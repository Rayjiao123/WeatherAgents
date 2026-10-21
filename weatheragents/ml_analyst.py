"""ML Analyst: a learned correction to the weather models, scored on its own.

Takes today's multi-model forecast for the contract's station (as issued 1-2 days ahead), plus weather
state, the model's recent errors at this station and yesterday's aerosols, and predicts the day's high
as a bell curve. The model is trained by ml_train.py on ~2.5 years of as-issued forecasts against the
station's own readings. Reads semantics only: no price, and nothing from the LLM agents.

Covers highs one or two days ahead. Anything else returns None, and the pipeline carries on without it.

    python -m weatheragents.ml_analyst        # prints the ML curve for the first few contracts
"""

import json
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np

from weatheragents import ml_data
from weatheragents.baseline import vector
from weatheragents.ensemble_analyst import _latest_contract_list, resolve_coordinates
from weatheragents.ml_data import ALL_FEATURES
from weatheragents.ml_train import ML_DIR, MLModel
from weatheragents.schemas import Contract, MLReport
from weatheragents.timeutil import now as ict_now

SUPPORTED_LEADS = (1, 2)
_LIVE_BUNDLES: dict[tuple[str, str, int], ml_data.Bundle] = {}  # neighbors are shared between events in one run


def station_coords(directory: Path = ML_DIR) -> dict[str, tuple[float, float]]:
    path = directory / "stations.json"
    return {c: (v[0], v[1]) for c, v in json.loads(path.read_text(encoding="utf-8")).items()} if path.exists() else {}


def live_neighbor(client: httpx.Client, code: str, coords: tuple[float, float], day, lead: int) -> ml_data.Bundle | None:
    """A neighbor's recent forecasts and station highs, fetched now (cached for the rest of the run)."""
    key = (code, day.isoformat(), lead)
    if key not in _LIVE_BUNDLES:
        try:
            fc = ml_data.fetch_forecasts(client, coords[0], coords[1], day - timedelta(days=10), day - timedelta(days=1), use_cache=False, leads=(lead,), variables=ml_data.TEMP_ONLY)
            obs = ml_data.fetch_obs(client, code, fc["tz"], day - timedelta(days=40), day - timedelta(days=1), use_cache=False)
        except (httpx.HTTPError, ValueError, KeyError):
            return None
        _LIVE_BUNDLES[key] = ml_data.Bundle(code, coords[0], coords[1], fc["elev"], fc["tz"],
                                            {lead: ml_data.hourly_to_daily(fc["hourly"], lead)}, obs, {})
    return _LIVE_BUNDLES[key]


def load_model(directory: Path = ML_DIR) -> tuple[MLModel, dict] | None:
    meta_path = directory / "meta.json"
    if not (meta_path.exists() and (directory / "mean_model.txt").exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = MLModel(
        meta["features"],
        lgb.Booster(model_file=str(directory / "mean_model.txt")),
        lgb.Booster(model_file=str(directory / "spread_model.txt")),
        meta["scale"],
    )
    return model, meta


def fresh_row(row: dict, fresh: dict | None) -> dict | None:
    """The same inputs with the newest weather-model runs swapped in for the day-ahead ones.

    The model was trained on day-ahead forecasts. A fresher forecast of the same day is simply more accurate, and a
    test on history showed the model gains from it without retraining (2-day-old inputs 0.98 C typical miss, 1-day-old
    inputs 0.89 C, against 0.885 C if also retrained on 1-day-old). Everything that describes past days is unchanged.
    """
    if not fresh:
        return None
    models = {m: v for m, v in fresh["models"].items() if m in ml_data.MODELS}
    if len(models) < 2:
        return None
    out = dict(row)
    for m in ml_data.MODELS:
        out[f"fc_{m}"] = models.get(m, row[f"fc_{m}"])
    vals = [out[f"fc_{m}"] for m in ml_data.MODELS if out[f"fc_{m}"] == out[f"fc_{m}"]]
    out["fc_mean"] = sum(vals) / len(vals)
    out["fc_std"] = statistics.pstdev(vals) if len(vals) >= 2 else float("nan")
    shift = out["fc_mean"] - row["fc_mean"]
    if row["dT"] == row["dT"]:
        out["dT"] = row["dT"] + shift  # yesterday's forecast for the day before is unchanged, so the change moves with today's
    if row["clim_mean"] == row["clim_mean"]:
        out["clim_anom"] = out["fc_mean"] - row["clim_mean"]
    return out


def analyze_contract(contract: Contract, client: httpx.Client, *, lead_days: int, loaded: tuple[MLModel, dict] | None) -> MLReport | None:
    return analyze_pair(contract, client, lead_days=lead_days, loaded=loaded)[0]


def analyze_pair(contract: Contract, client: httpx.Client, *, lead_days: int, loaded: tuple[MLModel, dict] | None,
                 fresh: dict | None = None) -> tuple[MLReport | None, MLReport | None]:
    """(the model on day-ahead inputs, the same model on the newest runs). The second is None without fresh forecasts."""
    s = contract.semantics_view()
    if loaded is None or s.kind != "highest" or lead_days not in SUPPORTED_LEADS or not s.station_code:
        return None, None
    model, meta = loaded
    lat, lon = resolve_coordinates(s.station_code)
    day = s.observation_date
    start = day - timedelta(days=10)

    fc = ml_data.fetch_forecasts(client, lat, lon, start, day, use_cache=False, leads=(lead_days,))
    try:
        obs = ml_data.fetch_obs(client, s.station_code, fc["tz"], start, day - timedelta(days=1), use_cache=False)
    except (httpx.HTTPError, ValueError, KeyError):
        obs = {}  # recent-error features go missing; the model handles that
    used = set(model.features)  # fetch only what the saved model reads: aerosols and neighbors tested at zero gain
    aero = {}
    if used & set(ml_data.FAMILIES["aerosol"]):
        try:
            aero = ml_data.fetch_aerosol(client, lat, lon, start, day - timedelta(days=1), use_cache=False)
        except (httpx.HTTPError, ValueError, KeyError):
            aero = {}

    daily = ml_data.hourly_to_daily(fc["hourly"], lead_days)
    older = None
    if "run_drift" in set(model.features) and lead_days == 1:
        try:
            old = ml_data.fetch_forecasts(client, lat, lon, day, day, use_cache=False, leads=(2,), variables=ml_data.TEMP_ONLY)
            older = ml_data.hourly_to_daily(old["hourly"], 2)
        except (httpx.HTTPError, ValueError, KeyError):
            older = None  # the drift input goes missing; the model handles that
    nb = None
    if used & set(ml_data.FAMILIES["neighbors"]):
        coords = station_coords()
        neighbors = []
        for code, dist in ml_data.nearest_neighbors(s.station_code, coords):
            b = live_neighbor(client, code, coords[code], day, lead_days)
            if b is not None:
                neighbors.append((b, dist))
        nb = ml_data.neighbor_features(day.isoformat(), lead_days, neighbors) if neighbors else None
    row = ml_data.make_features(day.isoformat(), lead_days, daily, obs, aero, lat, lon, fc["elev"], nb=nb,
                                normals=ml_data.normals_for(lat, lon), older=older)
    if row is None:
        return None, None
    k = 9 / 5 if s.unit == "F" else 1.0
    off = 32.0 if s.unit == "F" else 0.0
    present = [m for m in ml_data.MODELS if m in daily[day.isoformat()]["fc"]]
    caveats = [f"trained on station highs through {meta['trained_until']}; day-ahead forecasts only"]
    if len(present) < len(ml_data.MODELS):
        caveats.append(f"only {len(present)} of {len(ml_data.MODELS)} weather models had data for this date")
    if not obs:
        caveats.append("station history unavailable, recent-error features missing")
    def report(r: dict, inputs: str) -> MLReport:
        c, sd_ = model.predict(np.array([[r[f] for f in ALL_FEATURES]], dtype=float))
        center, sd = float(c[0]) * k + off, float(sd_[0]) * k
        return MLReport(
            event_id=s.event_id,
            station_code=s.station_code,
            lead_days=lead_days,
            center=round(center, 2),
            sd=round(sd, 2),
            multi_model_mean=round(r["fc_mean"] * k + off, 2),
            bracket_probabilities=vector(s, center=center, sd=sd),
            models_used=present,
            recent_error_mean=None if r["err_mean"] != r["err_mean"] else round(r["err_mean"] * k, 3),
            inputs=inputs,
            model_trained_until=meta["trained_until"],
            feature_set=meta.get("chosen_set", ""),
            caveats=caveats,
            generated_at=ict_now(),
        )

    fr = fresh_row(row, fresh)
    return report(row, "day-ahead run"), (report(fr, "newest run") if fr else None)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    loaded = load_model()
    if loaded is None:
        print("no trained model in data/ml; run `python -m weatheragents.ml_train` first", file=sys.stderr)
        return 1
    path = _latest_contract_list(Path("data"))
    contracts = [Contract.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    today = datetime.now().date()
    shown = 0
    with httpx.Client(timeout=60, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for c in contracts:
            lead = (c.semantics.observation_date - today).days
            try:
                rep = analyze_contract(c, client, lead_days=lead, loaded=loaded)
            except (httpx.HTTPError, LookupError, ValueError, KeyError) as exc:
                print(f"! {c.semantics.city}: {exc}", file=sys.stderr)
                continue
            if rep is None:
                continue
            top = max(rep.bracket_probabilities, key=lambda b: b.probability)
            print(f"{c.semantics.city:<18}{c.semantics.observation_date}  lead {lead}  models-mean {rep.multi_model_mean:>6.1f}  "
                  f"ML center {rep.center:>6.1f} sd {rep.sd:.2f}  top {top.label} ({top.probability:.0%})")
            shown += 1
            if shown >= 8:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

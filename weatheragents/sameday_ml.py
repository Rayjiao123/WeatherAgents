"""Same-day ML model: predicts the day's high from what has happened so far today.

    python -m weatheragents.sameday_ml --fetch    # download the hourly forecasts it needs (resumable)
    python -m weatheragents.sameday_ml            # build rows, measure against the heuristic projector, fit and save

At each local hour from 07:00 to 19:00 on every historical day, it sees the station's reports so far, the latest
hourly forecast, and how far the forecast has been missing today, and predicts the day's final high. The centre is
the high so far plus a learned rise, so it can never forecast below what has already happened; the spread is learned
too. It is scored against sameday_analyst.py, the hand-built projector it is meant to replace, on days it never saw.

Optional input groups (rt_obs.py), measured one at a time so only what helps stays:
  front signs  pressure change, wind shift, dew point, rain, cloud, visibility at the station: a front or rain arriving
  upstream     whether the stations in the direction the wind comes from are warming or cooling

Same data rules as the day-ahead model: nothing from after the cut-off hour is used, and the test period (from
2026-04-01) is never trained on.
"""

import argparse
import json
import math
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import lightgbm as lgb
import numpy as np
from scipy.special import ndtr

from weatheragents import ml_data, rt_obs
from weatheragents.baseline import vector
from weatheragents.ensemble_analyst import resolve_coordinates
from weatheragents.ml_train import ML_DIR, MLModel, data_end, fit_model, stations_to_fetch
from weatheragents.sameday_analyst import BIAS_DAMPING, FORECAST_URL, fetch_obs_today, project, spread_for
from weatheragents.schemas import Contract, SameDayReport
from weatheragents.timeutil import now as ict_now

SD_DIR = ML_DIR / "sameday"
DATASET = ML_DIR / "sameday_dataset.npz"
HOURS = range(7, 20)  # local cut-off hours
MIN_REPORTS = 4
MIN_GAIN = 0.01  # whole-degree log loss an optional input group must add to earn its place
SD_FEATURES = [
    "sd_high_so_far", "sd_last", "sd_dev_now", "sd_dev3", "sd_trend3", "sd_fc_rest_max", "sd_fc_day_max", "sd_fc_now",
    "sd_proj_rise", "sd_hour", "sd_since_high", "sd_fc_std", "doy_sin", "doy_cos", "lat", "lon", "elev",
]
SD_ALL = SD_FEATURES + rt_obs.FRONT_FEATURES + rt_obs.UP_FEATURES
NAN = math.nan


def make_sd_row(day: str, cutoff: float, reports: list[tuple[float, float]], path: list, models_max: dict, lat: float, lon: float, elev: float) -> dict | None:
    """Features at `cutoff` (local hour, decimal) from reports up to then and the day's hourly forecast path."""
    sofar = [(h, t) for h, t in reports if h <= cutoff]
    if len(sofar) < MIN_REPORTS or len(path) != 24:
        return None
    fc_at = lambda h: path[min(int(h), 23)]
    high = max(t for _, t in sofar)
    t_last, last = sofar[-1]
    recent = [t - fc_at(h) for h, t in sofar if h >= t_last - 3 and fc_at(h) is not None]
    dev3 = sum(recent) / len(recent) if recent else NAN
    fc_last = fc_at(t_last)
    ahead = [v for i, v in enumerate(path) if i > cutoff and v is not None]
    rest_max = max(ahead) if ahead else NAN
    older = [t for h, t in sofar if h <= t_last - 3]
    hour_of_high = max(h for h, t in sofar if t == high)
    d = date.fromisoformat(day)
    doy = d.timetuple().tm_yday
    vals = list(models_max.values())
    return {
        "sd_high_so_far": high, "sd_last": last,
        "sd_dev_now": last - fc_last if fc_last is not None else NAN, "sd_dev3": dev3,
        "sd_trend3": last - older[-1] if older else NAN,
        "sd_fc_rest_max": rest_max, "sd_fc_day_max": max(v for v in path if v is not None),
        "sd_fc_now": fc_at(cutoff) if fc_at(cutoff) is not None else NAN,
        "sd_proj_rise": (rest_max + BIAS_DAMPING * dev3 - high) if not (math.isnan(rest_max) or math.isnan(dev3)) else NAN,
        "sd_hour": float(cutoff), "sd_since_high": cutoff - hour_of_high,
        "sd_fc_std": statistics.pstdev(vals) if len(vals) >= 2 else NAN,
        "doy_sin": math.sin(2 * math.pi * doy / 365.25), "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "lat": lat, "lon": lon, "elev": elev,
    }


def heuristic(reports: list[tuple[float, float]], path: list, cutoff: float) -> tuple[float, float]:
    """The hand-built projector's centre and spread at the same moment, for comparison."""
    sofar = [(h, t) for h, t in reports if h <= cutoff]
    forecast = {float(i): v for i, v in enumerate(path) if v is not None}
    pr = project(sofar, forecast, cutoff)
    return pr["center"], pr["sd"]


def front_and_upstream(rt_rows: list[dict], cutoff: float, day: str, tz: str, lat: float, lon: float, neighbours: list) -> dict:
    """The optional input groups at one moment: the station's own signs of change, and what upwind stations are doing."""
    own = rt_obs.own_features(rt_rows, cutoff)
    hour = [r for r in rt_rows if cutoff - 1.0 <= r["h"] <= cutoff]
    now = [r for r in rt_rows if r["h"] <= cutoff]
    now_s = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo(tz)).timestamp() + cutoff * 3600
    up = rt_obs.upstream_features(lat, lon, rt_obs._wind_dir(hour), now_s, now[-1]["tmpc"] if now else None, neighbours)
    return {**own, **up}


# ---------------------------------------------------------------- data


def fetch_day0(stations: list[str], end: date) -> None:
    """Download the latest-run hourly forecasts (temperature only, 4 models) for every station; cached on disk."""
    with httpx.Client(timeout=120, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for i, code in enumerate(stations, 1):
            try:
                lat, lon = resolve_coordinates(code)
                ml_data.fetch_forecasts(client, lat, lon, ml_data.START - timedelta(days=10), end, use_cache=True,
                                        leads=(0,), variables=ml_data.TEMP_ONLY)
                print(f"[{i}/{len(stations)}] {code}: day-0 forecasts ready", flush=True)
            except (LookupError, KeyError, ValueError, httpx.HTTPError) as exc:
                print(f"[{i}/{len(stations)}] {code}: skipped ({type(exc).__name__}: {exc})", flush=True)


def _station_tz(client: httpx.Client, code: str, cache: dict, end: date) -> tuple[str, float, float, float]:
    """(timezone, lat, lon, elevation) for any tracked or helper station, from cached downloads."""
    if code not in cache:
        lat, lon = resolve_coordinates(code)
        start = ml_data.START - timedelta(days=10)
        # use whichever download is already on disk: tracked stations have the latest-run one, helper stations the day-ahead one
        if ml_data.forecast_cached(lat, lon, start, end, leads=(0,), variables=ml_data.TEMP_ONLY):
            leads = (0,)
        else:
            leads = ml_data.LEADS
        fc = ml_data.fetch_forecasts(client, lat, lon, start, end, use_cache=True, leads=leads, variables=ml_data.TEMP_ONLY)
        cache[code] = (fc["tz"], lat, lon, fc["elev"])
    return cache[code]


def build_rows(stations: list[str], end: date) -> dict[str, np.ndarray]:
    rows = []
    coords = {c: tuple(v) for c, v in json.loads((ML_DIR / "stations.json").read_text(encoding="utf-8")).items()} if (ML_DIR / "stations.json").exists() else {}
    tzc: dict = {}
    series_cache: dict = {}
    with httpx.Client(timeout=120, headers={"User-Agent": "weatheragents/0.1"}) as client:
        def series(code: str):
            if code not in series_cache:
                tz, *_ = _station_tz(client, code, tzc, end)
                rep = ml_data.fetch_reports(client, code, tz, ml_data.START - timedelta(days=10), end, use_cache=True)
                series_cache[code] = rt_obs.utc_series(rep, tz)
            return series_cache[code]

        for code in stations:
            try:
                tz, lat, lon, elev = _station_tz(client, code, tzc, end)
                fc = ml_data.fetch_forecasts(client, lat, lon, ml_data.START - timedelta(days=10), end, use_cache=True, leads=(0,), variables=ml_data.TEMP_ONLY)
                paths = ml_data.hourly_paths(fc["hourly"])
                reports = ml_data.fetch_reports(client, code, tz, ml_data.START - timedelta(days=10), end, use_cache=True)
                highs = ml_data.fetch_obs(client, code, tz, ml_data.START - timedelta(days=10), end, use_cache=True)
                rt_days = rt_obs.fetch_rt(client, code, tz, ml_data.START - timedelta(days=10), end, use_cache=True)
                neighbours = []
                for n, dist in ml_data.nearest_neighbors(code, coords):
                    ntz, nlat, nlon, _ = _station_tz(client, n, tzc, end)
                    neighbours.append((nlat, nlon, dist, series(n)))
            except (LookupError, KeyError, ValueError, httpx.HTTPError) as exc:
                print(f"{code}: skipped ({type(exc).__name__}: {exc})", flush=True)
                continue
            n0 = len(rows)
            d = ml_data.START
            while d <= end:
                day = d.isoformat()
                if day in highs and day in paths and day in reports and day in rt_days:
                    for h in HOURS:
                        row = make_sd_row(day, h, reports[day], paths[day]["path"], paths[day]["models"], lat, lon, elev)
                        if row is None:
                            continue
                        hc, hs = heuristic(reports[day], paths[day]["path"], h)
                        extra = front_and_upstream(rt_days[day], h, day, tz, lat, lon, neighbours)
                        rows.append({"station": code, "date": day, "y": highs[day], "h_center": hc, "h_sd": hs, **row, **extra})
                d += timedelta(days=1)
            print(f"{code}: {len(rows) - n0} rows", flush=True)
    (ML_DIR / "station_tz.json").write_text(json.dumps({c: v[0] for c, v in tzc.items()}), encoding="utf-8")
    data = {
        "X": np.array([[r[f] for f in SD_ALL] for r in rows], dtype=float),
        "y": np.array([r["y"] for r in rows]), "station": np.array([r["station"] for r in rows]), "date": np.array([r["date"] for r in rows]),
        "h_center": np.array([r["h_center"] for r in rows]), "h_sd": np.array([r["h_sd"] for r in rows]),
        "features": np.array(SD_ALL),
    }
    np.savez(DATASET, **data)
    print(f"same-day dataset: {len(rows)} rows -> {DATASET}")
    return data


# ---------------------------------------------------------------- scoring


def trunc_metrics(y: np.ndarray, c: np.ndarray, sd: np.ndarray, floor: np.ndarray) -> dict:
    """Whole-degree log loss of a bell curve cut off below the high so far (nothing lower can happen)."""
    k = np.round(y)
    hi, lo = k + 0.5, np.maximum(k - 0.5, floor)
    mass = np.maximum(ndtr((hi - c) / sd) - ndtr((lo - c) / sd), 0.0)
    tail = np.maximum(1.0 - ndtr((floor - c) / sd), 1e-9)
    p = np.clip(mass / tail, 1e-3, 1.0)
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(y - c))), "int_log_loss": float(np.mean(-np.log(p)))}


def feature_groups() -> list[tuple[str, list[str]]]:
    return [("same-day model, current inputs", SD_FEATURES),
            ("+ front signs (own station)", SD_FEATURES + rt_obs.FRONT_FEATURES),
            ("+ upwind stations", SD_FEATURES + rt_obs.UP_FEATURES),
            ("+ both", SD_ALL)]


def measure(data: dict, test_start: str):
    d, X = data["date"], data["X"]
    tr, te = d < test_start, d >= test_start
    floor = X[:, SD_ALL.index("sd_high_so_far")] - 0.5
    results, models = [], {}
    for name, feats in feature_groups():
        model = fit_model(X[tr], data["y"][tr], d[tr], feats, all_features=SD_ALL, anchor="sd_high_so_far")
        c, sd = model.predict(X[te])
        results.append({"name": name, "features": feats, **trunc_metrics(data["y"][te], c, sd, floor[te])})
        models[name] = (c, sd)
        print(f"  measured {name}", flush=True)
    results.append({"name": "heuristic projector", "features": [], **trunc_metrics(data["y"][te], data["h_center"][te], data["h_sd"][te], floor[te])})
    return results, models, floor[te]


def save(model: MLModel, meta: dict) -> None:
    SD_DIR.mkdir(parents=True, exist_ok=True)
    model.mean_model.save_model(str(SD_DIR / "mean_model.txt"))
    model.spread_model.save_model(str(SD_DIR / "spread_model.txt"))
    (SD_DIR / "meta.json").write_text(json.dumps({"features": model.features, "all_features": SD_ALL, "scale": model.scale, "fitted_at": ict_now().isoformat(), **meta}, indent=2), encoding="utf-8")


def load() -> tuple[MLModel, dict] | None:
    meta_path = SD_DIR / "meta.json"
    if not (meta_path.exists() and (SD_DIR / "mean_model.txt").exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = MLModel(meta["features"], lgb.Booster(model_file=str(SD_DIR / "mean_model.txt")),
                    lgb.Booster(model_file=str(SD_DIR / "spread_model.txt")), meta["scale"], meta.get("all_features", SD_FEATURES), "sd_high_so_far")
    return model, meta


# ---------------------------------------------------------------- live


def _neighbour_series_live(client: httpx.Client, s_code: str, today: str) -> list:
    coords = {c: tuple(v) for c, v in json.loads((ML_DIR / "stations.json").read_text(encoding="utf-8")).items()}
    tzs = json.loads((ML_DIR / "station_tz.json").read_text(encoding="utf-8"))
    out = []
    d = date.fromisoformat(today)
    for n, dist in ml_data.nearest_neighbors(s_code, coords):
        if n not in tzs:
            continue
        try:
            rep = ml_data.fetch_reports(client, n, tzs[n], d - timedelta(days=1), d, use_cache=False)
        except (httpx.HTTPError, ValueError, KeyError):
            continue
        out.append((coords[n][0], coords[n][1], dist, rt_obs.utc_series(rep, tzs[n])))
    return out


def analyze_contract(contract: Contract, client: httpx.Client, *, lead_days: int, loaded) -> SameDayReport | None:
    """Live same-day prediction for an event whose day is in progress; None when the model does not apply."""
    s = contract.semantics_view()
    if loaded is None or s.kind != "highest" or lead_days != 0 or not s.station_code:
        return None
    model, meta = loaded
    lat, lon = resolve_coordinates(s.station_code)
    r = client.get(FORECAST_URL, params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "models": ",".join(ml_data.MODELS),
                                         "forecast_days": 1, "timezone": "auto"})
    r.raise_for_status()
    p = r.json()
    local_now = datetime.now(timezone.utc) + timedelta(seconds=p["utc_offset_seconds"])
    today = local_now.date().isoformat()
    n_today = sum(1 for t in p["hourly"]["time"] if t[:10] == today)
    paths = ml_data.hourly_paths({"time": [t for t in p["hourly"]["time"] if t[:10] == today],
                                  **{k: v[:n_today] for k, v in p["hourly"].items() if k != "time"}})
    cutoff = local_now.hour + local_now.minute / 60
    if today not in paths or not (HOURS.start <= cutoff < HOURS.stop):
        return None  # the model was trained on 07:00-19:59 local; outside that the heuristic projector still covers it
    reports = fetch_obs_today(client, s.station_code, p["timezone"], today)
    row = make_sd_row(today, cutoff, reports, paths[today]["path"], paths[today]["models"], lat, lon, p.get("elevation", 0.0))
    if row is None:
        return None
    used = set(model.features)
    if used & set(rt_obs.FRONT_FEATURES + rt_obs.UP_FEATURES):
        try:
            rt_rows = rt_obs.fetch_rt(client, s.station_code, p["timezone"], date.fromisoformat(today) - timedelta(days=1), date.fromisoformat(today), use_cache=False).get(today, [])
            neighbours = _neighbour_series_live(client, s.station_code, today) if used & set(rt_obs.UP_FEATURES) else []
            row.update(front_and_upstream(rt_rows, cutoff, today, p["timezone"], lat, lon, neighbours))
        except (httpx.HTTPError, ValueError, KeyError, OSError):
            pass  # those inputs go missing; the model handles that
    allf = model.all_features
    c, sd = model.predict(np.array([[row.get(f, NAN) for f in allf]], dtype=float))
    k, off = (9 / 5, 32.0) if s.unit == "F" else (1.0, 0.0)
    center, spread, high = float(c[0]) * k + off, float(sd[0]) * k, row["sd_high_so_far"] * k + off
    return SameDayReport(
        event_id=s.event_id, station_code=s.station_code, local_hour=round(cutoff, 2), high_so_far=round(high, 1),
        remaining_forecast_high=None if row["sd_fc_rest_max"] != row["sd_fc_rest_max"] else round(row["sd_fc_rest_max"] * k + off, 1),
        bias_adjustment=round(row["sd_dev3"] * k, 2) if row["sd_dev3"] == row["sd_dev3"] else 0.0,
        center=round(center, 2), sd=round(spread, 2),
        bracket_probabilities=vector(s, center=center, sd=spread, lower=high - 0.5), generated_at=ict_now())


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Train and measure the same-day model")
    ap.add_argument("--fetch", action="store_true", help="download the hourly forecasts it needs, then stop")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--test-start", default="2026-04-01")
    args = ap.parse_args(argv)
    stations = stations_to_fetch(None)
    end = data_end()
    if args.fetch:
        fetch_day0(stations, end)
        return 0
    data = None
    if DATASET.exists() and not args.rebuild:
        z = np.load(DATASET)
        if "features" in z.files and list(z["features"]) == SD_ALL:
            data = {k: z[k] for k in z.files}
        else:
            print("cached same-day dataset has a different feature list; rebuilding from the raw cache")
    if data is None:
        data = build_rows(stations, end)
    results, models, floor_te = measure(data, args.test_start)
    n_test = int((data["date"] >= args.test_start).sum())
    print(f"\nTrained on days before {args.test_start}; scored on {n_test} station-hours after it (lower is better).")
    print(f"\n{'model':<34}{'typical miss (C)':>18}{'whole-degree log loss':>24}")
    for r in results:
        print(f"{r['name']:<34}{r['mae']:>18.2f}{r['int_log_loss']:>24.3f}")
    ml = [r for r in results if r["features"]]
    base = next(r for r in results if r["name"].startswith("same-day model, current"))
    best = min(ml, key=lambda r: r["int_log_loss"])
    if base["int_log_loss"] - best["int_log_loss"] < MIN_GAIN:
        best = base  # a gain this small is not worth the extra live downloads and failure modes
    print(f"\nchosen: {best['name']} ({best['int_log_loss'] - base['int_log_loss']:+.3f} against the current inputs; an optional group must gain {MIN_GAIN} to be kept)")
    c, sd = models[best["name"]]
    te = data["date"] >= args.test_start
    hours = data["X"][te, SD_ALL.index("sd_hour")]
    print("\nby local hour of the prediction (best model vs heuristic): log loss")
    for h in (8, 10, 12, 14, 16, 18):
        m = hours == h
        if m.sum() > 200:
            a = trunc_metrics(data["y"][te][m], c[m], sd[m], floor_te[m])
            b = trunc_metrics(data["y"][te][m], data["h_center"][te][m], data["h_sd"][te][m], floor_te[m])
            print(f"  {h:02d}:00   model {a['int_log_loss']:.3f}   heuristic {b['int_log_loss']:.3f}   miss {a['mae']:.2f} vs {b['mae']:.2f} C   (n={a['n']})")
    final = fit_model(data["X"], data["y"], data["date"], best["features"], all_features=SD_ALL, anchor="sd_high_so_far")
    save(final, {"trained_until": max(data["date"].tolist()), "n_rows": int(len(data["y"])), "chosen_set": best["name"],
                 "ml_log_loss": best["int_log_loss"], "heuristic_log_loss": results[-1]["int_log_loss"]})
    print(f"\nfinal model ({best['name']}) fitted on all {len(data['y'])} rows -> {SD_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

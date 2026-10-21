"""Trains and measures the ML Analyst.

    python -m weatheragents.ml_train                 # build data (cached), measure, fit final model
    python -m weatheragents.ml_train --test-start 2026-05-01
    python -m weatheragents.ml_train --skip-final    # measurement only

The ML predicts the station's daily high as (multi-model mean forecast + a learned correction) with a
learned uncertainty, so it is a post-processor on the weather models, not a rival to them.

Measurement is a time split: train on dates before --test-start, score on dates after it, against the
same per-station bias-and-spread correction the live baseline uses. Each feature family is then added
one at a time, so you can see what each kind of information is worth. Scores:

  int log loss   -log P(the station's whole-degree high) under the predicted bell curve; lower is better
  CRPS           distance between the predicted curve and the outcome, in degrees; lower is better
  z sd           spread of (error / predicted sd); 1.0 means the stated uncertainty is honest
"""

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np

from weatheragents import ml_data
from weatheragents.baseline import phi
from weatheragents.ensemble_analyst import _latest_contract_list, resolve_coordinates
from weatheragents.ml_data import ALL_FEATURES, FAMILIES, KEPT, ROUND2
from weatheragents.schemas import Contract
from weatheragents.timeutil import now as ict_now

ML_DIR = Path("data") / "ml"
DATASET = ML_DIR / "dataset.npz"
PARAMS = {
    "objective": "regression", "learning_rate": 0.04, "num_leaves": 15, "min_data_in_leaf": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
    "verbose": -1, "seed": 7, "num_threads": 4,
}
ROUNDS = 250
SD_MIN, SD_MAX = 0.3, 6.0
SQRT_HALF_PI = math.sqrt(math.pi / 2)  # E|x| = sd * sqrt(2/pi) for a normal, so sd = E|x| * sqrt(pi/2)


# ---------------------------------------------------------------- dataset


def stations_to_fetch(contracts_path: Path | None) -> list[str]:
    path = contracts_path or _latest_contract_list(Path("data"))
    contracts = [Contract.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    return sorted({c.semantics.station_code for c in contracts if c.semantics.station_code and c.semantics.kind == "highest"})


DATA_END = ML_DIR / "data_end.txt"


def data_end(refresh: bool = False) -> date:
    """The last day of training data. Pinned so a retrain reuses the on-disk cache; --refresh-data advances it."""
    if refresh or not DATA_END.exists():
        newest = date.today() - timedelta(days=3)
        ML_DIR.mkdir(parents=True, exist_ok=True)
        DATA_END.write_text(newest.isoformat() if refresh else "2026-10-10", encoding="utf-8")
    return date.fromisoformat(DATA_END.read_text(encoding="utf-8").strip())


def build_dataset(stations: list[str], *, force: bool = False, end: date | None = None, weather2: bool = False, more_models: bool = False) -> dict[str, np.ndarray]:
    end = end or data_end()
    if DATASET.exists() and not force:
        z = np.load(DATASET)
        aux_path = ML_DIR / "aux_stations.json"
        stale_aux = aux_path.exists() and aux_path.stat().st_mtime > DATASET.stat().st_mtime  # new helper stations picked since
        if "features" in z.files and list(z["features"]) == ALL_FEATURES and not stale_aux:
            return {k: z[k] for k in z.files}
        print("cached dataset is out of date (new features or new helper stations); rebuilding from the raw cache")
    bundles, failed = {}, []
    with httpx.Client(timeout=120, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for i, code in enumerate(stations, 1):
            try:
                lat, lon = resolve_coordinates(code)
                bundles[code] = ml_data.fetch_bundle(client, code, lat, lon, end=end, weather2=weather2, more_models=more_models)
                print(f"[{i}/{len(stations)}] {code}: fetched", flush=True)
            except (LookupError, KeyError, ValueError, httpx.HTTPError) as exc:
                failed.append(code)
                print(f"[{i}/{len(stations)}] {code}: skipped ({type(exc).__name__}: {exc})", flush=True)
    if not bundles:
        raise SystemExit("no training data built")
    helpers = {}  # nearby airports that only supply neighbor features (see ml_aux.py); never traded
    aux_path = ML_DIR / "aux_stations.json"
    if aux_path.exists():
        aux = json.loads(aux_path.read_text(encoding="utf-8"))
        with httpx.Client(timeout=120, headers={"User-Agent": "weatheragents/0.1"}) as client:
            for i, (code, (lat, lon)) in enumerate(aux.items(), 1):
                try:
                    helpers[code] = ml_data.fetch_bundle(client, code, lat, lon, light=True, end=end)
                    print(f"[helper {i}/{len(aux)}] {code}: fetched", flush=True)
                except (KeyError, ValueError, httpx.HTTPError) as exc:
                    print(f"[helper {i}/{len(aux)}] {code}: skipped ({type(exc).__name__})", flush=True)
    pool = {**bundles, **helpers}
    coords = {c: (b.lat, b.lon) for c, b in pool.items()}
    rows = []
    for code, b in bundles.items():
        nbs = [(pool[n], dist) for n, dist in ml_data.nearest_neighbors(code, coords)]
        rows += ml_data.bundle_rows(b, nbs, end=end)
    data = {
        "X": np.array([[r[f] for f in ALL_FEATURES] for r in rows], dtype=float),
        "y": np.array([r["y"] for r in rows], dtype=float),
        "station": np.array([r["station"] for r in rows]),
        "date": np.array([r["date"] for r in rows]),
        "lead": np.array([r["lead"] for r in rows], dtype=float),
        "features": np.array(ALL_FEATURES),
    }
    ML_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATASET, **data)
    (ML_DIR / "stations.json").write_text(json.dumps({c: list(v) for c, v in coords.items()}), encoding="utf-8")
    print(f"dataset: {len(rows)} rows, {len(bundles)} stations (+{len(helpers)} helpers), {len(failed)} failed -> {DATASET}")
    return data


# ---------------------------------------------------------------- model


def _cols(features: list[str], all_features: list[str] | None = None) -> list[int]:
    return [(all_features or ALL_FEATURES).index(f) for f in features]


def _fit(X: np.ndarray, y: np.ndarray) -> lgb.Booster:
    return lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=ROUNDS)


def _blocked_oof(X: np.ndarray, y: np.ndarray, order_key: np.ndarray, k: int = 4) -> np.ndarray:
    """Out-of-fold predictions using contiguous blocks of dates, so a row is never predicted by a model that saw it."""
    folds = np.array_split(np.argsort(order_key, kind="stable"), k)
    oof = np.empty(len(y))
    for i, te in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
        oof[te] = _fit(X[tr], y[tr]).predict(X[te])
    return oof


class MLModel:
    """Mean-correction model + spread model + one scale factor that makes the spread honest."""

    def __init__(self, features: list[str], mean_model: lgb.Booster, spread_model: lgb.Booster, scale: float,
                 all_features: list[str] | None = None, anchor: str = "fc_mean"):
        # the centre is the anchor column (the weather models' mean for the day-ahead model, the high so far for the
        # same-day model) plus a learned correction
        self.features, self.mean_model, self.spread_model, self.scale = features, mean_model, spread_model, scale
        self.all_features, self.anchor = all_features or ALL_FEATURES, anchor

    def predict(self, X_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = X_all[:, _cols(self.features, self.all_features)]
        center = X_all[:, self.all_features.index(self.anchor)] + self.mean_model.predict(X)
        sd = np.clip(self.scale * SQRT_HALF_PI * np.maximum(self.spread_model.predict(X), 0.05), SD_MIN, SD_MAX)
        return center, sd


def fit_model(X_all: np.ndarray, y: np.ndarray, dates: np.ndarray, features: list[str],
              all_features: list[str] | None = None, anchor: str = "fc_mean") -> MLModel:
    all_features = all_features or ALL_FEATURES
    X = X_all[:, _cols(features, all_features)]
    resid = y - X_all[:, all_features.index(anchor)]
    key = np.array([date.fromisoformat(d).toordinal() for d in dates])
    oof_mean = _blocked_oof(X, resid, key)
    abs_err = np.abs(resid - oof_mean)
    oof_abs = np.maximum(_blocked_oof(X, abs_err, key), 0.05)
    z = (resid - oof_mean) / (SQRT_HALF_PI * oof_abs)
    scale = float(np.std(z))  # makes the out-of-fold z-scores have unit spread
    return MLModel(features, _fit(X, resid), _fit(X, abs_err), scale, all_features, anchor)


# ---------------------------------------------------------------- baseline and metrics


def fit_baseline(station: np.ndarray, lead: np.ndarray, resid: np.ndarray) -> dict:
    """Per (station, lead) bias and spread of the multi-model mean, shrunk like calibration_fit does."""
    out, pooled = {}, {}
    for ld in np.unique(lead):
        m = lead == ld
        pooled[ld] = float(np.std(resid[m]))
        for s in np.unique(station[m]):
            r = resid[m & (station == s)]
            if len(r) >= 20:
                n = len(r)
                out[(s, ld)] = (float(np.mean(r) * n / (n + 15)), max(float(np.std(r)), 0.6))
    return {"stations": out, "pooled": pooled}


def baseline_predict(b: dict, station: np.ndarray, lead: np.ndarray, fc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bias = np.array([b["stations"].get((s, ld), (0.0, b["pooled"][ld]))[0] for s, ld in zip(station, lead)])
    sd = np.array([b["stations"].get((s, ld), (0.0, b["pooled"][ld]))[1] for s, ld in zip(station, lead)])
    return fc + bias, sd


def metrics(y: np.ndarray, c: np.ndarray, sd: np.ndarray) -> dict:
    z = (y - c) / sd
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = np.array([phi(v) for v in z])
    crps = sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))
    k = np.round(y)
    p = np.array([phi((kk + 0.5 - cc) / s) - phi((kk - 0.5 - cc) / s) for kk, cc, s in zip(k, c, sd)])
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - c))),
        "crps": float(np.mean(crps)),
        "int_log_loss": float(np.mean(-np.log(np.maximum(p, 1e-3)))),
        "z_sd": float(np.std(z)),
    }


# ---------------------------------------------------------------- experiments


def feature_sets_round2(include_weather2: bool, include_more_models: bool = False) -> list[tuple[str, list[str]]]:
    """Today's model, then today's model plus each new group on its own, then plus all of them."""
    kept = [f for fam in KEPT for f in FAMILIES[fam]]
    groups = [g for g in ROUND2 if (include_weather2 or g != "weather2") and (include_more_models or g != "more_models")]
    sets = [("ML: current model", kept)]
    sets += [(f"ML: current + {g}", kept + FAMILIES[g]) for g in groups]
    sets.append(("ML: current + all new groups", kept + [f for g in groups for f in FAMILIES[g]]))
    return sets


def feature_sets() -> list[tuple[str, list[str]]]:
    base = FAMILIES["base"]
    sets = [("ML: base only", base)]
    cum = list(base)
    for fam in ("multimodel", "weather", "recent_errors", "nowcast", "climatology", "aerosol", "neighbors"):
        cum = cum + FAMILIES[fam]
        sets.append((f"ML: + {fam} (cumulative)", list(cum)))
    for fam in ("multimodel", "weather", "recent_errors", "nowcast", "climatology", "aerosol", "neighbors"):
        sets.append((f"ML: base + {fam} only", base + FAMILIES[fam]))
    return sets


def measure(data: dict, test_start: str, sets: list[tuple[str, list[str]]] | None = None) -> list[dict]:
    d = data["date"]
    tr, te = d < test_start, d >= test_start
    if tr.sum() < 1000 or te.sum() < 200:
        raise SystemExit(f"split too small: {tr.sum()} train / {te.sum()} test rows")
    fc = data["X"][:, ALL_FEATURES.index("fc_mean")]
    results = []

    b = fit_baseline(data["station"][tr], data["lead"][tr], (data["y"] - fc)[tr])
    c, sd = baseline_predict(b, data["station"][te], data["lead"][te], fc[te])
    results.append({"name": "baseline: station bias + spread (what production uses)", "features": [], **metrics(data["y"][te], c, sd)})
    results.append({"name": "raw multi-model mean, one global spread", "features": [],
                    **metrics(data["y"][te], fc[te], np.full(te.sum(), float(np.std((data["y"] - fc)[tr]))))})

    for name, feats in (sets if sets is not None else feature_sets()):
        model = fit_model(data["X"][tr], data["y"][tr], d[tr], feats)
        c, sd = model.predict(data["X"][te])
        results.append({"name": name, "features": feats, **metrics(data["y"][te], c, sd)})
        print(f"  measured {name}", flush=True)
    return results


def measure_one(data: dict, test_start: str, name: str, features: list[str]) -> dict:
    """Score one specific feature set on the same time split as measure()."""
    d = data["date"]
    tr, te = d < test_start, d >= test_start
    model = fit_model(data["X"][tr], data["y"][tr], d[tr], features)
    c, sd = model.predict(data["X"][te])
    return {"name": name, "features": features, **metrics(data["y"][te], c, sd)}


def print_report(results: list[dict], test_start: str, n_train: int, n_test: int) -> None:
    base = results[0]
    print(f"\nTrained on {n_train} station-days before {test_start}; scored on {n_test} station-days after it.")
    print(f"\n{'model':<52}{'MAE':>6}{'CRPS':>7}{'int log loss':>14}{'vs baseline':>13}{'z sd':>7}")
    for r in results:
        d = r["int_log_loss"] - base["int_log_loss"]
        print(f"{r['name']:<52}{r['mae']:>6.2f}{r['crps']:>7.3f}{r['int_log_loss']:>14.3f}{d:>+13.3f}{r['z_sd']:>7.2f}")
    print("\n(vs baseline: negative = better than the live baseline)")


def fit_final(data: dict, features: list[str], meta: dict) -> None:
    model = fit_model(data["X"], data["y"], data["date"], features)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    model.mean_model.save_model(str(ML_DIR / "mean_model.txt"))
    model.spread_model.save_model(str(ML_DIR / "spread_model.txt"))
    imp = dict(zip(features, model.mean_model.feature_importance("gain")))
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:8]
    (ML_DIR / "meta.json").write_text(json.dumps({
        "features": features, "scale": model.scale, "trained_until": max(data["date"].tolist()),
        "n_rows": int(len(data["y"])), "fitted_at": ict_now().isoformat(),
        "top_features_by_gain": [[k, round(float(v), 1)] for k, v in top], **meta}, indent=2), encoding="utf-8")
    print(f"\nfinal model fitted on all {len(data['y'])} rows -> {ML_DIR}")
    print("top features:", ", ".join(f"{k}" for k, _ in top))


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Build data, measure the ML Analyst, fit the final model")
    ap.add_argument("--contracts", type=Path, default=None)
    ap.add_argument("--rebuild", action="store_true", help="re-download and rebuild the dataset")
    ap.add_argument("--test-start", default="2026-04-01")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--round2", action="store_true", help="measure today's model against today's model plus each new input group (saved as round2_report.json; does not refit the final model)")
    ap.add_argument("--weather2", action="store_true", help="include the large extra-weather download in --round2")
    ap.add_argument("--more-models", action="store_true", help="include the twelve extra weather models in --round2")
    ap.add_argument("--keep", default=None, help="comma-separated feature groups to keep (e.g. multimodel,weather,recent_errors,climatology); measures that set and refits the final model with it")
    ap.add_argument("--refresh-data", action="store_true", help="advance the training end date to 3 days ago and download what is new")
    ap.add_argument("--min-stations", type=int, default=38, help="refuse to overwrite the saved model if fewer stations were fetched (rate limits can skip some)")
    ap.add_argument("--final-only", action="store_true", help="skip measuring; refit the final model using the saved report.json")
    args = ap.parse_args(argv)

    data = build_dataset(stations_to_fetch(args.contracts), force=args.rebuild, end=data_end(args.refresh_data), weather2=args.weather2, more_models=args.more_models)
    if args.keep:
        fams = [f for f in args.keep.split(",") if f]
        feats = list(FAMILIES["base"]) + [f for fam in fams for f in FAMILIES[fam]]
        saved = json.loads((ML_DIR / "report.json").read_text(encoding="utf-8"))
        name = "ML: kept groups (" + ", ".join(fams) + ")"
        row = measure_one(data, saved["test_start"], name, feats)
        saved["results"] = [r for r in saved["results"] if not r["name"].startswith("ML: kept groups")] + [row]
        (ML_DIR / "report.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
        base_row = saved["results"][0]
        print(f"{name}: MAE {row['mae']:.2f}  CRPS {row['crps']:.3f}  int log loss {row['int_log_loss']:.3f}  vs baseline {row['int_log_loss'] - base_row['int_log_loss']:+.3f}")
        fit_final(data, feats, {"chosen_set": name, "test_start": saved["test_start"], "test_int_log_loss": row["int_log_loss"],
                                "baseline_int_log_loss": base_row["int_log_loss"]})
        return 0
    if args.final_only:
        saved = json.loads((ML_DIR / "report.json").read_text(encoding="utf-8"))
        results, args.test_start = saved["results"], saved["test_start"]
        _finish(data, results, args)
        return 0
    if args.round2:
        results = measure(data, args.test_start, feature_sets_round2(args.weather2, args.more_models))
        n_test = int((data["date"] >= args.test_start).sum())
        print_report(results, args.test_start, len(data["y"]) - n_test, n_test)
        (ML_DIR / "round2_report.json").write_text(json.dumps({"test_start": args.test_start, "results": results}, indent=2), encoding="utf-8")
        return 0
    results = measure(data, args.test_start)
    n_test = int((data["date"] >= args.test_start).sum())
    print_report(results, args.test_start, len(data["y"]) - n_test, n_test)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    (ML_DIR / "report.json").write_text(json.dumps({"test_start": args.test_start, "results": results}, indent=2), encoding="utf-8")

    n_stations = len(set(data["station"].tolist()))
    if not args.skip_final and n_stations < args.min_stations:
        print(f"\nonly {n_stations} stations were fetched (need {args.min_stations}); keeping the existing saved model. Re-run to resume from the cache.")
    elif not args.skip_final:
        _finish(data, results, args)
    return 0


def _finish(data: dict, results: list[dict], args) -> None:
    cumulative = [r for r in results if "(cumulative)" in r["name"] or r["name"] == "ML: base only"]
    best = min(cumulative, key=lambda r: r["int_log_loss"])
    print(f"\nfinal model uses the best cumulative feature set: {best['name']}")
    fit_final(data, best["features"], {"chosen_set": best["name"], "test_start": args.test_start,
                                       "test_int_log_loss": best["int_log_loss"], "baseline_int_log_loss": results[0]["int_log_loss"]})


if __name__ == "__main__":
    raise SystemExit(main())

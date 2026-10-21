"""What did the station record say before each of the biggest misses?

    python -m weatheragents.postmortem            # the 15 biggest misses of the day-ahead model on days it never saw
    python -m weatheragents.postmortem --top 30

For every big miss it lines up the day's forecast hour by hour against what the station actually reported, finds the
hour the day went off the forecast, and reads the station record in the hours before that moment: did pressure fall,
did the wind swing, did the dew point jump, was rain or fog reported? It then counts how many misses had a warning
sign at least two hours ahead. That share is a direct ceiling on what the real-time inputs (rt_obs.py) can achieve.

Signatures are rule-based and meant as a guide to what to look at, not a verdict.
"""

import argparse
import json
import math
import sys
from datetime import timedelta

import httpx
import numpy as np

from weatheragents import ml_data, rt_obs
from weatheragents.ensemble_analyst import resolve_coordinates
from weatheragents.ml_data import ALL_FEATURES
from weatheragents.ml_train import ML_DIR, data_end, fit_model

DIVERGE = 2.0  # degrees C between the hourly forecast and the station that counts as "went off the forecast"


def hourly_obs(rows: list[dict]) -> dict[int, float]:
    by_hour: dict[int, list[float]] = {}
    for r in rows:
        by_hour.setdefault(int(r["h"]), []).append(r["tmpc"])
    return {h: sum(v) / len(v) for h, v in by_hour.items()}


def diverge_hour(obs: dict[int, float], path: list, sign: int) -> int | None:
    """First hour from 06:00 when the station was DIVERGE degrees past the forecast in the direction of the miss (sign -1 = colder)."""
    for h in range(6, 22):
        if h in obs and path[h] is not None and sign * (obs[h] - path[h]) >= DIVERGE:
            return h
    return None


def signature(rows: list[dict], at: float, miss: float) -> dict:
    """Read the record in the six hours before `at`: what changed, and how early did it start?"""
    feats = rt_obs.own_features(rows, at)
    earlier = rt_obs.own_features(rows, at - 2.0)  # the same read two hours earlier: was the warning already there?
    pressure_fall = feats["ft_dp3"] <= -1.5 or earlier["ft_dp3"] <= -1.5
    shift = feats["ft_wshift"] >= 45 or earlier["ft_wshift"] >= 45
    dew_jump = feats["ft_dwp_chg3"] >= 3
    wet = feats["ft_wx_rain"] == 1 or (feats["ft_cloud"] >= 0.75 and feats["ft_dep_now"] <= 3)
    fog = feats["ft_wx_fog"] == 1 or (not math.isnan(feats["ft_vis"]) and feats["ft_vis"] < 3)
    early_warning = (earlier["ft_dp3"] <= -1.5) or (earlier["ft_wshift"] >= 45) or (earlier["ft_wx_rain"] == 1) or (earlier["ft_dwp_chg3"] >= 3)
    if miss < 0:
        label = "front or rain" if (pressure_fall or shift) and (wet or dew_jump) else "rain or thick cloud" if wet else "fog or low cloud" if fog else "cold air arriving" if (pressure_fall or shift) else "no clear sign"
    else:
        label = "warm air or downslope wind" if shift or pressure_fall else "clearing sky" if not wet else "no clear sign"
    return {"label": label, "early_warning": bool(early_warning), "dp3": feats["ft_dp3"], "shift": feats["ft_wshift"], "dew": feats["ft_dwp_chg3"]}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Read the station record before the biggest misses")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--test-start", default="2026-04-01")
    args = ap.parse_args(argv)

    z = np.load(ML_DIR / "dataset.npz")
    X, y, st, dt, ld = z["X"], z["y"], z["station"], z["date"], z["lead"]
    feats = json.loads((ML_DIR / "meta.json").read_text(encoding="utf-8"))["features"]
    tr, te = dt < args.test_start, dt >= args.test_start
    model = fit_model(X[tr], y[tr], dt[tr], feats)
    c, _ = model.predict(X[te])
    r = y[te] - c
    idx = np.where(te)[0]
    seen, picks = set(), []
    for i in np.argsort(-np.abs(r)):
        key = (st[idx[i]], dt[idx[i]])
        if key in seen:
            continue  # the same day appears at two lead times
        seen.add(key)
        picks.append((idx[i], r[i], c[i]))
        if len(picks) >= max(args.top * 3, 45):
            break
    end = data_end()
    results = []
    with httpx.Client(timeout=120, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for i, miss, centre in picks:
            code, day = st[i], dt[i]
            try:
                lat, lon = resolve_coordinates(code)
                fc = ml_data.fetch_forecasts(client, lat, lon, ml_data.START - timedelta(days=10), end, use_cache=True, leads=(0,), variables=ml_data.TEMP_ONLY)
                path = ml_data.hourly_paths(fc["hourly"]).get(day, {}).get("path")
                rows = rt_obs.fetch_rt(client, code, fc["tz"], ml_data.START - timedelta(days=10), end, use_cache=True).get(day)
            except (LookupError, KeyError, ValueError, httpx.HTTPError):
                continue
            if not path or not rows:
                continue
            obs = hourly_obs(rows)
            sign = -1 if miss < 0 else 1
            h = diverge_hour(obs, path, sign)
            pk_f = max(range(24), key=lambda q: path[q] if path[q] is not None else -99)
            pk_o = max(obs, key=obs.get)
            sig = signature(rows, float(h if h is not None else pk_o), miss) if h is not None else {"label": "never clearly left the forecast", "early_warning": False, "dp3": math.nan, "shift": math.nan, "dew": math.nan}
            results.append({"station": code, "day": day, "miss": miss, "forecast": float(X[i, ALL_FEATURES.index("fc_mean")]), "actual": float(y[i]),
                            "centre": float(centre), "peak_f": pk_f, "peak_o": pk_o, "div": h, **sig})
    results = results[: args.top * 2]
    print(f"{len(results)} of the biggest misses by the day-ahead model on days after {args.test_start}\n")
    print(f"{'station':<7}{'date':<12}{'forecast':>9}{'actual':>8}{'miss':>7}  {'peak hour fc/obs':>17}  {'off forecast at':>16}  {'pressure 3h':>12}  {'wind shift':>11}  what the record showed")
    for q in results[: args.top]:
        pk = f"{q['peak_f']:02d}h / {q['peak_o']:02d}h"
        dv = "-" if q["div"] is None else f"{q['div']:02d}:00"
        dp = "-" if math.isnan(q["dp3"]) else f"{q['dp3']:+.1f} hPa"
        sh = "-" if math.isnan(q["shift"]) else f"{q['shift']:.0f} deg"
        print(f"{q['station']:<7}{q['day']:<12}{q['forecast']:>8.1f}C{q['actual']:>7.1f}C{q['miss']:>+6.1f}C  {pk:>17}  {dv:>16}  {dp:>12}  {sh:>11}  {q['label']}")
    from collections import Counter

    counts = Counter(q["label"] for q in results)
    cold = [q for q in results if q["miss"] < 0]
    print("\nWhat the record showed across these misses: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    if results:
        warned = sum(1 for q in results if q["early_warning"])
        print(f"A warning sign (pressure fall, wind swing, rain, or a dew-point jump) was already on the record 2+ hours before the miss started on {warned} of {len(results)} ({warned / len(results):.0%}).")
        late = [q for q in cold if q["div"] is not None]
        if late:
            print(f"Cold misses: the day left the forecast at {sorted(q['div'] for q in late)[len(late) // 2]:02d}:00 (median hour); the forecast put the peak at {sorted(q['peak_f'] for q in late)[len(late) // 2]:02d}:00 (median).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

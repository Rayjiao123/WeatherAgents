"""Do national agencies' forecasts know something our models and the market do not?

    python -m weatheragents.agency_report

For each (city, date) where we have the agency's forecast high, the outcome, and a comparison forecast, it asks
whether the agency's disagreement with the comparison predicts the comparison's error. If it does, the agency has
information the comparison lacks:

    (actual - comparison)  =  b * (agency - comparison)  +  noise

  b near 0   the agency adds nothing: its number is just the comparison's with extra noise
  b near 1   the agency is right where the comparison is wrong: take the agency's number
  b negative the agency is worse than noise

Run against two comparisons: our ML/baseline forecast, and the market's implied average high. Beating the market
is the one that matters. Needs a few weeks of data before b is readable; the report says how sure it is.
"""

import argparse
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from weatheragents import agency_forecasts as af
from weatheragents import evaluate as ev
from weatheragents import prediction_log as plog
from weatheragents.market_monitor import SNAP_PATH, read_snapshots

MIN_N = 10


def label_value(label: str) -> float | None:
    nums = re.findall(r"-?\d+", label)
    return float(nums[0]) if nums else None


def slope(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Least-squares b through the origin, and its standard error."""
    sxx = sum(x * x for x in xs)
    if len(xs) < 3 or sxx == 0:
        return None
    b = sum(x * y for x, y in zip(xs, ys)) / sxx
    resid = sum((y - b * x) ** 2 for x, y in zip(xs, ys)) / (len(xs) - 1)
    return b, math.sqrt(resid / sxx)


def nearest_before(rows: list[dict], key: str, when: datetime) -> dict | None:
    best = None
    for r in rows:
        t = datetime.fromisoformat(r[key])
        if t <= when and (best is None or t > datetime.fromisoformat(best[key])):
            best = r
    return best


def build_pairs(agency_rows: list[dict], preds: dict[str, dict], outcomes: dict[str, dict], snaps: dict[tuple, list[dict]]) -> list[dict]:
    by_target = defaultdict(list)
    for r in agency_rows:
        by_target[(r["city"], r["target_date"])].append(r)
    pairs = []
    for eid, p in preds.items():
        o = outcomes.get(eid)
        if not o or p["kind"] != "highest" or p["unit"] != "C":
            continue
        rows = by_target.get((p["city"], p["observation_date"]))
        actual = label_value(o["resolved_label"])
        if not rows or actual is None:
            continue
        t_pred = datetime.fromisoformat(p["logged_at"])
        ag = nearest_before(rows, "collected_at", t_pred)
        if ag is None:
            continue
        ours = (p.get("ml") or {}).get("center")
        if ours is None:
            ours = (p.get("estimate") or {}).get("baseline_params", {}).get("center", p["ensemble"]["mean"])
        sn = nearest_before(snaps.get((p["city"], p["observation_date"]), []), "ts", datetime.fromisoformat(ag["collected_at"]))
        pairs.append({"city": p["city"], "date": p["observation_date"], "lead": ag["lead_days"], "agency": ag["high"], "ours": ours,
                      "market": sn["implied_mean"] if sn else None, "actual": actual})
    return pairs


def summarize(pairs: list[dict], against: str) -> str:
    use = [p for p in pairs if p.get(against) is not None]
    if len(use) < MIN_N:
        return f"  vs {against:<7} n={len(use):>3}  too few so far (need {MIN_N})"
    x = [p["agency"] - p[against] for p in use]
    y = [p["actual"] - p[against] for p in use]
    mae = lambda f: sum(abs(f(p)) for p in use) / len(use)
    res = slope(x, y)
    b, se = res if res else (float("nan"), float("nan"))
    verdict = ("agency adds information" if b - 2 * se > 0 else "agency adds nothing detectable" if b + 2 * se > 0 else "agency is worse than noise")
    return (f"  vs {against:<7} n={len(use):>3}  b = {b:+.2f} (+/-{2 * se:.2f})  {verdict}   "
            f"| typical miss: agency {mae(lambda p: p['actual'] - p['agency']):.2f}  {against} {mae(lambda p: p['actual'] - p[against]):.2f}  "
            f"average of the two {mae(lambda p: p['actual'] - (p['agency'] + p[against]) / 2):.2f} C")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Score agency forecasts against our forecasts and the market")
    ap.add_argument("--log-dir", type=Path, default=plog.LOG_DIR)
    args = ap.parse_args(argv)

    agency_rows = af.read()
    if not agency_rows:
        print("no agency forecasts yet: run `python -m weatheragents.agency_forecasts --watch 360`")
        return 0
    outcomes = {o["event_id"]: o for o in plog.read(args.log_dir / plog.OUTCOMES)}
    preds = ev.latest_before_resolution([p for p in plog.read(args.log_dir / plog.PREDICTIONS) if p["city"] in af.STATIONS])
    snaps = defaultdict(list)
    for r in read_snapshots(SNAP_PATH):
        if r["city"] in af.STATIONS:
            snaps[(r["city"], r["observation_date"])].append(r)
    for v in snaps.values():
        v.sort(key=lambda r: r["ts"])

    pairs = build_pairs(agency_rows, preds, outcomes, snaps)
    days = len({r["collected_at"][:10] for r in agency_rows})
    print(f"{len(agency_rows)} agency forecast rows over {days} day(s); {len(pairs)} resolved city-days to score")
    if not pairs:
        print("nothing to score yet: needs resolved events for Hong Kong, Tokyo or Singapore whose forecasts we collected beforehand")
        return 0
    for label, subset in (("ALL LEADS", pairs), ("1 DAY AHEAD", [p for p in pairs if p["lead"] == 1]),
                          ("2 DAYS AHEAD", [p for p in pairs if p["lead"] == 2]), ("3+ DAYS AHEAD", [p for p in pairs if p["lead"] >= 3])):
        print(f"\n{label}")
        for against in ("ours", "market"):
            print(summarize(subset, against))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

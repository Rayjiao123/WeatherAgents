"""What the market's price moves tell us.

    python -m weatheragents.market_moves

Reads the snapshots market_monitor.py has been taking. A move is a price change that someone paid for,
so it usually means information arrived. This report asks three questions:

1. WHEN does the market move? Moves bucketed by hour of day and by hours before resolution, to see when
   the market learns (new model runs, morning station reports, the afternoon peak).
2. HOW does it move? A shift in the implied average high is news about the centre; a change in the implied
   spread is news about confidence. A market that sharpens without its centre moving has learned the
   outcome is nearly settled.
3. DOES IT MOVE TOWARD US? For each prediction we logged, did the market's later average drift toward
   our forecast or away from it? This needs no outcomes, so it works before any event resolves. If the
   market keeps drifting toward our curve, our signal was early information. If it drifts away, we were
   noise.

Prices are Trader-side data. Nothing here feeds an analyst, a curve or the ML.
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from weatheragents import prediction_log as plog
from weatheragents.market_monitor import SNAP_PATH, read_snapshots

MOVE_MEAN = 0.5  # degrees C (scaled for Fahrenheit) of change in the implied average that counts as a move
MOVE_TV = 0.12  # or this much probability moved between brackets
MAX_GAP_MIN = 90  # snapshots are 30 minutes apart; anything further means the monitor was down in between
HEALTHY_BOOK = (0.85, 1.15)  # bracket mid prices of a liquid event add up to about 1


def tv_distance(a: list[float], b: list[float]) -> float:
    sa, sb = sum(a) or 1.0, sum(b) or 1.0
    return 0.5 * sum(abs(x / sa - y / sb) for x, y in zip(a, b))


def series_by_event(rows: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        if r.get("implied_mean") is not None:
            out[r["event_id"]].append(r)
    for v in out.values():
        v.sort(key=lambda r: datetime.fromisoformat(r["ts"]))
    return out


def find_moves(series: list[dict]) -> list[dict]:
    """Consecutive-snapshot changes big enough to count as a move."""
    moves = []
    for a, b in zip(series, series[1:]):
        if a["labels"] != b["labels"]:
            continue
        if (datetime.fromisoformat(b["ts"]) - datetime.fromisoformat(a["ts"])).total_seconds() > MAX_GAP_MIN * 60:
            continue  # the monitor was off between these two readings: a jump across the gap is not one move
        if not all(HEALTHY_BOOK[0] <= sum(x["mid"]) <= HEALTHY_BOOK[1] for x in (a, b)):
            continue  # a thin book (prices that do not add up to about 1) makes the implied centre jump around
        scale = 9 / 5 if a["unit"] == "F" else 1.0
        d_mean = b["implied_mean"] - a["implied_mean"]
        d_sd = (b["implied_sd"] or 0) - (a["implied_sd"] or 0)
        tv = tv_distance(a["mid"], b["mid"])
        if abs(d_mean) >= MOVE_MEAN * scale or tv >= MOVE_TV:
            t = datetime.fromisoformat(b["ts"])
            hours_left = (datetime.fromisoformat(b["resolves_at"]) - t).total_seconds() / 3600
            moves.append({"event_id": b["event_id"], "city": b["city"], "ts": b["ts"], "d_mean": d_mean, "d_sd": d_sd, "tv": tv,
                          "hour_ict": t.hour, "hours_left": hours_left, "gap_min": (t - datetime.fromisoformat(a["ts"])).total_seconds() / 60})
    return moves


def convergence(preds: list[dict], series: dict[str, list[dict]], source: str) -> list[float] | None:
    """For each prediction: how far the market's average later moved toward (+) or away from (-) our centre.

    Measured in units of the original gap: +1 means the market moved all the way to our number.
    """
    out = []
    for p in preds:
        cen = (p.get(source) or {}).get("center") if source == "ml" else (p.get("estimate") or {}).get("baseline_params", {}).get("center")
        snaps = series.get(p["event_id"], [])
        if cen is None or len(snaps) < 2:
            continue
        t0 = datetime.fromisoformat(p["logged_at"])
        before = [s for s in snaps if datetime.fromisoformat(s["ts"]) <= t0]
        after = [s for s in snaps if datetime.fromisoformat(s["ts"]) > t0]
        if not before or not after:
            continue
        m0, m1 = before[-1]["implied_mean"], after[-1]["implied_mean"]
        gap = cen - m0
        scale = 9 / 5 if p["unit"] == "F" else 1.0
        if abs(gap) < 0.3 * scale:
            continue  # we agreed with the market: nothing to learn about direction
        out.append((m1 - m0) / gap)
    return out or None


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Report on Polymarket price moves")
    ap.add_argument("--snapshots", type=Path, default=SNAP_PATH)
    ap.add_argument("--log-dir", type=Path, default=plog.LOG_DIR)
    args = ap.parse_args(argv)

    rows = read_snapshots(args.snapshots)
    if not rows:
        print("no snapshots yet: run `python -m weatheragents.market_monitor --watch 30`")
        return 0
    series = series_by_event(rows)
    times = sorted(r["ts"] for r in rows)
    n_snaps = len({r["ts"] for r in rows})
    print(f"{len(rows)} readings over {n_snaps} snapshots, {times[0][5:16]} to {times[-1][5:16]} ICT, {len(series)} events")

    moves = [m for s in series.values() for m in find_moves(s)]
    print(f"\nMOVES: {len(moves)} of {sum(max(len(s) - 1, 0) for s in series.values())} snapshot-to-snapshot changes cleared the bar")
    if moves:
        by_hour = Counter(m["hour_ict"] for m in moves)
        print("  when (hour of day, Thailand time) -> moves: " + ", ".join(f"{h:02d}h:{c}" for h, c in sorted(by_hour.items())))
        bucket = lambda h: "<3h" if h < 3 else "3-6h" if h < 6 else "6-12h" if h < 12 else "12-24h" if h < 24 else "24h+"
        by_left = Counter(bucket(m["hours_left"]) for m in moves)
        print("  hours before resolution -> moves: " + ", ".join(f"{k}:{by_left[k]}" for k in ("24h+", "12-24h", "6-12h", "3-6h", "<3h") if by_left[k]))
        sharpening = sum(1 for m in moves if m["d_sd"] < -0.05 and abs(m["d_mean"]) < MOVE_MEAN)
        print(f"  {sharpening} moves were the market sharpening (spread narrowing) without the centre moving: confidence, not news about direction")
        print("\n  biggest moves:")
        for m in sorted(moves, key=lambda m: -(abs(m["d_mean"]) + m["tv"]))[:8]:
            print(f"    {m['city']:<16}{m['ts'][5:16]}  centre {m['d_mean']:+.2f}  spread {m['d_sd']:+.2f}  probability moved {m['tv']:.0%}  {m['hours_left']:.0f}h before resolution")

    preds = [p for p in plog.read(args.log_dir / plog.PREDICTIONS) if p.get("pipeline_version", "").startswith("v3")]
    print("\nDOES THE MARKET MOVE TOWARD US? (1.0 = all the way to our number, negative = away from it)")
    any_row = False
    for source, label in (("ml", "ML Analyst"), ("baseline", "Baseline curve")):
        c = convergence(preds, series, source)
        if c:
            any_row = True
            toward = sum(1 for x in c if x > 0) / len(c)
            print(f"  {label:<15} n={len(c):>3}  median {statistics.median(c):+.2f}  moved toward us {toward:.0%} of the time")
    if not any_row:
        print("  not enough overlap yet: needs snapshots both before and after a logged prediction, where we disagreed with the market")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

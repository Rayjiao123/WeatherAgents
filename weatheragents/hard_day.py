"""Can we tell in advance which days will be hard to forecast?

    python -m weatheragents.hard_day

A day's miss is mostly the size of the surprise, and most of the surprise sits in a few days (the worst 10% of days
produce half the squared error). If a warning sign reliably precedes a big miss, the forecast's stated uncertainty can
use it: honest spreads on hard days are worth money, because a bracket price that is too confident on a hard day is a
mispriced bet. For every resolved event this ranks the warning signs we log by how well they predict the size of the
miss. The most useful one for the real question, the ensemble's own spread, has only just started to build up history.

  miss          |actual high - the model's centre|, in degrees C
  warning sign  anything known before the day: ensemble spread, disagreement between forecast systems, how far the
                forecast has drifted, how many alerts are active, the model's own stated spread, the market's
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

from weatheragents import evaluate as ev
from weatheragents import prediction_log as plog
from weatheragents.market_monitor import implied_stats
from weatheragents.scout import parse_bracket

MIN_N = 30


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for q in range(i, j + 1):
            out[order[q]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def signs(p: dict) -> dict[str, float]:
    """Warning signs for one prediction, in degrees C where they are temperatures."""
    k = 9 / 5 if p["unit"] == "F" else 1.0
    out: dict[str, float] = {}
    ens = p.get("ensemble") or {}
    if ens.get("std_dev") is not None:
        out["ensemble spread (119 members)"] = ens["std_dev"] / k
    means = [m["mean"] for m in (ens.get("model_summaries") or {}).values() if m.get("mean") is not None]
    if len(means) >= 2:
        out["disagreement between ensemble systems"] = statistics.pstdev(means) / k
    nc = p.get("nowcast") or {}
    if nc.get("drift_vs_3d") is not None:
        out["forecast drift over 3 days"] = abs(nc["drift_vs_3d"]) / k
    if nc.get("lead_days") is not None:
        out["days ahead"] = float(nc["lead_days"])
    if (p.get("alerts") or {}).get("flags") is not None:
        out["active alerts"] = float(len(p["alerts"]["flags"]))
    if (p.get("ml") or {}).get("sd") is not None:
        out["the model's own stated spread"] = p["ml"]["sd"] / k
    try:
        br = [parse_bracket(l)[0] for l in p["brackets"]]
        v = ev.vectors(p)["market"]
        if v:
            out["the market's implied spread"] = implied_stats(br, v)[1] / k
    except Exception:  # an unparseable bracket just means this sign is missing
        pass
    return out


def collect(preds: dict[str, dict], outcomes: dict[str, dict]) -> list[tuple[float, dict]]:
    rows = []
    for eid, p in preds.items():
        o = outcomes.get(eid)
        actual = ev.outcome_value(o["resolved_label"]) if o else None
        if actual is None or p["kind"] != "highest":
            continue
        centre = (p.get("ml") or {}).get("center")
        if centre is None:
            centre = (p.get("estimate") or {}).get("baseline_params", {}).get("center")
        if centre is None:
            continue
        k = 9 / 5 if p["unit"] == "F" else 1.0
        rows.append((abs(actual - centre) / k, signs(p)))
    return rows


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Which warning signs predict a big miss?")
    ap.add_argument("--log-dir", type=Path, default=plog.LOG_DIR)
    args = ap.parse_args(argv)
    outcomes = {o["event_id"]: o for o in plog.read(args.log_dir / plog.OUTCOMES)}
    preds = ev.latest_before_resolution(plog.read(args.log_dir / plog.PREDICTIONS))
    rows = collect(preds, outcomes)
    print(f"{len(rows)} resolved high-temperature events with a model centre")
    names = sorted({n for _, s in rows for n in s})
    table = []
    for name in names:
        pairs = [(s[name], err) for err, s in rows if name in s]
        if len(pairs) < MIN_N:
            continue
        xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
        srt = sorted(pairs)
        third = len(srt) // 3
        low, high = statistics.fmean(b for _, b in srt[:third]), statistics.fmean(b for _, b in srt[-third:])
        table.append((spearman(xs, ys), name, len(pairs), low, high))
    if not table:
        print(f"too few so far: needs {MIN_N}+ events per sign")
        return 0
    print(f"\n{'warning sign':<42}{'n':>5}{'rank correlation':>18}{'miss when low':>15}{'miss when high':>16}")
    for rho, name, n, low, high in sorted(table, reverse=True):
        print(f"{name:<42}{n:>5}{rho:>+18.2f}{low:>14.2f}C{high:>15.2f}C")
    print("\n(rank correlation: +1 = the sign always rises with the miss, 0 = no relationship. The bottom and top thirds of each sign are compared on the right.)")
    best = max(table)
    print(f"strongest so far: {best[1]} ({best[0]:+.2f}); with {len(rows)} events a correlation within about +/-{2 / math.sqrt(len(rows)):.2f} of zero is indistinguishable from chance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline evaluator: scores logged predictions against recorded outcomes.

    python -m weatheragents.evaluate
    python -m weatheragents.evaluate --final-only --by city

For each resolved event it uses the latest prediction logged before the
event resolved, and scores every source's probability vector on the same
event: the Manager (estimate), the raw ensemble, climatology, and the market's
own implied vector. The market is the bar to clear, not just accuracy.

Lower is better for log loss, RPS and Brier. All sources share the same
probability floor so none is rewarded for a hard zero.
"""

import argparse
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from weatheragents import prediction_log as plog

FLOOR = 1e-3
SOURCES = ("manager", "ml", "ml_fresh", "sameday_ml", "sameday", "baseline_plus", "baseline", "baseline_raw", "ensemble", "climatology", "market")


def normalize(vec: list[float]) -> list[float]:
    vec = [max(v, FLOOR) for v in vec]
    total = sum(vec)
    return [v / total for v in vec]


def market_vector(pred: dict) -> list[float] | None:
    """Implied probability per bracket: bid/ask mid, else last trade, else whichever quote exists.

    A bracket with no quote at all is deep out of the money and counts as ~0, so one empty
    tail doesn't throw the whole event away. Returns None only if no bracket has a price.
    """
    by_label = {b["label"]: b for b in pred["market_snapshot"]["brackets"]}
    out = []
    for label in pred["brackets"]:
        b = by_label.get(label)
        if b is None:
            return None
        bid, ask, last = b["best_bid"], b["best_ask"], b["last_trade"]
        if bid is not None and ask is not None:
            out.append((bid + ask) / 2)
        else:
            out.append(next((x for x in (last, ask, bid) if x is not None), 0.0))
    return normalize(out) if sum(out) > 0 else None


def vectors(pred: dict) -> dict[str, list[float] | None]:
    labels = pred["brackets"]

    def pick(rows):
        d = {r["label"]: r["probability"] for r in rows}
        return normalize([d[l] for l in labels]) if set(labels) <= set(d) else None

    est = pred["estimate"]
    return {
        "manager": pick(est["bracket_probabilities"]) if est.get("bracket_probabilities") else None,  # none in numeric mode
        # v3+: the ML Analyst's own curve, scored separately from the LLM agents that also see it.
        "ml": pick(pred["ml"]["bracket_probabilities"]) if pred.get("ml") else None,
        # the same model fed the newest weather-model runs instead of the day-ahead ones
        "ml_fresh": pick(pred["ml_fresh"]["bracket_probabilities"]) if pred.get("ml_fresh") else None,
        # events whose day was still in progress when predicted: the high projected from the station's own reports
        "sameday": pick(pred["sameday"]["bracket_probabilities"]) if pred.get("sameday") else None,
        # the same projection learned from history instead of hand-built: scored against the heuristic above
        "sameday_ml": pick(pred["sameday_ml"]["bracket_probabilities"]) if pred.get("sameday_ml") else None,
        # v2+: the statistical curve before the LLM's adjustment, with and without station-bias correction.
        # Comparing manager vs baseline measures what the LLM adds; baseline vs baseline_raw, the bias fit.
        "baseline": pick(est["baseline_probabilities"]) if est.get("baseline_probabilities") else None,
        # the baseline with last week's miss leaned on: a fairer yardstick for the ML (a separate source; the baseline stays fixed)
        "baseline_plus": pick(est["baseline_plus_probabilities"]) if est.get("baseline_plus_probabilities") else None,
        "baseline_raw": pick(est["baseline_uncorrected_probabilities"]) if est.get("baseline_uncorrected_probabilities") else None,
        "ensemble": pick(pred["ensemble"]["bracket_probabilities"]),
        "climatology": pick(pred["climatology"]["bracket_probabilities"]),
        "market": market_vector(pred),
    }


def log_loss(vec: list[float], k: int) -> float:
    return -math.log(vec[k])


def brier(vec: list[float], k: int) -> float:
    return sum((p - (1.0 if i == k else 0.0)) ** 2 for i, p in enumerate(vec))


def rps(vec: list[float], k: int) -> float:
    """Ranked probability score: brackets are ordered, so a near miss costs less than a far one."""
    cum_p = cum_o = total = 0.0
    for i, p in enumerate(vec[:-1]):
        cum_p += p
        cum_o += 1.0 if i == k else 0.0
        total += (cum_p - cum_o) ** 2
    return total / (len(vec) - 1)


def version_of(pred: dict) -> str:
    return pred.get("pipeline_version", "v1")


def latest_before_resolution(preds: list[dict]) -> dict[str, dict]:
    """Latest pre-resolution prediction per event. Filter by version first to compare versions."""
    chosen: dict[str, dict] = {}
    for p in preds:
        if datetime.fromisoformat(p["logged_at"]) >= datetime.fromisoformat(p["resolves_at"]):
            continue  # made after the fact; doesn't count
        cur = chosen.get(p["event_id"])
        if cur is None or datetime.fromisoformat(p["logged_at"]) > datetime.fromisoformat(cur["logged_at"]):
            chosen[p["event_id"]] = p
    return chosen


def score_events(preds: dict[str, dict], outcomes: dict[str, dict]) -> list[dict]:
    rows = []
    for eid, p in preds.items():
        o = outcomes.get(eid)
        if o is None or o["resolved_label"] not in p["brackets"]:
            continue
        k = p["brackets"].index(o["resolved_label"])
        row = {"event_id": eid, "city": p["city"], "kind": p["kind"], "unit": p["unit"], "k": k, "status": o["status"],
               "version": p.get("pipeline_version", "v1")}
        for name, vec in vectors(p).items():
            row[name] = None if vec is None else {"ll": log_loss(vec, k), "rps": rps(vec, k), "brier": brier(vec, k), "hit": max(range(len(vec)), key=vec.__getitem__) == k, "p_true": vec[k]}
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for name in SOURCES:
        sub = [r[name] for r in rows if r.get(name)]
        if sub:
            n = len(sub)
            out[name] = {
                "n": n,
                "log_loss": sum(x["ll"] for x in sub) / n,
                "rps": sum(x["rps"] for x in sub) / n,
                "brier": sum(x["brier"] for x in sub) / n,
                "top_hit": sum(x["hit"] for x in sub) / n,
            }
    return out


def calibration(preds: dict[str, dict], outcomes: dict[str, dict], source: str = "manager", bins: int = 5) -> list[tuple[float, float, float, int]]:
    """(bin low, mean predicted, observed frequency, count) over every bracket of every event."""
    pairs = []
    for eid, p in preds.items():
        o = outcomes.get(eid)
        vec = vectors(p)[source]
        if o is None or vec is None or o["resolved_label"] not in p["brackets"]:
            continue
        k = p["brackets"].index(o["resolved_label"])
        pairs += [(v, 1.0 if i == k else 0.0) for i, v in enumerate(vec)]
    edges = [0, 0.02, 0.1, 0.25, 0.5, 1.0001][: bins + 1]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sub = [(p, y) for p, y in pairs if lo <= p < hi]
        if sub:
            out.append((lo, sum(p for p, _ in sub) / len(sub), sum(y for _, y in sub) / len(sub), len(sub)))
    return out


def fmt_table(title: str, summary: dict[str, dict]) -> str:
    lines = [title, f"  {'source':<13}{'n':>4}{'log loss':>10}{'RPS':>8}{'Brier':>8}{'top-bracket hit':>17}"]
    for name, s in summary.items():
        lines.append(f"  {name:<13}{s['n']:>4}{s['log_loss']:>10.3f}{s['rps']:>8.3f}{s['brier']:>8.3f}{s['top_hit']:>17.0%}")
    return "\n".join(lines)


def print_version_comparison(all_preds: list[dict], outcomes: dict[str, dict]) -> None:
    """Each pipeline version scored on its own latest prediction, on events every version predicted."""
    by_version = defaultdict(list)
    for p in all_preds:
        by_version[version_of(p)].append(p)
    scored = {v: score_events(latest_before_resolution(ps), outcomes) for v, ps in by_version.items()}
    scored = {v: rows for v, rows in scored.items() if rows}
    print("\nBY VERSION")
    if len(scored) < 2:
        print("  fewer than two versions have resolved events yet: " + (", ".join(f"{v} (n={len(r)})" for v, r in scored.items()) or "none"))
        return
    shared = set.intersection(*({r["event_id"] for r in rows} for rows in scored.values()))
    print(f"  {len(shared)} events were predicted by every version: same events, apples to apples")
    for v, rows in sorted(scored.items()):
        print("\n" + fmt_table(f"{v} on shared events", summarize([r for r in rows if r["event_id"] in shared])))


def outcome_value(label: str) -> float | None:
    """The temperature a resolved bracket stands for: its degree, or the middle of a Fahrenheit pair.
    Open-ended brackets ("19C or higher") are censored (we only know a bound), so they return None."""
    import re

    if re.search(r"or (higher|below)", label):
        return None
    m = re.match(r"^(-?\d+)-(-?\d+)", label)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    n = re.findall(r"-?\d+", label)
    return float(n[0]) if n else None


def city_bias(preds: dict[str, dict], outcomes: dict[str, dict], source: str = "ml") -> list[dict]:
    """Per city: how far the real high sat from a source's centre, in degrees C (positive = the model ran too cold)."""
    by_city: dict[str, list[float]] = defaultdict(list)
    for eid, p in preds.items():
        o = outcomes.get(eid)
        if o is None or p["kind"] != "highest":
            continue
        centre = (p.get(source) or {}).get("center") if source == "ml" else None
        actual = outcome_value(o["resolved_label"])
        if centre is None or actual is None:
            continue
        k = 9 / 5 if p["unit"] == "F" else 1.0
        by_city[p["city"]].append((actual - centre) / k)
    out = []
    for city, v in by_city.items():
        n = len(v)
        mean = sum(v) / n
        sd = (sum((x - mean) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        se = sd / n ** 0.5 if n > 1 else float("nan")
        out.append({"city": city, "n": n, "bias": mean, "se": se, "t": mean / se if se and se == se and se > 0 else 0.0})
    return sorted(out, key=lambda r: -abs(r["t"]))


def print_bias(preds: dict[str, dict], outcomes: dict[str, dict], min_n: int = 5) -> None:
    rows = city_bias(preds, outcomes)
    print("\nLEFTOVER BIAS BY CITY (live ML: actual high minus the model's centre, degrees C; + means it ran too cold)")
    if not rows:
        print("  none yet: needs resolved events predicted by the v3 pipeline")
        return
    allv = sum(r["bias"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
    print(f"  all cities: {allv:+.2f} C over {sum(r['n'] for r in rows)} events")
    shown = [r for r in rows if r["n"] >= min_n]
    for r in shown[:10]:
        flag = "  <- off by more than chance" if abs(r["t"]) >= 2 and r["n"] >= 10 else ""
        print(f"  {r['city']:<18}{r['bias']:>+6.2f} C  (+/-{2 * r['se']:.2f}, n={r['n']}){flag}")
    sig = [r for r in rows if abs(r["t"]) >= 2 and r["n"] >= 10]
    print(f"  {len(sig)} of {len([r for r in rows if r['n'] >= 10])} cities with 10+ events are off by more than chance: " + ("a per-city corrector is worth fitting" if sig else "a corrector would be fitting noise"))


def print_blend(preds: dict[str, dict], outcomes: dict[str, dict]) -> None:
    """Log loss of w*market + (1-w)*source. If a source never improves on w=1, it holds nothing the market lacks."""
    sources = ("ml", "ml_fresh", "sameday_ml", "sameday", "baseline_plus", "baseline", "manager")
    rows: dict[str, list] = {s: [] for s in sources}
    for eid, p in preds.items():
        o = outcomes.get(eid)
        if o is None or o["resolved_label"] not in p["brackets"]:
            continue
        v = vectors(p)
        k = p["brackets"].index(o["resolved_label"])
        for s in sources:
            if v["market"] and v[s]:
                rows[s].append((v, k))
    live = [s for s in sources if rows[s]]
    print("\nBLEND WITH THE MARKET (log loss of w*market + (1-w)*source, on events where both exist; lower is better)")
    if not live:
        print("  none yet: needs resolved events predicted by the v3 pipeline")
        return
    print(f"  {'weight on market':<18}" + "".join(f"{s + ' (n=' + str(len(rows[s])) + ')':>16}" for s in live))
    for w in (0.0, 0.5, 0.7, 0.8, 0.9, 1.0):
        cells = []
        for s in live:
            ll = sum(-math.log(max(w * v["market"][k] + (1 - w) * v[s][k], FLOOR)) for v, k in rows[s]) / len(rows[s])
            cells.append(f"{ll:>16.3f}")
        print(f"  {w:<18.1f}" + "".join(cells))
    print("  (w=1.0 is the market alone; a source holds something the market lacks only if some w<1 beats that row)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score logged predictions against outcomes")
    parser.add_argument("--log-dir", type=Path, default=plog.LOG_DIR)
    parser.add_argument("--final-only", action="store_true", help="ignore provisional outcomes")
    parser.add_argument("--version", default=None, help="only predictions from this pipeline version (e.g. v1)")
    parser.add_argument("--by", choices=["city", "kind", "unit", "version"], default=None, help="also break results down by this field")
    parser.add_argument("--bias", action="store_true", help="per-city leftover bias of the live ML forecast, and whether it is more than chance")
    parser.add_argument("--blend", action="store_true", help="test whether mixing each source into the market beats the market alone")
    parser.add_argument("--min-n", type=int, default=5, help="hide slices smaller than this (default 5)")
    args = parser.parse_args(argv)

    all_preds = plog.read(args.log_dir / plog.PREDICTIONS)
    if args.version:
        all_preds = [p for p in all_preds if version_of(p) == args.version]
    preds = latest_before_resolution(all_preds)
    outcomes = {}
    for o in plog.read(args.log_dir / plog.OUTCOMES):
        outcomes[o["event_id"]] = o
    if args.final_only:
        outcomes = {k: o for k, o in outcomes.items() if o["status"] == "final"}

    rows = score_events(preds, outcomes)
    print(f"{len(preds)} events predicted, {len(rows)} resolved and scored")
    if not rows:
        print("nothing to score yet: run the pipeline, wait for events to resolve, then run `python -m weatheragents.resolver`")
        return 0

    print("\n" + fmt_table("ALL EVENTS (lower is better)", summarize(rows)))
    s = summarize(rows)
    for name in ("manager", "ml", "ml_fresh", "baseline_plus", "baseline"):
        # same events only: the market's score on all events would not be comparable with a source that covers fewer
        both = [r for r in rows if r.get(name) and r.get("market")]
        if both:
            diffs = [r[name]["ll"] - r["market"]["ll"] for r in both]
            d = sum(diffs) / len(diffs)
            se = (sum((x - d) ** 2 for x in diffs) / max(len(diffs) - 1, 1)) ** 0.5 / len(diffs) ** 0.5
            verdict = "beats the market" if d + 1.96 * se < 0 else "does NOT beat the market" if d - 1.96 * se > 0 else "is not clearly different from the market"
            print(f"\n  {name} vs market on the same {len(both)} events: {d:+.3f} log loss (95% range {d - 1.96 * se:+.3f} to {d + 1.96 * se:+.3f}); {name} {verdict}. Negative = {name} better.")

    if args.blend:
        print_blend(preds, outcomes)
    if args.bias:
        print_bias(preds, outcomes, args.min_n)

    cal_src = next((n for n in ("manager", "ml", "baseline") if n in s), "baseline")
    print(f"\n{cal_src.upper()} CALIBRATION (predicted vs how often that bracket won)")
    for lo, mean_p, freq, n in calibration(preds, outcomes, source=cal_src):
        print(f"  predicted ~{mean_p:>5.1%}   observed {freq:>5.1%}   (n={n})")

    if args.by == "version":
        print_version_comparison(all_preds, outcomes)
    elif args.by:
        groups = defaultdict(list)
        for r in rows:
            groups[r[args.by]].append(r)
        print(f"\nBY {args.by.upper()} (slices under n={args.min_n} hidden: too noisy to read)")
        for key, grp in sorted(groups.items()):
            if len(grp) >= args.min_n:
                print("\n" + fmt_table(f"{key}", summarize(grp)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

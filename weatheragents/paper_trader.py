"""Paper trading: what would betting on our forecasts at the open have made, after paying the spread?

    python -m weatheragents.paper_trader
    python -m weatheragents.paper_trader --source ml_fresh --stake 10

No real money and no orders: this replays logged predictions against the logged prices and Polymarket's results.

For each event it takes the EARLIEST prediction we logged (as close to the market opening as the logs allow) and the
prices quoted at that moment, finds the one bracket where our probability disagrees most with the price, and "buys" it:

  buy YES  when our probability is above the ask      edge = P(yes) - ask
  buy NO   when our probability is below the bid      edge = bid - P(yes)

Fills are honest: you pay the ask, or (for NO) one minus the bid, so the spread is a real cost, and the weather-market taker
fee (0.05 x p x (1 - p) per share, on buys and on sales; none at settlement) is charged. A second line shows the optimistic
case of getting filled at the middle price with a resting order, which pays no fee. Each trade is held to resolution (YES pays
1 per share if its bracket wins, NO pays 1 if it loses), and also marked to the market at fixed times before resolution,
where you would have to sell at the bid, to test the idea of trading the price move itself.

Edges are only counted when they exceed a threshold, so you can see whether bigger disagreements pay more.
"""

import argparse
import math
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

from weatheragents import evaluate as ev
from weatheragents import prediction_log as plog
from weatheragents.market_monitor import SNAP_PATH, read_snapshots

THRESHOLDS = (0.03, 0.05, 0.08, 0.12)
EXIT_HOURS = (24, 12, 6)
PRIMARY_HOURS = 12  # fixed in advance in PREREGISTERED.md
CURVE_HOURS = (3, 6, 12, 18, 24, 36)
WEATHER_FEE_RATE = 0.05  # Polymarket weather markets: takers pay shares * rate * p * (1 - p) in USDC, makers nothing (from the market feeSchedule)
FEE_RATE = WEATHER_FEE_RATE
MIN_PRICE, MAX_PRICE = 0.02, 0.98  # nothing is bought at a price where a one-cent quote error is the whole edge


def fee(price: float, rate: float | None = None) -> float:
    """Taker fee per share at this price."""
    return (FEE_RATE if rate is None else rate) * price * (1 - price)


def entry_vector(p: dict, source: str) -> list[float] | None:
    return ev.vectors(p).get(source)


def best_trade(p: dict, vec: list[float]) -> dict | None:
    """The single bracket with the biggest edge after crossing the spread, or None."""
    by_label = {b["label"]: b for b in p["market_snapshot"]["brackets"]}
    best = None
    for label, prob in zip(p["brackets"], vec):
        q = by_label.get(label)
        if not q:
            continue
        bid, ask = q["best_bid"], q["best_ask"]
        if ask is not None and MIN_PRICE <= ask <= MAX_PRICE:
            cand = {"label": label, "side": "YES", "price": ask, "mid_price": (ask + bid) / 2 if bid is not None else ask, "edge": prob - ask,
                    "spread": None if bid is None else ask - bid, "bracket_liquidity": q.get("liquidity")}
            if best is None or cand["edge"] > best["edge"]:
                best = cand
        if bid is not None and MIN_PRICE <= 1 - bid <= MAX_PRICE:
            cand = {"label": label, "side": "NO", "price": 1 - bid, "mid_price": 1 - ((ask + bid) / 2 if ask is not None else bid), "edge": bid - prob,
                    "spread": None if ask is None else ask - bid, "bracket_liquidity": q.get("liquidity")}
            if best is None or cand["edge"] > best["edge"]:
                best = cand
    return best


def resolve_trade(t: dict, winner: str) -> float:
    """Return on one unit staked, held to resolution, at the price paid."""
    wins = (t["label"] == winner) if t["side"] == "YES" else (t["label"] != winner)
    return ((1.0 if wins else 0.0) - t["price"] - t.get("fee_in", 0.0)) / t["price"]  # settlement pays 1 or 0 with no fee


def sell_price(t: dict, snap: dict) -> float | None:
    """What selling now would fetch: YES sells at the bid, NO sells at one minus the ask."""
    if t["label"] not in snap["labels"]:
        return None
    i = snap["labels"].index(t["label"])
    bid, ask = snap["bid"][i], snap["ask"][i]
    return bid if t["side"] == "YES" else (None if ask is None else 1 - ask)


def mark_trade(t: dict, snap: dict) -> float | None:
    """Return if instead sold now."""
    sell = sell_price(t, snap)
    return None if sell is None else (sell - fee(sell) - t["price"] - t.get("fee_in", 0.0)) / t["price"]


def exit_return(t: dict, rule: tuple) -> float:
    """Return under an exit rule, walking the saved price path; holds to resolution if the rule never fires.

    ("tp", up, down): sell once the sale price is `up` above the entry price (or `down` below it).
    ("hours", n): sell at the first quote n hours or more after entry.
    ("fair", margin): sell once the sale price reaches our own probability minus the margin (the gap has closed).
    """
    cost = t["price"] + t.get("fee_in", 0.0)
    for hrs, bid in t["path"]:
        sell = bid - fee(bid)  # what the sale nets after the taker fee
        kind = rule[0]
        fire = (kind == "tp" and (sell - cost >= rule[1] or cost - sell >= rule[2])
                or kind == "hours" and hrs >= rule[1]
                or kind == "fair" and sell >= t["price"] + t["edge"] - rule[1])
        if fire:
            return (sell - cost) / t["price"]
    return t["held"]


EXIT_RULES = [
    ("take profit at +10c, stop at -10c", ("tp", 0.10, 0.10)),
    ("take profit at +5c, stop at -5c", ("tp", 0.05, 0.05)),
    ("take profit at +10c, no stop", ("tp", 0.10, 9.0)),
    ("sell 6 hours after entry", ("hours", 6)),
    ("sell 12 hours after entry", ("hours", 12)),
    ("sell when price reaches our value (within 2c)", ("fair", 0.02)),
]


def snapshot_near(rs: list[dict], when: datetime, after: datetime) -> dict | None:
    cands = [r for r in rs if datetime.fromisoformat(r["ts"]) >= after]
    best = min(cands, key=lambda r: abs((datetime.fromisoformat(r["ts"]) - when).total_seconds()), default=None)
    return best if best and abs((datetime.fromisoformat(best["ts"]) - when).total_seconds()) <= 2700 else None  # within 45 minutes


def summarize(rets: list[float]) -> str:
    if not rets:
        return "-"
    m = statistics.fmean(rets)
    se = statistics.pstdev(rets) / math.sqrt(len(rets)) if len(rets) > 1 else float("nan")
    return f"{m:+.1%} (+/-{2 * se:.0%})"


def run(preds: dict[str, dict], outcomes: dict[str, dict], snaps: dict[str, list[dict]], source: str) -> list[dict]:
    trades = []
    for eid, p in preds.items():
        o = outcomes.get(eid)
        vec = entry_vector(p, source)
        if o is None or vec is None or not p.get("market_snapshot"):
            continue
        t = best_trade(p, vec)
        if t is None:
            continue
        t0 = datetime.fromisoformat(p["logged_at"])
        res = datetime.fromisoformat(p["resolves_at"])
        rs = sorted(snaps.get(eid, []), key=lambda r: r["ts"])
        t["fee_in"] = fee(t["price"])  # buying at the ask is a taker trade
        t.update(event=eid, city=p["city"], held=resolve_trade(t, o["resolved_label"]), mid_held=resolve_trade({**t, "price": t["mid_price"], "fee_in": 0.0}, o["resolved_label"]),
                 hours_ahead=(res - t0).total_seconds() / 3600)
        t["path"] = [((datetime.fromisoformat(r["ts"]) - t0).total_seconds() / 3600, sp) for r in rs
                     if datetime.fromisoformat(r["ts"]) > t0 and (sp := sell_price(t, r)) is not None]
        for h in EXIT_HOURS:
            snap = snapshot_near(rs, res - timedelta(hours=h), after=t0)
            t[f"exit{h}"] = mark_trade(t, snap) if snap else None
        trades.append(t)
    return trades


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Paper-trade the logged predictions")
    ap.add_argument("--source", default="ml", choices=["ml", "ml_fresh", "baseline_plus", "baseline", "manager"])
    ap.add_argument("--stake", type=float, default=10.0, help="dollars per trade, for the profit column")
    ap.add_argument("--log-dir", type=Path, default=plog.LOG_DIR)
    args = ap.parse_args(argv)

    outcomes = {o["event_id"]: o for o in plog.read(args.log_dir / plog.OUTCOMES)}
    allp = [p for p in plog.read(args.log_dir / plog.PREDICTIONS) if p["kind"] == "highest" and entry_vector(p, args.source)]
    first: dict[str, dict] = {}
    for p in allp:  # the earliest prediction per event: the closest we have to "at the open"
        if p["event_id"] not in first or p["logged_at"] < first[p["event_id"]]["logged_at"]:
            first[p["event_id"]] = p
    snaps: dict[str, list[dict]] = {}
    for r in read_snapshots(SNAP_PATH):
        snaps.setdefault(r["event_id"], []).append(r)
    trades = run(first, outcomes, snaps, args.source)
    print(f"source: {args.source} | {len(first)} events predicted with it, {len(trades)} resolved with a quoted price to trade against")
    if not trades:
        print("nothing to trade yet")
        return 0
    lead = statistics.median(t["hours_ahead"] for t in trades)
    print(f"entry was a median {lead:.0f} hours before resolution. Returns are per dollar staked, after paying the spread and the taker fee (a resting order at the mid price pays none).\n")
    print(f"{'edge at least':<14}{'trades':>7}{'hit rate':>10}{'held to the end':>20}{'at mid price':>20}  " + "".join(f"{'sold ' + str(h) + 'h out':>18}" for h in EXIT_HOURS))
    for th in THRESHOLDS:
        sel = [t for t in trades if t["edge"] >= th]
        if not sel:
            continue
        hit = statistics.fmean(1.0 if t["held"] > 0 else 0.0 for t in sel)
        exits = "".join(f"{summarize([t[f'exit{h}'] for t in sel if t[f'exit{h}'] is not None]):>18}" for h in EXIT_HOURS)
        print(f"{th * 100:>6.0f} cents  {len(sel):>7}{hit:>10.0%}{summarize([t['held'] for t in sel]):>20}{summarize([t['mid_held'] for t in sel]):>20}  {exits}")
    sel = [t for t in trades if t["edge"] >= THRESHOLDS[1]]
    if sel:
        print()
        print(f"exit rules on the trades with an edge of 5 cents or more ({len(sel)} trades; a rule that never fires holds to the end):")
        for name, rule in EXIT_RULES:
            print(f"  {name:<48}{summarize([exit_return(t, rule) for t in sel]):>18}")
        print(f"  {'hold to resolution':<48}{summarize([t['held'] for t in sel]):>18}")
        print()
        print("return by holding time (a diagnostic of how long any mispricing lasts, not a menu to pick the best from):")
        for h in CURVE_HOURS:
            print(f"  sell {h:>2} hours after entry{summarize([exit_return(t, ('hours', h)) for t in sel]):>40}")
    prim = [exit_return(t, ("hours", PRIMARY_HOURS)) for t in sel]
    print()
    print(f"PRE-REGISTERED TEST ({args.source}, edge >= 5c, sell {PRIMARY_HOURS}h after entry): {len(sel)} trades of the 100 needed. {summarize(prim)}")
    if sel:
        pl = sum(t["held"] for t in sel) * args.stake
        print(f"\nstaking ${args.stake:.0f} on each trade with an edge of 5 cents or more: {len(sel)} trades, profit ${pl:+,.0f} held to resolution")
    print("\n(+/- is two standard errors. With few trades the range will be wide; a result is real only if the range stays above zero.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

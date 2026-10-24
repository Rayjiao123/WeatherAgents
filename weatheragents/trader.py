"""The Trader: the one agent that sees both our probabilities and the market's prices. Paper only.

    python -m weatheragents.trader --watch 5      # poll live prices; every paper account trades on the same quotes
    python -m weatheragents.trader --report       # profit and loss for every account

It never places a real order. Each account keeps a ledger in data/trader/<name>.jsonl of what it WOULD have done at
the live quotes. Several accounts run side by side so each added rule is measured against the control:

  control    the rule fixed in advance: edge >= 5c, pay the ask, $10, sell 12 hours later
  careful    adds the filters: skip when the model is unsure, when the spread eats the edge, when the price is already
             running against us, and when too many trades pile up in one region
  patient    careful, plus a resting limit order inside the spread instead of paying the ask
  smart      patient, plus Kelly-style sizing and the full exit logic below
  flip       the clean opening test: buy at the open only (prediction under 30 minutes old), sell the moment the sale nets
             any profit after both fees, no stop, and sell at 12 hours if still not in profit
  quick      the control's entries, but sells the moment the sale price is 1 cent above what we paid (the spread is
             already paid: we buy at the ask and sell at the bid), with a 10 cent stop and the 12 hour limit

Risk gate (plain code, checked before any new trade; exits always run): no new trades while stakes at risk reach `max_exposure`,
after losing `daily_loss_limit` on trades closed today, after falling `kill_drawdown` from the best realized profit, or while a file
named KILL exists in data/trader. There is no agent debate about size: caps are code an agent cannot argue around.

Exit logic (checked at every poll for every open position, first rule that fires wins):
  settle   the market resolved: collect 1 or 0
  profit   (quick only) the best sale price is `take_profit` above what we paid: bank it
  stop     the best sale price fell `stop_loss` below what we paid: our read was wrong, cut it
  target   the best sale price is within `target_margin` of our value: the market has caught up, the edge is gone
  stale    what we still expect to gain by holding (our value minus the sale price) is under `min_hold_edge`: not worth the risk
  time     held `exit_hours`: free the money, and avoid sitting through the last hours, when the market is sharpest
  "our value" is our own probability shrunk toward the market's price by `fair_shrink` (1 = fully trust ourselves),
  because history says the market is usually the better judge.
"""

import argparse
import contextlib
import io
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from weatheragents import prediction_log as plog
from weatheragents.market_monitor import SNAP_PATH
from weatheragents.paper_trader import WEATHER_FEE_RATE, best_trade, entry_vector, fee
from weatheragents.scout import fetch_events, parse_event
from weatheragents.timeutil import now as ict_now

TRADER_DIR = Path("data") / "trader"
DRIFT_HOURS = 3.0


@dataclass
class Policy:
    name: str = "control"
    source: str = "ml"
    # entry
    min_edge: float = 0.05
    max_spread: float | None = None  # skip brackets quoted wider than this
    cost_mult: float | None = None  # require edge >= this many times the spread
    min_bracket_liquidity: float = 0.0
    max_sd: float | None = None  # skip when the model itself says it is unsure (sd in the contract's unit)
    max_adverse_drift: float | None = None  # skip when the side's price moved this much against us in the last 3 hours
    max_per_region: int | None = None  # open plus waiting trades sharing a station-code prefix
    max_entry_age_hours: float = 6.0  # the prediction must be this fresh
    only_at_open: bool = False  # trade only forecasts made when the market was funded, never catch-up refreshes
    max_new_per_day: int = 40
    max_open: int = 120
    # limit orders
    maker: bool = False
    maker_improve: float = 0.5  # rest at the ask minus this share of the spread
    maker_wait_hours: float = 3.0
    # size
    sizing: str = "fixed"  # or "kelly"
    stake: float = 10.0
    bankroll: float = 1000.0
    kelly_frac: float = 0.25
    min_stake: float = 2.0
    max_stake: float = 25.0
    # exit
    exit_hours: float | None = 12.0
    stop_loss: float | None = None
    take_profit: float | None = None  # sell as soon as the sale price is this far above what we paid (the spread is already in both prices)
    target_margin: float | None = None
    min_hold_edge: float | None = None
    fair_shrink: float = 1.0
    fee_rate: float = WEATHER_FEE_RATE
    # risk gate (plain code, not an agent): in pretend dollars per account
    max_exposure: float = 400.0  # stakes in open positions and waiting orders
    daily_loss_limit: float = 50.0  # stop opening trades for the day after losing this much on trades closed today
    kill_drawdown: float = 150.0  # stop opening trades after falling this far from the best realized profit


def presets() -> dict[str, Policy]:
    control = Policy()
    careful = replace(control, name="careful", max_spread=0.10, cost_mult=1.5, max_sd=2.0, max_adverse_drift=0.05, max_per_region=6, min_bracket_liquidity=300)
    patient = replace(careful, name="patient", maker=True)
    smart = replace(patient, name="smart", sizing="kelly", stop_loss=0.12, target_margin=0.03, min_hold_edge=0.02, fair_shrink=0.5, exit_hours=24.0)
    quick = replace(control, name="quick", take_profit=0.01, stop_loss=0.10)
    flip = replace(control, name="flip", max_entry_age_hours=0.5, take_profit=0.001, only_at_open=True)
    return {p.name: p for p in (control, careful, patient, smart, quick, flip)}


def side_prices(side: str, quote: dict) -> tuple[float | None, float | None, float | None]:
    """(price to buy at, price to sell at, middle) for the YES or NO side of one bracket."""
    bid, ask = quote["best_bid"], quote["best_ask"]
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None
    if side == "YES":
        return ask, bid, mid
    return (None if bid is None else 1 - bid), (None if ask is None else 1 - ask), (None if mid is None else 1 - mid)


def decide(p: dict, quotes: dict, policy: Policy, rate: float = WEATHER_FEE_RATE) -> dict | None:
    """The trade this policy would make, or None. `quotes` maps bracket label -> {best_bid, best_ask, liquidity}."""
    vec = entry_vector(p, policy.source)
    if vec is None:
        return None
    priced = {"brackets": p["brackets"], "market_snapshot": {"brackets": [{"label": k, **v} for k, v in quotes.items()]}}
    t = best_trade(priced, vec)
    if t is None or t["edge"] < policy.min_edge:
        return None
    if policy.max_spread is not None and (t["spread"] is None or t["spread"] > policy.max_spread):
        return None
    if policy.cost_mult is not None and (t["spread"] is None or t["edge"] < policy.cost_mult * t["spread"]):
        return None
    if (t["bracket_liquidity"] or 0.0) < policy.min_bracket_liquidity:
        return None
    sd = (p.get(policy.source) or {}).get("sd")
    if policy.max_sd is not None and (sd is None or sd > policy.max_sd):
        return None
    t["fair"] = t["price"] + t["edge"]
    t["edge_gross"] = t["edge"]
    t["edge"] = t["edge"] - fee(t["price"], rate)  # paying the ask is a taker trade
    if t["edge"] < policy.min_edge:
        return None
    return t


def stake_for(t: dict, price: float, policy: Policy) -> float:
    """Fixed, or a fraction of Kelly: the bet size that maximises growth if our probability is right."""
    if policy.sizing != "kelly":
        return policy.stake
    f = max(0.0, (t["fair"] - price - t.get("fee_in", 0.0)) / (1 - price))
    return round(min(policy.max_stake, max(policy.min_stake, policy.bankroll * policy.kelly_frac * f)), 2)


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(rec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def positions(ledger: list[dict]) -> dict[str, dict]:
    """Opening records keyed by event, each with a `closed` entry once it has been closed."""
    out: dict[str, dict] = {}
    for r in ledger:
        if r["action"] == "open":
            legacy_fee = 0.0 if r.get("entry") == "limit order filled" else fee(r["price"], WEATHER_FEE_RATE)  # trades logged before fees were modelled
            out[r["event_id"]] = {"fair": r["price"] + r["edge"], "fee_in": legacy_fee, **r, "closed": None}
        elif r["action"] == "close" and r["event_id"] in out:
            out[r["event_id"]]["closed"] = r
    return out


def waiting_orders(ledger: list[dict]) -> dict[str, dict]:
    """Limit orders neither filled nor cancelled."""
    out = {}
    for r in ledger:
        if r["action"] == "order":
            out[r["event_id"]] = r
        elif r["action"] in ("open", "cancel"):
            out.pop(r["event_id"], None)
    return out


def net_return(ps: dict, proceeds_per_share: float) -> float:
    """Profit per dollar of price paid: what a share finally brings in (after any sale fee) less what it cost (price plus any entry fee)."""
    return (proceeds_per_share - ps["price"] - ps.get("fee_in", 0.0)) / ps["price"]


def settle(pos: dict, winner: str) -> float:
    wins = (pos["label"] == winner) if pos["side"] == "YES" else (pos["label"] != winner)
    return 1.0 if wins else 0.0


def exit_reason(ps: dict, quote: dict, age_h: float, policy: Policy, rate: float | None = None) -> tuple[str, float] | None:
    """(reason, sale price) if an exit rule fires now, else None. Rules compare what the sale would really net after the fee."""
    rate = policy.fee_rate if rate is None else rate
    _, bid_side, mid = side_prices(ps["side"], quote)
    if bid_side is None:
        return None
    sell = bid_side - fee(bid_side, rate)
    cost = ps["price"] + ps.get("fee_in", 0.0)
    fair = ps["fair"] if mid is None else policy.fair_shrink * ps["fair"] + (1 - policy.fair_shrink) * mid
    if policy.stop_loss is not None and sell <= cost - policy.stop_loss:
        return "stop", bid_side
    if policy.take_profit is not None and sell - cost >= policy.take_profit:
        return "profit", bid_side
    if policy.target_margin is not None and sell >= fair - policy.target_margin:
        return "target", bid_side
    if policy.min_hold_edge is not None and age_h >= 1.0 and fair - sell < policy.min_hold_edge:
        return "stale", bid_side
    if policy.exit_hours is not None and age_h >= policy.exit_hours:
        return f"time ({policy.exit_hours:g}h)", bid_side
    return None


def region(p: dict) -> str:
    return (p.get("station_code") or "????")[:2]


def risk_block(pos: dict[str, dict], orders: dict[str, dict], now: datetime, policy: Policy) -> str | None:
    """Why no new trade may open right now, or None. Exits and settlement always run."""
    closed = sorted((p for p in pos.values() if p["closed"]), key=lambda p: p["closed"]["at"])
    peak = run = worst = 0.0
    for p in closed:
        run += p["stake"] * p["closed"]["ret"]
        peak = max(peak, run)
        worst = max(worst, peak - run)
    today = now.date().isoformat()
    lost_today = -sum(p["stake"] * p["closed"]["ret"] for p in closed if p["closed"]["at"][:10] == today)
    exposure = sum(p["stake"] for p in pos.values() if not p["closed"]) + policy.stake * len(orders)
    if peak - run >= policy.kill_drawdown:
        return f"drawdown kill: down ${peak - run:,.0f} from the best realized profit"
    if lost_today >= policy.daily_loss_limit:
        return f"daily loss limit: down ${lost_today:,.0f} on trades closed today"
    if exposure >= policy.max_exposure:
        return f"exposure cap: ${exposure:,.0f} at risk"
    return None


def step(live: dict[str, dict], preds: dict[str, dict], outcomes: dict[str, dict], ledger: list[dict], policy: Policy, now: datetime,
         drifts: dict[tuple[str, str, str], float] | None = None, halted: bool = False) -> list[dict]:
    """New ledger records for one account. `live` maps event_id -> {"quotes": {label: quote}, "resolves_at": datetime}."""
    drifts = drifts or {}
    recs: list[dict] = []
    stamp = now.isoformat(timespec="seconds")
    pos = positions(ledger)
    orders = waiting_orders(ledger)
    block = "kill switch file present" if halted else risk_block(pos, orders, now, policy)
    if block and not any(r["action"] == "blocked" and r["at"][:10] == now.date().isoformat() and r["reason"].split(":")[0] == block.split(":")[0] for r in ledger):
        recs.append({"action": "blocked", "event_id": "-", "at": stamp, "reason": block})  # once a day per reason

    for eid, ps in pos.items():  # exits
        if ps["closed"]:
            continue
        o = outcomes.get(eid)
        if o is not None:
            pay = settle(ps, o["resolved_label"])
            recs.append({"action": "close", "event_id": eid, "at": stamp, "reason": "settled", "exit_price": pay, "ret": net_return(ps, pay)})
            continue
        lv = live.get(eid)
        if not lv or ps["label"] not in lv["quotes"]:
            continue
        age = (now - datetime.fromisoformat(ps["at"])).total_seconds() / 3600
        rate = lv.get("fee_rate", policy.fee_rate)
        hit = exit_reason(ps, lv["quotes"][ps["label"]], age, policy, rate)
        if hit:
            recs.append({"action": "close", "event_id": eid, "at": stamp, "reason": hit[0], "exit_price": hit[1], "fee_out": round(fee(hit[1], rate), 4),
                         "ret": net_return(ps, hit[1] - fee(hit[1], rate))})

    def open_rec(eid, o, price, how):
        rate = live[eid].get("fee_rate", policy.fee_rate) if eid in live else policy.fee_rate
        fee_in = 0.0 if how == "limit order filled" else fee(price, rate)  # resting orders are makers: no fee, and a rebate we don't count
        return {"action": "open", "event_id": eid, "city": o["city"], "at": stamp, "label": o["label"], "side": o["side"], "price": price, "fee_in": round(fee_in, 4),
                "fair": o["fair"], "edge": round(o["fair"] - price - fee_in, 4), "spread": o.get("spread"), "stake": stake_for({**o, "fee_in": fee_in}, price, policy),
                "drift": o.get("drift"), "region": o["region"], "entry": how, "signal_at": o["signal_at"], "source": policy.source}

    for eid, o in orders.items():  # resting limit orders: filled once the price comes to us
        lv = live.get(eid)
        if eid in outcomes or lv is None or lv["resolves_at"] <= now or now >= datetime.fromisoformat(o["expires"]):
            recs.append({"action": "cancel", "event_id": eid, "at": stamp})
            continue
        q = lv["quotes"].get(o["label"])
        buy = side_prices(o["side"], q)[0] if q else None
        if buy is not None and buy < o["limit"] and not block:  # strictly through our price: a fill we can trust
            recs.append(open_rec(eid, o, o["limit"], "limit order filled"))

    if block:
        return recs  # exits and settlement ran above; nothing new opens

    open_regions: dict[str, int] = {}
    for eid, ps in pos.items():
        if not ps["closed"]:
            open_regions[ps.get("region", "")] = open_regions.get(ps.get("region", ""), 0) + 1
    for o in orders.values():
        open_regions[o["region"]] = open_regions.get(o["region"], 0) + 1
    n_open = sum(1 for ps in pos.values() if not ps["closed"]) + len(orders)
    n_today = sum(1 for ps in pos.values() if ps["at"][:10] == now.date().isoformat()) + sum(1 for r in recs if r["action"] == "order")
    for eid, p in preds.items():  # entries
        if eid in pos or eid in orders or eid in outcomes or eid not in live or n_open >= policy.max_open or n_today >= policy.max_new_per_day:
            continue
        if (now - datetime.fromisoformat(p["logged_at"])).total_seconds() / 3600 > policy.max_entry_age_hours or live[eid]["resolves_at"] <= now:
            continue
        if policy.only_at_open and p.get("trigger") != "open":
            continue
        t = decide(p, live[eid]["quotes"], policy, live[eid].get("fee_rate", policy.fee_rate))
        if t is None:
            continue
        reg = region(p)
        if policy.max_per_region is not None and open_regions.get(reg, 0) >= policy.max_per_region:
            continue
        drift = drifts.get((eid, t["label"], t["side"]))
        if policy.max_adverse_drift is not None and drift is not None and -drift > policy.max_adverse_drift:
            continue
        sig = {"city": p["city"], "label": t["label"], "side": t["side"], "fair": t["fair"], "spread": t["spread"], "drift": drift, "region": reg, "signal_at": stamp}
        open_regions[reg] = open_regions.get(reg, 0) + 1
        n_open += 1
        n_today += 1
        if policy.maker and t["spread"]:
            limit = round(t["price"] - policy.maker_improve * t["spread"], 3)
            recs.append({"action": "order", "event_id": eid, **sig, "limit": limit, "price_then": t["price"],
                         "expires": (now + timedelta(hours=policy.maker_wait_hours)).isoformat(timespec="seconds")})
        else:
            recs.append(open_rec(eid, sig, t["price"], "paid the ask"))
    return recs


def live_quotes(http: httpx.Client) -> dict[str, dict]:
    out = {}
    for e in fetch_events(http):
        try:
            c = parse_event(e)
        except (ValueError, KeyError, LookupError):
            continue
        m = (e.get("markets") or [{}])[0]
        rate = (m.get("feeSchedule") or {}).get("rate", WEATHER_FEE_RATE) if m.get("feesEnabled") else 0.0
        out[c.semantics.event_id] = {"resolves_at": c.semantics.resolves_at, "fee_rate": rate,
                                      "quotes": {b.label: {"best_bid": b.best_bid, "best_ask": b.best_ask, "liquidity": b.liquidity} for b in c.pricing.brackets}}
    return out


def recent_mids(path: Path = SNAP_PATH, hours: float = DRIFT_HOURS) -> dict[tuple[str, str], float]:
    """Each bracket's middle price from the oldest snapshot within the last `hours`, read from the tail of the snapshot file."""
    if not path.exists():
        return {}
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 6_000_000))
        rows = [json.loads(x) for x in f.read().decode("utf-8", "ignore").splitlines()[1:] if x.strip().endswith("}")]
    if not rows:
        return {}
    cutoff = (datetime.fromisoformat(rows[-1]["ts"]) - timedelta(hours=hours)).isoformat()
    out: dict[tuple[str, str], float] = {}
    for r in rows:
        if r["ts"] >= cutoff:
            for label, m in zip(r["labels"], r["mid"]):
                out.setdefault((r["event_id"], label), m)
    return out


def drifts_for(live: dict[str, dict], old: dict[tuple[str, str], float]) -> dict[tuple[str, str, str], float]:
    """How far each bracket's YES or NO middle price moved in the last 3 hours (positive = up)."""
    out = {}
    for eid, lv in live.items():
        for label, q in lv["quotes"].items():
            _, _, mid = side_prices("YES", q)
            was = old.get((eid, label))
            if mid is not None and was is not None:
                out[(eid, label, "YES")] = mid - was
                out[(eid, label, "NO")] = -(mid - was)
    return out


def newest_predictions(log_dir: Path, policy: Policy) -> dict[str, dict]:
    """The most recent logged prediction per event that has the policy's source."""
    best: dict[str, dict] = {}
    for p in plog.read(log_dir / plog.PREDICTIONS):
        if entry_vector(p, policy.source) is None:
            continue
        cur = best.get(p["event_id"])
        if cur is None or p["logged_at"] > cur["logged_at"]:
            best[p["event_id"]] = p
    return best


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser(description="Paper Trader")
    ap.add_argument("--watch", type=float, default=0, metavar="MINUTES")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--accounts", default="control,careful,patient,smart,quick,flip", help="comma-separated account names")
    ap.add_argument("--dir", type=Path, default=TRADER_DIR)
    ap.add_argument("--log-dir", type=Path, default=plog.LOG_DIR)
    args = ap.parse_args(argv)
    chosen = [presets()[n] for n in args.accounts.split(",")]
    if args.report:
        from weatheragents import trader_report

        ledgers = {pol.name: read_ledger(args.dir / f"{pol.name}.jsonl") for pol in chosen}
        try:
            with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1"}) as http:
                live = live_quotes(http)
        except httpx.HTTPError:
            live = None  # offline: open positions are then left out of the dollar total
        orders = {n: (sum(1 for r in lg if r["action"] == "order"), len(waiting_orders(lg))) for n, lg in ledgers.items()}
        print(trader_report.full_report({n: positions(lg) for n, lg in ledgers.items()}, orders, live))
        for n, lg in ledgers.items():
            blocks = [r for r in lg if r["action"] == "blocked"]
            if blocks:
                print(f"risk gate [{n}]: stopped new trades {len(blocks)} time(s); latest {blocks[-1]['at'][5:16]} - {blocks[-1]['reason']}")
        return 0
    last_resolve = 0.0
    while True:
        if time.time() - last_resolve > 3600:  # settlement needs Polymarket's results in the outcomes log; fetch them hourly
            try:
                from weatheragents import resolver

                with contextlib.redirect_stdout(io.StringIO()):
                    resolver.main(["--log-dir", str(args.log_dir)])
            except Exception as exc:
                print(f"{ict_now():%m-%d %H:%M} ! resolver: {type(exc).__name__}: {exc}", file=sys.stderr)
            last_resolve = time.time()
        try:
            with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1"}) as http:
                live = live_quotes(http)
            outcomes = {o["event_id"]: o for o in plog.read(args.log_dir / plog.OUTCOMES)}
            now = ict_now()
            drifts = drifts_for(live, recent_mids())
            for pol in chosen:
                path = args.dir / f"{pol.name}.jsonl"
                recs = step(live, newest_predictions(args.log_dir, pol), outcomes, read_ledger(path), pol, now, drifts, halted=(args.dir / "KILL").exists())
                for r in recs:
                    append(r, path)
                    what = {"open": lambda: f"{r['side']} {r['label']} at {r['price']:.2f} edge {r['edge']:.2f} (${r['stake']:g})",
                            "order": lambda: f"{r['side']} {r['label']} limit {r['limit']:.3f}",
                            "cancel": lambda: "limit order cancelled",
                            "blocked": lambda: f"no new trades: {r['reason']}",
                            "close": lambda: f"{r['reason']} return {r['ret']:+.1%}"}[r["action"]]()
                    print(f"{now:%m-%d %H:%M} [{pol.name}] {r['action'].upper()} {r['event_id']} {what}")
        except Exception as exc:
            print(f"{ict_now():%m-%d %H:%M} ! {type(exc).__name__}: {exc}", file=sys.stderr)
        if not args.watch:
            return 0
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    raise SystemExit(main())

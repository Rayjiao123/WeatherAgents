"""Records how Polymarket's prices move, so we can ask why.

    python -m weatheragents.market_monitor              # one snapshot now
    python -m weatheragents.market_monitor --watch 30   # a snapshot every 30 minutes, until stopped

Each snapshot stores, for every open temperature event, the price of every bracket and the forecast the
market implies: an average high and how wide the market thinks the outcome is. Appended to
data/market/snapshots.jsonl (Thailand time). History cannot be backfilled, so this should run whenever
the PC is on. The gaps while it was off are part of the record.

This is price data, so it belongs on the Trader side of the wall: no analyst or curve ever reads it.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx

from weatheragents.scout import GAMMA_URL, ParseError, fetch_events, parse_event
from weatheragents.schemas import Bracket, Contract
from weatheragents.timeutil import now as ict_now

SNAP_PATH = Path("data") / "market" / "snapshots.jsonl"
YOUNG_HOURS = 22  # markets open about 46h before resolution, so more than this to go means opened in the last day
MIN_LIQUIDITY = 5_000  # keep thinner markets than Scout does: their moves are information too


def bracket_value(b: Bracket) -> float:
    """A single number for a bracket, for averaging. Open-ended brackets sit just inside their edge."""
    if b.low is None:
        return b.high - 0.5
    if b.high is None:
        return b.low + 0.5
    return (b.low + b.high) / 2


def implied_stats(brackets: list[Bracket], mids: list[float]) -> tuple[float, float]:
    """The market's implied average high and spread (standard deviation) from bracket prices."""
    total = sum(mids)
    if total <= 0:
        return math.nan, math.nan
    p = [m / total for m in mids]
    vals = [bracket_value(b) for b in brackets]
    mean = sum(v * q for v, q in zip(vals, p))
    return mean, math.sqrt(sum(q * (v - mean) ** 2 for v, q in zip(vals, p)))


def snap_event(contract: Contract, ts: str) -> dict:
    s, pr = contract.semantics, contract.pricing
    bid, ask, mid = [], [], []
    for b in pr.brackets:
        bid.append(b.best_bid)
        ask.append(b.best_ask)
        if b.best_bid is not None and b.best_ask is not None:
            mid.append((b.best_bid + b.best_ask) / 2)
        else:
            mid.append(b.last_trade if b.last_trade is not None else 0.0)
    mean, sd = implied_stats(s.brackets, mid)
    r = lambda xs: [None if x is None else round(x, 4) for x in xs]
    return {
        "ts": ts, "event_id": s.event_id, "city": s.city, "kind": s.kind, "unit": s.unit,
        "observation_date": str(s.observation_date), "resolves_at": s.resolves_at.isoformat(),
        "labels": [b.label for b in s.brackets], "mid": r(mid), "bid": r(bid), "ask": r(ask),
        "liquidity": round(pr.total_liquidity), "volume_24h": round(pr.volume_24h),
        "implied_mean": None if math.isnan(mean) else round(mean, 3),
        "implied_sd": None if math.isnan(sd) else round(sd, 3),
    }


def take_snapshot(client: httpx.Client, path: Path = SNAP_PATH, *, young_only: bool = False) -> int:
    """Snapshot every event, or with young_only just those that opened recently (more than YOUNG_HOURS to go)."""
    now = ict_now()
    ts = now.isoformat(timespec="seconds")
    rows = []
    for event in fetch_events(client):
        try:
            c = parse_event(event)
        except (ParseError, KeyError, ValueError):
            continue
        if young_only and (c.semantics.resolves_at - now).total_seconds() / 3600 < YOUNG_HOURS:
            continue
        if c.pricing.total_liquidity >= MIN_LIQUIDITY:
            rows.append(snap_event(c, ts))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def read_snapshots(path: Path = SNAP_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Snapshot Polymarket temperature prices")
    ap.add_argument("--watch", type=float, default=None, metavar="MINUTES", help="keep snapshotting every N minutes")
    ap.add_argument("--young-every", type=float, default=None, metavar="MINUTES",
                    help="between full snapshots, also snapshot just the recently opened markets this often")
    args = ap.parse_args(argv)
    step = min(args.watch, args.young_every) if args.watch and args.young_every else args.watch
    last_full = 0.0
    with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1"}) as client:
        while True:
            try:
                full = args.watch is None or time.time() - last_full >= args.watch * 60 - 30
                n = take_snapshot(client, young_only=not full)
                last_full = time.time() if full else last_full
                print(f"{ict_now():%m-%d %H:%M} ICT: {n} events", flush=True)
            except (httpx.HTTPError, ValueError) as exc:
                print(f"{ict_now():%m-%d %H:%M} ICT: snapshot failed ({type(exc).__name__}); will retry", flush=True)
            if args.watch is None:
                return 0
            time.sleep(step * 60)


if __name__ == "__main__":
    raise SystemExit(main())

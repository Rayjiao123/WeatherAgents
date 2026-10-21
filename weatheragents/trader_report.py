"""Money scorecard for the paper accounts: what each would have made, with the numbers needed to judge whether it is real.

    python -m weatheragents.trader --report

All dollars are pretend. Each trade risks its stake: it buys stake/price shares and a share pays 1 if it wins, so a
winning trade makes stake * (1/price - 1) and a loser loses the stake. Open positions are valued at the price they could
be sold for right now (the bid), so they are honest about the spread. Polymarket's weather-market taker fee (rate 0.05 x p x (1-p)
per share) is charged when buying at the ask and when selling at the bid; resting limit orders and settlement pay none.
"""

import math
import statistics
from collections import defaultdict

from weatheragents.trader import WEATHER_FEE_RATE, fee, side_prices

START = 1000.0  # paper bankroll per account
BANDS = [("long shots (price under 20c)", 0.0, 0.20), ("middle (20-60c)", 0.20, 0.60), ("favourites (over 60c)", 0.60, 1.01)]


def pnl(ps: dict) -> float:
    return ps["stake"] * ps["closed"]["ret"]


def mark(ps: dict, live: dict | None) -> float | None:
    """Dollar value change of an open position if sold now; None when there is no live bid."""
    lv = (live or {}).get(ps["event_id"])
    q = lv["quotes"].get(ps["label"]) if lv else None
    sell = side_prices(ps["side"], q)[1] if q else None
    if sell is None:
        return None
    rate = lv.get("fee_rate", WEATHER_FEE_RATE)
    return ps["stake"] * (sell - fee(sell, rate) - ps["price"] - ps.get("fee_in", 0.0)) / ps["price"]


def interval(xs: list[float]) -> tuple[float, float]:
    """Mean and the two-standard-error half width."""
    if not xs:
        return 0.0, 0.0
    m = statistics.fmean(xs)
    return m, (2 * statistics.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan"))


def drawdown(closed: list[dict]) -> float:
    """Largest fall from a peak in the running total of realized profit, in dollars."""
    peak = run = worst = 0.0
    for ps in sorted(closed, key=lambda p: p["closed"]["at"]):
        run += pnl(ps)
        peak = max(peak, run)
        worst = max(worst, peak - run)
    return worst


def verdict(rets: list[float]) -> str:
    if len(rets) < 30:
        return f"too early: {len(rets)} closed trades, need about 100"
    m, h = interval(rets)
    if m - h > 0:
        return f"profitable beyond chance so far ({m:+.1%} per dollar, range {m - h:+.1%} to {m + h:+.1%})"
    if m + h < 0:
        return f"losing beyond chance so far ({m:+.1%} per dollar, range {m - h:+.1%} to {m + h:+.1%})"
    return f"cannot tell from luck yet ({m:+.1%} per dollar, range {m - h:+.1%} to {m + h:+.1%})"


def account_lines(name: str, pos: dict[str, dict], n_orders: int, n_waiting: int, live: dict | None) -> list[str]:
    closed = [p for p in pos.values() if p["closed"]]
    opened = [p for p in pos.values() if not p["closed"]]
    realized = sum(pnl(p) for p in closed)
    marks = [mark(p, live) for p in opened]
    unreal = sum(m for m in marks if m is not None)
    staked = sum(p["stake"] for p in closed)
    rets = [p["closed"]["ret"] for p in closed]
    out = [f"[{name}]  {len(pos)} trades opened, {len(closed)} closed, {len(opened)} open" + (f", {n_waiting} limit orders waiting" if n_waiting else "")]
    out.append(f"  profit: realized ${realized:+,.2f}, open positions ${unreal:+,.2f} at today's bids  ->  account ${START + realized + unreal:,.2f} from ${START:,.0f}")
    if closed:
        wins = [pnl(p) for p in closed if pnl(p) > 0]
        losses = [pnl(p) for p in closed if pnl(p) <= 0]
        out.append(f"  closed: staked ${staked:,.0f}, return {realized / staked:+.1%} on money staked, {len(wins)}/{len(closed)} winners "
                   f"(avg win ${statistics.fmean(wins) if wins else 0:,.2f}, avg loss ${statistics.fmean(losses) if losses else 0:,.2f}), worst drawdown ${drawdown(closed):,.2f}")
        daily: dict[str, float] = defaultdict(float)
        for p in closed:
            daily[p["closed"]["at"][:10]] += pnl(p)
        out.append(f"  by day: {sum(1 for v in daily.values() if v > 0)} of {len(daily)} days made money (days are the independent unit; one weather pattern hits many cities)")
        out.append(f"  verdict: {verdict(rets)}")
        by = defaultdict(list)
        for p in closed:
            by[p["closed"]["reason"]].append(p)
        out.append("  how trades ended: " + "; ".join(f"{k} {len(v)} trades ${sum(pnl(p) for p in v):+,.0f}" for k, v in sorted(by.items())))
        sides = defaultdict(list)
        for p in closed:
            sides[p["side"]].append(p)
        out.append("  by side: " + "; ".join(f"buy {k} {len(v)} trades ${sum(pnl(p) for p in v):+,.0f}" for k, v in sorted(sides.items())))
        out.append("  by price: " + "; ".join(f"{label} {len(g)} ${sum(pnl(p) for p in g):+,.0f}" for label, lo, hi in BANDS if (g := [p for p in closed if lo <= p["price"] < hi])))
        if n_orders:
            filled = sum(1 for p in pos.values() if p.get("entry") == "limit order filled")
            out.append(f"  limit orders: {n_orders} placed, {filled} filled ({filled / n_orders:.0%}); the rest never traded at our price")
    return out


def compare_lines(accounts: dict[str, dict[str, dict]]) -> list[str]:
    """Each account against the control on the events both traded, closed trades only."""
    base = {e: p for e, p in accounts.get("control", {}).items() if p["closed"]}
    out = []
    for name, pos in accounts.items():
        if name == "control":
            continue
        both = [(pos[e], base[e]) for e in base if e in pos and pos[e]["closed"]]
        if len(both) < 10:
            out.append(f"  {name} vs control: {len(both)} events traded by both, too few so far")
            continue
        diff = [a["closed"]["ret"] - b["closed"]["ret"] for a, b in both]
        m, h = interval(diff)
        out.append(f"  {name} vs control on {len(both)} shared events: {m:+.1%} per dollar better ({m - h:+.1%} to {m + h:+.1%})"
                   + ("  <- real" if m - h > 0 else "  <- worse than chance" if m + h < 0 else ""))
    return out


def full_report(accounts: dict[str, dict[str, dict]], orders: dict[str, tuple[int, int]], live: dict | None) -> str:
    lines = ["PAPER TRADING SCORECARD (pretend dollars; $1,000 per account; taker fees included)", ""]
    for name, pos in accounts.items():
        n_orders, n_waiting = orders.get(name, (0, 0))
        lines += account_lines(name, pos, n_orders, n_waiting, live) + [""]
    cmp = compare_lines(accounts)
    if cmp:
        lines += ["VERSUS THE CONTROL (same events, paired):"] + cmp + [""]
    lines.append("Read the verdict lines, not the dollars: a few lucky long shots can make any account look rich. A result counts once its range stays above zero over many days.")
    return "\n".join(lines)

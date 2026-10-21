from datetime import datetime, timedelta

import pytest

from weatheragents import paper_trader as pt

LABELS = ["a", "b", "c"]


@pytest.fixture(autouse=True)
def no_fees(monkeypatch):
    monkeypatch.setattr(pt, "FEE_RATE", 0.0)  # the older tests use round numbers; fees have their own test at the bottom


def pred(bids, asks):
    return {"brackets": LABELS, "market_snapshot": {"brackets": [{"label": l, "best_bid": b, "best_ask": a} for l, b, a in zip(LABELS, bids, asks)]}}


def test_best_trade_pays_the_spread_and_picks_the_biggest_edge():
    p = pred(bids=[0.10, 0.40, 0.20], asks=[0.14, 0.46, 0.26])
    t = pt.best_trade(p, [0.10, 0.70, 0.20])
    assert t["label"] == "b" and t["side"] == "YES" and t["price"] == 0.46 and t["edge"] == pytest.approx(0.24)
    assert t["mid_price"] == pytest.approx(0.43)  # a patient order at the middle would have paid less
    t = pt.best_trade(p, [0.30, 0.15, 0.20])  # the market has b far too high: buy NO there
    assert t["label"] == "b" and t["side"] == "NO" and t["price"] == pytest.approx(0.60) and t["edge"] == pytest.approx(0.25)
    assert pt.best_trade(pred([None] * 3, [None] * 3), [0.3, 0.3, 0.4]) is None  # no quotes, no trade


def test_a_tiny_price_is_never_bought():
    p = pred(bids=[0.0, 0.0, 0.0], asks=[0.01, 0.01, 0.01])  # one-cent quotes: the whole edge would be quote noise
    assert pt.best_trade(p, [0.5, 0.3, 0.2]) is None


def test_returns_held_to_resolution_and_marked_to_the_market():
    yes = {"label": "b", "side": "YES", "price": 0.40}
    assert pt.resolve_trade(yes, "b") == pytest.approx(1.5) and pt.resolve_trade(yes, "a") == -1.0
    no = {"label": "b", "side": "NO", "price": 0.60}
    assert pt.resolve_trade(no, "a") == pytest.approx(1 / 0.6 - 1) and pt.resolve_trade(no, "b") == -1.0
    snap = {"labels": LABELS, "bid": [0.1, 0.50, 0.2], "ask": [0.2, 0.56, 0.3]}
    assert pt.mark_trade(yes, snap) == pytest.approx(0.50 / 0.40 - 1)  # sold at the bid
    assert pt.mark_trade(no, snap) == pytest.approx((1 - 0.56) / 0.60 - 1)  # sold at one minus the ask
    assert pt.mark_trade({"label": "zzz", "side": "YES", "price": 0.4}, snap) is None


def test_exit_snapshots_must_be_after_entry_and_close_to_the_requested_time():
    t0 = datetime(2026, 10, 14, 12, 0)
    rs = [{"ts": (t0 + timedelta(hours=h)).isoformat()} for h in (0, 10, 20, 30)]
    assert pt.snapshot_near(rs, t0 + timedelta(hours=20, minutes=10), after=t0 + timedelta(hours=5))["ts"] == rs[2]["ts"]
    assert pt.snapshot_near(rs, t0 + timedelta(hours=1), after=t0 + timedelta(hours=5)) is None  # nothing near that time after entry


def test_exit_rules_walk_the_price_path():
    t = {"price": 0.40, "edge": 0.20, "held": -1.0, "path": [(2, 0.42), (5, 0.52), (9, 0.30)]}
    assert pt.exit_return(t, ("tp", 0.10, 0.10)) == pytest.approx(0.52 / 0.40 - 1)  # +12c: sell on the way up
    assert pt.exit_return(t, ("hours", 6)) == pytest.approx(0.30 / 0.40 - 1)  # first quote at or after 6h
    assert pt.exit_return(t, ("fair", 0.02)) == -1.0  # never reached 0.58: held to the end
    assert pt.exit_return({**t, "path": [(2, 0.30)]}, ("tp", 9.0, 0.05)) == pytest.approx(0.30 / 0.40 - 1)  # stop-loss


def test_fees_are_charged_on_buying_and_selling_but_not_at_settlement(monkeypatch):
    monkeypatch.setattr(pt, "FEE_RATE", 0.05)
    t = {"label": "b", "side": "YES", "price": 0.40, "fee_in": pt.fee(0.40), "edge": 0.3}
    assert pt.fee(0.40) == pytest.approx(0.05 * 0.4 * 0.6)
    assert pt.resolve_trade(t, "b") == pytest.approx((1 - 0.40 - 0.012) / 0.40)  # no fee when it settles
    assert pt.resolve_trade(t, "a") == pytest.approx(-(0.40 + 0.012) / 0.40)  # the entry fee is lost too
    snap = {"labels": LABELS, "bid": [0.1, 0.50, 0.2], "ask": [0.2, 0.56, 0.3]}
    assert pt.mark_trade(t, snap) == pytest.approx((0.50 - 0.05 * 0.25 - 0.40 - 0.012) / 0.40)  # the sale fee comes off the proceeds
    assert pt.exit_return({**t, "held": -1.0, "path": [(2, 0.42)]}, ("tp", 0.01, 9.0)) == -1.0  # 0.42 less fees is under what we paid: no profit to bank

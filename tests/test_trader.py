from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from weatheragents import trader as tr

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone(timedelta(hours=7)))
LABELS = ["a", "b", "c"]
CONTROL = tr.Policy(fee_rate=0.0)  # most tests use round numbers; fees have their own test below


def pred(logged=NOW - timedelta(hours=1), sd=1.0, station="KLAX"):
    vec = [{"label": l, "probability": v} for l, v in zip(LABELS, [0.1, 0.7, 0.2])]
    return {"event_id": "e", "city": "X", "kind": "highest", "station_code": station, "brackets": LABELS, "logged_at": logged.isoformat(),
            "resolves_at": (NOW + timedelta(hours=40)).isoformat(), "ml": {"bracket_probabilities": vec, "sd": sd},
            "market_snapshot": {"brackets": []}, "estimate": {}, "ensemble": {"bracket_probabilities": vec}, "climatology": {"bracket_probabilities": vec}}


def live(b=0.40, a=0.46):
    q = {"a": {"best_bid": 0.1, "best_ask": 0.14, "liquidity": 900}, "b": {"best_bid": b, "best_ask": a, "liquidity": 900}, "c": {"best_bid": 0.18, "best_ask": 0.22, "liquidity": 900}}
    return {"e": {"quotes": q, "resolves_at": NOW + timedelta(hours=40)}}


def run(policy, lv=None, p=None, ledger=(), when=NOW, outcomes=None, drifts=None):
    return tr.step(lv or live(), {"e": p or pred()}, outcomes or {}, list(ledger), policy, when, drifts)


def test_opens_the_biggest_edge_once():
    recs = run(CONTROL)
    assert len(recs) == 1 and recs[0]["label"] == "b" and recs[0]["side"] == "YES" and recs[0]["price"] == 0.46 and recs[0]["fair"] == pytest.approx(0.70)
    assert run(CONTROL, ledger=recs) == []  # no second position on the same event


def test_filters_and_stale_predictions_block_entry():
    assert run(replace(CONTROL, max_spread=0.10), lv=live(0.30, 0.46)) == []  # 16c spread
    assert run(replace(CONTROL, max_sd=2.0), p=pred(sd=2.5)) == []  # model unsure
    assert run(CONTROL, p=pred(logged=NOW - timedelta(hours=9))) == []  # prediction too old
    assert run(replace(CONTROL, cost_mult=5.0)) == []  # edge 24c is under five times the 6c spread
    assert run(replace(CONTROL, max_adverse_drift=0.05), drifts={("e", "b", "YES"): -0.08}) == []  # price already running against us
    assert len(run(replace(CONTROL, max_adverse_drift=0.05), drifts={("e", "b", "YES"): 0.08})) == 1


def test_region_cap_counts_trades_sharing_a_station_prefix():
    first = run(replace(CONTROL, max_per_region=1))
    other = {"e2": pred() | {"event_id": "e2"}}
    lv = {**live(), "e2": live()["e"]}
    recs = tr.step(lv, other, {}, first, replace(CONTROL, max_per_region=1), NOW)
    assert recs == []  # same KL prefix: capped
    assert len(tr.step(lv, {"e2": pred(station="ZGGG") | {"event_id": "e2"}}, {}, first, replace(CONTROL, max_per_region=1), NOW)) == 1


def test_limit_order_rests_then_fills_only_when_price_comes_through():
    pol = replace(CONTROL, maker=True)
    placed = run(pol)
    assert placed[0]["action"] == "order" and placed[0]["limit"] == pytest.approx(0.43)  # halfway into the 6c spread
    assert run(pol, ledger=placed, when=NOW + timedelta(minutes=30)) == []  # ask still 0.46: not filled
    filled = run(pol, lv=live(0.38, 0.42), ledger=placed, when=NOW + timedelta(minutes=30))
    assert filled[0]["action"] == "open" and filled[0]["price"] == 0.43 and filled[0]["entry"] == "limit order filled"
    expired = run(pol, ledger=placed, when=NOW + timedelta(hours=4))
    assert expired[0]["action"] == "cancel"
    assert tr.waiting_orders(placed + expired) == {}


def test_kelly_sizing_grows_with_edge_and_is_capped():
    big = {"fair": 0.90}
    small = {"fair": 0.50}
    pol = replace(CONTROL, sizing="kelly")
    assert tr.stake_for(big, 0.40, pol) == 25.0  # capped
    assert tr.stake_for(small, 0.46, pol) == pytest.approx(18.52, abs=0.01)  # Kelly fraction 0.04/0.54 = 7.4% of a 1000 bankroll, a quarter of it
    assert tr.stake_for(small, 0.50, pol) == 2.0  # no edge: the floor
    assert tr.stake_for(small, 0.46, CONTROL) == 10.0


def test_exit_rules_fire_in_order():
    opened = run(CONTROL)[0]
    ps = {**opened, "closed": None}
    q = lambda b, a: {"best_bid": b, "best_ask": a}
    pol = replace(CONTROL, stop_loss=0.10, target_margin=0.03, min_hold_edge=0.02, exit_hours=24.0, fair_shrink=1.0)
    assert tr.exit_reason(ps, q(0.34, 0.38), 2.0, pol) == ("stop", 0.34)  # paid 0.46, can only sell at 0.34
    assert tr.exit_reason(ps, q(0.68, 0.70), 2.0, pol) == ("target", 0.68)  # within 3c of our 0.70 value
    assert tr.exit_reason(ps, q(0.50, 0.54), 2.0, pol) is None  # still 20c of edge to collect
    assert tr.exit_reason(ps, q(0.50, 0.54), 25.0, pol)[0].startswith("time")
    shrunk = replace(pol, fair_shrink=0.0, target_margin=None, stop_loss=None)  # value = the market's own middle price
    assert tr.exit_reason(ps, q(0.50, 0.52), 2.0, shrunk) == ("stale", 0.50)  # nothing left to gain over the bid


def test_step_closes_on_a_rule_and_settles_at_resolution():
    pol = replace(CONTROL, exit_hours=12.0)
    opened = run(pol)
    assert run(pol, lv=live(0.60, 0.64), ledger=opened, when=NOW + timedelta(hours=5)) == []
    sold = run(pol, lv=live(0.60, 0.64), ledger=opened, when=NOW + timedelta(hours=12))
    assert sold[0]["action"] == "close" and sold[0]["exit_price"] == 0.60 and sold[0]["ret"] == pytest.approx(0.60 / 0.46 - 1)
    done = tr.step({}, {}, {"e": {"resolved_label": "b"}}, opened, pol, NOW + timedelta(hours=30))
    assert done[0]["reason"] == "settled" and done[0]["ret"] == pytest.approx(1 / 0.46 - 1)
    assert tr.positions(opened + sold)["e"]["closed"]["reason"].startswith("time")
    assert tr.side_prices("NO", {"best_bid": 0.40, "best_ask": 0.46}) == (pytest.approx(0.60), pytest.approx(0.54), pytest.approx(0.57))


def test_report_dollars_marks_and_verdicts():
    from weatheragents import trader_report as rp

    def pos(price, ret, at="2026-10-17T10:00:00+07:00", side="YES", stake=10.0):
        return {"event_id": str(price), "label": "b", "side": side, "price": price, "stake": stake, "closed": {"ret": ret, "at": at, "reason": "time (12h)"}}

    win, lose = pos(0.40, 1.5), pos(0.50, -1.0, at="2026-10-17T11:00:00+07:00")
    assert rp.pnl(win) == pytest.approx(15.0) and rp.pnl(lose) == pytest.approx(-10.0)
    assert rp.drawdown([win, lose]) == pytest.approx(10.0)
    openp = {"event_id": "x", "label": "b", "side": "YES", "price": 0.40, "stake": 10.0}
    assert rp.mark(openp, {"x": {"fee_rate": 0.0, "quotes": {"b": {"best_bid": 0.50, "best_ask": 0.54}}}}) == pytest.approx(2.5)
    assert rp.mark(openp, {"x": {"fee_rate": 0.05, "quotes": {"b": {"best_bid": 0.50, "best_ask": 0.54}}}}) == pytest.approx(10 * (0.50 - 0.0125 - 0.40) / 0.40)
    assert rp.mark(openp, None) is None
    assert rp.verdict([0.1] * 10).startswith("too early")
    assert "losing beyond chance" in rp.verdict([-0.5, -0.4] * 20) and "profitable beyond chance" in rp.verdict([0.5, 0.4] * 20)
    text = rp.full_report({"control": {"a": win, "b": lose}}, {}, None)
    assert "control" in text and "realized $+5.00" in text


def test_quick_account_banks_any_profit_after_the_spread_but_cuts_losses():
    ps = {**run(CONTROL)[0], "closed": None}  # bought at the 0.46 ask
    quick = replace(tr.presets()["quick"], fee_rate=0.0)
    q = lambda b, a: {"best_bid": b, "best_ask": a}
    assert tr.exit_reason(ps, q(0.46, 0.50), 1.0, quick) is None  # selling at 0.46 is break-even, not a profit
    assert tr.exit_reason(ps, q(0.48, 0.52), 1.0, quick) == ("profit", 0.48)
    assert tr.exit_reason(ps, q(0.35, 0.40), 1.0, quick) == ("stop", 0.35)
    assert tr.exit_reason(ps, q(0.40, 0.44), 13.0, quick)[0].startswith("time")


def test_fees_make_buying_and_selling_cost_more_and_profit_must_clear_them():
    pol = tr.presets()["quick"]  # 5% weather fee rate
    rec = run(replace(pol, maker=False))[0]
    fin = 0.05 * 0.46 * 0.54
    assert rec["fee_in"] == pytest.approx(fin, abs=1e-4) and rec["edge"] == pytest.approx(0.70 - 0.46 - fin, abs=1e-3)
    ps = {**rec, "closed": None}
    q = lambda b, a: {"best_bid": b, "best_ask": a}
    assert tr.exit_reason(ps, q(0.48, 0.52), 1.0, pol) is None  # 0.48 less its fee is still under what we paid in total
    assert tr.exit_reason(ps, q(0.50, 0.54), 1.0, pol) == ("profit", 0.50)
    sold = run(pol, lv=live(0.60, 0.64), ledger=[rec], when=NOW + timedelta(hours=13))[0]
    assert sold["ret"] == pytest.approx((0.60 - 0.05 * 0.6 * 0.4 - 0.46 - fin) / 0.46, abs=1e-3)
    assert tr.net_return(ps, 1.0) == pytest.approx((1.0 - 0.46 - fin) / 0.46, abs=1e-3)  # a winner pays 1 with no fee at settlement
    maker = run(replace(pol, maker=True))
    filled = run(replace(pol, maker=True), lv=live(0.38, 0.42), ledger=maker, when=NOW + timedelta(minutes=30))[0]
    assert filled["fee_in"] == 0.0  # a resting order pays no fee


def test_flip_account_buys_only_at_the_open_and_sells_at_any_net_profit_or_12h():
    flip = replace(tr.presets()["flip"], fee_rate=0.0)
    assert run(flip, p=pred(logged=NOW - timedelta(hours=2))) == []  # not an opening: the prediction is two hours old
    rec = run(flip, p=pred(logged=NOW - timedelta(minutes=10)))[0]
    ps = {**rec, "closed": None}
    q = lambda b, a: {"best_bid": b, "best_ask": a}
    assert tr.exit_reason(ps, q(0.46, 0.50), 1.0, flip) is None  # break-even is not a profit
    assert tr.exit_reason(ps, q(0.47, 0.51), 1.0, flip) == ("profit", 0.47)
    assert tr.exit_reason(ps, q(0.20, 0.24), 5.0, flip) is None  # no stop: it waits even when far under water
    assert tr.exit_reason(ps, q(0.20, 0.24), 12.0, flip)[0].startswith("time")


def test_risk_gate_blocks_new_trades_but_not_exits():
    opened = run(CONTROL)
    assert tr.step(live(), {"e": pred()}, {}, [], CONTROL, NOW, halted=True)[0]["action"] == "blocked"  # kill switch file
    capped = replace(CONTROL, max_exposure=10.0)
    first = run(capped)
    assert first[0]["action"] == "open"  # nothing at risk yet
    e2 = {"event_id": "e2"}
    lv = {**live(), "e2": live()["e"]}
    got = tr.step(lv, {"e2": pred() | e2}, {}, first, capped, NOW)
    assert [r["action"] for r in got] == ["blocked"] and got[0]["reason"].startswith("exposure cap")
    again = tr.step(lv, {"e2": pred() | e2}, {}, first + got, capped, NOW + timedelta(minutes=5))
    assert again == []  # reported once a day, not every poll
    sold = tr.step(lv, {}, {}, first, capped, NOW + timedelta(hours=13))
    assert any(r["action"] == "close" for r in sold)  # exits still run while blocked


def test_daily_loss_limit_and_drawdown_kill():
    lost = {"event_id": "x", "action": "open", "at": NOW.isoformat(), "price": 0.5, "edge": 0.1, "stake": 10.0, "label": "b", "side": "YES"}
    closed = {"action": "close", "event_id": "x", "at": NOW.isoformat(), "reason": "stop", "ret": -1.0, "exit_price": 0.0}
    pos = tr.positions([lost, closed])
    assert tr.risk_block(pos, {}, NOW, replace(CONTROL, daily_loss_limit=5.0)).startswith("daily loss limit")
    assert tr.risk_block(pos, {}, NOW, replace(CONTROL, daily_loss_limit=500.0, kill_drawdown=500.0)) is None
    assert tr.risk_block(pos, {}, NOW + timedelta(days=1), replace(CONTROL, daily_loss_limit=5.0, kill_drawdown=500.0)) is None  # a new day resets the daily limit

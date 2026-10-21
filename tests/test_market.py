import pytest

from weatheragents import market_moves as mm
from weatheragents.market_monitor import bracket_value, implied_stats
from weatheragents.schemas import Bracket

BR = [Bracket(label="a", low=None, high=20), Bracket(label="b", low=21, high=21), Bracket(label="c", low=22, high=22), Bracket(label="d", low=23, high=None)]


def snap(ts, mids, mean=None, sd=None, unit="C", event="1"):
    m, s = implied_stats(BR, mids)
    return {"ts": ts, "event_id": event, "city": "X", "unit": unit, "labels": ["a", "b", "c", "d"], "mid": mids,
            "implied_mean": m if mean is None else mean, "implied_sd": s if sd is None else sd, "resolves_at": "2026-10-14T19:00:00+07:00"}


def test_implied_stats_centre_and_spread_from_prices():
    mean, sd = implied_stats(BR, [0.0, 0.0, 1.0, 0.0])
    assert mean == 22 and sd == 0
    mean, sd = implied_stats(BR, [0.0, 0.5, 0.5, 0.0])
    assert mean == pytest.approx(21.5) and sd == pytest.approx(0.5)
    assert bracket_value(BR[0]) == 19.5 and bracket_value(BR[3]) == 23.5  # open ends sit just inside their edge
    assert implied_stats(BR, [0, 0, 0, 0])[0] != implied_stats(BR, [0, 0, 0, 0])[0]  # nan when nothing is priced


def test_a_move_needs_the_centre_or_the_probability_to_shift_enough():
    quiet = [snap("2026-10-13T20:00:00+07:00", [0.1, 0.4, 0.4, 0.1]), snap("2026-10-13T20:30:00+07:00", [0.1, 0.41, 0.39, 0.1])]
    assert mm.find_moves(quiet) == []
    jump = [snap("2026-10-13T20:00:00+07:00", [0.1, 0.4, 0.4, 0.1]), snap("2026-10-13T20:30:00+07:00", [0.0, 0.1, 0.4, 0.5])]
    (move,) = mm.find_moves(jump)
    assert move["d_mean"] > 0.5 and move["hour_ict"] == 20 and move["hours_left"] == pytest.approx(22.5) and move["gap_min"] == 30
    # Fahrenheit needs a bigger absolute shift for the same degree-Celsius significance
    f = [snap("2026-10-13T20:00:00+07:00", [0.25, 0.25, 0.25, 0.25], mean=70.0, unit="F"), snap("2026-10-13T20:30:00+07:00", [0.25, 0.25, 0.25, 0.25], mean=70.6, unit="F")]
    assert mm.find_moves(f) == []


def test_tv_distance_and_label_change_guard():
    assert mm.tv_distance([1, 0], [0, 1]) == 1.0 and mm.tv_distance([0.5, 0.5], [0.5, 0.5]) == 0
    changed = [snap("2026-10-13T20:00:00+07:00", [0.1, 0.4, 0.4, 0.1]), dict(snap("2026-10-13T20:30:00+07:00", [0, 0, 0, 1]), labels=["x", "y", "z", "w"])]
    assert mm.find_moves(changed) == []  # a different bracket set is a different market, not a move


def test_convergence_counts_only_events_where_we_disagreed_and_measures_the_drift_toward_us():
    series = {"1": [snap("2026-10-13T20:00:00+07:00", [0.1, 0.4, 0.4, 0.1]), snap("2026-10-13T22:00:00+07:00", [0.0, 0.1, 0.4, 0.5])]}
    m0 = series["1"][0]["implied_mean"]
    pred = {"event_id": "1", "logged_at": "2026-10-13T20:10:00+07:00", "unit": "C", "ml": {"center": m0 + 1.0}}
    (c,) = mm.convergence([pred], series, "ml")
    assert c == pytest.approx((series["1"][1]["implied_mean"] - m0) / 1.0) and c > 0  # market drifted up toward our higher centre
    agree = dict(pred, ml={"center": m0 + 0.1})
    assert mm.convergence([agree], series, "ml") is None  # we agreed with the market: no direction to test


def test_moves_ignore_jumps_across_a_collection_gap_and_thin_books():
    a = snap("2026-10-13T20:00:00+07:00", [0.1, 0.4, 0.4, 0.1])
    after_gap = snap("2026-10-14T09:00:00+07:00", [0.0, 0.1, 0.4, 0.5])  # 13 hours later: the monitor was off in between
    assert mm.find_moves([a, after_gap]) == []
    thin = snap("2026-10-13T20:30:00+07:00", [0.0, 0.0, 0.3, 0.0])  # prices add up to 0.3: no real market here
    assert mm.find_moves([a, thin]) == []
    real = snap("2026-10-13T20:30:00+07:00", [0.0, 0.1, 0.4, 0.5])
    assert len(mm.find_moves([a, real])) == 1


def test_young_only_snapshots_skip_markets_close_to_resolution(tmp_path, monkeypatch):
    from datetime import timedelta
    from weatheragents import market_monitor as mm

    class C:
        def __init__(self, hours):
            self.semantics = type("S", (), {"resolves_at": mm.ict_now() + timedelta(hours=hours)})()
            self.pricing = type("P", (), {"total_liquidity": 10_000})()

    cs = {"old": C(10), "young": C(40)}
    monkeypatch.setattr(mm, "fetch_events", lambda client: ["old", "young"])
    monkeypatch.setattr(mm, "parse_event", lambda e: cs[e])
    monkeypatch.setattr(mm, "snap_event", lambda c, ts: {"ts": ts, "hours": 1})
    assert mm.take_snapshot(None, tmp_path / "s.jsonl", young_only=True) == 1
    assert mm.take_snapshot(None, tmp_path / "s2.jsonl") == 2

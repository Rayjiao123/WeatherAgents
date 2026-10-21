import math
from datetime import datetime, timezone

import pytest

from weatheragents import evaluate as ev
from weatheragents import resolver as rs

LABELS = ["a", "b", "c"]


def test_scores_known_values():
    assert ev.log_loss([0.2, 0.5, 0.3], 1) == pytest.approx(-math.log(0.5))
    assert ev.brier([0.2, 0.5, 0.3], 1) == pytest.approx(0.04 + 0.25 + 0.09)
    # RPS is ordered: putting everything one bracket off costs less than two off
    assert ev.rps([0, 1, 0], 0) < ev.rps([0, 0, 1], 0)
    assert ev.rps([1, 0, 0], 0) == 0


def test_normalize_floors_hard_zeros():
    v = ev.normalize([0.0, 0.5, 0.5])
    assert min(v) > 0 and sum(v) == pytest.approx(1)


def pred(eid, logged, resolves="2026-10-07T12:00:00+00:00", probs=(0.2, 0.5, 0.3)):
    rows = lambda ps: [{"label": l, "probability": p} for l, p in zip(LABELS, ps)]
    return {
        "event_id": eid, "city": "X", "kind": "highest", "unit": "C", "logged_at": logged, "resolves_at": resolves,
        "brackets": LABELS, "estimate": {"bracket_probabilities": rows(probs)},
        "ensemble": {"bracket_probabilities": rows((0.0, 1.0, 0.0))},
        "climatology": {"bracket_probabilities": rows((1 / 3, 1 / 3, 1 / 3))},
        "market_snapshot": {"brackets": [
            {"label": l, "best_bid": b, "best_ask": a, "last_trade": None}
            for l, (b, a) in zip(LABELS, ((0.1, 0.2), (0.4, 0.5), (0.2, 0.4)))]},
    }


def test_uses_latest_prediction_made_before_resolution():
    early = pred("1", "2026-10-06T10:00:00+00:00", probs=(0.8, 0.1, 0.1))
    late = pred("1", "2026-10-07T06:00:00+00:00", probs=(0.1, 0.8, 0.1))
    after = pred("1", "2026-10-07T13:00:00+00:00", probs=(0.0, 0.0, 1.0))  # after resolution: ignored
    chosen = ev.latest_before_resolution([early, late, after])
    assert chosen["1"]["logged_at"] == late["logged_at"]


def test_score_events_includes_market_and_skips_unresolved():
    preds = {"1": pred("1", "2026-10-06T10:00:00+00:00"), "2": pred("2", "2026-10-06T10:00:00+00:00")}
    outcomes = {"1": {"event_id": "1", "resolved_label": "b", "status": "final"}}
    rows = ev.score_events(preds, outcomes)
    assert len(rows) == 1 and rows[0]["k"] == 1
    assert rows[0]["market"] is not None and rows[0]["manager"]["hit"] is True
    s = ev.summarize(rows)
    assert s["ensemble"]["log_loss"] < s["climatology"]["log_loss"]  # ensemble was sure and right


def market(prices, closed=False, end="2026-10-06T12:00:00Z"):
    return {"closed": closed, "endDate": end, "markets": [
        {"groupItemTitle": l, "outcomePrices": f'["{p}", "{1 - p}"]'} for l, p in zip(LABELS, prices)]}


def test_resolver_final_provisional_open():
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    assert rs.classify(market([0.0005, 0.9995, 0.0005], closed=True), now) == ("final", "b")
    assert rs.classify(market([0.0005, 0.9995, 0.0005]), now) == ("provisional", "b")
    assert rs.classify(market([0.0005, 0.9995, 0.0005], end="2026-10-08T12:00:00Z"), now) == ("open", None)
    assert rs.classify(market([0.3, 0.6, 0.1], closed=True), now) == ("open", None)  # no clear winner
    assert rs.winning_bracket({"markets": [{"groupItemTitle": "a", "outcomePrices": "bad"}]}) is None


def test_versions_are_scored_separately_on_shared_events(capsys):
    old = pred("1", "2026-10-06T10:00:00+00:00", probs=(0.1, 0.4, 0.5))  # no stamp: counts as v1
    new = dict(pred("1", "2026-10-07T06:00:00+00:00", probs=(0.1, 0.8, 0.1)), pipeline_version="v2-x")
    only_new = dict(pred("2", "2026-10-07T06:00:00+00:00"), pipeline_version="v2-x")
    outcomes = {e: {"event_id": e, "resolved_label": "b", "status": "final"} for e in ("1", "2")}
    ev.print_version_comparison([old, new, only_new], outcomes)
    out = capsys.readouterr().out
    assert "1 events were predicted by every version" in out
    assert "v1 on shared events" in out and "v2-x on shared events" in out
    # latest-per-event across versions would have hidden v1 entirely
    assert ev.version_of(old) == "v1"


def test_latest_prediction_compares_instants_across_utc_and_ict_offsets():
    # 10:00 UTC is 17:00 ICT. A string compare would call the ICT stamp "09:00+07:00" earlier than "10:00+00:00";
    # as instants, 09:00+07:00 is 02:00 UTC, so the UTC record really is later.
    utc_later = pred("1", "2026-10-06T10:00:00+00:00", probs=(0.8, 0.1, 0.1))
    ict_earlier = pred("1", "2026-10-06T09:00:00+07:00", probs=(0.1, 0.1, 0.8))
    assert ev.latest_before_resolution([ict_earlier, utc_later])["1"] is utc_later
    # and an ICT stamp that is later as an instant wins even though its string sorts lower
    ict_later = pred("1", "2026-10-06T23:00:00+07:00", probs=(0.1, 0.8, 0.1))  # 16:00 UTC
    assert ev.latest_before_resolution([utc_later, ict_later])["1"] is ict_later


def test_timeutil_is_ict_and_round_trips():
    from datetime import datetime, timezone

    from weatheragents import timeutil

    assert timeutil.now().utcoffset().total_seconds() == 7 * 3600
    noon_utc = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
    assert timeutil.fmt(noon_utc) == "10-08 19:00"
    assert timeutil.to_ict(noon_utc).isoformat() == "2026-10-08T19:00:00+07:00"


def test_numeric_mode_records_without_a_manager_still_score(capsys):
    p = pred("1", "2026-10-06T10:00:00+00:00")
    rows = lambda ps: [{"label": l, "probability": x} for l, x in zip(LABELS, ps)]
    p["estimate"] = {"bracket_probabilities": None, "baseline_probabilities": rows((0.2, 0.6, 0.2)), "baseline_uncorrected_probabilities": None}
    p["ml"] = {"bracket_probabilities": rows((0.1, 0.7, 0.2))}
    p["pipeline_version"] = "v3-numeric"
    v = ev.vectors(p)
    assert v["manager"] is None and v["baseline"] is not None and v["ml"] is not None
    outcomes = {"1": {"event_id": "1", "resolved_label": "b", "status": "final"}}
    summary = ev.summarize(ev.score_events({"1": p}, outcomes))
    assert "manager" not in summary and summary["ml"]["log_loss"] < summary["baseline"]["log_loss"]
    ev.print_blend({"1": p}, outcomes)
    out = capsys.readouterr().out
    assert "ml (n=1)" in out and "baseline (n=1)" in out and "manager" not in out


def test_baseline_plus_leans_on_last_weeks_miss_beyond_the_usual_bias():
    from weatheragents import baseline as bl

    p = bl.Params(center=22.0, center_raw=22.0, sd=1.4, lower=None, upper=None, station_fit=True, lead_days=1, bias=0.4)
    assert bl.recent_miss_adjustment(p, 1.4) == pytest.approx(0.5 * (1.4 - 0.4))  # the forecast has run 1 C colder than usual lately
    assert bl.recent_miss_adjustment(p, 0.4) == 0  # exactly its usual bias: nothing new
    assert bl.recent_miss_adjustment(p, None) == 0  # no station history: baseline+ is just the baseline


def test_baseline_plus_is_a_separate_source():
    p = pred("1", "2026-10-06T10:00:00+00:00")
    rows = lambda ps: [{"label": l, "probability": x} for l, x in zip(LABELS, ps)]
    p["estimate"] = {"bracket_probabilities": None, "baseline_probabilities": rows((0.2, 0.6, 0.2)), "baseline_plus_probabilities": rows((0.1, 0.8, 0.1))}
    v = ev.vectors(p)
    assert v["baseline"] != v["baseline_plus"] and v["baseline_plus"] is not None
    assert ev.vectors(dict(p, estimate={"bracket_probabilities": None, "baseline_probabilities": rows((0.2, 0.6, 0.2))}))["baseline_plus"] is None


def test_city_bias_flags_a_city_that_is_consistently_off_and_ignores_noise_and_censored_outcomes():
    def event(eid, city, centre, label, unit="C"):
        p = pred(eid, "2026-10-06T10:00:00+00:00")
        p.update(city=city, unit=unit, ml={"center": centre})
        return p, {"event_id": eid, "resolved_label": label, "status": "final"}

    preds, outs = {}, {}
    for i in range(12):  # Cold City: the high lands 1 C above the model's centre every time, with a little scatter
        p, o = event(f"c{i}", "Cold City", 20.0 + (i % 3) * 0.1, "21°C")
        preds[p["event_id"]], outs[o["event_id"]] = p, o
    for i in range(12):  # Fine City: misses in both directions
        centre = 20.0
        p, o = event(f"f{i}", "Fine City", centre, f"{20 + (1 if i % 2 else -1)}°C")
        preds[p["event_id"]], outs[o["event_id"]] = p, o
    p, o = event("open", "Fine City", 20.0, "25°C or higher")  # censored: only a bound is known
    preds["open"], outs["open"] = p, o
    rows = {r["city"]: r for r in ev.city_bias(preds, outs)}
    assert rows["Fine City"]["n"] == 12  # the censored outcome is left out
    assert rows["Cold City"]["bias"] == pytest.approx(0.9, abs=0.02) and abs(rows["Cold City"]["t"]) >= 2
    assert abs(rows["Fine City"]["bias"]) < 0.01 and abs(rows["Fine City"]["t"]) < 2
    assert ev.outcome_value("88-89°F") == 88.5 and ev.outcome_value("22°C") == 22 and ev.outcome_value("19°C or higher") is None
    # Fahrenheit misses are reported in Celsius so cities can be compared
    pf, of = event("fx", "Hot City", 80.0, "86-87°F", unit="F")
    assert ev.city_bias({"fx": pf}, {"fx": of})[0]["bias"] == pytest.approx((86.5 - 80.0) / 1.8)

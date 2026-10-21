from datetime import date, datetime, timezone

import pytest

from weatheragents import research_debate as rd
from weatheragents.schemas import (
    Bracket,
    BracketProbability,
    ClimatologyReport,
    ContractSemantics,
    EnsembleReport,
    LocalContextReport,
    NowcastReport,
)

NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)
LABELS = ["21°C or below", "22°C", "23°C", "24°C or higher"]


def semantics() -> ContractSemantics:
    brackets = [Bracket(label=LABELS[0], low=None, high=21), Bracket(label="22°C", low=22, high=22),
                Bracket(label="23°C", low=23, high=23), Bracket(label=LABELS[3], low=24, high=None)]
    return ContractSemantics(
        event_id="1", city="Seoul", kind="highest", observation_date=date(2026, 10, 8), unit="C",
        station_name="Incheon", station_code="RKSI", resolution_source=None, brackets=brackets,
        resolves_at=NOW, rules="PRICE-FREE RULES",
    )


def reports() -> rd.AnalystReports:
    probs = [BracketProbability(label=l, probability=p) for l, p in zip(LABELS, (0.0, 0.6, 0.4, 0.0))]
    return rd.AnalystReports(
        ensemble=EnsembleReport(event_id="1", station_code="RKSI", observation_date=date(2026, 10, 8), model="m",
                                member_count=2, member_values=[22.0, 23.0], mean=22.5, std_dev=0.5,
                                bracket_probabilities=probs, generated_at=NOW),
        nowcast=NowcastReport(event_id="1", station_code="RKSI", observation_date=date(2026, 10, 8), unit="C",
                              forecast_extreme=22.5, previous_run_extremes={"3d": 22.0}, drift_vs_3d=0.5,
                              trend="steady", observed_extreme_so_far=None, observed_through_local=None,
                              latest_observed_temp=None, metar_reports_used=0, caveats=[], generated_at=NOW),
        climatology=ClimatologyReport(event_id="1", station_code="RKSI", observation_date=date(2026, 10, 8),
                                      kind="highest", unit="C", years_used=30, window_days=7, sample_size=450,
                                      mean=21.0, std_dev=2.0, p10=18.0, p50=21.0, p90=24.0, recent_10yr_mean=21.5,
                                      bracket_probabilities=probs, caveats=[], generated_at=NOW),
        local=LocalContextReport(event_id="1", station_code="RKSI", sources=[], summary="clear", leans="neutral",
                                 local_only_warnings=[], caveats=[], generated_at=NOW),
    )


def fake_llm(*answers):
    """Stand-in LLM whose structured() returns each answer in turn, validated against the schema."""
    queue = list(answers)
    prompts = []

    class Fake:
        def structured(self, *, system, user, schema, max_tokens=4096):
            prompts.append(user)
            return schema.model_validate(queue.pop(0))

    return Fake(), prompts


def params(**kw):
    from weatheragents.baseline import Params

    base = dict(center=22.5, center_raw=22.5, sd=1.4, lower=None, upper=None, station_fit=True, lead_days=1)
    return Params(**{**base, **kw})


def cases():
    w = rd.ResearchCase(event_id="1", side="warmer", argument="a", favored_brackets=[], evidence=[])
    c = rd.ResearchCase(event_id="1", side="cooler", argument="b", favored_brackets=[], evidence=[])
    return w, c


def adjust(shift=0.0, mult=1.0):
    return {"rationale": "r", "shift": shift, "spread_multiplier": mult, "confidence": 0.6}


def test_manager_with_no_adjustment_returns_the_baseline():
    llm, _ = fake_llm(adjust())
    w, c = cases()
    est = rd.manage(llm, semantics(), reports().ensemble, w, c, [], params=params())
    assert [bp.probability for bp in est.bracket_probabilities] == pytest.approx(
        [bp.probability for bp in est.baseline_probabilities]
    )
    assert est.shift == 0 and est.spread_multiplier == 1.0


def test_manager_shift_moves_mass_and_adjustments_are_clamped():
    llm, _ = fake_llm(adjust(shift=1.0))
    w, c = cases()
    est = rd.manage(llm, semantics(), reports().ensemble, w, c, [], params=params())
    probs = {bp.label: bp.probability for bp in est.bracket_probabilities}
    base = {bp.label: bp.probability for bp in est.baseline_probabilities}
    assert probs["24°C or higher"] > base["24°C or higher"] and probs["21°C or below"] < base["21°C or below"]
    assert sum(probs.values()) == pytest.approx(1)

    llm, _ = fake_llm(adjust(shift=50.0, mult=9.0))
    est = rd.manage(llm, semantics(), reports().ensemble, w, c, [], params=params())
    assert est.shift == 2.0 and est.spread_multiplier == 1.6  # 2 C and the spread cap, for a Celsius contract


def test_baseline_respects_observed_floor_and_scales_for_fahrenheit():
    from weatheragents import baseline as bl

    s = semantics()
    # high so far was 23: brackets entirely below 23 are impossible
    vec = bl.vector(s, center=22.0, sd=1.5, lower=22.5)
    got = {bp.label: bp.probability for bp in vec}
    assert got["21°C or below"] == 0 and got["22°C"] == 0
    assert sum(got.values()) == pytest.approx(1)
    # truth far above the forecast: all mass lands on the nearest allowed bracket, no crash
    far = bl.vector(s, center=10.0, sd=0.5, lower=22.5)
    assert max(bp.probability for bp in far) == 1.0
    assert bl.unit_scale("F") == 1.8 and bl.unit_scale("C") == 1.0


def test_baseline_params_blends_bias_and_applies_floor_for_highs():
    from weatheragents import baseline as bl

    r = reports()
    nc = r.nowcast.model_copy(update={"forecast_extreme": 20.0, "observed_extreme_so_far": 21.0, "lead_days": 0})
    ens = r.ensemble.model_copy(update={"mean": 22.0, "std_dev": 0.4})
    calib = bl.Calibration({"stations": {"RKSI|highest": {"day0": {"bias_c": 2.0, "sd_c": 0.9}, "day1": {"bias_c": 2.5, "sd_c": 1.8}}},
                            "default": bl.FALLBACK})
    p = bl.baseline_params(semantics(), ens, nc, calib)
    assert p.center == pytest.approx((22.0 + (20.0 + 2.0)) / 2)  # day0 bias used for a same-day contract
    assert p.sd == 0.9 and p.station_fit and p.lower == 20.5 and p.upper is None
    p2 = bl.baseline_params(semantics(), ens, nc, bl.Calibration())  # nothing fitted: default spreads
    assert not p2.station_fit and p2.sd == bl.FALLBACK["day0"]["sd_c"]


def test_prompts_carry_no_price_or_rules_text():
    s, r = semantics(), reports()
    w, _ = cases()
    texts = [rd.case_prompt(s, r), rd.case_prompt(s, r, opponent=w), rd.manager_prompt(s, r.ensemble, w, w, [], params())]
    for t in texts:
        for banned in ("best_bid", "best_ask", "liquidity", "last_trade", "volume", "yes_token", "PRICE-FREE RULES"):
            assert banned not in t


def test_debate_rounds_and_label_filtering():
    case = {"argument": "x", "favored_brackets": ["22°C", "99°C"], "evidence": ["e"]}
    llm, prompts = fake_llm(case, case, case, case)
    w, c, history = rd.debate(llm, semantics(), reports(), rounds=2)
    assert len(history) == 4 and len(prompts) == 4
    assert w.favored_brackets == ["22°C"]  # unknown label dropped
    assert "rebuttal" in prompts[2]


def test_local_research_filters_betting_sites_and_handles_no_results():
    from weatheragents import local_research_analyst as lr

    assert lr.blocked("https://polymarket.com/event/x") and lr.blocked("https://www.kalshi.com/a")
    assert not lr.blocked("https://www.kma.go.kr/forecast")

    llm, _ = fake_llm({"queries": ["a", "b", "c"]})
    rep = lr.research(llm, semantics(), search_fn=lambda q: [])
    assert rep.leans == "no_information"

    llm, prompts = fake_llm(
        {"queries": ["q"]},
        {"sources": ["KMA"], "summary": "clear", "leans": "banana", "local_only_warnings": [], "caveats": []},
    )
    rep = lr.research(llm, semantics(), search_fn=lambda q: [{"title": "KMA", "href": "https://kma.go.kr", "body": "sunny"}])
    assert rep.leans == "no_information"  # unknown value coerced
    assert "PRICE-FREE RULES" not in prompts[0] + prompts[1]


def test_local_research_drops_market_pages_by_content_not_just_domain():
    from weatheragents import local_research_analyst as lr

    market = {"title": "marketmarket.io prediction page for highest temperature in Seoul", "href": "https://x.example/a", "body": "Yes 56c"}
    odds = {"title": "Seoul", "href": "https://y.example/b", "body": "Will the highest temperature in Seoul be 22C? odds"}
    good = {"title": "KMA forecast", "href": "https://kma.go.kr/f", "body": "Clear, high 21"}
    assert lr.looks_like_market(market) and lr.looks_like_market(odds) and not lr.looks_like_market(good)

    llm, prompts = fake_llm(
        {"queries": ["q"]},
        {"sources": ["KMA"], "summary": "s", "leans": "neutral", "local_only_warnings": [], "caveats": []},
    )
    lr.research(llm, semantics(), search_fn=lambda q: [market, odds, good])
    assert "kma.go.kr" in prompts[1] and "x.example" not in prompts[1] and "y.example" not in prompts[1]


def test_search_prefers_tavily_and_falls_back(monkeypatch):
    import httpx

    from weatheragents import local_research_analyst as lr

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(lr, "tavily_search", lambda q, api_key, max_results=6: [{"title": "t", "href": "https://kma.go.kr", "body": "b"}])
    assert lr.search("q")[0]["href"] == "https://kma.go.kr"

    def boom(q, api_key, max_results=6):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(lr, "tavily_search", boom)
    monkeypatch.setattr(lr, "_ddgs_search", lambda q, max_results=6: [{"title": "f", "href": "https://x.example", "body": "b"}])
    assert lr.search("q")[0]["title"] == "f"  # fell back, did not raise

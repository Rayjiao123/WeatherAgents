from weatheragents import open_watch as ow


def test_llm_budget_resets_each_day():
    st = {"llm_day": "2026-10-15", "llm_used": 2}
    assert ow.llm_allowed(st, 2, "2026-10-15") is False
    assert ow.llm_allowed(st, 2, "2026-10-16") is True and st["llm_used"] == 0


def test_only_funded_parseable_events_count(monkeypatch):
    def parse(e):
        if e["id"] == "bad":
            raise ValueError("x")
        sem = type("S", (), {"station_code": "KLAX", "event_id": e["id"]})()
        return type("C", (), {"semantics": sem, "pricing": type("P", (), {"total_liquidity": e["liq"]})()})()

    monkeypatch.setattr(ow, "parse_event", parse)
    events = [{"id": "1", "liq": 9000}, {"id": "2", "liq": 0}, {"id": "bad", "liq": 9000}]
    assert [c.semantics.event_id for c in ow.funded_contracts(events)] == ["1"]

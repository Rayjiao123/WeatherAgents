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


def test_stale_highs_picks_funded_highs_without_a_recent_forecast():
    from datetime import datetime, timedelta, timezone

    tz = timezone(timedelta(hours=7))
    now = datetime(2026, 10, 23, 12, 0, tzinfo=tz)

    def contract(eid, kind="highest", hours=40, liq=9000):
        sem = type("S", (), {"event_id": eid, "kind": kind, "resolves_at": now + timedelta(hours=hours)})()
        return type("C", (), {"semantics": sem, "pricing": type("P", (), {"total_liquidity": liq})()})()

    funded = [contract("a"), contract("b"), contract("c"), contract("d", kind="lowest"), contract("e", hours=5), contract("f", liq=20000)]
    newest = {"a": (now - timedelta(hours=3)).isoformat(), "b": (now - timedelta(hours=30)).isoformat()}
    got = ow.stale_highs(funded, newest, now, skip={"c"})
    assert [c.semantics.event_id for c in got] == ["f", "b"]  # a is fresh, c is skipped, d is a low, e is too close to resolution; deepest book first

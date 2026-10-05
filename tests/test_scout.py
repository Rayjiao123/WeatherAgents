import json
from datetime import datetime, timezone

import pytest

from weatheragents.scout import ParseError, parse_bracket, parse_event, scout


def market(label, bid="0.10", ask="0.12"):
    return {
        "groupItemTitle": label,
        "outcomes": json.dumps(["Yes", "No"]),
        "clobTokenIds": json.dumps([f"yes-{label}", f"no-{label}"]),
        "bestBid": bid,
        "bestAsk": ask,
        "lastTradePrice": "0.11",
        "liquidityNum": 1000,
        "active": True,
        "closed": False,
    }


def event(**overrides):
    base = {
        "id": "123",
        "title": "Highest temperature in Wuhan on October 5?",
        "endDate": "2026-10-05T12:00:00Z",
        "liquidity": "50000",
        "volume24hr": "3000",
        "resolutionSource": None,
        "description": (
            "This market will resolve to the temperature range that contains the highest temperature "
            "recorded by NOAA at the Wuhan Tianhe International Airport Station in degrees Celsius on 5 Oct '26.\n"
            "available here: https://www.weather.gov/wrh/timeseries?site=zhhh"
        ),
        "markets": [market("23°C"), market("21°C or below"), market("25°C or higher"), market("22°C"), market("24°C")],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "label, low, high, unit",
    [
        ("22°C", 22, 22, "C"),
        ("-3°C", -3, -3, "C"),
        ("21°C or below", None, 21, "C"),
        ("31°C or higher", 31, None, "C"),
        ("62-63°F", 62, 63, "F"),
        ("61°F or below", None, 61, "F"),
    ],
)
def test_parse_bracket(label, low, high, unit):
    bracket, parsed_unit = parse_bracket(label)
    assert (bracket.low, bracket.high, parsed_unit) == (low, high, unit)


def test_parse_bracket_rejects_unknown():
    with pytest.raises(ParseError):
        parse_bracket("between 20 and 22")


def test_parse_event_semantics():
    c = parse_event(event()).semantics
    assert (c.city, c.kind, c.unit) == ("Wuhan", "highest", "C")
    assert c.station_code == "ZHHH"
    assert c.station_name == "Wuhan Tianhe International Airport"
    assert str(c.observation_date) == "2026-10-05"
    assert [b.label for b in c.brackets] == ["21°C or below", "22°C", "23°C", "24°C", "25°C or higher"]


def test_pricing_stays_aligned_with_sorted_brackets():
    c = parse_event(event())
    assert [b.label for b in c.pricing.brackets] == [b.label for b in c.semantics.brackets]
    assert c.pricing.brackets[0].yes_token_id == "yes-21°C or below"


def test_semantics_view_carries_no_price_fields():
    dumped = json.dumps(parse_event(event()).semantics_view().model_dump(mode="json"))
    for leak in ("best_bid", "best_ask", "last_trade", "liquidity", "token"):
        assert leak not in dumped


def test_station_from_wunderground_and_hko():
    taipei = event(
        title="Lowest temperature in Taipei on October 6?",
        resolutionSource="https://www.wunderground.com/history/daily/tw/taipei/RCSS",
        description="resolve based on the lowest temperature recorded in the 'Daily Observations' table",
    )
    assert parse_event(taipei).semantics.station_code == "RCSS"
    hk = event(title="Highest temperature in Hong Kong on October 6?", description="recorded by the Hong Kong Observatory")
    assert parse_event(hk).semantics.station_code == "HKO"


def test_scout_filters():
    now = datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc)
    events = [
        event(id="ok", endDate="2026-10-06T12:00:00Z"),
        event(id="soon", endDate="2026-10-05T03:00:00Z"),
        event(id="thin", endDate="2026-10-06T12:00:00Z", liquidity="100"),
        event(id="junk", title="Measles cases in U.S. in 2026?"),
    ]
    contracts, skipped = scout(events, min_liquidity=20_000, min_hours_left=6, now=now)
    assert [c.semantics.event_id for c in contracts] == ["ok"]
    assert len(skipped) == 3

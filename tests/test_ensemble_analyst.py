from datetime import date

import pytest

from weatheragents.ensemble_analyst import (
    ForecastRangeError,
    analyze_contract,
    bracket_probabilities,
    daily_variable,
    ensemble_report_from_payload,
    extract_member_values,
    resolve_coordinates,
)
from weatheragents.schemas import (
    Bracket,
    Contract,
    ContractPricing,
    ContractSemantics,
)

BRACKETS = [
    Bracket(label="21C or below", low=None, high=21),
    Bracket(label="22C", low=22, high=22),
    Bracket(label="23C", low=23, high=23),
    Bracket(label="24C", low=24, high=24),
    Bracket(label="25C or higher", low=25, high=None),
]


def semantics(**overrides) -> ContractSemantics:
    base = dict(
        event_id="123",
        city="Wuhan",
        kind="highest",
        observation_date=date(2026, 10, 7),
        unit="C",
        station_name="Wuhan Tianhe International Airport",
        station_code="ZHHH",
        resolution_source=None,
        brackets=BRACKETS,
        resolves_at="2026-10-07T12:00:00Z",
        rules="",
    )
    base.update(overrides)
    return ContractSemantics(**base)


def contract(**overrides) -> Contract:
    s = semantics(**overrides)
    return Contract(
        semantics=s,
        pricing=ContractPricing(event_id=s.event_id, total_liquidity=0, volume_24h=0, brackets=[]),
    )


def daily_payload(variable: str, obs_date: str, values: list[float]) -> dict:
    payload = {"time": [obs_date], variable: [sum(values) / len(values)]}
    for i, v in enumerate(values, start=1):
        payload[f"{variable}_member{i:02d}"] = [v]
    return payload


def test_daily_variable_picks_max_or_min():
    assert daily_variable("highest") == "temperature_2m_max"
    assert daily_variable("lowest") == "temperature_2m_min"


def test_resolve_coordinates_known_icao():
    lat, lon = resolve_coordinates("ZHHH")
    assert 30 < lat < 31
    assert 114 < lon < 115


def test_resolve_coordinates_special_station():
    lat, lon = resolve_coordinates("HKO")
    assert (lat, lon) == (22.3019, 114.1741)


def test_resolve_coordinates_unknown_raises():
    with pytest.raises(LookupError):
        resolve_coordinates("ZZZZ")


def test_extract_member_values():
    daily = daily_payload("temperature_2m_max", "2026-10-07", [24.0, 25.5, 23.1])
    values = extract_member_values(daily, variable="temperature_2m_max", obs_date=date(2026, 10, 7))
    assert sorted(values) == [23.1, 24.0, 25.5]


def test_extract_member_values_missing_date_raises():
    daily = daily_payload("temperature_2m_max", "2026-10-07", [24.0])
    with pytest.raises(ForecastRangeError):
        extract_member_values(daily, variable="temperature_2m_max", obs_date=date(2026, 10, 8))


def test_bracket_probabilities_partitions_members():
    # 20 members: 2 at 21 (or below bucket), 10 at 22, 8 at 24, nothing elsewhere.
    values = [21.0, 20.9] + [22.0] * 10 + [24.3] * 8
    result = bracket_probabilities(BRACKETS, values)
    by_label = {bp.label: bp.probability for bp in result}
    assert by_label["21C or below"] == pytest.approx(2 / 20)
    assert by_label["22C"] == pytest.approx(10 / 20)
    assert by_label["23C"] == 0.0
    assert by_label["24C"] == pytest.approx(8 / 20)
    assert by_label["25C or higher"] == 0.0
    assert sum(by_label.values()) == pytest.approx(1.0)


def test_bracket_probabilities_rounds_to_nearest_degree():
    # 22.4 rounds to 22 (bracket "22C"), 22.6 rounds to 23 (bracket "23C").
    result = bracket_probabilities(BRACKETS, [22.4, 22.6])
    by_label = {bp.label: bp.probability for bp in result}
    assert by_label["22C"] == pytest.approx(0.5)
    assert by_label["23C"] == pytest.approx(0.5)


def test_bracket_probabilities_gap_raises():
    gappy = [Bracket(label="10C", low=10, high=10), Bracket(label="12C", low=12, high=12)]
    with pytest.raises(ValueError):
        bracket_probabilities(gappy, [11.0])


def test_ensemble_report_from_payload_shapes_output():
    daily = daily_payload("temperature_2m_max", "2026-10-07", [22.0, 24.0])
    report = ensemble_report_from_payload(semantics(), daily, model="icon_seamless")
    assert report.event_id == "123"
    assert report.station_code == "ZHHH"
    assert report.member_count == 2
    assert report.mean == pytest.approx(23.0)
    assert sum(bp.probability for bp in report.bracket_probabilities) == pytest.approx(1.0)


def test_analyze_contract_no_station_code_raises(monkeypatch):
    bad = contract(station_code=None)
    with pytest.raises(LookupError):
        analyze_contract(bad, client=None, model="icon_seamless")


def test_analyze_contract_wires_fetch_to_report(monkeypatch):
    import weatheragents.ensemble_analyst as mod

    captured = {}

    def fake_fetch(client, *, lat, lon, obs_date, variable, unit, model):
        captured.update(lat=lat, lon=lon, obs_date=obs_date, variable=variable, unit=unit, model=model)
        return daily_payload(variable, obs_date.isoformat(), [22.0, 23.0, 24.0])

    monkeypatch.setattr(mod, "fetch_daily_payload", fake_fetch)

    report = analyze_contract(contract(), client=object(), model="icon_seamless")

    assert captured["variable"] == "temperature_2m_max"
    assert captured["unit"] == "C"
    assert 30 < captured["lat"] < 31
    assert report.member_count == 3

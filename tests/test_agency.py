import json

import pytest

from weatheragents import agency_forecasts as af
from weatheragents import agency_report as ar


def test_parsers_read_each_agencys_format():
    hko = {"weatherForecast": [{"forecastDate": "20261014", "forecastMaxtemp": {"value": 31}, "forecastMintemp": {"value": 26}, "PSR": "Low", "forecastWeather": "Fine"}]}
    assert af.parse_hko(hko) == [{"target_date": "2026-10-14", "high": 31.0, "low": 26.0, "rain": "Low", "text": "Fine"}]
    jma = [{}, {"timeSeries": [
        {"timeDefines": ["2026-10-14", "2026-10-15"], "areas": [{"area": {"name": "x"}, "pops": ["", "10"]}]},
        {"timeDefines": ["2026-10-14", "2026-10-15"], "areas": [{"area": {"name": "東京"}, "tempsMax": ["", "24"], "tempsMin": ["", "18"]}]}]}]
    rows = af.parse_jma(jma)
    assert rows[0]["high"] is None and rows[1] == {"target_date": "2026-10-15", "high": 24.0, "low": 18.0, "rain": 10.0, "text": None}
    mss = {"data": {"records": [{"forecasts": [{"timestamp": "2026-10-14T00:00:00+08:00", "temperature": {"high": 34, "low": 25}, "forecast": {"summary": "Thundery showers"}}]}]}}
    assert af.parse_mss(mss)[0]["high"] == 34.0


def test_slope_tells_a_useful_forecaster_from_noise():
    xs = [-2.0, -1.0, 1.0, 2.0, -1.5, 1.5, 0.5, -0.5]
    b, se = ar.slope(xs, [0.9 * x for x in xs])  # when the agency disagrees, it is right
    assert b == pytest.approx(0.9) and se == pytest.approx(0.0, abs=1e-9)
    b, _ = ar.slope(xs, [0.0 for _ in xs])  # the comparison was already right
    assert b == 0
    assert ar.slope([1.0], [1.0]) is None


def test_pairs_use_the_agency_forecast_collected_just_before_our_prediction():
    agency = [
        {"collected_at": "2026-10-12T06:00:00+07:00", "city": "Tokyo", "target_date": "2026-10-14", "lead_days": 2, "high": 24.0},
        {"collected_at": "2026-10-13T06:00:00+07:00", "city": "Tokyo", "target_date": "2026-10-14", "lead_days": 1, "high": 25.0},
        {"collected_at": "2026-10-14T06:00:00+07:00", "city": "Tokyo", "target_date": "2026-10-14", "lead_days": 0, "high": 26.0},  # after our prediction: must not be used
    ]
    pred = {"event_id": "1", "kind": "highest", "unit": "C", "city": "Tokyo", "observation_date": "2026-10-14", "logged_at": "2026-10-13T20:00:00+07:00",
            "ml": {"center": 23.0}, "ensemble": {"mean": 22.0}}
    snaps = {("Tokyo", "2026-10-14"): [{"ts": "2026-10-13T05:30:00+07:00", "implied_mean": 24.5}]}
    (pair,) = ar.build_pairs(agency, {"1": pred}, {"1": {"resolved_label": "25°C"}}, snaps)
    assert pair["agency"] == 25.0 and pair["lead"] == 1 and pair["ours"] == 23.0 and pair["market"] == 24.5 and pair["actual"] == 25.0
    assert ar.build_pairs(agency, {"1": dict(pred, unit="F")}, {"1": {"resolved_label": "77-78°F"}}, snaps) == []  # Celsius cities only

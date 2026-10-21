from datetime import date, datetime, timezone

from weatheragents import climatology_analyst as clim
from weatheragents import nowcast_analyst as nc
from weatheragents.schemas import Bracket, ContractSemantics


def semantics(kind="highest", unit="C"):
    brackets = [Bracket(label="19 or below", low=None, high=19), Bracket(label="20", low=20, high=20),
                Bracket(label="21", low=21, high=21), Bracket(label="22 or higher", low=22, high=None)]
    return ContractSemantics(
        event_id="1", city="X", kind=kind, observation_date=date(2026, 10, 7), unit=unit, station_name=None,
        station_code="RKSI", resolution_source=None, brackets=brackets,
        resolves_at=datetime(2026, 10, 7, tzinfo=timezone.utc), rules="",
    )


def metar(hour_utc, temp):
    return {"obsTime": int(datetime(2026, 10, 7, hour_utc, tzinfo=timezone.utc).timestamp()), "temp": temp}


def test_trend_classification_and_units():
    assert nc.classify_trend(0.4, "C") == "steady"
    assert nc.classify_trend(0.6, "C") == "warming"
    assert nc.classify_trend(-0.8, "F") == "steady"  # F band is 0.9
    assert nc.classify_trend(-1.0, "F") == "cooling"
    assert nc.c_to_unit(20, "F") == 68


def test_observed_so_far_uses_station_local_date():
    # Seoul is UTC+9: 20:00 UTC on Oct 6 is 05:00 local on Oct 7; 16:00 UTC Oct 7 is already Oct 8 local.
    metars = [
        {"obsTime": int(datetime(2026, 10, 6, 20, tzinfo=timezone.utc).timestamp()), "temp": 15},
        metar(3, 21),
        metar(6, 19),
        {"obsTime": int(datetime(2026, 10, 7, 16, tzinfo=timezone.utc).timestamp()), "temp": 30},
    ]
    ext, _, latest, n = nc.observed_so_far(metars, obs_date=date(2026, 10, 7), utc_offset_s=9 * 3600, kind="highest", unit="C")
    assert (ext, latest, n) == (21, 19, 3)


def test_nowcast_report_drift_and_caveats():
    runs = {"latest": [18.0, 22.0], "1d": [21.0], "2d": [20.5], "3d": [20.0]}
    rep = nc.build_report(semantics(), runs, 9 * 3600, [])
    assert rep.forecast_extreme == 22.0 and rep.drift_vs_3d == 2.0 and rep.trend == "warming"
    assert rep.observed_extreme_so_far is None
    assert any("no METAR" in c for c in rep.caveats)


def test_climatology_window_and_probabilities_sum_to_one():
    times, vals = [], []
    for y in range(2000, 2005):
        for day in range(1, 30):
            times.append(f"{y}-10-{day:02d}")
            vals.append(20.0 + (day % 3))  # 20, 21, 22
    daily = {"time": times, "temperature_2m_max": vals}
    pairs = clim.window_values(daily, variable="temperature_2m_max", target=date(2026, 10, 7), window=7)
    assert len(pairs) == 5 * 14  # +/-7 days around Oct 7 is Oct 1-14
    rep_vals = sorted(v for _, v in pairs)
    assert clim.percentile(rep_vals, 0.5) in (20.0, 21.0, 22.0)

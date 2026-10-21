import pytest

from weatheragents import sameday_analyst as sd


def test_spread_narrows_as_the_peak_passes():
    assert sd.spread_for(6) > sd.spread_for(2) > sd.spread_for(0)
    assert sd.spread_for(-3) == sd.spread_for(0) == pytest.approx(0.4)  # past the usual peak: little left to learn
    assert sd.spread_for(40) == 1.8


def test_projection_floors_at_the_high_so_far_and_follows_todays_forecast_miss():
    forecast = {float(h): 20.0 + min(h, 15) * 0.4 - max(h - 15, 0) * 0.5 for h in range(24)}  # peaks 26.0 at 15:00
    # a cool day: the station has run 1.5 C under the forecast over the last hours
    cool = [(float(h), forecast[float(h)] - 1.5) for h in range(8, 12)]
    pr = sd.project(cool, forecast, 12.0)
    assert pr["bias_adjustment"] == pytest.approx(-1.5 * sd.BIAS_DAMPING)
    assert pr["center"] == pytest.approx(26.0 - 1.5 * sd.BIAS_DAMPING)
    # past the peak the high so far wins, even if the hourly forecast said more
    late = [(float(h), 27.0 if h == 14 else 22.0) for h in range(10, 19)]
    pr = sd.project(late, forecast, 18.5)
    assert pr["center"] == 27.0 and pr["sd"] == pytest.approx(0.4)

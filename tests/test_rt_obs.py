import math

import pytest

from weatheragents import rt_obs as rt


def rep(h, t=20.0, dwp=12.0, drct=90.0, sknt=8.0, sky=0.0, vsby=6.0, alti=29.92, p01i=0.0, wx=""):
    return {"h": h, "tmpc": t, "dwpc": dwp, "drct": drct, "sknt": sknt, "sky": sky, "vsby": vsby, "alti": alti, "p01i": p01i, "wx": wx}


STEADY = [rep(h / 2) for h in range(0, 30)]  # half-hourly from 00:00 to 14:30, nothing changing


def test_a_steady_morning_shows_no_change():
    f = rt.own_features(STEADY, 14.0)
    assert f["ft_dp3"] == pytest.approx(0) and f["ft_wshift"] == pytest.approx(0) and f["ft_dwp_chg3"] == 0
    assert f["ft_wx_rain"] == 0 and f["ft_cloud"] == 0 and f["ft_temp_chg1"] == 0


def test_a_front_shows_as_falling_pressure_a_wind_shift_rising_dew_point_and_rain():
    rows = []
    for h2 in range(0, 30):
        h = h2 / 2
        late = h >= 12.5  # the front arrives at 12:30
        rows.append(rep(h, t=22.0 - (6 if late else 0), dwp=12.0 + (4 if late else 0), drct=90.0 if not late else 270.0,
                        sknt=8.0 if not late else 18.0, alti=29.92 - (0.09 if late else 0), p01i=0.05 if late else 0.0, wx="-RA" if late else "", sky=1.0 if late else 0.0))
    f = rt.own_features(rows, 14.0)
    assert f["ft_dp3"] == pytest.approx(-0.09 * rt.HPA_PER_INHG, abs=0.05)  # a 3 hPa drop in 3 hours
    assert f["ft_wshift"] == pytest.approx(180.0) and f["ft_wind_chg"] == pytest.approx(10.0)
    assert f["ft_dwp_chg3"] == pytest.approx(4.0) and f["ft_wx_rain"] == 1.0 and f["ft_cloud"] == 1.0
    assert f["ft_rain3"] == pytest.approx(0.05 * 25.4)


def test_only_reports_up_to_the_cutoff_are_used():
    rows = STEADY + [rep(15.0, alti=28.0, drct=270.0, wx="TS")]
    assert rt.own_features(rows, 14.0)["ft_wx_rain"] == 0  # the storm at 15:00 has not happened yet at 14:00
    assert rt.own_features(rows, 15.0)["ft_wx_rain"] == 1
    assert all(math.isnan(v) for v in rt.own_features(STEADY[:2], 14.0).values())  # too few reports


def test_upstream_stations_count_only_when_the_wind_is_coming_from_their_direction():
    # target at (0, 0); a station due west (270) and one due east (90); an easterly wind comes FROM the east
    west = (0.0, -1.0, 110.0, ([0.0, 3 * 3600.0], [10.0, 10.0]))
    east = (0.0, 1.0, 110.0, ([0.0, 3 * 3600.0], [12.0, 18.0]))  # warmed by 6 C over 3 hours
    f = rt.upstream_features(0.0, 0.0, 90.0, 3 * 3600.0, 15.0, [west, east])
    assert f["up_n"] == 1.0  # the western station is downwind and is ignored
    assert f["up_trend3"] == pytest.approx(6.0) and f["up_grad"] == pytest.approx(3.0)
    assert rt.upstream_features(0.0, 0.0, None, 3 * 3600.0, 15.0, [east])["up_n"] == 0  # calm air: no upwind
    assert math.isnan(rt.upstream_features(0.0, 0.0, 90.0, 99 * 3600.0, 15.0, [east])["up_grad"])  # the neighbour's data is stale


def test_bearing_points_the_right_way():
    assert rt.bearing(0, 0, 1, 0) == pytest.approx(0) and rt.bearing(0, 0, 0, 1) == pytest.approx(90) and rt.bearing(0, 0, 0, -1) == pytest.approx(270)

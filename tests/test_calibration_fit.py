import pytest

from weatheragents import calibration_fit as cf


def test_daily_extremes_groups_by_local_date_and_needs_enough_readings():
    times = ["2026-10-01 00:00", "2026-10-01 12:00", "2026-10-02 00:00"]
    assert cf.daily_extremes(times, [10.0, 20.0, 5.0], "highest", 2) == {"2026-10-01": 20.0}
    assert cf.daily_extremes(times, [10.0, 20.0, 5.0], "lowest", 2) == {"2026-10-01": 10.0}
    assert cf.daily_extremes(times, [None, 20.0, 5.0], "highest", 2) == {}


def test_error_stats_shrinks_bias_floors_spread_and_drops_outliers():
    errors = [2.0] * 100 + [50.0]  # constant +2 bias, one absurd outlier
    st = cf.error_stats(errors)
    assert st["n"] == 100  # outlier dropped
    assert st["bias_c"] == pytest.approx(2.0 * 100 / 115, abs=1e-3)  # pulled toward 0
    assert st["sd_c"] == cf.SD_FLOOR_C  # zero real spread, floored
    with pytest.raises(ValueError):
        cf.error_stats([1.0] * 5)


def test_pooled_default_has_no_bias():
    stations = {"A|highest": {"day0": {"n": 50, "bias_c": 2.0, "sd_c": 1.0}, "day1": {"n": 50, "bias_c": 2.0, "sd_c": 2.0}},
                "B|highest": {"day0": {"n": 50, "bias_c": -1.0, "sd_c": 2.0}, "day1": {"n": 50, "bias_c": -1.0, "sd_c": 3.0}}}
    d = cf.pooled_default(stations)
    assert d["day0"]["bias_c"] == 0.0 and d["day0"]["sd_c"] == 1.5 and d["day1"]["sd_c"] == 2.5

import math

import numpy as np
import pytest

from weatheragents import sameday_ml as sm

PATH = [18 + min(h, 15) * 0.5 - max(h - 15, 0) * 0.4 for h in range(24)]  # peaks 25.5 at 15:00
REPORTS = [(h + m / 60, PATH[h] - 1.0) for h in range(0, 20) for m in (0, 30)]  # the station has run 1 C cool all day
MODELS = {"ecmwf_ifs025": 25.5, "gfs_global": 26.5}


def row(cutoff, reports=REPORTS):
    return sm.make_sd_row("2026-10-12", cutoff, reports, PATH, MODELS, 37.0, 127.0, 10.0)


def test_features_describe_the_day_so_far_and_the_forecast_miss():
    r = row(12.0)
    assert r["sd_high_so_far"] == pytest.approx(PATH[12] - 1.0)
    assert r["sd_dev3"] == pytest.approx(-1.0)  # running a degree under the forecast
    assert r["sd_fc_rest_max"] == pytest.approx(25.5)  # the forecast peak is still ahead
    assert r["sd_proj_rise"] == pytest.approx(25.5 - 0.7 - r["sd_high_so_far"])
    assert r["sd_hour"] == 12 and r["sd_fc_std"] == pytest.approx(0.5)
    assert set(r) == set(sm.SD_FEATURES)


def test_later_reports_cannot_leak_into_an_earlier_cutoff():
    base = row(12.0)
    leaky = REPORTS + [(15.0, 99.0), (16.5, 99.0)]  # a scorching afternoon that has not happened yet at 12:00
    assert row(12.0, sorted(leaky)) == base
    assert row(17.0, sorted(leaky))["sd_high_so_far"] == 99.0  # but by 17:00 it has


def test_past_the_peak_nothing_is_left_to_rise():
    r = row(18.0)
    assert r["sd_since_high"] > 2  # the high was hours ago
    assert math.isnan(row(23.5)["sd_fc_rest_max"]) or row(23.5)["sd_fc_rest_max"] <= PATH[23]


def test_too_few_reports_gives_no_row():
    assert sm.make_sd_row("2026-10-12", 7.0, [(0.0, 18.0), (1.0, 17.5)], PATH, MODELS, 37.0, 127.0, 10.0) is None


def test_truncated_scoring_never_credits_probability_below_the_high_so_far():
    y = np.array([22.0])
    floor = np.array([21.5])  # the station already recorded 22
    right = sm.trunc_metrics(y, np.array([22.0]), np.array([1.0]), floor)
    off = sm.trunc_metrics(y, np.array([25.0]), np.array([1.0]), floor)
    assert right["int_log_loss"] < off["int_log_loss"]
    # a centre far below the floor is treated as all mass just above it, not as an error or nan
    low = sm.trunc_metrics(y, np.array([15.0]), np.array([1.0]), floor)
    assert math.isfinite(low["int_log_loss"])
    # the cut-off makes a wide curve sharper where it matters: same centre, truncated beats untruncated
    flat = sm.trunc_metrics(y, np.array([22.0]), np.array([2.0]), floor)["int_log_loss"]
    assert flat < -math.log(0.2)

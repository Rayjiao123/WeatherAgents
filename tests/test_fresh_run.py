import pytest

import pytest

from weatheragents import fresh_run as fr


def payload(hours=48, models=("ecmwf_ifs025", "gfs_global", "icon_global")):
    times = [f"2026-10-15T{h:02d}:00" for h in range(24)] + [f"2026-10-16T{h:02d}:00" for h in range(24)]
    series = {f"temperature_2m_{m}": [20 + min(h % 24, 15) * 0.5 + i for h in range(hours)] for i, m in enumerate(models)}
    return {"hourly": {"time": times[:hours], **series}}


def test_fresh_forecast_takes_each_models_high_for_the_target_local_date_only():
    out = fr.parse_fresh(payload(), "2026-10-15")
    assert out["models"] == {"ecmwf_ifs025": 27.5, "gfs_global": 28.5, "icon_global": 29.5}
    assert out["mean"] == pytest.approx(28.5)
    assert fr.parse_fresh(payload(), "2026-10-17") is None  # a date the response does not cover


def test_needs_two_models_with_a_full_day():
    assert fr.parse_fresh(payload(models=("ecmwf_ifs025",)), "2026-10-15") is None
    short = payload(hours=10)  # only 10 hours of the day returned
    assert fr.parse_fresh(short, "2026-10-15") is None


def test_fresh_inputs_replace_only_todays_forecast_and_what_follows_from_it():
    import math

    from weatheragents import ml_analyst as ma
    from weatheragents.ml_data import ALL_FEATURES, MODELS

    row = {f: 0.0 for f in ALL_FEATURES}
    row.update({"fc_ecmwf_ifs025": 20.0, "fc_gfs_global": 22.0, "fc_icon_global": math.nan, "fc_jma_seamless": math.nan,
                "fc_mean": 21.0, "fc_std": 1.0, "dT": 0.5, "clim_mean": 19.0, "clim_anom": 2.0, "err_mean": 0.3, "err_lag2": 0.1})
    fresh = {"models": {"ecmwf_ifs025": 22.0, "gfs_global": 24.0, "icon_global": 23.0, "ukmo_seamless": 99.0}}  # an unknown model is ignored
    out = ma.fresh_row(row, fresh)
    assert out["fc_ecmwf_ifs025"] == 22.0 and out["fc_icon_global"] == 23.0  # a model that had no day-ahead data is filled in
    assert out["fc_mean"] == pytest.approx((22 + 24 + 23) / 3)
    shift = out["fc_mean"] - 21.0
    assert out["dT"] == pytest.approx(0.5 + shift) and out["clim_anom"] == pytest.approx(out["fc_mean"] - 19.0)
    assert out["err_mean"] == 0.3 and out["err_lag2"] == 0.1  # past days are untouched
    assert ma.fresh_row(row, None) is None
    assert ma.fresh_row(row, {"models": {"ecmwf_ifs025": 22.0}}) is None  # one model is not a forecast
    assert row["fc_mean"] == 21.0  # the original row is not modified

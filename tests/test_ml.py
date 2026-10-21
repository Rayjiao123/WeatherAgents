import math
import statistics
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from weatheragents import ml_analyst, ml_data, ml_train
from weatheragents.ml_data import ALL_FEATURES, FAMILIES
from weatheragents.schemas import Bracket, ContractSemantics


def synthetic_hourly(days=14, start=date(2026, 6, 1), lead=1):
    """Two models, hourly temps peaking at 15:00, plus flat weather variables."""
    times, series = [], {}
    for m, bump in (("ecmwf_ifs025", 0.0), ("gfs_global", 2.0)):
        for v in ("temperature_2m",) + ml_data.WEATHER_VARS:
            series[f"{v}_previous_day{lead}_{m}"] = []
    for d in range(days):
        day = start + timedelta(days=d)
        for h in range(24):
            times.append(f"{day.isoformat()}T{h:02d}:00")
            for m, bump in (("ecmwf_ifs025", 0.0), ("gfs_global", 2.0)):
                series[f"temperature_2m_previous_day{lead}_{m}"].append(20 + d * 0.1 + bump - abs(h - 15) * 0.5)
                for v in ml_data.WEATHER_VARS:
                    series[f"{v}_previous_day{lead}_{m}"].append(50.0 + h)
    return {"time": times, **series}


def test_hourly_to_daily_takes_each_models_high_and_averages_weather():
    daily = ml_data.hourly_to_daily(synthetic_hourly(), 1)
    rec = daily["2026-06-01"]
    assert rec["fc"] == {"ecmwf_ifs025": pytest.approx(20.0), "gfs_global": pytest.approx(22.0)}
    assert ml_data.fc_mean(rec) == pytest.approx(21.0)
    assert not math.isnan(rec["cloud"]) and not math.isnan(rec["pres"])


def same_features(a: dict, b: dict) -> bool:
    """Equal, treating two missing (NaN) values as the same."""
    return a.keys() == b.keys() and all(a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in a)


def test_features_cannot_see_the_target_days_outcome():
    """The point of the whole exercise: nothing from the target day, or the day before it, leaks in."""
    daily = ml_data.hourly_to_daily(synthetic_hourly(), 1)
    target = "2026-06-12"
    obs = {(date(2026, 6, 1) + timedelta(days=i)).isoformat(): 21.0 + i * 0.1 for i in range(14)}
    aero = {k: {"aod": 0.1, "dust": 1.0, "pm10": 10.0} for k in obs}
    base = ml_data.make_features(target, 1, daily, obs, aero, 37.0, 127.0, 10.0)

    leaky = dict(obs)
    leaky[target] = 99.0  # the outcome itself
    leaky["2026-06-11"] = -50.0  # the day before: still in progress when a day-ahead prediction is made
    leaky_aero = dict(aero)
    leaky_aero[target] = {"aod": 9.0, "dust": 999.0, "pm10": 999.0}  # same-day aerosol reflects the day itself
    assert same_features(ml_data.make_features(target, 1, daily, leaky, leaky_aero, 37.0, 127.0, 10.0), base)

    older = dict(obs)
    older["2026-06-10"] = 50.0  # day-2 is allowed to matter
    assert ml_data.make_features(target, 1, daily, older, aero, 37.0, 127.0, 10.0)["err_lag2"] != base["err_lag2"]
    aero2 = dict(aero)
    aero2["2026-06-11"] = {"aod": 5.0, "dust": 50.0, "pm10": 80.0}  # yesterday's aerosol is allowed
    assert ml_data.make_features(target, 1, daily, obs, aero2, 37.0, 127.0, 10.0)["aod_lag1"] == 5.0


def test_missing_models_become_nan_not_errors():
    daily = ml_data.hourly_to_daily(synthetic_hourly(), 1)
    row = ml_data.make_features("2026-06-12", 1, daily, {}, {}, 37.0, 127.0, 10.0)
    assert math.isnan(row["fc_icon_global"]) and math.isnan(row["err_mean"]) and math.isnan(row["aod_lag1"])
    assert set(ALL_FEATURES) == set(row)
    assert ml_data.make_features("2030-01-01", 1, daily, {}, {}, 0, 0, 0) is None  # no forecast for that day


def test_metrics_reward_honest_uncertainty():
    rng = np.random.default_rng(0)
    y = rng.normal(20.0, 1.5, 4000)
    honest = ml_train.metrics(y, np.full(4000, 20.0), np.full(4000, 1.5))
    overconfident = ml_train.metrics(y, np.full(4000, 20.0), np.full(4000, 0.5))
    assert honest["z_sd"] == pytest.approx(1.0, abs=0.05)
    assert honest["crps"] < overconfident["crps"] and honest["int_log_loss"] < overconfident["int_log_loss"]


def test_model_learns_a_real_signal_and_the_ablation_can_see_it(monkeypatch):
    monkeypatch.setattr(ml_train, "ROUNDS", 60)
    rng = np.random.default_rng(1)
    n = 2400
    X = np.full((n, len(ALL_FEATURES)), np.nan)
    col = ALL_FEATURES.index
    X[:, col("fc_mean")] = rng.normal(20, 5, n)
    X[:, col("doy_sin")] = rng.uniform(-1, 1, n)
    X[:, col("cloud")] = rng.uniform(0, 100, n)
    for f in FAMILIES["base"]:
        if np.isnan(X[:, col(f)]).all():
            X[:, col(f)] = 1.0
    y = X[:, col("fc_mean")] - 0.05 * X[:, col("cloud")] + rng.normal(0, 0.4, n)  # cloudier days run cooler than forecast
    dates = np.array([(date(2025, 1, 1) + timedelta(days=i // 6)).isoformat() for i in range(n)])
    tr, te = slice(0, 1800), slice(1800, n)

    without = ml_train.fit_model(X[tr], y[tr], dates[tr], FAMILIES["base"])
    with_wx = ml_train.fit_model(X[tr], y[tr], dates[tr], FAMILIES["base"] + FAMILIES["weather"])
    m0 = ml_train.metrics(y[te], *without.predict(X[te]))
    m1 = ml_train.metrics(y[te], *with_wx.predict(X[te]))
    assert m1["mae"] < m0["mae"] * 0.7  # the cloud signal is found when its family is included, and only then
    assert 0.7 < m1["z_sd"] < 1.3 and ml_train.SD_MIN <= with_wx.predict(X[te])[1].min()


def test_baseline_fit_applies_shrunk_bias_per_station():
    station = np.array(["A"] * 100 + ["B"] * 5)
    lead = np.ones(105)
    resid = np.concatenate([np.full(100, 2.0), np.full(5, 2.0)])
    b = ml_train.fit_baseline(station, lead, resid)
    assert ("A", 1.0) in b["stations"] and ("B", 1.0) not in b["stations"]  # too few rows for B: pooled spread, no bias
    c, sd = ml_train.baseline_predict(b, np.array(["A", "B"]), np.ones(2), np.array([20.0, 20.0]))
    assert 21.5 < c[0] < 22.0 and c[1] == 20.0 and sd[0] >= 0.6


def sem(kind="highest", unit="C"):
    brackets = [Bracket(label="a", low=None, high=20), Bracket(label="b", low=21, high=21), Bracket(label="c", low=22, high=None)]
    return ContractSemantics(event_id="1", city="X", kind=kind, observation_date=date(2026, 10, 12), unit=unit, station_name=None,
                             station_code="RKSI", resolution_source=None, brackets=brackets,
                             resolves_at=datetime(2026, 10, 12, tzinfo=timezone.utc), rules="")


def test_ml_analyst_declines_what_it_was_not_trained_for():
    from weatheragents.schemas import Contract, ContractPricing

    def contract(**kw):
        s = sem(**kw)
        return Contract(semantics=s, pricing=ContractPricing(event_id="1", total_liquidity=0, volume_24h=0, brackets=[]))

    loaded = (object(), {"trained_until": "2026-10-01"})
    assert ml_analyst.analyze_contract(contract(), None, lead_days=1, loaded=None) is None  # no model trained
    assert ml_analyst.analyze_contract(contract(kind="lowest"), None, lead_days=1, loaded=loaded) is None
    assert ml_analyst.analyze_contract(contract(), None, lead_days=0, loaded=loaded) is None
    assert ml_analyst.analyze_contract(contract(), None, lead_days=3, loaded=loaded) is None


def neighbor_bundle(code="NB", shift=0.0):
    daily = ml_data.hourly_to_daily(synthetic_hourly(days=40), 1)
    obs = {(date(2026, 6, 1) + timedelta(days=i)).isoformat(): 21.0 + i * 0.1 + shift for i in range(40)}
    return ml_data.Bundle(code, 36.0, 128.0, 5.0, "Asia/Seoul", {1: daily, 2: {}}, obs, {})


def test_nearest_neighbors_picks_close_stations_and_skips_far_or_self():
    coords = {"RKSI": (37.47, 126.45), "RKPK": (35.18, 128.94), "ZSQD": (36.27, 120.37), "RJTT": (35.55, 139.78), "KLAX": (33.9, -118.4)}
    near = ml_data.nearest_neighbors("RKSI", coords)
    assert [c for c, _ in near] == ["RKPK", "ZSQD"]  # Tokyo is ~1200 km away, LA is across the ocean
    assert near[0][1] < near[1][1] < ml_data.NEIGHBOR_RADIUS_KM
    assert ml_data.nearest_neighbors("NOPE", coords) == []


def test_neighbor_features_summarize_recent_errors_and_stay_in_the_past():
    day = "2026-07-05"
    b = neighbor_bundle()
    base = ml_data.neighbor_features(day, 1, [(b, 300.0)])
    assert base["nb_n"] == 1.0 and base["nb_dist"] == 300.0
    assert not math.isnan(base["nb_err_lag2"]) and not math.isnan(base["nb_err_mean"]) and not math.isnan(base["nb_anom_lag2"])

    leaky = neighbor_bundle()
    for k in (0, 1):  # the target day and the day before it
        leaky.obs[(date(2026, 7, 5) - timedelta(days=k)).isoformat()] = 99.0
    assert ml_data.neighbor_features(day, 1, [(leaky, 300.0)]) == base

    older = neighbor_bundle()
    older.obs["2026-07-03"] = 50.0  # day-2 is allowed to matter
    assert ml_data.neighbor_features(day, 1, [(older, 300.0)])["nb_err_lag2"] != base["nb_err_lag2"]


def test_no_neighbors_gives_missing_not_zero():
    f = ml_data.neighbor_features("2026-07-05", 1, [])
    assert f["nb_n"] == 0.0 and math.isnan(f["nb_err_lag2"]) and math.isnan(f["nb_dist"])
    daily = ml_data.hourly_to_daily(synthetic_hourly(), 1)
    row = ml_data.make_features("2026-06-12", 1, daily, {}, {}, 37.0, 127.0, 10.0)
    assert row["nb_n"] == 0.0 and math.isnan(row["nb_err_mean"])  # default when no neighbor data is passed in


def test_normals_average_a_window_across_years_and_feed_the_climatology_features():
    times, vals = [], []
    for y in range(2010, 2020):
        for i in range(60):  # Oct 1 .. Nov 29 each year
            d = date(y, 10, 1) + timedelta(days=i)
            times.append(d.isoformat())
            vals.append(20.0 if y == 2010 else 22.0)  # one cooler year and nine warmer ones
    normals = ml_data.build_normals({"time": times, "temperature_2m_max": vals})
    doy = ml_data.doy365(date(2026, 10, 15))
    mean, sd = normals[doy]
    assert mean == pytest.approx(21.8) and sd == pytest.approx(0.6, abs=1e-3)
    assert ml_data.doy365(date(2024, 12, 31)) == 365 and ml_data.doy365(date(2025, 12, 31)) == 365  # leap day folded in

    daily = ml_data.hourly_to_daily(synthetic_hourly(), 1)
    row = ml_data.make_features("2026-06-12", 1, daily, {}, {}, 37.0, 127.0, 10.0, normals={ml_data.doy365(date(2026, 6, 12)): (18.0, 2.0)})
    assert row["clim_mean"] == 18.0 and row["clim_sd"] == 2.0
    assert row["clim_anom"] == pytest.approx(row["fc_mean"] - 18.0)
    missing = ml_data.make_features("2026-06-12", 1, daily, {}, {}, 37.0, 127.0, 10.0)
    assert math.isnan(missing["clim_mean"]) and math.isnan(missing["clim_anom"])


def test_run_drift_is_the_gap_between_this_runs_forecast_and_the_one_a_day_earlier():
    day = "2026-06-12"
    newer = ml_data.hourly_to_daily(synthetic_hourly(), 1)
    older = ml_data.hourly_to_daily(synthetic_hourly(lead=2), 2)
    older = {d: {**rec, "fc": {m: v - 1.5 for m, v in rec["fc"].items()}} for d, rec in older.items()}  # the earlier run was 1.5 C cooler
    row = ml_data.make_features(day, 1, newer, {}, {}, 37.0, 127.0, 10.0, older=older)
    assert row["run_drift"] == pytest.approx(1.5)
    assert math.isnan(ml_data.make_features(day, 1, newer, {}, {}, 37.0, 127.0, 10.0)["run_drift"])  # no earlier run supplied
    assert "run_drift" in ml_data.FAMILIES["nowcast"] and set(ml_data.ALL_FEATURES) == set(row)


def round2_inputs():
    daily = ml_data.hourly_to_daily(synthetic_hourly(days=40), 1)
    obs = {(date(2026, 6, 1) + timedelta(days=i)).isoformat(): 21.5 + i * 0.1 for i in range(40)}
    normals = {ml_data.doy365(date(2026, 6, 1) + timedelta(days=i)): (20.0, 2.0) for i in range(-40, 80)}
    ctx = {"obs": {k: {"dwp": 12.0, "dep_aft": 9.0, "wind": 6.0, "wsin": 0.5, "wcos": 0.8, "cloud": 0.3, "vis": 6.0} for k in obs},
           "traits": {"st_reports_per_day": 48.0}}
    return daily, obs, normals, ctx


def test_round_two_groups_describe_recent_skill_volatility_and_anomaly():
    daily, obs, normals, ctx = round2_inputs()
    r = ml_data.make_features("2026-07-05", 1, daily, obs, {}, 37.0, 127.0, 10.0, normals=normals, ctx=ctx)
    # synthetic models: ecmwf peaks at 20.0 and gfs at 22.0 plus a 0.1/day drift; the station ran ~21.5+ so ecmwf under-forecast more
    assert r["err_gfs_global"] < r["err_ecmwf_ifs025"]  # the cooler model has missed low by more, so its error is larger
    assert math.isnan(r["err_icon_global"])  # a model with no data gives no skill estimate
    assert r["fc_skill_best"] == r["fc_gfs_global"]  # the model whose recent misses are smallest
    assert r["fc_skill_adj"] == pytest.approx(statistics.fmean([r["fc_ecmwf_ifs025"] + r["err_ecmwf_ifs025"], r["fc_gfs_global"] + r["err_gfs_global"]]))
    assert r["err_std"] >= 0 and r["err_absmax"] >= r["err_absmean"] > 0
    assert r["obs_anom3"] > 0 and r["obs_anom30"] > 0  # the station has been running above its normal of 20
    assert r["ob_dwp"] == 12.0 and r["ob_cloud"] == 0.3 and r["st_reports_per_day"] == 48.0
    assert set(ml_data.ALL_FEATURES) == set(r)


def test_round_two_groups_cannot_see_the_target_day_or_the_day_before():
    daily, obs, normals, ctx = round2_inputs()
    base = ml_data.make_features("2026-07-05", 1, daily, obs, {}, 37.0, 127.0, 10.0, normals=normals, ctx=ctx)
    leaky_obs = dict(obs)
    leaky_obs["2026-07-05"], leaky_obs["2026-07-04"] = 99.0, -40.0
    leaky_ctx = {"obs": {**ctx["obs"], "2026-07-05": {"dwp": 99.0}, "2026-07-04": {"dwp": 99.0}}, "traits": ctx["traits"]}
    after = ml_data.make_features("2026-07-05", 1, daily, leaky_obs, {}, 37.0, 127.0, 10.0, normals=normals, ctx=leaky_ctx)
    assert same_features(after, base)
    older = dict(obs)
    older["2026-07-03"] = 60.0  # day-2 is allowed to matter
    assert ml_data.make_features("2026-07-05", 1, daily, older, {}, 37.0, 127.0, 10.0, normals=normals, ctx=ctx)["obs_anom3"] != base["obs_anom3"]


def test_condition_summaries_turn_station_reports_into_a_days_airmass():
    rows = []
    for h in range(0, 24):
        rows.append([f"2026-10-08 {h:02d}:00", str(14 + min(h, 12) * 0.5), "10.0", "90", "10", "BKN" if h % 2 else "", "6.2"])
    out = ml_data.summarize_conditions(rows)["2026-10-08"]
    assert out["dwp"] == 10.0 and out["wind"] == 10.0 and out["vis"] == 6.2
    assert out["cloud"] == pytest.approx(0.375)  # half the reports broken cloud, half clear (blank = clear)
    assert out["wsin"] == pytest.approx(1.0) and abs(out["wcos"]) < 1e-6  # wind from due east
    assert out["dep_aft"] == pytest.approx(10.0)  # 20 C in the afternoon against a 10 C dew point
    assert ml_data.summarize_conditions(rows[:5]) == {}  # too few reports for a trustworthy day


def test_variable_sets_get_separate_cache_keys_but_the_standard_set_keeps_its_old_one():
    standard = ("temperature_2m",) + ml_data.WEATHER_VARS
    assert ml_data._vkey(standard) == "" and ml_data._vkey(ml_data.TEMP_ONLY) == "_temp"
    assert ml_data._vkey(standard + ml_data.EXTRA_VARS) not in ("", "_temp")


def test_more_models_summarize_the_extra_forecasts_and_how_high_resolution_ones_differ():
    rec = {"fc": {"ecmwf_ifs025": 20.0, "gfs_global": 22.0}}  # global mean 21.0
    xm = {"ukmo_seamless": 21.0, "icon_d2": 23.0, "jma_msm": 25.0}  # two of the three are regional high-resolution models
    f = ml_data._more_models_features(21.0, rec, xm)
    assert f["xm_n"] == 3 and f["xm_mean"] == pytest.approx(23.0)
    assert f["hires_mean"] == pytest.approx(24.0) and f["hires_gap"] == pytest.approx(3.0)
    assert f["all_mean"] == pytest.approx((20 + 22 + 21 + 23 + 25) / 5)
    none = ml_data._more_models_features(21.0, rec, None)  # outside every regional domain: missing, not zero
    assert none["xm_n"] == 0 and math.isnan(none["hires_mean"]) and math.isnan(none["hires_gap"]) and none["all_std"] > 0


def test_extra_model_highs_need_a_full_day_and_cache_keys_are_distinct(tmp_path, monkeypatch):
    times = [f"2026-06-01T{h:02d}:00" for h in range(24)] + [f"2026-06-02T{h:02d}:00" for h in range(10)]
    hourly = {"time": times, "temperature_2m_previous_day1_icon_d2": [20.0 + i * 0.1 for i in range(len(times))]}
    out = ml_data.extra_models_daily(hourly, 1)
    assert out == {"2026-06-01": {"icon_d2": pytest.approx(22.3)}}  # the half day of June 2 is not trusted

    monkeypatch.setattr(ml_data, "RAW_DIR", tmp_path)
    a, b = date(2026, 1, 1), date(2026, 1, 20)
    assert not ml_data.forecast_cached(1.0, 2.0, a, b)
    (tmp_path / (ml_data._fc_key(1.0, 2.0, a, b, ml_data.LEADS, ("temperature_2m",) + ml_data.WEATHER_VARS, ml_data.MODELS) + ".json")).write_text("{}")
    assert ml_data.forecast_cached(1.0, 2.0, a, b)  # the standard request now counts as on disk
    assert not ml_data.forecast_cached(1.0, 2.0, a, b, variables=ml_data.TEMP_ONLY, models=ml_data.EXTRA_MODELS)  # a different request does not
    assert len({ml_data._mkey(ml_data.MODELS), ml_data._mkey(ml_data.EXTRA_MODELS)}) == 2

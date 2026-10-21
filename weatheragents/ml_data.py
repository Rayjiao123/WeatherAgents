"""Data and features for the ML Analyst.

Everything here is something that existed at prediction time. Forecasts come from
Open-Meteo's previous-runs API, which keeps each model's forecast *as it was issued*
1 and 2 days before the day it covers (reanalysis would leak the outcome). Truth is
the station's own METAR record (Iowa State archive), the same family of source that
Polymarket resolves on. Rows are one per (station, date, lead).

Feature families (ml_train.py measures each one's contribution):
  base           the multi-model mean forecast, season, location, lead
  multimodel     each model's own forecast, their disagreement, day-over-day change
  weather        cloud, humidity, pressure and its change, sunshine, wind
  recent_errors  how wrong the forecast was at this station over the last week
  aerosol        yesterday's dust / haze / particulates (cuts daytime highs)
  neighbors      how wrong the forecast was at nearby stations on recent complete days, and how
                 unusual their weather was (model errors are often shared across a region)
  nowcast        how far the forecast has drifted between the run issued 2 days ahead and the one 1 day ahead
                 (the Nowcast Analyst's run-drift signal, now an input; its observed-so-far part is for same-day work)
  model_skill    each weather model's own miss at this station over the last week, and the forecast of whichever
                 model has been closest lately
  volatility     how erratic those misses have been (feeds the spread)
  anomaly        how far the station's recent highs ran from normal
  obs_conditions the station's last complete day of dew point, wind, cloud cover and visibility
  station_traits how often the station reports (more reports catch a higher maximum)
  weather2       precipitation, humidity, sunshine, wind direction, vapour-pressure deficit (a large extra download)
  more_models    twelve more weather models, including regional high-resolution ones: their average, their
                 disagreement, and how the high-resolution ones differ from the global average
  climatology    what is normal for this date (30-year average and spread) and how far the forecast
                 sits from it. The Climatology Analyst's job, folded in as an input.

Only highs are modelled; lows are rare in the contract list and behave differently.
"""

import csv
import io
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

from weatheragents.calibration_fit import IEM_URL, daily_extremes

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MODELS = ("ecmwf_ifs025", "gfs_global", "icon_global", "jma_seamless")
LEADS = (1, 2)
# Twelve more weather models that keep history: global ones from other countries, and regional high-resolution ones
# that only cover their own domain (elsewhere they return nothing, which the model treats as missing).
EXTRA_MODELS = ("ukmo_seamless", "meteofrance_seamless", "gem_seamless", "kma_seamless", "cma_grapes_global", "bom_access_global",
                "icon_d2", "meteofrance_arome_france_hd", "ukmo_uk_deterministic_2km", "jma_msm", "ncep_hrrr_conus", "ncep_nbm_conus")
HIRES_MODELS = ("icon_d2", "meteofrance_arome_france_hd", "ukmo_uk_deterministic_2km", "jma_msm", "ncep_hrrr_conus", "ncep_nbm_conus")
EXTRA_VARS = ("precipitation", "relative_humidity_2m", "sunshine_duration", "wind_direction_10m", "vapour_pressure_deficit")
WEATHER_VARS = ("cloud_cover", "dew_point_2m", "surface_pressure", "shortwave_radiation", "wind_speed_10m")
START = date(2024, 4, 1)  # multi-model history is complete from about here
RAW_DIR = Path("data") / "ml" / "raw"
NAN = math.nan

FAMILIES: dict[str, list[str]] = {
    "base": ["fc_mean", "doy_sin", "doy_cos", "lat", "lon", "elev", "lead"],
    "multimodel": [f"fc_{m}" for m in MODELS] + ["fc_std", "dT"],
    "weather": ["cloud", "dew", "pres", "pres_chg", "rad", "wind"],
    "recent_errors": ["err_lag2", "err_mean"],
    "aerosol": ["aod_lag1", "dust_lag1", "pm10_lag1"],
    "neighbors": ["nb_err_lag2", "nb_err_mean", "nb_anom_lag2", "nb_n", "nb_dist"],
    "climatology": ["clim_mean", "clim_anom", "clim_sd"],
    "nowcast": ["run_drift"],
    # round 2
    "model_skill": [f"err_{m}" for m in MODELS] + ["fc_skill_best", "fc_skill_adj"],
    "volatility": ["err_std", "err_absmax", "err_absmean"],
    "anomaly": ["obs_anom3", "obs_anom7", "obs_anom30"],
    "obs_conditions": ["ob_dwp", "ob_dep_aft", "ob_wind", "ob_wdir_sin", "ob_wdir_cos", "ob_cloud", "ob_vis_min"],
    "station_traits": ["st_reports_per_day"],
    "weather2": ["precip", "rh", "sun", "wdir_sin", "wdir_cos", "vpd"],
    "more_models": ["xm_n", "xm_mean", "xm_std", "all_mean", "all_std", "hires_mean", "hires_gap"],
}
ROUND2 = ["model_skill", "volatility", "anomaly", "obs_conditions", "station_traits", "weather2", "more_models"]
KEPT = ["base", "multimodel", "weather", "recent_errors", "climatology"]  # the groups the saved model uses
ALL_FEATURES = [f for fam in FAMILIES.values() for f in fam]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else NAN


def hourly_to_daily(hourly: dict, lead: int) -> dict[str, dict]:
    """Per local date: each model's forecast high, plus cross-model means of the weather variables."""
    idx: dict[str, list[int]] = {}
    for i, t in enumerate(hourly["time"]):
        idx.setdefault(t[:10], []).append(i)
    short = {"cloud_cover": "cloud", "dew_point_2m": "dew", "surface_pressure": "pres", "shortwave_radiation": "rad", "wind_speed_10m": "wind"}
    out = {}
    for day, ii in idx.items():
        fc, wx = {}, {v: [] for v in WEATHER_VARS}
        for m in MODELS:
            temp = hourly.get(f"temperature_2m_previous_day{lead}_{m}")
            if temp:
                vals = [temp[i] for i in ii if temp[i] is not None]
                if len(vals) >= 20:
                    fc[m] = max(vals)
            for v in WEATHER_VARS:
                s = hourly.get(f"{v}_previous_day{lead}_{m}")
                if not s:
                    continue
                if v in ("surface_pressure", "shortwave_radiation"):
                    vals = [s[i] for i in ii if s[i] is not None]
                    need = 20
                else:  # daytime only
                    vals = [s[i] for i in ii if s[i] is not None and 8 <= int(hourly["time"][i][11:13]) <= 18]
                    need = 8
                if len(vals) >= need:
                    wx[v].append(sum(vals) / len(vals) if v != "shortwave_radiation" else sum(vals) / 1000)
        out[day] = {"fc": fc, **{short[v]: _mean(wx[v]) for v in WEATHER_VARS}, **_extra_daily(hourly, ii, lead)}
    return out


def _extra_daily(hourly: dict, ii: list[int], lead: int) -> dict:
    """Round-2 weather columns, averaged over the models that have them; NaN when the series were not downloaded."""
    acc = {"precip": [], "rh": [], "sun": [], "wsin": [], "wcos": [], "vpd": []}
    for m in MODELS:
        def col(v):
            return hourly.get(f"{v}_previous_day{lead}_{m}")

        pr, rh, sun, wd, vpd = col("precipitation"), col("relative_humidity_2m"), col("sunshine_duration"), col("wind_direction_10m"), col("vapour_pressure_deficit")
        day = [i for i in ii if 8 <= int(hourly["time"][i][11:13]) <= 18]
        if pr and sum(pr[i] is not None for i in ii) >= 20:
            acc["precip"].append(sum(pr[i] for i in ii if pr[i] is not None))
        if rh and sum(rh[i] is not None for i in day) >= 8:
            acc["rh"].append(_mean([rh[i] for i in day if rh[i] is not None]))
        if sun and sum(sun[i] is not None for i in ii) >= 20:
            acc["sun"].append(sum(sun[i] for i in ii if sun[i] is not None) / 3600)
        if wd and sum(wd[i] is not None for i in day) >= 8:
            ang = [math.radians(wd[i]) for i in day if wd[i] is not None]
            acc["wsin"].append(_mean([math.sin(a) for a in ang]))
            acc["wcos"].append(_mean([math.cos(a) for a in ang]))
        if vpd and sum(vpd[i] is not None for i in day) >= 8:
            acc["vpd"].append(_mean([vpd[i] for i in day if vpd[i] is not None]))
    return {"precip": _mean(acc["precip"]), "rh": _mean(acc["rh"]), "sun": _mean(acc["sun"]),
            "wdir_s": _mean(acc["wsin"]), "wdir_c": _mean(acc["wcos"]), "vpd": _mean(acc["vpd"])}


def extra_models_daily(hourly: dict, lead: int) -> dict[str, dict[str, float]]:
    """{date: {model: forecast high}} for the extra models that have a full day of data."""
    idx: dict[str, list[int]] = {}
    for i, t in enumerate(hourly["time"]):
        idx.setdefault(t[:10], []).append(i)
    out: dict[str, dict[str, float]] = {}
    for m in EXTRA_MODELS:
        series = hourly.get(f"temperature_2m_previous_day{lead}_{m}")
        if not series:
            continue
        for day, ii in idx.items():
            vals = [series[i] for i in ii if series[i] is not None]
            if len(vals) >= 20:
                out.setdefault(day, {})[m] = max(vals)
    return out


def _more_models_features(mean: float, rec: dict, xm_day: dict | None) -> dict:
    vals = list((xm_day or {}).values())
    hi = [v for m, v in (xm_day or {}).items() if m in HIRES_MODELS]
    every = list(rec["fc"].values()) + vals
    return {"xm_n": float(len(vals)), "xm_mean": _mean(vals), "xm_std": statistics.pstdev(vals) if len(vals) >= 2 else NAN,
            "all_mean": _mean(every), "all_std": statistics.pstdev(every) if len(every) >= 2 else NAN,
            "hires_mean": _mean(hi), "hires_gap": _mean(hi) - mean if hi else NAN}


def fc_mean(day_rec: dict | None) -> float:
    vals = list(day_rec["fc"].values()) if day_rec else []
    return sum(vals) / len(vals) if len(vals) >= 2 else NAN


def doy365(d: date) -> int:
    """Day of year with the leap day folded into Dec 31, so every year has the same 365 slots."""
    return min(d.timetuple().tm_yday, 365)


def build_normals(daily: dict, variable: str = "temperature_2m_max", window: int = 7) -> dict[int, tuple[float, float]]:
    """Normal high and spread for each day of the year: all years' values within +/-window days of it."""
    by_doy: dict[int, list[float]] = {}
    for t, v in zip(daily["time"], daily[variable]):
        if v is not None:
            by_doy.setdefault(doy365(date.fromisoformat(t)), []).append(float(v))
    out = {}
    for doy in range(1, 366):
        vals = [x for k in range(-window, window + 1) for x in by_doy.get((doy - 1 + k) % 365 + 1, [])]
        if len(vals) >= 100:
            out[doy] = (statistics.fmean(vals), statistics.pstdev(vals))
    return out


_NORMALS: dict[tuple[float, float], dict | None] = {}


def normals_for(lat: float, lon: float) -> dict[int, tuple[float, float]] | None:
    """30-year normals for a station from the climatology analyst's on-disk ERA5 cache (None if unavailable).

    Built from 1996 through last year, so the 2026 test period is outside it; training years 2024-25 are
    inside it by about 1/30 of the sample, a leak too small to matter for a normal.
    """
    key = (round(lat, 3), round(lon, 3))
    if key not in _NORMALS:
        from weatheragents.climatology_analyst import _fetch_archive_c

        try:
            _NORMALS[key] = build_normals(_fetch_archive_c(lat, lon, "temperature_2m_max", date.today().year - 1))
        except (httpx.HTTPError, KeyError, ValueError):
            _NORMALS[key] = None
    return _NORMALS[key]


NB_EMPTY = {"nb_err_lag2": NAN, "nb_err_mean": NAN, "nb_anom_lag2": NAN, "nb_n": 0.0, "nb_dist": NAN}
NEIGHBOR_RADIUS_KM = 700.0
NEIGHBOR_COUNT = 4


@dataclass
class Bundle:
    """Everything fetched for one station: as-issued forecasts per lead, station highs, aerosols."""

    station: str
    lat: float
    lon: float
    elev: float
    tz: str
    daily: dict[int, dict[str, dict]] = field(default_factory=dict)
    obs: dict[str, float] = field(default_factory=dict)
    aero: dict[str, dict] = field(default_factory=dict)
    normals: dict[int, tuple[float, float]] | None = None
    obs_ctx: dict[str, dict] = field(default_factory=dict)  # last-day conditions by local date (dew point, wind, cloud, visibility)
    traits: dict[str, float] = field(default_factory=dict)  # static facts about the station, e.g. reports per day
    extra_daily: dict[int, dict[str, dict]] = field(default_factory=dict)  # lead -> date -> {extra model: forecast high}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def nearest_neighbors(code: str, coords: dict[str, tuple[float, float]], *, radius: float = NEIGHBOR_RADIUS_KM, k: int = NEIGHBOR_COUNT) -> list[tuple[str, float]]:
    """Other tracked stations within `radius` km, nearest first: [(code, distance_km)]."""
    if code not in coords:
        return []
    lat, lon = coords[code]
    near = sorted((haversine_km(lat, lon, la, lo), c) for c, (la, lo) in coords.items() if c != code)
    return [(c, d) for d, c in near if d <= radius][:k]


def neighbor_features(day: str, lead: int, neighbors: list[tuple[Bundle, float]]) -> dict:
    """Regional signals from complete days only (day-2 and earlier, like the station's own error features).

    nb_err_lag2   neighbors' (observed high - day-ahead forecast) on day-2, averaged
    nb_err_mean   the same averaged over day-2..day-8
    nb_anom_lag2  neighbors' day-2 high minus their own average of the 30 days before it
    """
    d = date.fromisoformat(day)
    iso = lambda k: (d - timedelta(days=k)).isoformat()
    e_lag2, e_mean, anom, dists = [], [], [], []
    for b, dist in neighbors:
        daily = b.daily.get(lead, {})
        used = False
        o2, f2 = b.obs.get(iso(2)), fc_mean(daily.get(iso(2)))
        if o2 is not None and not math.isnan(f2):
            e_lag2.append(o2 - f2)
            used = True
        errs = [b.obs[iso(k)] - fc_mean(daily.get(iso(k))) for k in range(2, 9) if iso(k) in b.obs and not math.isnan(fc_mean(daily.get(iso(k))))]
        if len(errs) >= 3:
            e_mean.append(sum(errs) / len(errs))
            used = True
        hist = [b.obs[iso(k)] for k in range(3, 33) if iso(k) in b.obs]
        if o2 is not None and len(hist) >= 10:
            anom.append(o2 - sum(hist) / len(hist))
            used = True
        if used:
            dists.append(dist)
    avg = lambda xs: sum(xs) / len(xs) if xs else NAN
    return {"nb_err_lag2": avg(e_lag2), "nb_err_mean": avg(e_mean), "nb_anom_lag2": avg(anom), "nb_n": float(len(dists)), "nb_dist": avg(dists)}


def make_features(
    day: str, lead: int, daily: dict[str, dict], obs: dict[str, float], aero: dict[str, dict], lat: float, lon: float, elev: float,
    nb: dict | None = None, normals: dict[int, tuple[float, float]] | None = None, older: dict[str, dict] | None = None,
    ctx: dict | None = None, xm: dict[str, dict] | None = None,
) -> dict | None:
    """Features for predicting `day`'s high. Uses only information from before `day` plus the
    day-ahead forecast for it: observations from day-2 and earlier, aerosols from day-1."""
    rec = daily.get(day)
    mean = fc_mean(rec)
    if math.isnan(mean):
        return None
    d = date.fromisoformat(day)
    prev = daily.get((d - timedelta(days=1)).isoformat())
    prev_mean = fc_mean(prev)
    vals = list(rec["fc"].values())
    errs = []
    for k in range(2, 9):  # day-2 .. day-8: complete days only
        dk = (d - timedelta(days=k)).isoformat()
        e = obs.get(dk, NAN) - fc_mean(daily.get(dk))
        errs.append(e)
    good = [e for e in errs if not math.isnan(e)]
    a1 = aero.get((d - timedelta(days=1)).isoformat(), {})
    doy = d.timetuple().tm_yday
    row = {
        "fc_mean": mean,
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "lat": lat, "lon": lon, "elev": elev, "lead": float(lead),
        **{f"fc_{m}": rec["fc"].get(m, NAN) for m in MODELS},
        "fc_std": statistics.pstdev(vals) if len(vals) >= 2 else NAN,
        "dT": mean - prev_mean if not math.isnan(prev_mean) else NAN,
        "cloud": rec["cloud"], "dew": rec["dew"], "pres": rec["pres"], "rad": rec["rad"], "wind": rec["wind"],
        "pres_chg": rec["pres"] - prev["pres"] if prev and not math.isnan(prev["pres"]) and not math.isnan(rec["pres"]) else NAN,
        "err_lag2": errs[0],
        "err_mean": sum(good) / len(good) if len(good) >= 4 else NAN,
        "aod_lag1": a1.get("aod", NAN), "dust_lag1": a1.get("dust", NAN), "pm10_lag1": a1.get("pm10", NAN),
        **(nb if nb is not None else NB_EMPTY),
    }
    norm = normals.get(doy365(d)) if normals else None
    older_mean = fc_mean(older.get(day)) if older else NAN
    row["run_drift"] = mean - older_mean if not math.isnan(older_mean) else NAN  # vs the forecast one run-day earlier
    row.update(_round2_features(d, mean, rec, daily, obs, normals, ctx))
    row.update(_more_models_features(mean, rec, (xm or {}).get(day)))
    row["clim_mean"] = norm[0] if norm else NAN
    row["clim_sd"] = norm[1] if norm else NAN
    row["clim_anom"] = mean - norm[0] if norm else NAN
    return row


def _round2_features(d: date, mean: float, rec: dict, daily: dict, obs: dict, normals: dict | None, ctx: dict | None) -> dict:
    """The round-2 groups. Same rule as the rest: only complete days before the target (day-2 and earlier)."""
    iso = lambda k: (d - timedelta(days=k)).isoformat()
    out: dict = {}
    # which model has been right lately
    errs_m: dict[str, float] = {}
    for m in MODELS:
        e = [obs[iso(k)] - daily[iso(k)]["fc"][m] for k in range(2, 9) if iso(k) in obs and iso(k) in daily and m in daily[iso(k)]["fc"]]
        out[f"err_{m}"] = sum(e) / len(e) if len(e) >= 4 else NAN
        if len(e) >= 4:
            errs_m[m] = sum(e) / len(e)
    usable = [m for m in errs_m if m in rec["fc"]]
    out["fc_skill_best"] = rec["fc"][min(usable, key=lambda m: abs(errs_m[m]))] if usable else NAN
    out["fc_skill_adj"] = _mean([rec["fc"][m] + errs_m[m] for m in usable])
    # how erratic the recent misses have been
    errs = [obs[iso(k)] - fc_mean(daily.get(iso(k))) for k in range(2, 9) if iso(k) in obs and not math.isnan(fc_mean(daily.get(iso(k))))]
    out["err_std"] = statistics.pstdev(errs) if len(errs) >= 4 else NAN
    out["err_absmax"] = max(abs(e) for e in errs) if len(errs) >= 4 else NAN
    out["err_absmean"] = _mean([abs(e) for e in errs]) if len(errs) >= 4 else NAN
    # how far the station's recent highs ran from normal
    for n in (3, 7, 30):
        a = [obs[iso(k)] - normals[doy365(d - timedelta(days=k))][0] for k in range(2, n + 2)
             if iso(k) in obs and normals and doy365(d - timedelta(days=k)) in normals]
        out[f"obs_anom{n}"] = sum(a) / len(a) if len(a) >= max(2, n // 2) else NAN
    # the station's last complete day of conditions, and static traits
    oc = (ctx or {}).get("obs", {}).get(iso(2), {})
    out.update({"ob_dwp": oc.get("dwp", NAN), "ob_dep_aft": oc.get("dep_aft", NAN), "ob_wind": oc.get("wind", NAN),
                "ob_wdir_sin": oc.get("wsin", NAN), "ob_wdir_cos": oc.get("wcos", NAN), "ob_cloud": oc.get("cloud", NAN), "ob_vis_min": oc.get("vis", NAN)})
    out["st_reports_per_day"] = (ctx or {}).get("traits", {}).get("st_reports_per_day", NAN)
    # extra weather columns (NaN unless downloaded)
    out.update({"precip": rec.get("precip", NAN), "rh": rec.get("rh", NAN), "sun": rec.get("sun", NAN), "wdir_sin": rec.get("wdir_s", NAN),
                "wdir_cos": rec.get("wdir_c", NAN), "vpd": rec.get("vpd", NAN)})
    return out


# ---------------------------------------------------------------- fetching


def _cached(key: str, use_cache: bool, fn):
    path = RAW_DIR / f"{key}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = fn()
    if use_cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _get(client: httpx.Client, url: str, params: dict, tries: int = 6) -> httpx.Response:
    """GET with retries for rate limits (429) and dropped connections."""
    for attempt in range(tries):
        try:
            resp = client.get(url, params=params)
        except httpx.TransportError:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 429 and "Daily" not in resp.text and attempt < tries - 1:
            time.sleep(20 * (attempt + 1))  # a per-minute or per-hour limit clears soon; a daily one does not
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable")


TEMP_ONLY = ("temperature_2m",)


def _vkey(variables: tuple[str, ...]) -> str:
    """Cache-key suffix: nothing for the standard variable set, so older downloads keep working; distinct for others."""
    import hashlib

    if tuple(variables) == ("temperature_2m",) + WEATHER_VARS:
        return ""
    return "_temp" if tuple(variables) == TEMP_ONLY else "_v" + hashlib.md5(",".join(variables).encode()).hexdigest()[:6]


def _mkey(models: tuple[str, ...]) -> str:
    import hashlib

    return "" if tuple(models) == MODELS else "_m" + hashlib.md5(",".join(models).encode()).hexdigest()[:6]


def _fc_key(lat: float, lon: float, cs: date, ce: date, leads, variables, models) -> str:
    return f"fc_{lat:.3f}_{lon:.3f}_{cs}_{ce}" + ("" if leads == LEADS else f"_l{leads}") + _vkey(variables) + _mkey(models)


def forecast_cached(lat: float, lon: float, start: date, end: date, *, leads=LEADS, variables=("temperature_2m",) + WEATHER_VARS, models=MODELS) -> bool:
    """True when every download chunk for this request is already on disk."""
    cur = start
    while cur <= end:
        ce = min(cur + timedelta(days=119), end)
        if not (RAW_DIR / (_fc_key(lat, lon, cur, ce, leads, variables, models) + ".json")).exists():
            return False
        cur = ce + timedelta(days=1)
    return True


def fetch_forecasts(client: httpx.Client, lat: float, lon: float, start: date, end: date, *, use_cache: bool,
                    leads: tuple[int, ...] = LEADS, variables: tuple[str, ...] = ("temperature_2m",) + WEATHER_VARS,
                    models: tuple[str, ...] = MODELS) -> dict:
    """Hourly as-issued forecasts for 4 models x 6 variables x 2 leads. Returns {hourly, tz, elev}."""
    hourly_names = ",".join((f"{v}_previous_day{d}" if d > 0 else v) for v in variables for d in leads)
    merged: dict[str, list] = {}
    tz, elev = "UTC", 0.0
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=119), end)

        def go(cs=cur, ce=chunk_end):
            r = _get(client, PREVIOUS_RUNS_URL, {
                "latitude": lat, "longitude": lon, "hourly": hourly_names, "models": ",".join(models),
                "start_date": cs.isoformat(), "end_date": ce.isoformat(), "timezone": "auto"}).json()
            return {"hourly": r["hourly"], "tz": r["timezone"], "elev": r.get("elevation", 0.0)}

        part = _cached(_fc_key(lat, lon, cur, chunk_end, leads, variables, models), use_cache and chunk_end <= date.today() - timedelta(days=3), go)
        tz, elev = part["tz"], part["elev"]
        for k, v in part["hourly"].items():
            merged.setdefault(k, []).extend(v)
        cur = chunk_end + timedelta(days=1)
    return {"hourly": merged, "tz": tz, "elev": elev}


def _obs_parts(client: httpx.Client, station: str, tz: str, start: date, end: date, *, use_cache: bool):
    """Station reports in 90-day chunks (local time strings and temperatures), cached on disk."""
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=89), end)

        def go(cs=cur, ce=chunk_end):
            resp = _get(client, IEM_URL, {
                "station": station, "data": "tmpc", "year1": cs.year, "month1": cs.month, "day1": cs.day,
                "year2": ce.year, "month2": ce.month, "day2": ce.day, "tz": tz, "format": "onlycomma",
                "latlon": "no", "elev": "no", "missing": "empty", "trace": "T", "direct": "no", "report_type": [3, 4]})
            times, vals = [], []
            for row in csv.DictReader(io.StringIO(resp.text)):
                times.append(row["valid"])
                vals.append(float(row["tmpc"]) if row.get("tmpc") not in (None, "", "M") else None)
            return {"times": times, "vals": vals}

        yield _cached(f"obs_{station}_{cur}_{chunk_end}", use_cache and chunk_end <= date.today() - timedelta(days=3), go)
        cur = chunk_end + timedelta(days=1)


def fetch_obs(client: httpx.Client, station: str, tz: str, start: date, end: date, *, use_cache: bool) -> dict[str, float]:
    """Station daily highs (local date -> degrees C) from the Iowa State METAR archive."""
    out: dict[str, float] = {}
    for part in _obs_parts(client, station, tz, start, end, use_cache=use_cache):
        out.update(daily_extremes(part["times"], part["vals"], "highest", 8))
    return out


def fetch_reports(client: httpx.Client, station: str, tz: str, start: date, end: date, *, use_cache: bool) -> dict[str, list[tuple[float, float]]]:
    """Every station report by local date: {date: [(local hour as a decimal, degrees C)]}, in time order."""
    out: dict[str, list[tuple[float, float]]] = {}
    for part in _obs_parts(client, station, tz, start, end, use_cache=use_cache):
        for t, v in zip(part["times"], part["vals"]):
            if v is not None:
                out.setdefault(t[:10], []).append((int(t[11:13]) + int(t[14:16]) / 60, float(v)))
    for v in out.values():
        v.sort()
    return out


SKY = {"CLR": 0.0, "SKC": 0.0, "NCD": 0.0, "NSC": 0.0, "FEW": 0.19, "SCT": 0.44, "BKN": 0.75, "OVC": 1.0, "VV": 1.0}


def summarize_conditions(rows: list[list]) -> dict[str, dict]:
    """Per local date: mean dew point, afternoon temperature-minus-dew-point, wind, wind direction, cloud cover, lowest visibility."""
    days: dict[str, list[list]] = {}
    for r in rows:
        days.setdefault(r[0][:10], []).append(r)
    out = {}
    for day, rs in days.items():
        if len(rs) < 8:
            continue
        num = lambda i: [float(r[i]) for r in rs if r[i] not in (None, "", "M")]
        dwp, wind, vis = num(2), [float(r[4]) for r in rs if r[4] not in (None, "", "M")], num(6)
        aft = [float(r[1]) - float(r[2]) for r in rs if r[1] not in (None, "", "M") and r[2] not in (None, "", "M") and 12 <= int(r[0][11:13]) <= 18]
        wd = [(float(r[3]), float(r[4])) for r in rs if r[3] not in (None, "", "M") and r[4] not in (None, "", "M") and float(r[4]) >= 3]
        sky = [SKY.get(str(r[5]).upper(), 0.0) for r in rs]  # a blank sky field on a report means clear (CAVOK / no significant cloud)
        out[day] = {"dwp": _mean(dwp), "dep_aft": _mean(aft), "wind": _mean(wind),
                    "wsin": _mean([math.sin(math.radians(a)) * w for a, w in wd]) / _mean([w for _, w in wd]) if wd else NAN,
                    "wcos": _mean([math.cos(math.radians(a)) * w for a, w in wd]) / _mean([w for _, w in wd]) if wd else NAN,
                    "cloud": _mean(sky), "vis": min(vis) if vis else NAN}
    return out


def fetch_conditions(client: httpx.Client, station: str, tz: str, start: date, end: date, *, use_cache: bool) -> tuple[dict[str, dict], float]:
    """(daily condition summaries, average reports per day) from the station archive; chunks cached on disk."""
    rows: list[list] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=89), end)

        def go(cs=cur, ce=chunk_end):
            resp = _get(client, IEM_URL, {
                "station": station, "data": ["tmpc", "dwpc", "drct", "sknt", "skyc1", "vsby"], "year1": cs.year, "month1": cs.month, "day1": cs.day,
                "year2": ce.year, "month2": ce.month, "day2": ce.day, "tz": tz, "format": "onlycomma",
                "latlon": "no", "elev": "no", "missing": "empty", "trace": "T", "direct": "no", "report_type": [3, 4]})
            return {"rows": [[r["valid"], r["tmpc"], r["dwpc"], r["drct"], r["sknt"], r["skyc1"], r["vsby"]] for r in csv.DictReader(io.StringIO(resp.text))]}

        rows += _cached(f"obsx_{station}_{cur}_{chunk_end}", use_cache and chunk_end <= date.today() - timedelta(days=3), go)["rows"]
        cur = chunk_end + timedelta(days=1)
    days = {r[0][:10] for r in rows}
    return summarize_conditions(rows), len(rows) / max(len(days), 1)


def hourly_paths(hourly: dict) -> dict[str, dict]:
    """Latest-run hourly forecast by local date: {date: {"path": mean over models by hour, "models": {model: day max}}}."""
    by_day: dict[str, list[int]] = {}
    for i, t in enumerate(hourly["time"]):
        by_day.setdefault(t[:10], []).append(i)
    out = {}
    for day, ii in by_day.items():
        series = {m: hourly.get(f"temperature_2m_{m}") for m in MODELS}
        path = []
        for i in ii:
            vals = [x[i] for x in series.values() if x and x[i] is not None]
            path.append(sum(vals) / len(vals) if vals else None)
        maxes = {m: max(x[i] for i in ii if x[i] is not None) for m, x in series.items() if x and any(x[i] is not None for i in ii)}
        if len(path) == 24 and sum(v is not None for v in path) >= 20 and maxes:
            out[day] = {"path": path, "models": maxes}
    return out


def fetch_aerosol(client: httpx.Client, lat: float, lon: float, start: date, end: date, *, use_cache: bool) -> dict[str, dict]:
    """Daily mean aerosol optical depth, dust and PM10 by local date."""
    out: dict[str, dict] = {}
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=364), end)

        def go(cs=cur, ce=chunk_end):
            return _get(client, AIR_URL, {
                "latitude": lat, "longitude": lon, "hourly": "aerosol_optical_depth,dust,pm10",
                "start_date": cs.isoformat(), "end_date": ce.isoformat(), "timezone": "auto"}).json()["hourly"]

        h = _cached(f"aero_{lat:.3f}_{lon:.3f}_{cur}_{chunk_end}", use_cache and chunk_end <= date.today() - timedelta(days=3), go)
        days: dict[str, dict[str, list[float]]] = {}
        for i, t in enumerate(h["time"]):
            for key, name in (("aerosol_optical_depth", "aod"), ("dust", "dust"), ("pm10", "pm10")):
                v = h[key][i]
                if v is not None:
                    days.setdefault(t[:10], {}).setdefault(name, []).append(v)
        out.update({d: {k: sum(v) / len(v) for k, v in m.items()} for d, m in days.items()})
        cur = chunk_end + timedelta(days=1)
    return out


def fetch_bundle(client: httpx.Client, station: str, lat: float, lon: float, *, start: date = START, end: date | None = None,
                 light: bool = False, weather2: bool = False, more_models: bool = False) -> Bundle:
    """Training-time fetch for one station (cached on disk; windows are fixed so the cache is reused).

    light=True is for helper stations that only supply neighbor features: temperature forecasts and
    station highs, no weather variables and no aerosols, which is several times cheaper to download.
    """
    end = end or (date.today() - timedelta(days=3))
    fc = fetch_forecasts(client, lat, lon, start - timedelta(days=10), end, use_cache=True, variables=TEMP_ONLY if light else ("temperature_2m",) + WEATHER_VARS + (EXTRA_VARS if weather2 else ()))
    obs = fetch_obs(client, station, fc["tz"], start - timedelta(days=10), end, use_cache=True)
    try:
        aero = {} if light else fetch_aerosol(client, lat, lon, start - timedelta(days=10), end, use_cache=True)
    except httpx.HTTPError:
        aero = {}
    obs_ctx, per_day = ({}, NAN) if light else _safe_conditions(client, station, fc["tz"], start - timedelta(days=10), end)
    extra: dict[int, dict] = {}
    if more_models and not light:
        xf = fetch_forecasts(client, lat, lon, start - timedelta(days=10), end, use_cache=True, variables=TEMP_ONLY, models=EXTRA_MODELS)
        extra = {lead: extra_models_daily(xf["hourly"], lead) for lead in LEADS}
    return Bundle(station, lat, lon, fc["elev"], fc["tz"], {lead: hourly_to_daily(fc["hourly"], lead) for lead in LEADS}, obs, aero,
                  normals=None if light else normals_for(lat, lon), obs_ctx=obs_ctx, traits={"st_reports_per_day": per_day}, extra_daily=extra)


def _safe_conditions(client: httpx.Client, station: str, tz: str, start: date, end: date) -> tuple[dict, float]:
    try:
        return fetch_conditions(client, station, tz, start, end, use_cache=True)
    except (httpx.HTTPError, ValueError, KeyError):
        return {}, NAN  # these inputs go missing; the model handles that


def bundle_rows(b: Bundle, neighbors: list[tuple[Bundle, float]], *, start: date = START, end: date | None = None) -> list[dict]:
    """Training rows for one station: features, the target (station high, deg C) and identifiers."""
    end = end or (date.today() - timedelta(days=3))
    rows = []
    for lead in LEADS:
        d = start
        while d <= end:
            day = d.isoformat()
            if day in b.obs:
                feats = make_features(day, lead, b.daily[lead], b.obs, b.aero, b.lat, b.lon, b.elev,
                                      nb=neighbor_features(day, lead, neighbors), normals=b.normals,
                                      older=b.daily.get(lead + 1), ctx={"obs": b.obs_ctx, "traits": b.traits}, xm=b.extra_daily.get(lead))
                if feats is not None:
                    rows.append({"station": b.station, "date": day, "y": b.obs[day], **feats})
            d += timedelta(days=1)
    return rows

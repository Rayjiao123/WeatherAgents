"""Real-time signs of a front, rain or fog arriving, from the station's own reports and its upwind neighbours.

A day's high usually fails when something changes faster than the forecast said: a front arrives early, rain starts,
fog does not burn off. Those show up in the station record hours before the temperature does:

  pressure    falling ahead of a front, rising sharply behind it
  wind        a sudden shift in direction as the front passes
  dew point   rising as moist air arrives, falling behind a cold front
  cloud, visibility, rain, weather codes (RA, TS, FG)
  upwind      stations in the direction the wind is coming from, and whether they are warming or cooling: the air that
              reaches this station a few hours later

Used by the same-day model, and by the post-mortem tool to show what the record said before each big miss.
Reports come from the same station archive as the rest. No price data, no quota.
"""

import csv
import io
import math
from bisect import bisect_right
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from weatheragents.calibration_fit import IEM_URL
from weatheragents.ml_data import SKY, _cached, _get, haversine_km

NAN = math.nan
HPA_PER_INHG = 33.8639
FIELDS = ["tmpc", "dwpc", "drct", "sknt", "skyc1", "vsby", "alti", "p01i", "wxcodes"]
FRONT_FEATURES = [
    "ft_dp3", "ft_dp1", "ft_wshift", "ft_wind", "ft_wind_chg", "ft_dwp_chg3", "ft_dep_now",
    "ft_vis", "ft_vis_chg3", "ft_cloud", "ft_rain3", "ft_wx_rain", "ft_wx_fog", "ft_temp_chg1",
]
UP_FEATURES = ["up_grad", "up_trend3", "up_dist", "up_n"]


def _num(x) -> float | None:
    try:
        return None if x in (None, "", "M") else float(x)
    except ValueError:
        return None


def parse_rt(rows: list[dict]) -> dict[str, list[dict]]:
    """CSV rows -> {local date: reports in time order}, numbers parsed, blank cloud field meaning clear."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        t = _num(r.get("tmpc"))
        if t is None:
            continue
        out.setdefault(r["valid"][:10], []).append({
            "h": int(r["valid"][11:13]) + int(r["valid"][14:16]) / 60, "tmpc": t, "dwpc": _num(r.get("dwpc")), "drct": _num(r.get("drct")),
            "sknt": _num(r.get("sknt")), "sky": SKY.get((r.get("skyc1") or "").upper(), 0.0), "vsby": _num(r.get("vsby")),
            "alti": _num(r.get("alti")), "p01i": _num(r.get("p01i")), "wx": (r.get("wxcodes") or "").upper()})
    for v in out.values():
        v.sort(key=lambda r: r["h"])
    return out


def fetch_rt(client: httpx.Client, station: str, tz: str, start: date, end: date, *, use_cache: bool) -> dict[str, list[dict]]:
    """Every report with pressure, wind, dew point, cloud, visibility, rain and weather codes, by local date."""
    rows: list[dict] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=89), end)

        def go(cs=cur, ce=chunk_end):
            resp = _get(client, IEM_URL, {
                "station": station, "data": FIELDS, "year1": cs.year, "month1": cs.month, "day1": cs.day,
                "year2": ce.year, "month2": ce.month, "day2": ce.day, "tz": tz, "format": "onlycomma",
                "latlon": "no", "elev": "no", "missing": "empty", "trace": "T", "direct": "no", "report_type": [3, 4]})
            return {"rows": list(csv.DictReader(io.StringIO(resp.text)))}

        rows += _cached(f"rt_{station}_{cur}_{chunk_end}", use_cache and chunk_end <= date.today() - timedelta(days=3), go)["rows"]
        cur = chunk_end + timedelta(days=1)
    return parse_rt(rows)


# ---------------------------------------------------------------- the station's own signs


def _near(rows: list[dict], h: float, tol: float) -> dict | None:
    best = min(rows, key=lambda r: abs(r["h"] - h), default=None)
    return best if best is not None and abs(best["h"] - h) <= tol else None


def _wind_dir(rows: list[dict]) -> float | None:
    """Circular mean of the direction the wind is coming from, over reports with a real wind (3 knots or more)."""
    pts = [(r["drct"], r["sknt"]) for r in rows if r["drct"] is not None and r["sknt"] is not None and r["sknt"] >= 3]
    if not pts:
        return None
    s = sum(math.sin(math.radians(d)) * w for d, w in pts)
    c = sum(math.cos(math.radians(d)) * w for d, w in pts)
    return math.degrees(math.atan2(s, c)) % 360


def _diff(a: float | None, b: float | None, scale: float = 1.0) -> float:
    return (a - b) * scale if a is not None and b is not None else NAN


def own_features(rows: list[dict], cutoff: float) -> dict:
    """Signs of change at the station, using only reports up to `cutoff` (local decimal hour)."""
    cur = [r for r in rows if r["h"] <= cutoff]
    if len(cur) < 4:
        return {f: NAN for f in FRONT_FEATURES}
    now = cur[-1]
    hour = [r for r in cur if r["h"] >= now["h"] - 1.0]
    r3, r1 = _near(cur, now["h"] - 3.0, 0.75), _near(cur, now["h"] - 1.0, 0.5)
    around3 = [r for r in cur if abs(r["h"] - (now["h"] - 3.0)) <= 0.5]
    d_now, d_3 = _wind_dir(hour), _wind_dir(around3)
    shift = abs((d_now - d_3 + 180) % 360 - 180) if d_now is not None and d_3 is not None else NAN
    winds_now = [r["sknt"] for r in hour if r["sknt"] is not None]
    winds_3 = [r["sknt"] for r in around3 if r["sknt"] is not None]
    last3 = [r for r in cur if r["h"] >= now["h"] - 3.0]
    rain = [r["p01i"] for r in last3 if r["p01i"] is not None]
    codes = " ".join(r["wx"] for r in last3)
    return {
        "ft_dp3": _diff(now["alti"], r3["alti"] if r3 else None, HPA_PER_INHG),
        "ft_dp1": _diff(now["alti"], r1["alti"] if r1 else None, HPA_PER_INHG),
        "ft_wshift": shift,
        "ft_wind": sum(winds_now) / len(winds_now) if winds_now else NAN,
        "ft_wind_chg": (sum(winds_now) / len(winds_now) - sum(winds_3) / len(winds_3)) if winds_now and winds_3 else NAN,
        "ft_dwp_chg3": _diff(now["dwpc"], r3["dwpc"] if r3 else None),
        "ft_dep_now": _diff(now["tmpc"], now["dwpc"]),
        "ft_vis": now["vsby"] if now["vsby"] is not None else NAN,
        "ft_vis_chg3": _diff(now["vsby"], r3["vsby"] if r3 else None),
        "ft_cloud": now["sky"],
        "ft_rain3": max(rain) * 25.4 if rain else NAN,
        "ft_wx_rain": float(any(c in codes for c in ("RA", "DZ", "SH", "TS", "SN"))),
        "ft_wx_fog": float(any(c in codes for c in ("FG", "BR", "HZ"))),
        "ft_temp_chg1": _diff(now["tmpc"], r1["tmpc"] if r1 else None),
    }


# ---------------------------------------------------------------- upwind neighbours


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def utc_series(reports_by_day: dict[str, list[tuple[float, float]]], tz: str) -> tuple[list[float], list[float]]:
    """A station's (local hour, temperature) reports as parallel lists of UTC seconds and temperatures, time ordered."""
    z = ZoneInfo(tz)
    secs, temps = [], []
    for day in sorted(reports_by_day):
        base = datetime.fromisoformat(day).replace(tzinfo=z).timestamp()
        for h, t in reports_by_day[day]:
            secs.append(base + h * 3600)
            temps.append(t)
    return secs, temps


def _temp_at(series: tuple[list[float], list[float]], when: float, tol: float = 5400) -> float | None:
    """The latest report at or before `when`, if it is within `tol` seconds (default 90 minutes)."""
    secs, temps = series
    i = bisect_right(secs, when) - 1
    return temps[i] if i >= 0 and when - secs[i] <= tol else None


def upstream_features(lat: float, lon: float, wind_from: float | None, now_s: float, t_now: float | None,
                      neighbours: list[tuple[float, float, float, tuple[list[float], list[float]]]]) -> dict:
    """What the stations upwind are doing: neighbours = [(lat, lon, distance_km, utc_series)].

    Each is weighted by how squarely it lies in the direction the wind comes from (zero if it is downwind).
    up_grad   their current temperature minus this station's: warm air upwind is on its way
    up_trend3 how much they warmed or cooled over the last 3 hours: a front passing there reaches here later
    """
    empty = {"up_grad": NAN, "up_trend3": NAN, "up_dist": NAN, "up_n": 0.0}
    if wind_from is None or t_now is None:
        return empty
    ws, grads, trends, dists = [], [], [], []
    for nlat, nlon, dist, series in neighbours:
        w = math.cos(math.radians(bearing(lat, lon, nlat, nlon) - wind_from))
        if w <= 0.3:
            continue
        a, b = _temp_at(series, now_s), _temp_at(series, now_s - 3 * 3600)
        if a is None or b is None:
            continue
        ws.append(w)
        grads.append(a - t_now)
        trends.append(a - b)
        dists.append(dist)
    if not ws:
        return empty
    tot = sum(ws)
    return {"up_grad": sum(w * g for w, g in zip(ws, grads)) / tot, "up_trend3": sum(w * t for w, t in zip(ws, trends)) / tot,
            "up_dist": sum(w * d for w, d in zip(ws, dists)) / tot, "up_n": float(len(ws))}

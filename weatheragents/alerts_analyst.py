"""Alerts Analyst: official warnings that can push a day's high away from the models.

Flags, for the contract's station:
  cyclone_within_800km   an active tropical cyclone (GDACS) close enough to change clouds, rain and wind
  nws_heat / nws_cold / nws_fog / nws_rain / nws_wind   US National Weather Service alerts at the station

There is no history of these to train on, so this does not feed a model yet. Every event's flags are
logged, and the evaluator can later check whether the forecasts do worse on flagged days, and by how
much. Reads semantics only. No price, no credentials.
"""

import math
import re
import time

import httpx

from weatheragents.ensemble_analyst import resolve_coordinates
from weatheragents.ml_data import haversine_km
from weatheragents.schemas import AlertsReport, Contract
from weatheragents.timeutil import now as ict_now

GDACS_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/EVENTS4APP?eventlist=TC"
NWS_URL = "https://api.weather.gov/alerts/active"
CYCLONE_RADIUS_KM = 800.0
NWS_FLAGS = {
    "nws_heat": re.compile(r"heat", re.I),
    "nws_cold": re.compile(r"freeze|frost|cold|winter|snow|ice|blizzard|chill", re.I),
    "nws_fog": re.compile(r"fog|smoke|dust|haze", re.I),
    "nws_rain": re.compile(r"flood|rain|storm|thunder|tornado|hurricane|tropical", re.I),
    "nws_wind": re.compile(r"wind|gale", re.I),
}
_CYCLONES: dict = {"at": 0.0, "data": []}


def active_cyclones(client: httpx.Client, *, max_age_s: float = 600) -> list[dict]:
    """Current tropical cyclones worldwide, fetched at most once per 10 minutes."""
    if time.time() - _CYCLONES["at"] > max_age_s:
        resp = client.get(GDACS_URL)
        resp.raise_for_status()
        out = []
        for f in resp.json().get("features", []):
            p = f.get("properties", {})
            if p.get("eventtype") != "TC" or not p.get("iscurrent") or f.get("geometry", {}).get("type") != "Point":
                continue
            lon, lat = f["geometry"]["coordinates"][:2]
            sev = p.get("severitydata") or {}
            out.append({"name": p.get("name", ""), "lat": lat, "lon": lon, "alert": p.get("alertlevel"), "wind_kmh": sev.get("severity")})
        _CYCLONES.update(at=time.time(), data=out)
    return _CYCLONES["data"]


def nws_alerts(client: httpx.Client, lat: float, lon: float) -> list[str]:
    resp = client.get(NWS_URL, params={"point": f"{lat:.4f},{lon:.4f}"}, headers={"User-Agent": "weatheragents/0.1 (research)"})
    resp.raise_for_status()
    return [f["properties"].get("event", "") for f in resp.json().get("features", [])]


def flags_from(cyclone_km: float | None, nws: list[str]) -> list[str]:
    flags = []
    if cyclone_km is not None and cyclone_km <= CYCLONE_RADIUS_KM:
        flags.append("cyclone_within_800km")
    for name, rx in NWS_FLAGS.items():
        if any(rx.search(e) for e in nws):
            flags.append(name)
    return flags


def analyze_contract(contract: Contract, client: httpx.Client) -> AlertsReport:
    s = contract.semantics_view()
    lat, lon = resolve_coordinates(s.station_code or "")
    nearest = min(((haversine_km(lat, lon, c["lat"], c["lon"]), c) for c in active_cyclones(client)), key=lambda t: t[0], default=None)
    km = None if nearest is None else round(nearest[0])
    nws: list[str] = []
    if s.unit == "F":  # the NWS only covers the US, and Fahrenheit contracts are the US ones
        try:
            nws = nws_alerts(client, lat, lon)
        except httpx.HTTPError:
            nws = []
    return AlertsReport(
        event_id=s.event_id,
        station_code=s.station_code or "",
        nearest_cyclone_km=km,
        nearest_cyclone=nearest[1]["name"] if nearest and km is not None and km <= 3000 else None,
        cyclone_alert=nearest[1]["alert"] if nearest and km is not None and km <= 3000 else None,
        cyclone_wind_kmh=nearest[1]["wind_kmh"] if nearest and km is not None and km <= 3000 else None,
        nws_alerts=nws,
        flags=flags_from(km, nws),
        generated_at=ict_now(),
    )

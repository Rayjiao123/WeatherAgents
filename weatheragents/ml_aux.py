"""Picks helper weather stations near each tracked city, to feed the ML Analyst's neighbor features.

    python -m weatheragents.ml_aux

The tracked cities are far apart, so only 39% of training rows had any neighbor. This finds up to
three nearby airports per tracked station (60-450 km away, not in the same town) that really publish
hourly weather reports, and writes them to data/ml/aux_stations.json. They are never traded; their only
job is to show how wrong the forecasts have been around a city, which ml_train.py then tests.
"""

import argparse
import csv
import io
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx

from weatheragents.calibration_fit import IEM_URL
from weatheragents.ensemble_analyst import AIRPORTS_CSV
from weatheragents.ml_data import haversine_km
from weatheragents.ml_train import ML_DIR

AUX_PATH = ML_DIR / "aux_stations.json"
MIN_KM, MAX_KM, PER_STATION = 60.0, 450.0, 3
SAME_TOWN_KM = 40.0  # two helpers closer than this are the same weather
PROBE_DAYS, PROBE_MIN_READINGS = 10, 150
SKIP_NAME = re.compile(r"heliport|seaplane|strip|ranch|private|glider|ultralight|balloon|farm|helipad|stolport|hospital", re.I)


def load_airports(path: Path = AIRPORTS_CSV) -> list[tuple[str, str, float, float]]:
    """(icao, name, lat, lon) for plausible weather-reporting airports: 4-letter codes, no heliports and the like."""
    out = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            icao = row["icao"]
            if len(icao) == 4 and icao.isalpha() and not SKIP_NAME.search(row["name"]):
                out.append((icao, row["name"], float(row["latitude_deg"]), float(row["longitude_deg"])))
    return out


def candidates(lat: float, lon: float, airports: list[tuple[str, str, float, float]], *, exclude: set[str]) -> list[tuple[float, str]]:
    """Airports 60-450 km away, nearest first."""
    near = []
    for icao, _, la, lo in airports:
        if icao in exclude:
            continue
        d = haversine_km(lat, lon, la, lo)
        if MIN_KM <= d <= MAX_KM:
            near.append((d, icao))
    return sorted(near)


def has_recent_readings(client: httpx.Client, icao: str) -> bool:
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=PROBE_DAYS)
    try:
        resp = client.get(IEM_URL, params={
            "station": icao, "data": "tmpc", "year1": start.year, "month1": start.month, "day1": start.day,
            "year2": end.year, "month2": end.month, "day2": end.day, "tz": "UTC", "format": "onlycomma",
            "latlon": "no", "elev": "no", "missing": "empty", "trace": "T", "direct": "no", "report_type": [3, 4]})
        resp.raise_for_status()
    except httpx.HTTPError:
        return False
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r.get("tmpc") not in (None, "", "M")]
    return len(rows) >= PROBE_MIN_READINGS


def pick_aux(tracked: dict[str, tuple[float, float]], client: httpx.Client, airports=None, probe=has_recent_readings) -> dict[str, tuple[float, float]]:
    airports = airports if airports is not None else load_airports()
    coords = {icao: (la, lo) for icao, _, la, lo in airports}
    chosen: dict[str, tuple[float, float]] = {}
    verdict: dict[str, bool] = {}
    for code, (lat, lon) in sorted(tracked.items()):
        have = [c for c in chosen if haversine_km(lat, lon, *chosen[c]) <= MAX_KM]  # helpers already near this city count
        picked = len(have)
        for _, icao in candidates(lat, lon, airports, exclude=set(tracked)):
            if picked >= PER_STATION:
                break
            if icao in chosen:
                continue
            la, lo = coords[icao]
            if any(haversine_km(la, lo, *chosen[c]) < SAME_TOWN_KM for c in chosen) or any(
                haversine_km(la, lo, *tracked[t]) < SAME_TOWN_KM for t in tracked
            ):
                continue
            if icao not in verdict:
                verdict[icao] = probe(client, icao)
            if verdict[icao]:
                chosen[icao] = (la, lo)
                picked += 1
    return chosen


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Pick helper stations near the tracked cities").parse_args(argv)
    stations = ML_DIR / "stations.json"
    if not stations.exists():
        print("run `python -m weatheragents.ml_train` first (it writes data/ml/stations.json)", file=sys.stderr)
        return 1
    tracked = {c: tuple(v) for c, v in json.loads(stations.read_text(encoding="utf-8")).items() if c not in json.loads(AUX_PATH.read_text(encoding="utf-8"))} if AUX_PATH.exists() else {c: tuple(v) for c, v in json.loads(stations.read_text(encoding="utf-8")).items()}
    with httpx.Client(timeout=60, headers={"User-Agent": "weatheragents/0.1"}) as client:
        aux = pick_aux(tracked, client)
    AUX_PATH.write_text(json.dumps({c: list(v) for c, v in aux.items()}, indent=2), encoding="utf-8")
    print(f"{len(aux)} helper stations for {len(tracked)} tracked cities -> {AUX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

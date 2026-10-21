"""Fits forecast error per station from history, so spreads come from data, not a guess.

    python -m weatheragents.calibration_fit            # stations in the latest contract list
    python -m weatheragents.calibration_fit --days 150

For each (station, highest|lowest) it compares the forecast Open-Meteo had at
the time (same-day run and the run issued a day earlier) against what the
station actually recorded, using Iowa State's METAR archive for the
observations. The result, data/calibration.json, holds per-station bias and
error spread in degrees C; baseline.py reads it.

Stations the observation archive doesn't cover (e.g. Hong Kong's HKO) fall
back to the pooled default.
"""

import argparse
import csv
import io
import json
import statistics
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from weatheragents.ensemble_analyst import _latest_contract_list, resolve_coordinates
from weatheragents.timeutil import now as ict_now, stamp as ict_stamp, to_ict
from weatheragents.schemas import Contract

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
CALIBRATION_PATH = Path("data") / "calibration.json"
MIN_HOURS = 20  # forecast hours needed for a trustworthy daily extreme
MIN_OBS = 8  # station readings needed for a trustworthy daily extreme
OUTLIER_C = 8.0
BIAS_SHRINK = 15  # bias is pulled toward 0 by n / (n + 15): thin samples shouldn't move the center
SD_FLOOR_C = 0.6


def daily_extremes(times: list[str], values: list[float | None], kind: str, min_count: int) -> dict[str, float]:
    """Group readings by their local date string (first 10 chars) and take max or min."""
    by_day: dict[str, list[float]] = {}
    for t, v in zip(times, values):
        if v is not None:
            by_day.setdefault(t[:10], []).append(float(v))
    pick = max if kind == "highest" else min
    return {d: pick(vs) for d, vs in by_day.items() if len(vs) >= min_count}


def fetch_forecast_history(client: httpx.Client, lat: float, lon: float, start: date, end: date) -> tuple[dict, str]:
    resp = client.get(
        PREVIOUS_RUNS_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,temperature_2m_previous_day1",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "auto",
        },
    )
    resp.raise_for_status()
    p = resp.json()
    return p["hourly"], p["timezone"]


def fetch_station_obs(client: httpx.Client, station: str, tz: str, start: date, end: date) -> tuple[list[str], list[float | None]]:
    resp = client.get(
        IEM_URL,
        params={
            "station": station,
            "data": "tmpc",
            "year1": start.year, "month1": start.month, "day1": start.day,
            "year2": end.year, "month2": end.month, "day2": end.day,
            "tz": tz,
            "format": "onlycomma",
            "latlon": "no", "elev": "no", "missing": "empty", "trace": "T", "direct": "no",
            "report_type": [3, 4],
        },
    )
    resp.raise_for_status()
    times, vals = [], []
    for row in csv.DictReader(io.StringIO(resp.text)):
        times.append(row["valid"])
        vals.append(float(row["tmpc"]) if row.get("tmpc") not in (None, "", "M") else None)
    return times, vals


def error_stats(errors: list[float]) -> dict:
    errors = [e for e in errors if abs(e) <= OUTLIER_C]
    n = len(errors)
    if n < 20:
        raise ValueError(f"only {n} usable days")
    raw_bias = statistics.fmean(errors)
    sd = statistics.pstdev(errors)  # spread around the bias, since the bias gets corrected
    return {"n": n, "bias_c": round(raw_bias * n / (n + BIAS_SHRINK), 3), "sd_c": round(max(sd, SD_FLOOR_C), 3)}


def fit_station(client: httpx.Client, station: str, kind: str, *, days: int) -> dict:
    lat, lon = resolve_coordinates(station)
    end = datetime.now(timezone.utc).date() - timedelta(days=2)
    start = end - timedelta(days=days)
    hourly, tz = fetch_forecast_history(client, lat, lon, start, end)
    times, obs = fetch_station_obs(client, station, tz, start, end)
    observed = daily_extremes(times, obs, kind, MIN_OBS)
    out = {}
    for lead, key in ((0, "temperature_2m"), (1, "temperature_2m_previous_day1")):
        fc = daily_extremes(hourly["time"], hourly[key], kind, MIN_HOURS)
        errs = [observed[d] - fc[d] for d in fc if d in observed]
        out[f"day{lead}"] = error_stats(errs)
    return out


def pooled_default(stations: dict[str, dict]) -> dict:
    out = {}
    for lead in ("day0", "day1"):
        rows = [v[lead] for v in stations.values()]
        n = sum(r["n"] for r in rows)
        out[lead] = {
            "n": n,
            "bias_c": 0.0,  # a pooled bias would be wrong for any particular station
            "sd_c": round(statistics.fmean(r["sd_c"] for r in rows), 3),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit per-station forecast error from history")
    parser.add_argument("--contracts", type=Path, default=None)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--out", type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args(argv)

    path = args.contracts or _latest_contract_list(Path("data"))
    if path is None or not path.exists():
        print("no contract_list file found; run `python -m weatheragents.scout` first", file=sys.stderr)
        return 1
    contracts = [Contract.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    pairs = sorted({(c.semantics.station_code, c.semantics.kind) for c in contracts if c.semantics.station_code})

    stations, skipped = {}, []
    with httpx.Client(timeout=90, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for code, kind in pairs:
            try:
                stations[f"{code}|{kind}"] = fit_station(client, code, kind, days=args.days)
            except (ValueError, LookupError, KeyError, httpx.HTTPError) as exc:
                skipped.append(f"{code} {kind}: {type(exc).__name__}: {exc}")
            time.sleep(0.5)  # be polite to both free APIs
    if not stations:
        print("fitted nothing", file=sys.stderr)
        return 1

    result = {
        "fitted_at": ict_now().isoformat(),
        "days": args.days,
        "stations": stations,
        "default": pooled_default(stations),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"fitted {len(stations)} of {len(pairs)} station/kind pairs; {len(skipped)} fell back to the default")
    for line in skipped:
        print(f"  ! {line}", file=sys.stderr)
    d = result["default"]
    print(f"pooled default: day0 sd {d['day0']['sd_c']}C, day1 sd {d['day1']['sd_c']}C")
    print(f"\n{'station':<16}{'n':>4}{'day0 bias':>11}{'day0 sd':>9}{'day1 bias':>11}{'day1 sd':>9}")
    for key, v in sorted(stations.items()):
        print(f"{key:<16}{v['day1']['n']:>4}{v['day0']['bias_c']:>+11.2f}{v['day0']['sd_c']:>9.2f}{v['day1']['bias_c']:>+11.2f}{v['day1']['sd_c']:>9.2f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

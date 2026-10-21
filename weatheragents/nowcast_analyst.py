"""Agent 03 - Nowcast Analyst.

Checks whether the latest forecast run and the station's own recent
observations are pulling toward warmer or cooler brackets than the ensemble
centers on. Reads a contract's semantics only. No LLM, no credentials.

    python -m weatheragents.nowcast_analyst

Sources: Open-Meteo previous-runs API (the same forecast as issued 1-3 days
ago, for run-to-run drift) and aviationweather.gov METARs (what the station
has actually recorded so far on the observation date).
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from weatheragents.ensemble_analyst import _latest_contract_list, resolve_coordinates
from weatheragents.timeutil import now as ict_now, stamp as ict_stamp, to_ict
from weatheragents.schemas import Contract, ContractSemantics, NowcastReport

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
METAR_URL = "https://aviationweather.gov/api/data/metar"
RUN_KEYS = {
    "latest": "temperature_2m",
    "1d": "temperature_2m_previous_day1",
    "2d": "temperature_2m_previous_day2",
    "3d": "temperature_2m_previous_day3",
}
STEADY_BAND = 0.5  # degrees C (or 0.9 F) of drift still counted as "steady"


def c_to_unit(value_c: float, unit: str) -> float:
    return value_c * 9 / 5 + 32 if unit == "F" else value_c


def extreme(values: list[float], kind: str) -> float:
    return max(values) if kind == "highest" else min(values)


def fetch_runs(client: httpx.Client, *, lat: float, lon: float, obs_date: date, unit: str) -> tuple[dict, int]:
    """Returns ({run: hourly values}, utc_offset_seconds) for the observation date."""
    resp = client.get(
        PREVIOUS_RUNS_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(RUN_KEYS.values()),
            "start_date": obs_date.isoformat(),
            "end_date": obs_date.isoformat(),
            "temperature_unit": "fahrenheit" if unit == "F" else "celsius",
            "timezone": "auto",
        },
    )
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload["hourly"]
    runs = {}
    for name, key in RUN_KEYS.items():
        vals = [v for v in hourly.get(key, []) if v is not None]
        if vals:
            runs[name] = vals
    return runs, payload.get("utc_offset_seconds", 0)


def fetch_metars(client: httpx.Client, station_code: str) -> list[dict]:
    if len(station_code) != 4 or not station_code.isalpha():
        return []  # e.g. HKO has no METAR feed
    resp = client.get(METAR_URL, params={"ids": station_code, "format": "json", "hours": 48})
    resp.raise_for_status()
    return resp.json() if resp.text.strip() else []


def observed_so_far(
    metars: list[dict], *, obs_date: date, utc_offset_s: int, kind: str, unit: str
) -> tuple[float | None, str | None, float | None, int]:
    """Extreme of station temps recorded on obs_date (station-local date)."""
    local = []
    for m in metars:
        if m.get("temp") is None or m.get("obsTime") is None:
            continue
        t = datetime.fromtimestamp(m["obsTime"], tz=timezone.utc) + timedelta(seconds=utc_offset_s)
        if t.date() == obs_date:
            local.append((t, float(m["temp"])))
    if not local:
        return None, None, None, 0
    local.sort()
    temps = [c_to_unit(t, unit) for _, t in local]
    return extreme(temps, kind), local[-1][0].strftime("%Y-%m-%d %H:%M"), temps[-1], len(local)


def classify_trend(drift: float, unit: str) -> str:
    band = STEADY_BAND * 9 / 5 if unit == "F" else STEADY_BAND
    if drift > band:
        return "warming"
    if drift < -band:
        return "cooling"
    return "steady"


def build_report(
    semantics: ContractSemantics,
    runs: dict[str, list[float]],
    utc_offset_s: int,
    metars: list[dict],
    lead_days: int = 1,
) -> NowcastReport:
    kind, unit = semantics.kind, semantics.unit
    if "latest" not in runs:
        raise ValueError("no current forecast values returned")
    extremes = {name: extreme(vals, kind) for name, vals in runs.items()}
    prev = {k: v for k, v in extremes.items() if k != "latest"}
    drift = extremes["latest"] - prev["3d"] if "3d" in prev else 0.0

    obs_ext, obs_through, latest_temp, n = observed_so_far(
        metars, obs_date=semantics.observation_date, utc_offset_s=utc_offset_s, kind=kind, unit=unit
    )
    caveats = ["forecast extreme is the hourly 2 m model value, not the station's rounded METAR reading"]
    if not metars:
        caveats.append("no METAR feed for this station")
    elif obs_ext is None:
        caveats.append("no observations yet on the observation date (still in the future)")
    elif kind == "highest":
        caveats.append("observed high so far can still rise; it is a floor for the day's final high")
    else:
        caveats.append("observed low so far can still fall; it is a ceiling for the day's final low")
    if "3d" not in prev:
        caveats.append("3-day-old run unavailable; drift set to 0")

    return NowcastReport(
        event_id=semantics.event_id,
        station_code=semantics.station_code or "",
        observation_date=semantics.observation_date,
        unit=unit,
        forecast_extreme=round(extremes["latest"], 2),
        previous_run_extremes={k: round(v, 2) for k, v in prev.items()},
        drift_vs_3d=round(drift, 2),
        trend=classify_trend(drift, unit),
        observed_extreme_so_far=None if obs_ext is None else round(obs_ext, 1),
        observed_through_local=obs_through,
        latest_observed_temp=None if latest_temp is None else round(latest_temp, 1),
        metar_reports_used=n,
        lead_days=lead_days,
        caveats=caveats,
        generated_at=ict_now(),
    )


def analyze_contract(contract: Contract, client: httpx.Client) -> NowcastReport:
    """Agent 03's entry point: Contract in, NowcastReport out. Price-blind."""
    s = contract.semantics_view()
    if not s.station_code:
        raise LookupError(f"contract {s.event_id} has no station_code")
    lat, lon = resolve_coordinates(s.station_code)
    runs, offset = fetch_runs(client, lat=lat, lon=lon, obs_date=s.observation_date, unit=s.unit)
    try:
        metars = fetch_metars(client, s.station_code)
    except httpx.HTTPError:
        metars = []
    local_today = (datetime.now(timezone.utc) + timedelta(seconds=offset)).date()
    return build_report(s, runs, offset, metars, lead_days=(s.observation_date - local_today).days)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nowcast Analyst: run drift and station observations")
    parser.add_argument("--contracts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    path = args.contracts or _latest_contract_list(Path("data"))
    if path is None or not path.exists():
        print("no contract_list file found; run `python -m weatheragents.scout` first", file=sys.stderr)
        return 1
    contracts = [Contract.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]

    reports, skipped = [], []
    with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for c in contracts:
            s = c.semantics
            try:
                reports.append(analyze_contract(c, client))
            except (LookupError, ValueError, KeyError, httpx.HTTPError) as exc:
                skipped.append(f"{s.city} {s.station_code} {s.observation_date}: {exc}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = ict_stamp()
    out_file = args.out / f"nowcast_reports_{stamp}.json"
    out_file.write_text(json.dumps([r.model_dump(mode="json") for r in reports], indent=2), encoding="utf-8")

    print(f"{len(contracts)} contracts -> {len(reports)} reports, {len(skipped)} skipped")
    for reason in skipped:
        print(f"  ! {reason}", file=sys.stderr)
    print(f"\n{'event':<10}{'station':<9}{'forecast':>9}{'3d ago':>8}{'drift':>7}  {'trend':<9}{'observed so far'}")
    for r in reports:
        obs = "-" if r.observed_extreme_so_far is None else f"{r.observed_extreme_so_far} (n={r.metar_reports_used})"
        print(
            f"{r.event_id:<10}{r.station_code:<9}{r.forecast_extreme:>9.1f}"
            f"{r.previous_run_extremes.get('3d', float('nan')):>8.1f}{r.drift_vs_3d:>+7.1f}  {r.trend:<9}{obs}"
        )
    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

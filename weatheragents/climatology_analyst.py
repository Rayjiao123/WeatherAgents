"""Agent 04 - Climatology Analyst.

Supplies the base rate for the contract's station and calendar window: 30
years of ERA5 daily extremes within +/- 7 days of the observation date,
turned into a probability for each bracket. Reads semantics only. No LLM, no
credentials.

    python -m weatheragents.climatology_analyst

National-agency archives (the artifact's optional second layer) are not wired
in yet; this is the Open-Meteo-only baseline.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

import httpx

from weatheragents.ensemble_analyst import (
    _latest_contract_list,
    bracket_probabilities,
    daily_variable,
    resolve_coordinates,
)
from weatheragents.timeutil import now as ict_now, stamp as ict_stamp, to_ict
from weatheragents.schemas import ClimatologyReport, Contract, ContractSemantics

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
YEARS = 30
WINDOW_DAYS = 7


def window_values(daily: dict, *, variable: str, target: date, window: int) -> list[tuple[int, float]]:
    """(year, value) pairs within +/- window days of target's month/day, any year."""
    out = []
    for t, v in zip(daily["time"], daily[variable]):
        if v is None:
            continue
        d = date.fromisoformat(t)
        try:
            anchor = date(d.year, target.month, target.day)
        except ValueError:  # Feb 29 target in a non-leap year
            anchor = date(d.year, target.month, 28)
        if abs((d - anchor).days) <= window:
            out.append((d.year, float(v)))
    return out


def percentile(sorted_vals: list[float], q: float) -> float:
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


CACHE_DIR = Path("data") / "cache"


def _to_unit(daily: dict, variable: str, unit: str) -> dict:
    if unit != "F":
        return daily
    out = dict(daily)
    out[variable] = [None if v is None else v * 9 / 5 + 32 for v in daily[variable]]
    return out


@lru_cache(maxsize=256)
def _fetch_archive_c(lat: float, lon: float, variable: str, last_year: int) -> dict:
    """30 years of daily values in Celsius, cached on disk (history doesn't change)."""
    cache = CACHE_DIR / f"era5_{lat:.3f}_{lon:.3f}_{variable}_{last_year}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    with httpx.Client(timeout=90, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for attempt in range(6):
            resp = client.get(
                ARCHIVE_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": variable,
                    "start_date": f"{last_year - YEARS + 1}-01-01",
                    "end_date": f"{last_year}-12-31",
                    "timezone": "auto",
                },
            )
            if resp.status_code == 429 and attempt < 5:
                time.sleep(15 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
    daily = resp.json()["daily"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(daily), encoding="utf-8")
    return daily


def build_report(semantics: ContractSemantics, daily: dict) -> ClimatologyReport:
    variable = daily_variable(semantics.kind)
    pairs = window_values(daily, variable=variable, target=semantics.observation_date, window=WINDOW_DAYS)
    if len(pairs) < 100:
        raise ValueError(f"only {len(pairs)} historical samples")
    values = sorted(v for _, v in pairs)
    last_year = max(y for y, _ in pairs)
    recent = [v for y, v in pairs if y > last_year - 10]
    return ClimatologyReport(
        event_id=semantics.event_id,
        station_code=semantics.station_code or "",
        observation_date=semantics.observation_date,
        kind=semantics.kind,
        unit=semantics.unit,
        years_used=len({y for y, _ in pairs}),
        window_days=WINDOW_DAYS,
        sample_size=len(values),
        mean=round(statistics.fmean(values), 2),
        std_dev=round(statistics.pstdev(values), 2),
        p10=round(percentile(values, 0.10), 1),
        p50=round(percentile(values, 0.50), 1),
        p90=round(percentile(values, 0.90), 1),
        recent_10yr_mean=round(statistics.fmean(recent), 2),
        bracket_probabilities=bracket_probabilities(semantics.brackets, values),
        caveats=[
            "ERA5 is a ~25 km reanalysis grid, not the station; it can run 1-2 degrees off the official reading",
            "base rate only: says nothing about this particular week's weather",
            "no national-agency archive layered in yet",
        ],
        generated_at=ict_now(),
    )


def analyze_contract(contract: Contract) -> ClimatologyReport:
    """Agent 04's entry point: Contract in, ClimatologyReport out. Price-blind."""
    s = contract.semantics_view()
    if not s.station_code:
        raise LookupError(f"contract {s.event_id} has no station_code")
    lat, lon = resolve_coordinates(s.station_code)
    variable = daily_variable(s.kind)
    daily = _fetch_archive_c(lat, lon, variable, datetime.now(timezone.utc).year - 1)
    return build_report(s, _to_unit(daily, variable, s.unit))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Climatology Analyst: 30-year base rates per bracket")
    parser.add_argument("--contracts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    path = args.contracts or _latest_contract_list(Path("data"))
    if path is None or not path.exists():
        print("no contract_list file found; run `python -m weatheragents.scout` first", file=sys.stderr)
        return 1
    contracts = [Contract.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]

    reports, skipped = [], []
    for c in contracts:
        s = c.semantics
        try:
            reports.append(analyze_contract(c))
        except (LookupError, ValueError, KeyError, httpx.HTTPError) as exc:
            skipped.append(f"{s.city} {s.station_code} {s.observation_date}: {exc}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = ict_stamp()
    out_file = args.out / f"climatology_reports_{stamp}.json"
    out_file.write_text(json.dumps([r.model_dump(mode="json") for r in reports], indent=2), encoding="utf-8")

    print(f"{len(contracts)} contracts -> {len(reports)} reports, {len(skipped)} skipped")
    for reason in skipped:
        print(f"  ! {reason}", file=sys.stderr)
    print(f"\n{'event':<10}{'station':<9}{'mean':>7}{'std':>6}{'p10':>6}{'p90':>6}{'recent10':>10}  most likely bracket")
    for r in reports:
        top = max(r.bracket_probabilities, key=lambda bp: bp.probability)
        print(
            f"{r.event_id:<10}{r.station_code:<9}{r.mean:>7.1f}{r.std_dev:>6.1f}{r.p10:>6.1f}"
            f"{r.p90:>6.1f}{r.recent_10yr_mean:>10.1f}  {top.label} ({top.probability:.0%})"
        )
    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

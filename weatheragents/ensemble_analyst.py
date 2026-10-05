"""Agent 02 - Ensemble Analyst.

Reads a contract's semantics (station, observation date, unit, brackets) -
never its price - and turns Open-Meteo's ensemble forecast members for that
station into a probability distribution over the contract's brackets. No
LLM, no credentials: this is a numerical step, like Scout.

    python -m weatheragents.ensemble_analyst --model icon_seamless

Reads the most recent data/contract_list_*.json (or --contracts PATH) that
Market Scout produced, and writes data/ensemble_reports_<timestamp>.json.
"""

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

import httpx

from weatheragents.schemas import (
    Bracket,
    BracketProbability,
    Contract,
    ContractSemantics,
    EnsembleReport,
)

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
AIRPORTS_CSV = Path(__file__).parent / "data" / "airports.csv"
DEFAULT_MODEL = "icon_seamless"

# Station codes Scout emits that aren't ICAO airport codes, per
# ContractSemantics.station_code's docstring. Hong Kong Observatory HQ,
# King's Park, Kowloon.
SPECIAL_STATIONS: dict[str, tuple[float, float]] = {
    "HKO": (22.3019, 114.1741),
}


class ForecastRangeError(ValueError):
    """The requested date isn't covered by this model's forecast window."""


@lru_cache(maxsize=1)
def _airport_coords() -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    with AIRPORTS_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            coords[row["icao"]] = (float(row["latitude_deg"]), float(row["longitude_deg"]))
    return coords


def resolve_coordinates(station_code: str) -> tuple[float, float]:
    code = station_code.upper()
    if code in SPECIAL_STATIONS:
        return SPECIAL_STATIONS[code]
    coords = _airport_coords().get(code)
    if coords is None:
        raise LookupError(f"no coordinates for station {station_code!r}")
    return coords


def daily_variable(kind: str) -> str:
    return "temperature_2m_max" if kind == "highest" else "temperature_2m_min"


def fetch_daily_payload(
    client: httpx.Client,
    *,
    lat: float,
    lon: float,
    obs_date: date,
    variable: str,
    unit: str,
    model: str,
) -> dict:
    """Hit Open-Meteo's ensemble API for one station/day. Returns the raw `daily` dict."""
    resp = client.get(
        ENSEMBLE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": variable,
            "models": model,
            "start_date": obs_date.isoformat(),
            "end_date": obs_date.isoformat(),
            "temperature_unit": "fahrenheit" if unit == "F" else "celsius",
            "timezone": "auto",
        },
    )
    resp.raise_for_status()
    payload = resp.json()
    daily = payload.get("daily")
    if not daily:
        raise ForecastRangeError(f"model {model!r} returned no daily data for {obs_date}")
    return daily


def extract_member_values(daily: dict, *, variable: str, obs_date: date) -> list[float]:
    """Pull every ensemble member's value for `obs_date` out of a `daily` payload."""
    times = daily.get("time", [])
    date_str = obs_date.isoformat()
    try:
        idx = times.index(date_str)
    except ValueError:
        raise ForecastRangeError(
            f"{date_str} not in this model's forecast window (have {times or 'nothing'})"
        )
    prefix = f"{variable}_member"
    values = [
        float(series[idx])
        for key, series in daily.items()
        if key.startswith(prefix) and series[idx] is not None
    ]
    if not values:
        raise ForecastRangeError(f"no ensemble members returned for {date_str}")
    return values


def bracket_probabilities(brackets: list[Bracket], values: list[float]) -> list[BracketProbability]:
    """Empirical P(bracket) = fraction of members landing in it.

    Members are rounded to the nearest whole degree first: brackets are
    integer-bounded (e.g. "21C or below", "22C" ... "25C or higher") because
    that's how the station rounds its official reading, and Scout's brackets
    are built to partition the integers exactly once each.
    """
    rounded = [round(v) for v in values]
    n = len(rounded)
    out = []
    covered = 0
    for b in brackets:
        lo = b.low if b.low is not None else -math.inf
        hi = b.high if b.high is not None else math.inf
        count = sum(1 for v in rounded if lo <= v <= hi)
        covered += count
        out.append(BracketProbability(label=b.label, probability=count / n))
    if covered != n:
        raise ValueError(
            f"brackets don't partition the ensemble spread: {covered}/{n} members covered "
            f"(check Scout's bracket parsing for gaps)"
        )
    return out


def ensemble_report_from_payload(
    semantics: ContractSemantics, daily: dict, *, model: str
) -> EnsembleReport:
    variable = daily_variable(semantics.kind)
    values = extract_member_values(daily, variable=variable, obs_date=semantics.observation_date)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return EnsembleReport(
        event_id=semantics.event_id,
        station_code=semantics.station_code or "",
        observation_date=semantics.observation_date,
        model=model,
        member_count=len(values),
        member_values=values,
        mean=mean,
        std_dev=math.sqrt(variance),
        bracket_probabilities=bracket_probabilities(semantics.brackets, values),
        generated_at=datetime.now(timezone.utc),
    )


def analyze_contract(contract: Contract, client: httpx.Client, *, model: str = DEFAULT_MODEL) -> EnsembleReport:
    """Agent 02's entry point: Contract in, EnsembleReport out. Price-blind."""
    semantics = contract.semantics_view()
    if not semantics.station_code:
        raise LookupError(f"contract {semantics.event_id} has no station_code")
    lat, lon = resolve_coordinates(semantics.station_code)
    daily = fetch_daily_payload(
        client,
        lat=lat,
        lon=lon,
        obs_date=semantics.observation_date,
        variable=daily_variable(semantics.kind),
        unit=semantics.unit,
        model=model,
    )
    return ensemble_report_from_payload(semantics, daily, model=model)


def _latest_contract_list(folder: Path) -> Path | None:
    candidates = sorted(folder.glob("contract_list_*.json"))
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ensemble Analyst: bracket probabilities from Open-Meteo")
    parser.add_argument("--contracts", type=Path, default=None, help="contract_list JSON from Scout (default: latest in data/)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Open-Meteo ensemble model (default icon_seamless)")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output folder (default data/)")
    args = parser.parse_args(argv)

    contracts_path = args.contracts or _latest_contract_list(Path("data"))
    if contracts_path is None or not contracts_path.exists():
        print("no contract_list file found; run `python -m weatheragents.scout` first", file=sys.stderr)
        return 1

    raw = json.loads(contracts_path.read_text(encoding="utf-8"))
    contracts = [Contract.model_validate(d) for d in raw]

    reports, skipped = [], []
    with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for contract in contracts:
            s = contract.semantics
            try:
                reports.append(analyze_contract(contract, client, model=args.model))
            except (LookupError, ForecastRangeError, ValueError, httpx.HTTPError) as exc:
                skipped.append(f"{s.city} {s.station_code} {s.observation_date}: {exc}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = args.out / f"ensemble_reports_{stamp}.json"
    out_file.write_text(
        json.dumps([r.model_dump(mode="json") for r in reports], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"read {len(contracts)} contracts from {contracts_path.name} -> {len(reports)} reports, {len(skipped)} skipped")
    for reason in skipped:
        print(f"  ! {reason}", file=sys.stderr)
    print(f"\n{'event':<10}{'station':<9}{'model':<16}{'members':>8}{'mean':>8}{'std':>7}  top bracket")
    for r in reports:
        top = max(r.bracket_probabilities, key=lambda bp: bp.probability)
        print(
            f"{r.event_id:<10}{r.station_code:<9}{r.model:<16}{r.member_count:>8}"
            f"{r.mean:>8.1f}{r.std_dev:>7.2f}  {top.label} ({top.probability:.0%})"
        )
    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

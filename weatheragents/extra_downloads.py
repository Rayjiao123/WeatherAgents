"""Paced download of the big optional inputs: extra weather variables and twelve extra weather models.

    python -m weatheragents.extra_downloads --stations 12

Each run fetches both downloads for the next N stations that do not have them yet, then stops, so the history
service's daily allowance is shared with the live passes instead of being spent in one go. Safe to repeat:
everything lands in the on-disk cache and finished stations are skipped. When every station is done it says so.
When the daily limit is hit it stops at once and the next run picks up where this one stopped.
"""

import argparse
import sys
from datetime import timedelta

import httpx

from weatheragents import ml_data
from weatheragents.ensemble_analyst import resolve_coordinates
from weatheragents.ml_train import data_end, stations_to_fetch

W2_VARS = ("temperature_2m",) + ml_data.WEATHER_VARS + ml_data.EXTRA_VARS


def missing(code: str, end) -> list[str]:
    lat, lon = resolve_coordinates(code)
    start = ml_data.START - timedelta(days=10)
    need = []
    if not ml_data.forecast_cached(lat, lon, start, end, variables=W2_VARS):
        need.append("weather2")
    if not ml_data.forecast_cached(lat, lon, start, end, variables=ml_data.TEMP_ONLY, models=ml_data.EXTRA_MODELS):
        need.append("more_models")
    return need


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Paced download of the optional large inputs")
    ap.add_argument("--stations", type=int, default=12, help="stations to fetch this run (default 12)")
    args = ap.parse_args(argv)
    end = data_end()
    todo = [(c, m) for c in stations_to_fetch(None) if (m := missing(c, end))]
    if not todo:
        print("all stations complete")
        return 0
    print(f"{len(todo)} stations still need downloads; fetching up to {args.stations} now")
    done = 0
    with httpx.Client(timeout=120, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for code, need in todo[: args.stations]:
            lat, lon = resolve_coordinates(code)
            start = ml_data.START - timedelta(days=10)
            try:
                if "weather2" in need:
                    ml_data.fetch_forecasts(client, lat, lon, start, end, use_cache=True, variables=W2_VARS)
                if "more_models" in need:
                    ml_data.fetch_forecasts(client, lat, lon, start, end, use_cache=True, variables=ml_data.TEMP_ONLY, models=ml_data.EXTRA_MODELS)
                done += 1
                print(f"{code}: done ({', '.join(need)})", flush=True)
            except httpx.HTTPStatusError as exc:
                print(f"{code}: stopped ({exc.response.status_code}: {exc.response.text[:80]})", flush=True)
                break
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                print(f"{code}: skipped ({type(exc).__name__})", flush=True)
    print(f"{done} stations finished this run; {len(todo) - done} to go")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Same-Day Analyst: for events whose day is still in progress, project the high from what has happened.

By afternoon most of a day's high is already on the station's record. Resolution uses every report in the
local calendar day (resolution_check.py shows this rebuilds Polymarket's result in 75 of 75 cases), so the
high so far is a hard floor, and the rest of the day can be projected from the hourly forecast:

  remaining high = the hourly forecast for the hours still to come, shifted by how far the forecast has been
                   missing over the last 3 hours (damped, since a miss does not fully persist)
  centre         = the larger of the high so far and the remaining high
  spread         = narrow once the usual peak (about 16:00 local) has passed, wider the earlier it is
  floor          = the high so far, less half a degree for rounding

Highs only, and only on the event's own day. Everything else returns None. No price, no LLM.
Weather forecast data comes from the plain forecast API, which has its own quota, separate from the
previous-runs API the other analysts use.
"""

import csv
import io
from datetime import datetime, timedelta, timezone

import httpx

from weatheragents.baseline import vector
from weatheragents.calibration_fit import IEM_URL
from weatheragents.ensemble_analyst import resolve_coordinates
from weatheragents.schemas import Contract, SameDayReport
from weatheragents.timeutil import now as ict_now

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PEAK_HOUR = 16.0  # when a day's high is usually reached, local time
BIAS_DAMPING = 0.7


def spread_for(hours_to_peak: float) -> float:
    """Degrees C of uncertainty left in the day's high: small once the peak is behind us."""
    return min(0.4 + 0.22 * max(hours_to_peak, 0.0), 1.8)


def project(obs_so_far: list[tuple[float, float]], forecast: dict[float, float], local_hour: float) -> dict:
    """Pure projection. obs_so_far: [(local hour, temp)]; forecast: {hour: temp}. Degrees C."""
    high_so_far = max(t for _, t in obs_so_far)
    recent = [(h, t) for h, t in obs_so_far if h >= local_hour - 3]
    diffs = [t - forecast[int(h)] for h, t in recent if int(h) in forecast]
    adj = BIAS_DAMPING * sum(diffs) / len(diffs) if diffs else 0.0
    ahead = [t + adj for h, t in forecast.items() if h > local_hour]
    remaining_high = max(ahead) if ahead else float("-inf")
    return {
        "high_so_far": high_so_far,
        "remaining_high": None if not ahead else remaining_high,
        "bias_adjustment": adj,
        "center": max(high_so_far, remaining_high),
        "hours_to_peak": PEAK_HOUR - local_hour,
        "sd": spread_for(PEAK_HOUR - local_hour),
    }


def fetch_today(client: httpx.Client, lat: float, lon: float) -> tuple[dict[float, float], float, str, str]:
    """(hourly forecast for today by local hour, current local hour, tz name, today's local date)."""
    r = client.get(FORECAST_URL, params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "forecast_days": 1, "timezone": "auto"})
    r.raise_for_status()
    p = r.json()
    local_now = datetime.now(timezone.utc) + timedelta(seconds=p["utc_offset_seconds"])
    today = local_now.date().isoformat()
    fc = {float(int(t[11:13])): v for t, v in zip(p["hourly"]["time"], p["hourly"]["temperature_2m"]) if t[:10] == today and v is not None}
    return fc, local_now.hour + local_now.minute / 60, p["timezone"], today


def fetch_obs_today(client: httpx.Client, code: str, tz: str, today: str) -> list[tuple[float, float]]:
    d = datetime.fromisoformat(today)
    r = client.get(IEM_URL, params={
        "station": code, "data": "tmpc", "year1": d.year, "month1": d.month, "day1": d.day,
        "year2": (d + timedelta(days=1)).year, "month2": (d + timedelta(days=1)).month, "day2": (d + timedelta(days=1)).day,
        "tz": tz, "format": "onlycomma", "latlon": "no", "elev": "no", "missing": "empty", "trace": "T", "direct": "no", "report_type": [3, 4]})
    r.raise_for_status()
    out = []
    for row in csv.DictReader(io.StringIO(r.text)):
        if row["valid"][:10] == today and row.get("tmpc") not in (None, "", "M"):
            out.append((int(row["valid"][11:13]) + int(row["valid"][14:16]) / 60, float(row["tmpc"])))
    return out


def analyze_contract(contract: Contract, client: httpx.Client, *, lead_days: int) -> SameDayReport | None:
    s = contract.semantics_view()
    if s.kind != "highest" or lead_days != 0 or not s.station_code:
        return None
    lat, lon = resolve_coordinates(s.station_code)
    fc, hour, tz, today = fetch_today(client, lat, lon)
    obs = fetch_obs_today(client, s.station_code, tz, today)
    if not obs or not fc:
        return None
    pr = project(obs, fc, hour)
    k, off = (9 / 5, 32.0) if s.unit == "F" else (1.0, 0.0)
    center, sd, floor = pr["center"] * k + off, pr["sd"] * k, pr["high_so_far"] * k + off - 0.5
    return SameDayReport(
        event_id=s.event_id,
        station_code=s.station_code,
        local_hour=round(hour, 2),
        high_so_far=round(pr["high_so_far"] * k + off, 1),
        remaining_forecast_high=None if pr["remaining_high"] is None else round(pr["remaining_high"] * k + off, 1),
        bias_adjustment=round(pr["bias_adjustment"] * k, 2),
        center=round(center, 2),
        sd=round(sd, 2),
        bracket_probabilities=vector(s, center=center, sd=sd, lower=floor),
        generated_at=ict_now(),
    )

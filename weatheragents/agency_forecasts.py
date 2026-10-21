"""Collects national weather agencies' own forecasts, to test whether human forecasters know things our models do not.

    python -m weatheragents.agency_forecasts              # one collection now
    python -m weatheragents.agency_forecasts --watch 360  # every 6 hours, until stopped

Three agencies publish free, structured bulletins written by their forecasters, for cities Polymarket lists:
  Hong Kong Observatory   9-day highs and lows, rain outlook        (station HKO)
  Japan Meteorological    7-day highs and lows, rain probability    (Tokyo, RJTT)
  Singapore Met Service   4-day highs and lows, weather summary     (WSSS)

Each row keeps the time it was collected, the date it covers, and how many days ahead that is, because the
question is whether the agency's number, issued days ahead, carries information beyond the weather models'.
agency_report.py scores that once outcomes exist. Appended to data/agency/forecasts.jsonl (Thailand time).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from weatheragents.timeutil import now as ict_now

PATH = Path("data") / "agency" / "forecasts.jsonl"
HKO_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=en"
JMA_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast/130000.json"
MSS_URL = "https://api-open.data.gov.sg/v2/real-time/api/four-day-outlook"
UTC_OFFSET_H = {"Hong Kong": 8, "Tokyo": 9, "Singapore": 8}  # none observes daylight saving
STATIONS = {"Hong Kong": "HKO", "Tokyo": "RJTT", "Singapore": "WSSS"}


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_hko(payload: dict) -> list[dict]:
    out = []
    for f in payload.get("weatherForecast", []):
        d = f["forecastDate"]
        out.append({"target_date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "high": _num(f["forecastMaxtemp"]["value"]), "low": _num(f["forecastMintemp"]["value"]),
                    "rain": f.get("PSR"), "text": f.get("forecastWeather")})
    return out


def parse_jma(payload: list) -> list[dict]:
    """Tokyo's 7-day series: highs, lows and daily rain probability. The first day often has no high yet."""
    week = payload[1]["timeSeries"]
    pops = next((ts for ts in week if "pops" in ts["areas"][0]), None)
    temps = next((ts for ts in week if "tempsMax" in ts["areas"][0]), None)
    if temps is None:
        return []
    area = next((a for a in temps["areas"] if a["area"]["name"] == "東京"), temps["areas"][0])
    out = []
    for i, t in enumerate(temps["timeDefines"]):
        pop = pops["areas"][0]["pops"][i] if pops and i < len(pops["areas"][0]["pops"]) else None
        out.append({"target_date": t[:10], "high": _num(area["tempsMax"][i]), "low": _num(area["tempsMin"][i]), "rain": _num(pop), "text": None})
    return out


def parse_mss(payload: dict) -> list[dict]:
    records = payload["data"]["records"]
    out = []
    for f in records[0]["forecasts"] if records else []:
        out.append({"target_date": f["timestamp"][:10], "high": _num(f["temperature"]["high"]), "low": _num(f["temperature"]["low"]),
                    "rain": None, "text": f["forecast"]["summary"]})
    return out


def collect(client: httpx.Client) -> list[dict]:
    collected = ict_now()
    rows = []
    for city, url, parser in (("Hong Kong", HKO_URL, parse_hko), ("Tokyo", JMA_URL, parse_jma), ("Singapore", MSS_URL, parse_mss)):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            items = parser(resp.json())
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            continue  # one agency being down must not lose the others
        local_today = (datetime.now(timezone.utc) + timedelta(hours=UTC_OFFSET_H[city])).date()
        for it in items:
            if it["high"] is None:
                continue
            lead = (datetime.fromisoformat(it["target_date"]).date() - local_today).days
            rows.append({"collected_at": collected.isoformat(timespec="seconds"), "city": city, "station": STATIONS[city],
                         "agency": {"Hong Kong": "HKO", "Tokyo": "JMA", "Singapore": "MSS"}[city], "lead_days": lead, **it})
    return rows


def read(path: Path = PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Collect national agency forecasts")
    ap.add_argument("--watch", type=float, default=None, metavar="MINUTES")
    args = ap.parse_args(argv)
    with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1 (research)"}) as client:
        while True:
            rows = collect(client)
            PATH.parent.mkdir(parents=True, exist_ok=True)
            with PATH.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{ict_now():%m-%d %H:%M} ICT: {len(rows)} agency forecast rows from {len({r['agency'] for r in rows})} agencies", flush=True)
            if args.watch is None:
                return 0
            time.sleep(args.watch * 60)


if __name__ == "__main__":
    raise SystemExit(main())

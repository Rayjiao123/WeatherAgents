"""Which way of building the day's high reproduces Polymarket's actual results?

    python -m weatheragents.resolution_check

A contract resolves on one source's daily high, and the details matter: whether the day runs on
local clock or UTC, whether daylight-saving time counts, whether every half-hourly report is used or
only some, how Fahrenheit is rounded. Each choice can move the high by a degree. This rebuilds the
high from the station's own reports under each rule and counts how often it lands in the bracket
Polymarket actually paid. The rule that matches most is the one to train on and to forecast.
"""

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from weatheragents import prediction_log as plog
from weatheragents.calibration_fit import IEM_URL
from weatheragents.scout import ParseError, parse_bracket

META_URL = "https://mesonet.agron.iastate.edu/api/1/station/{code}.json"
RULES = {
    "A  local day, all reports": ("local", None),
    "B  local day, routine reports only": ("local", {3}),
    "C  local standard-time day, all reports": ("standard", None),
    "D  UTC day, all reports": ("utc", None),
}
F_VARIANTS = ("tmpf rounded", "tmpc converted then rounded")


def station_tz(client: httpx.Client, code: str, cache: dict) -> str | None:
    if code not in cache:
        try:
            r = client.get(META_URL.format(code=code))
            r.raise_for_status()
            cache[code] = (r.json().get("data") or [{}])[0].get("tzname")
        except (httpx.HTTPError, ValueError):
            cache[code] = None
    return cache[code]


def fetch_reports(client: httpx.Client, code: str, day: date) -> list[dict]:
    """All reports from the day before to the day after, in UTC, tagged with IEM's report type."""
    out = []
    d0, d1 = day - timedelta(days=1), day + timedelta(days=2)
    for rt in (3, 4):
        resp = client.get(IEM_URL, params={
            "station": code, "data": ["tmpc", "tmpf"], "year1": d0.year, "month1": d0.month, "day1": d0.day,
            "year2": d1.year, "month2": d1.month, "day2": d1.day, "tz": "UTC", "format": "onlycomma",
            "latlon": "no", "elev": "no", "missing": "empty", "trace": "T", "direct": "no", "report_type": rt})
        resp.raise_for_status()
        for row in csv.DictReader(io.StringIO(resp.text)):
            if row.get("tmpc") in (None, "", "M"):
                continue
            out.append({"utc": datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc),
                        "tmpc": float(row["tmpc"]), "tmpf": float(row["tmpf"]) if row.get("tmpf") not in (None, "", "M") else None, "rt": rt})
    return out


def standard_offset(tz: str) -> timedelta:
    z = ZoneInfo(tz)
    return min(z.utcoffset(datetime(2026, 1, 15)), z.utcoffset(datetime(2026, 7, 15)))


def day_of(utc: datetime, how: str, tz: str) -> date:
    if how == "utc":
        return utc.date()
    if how == "standard":
        return (utc + standard_offset(tz)).date()
    return utc.astimezone(ZoneInfo(tz)).date()


def high(reports: list[dict], day: date, how: str, types: set | None, tz: str, unit: str, fvariant: str) -> int | None:
    sel = [r for r in reports if day_of(r["utc"], how, tz) == day and (types is None or r["rt"] in types)]
    if not sel:
        return None
    if unit == "C":
        return round(max(r["tmpc"] for r in sel))
    if fvariant == F_VARIANTS[0]:
        fs = [r["tmpf"] for r in sel if r["tmpf"] is not None]
        return round(max(fs)) if fs else None
    return round(max(r["tmpc"] for r in sel) * 9 / 5 + 32)


def bracket_label(labels: list[str], value: int) -> str | None:
    for lab in labels:
        try:
            b, _ = parse_bracket(lab)
        except ParseError:
            continue
        if (b.low is None or value >= b.low) and (b.high is None or value <= b.high):
            return lab
    return None


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Test rules for building the day's high against Polymarket's results")
    ap.add_argument("--log-dir", type=Path, default=plog.LOG_DIR)
    ap.add_argument("--final-only", action="store_true")
    args = ap.parse_args(argv)

    preds = {}
    for p in plog.read(args.log_dir / plog.PREDICTIONS):
        preds[p["event_id"]] = p
    outcomes = {o["event_id"]: o for o in plog.read(args.log_dir / plog.OUTCOMES)}
    events = [(preds[e], o) for e, o in outcomes.items() if e in preds and preds[e]["kind"] == "highest"
              and preds[e].get("station_code") and (o["status"] == "final" or not args.final_only)]
    print(f"{len(events)} resolved high-temperature events to check")

    tally: dict[str, Counter] = defaultdict(Counter)
    misses = defaultdict(list)
    tzcache: dict = {}
    with httpx.Client(timeout=60, headers={"User-Agent": "weatheragents/0.1"}) as client:
        for p, o in events:
            code, unit, day = p["station_code"], p["unit"], date.fromisoformat(p["observation_date"])
            tz = station_tz(client, code, tzcache)
            if not tz:
                continue
            try:
                reports = fetch_reports(client, code, day)
            except httpx.HTTPError:
                continue
            region = "US (Fahrenheit)" if unit == "F" else "elsewhere (Celsius)"
            variants = F_VARIANTS if unit == "F" else (None,)
            for name, (how, types) in RULES.items():
                for fv in variants:
                    label = name if fv is None else f"{name} | {fv}"
                    v = high(reports, day, how, types, tz, unit, fv)
                    got = bracket_label(p["brackets"], v) if v is not None else None
                    key = (region, label)
                    tally[key]["n"] += 1
                    tally[key]["hit"] += int(got == o["resolved_label"])
                    if got != o["resolved_label"]:
                        misses[key].append((p["city"], str(day), o["resolved_label"], got, v))

    for region in ("elsewhere (Celsius)", "US (Fahrenheit)"):
        rows = sorted(((k[1], c["hit"], c["n"]) for k, c in tally.items() if k[0] == region), key=lambda r: -r[1] / max(r[2], 1))
        if not rows:
            continue
        print(f"\n{region}: how often each rule lands in the bracket Polymarket paid")
        for label, hit, n in rows:
            print(f"  {hit / n:>5.0%}  ({hit}/{n})  {label}")
        best = (region, rows[0][0])
        print(f"  misses under the best rule: " + "; ".join(f"{c} {d} paid {lab} vs {got}" for c, d, lab, got, v in misses[best][:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

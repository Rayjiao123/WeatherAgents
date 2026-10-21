"""Records what the newest weather-model runs say, every time the pipeline predicts.

    python -m weatheragents.fresh_run     # once outcomes exist: is the newest run more accurate than the one we model on?

The day-ahead ML is trained on forecasts issued a day before the target date (the history service keeps those). The
market trades on the newest runs, which are fresher and, on history, noticeably more accurate (1.00 C typical miss
a day ahead against 0.81 C for the run on the day itself). History cannot give us the newest-run vintage, so this
logs it from now on: for every prediction, the four models' forecast highs for the target date at the moment of the
pass. Two uses, later: a score of how much better the fresh runs are on live outcomes, and a training set of the
vintage the market actually sees.
"""

import math
import statistics
import sys
from datetime import datetime

import httpx

from weatheragents import evaluate as ev
from weatheragents import prediction_log as plog
from weatheragents.ensemble_analyst import resolve_coordinates
from weatheragents.ml_data import MODELS
from weatheragents.schemas import Contract
from weatheragents.timeutil import now as ict_now

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def parse_fresh(payload: dict, obs_date: str) -> dict | None:
    """Each model's forecast high for the target local date, from a multi-model hourly response."""
    h = payload.get("hourly") or {}
    idx = [i for i, t in enumerate(h.get("time", [])) if t[:10] == obs_date]
    models = {}
    for m in MODELS:
        series = h.get(f"temperature_2m_{m}")
        vals = [series[i] for i in idx if series and series[i] is not None]
        if len(vals) >= 20:
            models[m] = max(vals)
    if len(models) < 2:
        return None
    return {"models": models, "mean": statistics.fmean(models.values())}


def fetch(contract: Contract, client: httpx.Client) -> dict | None:
    s = contract.semantics_view()
    if s.kind != "highest" or not s.station_code:
        return None
    lat, lon = resolve_coordinates(s.station_code)
    r = client.get(FORECAST_URL, params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "models": ",".join(MODELS),
                                         "start_date": str(s.observation_date), "end_date": str(s.observation_date), "timezone": "auto"})
    r.raise_for_status()
    out = parse_fresh(r.json(), str(s.observation_date))
    return None if out is None else {"fetched_at": ict_now().isoformat(timespec="seconds"), "unit": "C", **out}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    preds = ev.latest_before_resolution([p for p in plog.read(plog.LOG_DIR / plog.PREDICTIONS) if p.get("fresh_run") and p.get("ml")])
    outs = {o["event_id"]: o for o in plog.read(plog.LOG_DIR / plog.OUTCOMES)}
    rows = []
    for eid, p in preds.items():
        o = outs.get(eid)
        actual = ev.outcome_value(o["resolved_label"]) if o else None
        if actual is None or p["unit"] != "C":
            continue
        rows.append({"actual": actual, "fresh": p["fresh_run"]["mean"], "stale": p["ml"]["multi_model_mean"], "ml": p["ml"]["center"], "city": p["city"]})
    print(f"{len(rows)} resolved Celsius events that logged a fresh-run forecast")
    if len(rows) < 10:
        print("too few so far: the logging started with the first pass after this change; outcomes arrive as events resolve")
        return 0
    mean_err = lambda key: statistics.fmean(r["actual"] - r[key] for r in rows)
    print(f"\n  {'forecast':<42}{'average miss':>13}{'typical miss':>14}")
    for label, key in (("models' mean, run issued a day earlier (ML input)", "stale"), ("models' mean, newest run at prediction time", "fresh"), ("the ML's own centre", "ml")):
        print(f"  {label:<42}{mean_err(key):>+12.2f}C{statistics.fmean(abs(r['actual'] - r[key]) for r in rows):>13.2f}C")
    d = [abs(r["actual"] - r["fresh"]) - abs(r["actual"] - r["stale"]) for r in rows]
    se = statistics.pstdev(d) / math.sqrt(len(d))
    print(f"\n  newest run minus older run, typical miss: {statistics.fmean(d):+.2f}C (range {statistics.fmean(d) - 1.96 * se:+.2f} to {statistics.fmean(d) + 1.96 * se:+.2f}); negative = fresher is better")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

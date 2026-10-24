"""Catch new Polymarket temperature events the moment they open and run our forecasts on them straight away.

    python -m weatheragents.open_watch --watch 5        # poll every 5 minutes
    python -m weatheragents.open_watch                  # one pass

The first pass only records which events already have a funded order book (they opened before we were watching).
After that, every event whose book gets funded is treated as an opening:
  1. its opening prices are saved to data/market/openings.jsonl
  2. the numeric pipeline runs (ML, newest-run ML, baselines) and is logged as v3-numeric
  3. the full pipeline with the LLM researchers and Manager also runs, logged as v3-ml-analyst, up to a daily budget
     (web search credits are limited), so the two can be scored against each other.

Nothing here places a bet. `paper_trader` scores the logged predictions against the opening prices.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

from weatheragents import ml_analyst, sameday_ml
from weatheragents import prediction_log as plog
from weatheragents.baseline import Calibration
from weatheragents.llm import make_client, model_name
from weatheragents.market_monitor import MIN_LIQUIDITY, snap_event
from weatheragents.pipeline import run_event
from weatheragents.scout import ParseError, fetch_events, parse_event
from weatheragents.timeutil import now as ict_now

SEEN_PATH = Path("data") / "market" / "seen_events.json"
OPENINGS_PATH = Path("data") / "market" / "openings.jsonl"
LOG_PATH = Path("data") / "log" / plog.PREDICTIONS
CATCHUP_AGE_HOURS = 24.0  # refresh a market's forecast when its newest one is older than this
CATCHUP_PER_PASS = 8  # and only this many per pass, to stay inside the weather API quota
MIN_HOURS_LEFT = 20.0
DEFAULT_LLM_BUDGET = 0  # full (LLM + web search) runs per ICT calendar day; off since 2026-10-16, it scored 0.44 behind the market


def load_seen(path: Path = SEEN_PATH) -> dict:
    if not path.exists():
        return {"ids": [], "llm_day": "", "llm_used": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def save_seen(state: dict, path: Path = SEEN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def funded_contracts(events: list[dict]) -> list:
    """Parseable events that have a funded order book. Events are created hours before anyone can trade them, so the
    opening is when liquidity first appears, not when the event first shows up on Gamma."""
    out = []
    for e in events:
        try:
            c = parse_event(e)
        except (ParseError, KeyError, ValueError):
            continue
        if c.pricing.total_liquidity >= MIN_LIQUIDITY and c.semantics.station_code:
            out.append(c)
    return out


def llm_allowed(state: dict, budget: int, today: str) -> bool:
    if state.get("llm_day") != today:
        state["llm_day"], state["llm_used"] = today, 0
    return state["llm_used"] < budget


def stale_highs(funded: list, newest: dict[str, str], now: datetime, skip: set[str]) -> list:
    """Funded high-temperature markets with no forecast newer than CATCHUP_AGE_HOURS, nearest to resolution last."""
    out = []
    for c in funded:
        s = c.semantics
        if s.kind != "highest" or s.event_id in skip or (s.resolves_at - now).total_seconds() / 3600 < MIN_HOURS_LEFT:
            continue
        last = newest.get(s.event_id)
        if last is None or (now - datetime.fromisoformat(last)).total_seconds() / 3600 > CATCHUP_AGE_HOURS:
            out.append(c)
    return sorted(out, key=lambda c: -c.pricing.total_liquidity)


def one_pass(http: httpx.Client, state: dict, *, budget: int, llm, calib, ml_model, sameday_model) -> int:
    funded = funded_contracts(fetch_events(http))
    done = set(state["ids"])
    first = not done
    fresh = [] if first else [c for c in funded if c.semantics.event_id not in done]
    state["ids"] = sorted(done | {c.semantics.event_id for c in funded})
    if first:
        save_seen(state)
        print(f"{ict_now():%H:%M} baseline: {len(funded)} funded events recorded; the next to get funded counts as an opening", flush=True)
        return 0
    handled = 0
    for c in fresh:
        s = c.semantics
        ts = ict_now().isoformat(timespec="seconds")
        OPENINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OPENINGS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snap_event(c, ts), ensure_ascii=False) + "\n")
        run_id = plog.new_run_id()
        print(f"{ict_now():%H:%M} OPENED {s.city} {s.kind} {s.observation_date} (liquidity ${c.pricing.total_liquidity:,.0f})", flush=True)
        runs = [(None, plog.NUMERIC_VERSION, "none")]
        if llm is not None and llm_allowed(state, budget, ict_now().strftime("%Y-%m-%d")):
            runs.append((llm, plog.PIPELINE_VERSION, model_name()))
            state["llm_used"] += 1
        for client, version, model in runs:
            try:
                result = run_event(c, http, client, rounds=2, calib=calib, ml_model=ml_model, sameday_model=sameday_model)
                plog.append(LOG_PATH, plog.build_record(c, result, run_id=run_id, model=model, version=version, trigger="open"))
                print(f"        logged {version}", flush=True)
            except Exception as exc:  # one failing run must not stop the watcher
                print(f"        ! {version}: {type(exc).__name__}: {exc}", file=sys.stderr)
        handled += 1
        save_seen(state)
    save_seen(state)
    newest: dict[str, str] = {}  # markets we missed at the open (PC asleep, or listed before we watched) still need a forecast to trade on
    for p in plog.read(LOG_PATH):
        if p.get("ml") and p["logged_at"] > newest.get(p["event_id"], ""):
            newest[p["event_id"]] = p["logged_at"]
    for c in stale_highs(funded, newest, ict_now(), {x.semantics.event_id for x in fresh})[:CATCHUP_PER_PASS]:
        try:
            result = run_event(c, http, None, rounds=2, calib=calib, ml_model=ml_model, sameday_model=sameday_model)
            plog.append(LOG_PATH, plog.build_record(c, result, run_id=plog.new_run_id(), model="none", version=plog.NUMERIC_VERSION, trigger="catchup"))
            print(f"{ict_now():%H:%M} catch-up forecast {c.semantics.city} {c.semantics.observation_date}", flush=True)
        except Exception as exc:
            print(f"        ! catch-up {c.semantics.city}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return handled


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser(description="Run forecasts on newly opened markets")
    ap.add_argument("--watch", type=int, default=0, metavar="MINUTES", help="poll forever at this interval")
    ap.add_argument("--llm-budget", type=int, default=DEFAULT_LLM_BUDGET, help="full LLM runs per day (0 turns the LLM off)")
    args = ap.parse_args(argv)
    llm = make_client() if args.llm_budget > 0 else None
    calib, ml_model, sameday_model = Calibration.load(), ml_analyst.load_model(), sameday_ml.load()
    state = load_seen()
    while True:
        try:
            with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1"}) as http:
                one_pass(http, state, budget=args.llm_budget, llm=llm, calib=calib, ml_model=ml_model, sameday_model=sameday_model)
        except Exception as exc:
            print(f"{ict_now():%H:%M} ! pass failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not args.watch:
            return 0
        sys.stdout.flush()
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    raise SystemExit(main())

"""Runs agents 01-08 end to end for a few events and prints the result.

    python -m weatheragents.pipeline --limit 3
    python -m weatheragents.pipeline --city Seoul --city Houston

Reads the latest data/contract_list_*.json from Market Scout (run
`python -m weatheragents.scout` first). Writes data/research_<timestamp>.json
with every agent's output per event.

The market column in the printout is for the human reading it. It is read
after the pipeline finishes and never reaches an agent; the Trader (agent 09)
is where price and probability will meet.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

from weatheragents import prediction_log as plog
from weatheragents import alerts_analyst, fresh_run, ml_analyst, sameday_analyst, sameday_ml
from weatheragents import climatology_analyst, ensemble_analyst, local_research_analyst, nowcast_analyst
from weatheragents.baseline import Calibration, baseline_params, baseline_vector, recent_miss_adjustment
from weatheragents.ensemble_analyst import _latest_contract_list
from weatheragents.llm import make_client, model_name
from weatheragents.research_debate import AnalystReports, debate, manage
from weatheragents.timeutil import now as ict_now, stamp as ict_stamp, to_ict
from weatheragents.schemas import Contract


# Switched off 2026-10-16: neither has earned a place. The same-day model has no live score and no trader uses it; the LLM layer
# scored 0.44 worse than the market (94 events) and worse than the plain baseline. The code stays; set True to run it again.
SAMEDAY_ON = False


def run_event(contract: Contract, http: httpx.Client, llm, *, rounds: int, calib: Calibration, ml_model=None, sameday_model=None) -> dict:
    """llm=None runs the numeric pipeline: no local research, debate or Manager (and no web search)."""
    s = contract.semantics_view()  # everything below sees semantics only
    ensemble = ensemble_analyst.analyze_contract(contract, http)
    nowcast = nowcast_analyst.analyze_contract(contract, http)
    climo = climatology_analyst.analyze_contract(contract)
    local = local_research_analyst.research(llm, s) if llm else None
    try:
        fresh = fresh_run.fetch(contract, http)
    except (httpx.HTTPError, LookupError, ValueError, KeyError):
        fresh = None  # the newest-run forecast is optional: logged, and fed to the model as a second version
    try:
        ml, ml_fresh = ml_analyst.analyze_pair(contract, http, lead_days=nowcast.lead_days, loaded=ml_model, fresh=fresh)
    except (httpx.HTTPError, LookupError, ValueError, KeyError):
        ml = ml_fresh = None  # the ML Analyst is optional; the pipeline carries on without it
    try:
        alerts = alerts_analyst.analyze_contract(contract, http)
    except (httpx.HTTPError, LookupError, ValueError, KeyError):
        alerts = None  # logged for later study only; never blocks a prediction
    sameday = sameday_learned = None
    if SAMEDAY_ON:
        try:
            sameday = sameday_analyst.analyze_contract(contract, http, lead_days=nowcast.lead_days)
        except (httpx.HTTPError, LookupError, ValueError, KeyError):
            sameday = None  # only applies to events whose day is still in progress
        try:
            sameday_learned = sameday_ml.analyze_contract(contract, http, lead_days=nowcast.lead_days, loaded=sameday_model)
        except (httpx.HTTPError, LookupError, ValueError, KeyError):
            sameday_learned = None
    params = baseline_params(s, ensemble, nowcast, calib)
    plus = baseline_vector(s, params, shift=recent_miss_adjustment(params, ml.recent_error_mean if ml else None))
    if llm:
        reports = AnalystReports(ensemble=ensemble, nowcast=nowcast, climatology=climo, local=local, ml=ml)
        warmer, cooler, history = debate(llm, s, reports, rounds=rounds)
        estimate = manage(llm, s, ensemble, warmer, cooler, history, params=params, ml=ml).model_dump(mode="json")
        estimate["baseline_plus_probabilities"] = [b.model_dump() for b in plus]
    else:
        warmer = cooler = None
        history = []
        estimate = {  # no Manager vector; the baseline curves are still logged so they can be scored
            "bracket_probabilities": None,
            "baseline_probabilities": [b.model_dump() for b in baseline_vector(s, params)],
            "baseline_uncorrected_probabilities": [b.model_dump() for b in baseline_vector(s, params, corrected=False)],
            "baseline_plus_probabilities": [b.model_dump() for b in plus],
            "baseline_params": params.as_dict(),
        }
    return {
        "event_id": s.event_id,
        "city": s.city,
        "observation_date": str(s.observation_date),
        "ensemble_report": ensemble.model_dump(mode="json"),
        "nowcast_report": nowcast.model_dump(mode="json"),
        "climatology_report": climo.model_dump(mode="json"),
        "local_context_report": local.model_dump(mode="json") if local else None,
        "ml_report": ml.model_dump(mode="json") if ml else None,
        "ml_fresh_report": ml_fresh.model_dump(mode="json") if ml_fresh else None,
        "alerts_report": alerts.model_dump(mode="json") if alerts else None,
        "sameday_report": sameday.model_dump(mode="json") if sameday else None,
        "sameday_ml_report": sameday_learned.model_dump(mode="json") if sameday_learned else None,
        "fresh_run": fresh,
        "warmer_case": warmer.model_dump(mode="json") if warmer else None,
        "cooler_case": cooler.model_dump(mode="json") if cooler else None,
        "debate_history": history,
        "probability_estimate": estimate,
    }


def print_event(contract: Contract, result: dict) -> None:
    s, p = contract.semantics, contract.pricing
    ens = {b["label"]: b["probability"] for b in result["ensemble_report"]["bracket_probabilities"]}
    clim = {b["label"]: b["probability"] for b in result["climatology_report"]["bracket_probabilities"]}
    est = result["probability_estimate"]
    mgr = {b["label"]: b["probability"] for b in est["bracket_probabilities"]} if est["bracket_probabilities"] else {}
    base = {b["label"]: b["probability"] for b in result["probability_estimate"]["baseline_probabilities"]}
    mkt = {
        b.label: (b.best_bid + b.best_ask) / 2 if b.best_bid is not None and b.best_ask is not None else b.last_trade
        for b in p.brackets
    }
    nc = result["nowcast_report"]
    print(f"\n=== {s.city} {s.kind} on {s.observation_date} ({s.station_code}, {s.unit}) ===")
    print(
        f"nowcast: forecast {nc['forecast_extreme']} (3d ago {nc['previous_run_extremes'].get('3d')}, {nc['trend']}); "
        f"observed so far: {nc['observed_extreme_so_far']}"
    )
    if result["local_context_report"]:
        print(f"local: leans {result['local_context_report']['leans']} - {result['local_context_report']['summary'][:200]}")
    bp = est["baseline_params"]
    print(
        (f"adjustment: shift {est['shift']:+.2f}, spread x{est['spread_multiplier']:.2f} " if mgr else "")
        + f"(baseline center {bp['center']:.1f}, sd {bp['sd']:.2f}{'' if bp['station_fit'] else ', pooled default sd'})"
    )
    ml = result.get("ml_report")
    if ml:
        print(f"ML analyst: center {ml['center']:.1f} (weather models' mean {ml['multi_model_mean']:.1f}), sd {ml['sd']:.2f}")
    mlp = {b["label"]: b["probability"] for b in ml["bracket_probabilities"]} if ml else {}
    print(f"\n{'bracket':<16}{'climo':>7}{'ensemble':>10}{'baseline':>10}{'ML':>6}{'MANAGER':>9}{'market*':>9}")
    for b in s.brackets:
        m = mkt.get(b.label)
        print(
            f"{b.label:<16}{clim[b.label]:>7.0%}{ens[b.label]:>10.0%}{base[b.label]:>10.0%}{(format(mlp[b.label], '.0%') if mlp else '-'):>6}{(format(mgr[b.label], '.0%') if mgr else '-'):>9}"
            f"{'' if m is None else format(m, '.0%'):>9}"
        )
    if mgr:
        print(f"confidence {est['confidence']:.2f} - {est['rationale'][:300]}")
    print("* market mid, shown for you only; no agent saw it")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # degree signs on a Windows console
    parser = argparse.ArgumentParser(description="Run agents 01-08 on a few events")
    parser.add_argument("--contracts", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=3, help="most-liquid N events (default 3)")
    parser.add_argument("--city", action="append", default=[], help="only these cities (repeatable, substring match)")
    parser.add_argument("--rounds", type=int, default=2, help="debate rounds (default 2)")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--no-log", action="store_true", help="don't write to the prediction log")
    parser.add_argument("--mode", choices=["numeric", "full"], default="numeric",
                        help="numeric (default): analysts, ML and baseline only. full: also local research, debate and Manager (uses DeepSeek and web search)")
    args = parser.parse_args(argv)

    path = args.contracts or _latest_contract_list(Path("data"))
    if path is None or not path.exists():
        print("no contract_list file found; run `python -m weatheragents.scout` first", file=sys.stderr)
        return 1
    contracts = [Contract.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    if args.city:
        contracts = [c for c in contracts if any(q.lower() in c.semantics.city.lower() for q in args.city)]
    contracts = contracts[: args.limit]
    if not contracts:
        print("no matching contracts", file=sys.stderr)
        return 1

    calib = Calibration.load()
    if not calib.fitted:
        print("note: data/calibration.json missing; using default error spreads. Run `python -m weatheragents.calibration_fit`.", file=sys.stderr)
    ml_model = ml_analyst.load_model()
    sameday_model = sameday_ml.load()
    if ml_model is None:
        print("note: no trained ML model in data/ml; the ML Analyst will be skipped.", file=sys.stderr)
    llm = make_client() if args.mode == "full" else None
    print(f"mode {args.mode}{f' (model {model_name()})' if llm else ' (no LLM, no web search)'}, {len(contracts)} events from {path.name}")
    results = []
    run_id = plog.new_run_id()
    log_path = args.out / "log" / plog.PREDICTIONS
    with httpx.Client(timeout=30, headers={"User-Agent": "weatheragents/0.1"}) as http:
        for c in contracts:
            try:
                result = run_event(c, http, llm, rounds=args.rounds, calib=calib, ml_model=ml_model, sameday_model=sameday_model)
            except Exception as exc:  # keep going; report which event failed and why
                print(f"\n! {c.semantics.city} {c.semantics.observation_date}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            results.append(result)
            if not args.no_log:
                plog.append(log_path, plog.build_record(c, result, run_id=run_id, model=model_name() if llm else "none",
                                                          version=plog.PIPELINE_VERSION if llm else plog.NUMERIC_VERSION))
            print_event(c, result)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = ict_stamp()
    out_file = args.out / f"research_{stamp}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

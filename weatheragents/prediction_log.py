"""Append-only prediction log: what every agent said before the outcome was known.

    data/log/predictions.jsonl   one JSON line per (event, pipeline run)
    data/log/outcomes.jsonl      one JSON line per resolved event (see resolver.py)

Records are never edited. The evaluator reads these files and nothing else,
so it can't be fooled by a later re-run. `market_snapshot` is stored only so
the evaluator can compare against the market; no agent ever receives it.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from weatheragents.timeutil import now as ict_now, stamp as ict_stamp, to_ict

# Bump when an agent's prompt or logic changes, so the evaluator can score versions apart.
# v1 (no stamp): LLM wrote all bracket probabilities, single-model ensemble.
# v2: multi-model ensemble, calibrated baseline curve, LLM proposes shift + spread only.
# v3: adds the ML Analyst (logged as its own source, and shown to the researchers and Manager).
PIPELINE_VERSION = "v3-ml-analyst"
# Added to v3-numeric records without a version bump (they only add fields): alerts, sameday.
# v3-numeric: same analysts and baseline, but no LLM stages and no web search (no local research, debate or Manager).
NUMERIC_VERSION = "v3-numeric"

LOG_DIR = Path("data") / "log"
PREDICTIONS = "predictions.jsonl"
OUTCOMES = "outcomes.jsonl"


def new_run_id() -> str:
    return ict_stamp() + "-" + uuid.uuid4().hex[:6]


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_record(contract, result: dict, *, run_id: str, model: str, version: str = PIPELINE_VERSION) -> dict:
    """One prediction. `contract` is a schemas.Contract, `result` is pipeline.run_event's dict."""
    s = contract.semantics
    return {
        "run_id": run_id,
        "logged_at": ict_now().isoformat(),
        "llm_model": model,
        "pipeline_version": version,
        "event_id": s.event_id,
        "city": s.city,
        "station_code": s.station_code,
        "kind": s.kind,
        "unit": s.unit,
        "observation_date": str(s.observation_date),
        "resolves_at": to_ict(s.resolves_at).isoformat(),
        "brackets": [b.label for b in s.brackets],
        "ensemble": result["ensemble_report"],
        "nowcast": result["nowcast_report"],
        "climatology": result["climatology_report"],
        "local_context": result["local_context_report"],
        "ml": result.get("ml_report"),
        "ml_fresh": result.get("ml_fresh_report"),
        "alerts": result.get("alerts_report"),
        "sameday": result.get("sameday_report"),
        "sameday_ml": result.get("sameday_ml_report"),
        "fresh_run": result.get("fresh_run"),
        "warmer_case": result["warmer_case"],
        "cooler_case": result["cooler_case"],
        "debate_history": result["debate_history"],
        "estimate": result["probability_estimate"],
        "market_snapshot": contract.pricing.model_dump(mode="json"),
    }

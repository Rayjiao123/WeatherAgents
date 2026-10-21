"""Agents 06-08 - Warmer/Cooler Researchers and the Research Manager.

One debate per event. The researchers argue from the four analyst reports
only; the manager turns the debate into a probability for every bracket.
Nothing here ever receives a price: prompts are built from ContractSemantics
and analyst reports, and tests assert that.
"""

import json
from datetime import datetime

from pydantic import BaseModel

from weatheragents.timeutil import now as ict_now, stamp as ict_stamp, to_ict
from weatheragents.baseline import Params, baseline_vector, unit_scale
from weatheragents.schemas import (
    BracketProbability,
    ClimatologyReport,
    ContractSemantics,
    EnsembleReport,
    LocalContextReport,
    MLReport,
    NowcastReport,
    ProbabilityEstimate,
    ResearchCase,
)


class _CaseOut(BaseModel):
    argument: str
    favored_brackets: list[str]
    evidence: list[str]


class _AdjustOut(BaseModel):
    rationale: str  # first, so the model reasons before it commits to numbers
    shift: float  # degrees in the contract's unit; 0 unless the evidence says otherwise
    spread_multiplier: float  # 1.0 = keep the calibrated spread
    confidence: float


class AnalystReports(BaseModel):
    ensemble: EnsembleReport
    nowcast: NowcastReport
    climatology: ClimatologyReport
    local: LocalContextReport
    ml: MLReport | None = None


def _dump(model: BaseModel, drop: tuple[str, ...] = ("member_values", "generated_at", "event_id")) -> dict:
    return {k: v for k, v in model.model_dump(mode="json").items() if k not in drop}


def contract_block(s: ContractSemantics) -> str:
    return json.dumps(
        {
            "city": s.city,
            "station": s.station_name or s.station_code,
            "measures": f"the day's {s.kind} temperature, whole degrees {s.unit}",
            "observation_date": str(s.observation_date),
            "brackets": [b.label for b in s.brackets],
        },
        indent=2,
        ensure_ascii=False,
    )


def reports_block(r: AnalystReports) -> str:
    return json.dumps(
        {
            "ensemble_report": _dump(r.ensemble),
            "nowcast_report": _dump(r.nowcast),
            "climatology_report": _dump(r.climatology),
            "local_context_report": _dump(r.local),
            **({"ml_report": _dump(r.ml)} if r.ml else {}),
        },
        indent=2,
        ensure_ascii=False,
    )


CASE_SYSTEM = {
    "warmer": (
        "You are the Warmer-case researcher in a weather-forecast debate. Build the strongest honest "
        "case that the observed temperature will land warmer than the center of the ensemble forecast. "
        "Use only the analyst reports provided; cite their numbers. If the evidence for a warmer outcome "
        "is thin, say how thin rather than inventing support. favored_brackets must be labels copied "
        "exactly from the bracket list."
    ),
    "cooler": (
        "You are the Cooler-case researcher in a weather-forecast debate. Build the strongest honest "
        "case that the observed temperature will land cooler than the center of the ensemble forecast. "
        "Use only the analyst reports provided; cite their numbers. If the evidence for a cooler outcome "
        "is thin, say how thin rather than inventing support. favored_brackets must be labels copied "
        "exactly from the bracket list."
    ),
}


def case_prompt(s: ContractSemantics, r: AnalystReports, *, opponent: ResearchCase | None = None) -> str:
    text = f"EVENT\n{contract_block(s)}\n\nANALYST REPORTS\n{reports_block(r)}\n"
    if opponent is None:
        return text + "\nWrite your opening case."
    return (
        text
        + f"\nYOUR OPPONENT ({opponent.side} case) ARGUED:\n{opponent.argument}\n"
        + f"Evidence they cited: {opponent.evidence}\n"
        + "\nWrite your rebuttal: answer their strongest points and restate your case, conceding what is fair."
    )


def argue(client, s: ContractSemantics, r: AnalystReports, side: str, opponent: ResearchCase | None = None) -> ResearchCase:
    out = client.structured(system=CASE_SYSTEM[side], user=case_prompt(s, r, opponent=opponent), schema=_CaseOut)
    labels = {b.label for b in s.brackets}
    return ResearchCase(
        event_id=s.event_id,
        side=side,
        argument=out.argument,
        favored_brackets=[b for b in out.favored_brackets if b in labels],
        evidence=out.evidence,
    )


MAX_SHIFT_C = 2.0
SPREAD_RANGE = (0.7, 1.6)

MANAGER_SYSTEM = (
    "You are the Research Manager. Two researchers argued over where a day's temperature will land. A "
    "statistical baseline already turns the forecast into a probability for every bracket: its center blends "
    "the multi-model ensemble with a station-bias-corrected forecast, and its spread was fitted to this "
    "station's real forecast errors over the last 120 days. It already accounts for what the station has "
    "recorded so far today. So do NOT widen it for generic forecast uncertainty and do NOT rewrite the "
    "probabilities. Your only job is to propose two adjustments to that curve, only where the evidence "
    "warrants:\n"
    "- shift: degrees to move the center (0 on most days). Use it when the nowcast, run-to-run drift, or local "
    "reports show a clear warmer or cooler outcome than the baseline center.\n"
    "- spread_multiplier: 1.0 keeps the fitted spread. Raise it (up to 1.6) when the models disagree, a front or "
    "unusual weather makes the day hard to call, or the reports conflict. Lower it (down to 0.7) when everything "
    "agrees and the day is quiet.\n"
    "The researchers argue their side, so discount their enthusiasm; weigh the numbers. If an ML ANALYST block is present, it is a learned correction to the weather models, trained on years of forecast errors at this station: treat its center and spread as a serious second opinion, not as the answer. Say in the rationale what "
    "evidence moved you, or that nothing did. You have no information about market prices and must not guess "
    "them. confidence is 0 to 1."
)


def manager_prompt(
    s: ContractSemantics,
    ensemble: EnsembleReport,
    warmer: ResearchCase,
    cooler: ResearchCase,
    history: list[dict],
    params: Params,
    ml: MLReport | None = None,
) -> str:
    base_vec = baseline_vector(s, params)
    bounds = {k: v for k, v in (("cannot_end_below", params.lower), ("cannot_end_above", params.upper)) if v is not None}
    return (
        f"EVENT\n{contract_block(s)}\n\n"
        f"BASELINE (units: degrees {s.unit})\n"
        f"{json.dumps({'center': round(params.center, 2), 'sd': round(params.sd, 2), 'lead_days': params.lead_days, 'sd_fitted_to_this_station': params.station_fit, **bounds}, indent=2)}\n"
        f"baseline probabilities: {json.dumps({bp.label: round(bp.probability, 3) for bp in base_vec}, ensure_ascii=False)}\n\n"
        f"ENSEMBLE REPORT\n{json.dumps(_dump(ensemble), indent=2, ensure_ascii=False)}\n\n"
        + (f"ML ANALYST (a learned correction to the weather models; its curve is a second opinion on the center and spread)\n"
           f"{json.dumps(_dump(ml), indent=2, ensure_ascii=False)}\n\n" if ml else "")
        + f"WARMER CASE\n{warmer.model_dump_json(indent=2, exclude={'event_id'})}\n\n"
        f"COOLER CASE\n{cooler.model_dump_json(indent=2, exclude={'event_id'})}\n\n"
        f"DEBATE HISTORY (round, side, argument)\n{json.dumps(history, indent=2, ensure_ascii=False)}\n"
    )


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def manage(
    client,
    s: ContractSemantics,
    ensemble: EnsembleReport,
    warmer: ResearchCase,
    cooler: ResearchCase,
    history: list[dict],
    *,
    params: Params,
    ml: MLReport | None = None,
) -> ProbabilityEstimate:
    out = client.structured(
        system=MANAGER_SYSTEM, user=manager_prompt(s, ensemble, warmer, cooler, history, params, ml), schema=_AdjustOut
    )
    shift = clamp(out.shift, -MAX_SHIFT_C * unit_scale(s.unit), MAX_SHIFT_C * unit_scale(s.unit))
    mult = clamp(out.spread_multiplier, *SPREAD_RANGE)
    return ProbabilityEstimate(
        event_id=s.event_id,
        bracket_probabilities=baseline_vector(s, params, shift=shift, spread_multiplier=mult),
        confidence=clamp(out.confidence, 0.0, 1.0),
        rationale=out.rationale,
        generated_at=ict_now(),
        baseline_probabilities=baseline_vector(s, params),
        baseline_uncorrected_probabilities=baseline_vector(s, params, corrected=False),
        shift=round(shift, 3),
        spread_multiplier=round(mult, 3),
        baseline_params=params.as_dict(),
    )


def debate(client, s: ContractSemantics, r: AnalystReports, *, rounds: int = 2) -> tuple[ResearchCase, ResearchCase, list[dict]]:
    """Round 1: independent opening cases. Later rounds: each rebuts the other's latest."""
    history: list[dict] = []
    warmer = argue(client, s, r, "warmer")
    cooler = argue(client, s, r, "cooler")
    history += [
        {"round": 1, "side": "warmer", "argument": warmer.argument},
        {"round": 1, "side": "cooler", "argument": cooler.argument},
    ]
    for n in range(2, rounds + 1):
        prev_w, prev_c = warmer, cooler
        warmer = argue(client, s, r, "warmer", opponent=prev_c)
        cooler = argue(client, s, r, "cooler", opponent=prev_w)
        history += [
            {"round": n, "side": "warmer", "argument": warmer.argument},
            {"round": n, "side": "cooler", "argument": cooler.argument},
        ]
    return warmer, cooler, history

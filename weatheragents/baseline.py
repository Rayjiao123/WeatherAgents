"""Deterministic bracket distribution that the Research Manager adjusts.

A normal curve over the day's extreme, discretized onto the contract's
integer brackets:

  center  where the models say it lands, nudged by the station's fitted bias
  sd      how wrong forecasts at this station usually are (calibration_fit.py)
  bounds  what the station has already recorded today. A day's high can't end
          below the high so far; a low can't end above the low so far.

The LLM no longer writes eleven probabilities. It proposes a shift and a
spread multiplier on this curve (research_debate.py), so the numbers stay
coherent and the LLM's contribution is measurable on its own.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from weatheragents.schemas import BracketProbability, ContractSemantics, EnsembleReport, NowcastReport

CALIBRATION_PATH = Path("data") / "calibration.json"
# Baseline+ leans this fraction of the station's last-week forecast miss (beyond its usual bias) onto the centre.
# Fitted on 2024-26 history before 2026-04-01: 0.50. It takes the baseline from 1.706 to 1.669 on later dates.
RECENT_MISS_K = 0.5
FALLBACK = {"day0": {"bias_c": 0.0, "sd_c": 1.0}, "day1": {"bias_c": 0.0, "sd_c": 1.5}}


def unit_scale(unit: str) -> float:
    return 9 / 5 if unit == "F" else 1.0


def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class Calibration:
    def __init__(self, data: dict | None = None):
        data = data or {}
        self.stations = data.get("stations", {})
        self.default = data.get("default", FALLBACK)
        self.fitted = bool(data.get("stations"))

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "Calibration":
        return cls(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else cls()

    def lookup(self, station: str | None, kind: str, lead_days: int) -> tuple[dict, bool]:
        """(bias_c / sd_c for this lead, whether it is this station's own fit)."""
        lead = "day0" if lead_days <= 0 else "day1"
        own = self.stations.get(f"{station}|{kind}")
        return (own[lead], True) if own else (self.default[lead], False)


@dataclass(frozen=True)
class Params:
    center: float  # bias-corrected, in the contract's unit
    center_raw: float  # uncorrected ensemble mean, kept so the correction can be evaluated
    sd: float
    lower: float | None  # truth can't be below this (high so far)
    upper: float | None  # truth can't be above this (low so far)
    station_fit: bool
    lead_days: int
    bias: float = 0.0  # the station bias already folded into the centre, in the contract's unit

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def baseline_params(s: ContractSemantics, ensemble: EnsembleReport, nowcast: NowcastReport, calib: Calibration) -> Params:
    k = unit_scale(s.unit)
    cal, own = calib.lookup(s.station_code, s.kind, nowcast.lead_days)
    bias, sd_cal = cal["bias_c"] * k, cal["sd_c"] * k
    # The bias was fitted on the deterministic forecast, not the ensemble mean, so correcting the
    # ensemble outright could double-count. Average the two independent estimates of the center.
    corrected_det = nowcast.forecast_extreme + bias
    center = (ensemble.mean + corrected_det) / 2
    sd = max(sd_cal, ensemble.std_dev)  # honor model disagreement when it exceeds typical error

    lower = upper = None
    if nowcast.observed_extreme_so_far is not None:
        # station readings are whole degrees, so allow half a degree of rounding slack
        if s.kind == "highest":
            lower = nowcast.observed_extreme_so_far - 0.5
        else:
            upper = nowcast.observed_extreme_so_far + 0.5
    return Params(center, ensemble.mean, sd, lower, upper, own, nowcast.lead_days, bias)


def vector(s: ContractSemantics, *, center: float, sd: float, lower: float | None = None, upper: float | None = None) -> list[BracketProbability]:
    """P(bracket) for a normal(center, sd) rounded to whole degrees, truncated to [lower, upper]."""
    lo_bound = -math.inf if lower is None else lower
    hi_bound = math.inf if upper is None else upper
    z = phi((hi_bound - center) / sd) - phi((lo_bound - center) / sd)
    masses = []
    for b in s.brackets:
        lo = (b.low - 0.5) if b.low is not None else -math.inf
        hi = (b.high + 0.5) if b.high is not None else math.inf
        a, c = max(lo, lo_bound), min(hi, hi_bound)
        masses.append(max(phi((c - center) / sd) - phi((a - center) / sd), 0.0) if c > a else 0.0)
    if z < 1e-9 or sum(masses) < 1e-9:
        # the truth is already far outside the forecast: put everything on the nearest allowed bracket
        edge = lo_bound if math.isfinite(lo_bound) else hi_bound
        nearest = min(
            range(len(s.brackets)),
            key=lambda i: abs(((s.brackets[i].low if s.brackets[i].low is not None else s.brackets[i].high) or 0) - edge),
        )
        masses = [1.0 if i == nearest else 0.0 for i in range(len(masses))]
    total = sum(masses)
    return [BracketProbability(label=b.label, probability=m / total) for b, m in zip(s.brackets, masses)]


def recent_miss_adjustment(p: Params, recent_error_mean: float | None) -> float:
    """How far to move the centre for baseline+: half of last week's miss that the usual bias doesn't explain."""
    if recent_error_mean is None:
        return 0.0
    return RECENT_MISS_K * (recent_error_mean - p.bias)


def baseline_vector(s: ContractSemantics, p: Params, *, shift: float = 0.0, spread_multiplier: float = 1.0, corrected: bool = True):
    center = (p.center if corrected else p.center_raw) + shift
    return vector(s, center=center, sd=p.sd * spread_multiplier, lower=p.lower, upper=p.upper)

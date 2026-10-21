"""Data objects passed between agents.

The price wall lives here: a Contract keeps what the market is asking
(semantics) separate from what it is paying (pricing). Analysts and the
research debate only ever receive `Contract.semantics_view()`; the Trader is
the only consumer of `pricing`.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, model_validator


class Bracket(BaseModel):
    """One outcome bucket, e.g. "22°C", "62-63°F", "31°C or higher".

    Bounds are inclusive and in the contract's unit; None means open-ended.
    """

    label: str
    low: int | None
    high: int | None


class BracketPrice(BaseModel):
    label: str
    yes_token_id: str
    best_bid: float | None
    best_ask: float | None
    last_trade: float | None
    liquidity: float


class ContractSemantics(BaseModel):
    """Everything an analyst needs to answer the question. No price fields."""

    event_id: str
    city: str
    kind: Literal["highest", "lowest"]
    observation_date: date
    unit: Literal["C", "F"]
    station_name: str | None
    station_code: str | None  # ICAO (e.g. ZHHH), or "HKO" for Hong Kong Observatory
    resolution_source: str | None
    brackets: list[Bracket]
    resolves_at: datetime
    rules: str


class ContractPricing(BaseModel):
    """What the market is paying. Trader-only."""

    event_id: str
    total_liquidity: float
    volume_24h: float
    brackets: list[BracketPrice]


class Contract(BaseModel):
    semantics: ContractSemantics
    pricing: ContractPricing

    def semantics_view(self) -> ContractSemantics:
        return self.semantics


class BracketProbability(BaseModel):
    """P(observation falls in this bracket), per an analyst's model."""

    label: str
    probability: float


class EnsembleReport(BaseModel):
    """Agent 02 (Ensemble Analyst) output.

    A full probability distribution over the contract's brackets, built from
    Open-Meteo ensemble forecast members for the station's coordinates and
    observation date. No price fields — this is fed to the research debate,
    never the Trader directly.
    """

    event_id: str
    station_code: str
    observation_date: date
    model: str
    member_count: int
    member_values: list[float]
    mean: float
    std_dev: float
    bracket_probabilities: list[BracketProbability]
    model_summaries: dict[str, dict[str, float]] = {}  # per-model members / mean / std_dev
    generated_at: datetime


class MLReport(BaseModel):
    """ML Analyst output: a learned correction to the multi-model forecast. No price fields."""

    event_id: str
    station_code: str
    lead_days: int
    center: float  # predicted high, in the contract's unit
    sd: float
    multi_model_mean: float  # the weather models' own average, for comparison
    bracket_probabilities: list[BracketProbability]
    models_used: list[str]
    recent_error_mean: float | None = None  # last week's average forecast miss at the station, contract unit (a difference)
    inputs: str = "day-ahead run"  # which vintage of weather-model forecast fed the model: the previous-day run, or the newest run
    model_trained_until: str
    feature_set: str
    caveats: list[str]
    generated_at: datetime


class AlertsReport(BaseModel):
    """Alerts Analyst output: official warnings near the station. Logged for later study; no price fields."""

    event_id: str
    station_code: str
    nearest_cyclone_km: int | None
    nearest_cyclone: str | None
    cyclone_alert: str | None
    cyclone_wind_kmh: float | None
    nws_alerts: list[str]
    flags: list[str]
    generated_at: datetime


class SameDayReport(BaseModel):
    """Same-Day Analyst output: the day's high projected from what the station has already recorded."""

    event_id: str
    station_code: str
    local_hour: float
    high_so_far: float  # in the contract's unit
    remaining_forecast_high: float | None
    bias_adjustment: float
    center: float
    sd: float
    bracket_probabilities: list[BracketProbability]
    generated_at: datetime


class ResearchCase(BaseModel):
    """Agents 06/07 output: one side of the per-event debate.

    The warmer-case researcher argues the observation lands above where the
    ensemble centers; the cooler-case researcher argues below. Price-blind.
    """

    event_id: str
    side: Literal["warmer", "cooler"]
    argument: str
    favored_brackets: list[str]  # bracket labels this case argues for
    evidence: list[str]


class ProbabilityEstimate(BaseModel):
    """Agent 08 (Research Manager) output.

    A calibrated probability for every bracket of the event, summing to 1.
    Price-blind by construction: the Trader is the first agent to see prices.
    """

    event_id: str
    bracket_probabilities: list[BracketProbability]
    confidence: float
    rationale: str
    generated_at: datetime
    # How the vector was made (see baseline.py). Optional so older log records still load.
    baseline_probabilities: list[BracketProbability] | None = None
    baseline_uncorrected_probabilities: list[BracketProbability] | None = None  # no station-bias correction
    baseline_plus_probabilities: list[BracketProbability] | None = None  # baseline + a recent-miss correction
    shift: float | None = None  # the LLM's proposed shift, in the contract's unit
    spread_multiplier: float | None = None
    baseline_params: dict | None = None

    @model_validator(mode="after")
    def _is_distribution(self):
        labels = [bp.label for bp in self.bracket_probabilities]
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate bracket labels")
        if any(not 0 <= bp.probability <= 1 for bp in self.bracket_probabilities):
            raise ValueError("bracket probability outside [0, 1]")
        total = sum(bp.probability for bp in self.bracket_probabilities)
        if abs(total - 1) > 1e-3:
            raise ValueError(f"bracket probabilities sum to {total:.4f}, expected 1")
        return self


class NowcastReport(BaseModel):
    """Agent 03 (Nowcast Analyst) output. No price fields.

    Extremes are for the contract's observation date, in the contract's unit.
    `previous_run_extremes` holds the same forecast as issued 1, 2 and 3 days
    earlier, so a reader can see which way the model is drifting.
    """

    event_id: str
    station_code: str
    observation_date: date
    unit: Literal["C", "F"]
    forecast_extreme: float
    previous_run_extremes: dict[str, float]
    drift_vs_3d: float
    trend: Literal["warming", "cooling", "steady"]
    observed_extreme_so_far: float | None
    observed_through_local: str | None
    latest_observed_temp: float | None
    metar_reports_used: int
    lead_days: int = 1  # observation date minus the station's local date today; 0 = same day
    caveats: list[str]
    generated_at: datetime


class ClimatologyReport(BaseModel):
    """Agent 04 (Climatology Analyst) output. No price fields.

    Base rate for the station and calendar window from 30 years of ERA5
    reanalysis, as a probability for each bracket.
    """

    event_id: str
    station_code: str
    observation_date: date
    kind: Literal["highest", "lowest"]
    unit: Literal["C", "F"]
    years_used: int
    window_days: int
    sample_size: int
    mean: float
    std_dev: float
    p10: float
    p50: float
    p90: float
    recent_10yr_mean: float
    bracket_probabilities: list[BracketProbability]
    caveats: list[str]
    generated_at: datetime


class LocalContextReport(BaseModel):
    """Agent 05 (Local Research Analyst) output. No price fields."""

    event_id: str
    station_code: str
    sources: list[str]
    summary: str
    leans: Literal["warmer", "cooler", "neutral", "no_information"]
    local_only_warnings: list[str]
    caveats: list[str]
    generated_at: datetime

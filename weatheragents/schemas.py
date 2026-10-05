"""Data objects passed between agents.

The price wall lives here: a Contract keeps what the market is asking
(semantics) separate from what it is paying (pricing). Analysts and the
research debate only ever receive `Contract.semantics_view()`; the Trader is
the only consumer of `pricing`.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel


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
    generated_at: datetime

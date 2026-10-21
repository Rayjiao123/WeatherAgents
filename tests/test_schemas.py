from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from weatheragents.schemas import BracketProbability, ProbabilityEstimate


def _estimate(probs):
    return ProbabilityEstimate(
        event_id="1",
        bracket_probabilities=[BracketProbability(label=l, probability=p) for l, p in probs],
        confidence=0.6,
        rationale="test",
        generated_at=datetime.now(timezone.utc),
    )


def test_valid_distribution():
    est = _estimate([("21°C", 0.08), ("22°C", 0.47), ("23°C", 0.34), ("24°C", 0.09), ("25°C or higher", 0.02)])
    assert len(est.bracket_probabilities) == 5


def test_must_sum_to_one():
    with pytest.raises(ValidationError):
        _estimate([("22°C", 0.5), ("23°C", 0.3)])


def test_rejects_duplicate_labels():
    with pytest.raises(ValidationError):
        _estimate([("22°C", 0.5), ("22°C", 0.5)])


def test_rejects_out_of_range():
    with pytest.raises(ValidationError):
        _estimate([("22°C", 1.2), ("23°C", -0.2)])

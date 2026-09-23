"""Tests for outcome evidence: counts, decay and the score they give.

The properties are checked over a grid of counts rather than a few examples,
since they are what the rest of the system relies on.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from runtime_memory.core.models import Memory, MemoryCategory, Outcome
from runtime_memory.core.outcomes import EVIDENCE_OF, OutcomeModel, pseudo_counts

COUNTS = [0.0, 0.25, 1.0, 2.0, 5.0, 10.0, 100.0]
MODEL = OutcomeModel()
NOW = datetime(2026, 9, 24, tzinfo=UTC)


def _memory(worked: float = 0.0, failed: float = 0.0, days_ago: float | None = None) -> Memory:
    memory = Memory(content="m", category=MemoryCategory.PATTERN, worked=worked, failed=failed)
    if days_ago is not None:
        memory.evidence_at = NOW - timedelta(days=days_ago)
    return memory


class TestScore:
    """The score of a pair of counts."""

    @pytest.mark.parametrize(("worked", "failed"), itertools.product(COUNTS, COUNTS))
    def test_stays_strictly_inside_minus_one_to_one(self, worked: float, failed: float) -> None:
        assert -1.0 < MODEL.score(worked, failed) < 1.0

    def test_no_evidence_is_neutral(self) -> None:
        assert MODEL.score(0.0, 0.0) == 0.0

    @pytest.mark.parametrize(("worked", "failed"), itertools.product(COUNTS, COUNTS))
    def test_a_success_never_lowers_it_and_a_failure_never_raises_it(
        self, worked: float, failed: float
    ) -> None:
        score = MODEL.score(worked, failed)
        assert MODEL.score(worked + 1, failed) > score
        assert MODEL.score(worked, failed + 1) < score

    @pytest.mark.parametrize("n", [1.0, 2.0, 5.0, 10.0])
    def test_more_of_the_same_evidence_reads_stronger(self, n: float) -> None:
        """One observation should not read like ten."""
        assert MODEL.score(n, 0) < MODEL.score(n + 1, 0)
        assert MODEL.score(0, n) > MODEL.score(0, n + 1)

    def test_a_failure_outweighs_a_success(self) -> None:
        assert MODEL.score(1.0, 1.0) < 0.0
        assert abs(MODEL.score(0.0, 1.0)) > MODEL.score(1.0, 0.0)

    def test_the_documented_values(self) -> None:
        assert MODEL.score(1, 0) == pytest.approx(1 / 3)
        assert MODEL.score(10, 0) == pytest.approx(10 / 12)
        assert MODEL.score(0, 1) == pytest.approx(-1.5 / 3.5)
        assert MODEL.score(0, 2) == pytest.approx(-0.6)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"half_life_days": 0}, "half_life_days"),
            ({"prior": 0}, "prior"),
            ({"failure_weight": -1}, "failure_weight"),
        ],
    )
    def test_bad_parameters_are_rejected(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            OutcomeModel(**kwargs)


class TestDecay:
    """Counts halve every half-life."""

    @pytest.mark.parametrize("half_lives", [0, 1, 2, 3.5])
    def test_halves_per_half_life(self, half_lives: float) -> None:
        memory = _memory(worked=8.0, failed=4.0, days_ago=90 * half_lives)

        worked, failed = MODEL.evidence(memory, NOW)

        assert worked == pytest.approx(8.0 * 0.5**half_lives)
        assert failed == pytest.approx(4.0 * 0.5**half_lives)

    def test_evidence_with_no_date_is_taken_as_is(self) -> None:
        assert MODEL.evidence(_memory(worked=3.0), NOW) == (3.0, 0.0)

    def test_a_future_date_does_not_grow_the_counts(self) -> None:
        assert MODEL.evidence(_memory(worked=3.0, days_ago=-10), NOW) == (3.0, 0.0)

    @pytest.mark.parametrize(("worked", "failed"), [(5.0, 0.0), (0.0, 5.0), (4.0, 1.0)])
    def test_fading_evidence_moves_the_score_toward_neutral(
        self, worked: float, failed: float
    ) -> None:
        fresh = MODEL.score(*MODEL.evidence(_memory(worked, failed, days_ago=0), NOW))
        old = MODEL.score(*MODEL.evidence(_memory(worked, failed, days_ago=365), NOW))

        assert abs(old) < abs(fresh)
        assert (old > 0) == (fresh > 0)


class TestAdd:
    """Adding evidence to a memory."""

    def test_old_counts_are_decayed_before_new_ones_are_added(self) -> None:
        memory = _memory(worked=4.0, days_ago=90)

        worked, failed, score = MODEL.add(memory, 1.0, 0.0, NOW)

        assert (worked, failed) == pytest.approx((3.0, 0.0))
        assert score == pytest.approx(MODEL.score(3.0, 0.0))

    @pytest.mark.parametrize("outcome", list(Outcome))
    def test_every_outcome_adds_evidence(self, outcome: Outcome) -> None:
        worked, failed = EVIDENCE_OF[outcome]
        assert worked + failed > 0


class TestPseudoCounts:
    """3.x scores read back as the counts they stand for."""

    @pytest.mark.parametrize(
        ("score", "counts"),
        [(0.0, (0.0, 0.0)), (0.2, (1.0, 0.0)), (1.0, (5.0, 0.0)), (-0.3, (0.0, 1.0)), (-0.6, (0.0, 2.0))],
    )
    def test_steps_become_counts(self, score: float, counts: tuple[float, float]) -> None:
        assert pseudo_counts(score) == pytest.approx(counts)

    @pytest.mark.parametrize("score", [-1.0, -0.6, -0.3, 0.0, 0.2, 0.6, 1.0])
    def test_the_sign_survives(self, score: float) -> None:
        migrated = MODEL.score(*pseudo_counts(score))
        assert (migrated > 0) == (score > 0)
        assert (migrated < 0) == (score < 0)

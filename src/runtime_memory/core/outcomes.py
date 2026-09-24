"""How recorded outcomes become evidence, and evidence becomes a score.

Before 4.0.0 an outcome moved a memory's score by a fixed step, +0.2 worked,
-0.3 failed, +0.05 partial, clamped to [-1, 1]. One observation then read like a
settled record, nothing aged, and a memory at the floor lost nothing more from
another failure. In the Tier 2 evaluation a memory failed on every use and kept
its place that way.

A memory now keeps decayed counts of the times it worked and the times it failed.
Its outcome score is their difference over their total plus a prior, so a single
observation moves it a little and ten move it a lot, and every count halves over
``half_life_days``. The counts are what an agent is shown, as "worked 3 times,
failed 1 time", because a score alone does not say how much evidence is behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from runtime_memory.core.models import Outcome

if TYPE_CHECKING:
    from datetime import datetime

    from runtime_memory.core.models import Memory

EVIDENCE_OF: dict[Outcome, tuple[float, float]] = {
    Outcome.WORKED: (1.0, 0.0),
    Outcome.FAILED: (0.0, 1.0),
    # A quarter of a success, the ratio the 3.x steps gave partial to worked.
    Outcome.PARTIAL: (0.25, 0.0),
}
"""What one recorded outcome adds, as (worked, failed) observations."""


@dataclass(frozen=True)
class OutcomeModel:
    """Turns worked and failed counts into a score in [-1, 1]."""

    half_life_days: float = 90.0
    """Days over which a count halves. Evidence about a project goes stale as the
    project changes; a quarter of a year keeps a season's record at full weight
    and lets last year's fade."""

    failure_weight: float = 1.5
    """How many successes one failure is worth. Advice that misleads costs more
    than advice that was not needed, so a memory that failed once needs more than
    one success to read as reliable again. 1.5 keeps the 3.x ratio of 0.3 to 0.2."""

    prior: float = 2.0
    """Pseudo-observations at neutral. A single success scores 1/3 and ten score
    10/12, where 3.x gave 0.2 and a clamped 1.0."""

    def __post_init__(self) -> None:
        """Check the parameters.

        Raises:
            ValueError: If the half-life or prior is not positive, or the failure
                weight is negative.
        """
        if self.half_life_days <= 0:
            raise ValueError(f"half_life_days must be positive, got {self.half_life_days}")
        if self.prior <= 0:
            raise ValueError(f"prior must be positive, got {self.prior}")
        if self.failure_weight < 0:
            raise ValueError(f"failure_weight must not be negative, got {self.failure_weight}")

    def decayed(self, count: float, since: datetime | None, now: datetime) -> float:
        """A count as it stands at ``now``, having halved every half-life since ``since``."""
        if count == 0 or since is None:
            return count
        days = max((now - since).total_seconds() / 86400, 0.0)
        return count * math.pow(0.5, days / self.half_life_days)

    def evidence(self, memory: Memory, now: datetime) -> tuple[float, float]:
        """A memory's worked and failed counts, decayed to ``now``."""
        return (
            self.decayed(memory.worked, memory.evidence_at, now),
            self.decayed(memory.failed, memory.evidence_at, now),
        )

    def score(self, worked: float, failed: float) -> float:
        """The outcome score of these counts: (worked - weighted failed) / (total + prior)."""
        weighted_failed = self.failure_weight * failed
        return (worked - weighted_failed) / (worked + weighted_failed + self.prior)

    def add(
        self, memory: Memory, worked: float, failed: float, now: datetime
    ) -> tuple[float, float, float]:
        """A memory's counts after adding evidence, and its score from them.

        The stored counts are first decayed to ``now``, so the stored pair and
        ``evidence_at`` together always describe the record as of that moment.

        Returns:
            (worked, failed, score).
        """
        current_worked, current_failed = self.evidence(memory, now)
        new_worked = current_worked + worked
        new_failed = current_failed + failed
        return new_worked, new_failed, self.score(new_worked, new_failed)


def pseudo_counts(outcome_score: float) -> tuple[float, float]:
    """The counts a 3.x outcome score stands for, used to migrate stored scores.

    3.x moved the score +0.2 per success and -0.3 per failure, so a positive
    score is read as that many successes and a negative one as that many
    failures. The clamp and any mix of the two are lost; the sign and the
    rough weight of the record survive.
    """
    if outcome_score > 0:
        return outcome_score / 0.2, 0.0
    if outcome_score < 0:
        return 0.0, -outcome_score / 0.3
    return 0.0, 0.0

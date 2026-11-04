"""A genome's score on observed outcomes: a mean and its standard error."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Score:
    """Mean outcome and the standard error of that mean.

    `disqualified` is a hard veto (guardrail), independent of the mean.
    A disqualified genome is scored so the log stays honest, and it cannot
    be promoted.
    """

    mean: float
    se: float
    n: int = 0
    disqualified: bool = False
    detail: dict = field(default_factory=dict)


class Evaluator(Protocol):
    """Score genomes from observed outcomes.

    `outcomes` is domain data: a table of labelled rows, a window of page
    metrics, a replay of past decisions. The core never inspects it.
    """

    def score(
        self, genomes: Sequence[Any], outcomes: Any = None
    ) -> Mapping[str, Score]:
        ...

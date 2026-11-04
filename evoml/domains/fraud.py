"""Fraud-bench adapter.

The bench keeps its own fitting loop (balanced weights, purged time split,
bootstrap standard errors). Promotion is the core rule in `max` mode: the
challenger must beat the incumbent by more than the larger of the two
standard errors, which is the comparison the bench already ran.
"""
from __future__ import annotations

import json
import random
from typing import Any

from evoml.core.arena import beats
from evoml.core.evaluator import Score
from evoml.core.genome import Genome


def consider_succession(
    incumbent_mean: float,
    incumbent_se: float,
    challenger_mean: float,
    challenger_se: float,
    *,
    se_mult: float = 1.0,
) -> bool:
    """True when the challenger clears the bench's one-SE gate."""
    return beats(
        Score(challenger_mean, challenger_se),
        Score(incumbent_mean, incumbent_se),
        se_mult=se_mult,
        mode="max",
    )


class FraudGenome(Genome):
    """Wraps a fraud-bench genome dict. `n_feat` is the column count."""

    def __init__(self, data: dict, n_feat: int):
        self.data = json.loads(json.dumps(data))
        self.n_feat = int(n_feat)

    def mutate(self, rng: random.Random) -> FraudGenome:
        from bench.fraud_bench import mutate

        return FraudGenome(mutate(self.data, rng, self.n_feat), self.n_feat)

    @classmethod
    def random(cls, rng: random.Random, n_feat: int = 29, kind: str | None = None) -> FraudGenome:
        from bench.fraud_bench import random_genome

        return cls(random_genome(rng, n_feat, kind=kind), n_feat)

    def serialize(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data))

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> FraudGenome:
        raise TypeError("FraudGenome.deserialize needs n_feat; use FraudGenome(data, n_feat)")

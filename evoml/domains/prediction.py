"""Crypto-direction adapter.

The genome is the forecaster genome in `memescalp.mlpred`. Succession uses
the core rule with the shared holdout standard error the live loop already
uses (`sqrt(0.25 / n)`), so a challenger still has to clear that noise
floor and a tie still stays with the incumbent.
"""
from __future__ import annotations

import json
import random
from typing import Any

from evoml.core.arena import Arena, Contender
from evoml.core.evaluator import Score
from evoml.core.genome import Genome


class PredictionGenome(Genome):
    """Wraps a memescalp genome dict. Mutation delegates to `mutate_genome`."""

    def __init__(self, data: dict):
        import memescalp.mlpred as mlp

        self.data = mlp._ensure_genome_defaults(json.loads(json.dumps(data)))

    def mutate(self, rng: random.Random, report: list[float] | None = None) -> PredictionGenome:
        import memescalp.mlpred as mlp

        return PredictionGenome(mlp.mutate_genome(self.data, rng, report))

    @classmethod
    def random(cls, rng: random.Random, exclude_kind: str | None = None) -> PredictionGenome:
        import memescalp.mlpred as mlp

        return cls(mlp.random_genome(rng, exclude_kind=exclude_kind))

    def crossover(self, other: Genome, rng: random.Random) -> PredictionGenome:
        import memescalp.mlpred as mlp

        if not isinstance(other, PredictionGenome):
            raise TypeError("crossover expects another PredictionGenome")
        base = self.serialize() if rng.random() < 0.5 else other.serialize()
        child = json.loads(json.dumps(base))
        feats = set(int(i) for i in self.data.get("features", []))
        for feature in other.data.get("features", []):
            feature = int(feature)
            if rng.random() < 0.5:
                feats.add(feature)
            elif feature in feats and len(feats) > mlp.MIN_FEATURE_GENES:
                feats.discard(feature)
        child["features"] = sorted(feats)
        if rng.random() < 0.5:
            child["synth"] = json.loads(json.dumps(other.data.get("synth", [])))
        return PredictionGenome(child)

    def serialize(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data))

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> PredictionGenome:
        return cls(data)


def select_champion(
    scores: dict[str, float],
    roles: dict[str, str],
    incumbent_key: str | None,
    noise: float,
) -> tuple[str, str]:
    """Core succession for the live forecaster.

    `noise` is the shared holdout standard error already scaled by
    `SUCCESSION_SE_MULT`. Mode `shared` compares the margin to that one
    floor. Growth bias stays in the forecaster: a bigger body within the
    noise is a domain rule, not a core rule.
    """
    if not scores:
        raise ValueError("select_champion needs at least one score")
    score_map = {
        key: Score(mean=value, se=noise, n=0) for key, value in scores.items()
    }
    contenders = [
        Contender(key=key, role=roles.get(key, "extra")) for key in scores
    ]
    arena = Arena(
        name="prediction",
        se_mult=1.0,
        se_mode="shared",
        require_beat_control=False,
        cooldown_generations=0,
    )
    decision = arena.decide(
        contenders,
        score_map,
        generation=1,
        incumbent_key=incumbent_key if incumbent_key in scores else None,
    )
    return decision.winner_key, decision.succession

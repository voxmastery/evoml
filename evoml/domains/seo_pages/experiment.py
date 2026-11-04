"""Run the page-recipe arena.

The population is the reigning recipe, two mutants, one fresh challenger
and the untouched-page control. Promotion uses the core one-SE rule and
also has to beat the control. Guardrail failures are blacklisted.
"""
from __future__ import annotations

import random
from pathlib import Path

from evoml.core.arena import Arena, Contender, Decision
from evoml.core.audit import AuditLog
from evoml.core.memory import MemoryHook, NullMemory

from .evaluator import PageEvaluator, RecipeSimulator
from .genome import PageRecipe
from .signals import METRIC_NAMES


DEFAULT_WEIGHTS = {name: 1.0 for name in METRIC_NAMES}


class SEOExperiment:
    def __init__(
        self,
        *,
        seed: int = 0,
        generations: int = 8,
        weights: dict[str, float] | None = None,
        cohort_size: int = 8,
        mut_rate: float = 1.5,
        se_mult: float = 1.0,
        cooldown_generations: int = 0,
        audit_path: str | Path | None = None,
        memory: MemoryHook | None = None,
        verified: bool = False,
        champion: PageRecipe | None = None,
        simulator: RecipeSimulator | None = None,
    ):
        self.seed = seed
        self.generations = generations
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.cohort_size = cohort_size
        self.mut_rate = mut_rate
        self.verified = verified
        self.champion = champion or PageRecipe.weak()
        self.champion.mut_rate = mut_rate
        self.simulator = simulator or RecipeSimulator(seed=seed)
        self.evaluator = PageEvaluator(
            weights=self.weights,
            cohort_size=cohort_size,
            simulator=self.simulator,
        )
        self.audit = AuditLog(audit_path) if audit_path is not None else None
        self.memory = memory if memory is not None else NullMemory()
        self.arena = Arena(
            name="seo_pages",
            se_mult=se_mult,
            se_mode="difference",
            n_variants=2,
            cooldown_generations=cooldown_generations,
            require_beat_control=True,
            audit=self.audit,
            memory=self.memory,
        )
        self.journal: list[Decision] = []

    def run(self, generations: int | None = None) -> list[Decision]:
        rng = random.Random(self.seed)
        report = _signal_report(self.simulator, self.cohort_size)
        n = self.generations if generations is None else generations
        self.journal = []
        for generation in range(1, n + 1):
            variants = [
                self.champion.mutate(rng, report, rate=self.mut_rate)
                for _ in range(2)
            ]
            challenger = PageRecipe.random(rng)
            control = PageRecipe.control()
            contenders = _dedupe(
                [
                    Contender(self.champion.key(), "champion", self.champion),
                    *[Contender(variant.key(), "variant", variant) for variant in variants],
                    Contender(challenger.key(), "challenger", challenger),
                    Contender("control", "control", control),
                ]
            )
            scores = self.evaluator.score(
                [c.genome for c in contenders if c.role != "control"]
            )
            decision = self.arena.decide(
                contenders,
                scores,
                generation=generation,
                incumbent_key=self.champion.key(),
            )
            winner = next(
                (c.genome for c in contenders if c.key == decision.winner_key),
                self.champion,
            )
            self.arena.commit(
                decision,
                genome=winner,
                verified=self.verified,
                recall_cue="seo page recipe " + decision.succession,
            )
            if decision.promoted and isinstance(winner, PageRecipe) and not winner.is_control:
                self.champion = winner
            self.journal.append(decision)
        return self.journal


def _dedupe(contenders: list[Contender]) -> list[Contender]:
    seen: set[str] = set()
    out: list[Contender] = []
    for contender in contenders:
        if contender.key in seen:
            continue
        seen.add(contender.key)
        out.append(contender)
    return out


def _signal_report(simulator: RecipeSimulator, cohort_size: int) -> dict[str, float]:
    """Correlation of each raw signal with bookings, over a spread of recipes.

    The report is what `invent_gene` reads. It is computed from simulated
    observations of random recipes plus the weak champion, not from a
    hand-set ranking.
    """
    rng = random.Random(simulator.seed + 17)
    recipes = [PageRecipe.weak()] + [PageRecipe.random(rng) for _ in range(12)]
    rows = []
    for recipe in recipes:
        rows.extend(simulator.observations(recipe, cohort_size=max(4, cohort_size // 2)))
    return report_from_rows(rows)


def report_from_rows(rows: list) -> dict[str, float]:
    """Pearson correlation of each signal column with booking starts."""
    from .genome import SIGNAL_NAMES

    if len(rows) < 5:
        return {name: 0.0 for name in SIGNAL_NAMES}
    target = [float(row.booking_starts) for row in rows]
    report = {}
    for name in SIGNAL_NAMES:
        xs = []
        for row in rows:
            if name == "position_delta":
                xs.append(float(row.position_delta))
            else:
                xs.append(float(getattr(row, name)))
        report[name] = _pearson(xs, target)
    # A perfect self-correlation would make invention ignore every other
    # signal. Keep booking_starts informative but not infinite.
    report["booking_starts"] = min(0.99, report["booking_starts"])
    return report


def _pearson(xs: list[float], ys: list[float]) -> float:
    import math

    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


__all__ = ["SEOExperiment"]

"""Score page recipes from observed rows or from the simulator."""
from __future__ import annotations

import random
import zlib
from typing import Sequence

from evoml.core.evaluator import Score

from .genome import PageRecipe, quality
from .guardrails import rates, spiked
from .signals import (
    METRIC_NAMES,
    PageObservation,
    composite,
    mean_se,
)


def _limit_pages(rows: list[PageObservation], cohort_size: int) -> list[PageObservation]:
    pages = sorted({row.page_id for row in rows})[:cohort_size]
    keep = set(pages)
    return [row for row in rows if row.page_id in keep]


def score_rows(
    rows: list[PageObservation],
    weights: dict[str, float],
    *,
    control_rows: list[PageObservation] | None = None,
) -> Score:
    """Mean composite across pages, plus a guardrail veto against `control_rows`."""
    vectors_by_page: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        vectors_by_page.setdefault(row.page_id, []).append(row.vector())
    values = []
    for page_id in sorted(vectors_by_page):
        acc = {name: 0.0 for name in METRIC_NAMES}
        group = vectors_by_page[page_id]
        for vector in group:
            for name in METRIC_NAMES:
                acc[name] += vector[name]
        n = float(len(group))
        values.append(composite({name: acc[name] / n for name in METRIC_NAMES}, weights))
    mean, se, n = mean_se(values)
    reason = None
    if control_rows is not None:
        reason = spiked(rates(rows), rates(control_rows))
    return Score(
        mean=mean,
        se=se,
        n=n,
        disqualified=reason is not None,
        detail={"guardrail": reason} if reason else {},
    )


def _seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts)
    return zlib.crc32(raw.encode()) & 0xFFFFFFFF


class RecipeSimulator:
    """Counterfactual outcomes for a recipe that has not been deployed.

    Tests and the dry-run use this. Measured runs use `load_csv` / `load_json`
    and never call it. No network.
    """

    def __init__(self, seed: int = 0, noise: float = 0.04):
        self.seed = seed
        self.noise = noise

    def observations(
        self,
        recipe: PageRecipe,
        *,
        cohort_size: int,
        day: int = 0,
    ) -> list[PageObservation]:
        level = quality(recipe)
        spike = recipe.causes_spike()
        return self._rows(
            key=recipe.key(),
            level=level,
            cohort_size=cohort_size,
            day=day,
            spike=spike,
            ctr_boost=0.004 * level,
        )

    def control_observations(self, cohort_size: int, *, day: int = 0) -> list[PageObservation]:
        return self._rows(
            key="control",
            level=1.0,
            cohort_size=cohort_size,
            day=day,
            spike=False,
            ctr_boost=0.0,
        )

    def _rows(
        self,
        *,
        key: str,
        level: float,
        cohort_size: int,
        day: int,
        spike: bool,
        ctr_boost: float,
    ) -> list[PageObservation]:
        rng = random.Random(_seed(self.seed, key, day, cohort_size))
        rows = []
        for index in range(cohort_size):
            bookings = max(0.0, level + rng.gauss(0, self.noise))
            ctr = max(0.0, 0.03 + ctr_boost + rng.gauss(0, self.noise / 20))
            rows.append(
                PageObservation(
                    page_id=f"{key[:24]}-{index}",
                    day=day,
                    group=key,
                    clicks=ctr * 100.0,
                    impressions=100.0,
                    ctr=ctr,
                    position=8.0 - min(4.0, level),
                    position_delta=-min(2.0, level / 4.0),
                    scroll_depth=min(1.0, 0.35 + 0.05 * level),
                    heatmap_cta=min(1.0, 0.15 + 0.04 * level),
                    time_on_page=40.0 + 8.0 * level,
                    booking_starts=bookings,
                    http_404=0.25 if spike else 0.0,
                    deindexed=0.08 if spike else 0.0,
                    noindex=0.08 if spike else 0.0,
                )
            )
        return rows


class PageEvaluator:
    """Evaluator for the page-recipe arena.

    `weights`, `cohort_size` and the window are the knobs the meta-genome
    tunes. On the simulator every recipe can be scored, including a mutant
    that has never been deployed. On a table of measured rows, only groups
    that actually appear are scored; missing groups are omitted.
    """

    def __init__(
        self,
        *,
        weights: dict[str, float],
        cohort_size: int = 8,
        simulator: RecipeSimulator | None = None,
        observations: Sequence[PageObservation] | None = None,
        window: tuple[int, int] | None = None,
    ):
        self.weights = dict(weights)
        self.cohort_size = int(cohort_size)
        self.simulator = simulator
        self.observations = list(observations or [])
        self.window = window

    def score(self, genomes: Sequence[PageRecipe], outcomes: object = None) -> dict[str, Score]:
        del outcomes
        if self.simulator is not None:
            return self._score_simulated(genomes)
        return self._score_measured(genomes)

    def _score_simulated(self, genomes: Sequence[PageRecipe]) -> dict[str, Score]:
        control_rows = self.simulator.control_observations(self.cohort_size)  # type: ignore[union-attr]
        scores = {"control": score_rows(control_rows, self.weights)}
        for recipe in genomes:
            if recipe.is_control or recipe.key() == "control":
                continue
            rows = self.simulator.observations(recipe, cohort_size=self.cohort_size)  # type: ignore[union-attr]
            scores[recipe.key()] = score_rows(rows, self.weights, control_rows=control_rows)
        return scores

    def _score_measured(self, genomes: Sequence[PageRecipe]) -> dict[str, Score]:
        rows = self._windowed(self.observations)
        control_rows = _limit_pages([r for r in rows if r.group == "control"], self.cohort_size)
        scores: dict[str, Score] = {}
        if control_rows:
            scores["control"] = score_rows(control_rows, self.weights)
        for recipe in genomes:
            key = "control" if recipe.is_control else recipe.key()
            if key == "control":
                continue
            group = _limit_pages([r for r in rows if r.group == key], self.cohort_size)
            if not group:
                continue
            scores[key] = score_rows(
                group, self.weights, control_rows=control_rows or None
            )
        return scores

    def _windowed(self, rows: list[PageObservation]) -> list[PageObservation]:
        if self.window is None:
            return rows
        start, end = self.window
        return [row for row in rows if start <= row.day < end]

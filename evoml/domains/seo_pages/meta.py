"""Meta-genome: the SEO adapter tuned by the same arena.

The genes are the score weights, the evaluation window, the mutation
rate, the cohort size and the promotion threshold (`se_mult`). A
meta-genome is scored by replaying past page assignments: at each
decision day it may promote a recipe only when the margin beats its own
threshold, then it is judged on that recipe's *later* booking starts.
The control is the untouched pages over the same payoff windows.

A meta-genome that promotes a recipe which then coincides with a 404,
deindex or noindex spike is blacklisted.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from evoml.core.arena import Arena, Contender, Decision, beats
from evoml.core.audit import AuditLog
from evoml.core.evaluator import Score
from evoml.core.genome import Genome
from evoml.core.memory import MemoryHook, NullMemory

from .evaluator import score_rows
from .genome import PageRecipe
from .guardrails import rates, spiked
from .signals import METRIC_NAMES, PageObservation, mean_se


WINDOW_CHOICES = (14, 21, 28)
PAYOFF_DAYS = 14
DECISION_DAYS = (14, 28, 42)


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    clean = {name: max(0.0, float(weights.get(name, 0.0))) for name in METRIC_NAMES}
    total = sum(clean.values())
    if total <= 0:
        return {name: 1.0 / len(METRIC_NAMES) for name in METRIC_NAMES}
    return {name: clean[name] / total for name in METRIC_NAMES}


@dataclass
class MetaGenome(Genome):
    weights: dict[str, float]
    window_days: int = 21
    mut_rate: float = 1.0
    cohort_size: int = 8
    se_mult: float = 1.0
    control_policy: bool = False

    def __post_init__(self) -> None:
        self.weights = _normalise(self.weights)
        self.window_days = _snap_window(self.window_days)
        self.mut_rate = min(3.0, max(0.5, float(self.mut_rate)))
        self.cohort_size = int(min(24, max(2, self.cohort_size)))
        self.se_mult = min(3.0, max(0.5, float(self.se_mult)))

    def mutate(self, rng: random.Random, report: dict[str, float] | None = None) -> MetaGenome:
        child = self.clone()
        child.control_policy = False
        # One step toward the signal that has actually lined up with later
        # bookings, then `mut_rate` further edits. Same idea as the
        # forecaster reading its gene report before it edits.
        if report:
            target = max(METRIC_NAMES, key=lambda name: report.get(name, 0.0))
            child._shift(target, 0.45)
        edits = max(1, round(child.mut_rate))
        for _ in range(edits):
            op = rng.choice(["weight", "weight", "window", "rate", "cohort", "threshold"])
            if op == "weight":
                if report:
                    target = max(METRIC_NAMES, key=lambda name: report.get(name, 0.0))
                else:
                    target = rng.choice(METRIC_NAMES)
                child._shift(target, 0.12)
            elif op == "window":
                child.window_days = rng.choice(WINDOW_CHOICES)
            elif op == "rate":
                child.mut_rate *= rng.choice([1 / 1.3, 1.3])
            elif op == "cohort":
                child.cohort_size += rng.choice([-2, 2])
            else:
                child.se_mult += rng.choice([-0.25, 0.25])
        child.__post_init__()
        return child

    def _shift(self, target: str, amount: float) -> None:
        donor = max(self.weights, key=self.weights.get)
        if donor == target:
            return
        shift = min(amount, max(0.0, self.weights[donor] - 0.02))
        self.weights[donor] -= shift
        self.weights[target] = self.weights.get(target, 0.0) + shift
        self.weights = _normalise(self.weights)

    @classmethod
    def random(cls, rng: random.Random) -> MetaGenome:
        dominant = rng.choice(METRIC_NAMES)
        weights = {name: 0.05 for name in METRIC_NAMES}
        weights[dominant] = 1.0
        return cls(
            weights=weights,
            window_days=rng.choice(WINDOW_CHOICES),
            mut_rate=rng.choice([0.5, 1.0, 1.5, 2.0]),
            cohort_size=rng.choice([4, 8, 12]),
            se_mult=rng.choice([0.5, 1.0, 1.5]),
        )

    @classmethod
    def ctr_start(cls) -> MetaGenome:
        """A starting champion that trusts click-through and a short window."""
        weights = {name: 0.02 for name in METRIC_NAMES}
        weights["ctr"] = 0.88
        return cls(weights=weights, window_days=14, mut_rate=1.0, cohort_size=8, se_mult=1.0)

    @classmethod
    def control(cls) -> MetaGenome:
        genome = cls.ctr_start()
        genome.control_policy = True
        return genome

    def clone(self) -> MetaGenome:
        return MetaGenome.deserialize(self.serialize())

    def proposals(self, rng: random.Random) -> list[PageRecipe]:
        """Page-recipe variants this policy would try. Count follows mut_rate."""
        parent = PageRecipe.weak()
        parent.mut_rate = self.mut_rate
        return [parent.mutate(rng, rate=self.mut_rate) for _ in range(max(1, round(self.mut_rate)))]

    def crossover(self, other: Genome, rng: random.Random) -> MetaGenome:
        if not isinstance(other, MetaGenome):
            raise TypeError("crossover expects another MetaGenome")
        weights = {}
        for name in METRIC_NAMES:
            weights[name] = self.weights[name] if rng.random() < 0.5 else other.weights[name]
        return MetaGenome(
            weights=weights,
            window_days=self.window_days if rng.random() < 0.5 else other.window_days,
            mut_rate=(self.mut_rate + other.mut_rate) / 2,
            cohort_size=self.cohort_size if rng.random() < 0.5 else other.cohort_size,
            se_mult=(self.se_mult + other.se_mult) / 2,
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "weights": {name: round(self.weights[name], 6) for name in METRIC_NAMES},
            "window_days": int(self.window_days),
            "mut_rate": round(float(self.mut_rate), 6),
            "cohort_size": int(self.cohort_size),
            "se_mult": round(float(self.se_mult), 6),
            "control_policy": bool(self.control_policy),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> MetaGenome:
        return cls(
            weights=dict(data["weights"]),
            window_days=int(data.get("window_days", 21)),
            mut_rate=float(data.get("mut_rate", 1.0)),
            cohort_size=int(data.get("cohort_size", 8)),
            se_mult=float(data.get("se_mult", 1.0)),
            control_policy=bool(data.get("control_policy", False)),
        )

    def key(self) -> str:
        if self.control_policy:
            return "control"
        return super().key()


def _snap_window(days: int) -> int:
    return min(WINDOW_CHOICES, key=lambda choice: abs(choice - int(days)))


@dataclass
class Replay:
    """One meta-genome's decisions and the later outcomes they led to."""

    choices: list[str] = field(default_factory=list)
    later: list[float] = field(default_factory=list)
    spiked: bool = False
    proposal_count: int = 0


class MetaEvaluator:
    """Replay deployed groups. Optionally score extra recipe proposals.

    `history` rows use `group` in ``good``, ``trap``, ``weak``, ``control``
    (or whatever the caller deployed). The payoff is booking starts on the
    next `PAYOFF_DAYS`, which does not depend on the weights being judged.
    """

    def __init__(
        self,
        history: Sequence[PageObservation],
        *,
        groups: Sequence[str] = ("good", "trap", "weak"),
        decision_days: Sequence[int] = DECISION_DAYS,
        payoff_days: int = PAYOFF_DAYS,
        proposal_source: Any = None,
    ):
        self.history = list(history)
        self.groups = list(groups)
        self.decision_days = list(decision_days)
        self.payoff_days = payoff_days
        self.proposal_source = proposal_source

    def score(self, genomes: Sequence[MetaGenome], outcomes: object = None) -> dict[str, Score]:
        del outcomes
        scores: dict[str, Score] = {}
        for genome in genomes:
            if genome.control_policy or genome.key() == "control":
                scores["control"] = self._control_score()
                continue
            replay = self.replay(genome)
            mean, se, n = mean_se(replay.later)
            scores[genome.key()] = Score(
                mean=mean,
                se=se,
                n=n,
                disqualified=replay.spiked,
                detail={"choices": list(replay.choices), "proposals": replay.proposal_count},
            )
        if "control" not in scores:
            scores["control"] = self._control_score()
        return scores

    def replay(self, genome: MetaGenome, *, rng: random.Random | None = None) -> Replay:
        incumbent = "weak"
        replay = Replay()
        draw = rng or random.Random(zlib_seed(genome.key()))
        for day in self.decision_days:
            proposals = []
            if self.proposal_source is not None:
                proposals = genome.proposals(draw)
                replay.proposal_count += len(proposals)
            chosen, harmful = self.choose(genome, day, incumbent, proposals=proposals)
            replay.choices.append(chosen)
            replay.later.append(self._payoff(chosen, day))
            if harmful or self._payoff_spike(chosen, day):
                replay.spiked = True
            incumbent = chosen
        return replay

    def choose(
        self,
        genome: MetaGenome,
        day: int,
        incumbent: str,
        proposals: Sequence[PageRecipe] | None = None,
    ) -> tuple[str, bool]:
        """Return the group this policy would deploy, and whether it is already toxic."""
        start = day - genome.window_days
        control_rows = self._slice("control", start, day, genome.cohort_size)
        scored: dict[str, Score] = {}
        for group in self.groups:
            rows = self._slice(group, start, day, genome.cohort_size)
            if not rows:
                continue
            scored[group] = score_rows(rows, genome.weights, control_rows=control_rows)
        for recipe in proposals or []:
            if self.proposal_source is None:
                continue
            rows = self.proposal_source(recipe, start, day, genome.cohort_size)
            if not rows:
                continue
            scored[recipe.key()] = score_rows(rows, genome.weights, control_rows=control_rows)
        eligible = [key for key, score in scored.items() if not score.disqualified]
        if not eligible:
            return incumbent, True
        best = max(eligible, key=lambda key: scored[key].mean)
        if incumbent not in scored or scored[incumbent].disqualified:
            return best, False
        if best == incumbent:
            return incumbent, False
        if beats(scored[best], scored[incumbent], se_mult=genome.se_mult, mode="difference"):
            return best, False
        return incumbent, False

    def _slice(self, group: str, start: int, end: int, cohort_size: int) -> list[PageObservation]:
        rows = [row for row in self.history if row.group == group and start <= row.day < end]
        pages = sorted({row.page_id for row in rows})[:cohort_size]
        keep = set(pages)
        return [row for row in rows if row.page_id in keep]

    def _payoff(self, group: str, day: int) -> float:
        rows = [
            row
            for row in self.history
            if row.group == group and day <= row.day < day + self.payoff_days
        ]
        if not rows:
            return 0.0
        return sum(row.booking_starts for row in rows) / len(rows)

    def _payoff_spike(self, group: str, day: int) -> bool:
        rows = [
            row
            for row in self.history
            if row.group == group and day <= row.day < day + self.payoff_days
        ]
        control = [
            row
            for row in self.history
            if row.group == "control" and day <= row.day < day + self.payoff_days
        ]
        if not rows:
            return False
        return spiked(rates(rows), rates(control)) is not None

    def _control_score(self) -> Score:
        laters = [self._payoff("control", day) for day in self.decision_days]
        mean, se, n = mean_se(laters)
        return Score(mean=mean, se=se, n=n, detail={"arm": "untouched"})


def zlib_seed(text: str) -> int:
    import zlib

    return zlib.crc32(text.encode()) & 0xFFFFFFFF


def weight_report(history: Sequence[PageObservation], decision_days: Sequence[int] = DECISION_DAYS) -> dict[str, float]:
    """Which scaled metric, in a 14-day lookback, lines up with later bookings.

    Pairs are (group, decision day). The target is later booking starts,
    not the metric itself, so a metric that is high just before a collapse
    gets a negative report.
    """
    groups = sorted({row.group for row in history if row.group != "control"})
    columns = {name: [] for name in METRIC_NAMES}
    target: list[float] = []
    evaluator = MetaEvaluator(history, groups=groups, decision_days=decision_days)
    for day in decision_days:
        later_by_group = {}
        for group in groups:
            later_by_group[group] = evaluator._payoff(group, day)
        for group in groups:
            rows = evaluator._slice(group, day - 14, day, cohort_size=10_000)
            if not rows:
                continue
            # One vector: mean of page means, then each metric's scaled value
            # via a one-hot weight.
            target.append(later_by_group[group])
            for name in METRIC_NAMES:
                lone = {metric: (1.0 if metric == name else 0.0) for metric in METRIC_NAMES}
                columns[name].append(_group_metric(rows, lone))
    report = {}
    for name in METRIC_NAMES:
        report[name] = _pearson(columns[name], target)
    return report


def _group_metric(rows: list[PageObservation], weights: dict[str, float]) -> float:
    scored = score_rows(rows, weights)
    return scored.mean


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def simulate_bangalore_history(
    *,
    seed: int = 0,
    days: int = 56,
    pages: int = 8,
    noise: float = 0.05,
) -> list[PageObservation]:
    """Simulated Search Console + engagement + bookings for four cohorts.

    ``trap`` looks strong on CTR for the first 14 days and then 404s.
    ``good`` keeps booking starts up. ``weak`` and ``control`` stay flat.
    This is a data-generating process for tests, not a description of
    nammaambulance.in traffic.
    """
    rng = random.Random(seed)
    rows: list[PageObservation] = []
    for day in range(days):
        for group in ("good", "trap", "weak", "control"):
            for index in range(pages):
                rows.append(_history_row(group, day, index, rng, noise))
    return rows


def _history_row(
    group: str, day: int, index: int, rng: random.Random, noise: float
) -> PageObservation:
    late = day >= 14
    if group == "good":
        bookings, ctr, delta, http_404 = 4.2, 0.055, -0.8, 0.0
    elif group == "trap":
        if late:
            bookings, ctr, delta, http_404 = 0.15, 0.01, 2.5, 0.30
        else:
            bookings, ctr, delta, http_404 = 1.0, 0.20, -2.0, 0.0
    elif group == "weak":
        bookings, ctr, delta, http_404 = 1.3, 0.03, 0.1, 0.0
    else:
        bookings, ctr, delta, http_404 = 1.15, 0.04, 0.0, 0.0
    bookings = max(0.0, bookings + rng.gauss(0, noise))
    ctr = max(0.0, ctr + rng.gauss(0, noise / 50))
    return PageObservation(
        page_id=f"{group}-p{index:02d}",
        day=day,
        group=group,
        clicks=ctr * 200.0,
        impressions=200.0,
        ctr=ctr,
        position=6.0 + delta,
        position_delta=delta,
        scroll_depth=0.55,
        heatmap_cta=0.25,
        time_on_page=70.0,
        booking_starts=bookings,
        http_404=http_404,
        deindexed=0.05 if http_404 else 0.0,
        noindex=0.0,
    )


class MetaExperiment:
    """Evolve the SEO policy with the same arena used for page recipes."""

    def __init__(
        self,
        history: Sequence[PageObservation] | None = None,
        *,
        seed: int = 0,
        audit_path: str | Path | None = None,
        memory: MemoryHook | None = None,
        verified: bool = False,
        champion: MetaGenome | None = None,
        proposal_source: Any = None,
    ):
        self.history = list(history) if history is not None else simulate_bangalore_history(seed=seed)
        self.seed = seed
        self.verified = verified
        self.champion = champion or MetaGenome.ctr_start()
        self.evaluator = MetaEvaluator(self.history, proposal_source=proposal_source)
        self.report = weight_report(self.history)
        self.audit = AuditLog(audit_path) if audit_path is not None else None
        self.memory = memory if memory is not None else NullMemory()
        self.arena = Arena(
            name="seo_meta",
            se_mult=1.0,
            se_mode="difference",
            n_variants=2,
            cooldown_generations=0,
            require_beat_control=True,
            audit=self.audit,
            memory=self.memory,
        )
        self.journal: list[Decision] = []

    def run(self, generations: int = 6) -> list[Decision]:
        rng = random.Random(self.seed)
        self.journal = []
        for generation in range(1, generations + 1):
            variants = [self.champion.mutate(rng, self.report) for _ in range(2)]
            challenger = MetaGenome.random(rng)
            control = MetaGenome.control()
            contenders = _dedupe(
                [
                    Contender(self.champion.key(), "champion", self.champion),
                    *[Contender(v.key(), "variant", v) for v in variants],
                    Contender(challenger.key(), "challenger", challenger),
                    Contender("control", "control", control),
                ]
            )
            scores = self.evaluator.score([c.genome for c in contenders])
            # A disqualified incumbent is blacklisted inside decide and cannot
            # keep the crown. Passing its key is what triggers that veto.
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
                recall_cue="seo meta policy " + decision.succession,
            )
            if decision.promoted and isinstance(winner, MetaGenome) and not winner.control_policy:
                self.champion = winner
            self.journal.append(decision)
        return self.journal


def _dedupe(contenders: list[Contender]) -> list[Contender]:
    seen: set[str] = set()
    out = []
    for contender in contenders:
        if contender.key in seen:
            continue
        seen.add(contender.key)
        out.append(contender)
    return out

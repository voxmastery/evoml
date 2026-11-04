"""One agentic cycle: sense, recall, plan, evolve, act, log.

The planner only proposes. Promotion is the arena's one-standard-error
rule, including the control and the guardrails. The actuator defaults
to a dry run and never deploys.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from evoml.core.memory import NullMemory
from evoml.domains.seo_pages.evaluator import RecipeSimulator
from evoml.domains.seo_pages.experiment import SEOExperiment
from evoml.domains.seo_pages.genome import PageRecipe

from .actuator import SeoActuator
from .config import AgentConfig
from .goal import Goal
from .memory import CausalMemory, episode_from_cycle, open_memory
from .planner import Hypothesis, Planner, apply_mutations, make_planner
from .sensor import FileSensor, SimulatorSensor, booking_mean


@dataclass
class CycleReport:
    cycle: int
    recalled: int
    hypotheses: list[Hypothesis]
    proposal_keys: list[str]
    rejected_keys: list[str]
    verdict: str
    promoted: bool
    succession: str
    champion_sections: list[str]
    booking_mean: float | None
    control_booking_mean: float | None
    manifest: dict
    meta_booking_weight: float | None
    subgoals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "recalled": self.recalled,
            "hypotheses": [
                {"rationale": item.rationale, "mutations": list(item.mutations)}
                for item in self.hypotheses
            ],
            "proposal_keys": self.proposal_keys,
            "rejected_keys": self.rejected_keys,
            "verdict": self.verdict,
            "promoted": self.promoted,
            "succession": self.succession,
            "champion_sections": self.champion_sections,
            "booking_mean": self.booking_mean,
            "control_booking_mean": self.control_booking_mean,
            "manifest": self.manifest,
            "meta_booking_weight": self.meta_booking_weight,
            "subgoals": self.subgoals,
            "simulated": True,
        }


class AgentSession:
    """State that survives from one cron invocation to the next via audit, files and memory."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        planner: Planner | None = None,
        memory: CausalMemory | None = None,
    ):
        if config.domain != "seo_pages":
            raise ValueError(
                f"domain {config.domain!r} is not wired yet; seo_pages is the first domain"
            )
        self.config = config
        self.goal: Goal = config.goal()
        self.memory = memory if memory is not None else open_memory(config.memory, config.memory_dir)
        self.planner = planner if planner is not None else make_planner(config.planner)
        self.actuator = SeoActuator(config.output_dir)
        self.completed = 0
        self.champion = PageRecipe.weak()
        self.meta_champion = None
        self.meta_booking_weight: float | None = None
        if config.measured:
            self.experiment = None
            self.simulator = None
            self.measured_rows = FileSensor(config.signals).read()
        else:
            self.measured_rows = None
            self.experiment = SEOExperiment(
                seed=config.seed,
                generations=config.generations,
                cohort_size=config.cohort_size,
                cooldown_generations=self.goal.budget.cooldown_generations,
                audit_path=config.audit_path,
                memory=NullMemory(),
                verified=False,
                champion=self.champion,
            )
            self.simulator = self.experiment.simulator
            self.champion = self.experiment.champion

    def pages(self) -> list[str]:
        if self.config.pages:
            return list(self.config.pages)
        count = max(self.config.cohort_size, self.config.max_pages_changed)
        return [f"NA-{index:05d}" for index in range(1, count + 1)]

    def cycle(self) -> CycleReport:
        """Sense, recall, plan, evolve, act, log. One pass."""
        goal = self.goal
        subgoals = [item.name for item in goal.decompose()]
        if self.config.measured:
            sensed = list(self.measured_rows or [])
        else:
            assert self.simulator is not None
            sensor = SimulatorSensor(
                self.simulator,
                self.champion,
                cohort_size=self.config.cohort_size,
                day=self.completed,
            )
            sensed = sensor.read()
        hits = self.memory.recall(goal.metric)
        audit_rows = self._audit_rows()
        hypotheses = list(
            self.planner.propose(
                goal=goal,
                memory_hits=hits,
                audit_rows=audit_rows,
                champion=self.champion,
            )
        )
        proposals = [apply_mutations(self.champion, item.mutations) for item in hypotheses]
        decision = self._evolve(proposals)
        winner = self.champion
        manifest = self.actuator.apply(
            winner,
            self.pages(),
            dry_run=self.config.dry_run,
            max_pages=goal.budget.max_pages_changed,
        )
        rejected = [
            key for key, score in decision.scores.items() if getattr(score, "disqualified", False)
        ]
        verdict = "promoted" if decision.promoted else "held"
        if rejected and not decision.promoted:
            verdict = "held"
        blocked = _blocked_section(proposals, decision.scores, goal.budget.blacklist)
        treated_key = self.champion.key()
        treated = booking_mean(sensed, treated_key)
        if treated is None and not self.config.measured:
            treated = _mean_bookings(self.simulator, self.champion, self.config.cohort_size, self.completed)  # type: ignore[arg-type]
        control_mean = booking_mean(sensed, "control")
        if control_mean is None and self.simulator is not None:
            control_rows = self.simulator.control_observations(self.config.cohort_size, day=self.completed)
            control_mean = booking_mean(control_rows, "control")
        score = decision.scores.get(decision.winner_key)
        outcome = (
            f"sensed_booking_mean={_fmt(treated)} control_booking_mean={_fmt(control_mean)} "
            f"decision_mean={_fmt(getattr(score, 'mean', None))} "
            f"se={_fmt(getattr(score, 'se', None))} succession={decision.succession}"
        )
        episode = episode_from_cycle(
            rationales=[item.rationale for item in hypotheses],
            action=(
                f"sections={list(self.champion.sections)} "
                f"pages={manifest['pages_changed']} dry_run={manifest['dry_run']}"
            ),
            outcome=outcome,
            verdict=verdict,
            blocked_section=blocked,
        )
        source = None if not self.config.measured else self.config.signals
        self.memory.remember(episode, verified=self.config.measured, source_uri=source)
        self.completed += 1
        return CycleReport(
            cycle=self.completed,
            recalled=len(hits),
            hypotheses=hypotheses,
            proposal_keys=[recipe.key() for recipe in proposals],
            rejected_keys=rejected,
            verdict=verdict,
            promoted=bool(decision.promoted),
            succession=str(decision.succession),
            champion_sections=list(self.champion.sections),
            booking_mean=treated if self.config.measured else _mean_bookings(
                self.simulator, self.champion, self.config.cohort_size, self.completed - 1
            ),
            control_booking_mean=control_mean,
            manifest=manifest,
            meta_booking_weight=self.meta_booking_weight,
            subgoals=subgoals,
        )

    def _evolve(self, proposals: list[PageRecipe]):
        if self.config.measured or self.experiment is None:
            return _measured_hold(self.champion, proposals, self.measured_rows or [])
        self._apply_meta()
        blacklist = set(self.goal.budget.blacklist)

        def disqualify(recipe: PageRecipe) -> str | None:
            if any(section in blacklist for section in recipe.sections):
                return "blacklist"
            if recipe.cohort_id and recipe.cohort_id in blacklist:
                return "blacklist"
            return None

        journal = self.experiment.run(
            self.config.generations,
            proposals,
            generation_offset=self.completed,
            disqualify=disqualify,
        )
        self.champion = self.experiment.champion
        return journal[-1]

    def _apply_meta(self) -> None:
        if not self.config.meta or self.experiment is None:
            if self.experiment is not None:
                weights = self.experiment.evaluator.weights
                total = sum(weights.values()) or 1.0
                self.meta_booking_weight = weights.get("booking_starts", 0.0) / total
            return
        from evoml.domains.seo_pages.meta import MetaExperiment

        meta = MetaExperiment(
            seed=self.config.seed + self.completed,
            champion=getattr(self, "meta_champion", None),
        )
        meta.run(1)
        self.meta_champion = meta.champion
        weights = dict(meta.champion.weights)
        self.experiment.weights = weights
        self.experiment.evaluator.weights = weights
        self.experiment.mut_rate = meta.champion.mut_rate
        self.experiment.arena.se_mult = meta.champion.se_mult
        self.experiment.cohort_size = meta.champion.cohort_size
        self.experiment.evaluator.cohort_size = meta.champion.cohort_size
        self.meta_booking_weight = weights.get("booking_starts")

    def _audit_rows(self) -> list[dict]:
        if self.experiment is None or self.experiment.audit is None:
            return []
        return self.experiment.audit.read()


def run_cycle(config: AgentConfig, **kwargs) -> CycleReport:
    """Run a single cycle. Suitable for cron: one process, then exit."""
    return AgentSession(config, **kwargs).cycle()


def _mean_bookings(simulator: RecipeSimulator | None, recipe: PageRecipe, cohort_size: int, day: int) -> float | None:
    if simulator is None:
        return None
    rows = simulator.observations(recipe, cohort_size=cohort_size, day=day)
    if not rows:
        return None
    return sum(row.booking_starts for row in rows) / len(rows)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def _blocked_section(proposals, scores, blacklist: tuple[str, ...]) -> str | None:
    forbidden = set(blacklist)
    for recipe in proposals:
        score = scores.get(recipe.key())
        if score is None or not score.disqualified:
            continue
        for section in recipe.sections:
            if section in forbidden:
                return section
        if recipe.causes_spike():
            return "UnsafeExperiment"
    return None


class _MeasuredDecision:
    def __init__(self, champion: PageRecipe):
        self.promoted = False
        self.succession = "held (proposals have no measured rows yet)"
        self.winner_key = champion.key()
        self.scores: dict = {}
        self.blacklisted: tuple = ()


def _measured_hold(champion: PageRecipe, proposals: list[PageRecipe], rows: list) -> _MeasuredDecision:
    """A file export can score only cohorts that are already in the file.

    New proposals have no rows, so they are not promoted. The cycle still
    records the hypothesis. Promoting them would invent an outcome.
    """
    del proposals
    decision = _MeasuredDecision(champion)
    present = {row.group for row in rows}
    if champion.key() not in present:
        decision.succession = "held (champion cohort is not in the export)"
    return decision

"""A goal is a target metric plus the constraints the loop is not allowed to break."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Subgoal:
    name: str
    detail: str


@dataclass(frozen=True)
class Budget:
    """Hard limits. The arena cannot spend past these."""

    max_pages_changed: int = 8
    cooldown_generations: int = 0
    blacklist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_pages_changed < 1:
            raise ValueError("max_pages_changed must be at least 1")
        if self.cooldown_generations < 0:
            raise ValueError("cooldown_generations cannot be negative")


@dataclass(frozen=True)
class Goal:
    """One measurable target.

    `metric` is the outcome the loop is trying to move (for SEO,
    booking starts). `direction` is ``maximize`` or ``minimize``.
    The budget caps how many pages a cycle may touch, how long a
    promotion must sit, and which sections or cohort ids are forbidden.
    """

    metric: str
    direction: str = "maximize"
    constraints: tuple[str, ...] = ()
    budget: Budget = field(default_factory=Budget)

    def __post_init__(self) -> None:
        if self.direction not in ("maximize", "minimize"):
            raise ValueError("direction must be maximize or minimize")
        if not self.metric:
            raise ValueError("metric is required")

    def decompose(self) -> tuple[Subgoal, ...]:
        """The five steps of one cycle, written as subgoals the planner can read."""
        blacklist = ", ".join(self.budget.blacklist) or "none"
        return (
            Subgoal("measure", f"Read {self.metric} on treated pages and on the untouched control."),
            Subgoal("recall", "Read causal memory of prior hypotheses, actions, outcomes and verdicts."),
            Subgoal("propose", "Propose recipe changes that could move the metric. Do not crown a winner."),
            Subgoal(
                "evolve",
                "Promote a proposal only when its margin beats one standard error and the control.",
            ),
            Subgoal(
                "act",
                f"Touch at most {self.budget.max_pages_changed} pages. "
                f"Cooldown {self.budget.cooldown_generations} generation(s). "
                f"Blacklist: {blacklist}.",
            ),
        )

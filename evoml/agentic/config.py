"""JSON config for one agent cycle."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .goal import Budget, Goal


@dataclass
class AgentConfig:
    domain: str = "seo_pages"
    seed: int = 0
    dry_run: bool = True
    planner: str = "stub"
    metric: str = "booking_starts"
    direction: str = "maximize"
    max_pages_changed: int = 8
    cooldown_generations: int = 0
    blacklist: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    output_dir: str = "out/seo-dry-run"
    audit_path: str | None = None
    memory: str = "none"
    memory_dir: str | None = None
    meta: bool = True
    cycles: int = 1
    signals: str = "simulator"
    pages: list[str] = field(default_factory=list)
    cohort_size: int = 8
    generations: int = 1

    def goal(self) -> Goal:
        return Goal(
            metric=self.metric,
            direction=self.direction,
            constraints=tuple(self.constraints),
            budget=Budget(
                max_pages_changed=self.max_pages_changed,
                cooldown_generations=self.cooldown_generations,
                blacklist=tuple(self.blacklist),
            ),
        )

    @property
    def measured(self) -> bool:
        return self.signals != "simulator"


def load_config(path: str | Path) -> AgentConfig:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    known = set(AgentConfig.__dataclass_fields__)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown config fields: {', '.join(unknown)}")
    return AgentConfig(**data)

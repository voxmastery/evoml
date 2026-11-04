"""Sensors read outcomes. They do not fetch URLs.

File sensors use the CSV and JSON loaders already in the SEO adapter.
The simulator sensor is for tests and the ablation harness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from evoml.domains.seo_pages.evaluator import RecipeSimulator
from evoml.domains.seo_pages.genome import PageRecipe
from evoml.domains.seo_pages.signals import PageObservation, load_csv, load_json


class Sensor(Protocol):
    def read(self) -> list[PageObservation]:
        """Current outcome rows. Search Console, engagement and bookings share a row."""

    @property
    def measured(self) -> bool:
        """True when the rows came from an export rather than the simulator."""


class FileSensor:
    """Load a local CSV or JSON export. No network."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.suffix.lower() not in {".csv", ".json"}:
            raise ValueError(f"unsupported signal file {self.path.name}")

    def read(self) -> list[PageObservation]:
        if self.path.suffix.lower() == ".csv":
            return load_csv(self.path)
        return load_json(self.path)

    @property
    def measured(self) -> bool:
        return True


class SimulatorSensor:
    """Counterfactual rows for the champion and the untouched control."""

    def __init__(
        self,
        simulator: RecipeSimulator,
        champion: PageRecipe,
        *,
        cohort_size: int,
        day: int = 0,
    ):
        self.simulator = simulator
        self.champion = champion
        self.cohort_size = cohort_size
        self.day = day

    def read(self) -> list[PageObservation]:
        treated = self.simulator.observations(
            self.champion, cohort_size=self.cohort_size, day=self.day
        )
        control = self.simulator.control_observations(self.cohort_size, day=self.day)
        return list(treated) + list(control)

    @property
    def measured(self) -> bool:
        return False


def booking_mean(rows: Sequence[PageObservation], group: str | None = None) -> float | None:
    chosen = [row for row in rows if group is None or row.group == group]
    if not chosen:
        return None
    return sum(row.booking_starts for row in chosen) / len(chosen)

"""Hard vetoes for recipes that line up with damage.

A recipe is blacklisted when its pages show a spike in 404s, deindexing,
or noindex relative to the untouched control pages in the same window.
The score is still recorded; the arena will not promote it.
"""
from __future__ import annotations

from .signals import PageObservation

SPIKE_FLOOR = 0.02
GUARDED = ("http_404", "deindexed", "noindex")


def rates(rows: list[PageObservation]) -> dict[str, float]:
    n = max(1, len(rows))
    return {name: sum(getattr(row, name) for row in rows) / n for name in GUARDED}


def spiked(recipe_rates: dict[str, float], control_rates: dict[str, float]) -> str | None:
    """Return the signal name that spiked, or None.

    A spike is a rate at or above `SPIKE_FLOOR` and at least that far
    above the control rate in the same window.
    """
    for name in GUARDED:
        recipe = float(recipe_rates.get(name, 0.0))
        control = float(control_rates.get(name, 0.0))
        if recipe >= SPIKE_FLOOR and recipe >= control + SPIKE_FLOOR:
            return name
    return None

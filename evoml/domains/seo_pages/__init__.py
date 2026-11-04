"""SEO landing-page adapter.

Recipes evolve against measured (or simulated) page outcomes. The policy
that scores them — weights, window, mutation rate, cohort size, promotion
threshold — evolves on the same arena.
"""

from .experiment import SEOExperiment
from .genome import PageRecipe, invent_gene
from .meta import MetaExperiment, MetaGenome, simulate_bangalore_history
from .signals import load_csv, load_json

__all__ = [
    "MetaExperiment",
    "MetaGenome",
    "PageRecipe",
    "SEOExperiment",
    "invent_gene",
    "load_csv",
    "load_json",
    "simulate_bangalore_history",
]

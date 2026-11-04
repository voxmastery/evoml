"""Interfaces shared by every EvoML domain.

Nothing in this package knows about prices, fraud labels, or web pages.
"""

from .arena import Arena, Contender, Decision, beats
from .audit import AuditLog
from .evaluator import Score
from .genome import Genome
from .memory import FluctlightMemory, MemoryHook, NullMemory

__all__ = [
    "Arena",
    "AuditLog",
    "Contender",
    "Decision",
    "FluctlightMemory",
    "Genome",
    "MemoryHook",
    "NullMemory",
    "Score",
    "beats",
]

"""Causal memory: hypothesis, action, outcome, verdict.

Stored through the existing memory hook. FluctlightDB is optional. With
no hook, or with `NullMemory`, recall returns nothing and the cycle
still runs. `verified=True` is set only when the sensor read an export.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from evoml.core.memory import FluctlightMemory, MemoryHook, NullMemory


@dataclass(frozen=True)
class Episode:
    hypothesis: str
    action: str
    outcome: str
    verdict: str
    blocked_section: str | None = None

    def render(self) -> str:
        blocked = f"\nblocked_section:{self.blocked_section}" if self.blocked_section else ""
        return (
            f"hypothesis: {self.hypothesis}\n"
            f"action: {self.action}\n"
            f"outcome: {self.outcome}\n"
            f"verdict: {self.verdict}"
            f"{blocked}"
        )


class InMemoryHook:
    """Process-local hook so a multi-cycle run can recall what it just stored.

    Cron across processes should use ``fluctlight`` instead. This hook is
    what the ablation's memory arms use, and it never touches the network.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict]] = []

    def experience(self, content: str, **kwargs):
        self.rows.append((content, dict(kwargs)))
        return {"engram_id": str(len(self.rows))}

    def recall(self, cue: str, limit: int | None = None) -> list:
        hits = [
            {"episode": {"content": content}, "verified": bool(meta.get("verified"))}
            for content, meta in self.rows
        ]
        if cue:
            matched = [hit for hit in hits if cue in hit["episode"]["content"]]
            hits = matched or hits
        if limit is not None:
            return hits[-limit:]
        return hits


class CausalMemory:
    def __init__(self, hook: MemoryHook | None = None):
        self.hook: MemoryHook = hook if hook is not None else NullMemory()

    def recall(self, cue: str, limit: int = 8) -> list[Any]:
        return list(self.hook.recall(cue, limit=limit))

    def remember(self, episode: Episode, *, verified: bool, source_uri: str | None = None) -> None:
        self.hook.experience(
            episode.render(),
            context="evoml:agentic",
            salience=0.8 if episode.verdict == "promoted" else 0.5,
            outcome=episode.verdict,
            verified=verified,
            source_uri=source_uri,
        )


def open_memory(kind: str, directory: str | None = None) -> CausalMemory:
    """``none`` uses the no-op hook. ``fluctlight`` opens an embedded brain."""
    if kind in ("none", "", "null"):
        return CausalMemory(NullMemory())
    if kind == "fluctlight":
        if not directory:
            raise ValueError("fluctlight memory needs a directory")
        return CausalMemory(FluctlightMemory(directory))
    raise ValueError(f"unknown memory kind {kind!r}")


def episode_from_cycle(
    *,
    rationales: Sequence[str],
    action: str,
    outcome: str,
    verdict: str,
    blocked_section: str | None = None,
) -> Episode:
    hypothesis = " | ".join(rationales) if rationales else "no hypothesis proposed"
    return Episode(
        hypothesis=hypothesis,
        action=action,
        outcome=outcome,
        verdict=verdict,
        blocked_section=blocked_section,
    )

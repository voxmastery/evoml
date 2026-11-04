"""Optional memory of outcomes and decisions.

The default hook does nothing, so the engine runs with no extra dependency.
When FluctlightDB is installed, `FluctlightMemory` writes through
`connect_embedded` and reads back with `recall`.
"""
from __future__ import annotations

from typing import Any, Protocol


class MemoryHook(Protocol):
    def experience(self, content: str, **kwargs: Any) -> Any:
        """Record one outcome or decision."""

    def recall(self, cue: str, limit: int | None = None) -> list[Any]:
        """Return memories related to `cue`. Empty when nothing matches."""


class NullMemory:
    """No-op default. Calls succeed and remember nothing."""

    def experience(self, content: str, **kwargs: Any) -> None:
        return None

    def recall(self, cue: str, limit: int | None = None) -> list[Any]:
        return []


class FluctlightMemory:
    """FluctlightDB-backed hook.

    `experience(..., verified=True)` is the provenance flag from the
    embedded API. `recall` uses `brain.recall`; `activate` is only a
    fallback for an older binding that has no `recall`.
    """

    def __init__(self, directory: str | None = None, brain: Any = None):
        if brain is None:
            if not directory:
                raise ValueError("FluctlightMemory needs a directory or a brain")
            from fluctlightdb import connect_embedded

            brain = connect_embedded(directory)
        self._brain = brain

    def experience(
        self,
        content: str,
        *,
        context: str = "evoml",
        salience: float = 0.5,
        outcome: str | None = None,
        verified: bool = True,
        source_uri: str | None = None,
        **kwargs: Any,
    ) -> Any:
        result = self._brain.experience(
            content,
            context=context,
            salience=salience,
            outcome=outcome,
            verified=verified,
            source_uri=source_uri,
            **kwargs,
        )
        checkpoint = getattr(self._brain, "checkpoint", None)
        if checkpoint is not None:
            checkpoint()
        return result

    def recall(self, cue: str, limit: int | None = None) -> list[Any]:
        recall = getattr(self._brain, "recall", None)
        if recall is not None:
            try:
                out = recall(cue) if limit is None else recall(cue, limit=limit)
            except TypeError:
                out = recall(cue)
            return _hits(out)
        activate = getattr(self._brain, "activate", None)
        if activate is None:
            return []
        try:
            out = activate(cue) if limit is None else activate(cue, limit=limit)
        except TypeError:
            out = activate(cue)
        if isinstance(out, dict):
            return list(out.get("recalls") or [])
        return list(out)


def _hits(out: Any) -> list[Any]:
    if isinstance(out, dict):
        hits = out.get("hits")
        if hits is None:
            hits = out.get("recalls") or []
        return list(hits)
    if out is None:
        return []
    return list(out)

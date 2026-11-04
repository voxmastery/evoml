"""Champion, two variants, one challenger, and a control.

Promotion happens only when the margin beats one standard error of the
outcome estimate. A control is scored on the same outcomes and is never
crowned. Cooldown holds the champion for a fixed number of generations
after a promotion. Disqualified genomes (guardrails) cannot be promoted.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .audit import AuditLog
from .evaluator import Score
from .genome import Genome
from .memory import MemoryHook, NullMemory


def _noise(challenger: Score, incumbent: Score, mode: str) -> float:
    if mode == "difference":
        return math.sqrt(challenger.se ** 2 + incumbent.se ** 2)
    if mode == "max":
        return max(challenger.se, incumbent.se)
    if mode == "shared":
        # One noise floor, carried on the incumbent (crypto holdout SE).
        return incumbent.se
    raise ValueError(f"unknown se_mode {mode!r}")


def beats(
    challenger: Score,
    incumbent: Score,
    *,
    se_mult: float = 1.0,
    mode: str = "difference",
) -> bool:
    """True when `challenger.mean - incumbent.mean` exceeds `se_mult` SEs.

    Equality does not promote: a tie stays with the incumbent.
    """
    if challenger.disqualified or incumbent.disqualified:
        return False
    margin = challenger.mean - incumbent.mean
    return margin > se_mult * _noise(challenger, incumbent, mode)


@dataclass
class Contender:
    key: str
    role: str
    genome: Any = None


@dataclass
class Decision:
    generation: int
    incumbent_key: str | None
    winner_key: str
    promoted: bool
    succession: str
    scores: dict[str, Score]
    control_key: str | None = None
    blacklisted: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)


class Arena:
    """Scheduler for one population.

    Population shape, assembled by `assemble` or supplied by the domain:
    the reigning champion, `n_variants` mutated children (default 2),
    one fresh challenger, and an optional control.
    """

    def __init__(
        self,
        *,
        name: str = "arena",
        se_mult: float = 1.0,
        se_mode: str = "difference",
        n_variants: int = 2,
        cooldown_generations: int = 0,
        require_beat_control: bool = False,
        audit: AuditLog | None = None,
        memory: MemoryHook | None = None,
    ):
        if n_variants < 1:
            raise ValueError("n_variants must be at least 1")
        self.name = name
        self.se_mult = float(se_mult)
        self.se_mode = se_mode
        self.n_variants = int(n_variants)
        self.cooldown_generations = int(cooldown_generations)
        self.require_beat_control = bool(require_beat_control)
        self.audit = audit
        self.memory: MemoryHook = memory if memory is not None else NullMemory()
        self.blacklist: set[str] = set()
        self._last_promotion_gen: int | None = None

    def assemble(
        self,
        champion: Genome,
        rng: Any,
        *,
        challenger: Genome | None = None,
        control: Genome | None = None,
        extras: Sequence[Genome] | None = None,
    ) -> list[Contender]:
        """Champion, `n_variants` mutants, one fresh genome, optional control."""
        contenders = [Contender(champion.key(), "champion", champion)]
        for _ in range(self.n_variants):
            child = champion.mutate(rng)
            contenders.append(Contender(child.key(), "variant", child))
        fresh = challenger if challenger is not None else type(champion).random(rng)
        contenders.append(Contender(fresh.key(), "challenger", fresh))
        if control is not None:
            contenders.append(Contender(control.key(), "control", control))
        for extra in extras or []:
            contenders.append(Contender(extra.key(), "extra", extra))
        return _dedupe(contenders)

    def in_cooldown(self, generation: int) -> bool:
        last = self._last_promotion_gen
        if last is None or self.cooldown_generations <= 0:
            return False
        return 0 < (generation - last) <= self.cooldown_generations

    def decide(
        self,
        contenders: Sequence[Contender],
        scores: Mapping[str, Score],
        *,
        generation: int,
        incumbent_key: str | None,
    ) -> Decision:
        """Pick the successor. Does not write the audit; call `commit`."""
        by_key = {c.key: c for c in contenders}
        blacklisted: list[str] = []
        for key, score in scores.items():
            if score.disqualified or key in self.blacklist:
                blacklisted.append(key)
                self.blacklist.add(key)

        control_key = next((c.key for c in contenders if c.role == "control"), None)

        def eligible(key: str) -> bool:
            contender = by_key.get(key)
            if contender is None or contender.role == "control":
                return False
            if key in self.blacklist:
                return False
            score = scores.get(key)
            return score is not None and not score.disqualified

        eligible_keys = [c.key for c in contenders if eligible(c.key)]
        incumbent_ok = incumbent_key is not None and incumbent_key in scores

        def best_of(keys: list[str]) -> str | None:
            if not keys:
                return None
            return max(keys, key=lambda k: scores[k].mean)

        best = best_of(eligible_keys)
        succession = "new lineage"
        winner = best
        promoted = False

        if best is None:
            winner = incumbent_key if incumbent_ok else (contenders[0].key if contenders else "")
            succession = "held (no eligible challenger)"
            promoted = False
        elif not incumbent_ok or (incumbent_key in self.blacklist):
            winner, promoted, succession = self._crown_fresh(
                best, control_key, scores
            )
            if incumbent_key in self.blacklist and promoted:
                succession = "blacklisted"
        else:
            inc = scores[incumbent_key]
            if best == incumbent_key:
                winner, succession, promoted = incumbent_key, "held", False
            elif self.in_cooldown(generation):
                winner, succession, promoted = incumbent_key, "held (cooldown)", False
            elif not beats(scores[best], inc, se_mult=self.se_mult, mode=self.se_mode):
                winner = incumbent_key
                succession = "held (challenger within noise)"
                promoted = False
            elif not self._beats_control(scores[best], control_key, scores):
                winner = incumbent_key
                succession = "held (control)"
                promoted = False
            else:
                winner, succession, promoted = best, "dethroned", True

        if promoted:
            self._last_promotion_gen = generation

        return Decision(
            generation=generation,
            incumbent_key=incumbent_key,
            winner_key=winner or "",
            promoted=promoted,
            succession=succession,
            scores=dict(scores),
            control_key=control_key,
            blacklisted=tuple(blacklisted),
        )

    def _crown_fresh(
        self,
        best: str,
        control_key: str | None,
        scores: Mapping[str, Score],
    ) -> tuple[str, bool, str]:
        if self._beats_control(scores[best], control_key, scores):
            return best, True, "new lineage"
        return best, False, "held (control)"

    def _beats_control(
        self, score: Score, control_key: str | None, scores: Mapping[str, Score]
    ) -> bool:
        if not self.require_beat_control or control_key is None:
            return True
        control = scores.get(control_key)
        if control is None or control.disqualified:
            return True
        return beats(score, control, se_mult=self.se_mult, mode=self.se_mode)

    def commit(
        self,
        decision: Decision,
        *,
        genome: Any = None,
        verified: bool = False,
        source_uri: str | None = None,
        recall_cue: str | None = None,
    ) -> Decision:
        """Append the decision and, when a memory hook is set, remember it.

        `verified=True` marks the memory as coming from a measured source
        (a loader export), not from the simulator.
        """
        cue = recall_cue or decision.winner_key
        recalled = self.memory.recall(cue, limit=5)
        decision.detail["recall"] = recalled
        if genome is not None and hasattr(genome, "serialize"):
            serialized: Any = genome.serialize()
        else:
            serialized = genome if isinstance(genome, dict) else {"key": decision.winner_key}
        record = {
            "ts": time.time(),
            "arena": self.name,
            "generation": decision.generation,
            "champion": decision.winner_key,
            "promoted": decision.promoted,
            "succession": decision.succession,
            "genome": serialized,
            "scores": {
                key: {
                    "mean": score.mean,
                    "se": score.se,
                    "n": score.n,
                    "disqualified": score.disqualified,
                }
                for key, score in decision.scores.items()
            },
            "blacklisted": list(decision.blacklisted),
            "control": decision.control_key,
            "incumbent": decision.incumbent_key,
        }
        if self.audit is not None:
            self.audit.append(record)
        content = (
            f"{self.name} gen {decision.generation}: {decision.succession} "
            f"champion={decision.winner_key}"
        )
        self.memory.experience(
            content,
            context=f"evoml:{self.name}",
            salience=0.7 if decision.promoted else 0.4,
            outcome=decision.succession,
            verified=verified,
            source_uri=source_uri,
        )
        return decision


def _dedupe(contenders: Sequence[Contender]) -> list[Contender]:
    seen: set[str] = set()
    out: list[Contender] = []
    for contender in contenders:
        if contender.key in seen:
            continue
        seen.add(contender.key)
        out.append(contender)
    return out

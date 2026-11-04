"""Core arena: promotion, control, cooldown, audit, memory, domain gates."""
import random
import sqlite3

import pytest

from evoml.core.arena import Arena, Contender, beats
from evoml.core.audit import AuditLog
from evoml.core.evaluator import Score
from evoml.core.memory import FluctlightMemory, NullMemory
from evoml.domains.fraud import FraudGenome, consider_succession
from evoml.domains.prediction import PredictionGenome, select_champion


def _score(mean, se=0.01, n=20, disqualified=False):
    return Score(mean=mean, se=se, n=n, disqualified=disqualified)


def test_margin_must_clear_one_standard_error():
    incumbent = _score(0.50, se=0.05)
    within = _score(0.54, se=0.05)
    clear = _score(0.70, se=0.05)
    # difference SE = sqrt(0.05**2 + 0.05**2) ≈ 0.0707
    assert not beats(within, incumbent)
    assert beats(clear, incumbent)
    # equality does not promote
    tied = _score(0.50 + (0.05 ** 2 + 0.05 ** 2) ** 0.5, se=0.05)
    assert not beats(tied, incumbent)


def test_prediction_gate_matches_the_shared_holdout_floor():
    noise = 0.03
    roles = {"champ": "champion", "next": "challenger"}
    assert select_champion(
        {"champ": 0.50, "next": 0.52}, roles, "champ", noise
    ) == ("champ", "held (challenger within noise)")
    winner, succession = select_champion(
        {"champ": 0.50, "next": 0.54}, roles, "champ", noise
    )
    assert winner == "next" and succession == "dethroned"
    winner, succession = select_champion({"next": 0.70}, {"next": "challenger"}, None, noise)
    assert winner == "next" and succession == "new lineage"


def test_fraud_gate_uses_the_larger_standard_error():
    assert consider_succession(0.40, 0.02, 0.45, 0.01)
    # margin equal to the larger SE does not promote
    assert not consider_succession(0.40, 0.02, 0.42, 0.01)
    assert not consider_succession(0.40, 0.05, 0.44, 0.01)


def test_control_is_scored_and_never_crowned():
    arena = Arena(require_beat_control=False, se_mult=1.0)
    contenders = [
        Contender("champ", "champion"),
        Contender("next", "challenger"),
        Contender("control", "control"),
    ]
    scores = {
        "champ": _score(0.40),
        "next": _score(0.55),
        "control": _score(0.99),
    }
    decision = arena.decide(contenders, scores, generation=1, incumbent_key="champ")
    assert decision.winner_key == "next"
    assert decision.control_key == "control"
    assert decision.promoted
    # the same control blocks a challenger that does not beat it
    strict = Arena(require_beat_control=True)
    held = strict.decide(contenders, scores, generation=1, incumbent_key="champ")
    assert held.winner_key == "champ"
    assert held.succession == "held (control)"
    assert not held.promoted


def test_disqualified_genome_cannot_be_promoted():
    arena = Arena(require_beat_control=False)
    contenders = [
        Contender("champ", "champion"),
        Contender("toxic", "challenger"),
    ]
    scores = {"champ": _score(0.4), "toxic": _score(0.95, disqualified=True)}
    decision = arena.decide(contenders, scores, generation=1, incumbent_key="champ")
    assert decision.winner_key == "champ"
    assert "toxic" in decision.blacklisted
    assert "toxic" in arena.blacklist


def test_cooldown_holds_the_champion():
    arena = Arena(cooldown_generations=2, require_beat_control=False)
    contenders = [Contender("champ", "champion"), Contender("next", "challenger")]
    first = {"champ": _score(0.4), "next": _score(0.8)}
    opened = arena.decide(contenders, first, generation=1, incumbent_key="champ")
    assert opened.promoted and opened.winner_key == "next"
    later = {"next": _score(0.5), "better": _score(0.9)}
    contenders = [Contender("next", "champion"), Contender("better", "challenger")]
    held = arena.decide(contenders, later, generation=2, incumbent_key="next")
    assert held.succession == "held (cooldown)" and not held.promoted
    still = arena.decide(contenders, later, generation=3, incumbent_key="next")
    assert still.succession == "held (cooldown)"
    released = arena.decide(contenders, later, generation=4, incumbent_key="next")
    assert released.promoted and released.winner_key == "better"


def test_audit_log_is_append_only(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    first = log.append(
        {"arena": "seo", "generation": 1, "champion": "a", "promoted": False,
         "succession": "held", "genome": {"k": 1}, "scores": {"a": {"mean": 0.2}}}
    )
    second = log.append(
        {"arena": "seo", "generation": 2, "champion": "b", "promoted": True,
         "succession": "dethroned", "genome": {"k": 2}, "scores": {}}
    )
    rows = log.read()
    assert [row["id"] for row in rows] == [first, second]
    assert rows[0]["champion"] == "a" and rows[1]["succession"] == "dethroned"
    with pytest.raises(sqlite3.IntegrityError):
        log._conn.execute("DELETE FROM audit WHERE id = ?", (first,))
    with pytest.raises(sqlite3.IntegrityError):
        log._conn.execute("UPDATE audit SET champion = 'rewritten' WHERE id = ?", (first,))
    assert log.read()[0]["champion"] == "a"
    log.close()


def test_null_memory_and_fluctlight_hook_shape():
    hook = NullMemory()
    assert hook.experience("nothing happened") is None
    assert hook.recall("nothing") == []

    class Brain:
        def __init__(self):
            self.rows = []
            self.checkpointed = 0

        def experience(self, content, **kwargs):
            self.rows.append((content, kwargs))
            return {"engram_id": "e1", "deduplicated": False}

        def recall(self, cue, limit=None):
            return {"hits": [{"episode": {"content": cue}}], "lanes_used": ["lex"], "mode": "lexical"}

        def checkpoint(self):
            self.checkpointed += 1

    brain = Brain()
    memory = FluctlightMemory(brain=brain)
    memory.experience("promoted recipe", context="evoml:seo", verified=True, source_uri="file://signals.csv")
    assert brain.checkpointed == 1
    assert brain.rows[0][1]["verified"] is True
    assert brain.rows[0][1]["source_uri"] == "file://signals.csv"
    hits = memory.recall("promoted recipe", limit=3)
    assert hits and hits[0]["episode"]["content"] == "promoted recipe"


def test_prediction_genome_roundtrip_and_crossover():
    rng = random.Random(4)
    parent = PredictionGenome.random(rng)
    other = PredictionGenome.random(rng)
    child = parent.mutate(rng)
    assert child.serialize()["kind"] in ("logreg", "mlp", "hgb", "net")
    again = PredictionGenome.deserialize(child.serialize())
    assert again.serialize() == child.serialize()
    crossed = parent.crossover(other, rng)
    assert crossed.serialize()["features"]


def test_fraud_genome_mutation_stays_in_bounds():
    rng = random.Random(2)
    genome = FraudGenome.random(rng, n_feat=29, kind="net")
    for _ in range(50):
        genome = genome.mutate(rng)
        assert 12 <= len(genome.data["features"]) <= 29
    assert genome.serialize()["kind"] == "net"

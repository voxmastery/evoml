"""Fraud bench: genome mutation stays in bounds; budget metrics are sane."""
import random

import numpy as np

from bench.fraud_bench import MIN_FEATURES, at_budget, label, mutate, random_genome


def test_mutation_respects_bounds():
    rng = random.Random(4)
    for kind in ("net", "hgb", "logreg"):
        g = random_genome(rng, 29, kind=kind)
        for _ in range(300):
            g = mutate(g, rng, 29)
            assert MIN_FEATURES <= len(g["features"]) <= 29
            assert all(0 <= f < 29 for f in g["features"])
            if kind == "net":
                assert all(4 <= h <= 64 for h in g["hidden"])
                assert 1e-4 <= g["lr"] <= 3e-2 and 0.0 <= g["prune"] <= 0.5
        assert label(g).startswith(kind)


def test_at_budget_counts_true_positives():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = at_budget(y, s, 0.2)
    assert r["alerts"] == 2 and r["true_positives"] == 2
    assert r["precision"] == 1.0 and r["recall"] == 1.0

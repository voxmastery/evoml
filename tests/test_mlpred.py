"""ML forecaster arm: dataset building, training gate, and abstention."""
import pytest

from memescalp.db import Database
from memescalp.mlpred import (
    MIN_TRAIN_ROWS, MLForecaster, build_dataset, feature_vector,
)
from memescalp.catalog import CatalogEntry
from tests.test_predict import entry


def seed_db(db: Database, n_windows: int = 40, up: bool = True):
    """Catalog snapshots + prices such that price always moves after them."""
    for i in range(n_windows):
        ts = 1000.0 + i * 60.0
        price = 1.0 + i * 0.001
        later = price * (1.01 if up else 0.99)
        db.insert_catalog(ts, [CatalogEntry(
            mint="M1", symbol="AAA", price_usd=price, liquidity_usd=100_000.0,
            volume_h24=1e6, price_change_h1=1.0, price_change_h24=2.0,
            volatility_pct=0.5, est_cost_pct=2.9, score=5.0, rank=1)])
        db.insert_price(ts, "M1", "AAA", price, "jupiter")
        db.insert_price(ts + 120.0, "M1", "AAA", later, "jupiter")


def test_feature_vector_shape_and_flow_ratio():
    x = feature_vector(1.0, 2.0, 3.0, 0.5, 100_000, 10.0, None, 9, 3)
    assert len(x) == 19
    assert x[9] == pytest.approx((9 / 12 - 0.5) * 2)   # buy-pressure feature
    x2 = feature_vector(0, 0, 0, 0, 100_000, 0, None, None, None)
    assert x2[9] == 0.0


def test_build_dataset_labels_direction(settings, monkeypatch):
    import time as _t
    db = Database(settings.db_path)
    seed_db(db, n_windows=10, up=True)
    monkeypatch.setattr(_t, "time", lambda: 1000.0 + 11 * 60.0 + 400.0)
    X, y, ts = build_dataset(db, horizon_s=120.0)
    assert len(X) == len(y) > 0
    assert all(label == 1 for label in y)   # price always rose afterwards


def test_forecaster_abstains_until_trained(settings):
    db = Database(settings.db_path)
    ml = MLForecaster()
    assert not ml.trained
    p = ml.make_prediction([entry()], {}, 2.0, db)
    assert p.arm == "ml" and p.direction == "SKIP"
    assert "untrained" in p.response


def test_retrain_requires_min_rows_and_both_classes(settings, monkeypatch):
    import time as _t
    db = Database(settings.db_path)
    seed_db(db, n_windows=20, up=True)
    monkeypatch.setattr(_t, "time", lambda: 1000.0 + 21 * 60.0 + 400.0)
    ml = MLForecaster()
    result = ml.retrain(db, 120.0)
    # 20 one-class rows: below MIN_TRAIN_ROWS and single-class -> no model.
    assert not result["trained"] and not ml.trained
    assert result["rows"] <= MIN_TRAIN_ROWS


def test_adaptive_threshold_rules():
    from memescalp.mlpred import (
        BOLD_PROB, CAUTIOUS_PROB, MIN_CALL_PROB, adaptive_threshold,
    )
    assert adaptive_threshold(None, 0.51) is None          # self-benched
    assert adaptive_threshold(0.7, 0.50) is None           # holdout rules all
    assert adaptive_threshold(0.45, 0.56) == CAUTIOUS_PROB  # live gone cold
    assert adaptive_threshold(0.62, 0.56) == BOLD_PROB      # hot on both
    assert adaptive_threshold(0.55, 0.56) == MIN_CALL_PROB  # normal
    assert adaptive_threshold(None, 0.56) == MIN_CALL_PROB  # no live data yet
    assert adaptive_threshold(0.62, None) == MIN_CALL_PROB  # no holdout: default


def test_champion_challenger_picks_winner(settings, monkeypatch):
    import memescalp.mlpred as mlp
    import random as _r
    rng = _r.Random(11)
    # Separable synthetic data: label follows the sign of feature 0.
    X, y = [], []
    for _ in range(600):
        base = rng.uniform(-2, 2)
        row = [base] + [rng.uniform(-1, 1) for _ in range(18)]
        X.append(row)
        y.append(1 if base > 0 else 0)
    ts = [float(i) for i in range(len(X))]
    monkeypatch.setattr(mlp, "build_dataset",
                        lambda db, h, max_rows=8000: (X, y, ts))
    from memescalp.db import Database
    db = Database(settings.db_path)
    ml = mlp.MLForecaster()
    result = ml.retrain(db, 120.0)
    assert result["trained"] and ml.trained
    assert result["n_params"] > 0
    assert result["generation"] == 1
    assert db.get_meta("ml_genome") is not None
    # A second generation seeds from the survivor and increments the counter.
    result2 = mlp.MLForecaster().retrain(db, 120.0)
    assert result2["generation"] == 2
    assert result["holdout_accuracy"] > 0.9   # separable -> champion aces it


def test_self_benched_model_refuses_to_call(settings):
    from memescalp.db import Database
    from memescalp.mlpred import MLForecaster
    db = Database(settings.db_path)
    ml = MLForecaster()
    ml._pipeline = object()      # pretend trained
    ml.champion = "logreg"
    ml.holdout_accuracy = 0.50   # below the 52% self-bench floor
    p = ml.make_prediction([entry()], {}, 2.0, db)
    assert p.direction == "SKIP"
    assert "self-benched" in p.response



def test_mutation_respects_bounds():
    import random as _r
    from memescalp.mlpred import mutate_genome, random_genome
    rng = _r.Random(3)
    g = {"kind": "mlp", "hidden": [16, 8], "alpha": 1e-4}
    for _ in range(200):
        g = mutate_genome(g, rng)
        assert 1 <= len(g["hidden"]) <= 3
        assert all(4 <= h <= 64 for h in g["hidden"])
        assert 1e-6 <= g["alpha"] <= 0.1
    h = {"kind": "hgb", "depth": 3, "lr": 0.08, "iters": 120}
    for _ in range(200):
        h = mutate_genome(h, rng)
        assert 2 <= h["depth"] <= 6
        assert 0.01 <= h["lr"] <= 0.3
        assert 40 <= h["iters"] <= 400
    for _ in range(30):
        imm = random_genome(rng, exclude_kind="mlp")
        assert imm["kind"] in ("logreg", "hgb", "net")


def test_invented_genes_evaluate_and_apply():
    from memescalp.mlpred import (
        MAX_SYNTH_GENES, NUM_FEATURES, apply_genome, eval_synth,
        invent_gene, mutate_genome, synth_str, _ensure_genome_defaults,
    )
    base = [float(i) for i in range(NUM_FEATURES)]
    assert eval_synth(["mul", 2, 3], base) == 6.0
    assert eval_synth(["sub", 5, 2], base) == 3.0
    assert eval_synth(["neg", 4], base) == -4.0
    assert eval_synth(["sign", 0], base) == 0.0
    assert eval_synth(["div", 1, 0], base) == 1e3   # safe divide, clipped
    assert eval_synth(["sq", 18], base) == 324.0
    assert synth_str(["mul", 9, 6]) == "mul(buy_ratio, ret_60s)"
    g = _ensure_genome_defaults({"kind": "logreg", "C": 0.5})
    g["synth"] = [["mul", 0, 1], ["abs", 2]]
    out = apply_genome(g, base)
    assert len(out) == NUM_FEATURES + 2 and out[-2:] == [0.0, 2.0]
    # Evolution can invent and kill synth genes, never beyond the cap.
    import random as _r
    rng = _r.Random(5)
    report = [0.0] * NUM_FEATURES
    report[9] = 0.3
    seen_synth = False
    for _ in range(300):
        g = mutate_genome(g, rng, report)
        assert len(g["synth"]) <= MAX_SYNTH_GENES
        seen_synth = seen_synth or bool(g["synth"])
    assert seen_synth
    expr = invent_gene(rng, report)
    assert expr[0] in ("neg", "abs", "sq", "sign", "mul", "sub", "div", "max", "min")


def test_self_modifying_genome_bounds():
    import random as _r
    from memescalp.mlpred import (
        MIN_FEATURE_GENES, NUM_FEATURES, _ensure_genome_defaults,
        mutate_genome, random_genome,
    )
    rng = _r.Random(9)
    g = _ensure_genome_defaults({"kind": "hgb", "depth": 3, "lr": 0.08,
                                 "iters": 120})
    assert g["features"] == list(range(NUM_FEATURES))
    sizes = set()
    for _ in range(400):
        g = mutate_genome(g, rng)
        assert MIN_FEATURE_GENES <= len(g["features"]) <= NUM_FEATURES
        assert all(0 <= i < NUM_FEATURES for i in g["features"])
        assert 0.54 <= g["thr"] <= 0.70
        assert 0.5 <= g["mut_rate"] <= 3.0
        assert 0.62 <= g["max_conf"] <= 0.95
        assert 0.0 <= g["min_fam"] <= 0.9
        assert isinstance(g["poly"], bool)
        sizes.add(len(g["features"]))
    assert len(sizes) > 1          # the feature chromosome actually evolves


def test_familiarity_and_regime_expression():
    from memescalp.mlpred import (
        NUM_FEATURES, REGIME_VR_INDEX, familiarity, fit_familiarity, regime_of,
    )
    import random as _r
    rng = _r.Random(2)
    X = [[rng.gauss(0, 1) for _ in range(NUM_FEATURES)] for _ in range(500)]
    mu, sd = fit_familiarity(X)
    assert familiarity([0.0] * NUM_FEATURES, mu, sd) == 1.0      # dead centre
    far = [50.0] * NUM_FEATURES
    assert familiarity(far, mu, sd) == 0.0                        # alien input
    x = [0.0] * NUM_FEATURES
    x[REGIME_VR_INDEX] = 0.2
    assert regime_of(x) == "trend"
    x[REGIME_VR_INDEX] = -0.2
    assert regime_of(x) == "revert"
    x[REGIME_VR_INDEX] = 0.0
    assert regime_of(x) == "walk"


def test_organism_persists_and_grows_online(settings, monkeypatch):
    """The champion network survives a restart and learns from resolved
    windows without being retrained from zero."""
    import numpy as np
    import memescalp.mlpred as mlp
    from memescalp.db import Database
    from memescalp.evonet import EvoNet, NetPipeline

    db = Database(settings.db_path)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, mlp.NUM_FEATURES)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    net = EvoNet(mlp.NUM_FEATURES, [8], lr=1e-2, epochs=5).fit(X, y)
    ml = mlp.MLForecaster()
    ml._pipeline = NetPipeline(net)
    ml._pipes = {"global": ml._pipeline}
    ml._parent_net = net
    ml._genome = mlp._ensure_genome_defaults({"kind": "net", "hidden": [8]})
    ml.champion = "net8"
    ml.holdout_accuracy = 0.6
    ml._save_state(db)

    fresh = mlp.MLForecaster()
    assert fresh.load_state(db) and fresh.trained
    assert np.allclose(fresh._pipeline.predict_proba(X[:5]),
                       ml._pipeline.predict_proba(X[:5]))
    assert fresh.champion == "net8" and fresh.n_params == net.n_params()

    # A resolved window: price rose 1% after the stash -> label 1.
    ts0 = 1000.0
    db.insert_price(ts0, "M1", "AAA", 1.0, "jupiter")
    db.insert_price(ts0 + 120.0, "M1", "AAA", 1.01, "jupiter")
    fresh.remember_window(ts0, [("M1", [float(v) for v in X[0]])])
    import time as _t
    monkeypatch.setattr(_t, "time", lambda: ts0 + 120.0 + 200.0)
    out = fresh.grow_step(db, 120.0)
    assert out["rows"] == 1 and out["absorbed"] == 1
    assert fresh.experience == 1
    assert json_loads(db.get_meta(mlp.META_GROWTH))["experience"] == 1


def json_loads(raw):
    import json
    return json.loads(raw)


def test_organism_widens_with_experience(settings, monkeypatch):
    import numpy as np
    import memescalp.mlpred as mlp
    from memescalp.db import Database
    from memescalp.evonet import EvoNet, NetPipeline

    monkeypatch.setattr(mlp, "GROW_EVERY_ROWS", 3)
    db = Database(settings.db_path)
    rng = np.random.default_rng(5)
    X = rng.standard_normal((300, mlp.NUM_FEATURES)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    net = EvoNet(mlp.NUM_FEATURES, [8, 4], lr=1e-2, epochs=3).fit(X, y)
    ml = mlp.MLForecaster()
    ml._pipeline = NetPipeline(net)
    ml._pipes = {"global": ml._pipeline}
    ml._genome = mlp._ensure_genome_defaults({"kind": "net", "hidden": [8, 4]})
    before = net.predict_proba(X[:20])[:, 1].copy()
    ts0 = 1000.0
    rows = []
    for i in range(4):
        m = f"M{i}"
        db.insert_price(ts0, m, m, 1.0, "jupiter")
        db.insert_price(ts0 + 120.0, m, m, 1.02, "jupiter")
        rows.append((m, [float(v) for v in X[i]]))
    ml.remember_window(ts0, rows)
    import time as _t
    monkeypatch.setattr(_t, "time", lambda: ts0 + 400.0)
    out = ml.grow_step(db, 120.0)
    assert out["absorbed"] == 4 and out.get("grew", "").startswith("widen")
    assert net.hidden == [8, 6] and ml._genome["hidden"] == [8, 6]
    assert ml.n_params == net.n_params() > 0
    # growth kept the function close (one small Adam step moved it slightly)
    assert np.abs(net.predict_proba(X[:20])[:, 1] - before).max() < 0.2

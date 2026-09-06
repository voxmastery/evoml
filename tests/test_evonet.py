"""EvoNet: from-scratch network — gradients, learning, inheritance, plumbing."""
import numpy as np
import pytest

from memescalp.evonet import EvoNet, NetPipeline


def separable(n=600, d=6, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y


def test_backward_matches_finite_differences():
    """The hand-derived gradient must agree with numerical differentiation."""
    rng = np.random.default_rng(0)
    net = EvoNet(4, [5, 3], act="tanh", l2=1e-3, seed=1)
    X = rng.standard_normal((16, 4)).astype(np.float32)
    y = (rng.random(16) > 0.5).astype(np.float32)
    w = np.ones(16, dtype=np.float32)
    zs, acts = net._forward(X)
    gW, gb = net._backward(zs, acts, y, w)
    eps = 1e-2
    for li in range(len(net.W)):
        for (i, j) in [(0, 0), (net.W[li].shape[0] - 1, net.W[li].shape[1] - 1)]:
            old = float(net.W[li][i, j])
            net.W[li][i, j] = old + eps; lp = net.loss(X, y, w)
            net.W[li][i, j] = old - eps; lm = net.loss(X, y, w)
            net.W[li][i, j] = old
            numeric = (lp - lm) / (2 * eps)
            assert abs(numeric - float(gW[li][i, j])) < 2e-2, (li, i, j, numeric, gW[li][i, j])


def test_learns_a_separable_problem():
    X, y = separable()
    net = EvoNet(6, [8], lr=1e-2, epochs=30).fit(X[:500], y[:500])
    p = net.predict_proba(X[500:])[:, 1]
    acc = ((p >= 0.5) == (y[500:] > 0.5)).mean()
    assert acc > 0.9
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_sample_weights_and_pruning():
    X, y = separable()
    net = EvoNet(6, [8], lr=1e-2, epochs=20, prune=0.3)
    net.fit(X, y, sample_weight=np.linspace(0.1, 1.0, len(y)))
    kept = sum(int(m.sum()) for m in net.mask)
    total = sum(m.size for m in net.mask)
    assert kept < total                      # some weights silenced
    assert net.n_params() == kept + sum(b.size for b in net.b)


def test_lamarckian_inheritance_copies_overlapping_blocks():
    X, y = separable()
    parent = EvoNet(6, [8, 4], lr=1e-2, epochs=20).fit(X, y)
    child = EvoNet(6, [16, 4], seed=99)      # grown first layer
    child.inherit(parent)
    assert np.allclose(child.W[0][:, :8], parent.W[0])
    assert np.allclose(child.W[-1][:4, :], parent.W[-1])
    assert np.allclose(child._mu, parent._mu)
    # Inherited child should start already skilled, before any training.
    p = child.predict_proba(X)[:, 1]
    assert ((p >= 0.5) == (y > 0.5)).mean() > 0.7


def test_pipeline_duck_typing():
    X, y = separable()
    pipe = NetPipeline(EvoNet(6, [8], epochs=5))
    pipe.fit(X, y, sample_weight=np.ones(len(y)))
    proba = pipe.predict_proba(X[:3])
    assert proba.shape == (3, 2)
    assert pipe[-1] is pipe.net
    assert hasattr(pipe[-1], "coefs_") and hasattr(pipe[-1], "intercepts_")


def test_net_genome_builds_and_labels():
    from memescalp.mlpred import genome_label, genome_to_pipeline, _ensure_genome_defaults
    g = _ensure_genome_defaults({"kind": "net", "hidden": [16, 8], "act": "relu",
                                 "lr": 3e-3, "prune": 0.2})
    pipe, wkey = genome_to_pipeline(g)
    assert isinstance(pipe, NetPipeline) and wkey == "sample_weight"
    assert pipe.net.n_in == 19
    assert genome_label(g).startswith("net16x8(relu,lr=0.003,p=0.2)")


def test_growth_preserves_function_and_persists():
    X, y = separable()
    net = EvoNet(6, [8, 4], act="relu", lr=1e-2, epochs=15).fit(X, y)
    before = net.predict_proba(X)[:, 1]
    net.widen(0, 4)
    assert net.hidden == [12, 4]
    assert np.allclose(net.predict_proba(X)[:, 1], before, atol=2e-2)
    net.deepen(1)
    assert len(net.hidden) == 3 and net.acts[2] == "relu"
    assert np.allclose(net.predict_proba(X)[:, 1], before, atol=2e-2)
    assert len(net.growth) == 2
    # tanh nets deepen with an exact linear identity layer
    t = EvoNet(6, [8], act="tanh", epochs=5).fit(X, y)
    pt = t.predict_proba(X)[:, 1]
    t.deepen(0)
    assert t.acts == ["tanh", "linear"]
    assert np.allclose(t.predict_proba(X)[:, 1], pt, atol=1e-5)
    # state round-trips, optimiser moments included
    d = net.state_dict()
    back = EvoNet.from_state(d)
    assert np.allclose(back.predict_proba(X)[:, 1], net.predict_proba(X)[:, 1])
    assert back._t == net._t and back.growth == net.growth


def test_absorb_learns_online_without_refit():
    X, y = separable(n=800)
    net = EvoNet(6, [8], lr=1e-2, epochs=3).fit(X[:200], y[:200])
    acc0 = ((net.predict_proba(X[600:])[:, 1] >= 0.5) == (y[600:] > 0.5)).mean()
    replay = (X[:200], y[:200], np.ones(200, dtype=np.float32))
    for s in range(200, 600, 20):
        net.absorb(X[s:s + 20], y[s:s + 20], replay=replay, steps=3)
    acc1 = ((net.predict_proba(X[600:])[:, 1] >= 0.5) == (y[600:] > 0.5)).mean()
    assert net.experience == 400
    assert acc1 >= acc0 - 0.02 and acc1 > 0.85

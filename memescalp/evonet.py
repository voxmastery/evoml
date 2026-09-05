"""EvoNet: a neural network written from first principles in NumPy.

No scikit-learn, no autograd. Every number here is ours: the forward pass,
the backward pass (hand-derived gradients), the Adam optimizer, weighted
binary cross-entropy, L2 regularization, magnitude pruning, and Lamarckian
weight inheritance (children start from their parent's learned weights).

Hardware notes: float32 throughout so AVX2 handles 8 lanes per instruction;
mini-batches sized so a batch (256 x inputs) plus weights stay inside L2;
matrix products go through BLAS, which uses every core.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.float32
BATCH = 256


def _act(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, 0.0)
    return np.tanh(z)


def _act_grad(name: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    if name == "relu":
        return (z > 0.0).astype(DTYPE)
    return 1.0 - a * a          # d tanh / dz


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class EvoNet:
    """Fully-connected network: inputs -> hidden layers -> 1 logit."""

    def __init__(self, n_in: int, hidden: list[int], act: str = "tanh",
                 lr: float = 3e-3, l2: float = 1e-4, epochs: int = 40,
                 prune: float = 0.0, seed: int = 7):
        self.n_in, self.hidden, self.act = int(n_in), [int(h) for h in hidden], act
        self.lr, self.l2, self.epochs, self.prune = lr, l2, int(epochs), prune
        rng = np.random.default_rng(seed)
        sizes = [self.n_in] + self.hidden + [1]
        # He/Glorot-style init: variance scaled by fan-in so signals neither
        # explode nor vanish through depth.
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(DTYPE)
                  for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, dtype=DTYPE) for b in sizes[1:]]
        self.mask = [np.ones_like(w) for w in self.W]
        self._mu = np.zeros(self.n_in, dtype=DTYPE)
        self._sd = np.ones(self.n_in, dtype=DTYPE)

    # --- sklearn-compatible surface used by the tournament -------------------
    @property
    def coefs_(self):
        return self.W

    @property
    def intercepts_(self):
        return self.b

    def n_params(self) -> int:
        return int(sum(int(m.sum()) for m in self.mask) + sum(b.size for b in self.b))

    # --- Lamarckian inheritance -------------------------------------------------
    def inherit(self, parent: "EvoNet") -> None:
        """Start from the parent's learned weights. Where shapes differ, copy the
        overlapping block so knowledge survives a grown or shrunk layer."""
        for i in range(min(len(self.W), len(parent.W))):
            # The output layer must stay last; only inherit matching positions
            # except the final layer, which inherits from the parent's final.
            src_w, src_b = parent.W[i], parent.b[i]
            if i == len(self.W) - 1:
                src_w, src_b = parent.W[-1], parent.b[-1]
            r = min(self.W[i].shape[0], src_w.shape[0])
            c = min(self.W[i].shape[1], src_w.shape[1])
            self.W[i][:r, :c] = src_w[:r, :c]
            self.b[i][:c] = src_b[:c]
        self._mu, self._sd = parent._mu.copy(), parent._sd.copy()

    # --- forward / backward -------------------------------------------------------
    def _forward(self, X: np.ndarray):
        zs, acts = [], [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ (W * self.mask[i]) + b
            zs.append(z)
            a = _sigmoid(z) if i == len(self.W) - 1 else _act(self.act, z)
            acts.append(a)
        return zs, acts

    def _backward(self, zs, acts, y, w):
        """Hand-derived gradients of weighted BCE + L2 w.r.t. every W and b."""
        n = y.shape[0]
        p = acts[-1][:, 0]
        # dL/dz_out for sigmoid + BCE collapses to (p - y), scaled by weights.
        delta = ((p - y) * w / max(1.0, w.sum()))[:, None].astype(DTYPE)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ delta + self.l2 * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ (self.W[i] * self.mask[i]).T) \
                    * _act_grad(self.act, zs[i - 1], acts[i])
        return gW, gb

    # --- training -----------------------------------------------------------------
    def fit(self, X, y, sample_weight=None) -> "EvoNet":
        X = np.asarray(X, dtype=DTYPE)
        y = np.asarray(y, dtype=DTYPE)
        w = (np.ones(len(y), dtype=DTYPE) if sample_weight is None
             else np.asarray(sample_weight, dtype=DTYPE))
        self._mu = X.mean(axis=0).astype(DTYPE)
        self._sd = (X.std(axis=0) + 1e-6).astype(DTYPE)
        Xn = (X - self._mu) / self._sd
        rng = np.random.default_rng(11)
        # Adam (Kingma & Ba 2015), written out: first/second moment estimates.
        mW = [np.zeros_like(W) for W in self.W]; vW = [np.zeros_like(W) for W in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        n = len(y)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, BATCH):
                idx = order[s:s + BATCH]
                zs, acts = self._forward(Xn[idx])
                gW, gb = self._backward(zs, acts, y[idx], w[idx])
                t += 1
                for i in range(len(self.W)):
                    mW[i] = b1 * mW[i] + (1 - b1) * gW[i]
                    vW[i] = b2 * vW[i] + (1 - b2) * gW[i] * gW[i]
                    mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
                    vb[i] = b2 * vb[i] + (1 - b2) * gb[i] * gb[i]
                    mhat = mW[i] / (1 - b1 ** t); vhat = vW[i] / (1 - b2 ** t)
                    self.W[i] -= (self.lr * mhat / (np.sqrt(vhat) + eps)).astype(DTYPE)
                    mhat_b = mb[i] / (1 - b1 ** t); vhat_b = vb[i] / (1 - b2 ** t)
                    self.b[i] -= (self.lr * mhat_b / (np.sqrt(vhat_b) + eps)).astype(DTYPE)
        if self.prune > 0:
            # Structural sparsity: silence the smallest |weights| for good.
            for i, W in enumerate(self.W):
                thresh = np.quantile(np.abs(W), self.prune)
                self.mask[i] = (np.abs(W) >= thresh).astype(DTYPE)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=DTYPE)
        Xn = (X - self._mu) / self._sd
        p = self._forward(Xn)[1][-1][:, 0]
        return np.stack([1.0 - p, p], axis=1)

    # --- numerical gradient check (used by tests) ---------------------------------
    def loss(self, X, y, w=None) -> float:
        X = np.asarray(X, dtype=DTYPE); y = np.asarray(y, dtype=DTYPE)
        w = np.ones(len(y), dtype=DTYPE) if w is None else np.asarray(w, dtype=DTYPE)
        p = np.clip(self._forward(X)[1][-1][:, 0], 1e-7, 1 - 1e-7)
        bce = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / max(1.0, w.sum())
        reg = 0.5 * self.l2 * sum(float((W * W).sum()) for W in self.W)
        return float(bce + reg)


class NetPipeline:
    """Duck-types the slice of sklearn's Pipeline the tournament uses:
    fit(X, y, sample_weight=...), predict_proba(X), and pipe[-1]."""

    def __init__(self, net: EvoNet):
        self.net = net

    def fit(self, X, y, sample_weight=None):
        self.net.fit(X, y, sample_weight)
        return self

    def predict_proba(self, X):
        return self.net.predict_proba(X)

    def __getitem__(self, i):
        return self.net

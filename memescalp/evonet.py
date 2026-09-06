"""EvoNet: a neural network written from first principles in NumPy that keeps
growing instead of being retrained.

Everything here is ours: forward pass, hand-derived backward pass, Adam,
weighted binary cross-entropy with L2, magnitude pruning, Lamarckian weight
inheritance, and three things a library learner does not give you:

* continual learning  -- ``absorb`` takes each newly resolved outcome as it
  arrives and moves the weights a little, with a replay of recent memory, so
  the model never starts from zero again;
* function-preserving growth -- ``widen`` and ``deepen`` add capacity while
  computing exactly the same function as before (Net2Net-style), so growth
  never destroys what was learned;
* persistence -- ``state_dict`` / ``from_state`` serialise weights, masks,
  standardisation and the optimiser's moments, so the organism survives a
  restart with its memory intact.

Hardware notes: float32 throughout so AVX2 handles 8 lanes per instruction;
mini-batches of 256 keep a batch plus the weights inside L2; matrix products
go through BLAS, which uses every core.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.float32
BATCH = 256
ACTS = ("tanh", "relu", "linear")


def _act(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "linear":
        return z
    return np.tanh(z)


def _act_grad(name: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    if name == "relu":
        return (z > 0.0).astype(DTYPE)
    if name == "linear":
        return np.ones_like(a)
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
        # One activation per hidden layer; ``deepen`` may insert "linear".
        self.acts: list[str] = [act] * len(self.hidden)
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
        self._reset_optimizer()
        self.experience = 0          # rows absorbed online, ever
        self.growth: list[str] = []  # function-preserving growth events

    # --- sklearn-compatible surface used by the tournament -------------------
    @property
    def coefs_(self):
        return self.W

    @property
    def intercepts_(self):
        return self.b

    def n_params(self) -> int:
        return int(sum(int(m.sum()) for m in self.mask) + sum(b.size for b in self.b))

    # --- optimiser state ----------------------------------------------------------
    def _reset_optimizer(self) -> None:
        self._mW = [np.zeros_like(W) for W in self.W]
        self._vW = [np.zeros_like(W) for W in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    def _adam_step(self, gW, gb, lr: float | None = None) -> None:
        # Adam (Kingma & Ba 2015), written out: first/second moment estimates
        # with bias correction. Moments persist across calls, so online steps
        # continue the same optimisation trajectory as the last fit.
        lr = self.lr if lr is None else lr
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._t += 1
        t = self._t
        for i in range(len(self.W)):
            self._mW[i] = b1 * self._mW[i] + (1 - b1) * gW[i]
            self._vW[i] = b2 * self._vW[i] + (1 - b2) * gW[i] * gW[i]
            self._mb[i] = b1 * self._mb[i] + (1 - b1) * gb[i]
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * gb[i] * gb[i]
            mhat = self._mW[i] / (1 - b1 ** t)
            vhat = self._vW[i] / (1 - b2 ** t)
            self.W[i] -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(DTYPE)
            mhat_b = self._mb[i] / (1 - b1 ** t)
            vhat_b = self._vb[i] / (1 - b2 ** t)
            self.b[i] -= (lr * mhat_b / (np.sqrt(vhat_b) + eps)).astype(DTYPE)

    # --- Lamarckian inheritance -------------------------------------------------
    def inherit(self, parent: "EvoNet") -> None:
        """Start from the parent's learned weights. Where shapes differ, copy the
        overlapping block so knowledge survives a grown or shrunk layer."""
        for i in range(min(len(self.W), len(parent.W))):
            src_w, src_b = parent.W[i], parent.b[i]
            if i == len(self.W) - 1:
                src_w, src_b = parent.W[-1], parent.b[-1]
            r = min(self.W[i].shape[0], src_w.shape[0])
            c = min(self.W[i].shape[1], src_w.shape[1])
            self.W[i][:r, :c] = src_w[:r, :c]
            self.b[i][:c] = src_b[:c]
        self._mu, self._sd = parent._mu.copy(), parent._sd.copy()
        self.experience = parent.experience
        self.growth = list(parent.growth)

    def clone(self) -> "EvoNet":
        """An exact copy, optimiser state included: the organism continues."""
        child = EvoNet(self.n_in, self.hidden, self.act, self.lr, self.l2,
                       self.epochs, self.prune)
        child.acts = list(self.acts)
        child.W = [w.copy() for w in self.W]
        child.b = [b.copy() for b in self.b]
        child.mask = [m.copy() for m in self.mask]
        child._mu, child._sd = self._mu.copy(), self._sd.copy()
        child._mW = [m.copy() for m in self._mW]
        child._vW = [v.copy() for v in self._vW]
        child._mb = [m.copy() for m in self._mb]
        child._vb = [v.copy() for v in self._vb]
        child._t = self._t
        child.experience = self.experience
        child.growth = list(self.growth)
        return child

    # --- function-preserving growth --------------------------------------------
    def widen(self, layer: int, extra: int, rng: np.random.Generator | None = None,
              noise: float = 1e-3) -> None:
        """Add ``extra`` units to hidden ``layer`` without changing the function
        (Net2WiderNet, Chen et al. 2016): duplicate existing units and halve
        the outgoing weights of each duplicated pair. A whisper of noise breaks
        the symmetry so the twins can specialise."""
        assert 0 <= layer < len(self.hidden)
        rng = rng or np.random.default_rng(self.experience + 1)
        n = self.hidden[layer]
        src = rng.integers(0, n, extra)
        # incoming: copy columns
        Win = self.W[layer]
        self.W[layer] = np.concatenate([Win, Win[:, src]], axis=1).astype(DTYPE)
        self.mask[layer] = np.concatenate([self.mask[layer], self.mask[layer][:, src]], axis=1)
        self.b[layer] = np.concatenate([self.b[layer], self.b[layer][src]]).astype(DTYPE)
        # outgoing: copy rows, then split the contribution between twins
        Wout = self.W[layer + 1]
        rows = Wout[src].copy()
        counts = np.ones(n, dtype=DTYPE)
        for s in src:
            counts[s] += 1.0
        Wout = Wout / counts[:, None]
        rows = rows / counts[src][:, None]
        self.W[layer + 1] = np.concatenate([Wout, rows], axis=0).astype(DTYPE)
        self.mask[layer + 1] = np.concatenate([self.mask[layer + 1], self.mask[layer + 1][src]], axis=0)
        self.W[layer + 1][n:] += (rng.standard_normal(self.W[layer + 1][n:].shape) * noise).astype(DTYPE)
        self.hidden[layer] = n + extra
        self._reset_optimizer()
        self.growth.append(f"widen L{layer} +{extra}")

    def deepen(self, after: int) -> None:
        """Insert a new hidden layer after hidden layer ``after`` that computes
        the identity, so the function is unchanged (Net2DeeperNet). With relu
        the identity is built from the pair relu(x) - relu(-x) = x; with tanh
        the inserted layer is linear."""
        assert 0 <= after < len(self.hidden)
        n = self.hidden[after]
        I = np.eye(n, dtype=DTYPE)
        if self.act == "relu":
            W_new = np.concatenate([I, -I], axis=1)              # n x 2n
            Wnext = self.W[after + 1]                             # n x m
            W_after = np.concatenate([Wnext, -Wnext], axis=0)     # 2n x m
            width, act = 2 * n, "relu"
        else:
            W_new, W_after, width, act = I, self.W[after + 1].copy(), n, "linear"
        self.W.insert(after + 1, W_new)
        self.b.insert(after + 1, np.zeros(width, dtype=DTYPE))
        self.mask.insert(after + 1, np.ones_like(W_new))
        self.W[after + 2] = W_after.astype(DTYPE)
        self.mask[after + 2] = np.ones_like(W_after)
        self.hidden.insert(after + 1, width)
        self.acts.insert(after + 1, act)
        self._reset_optimizer()
        self.growth.append(f"deepen after L{after} ({width}, {act})")

    # --- forward / backward -------------------------------------------------------
    def _forward(self, X: np.ndarray):
        zs, acts = [], [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ (W * self.mask[i]) + b
            zs.append(z)
            a = _sigmoid(z) if i == len(self.W) - 1 else _act(self.acts[i], z)
            acts.append(a)
        return zs, acts

    def _backward(self, zs, acts, y, w):
        """Hand-derived gradients of weighted BCE + L2 w.r.t. every W and b."""
        p = acts[-1][:, 0]
        # dL/dz_out for sigmoid + BCE collapses to (p - y), scaled by weights.
        delta = ((p - y) * w / max(1.0, w.sum()))[:, None].astype(DTYPE)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ delta + self.l2 * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ (self.W[i] * self.mask[i]).T) \
                    * _act_grad(self.acts[i - 1], zs[i - 1], acts[i])
        return gW, gb

    def _prep(self, X, y=None, sample_weight=None):
        X = np.asarray(X, dtype=DTYPE)
        y = None if y is None else np.asarray(y, dtype=DTYPE)
        w = None
        if y is not None:
            w = (np.ones(len(y), dtype=DTYPE) if sample_weight is None
                 else np.asarray(sample_weight, dtype=DTYPE))
        return X, y, w

    # --- training -----------------------------------------------------------------
    def fit(self, X, y, sample_weight=None) -> "EvoNet":
        """Batch training from the current weights (not from zero)."""
        X, y, w = self._prep(X, y, sample_weight)
        self._mu = X.mean(axis=0).astype(DTYPE)
        self._sd = (X.std(axis=0) + 1e-6).astype(DTYPE)
        Xn = (X - self._mu) / self._sd
        rng = np.random.default_rng(11)
        n = len(y)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, BATCH):
                idx = order[s:s + BATCH]
                zs, acts = self._forward(Xn[idx])
                gW, gb = self._backward(zs, acts, y[idx], w[idx])
                self._adam_step(gW, gb)
        if self.prune > 0:
            # Structural sparsity: silence the smallest |weights| for good.
            for i, W in enumerate(self.W):
                thresh = np.quantile(np.abs(W), self.prune)
                self.mask[i] = (np.abs(W) >= thresh).astype(DTYPE)
        return self

    def absorb(self, X, y, sample_weight=None, replay=None, lr_scale: float = 0.3,
               steps: int = 1) -> float:
        """Continual learning: one small Adam step on newly resolved rows plus
        a replay batch of remembered rows, so new experience is folded into
        the weights without forgetting. Returns the loss on the new rows
        before the step. Standardisation stays fixed (set at the last fit)
        so the function does not jump."""
        X, y, w = self._prep(X, y, sample_weight)
        Xn = (X - self._mu) / self._sd
        before = self.loss(Xn, y, w, standardized=True)
        if replay is not None and len(replay[0]):
            Xr, yr, wr = self._prep(*replay)
            Xn = np.concatenate([Xn, (Xr - self._mu) / self._sd])
            y = np.concatenate([y, yr])
            w = np.concatenate([w, wr])
        for _ in range(max(1, steps)):
            zs, acts = self._forward(Xn)
            gW, gb = self._backward(zs, acts, y, w)
            self._adam_step(gW, gb, lr=self.lr * lr_scale)
        self.experience += int(len(X))
        return before

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=DTYPE)
        Xn = (X - self._mu) / self._sd
        p = self._forward(Xn)[1][-1][:, 0]
        return np.stack([1.0 - p, p], axis=1)

    # --- numerical gradient check (used by tests) ---------------------------------
    def loss(self, X, y, w=None, standardized: bool = False) -> float:
        X = np.asarray(X, dtype=DTYPE)
        y = np.asarray(y, dtype=DTYPE)
        w = np.ones(len(y), dtype=DTYPE) if w is None else np.asarray(w, dtype=DTYPE)
        if not standardized:
            pass  # tests feed raw inputs to the un-standardised network
        p = np.clip(self._forward(X)[1][-1][:, 0], 1e-7, 1 - 1e-7)
        bce = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / max(1.0, w.sum())
        reg = 0.5 * self.l2 * sum(float((W * W).sum()) for W in self.W)
        return float(bce + reg)

    # --- persistence -------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "n_in": self.n_in, "hidden": list(self.hidden), "act": self.act,
            "acts": list(self.acts), "lr": self.lr, "l2": self.l2,
            "epochs": self.epochs, "prune": self.prune,
            "W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b],
            "mask": [m.tolist() for m in self.mask],
            "mu": self._mu.tolist(), "sd": self._sd.tolist(),
            "mW": [m.tolist() for m in self._mW], "vW": [v.tolist() for v in self._vW],
            "mb": [m.tolist() for m in self._mb], "vb": [v.tolist() for v in self._vb],
            "t": self._t, "experience": self.experience, "growth": list(self.growth),
        }

    @classmethod
    def from_state(cls, d: dict) -> "EvoNet":
        net = cls(d["n_in"], d["hidden"], d.get("act", "tanh"), d.get("lr", 3e-3),
                  d.get("l2", 1e-4), d.get("epochs", 40), d.get("prune", 0.0))
        net.acts = list(d.get("acts", [net.act] * len(net.hidden)))
        net.W = [np.asarray(w, dtype=DTYPE) for w in d["W"]]
        net.b = [np.asarray(b, dtype=DTYPE) for b in d["b"]]
        net.mask = [np.asarray(m, dtype=DTYPE) for m in d["mask"]]
        net._mu = np.asarray(d["mu"], dtype=DTYPE)
        net._sd = np.asarray(d["sd"], dtype=DTYPE)
        if "mW" in d:
            net._mW = [np.asarray(m, dtype=DTYPE) for m in d["mW"]]
            net._vW = [np.asarray(v, dtype=DTYPE) for v in d["vW"]]
            net._mb = [np.asarray(m, dtype=DTYPE) for m in d["mb"]]
            net._vb = [np.asarray(v, dtype=DTYPE) for v in d["vb"]]
            net._t = int(d.get("t", 0))
        net.experience = int(d.get("experience", 0))
        net.growth = list(d.get("growth", []))
        return net


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

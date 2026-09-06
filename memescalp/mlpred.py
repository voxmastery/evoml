"""Machine-learning forecaster arm: retrained repeatedly on this experiment's
own append-only logs. Open-source arsenal: scikit-learn.

Design (each element literature-grounded):
- Magnitude-aware labels: direction is trained only on moves >= MIN_LABEL_MOVE
  — short-horizon direction is noise on quiet coins; signal lives in movers.
- Forward, PURGED validation (Lopez de Prado 2018): train on the past, test
  on the newest slice, and drop train rows whose label window touches the
  holdout so overlapping horizons cannot leak.
- Recency-weighted fitting: exponential decay (tau ~6h) so the model tracks
  regime shifts instead of dragging all history equally.
- Champion/challenger tournament: logistic regression vs a small neural net
  vs gradient-boosted trees; the purged-holdout winner takes the job.
- Platt-calibrated probabilities (Platt 1999): a held-out calibration slice
  maps raw model scores to honest probabilities, so Kelly sizing and the
  call threshold mean what they say.
- Live movement gate + drift monitor + self-benching (Chow 1970 reject
  option): only call on coins that are actually moving, demand more evidence
  when live accuracy runs cold, refuse entirely when holdout drops to chance.
"""
from __future__ import annotations

import json
import logging
import math
import random as _random
import time
from collections import deque as _deque

from .catalog import CatalogEntry
from .db import Database
from .evonet import EvoNet, NetPipeline
from .indicators import ema, one_minute_closes
from .mathpred import signal_for, variance_ratio_2, zscore_20
from .micro import MicroFeatures, micro_features
from .predictor import Prediction

log = logging.getLogger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"

MIN_TRAIN_ROWS = 300
MIN_CALL_PROB = 0.58
HOLDOUT_FRACTION = 0.2
CAL_FRACTION = 0.25       # tail of the train slice reserved for Platt fitting
RECENCY_TAU_S = 6 * 3600.0
MIN_LABEL_MOVE = 0.15     # % — below this a 15-min move is noise
GLITCH_MOVE = 200.0       # % — beyond this a print is a data glitch
MOVE_GATE_VOL = 0.03      # live gate: micro_vol >= this ...
MOVE_GATE_RET = 0.15      # ... or |5-min return| >= this (%)

# --- self-monitoring (drift detection + selective prediction) ---------------
ROLLING_WINDOW = 30
ROLLING_MIN_N = 10
SELF_BENCH_HOLDOUT = 0.52
CAUTIOUS_PROB = 0.65
BOLD_PROB = 0.55

FEATURE_NAMES = [
    "chg_m5", "chg_h1", "chg_h24", "vol_1m", "log_liq", "turnover",
    "ret_60s", "ret_300s", "uptick", "buy_ratio",
    "hour_sin", "hour_cos", "sol_ret5",
    "z20", "vr2c", "ema_gap", "strat_sig",
    "tsfm_ret", "tsfm_spread",
]


def strategy_features(closes: list[float]) -> tuple[float, float, float, float]:
    """The classical strategies, served to the ML model as features
    (stacked generalization, Wolpert 1992): the model learns WHEN the old
    formulas work instead of merely competing with them."""
    if len(closes) < 25:
        return 0.0, 0.0, 0.0, 0.0
    z = zscore_20(closes) or 0.0
    z = max(-5.0, min(5.0, z))
    vr = variance_ratio_2(closes)
    vr_c = (vr - 1.0) if vr is not None else 0.0
    e5, e20 = ema(closes, 5), ema(closes, 20)
    gap = ((e5 - e20) / e20 * 100.0
           if e5 is not None and e20 is not None and e20 > 0 else 0.0)
    gap = max(-20.0, min(20.0, gap))
    sig = signal_for(closes)
    strat = 0.0
    if sig is not None:
        direction, strength, _ = sig
        strat = strength if direction == "UP" else -strength
    return z, vr_c, gap, strat


def adaptive_threshold(rolling_acc: float | None,
                       holdout: float | None,
                       base: float = MIN_CALL_PROB) -> float | None:
    """Self-regulation rule; None means self-benched entirely. `base` is the
    genome's own evolved call threshold, used in the normal regime."""
    if holdout is not None and holdout < SELF_BENCH_HOLDOUT:
        return None
    if rolling_acc is not None:
        if rolling_acc < 0.50:
            return CAUTIOUS_PROB
        if rolling_acc >= 0.60 and (holdout or 0.0) >= 0.55:
            return BOLD_PROB
    return base


def context_features(db: Database, ts: float) -> tuple[float, float, float]:
    """Time-of-day encoding + SOL 5-minute return (market regime)."""
    hour = (ts % 86400.0) / 86400.0 * 2.0 * math.pi
    sol_ret5 = 0.0
    p0 = db.price_at_or_after(SOL_MINT, ts - 300.0)
    p1 = db.price_at_or_after(SOL_MINT, ts - 10.0)
    if p0 and p1 and p0[1] > 0 and p0[0] < ts:
        sol_ret5 = (p1[1] / p0[1] - 1.0) * 100.0
    return math.sin(hour), math.cos(hour), sol_ret5


def feature_vector(chg_m5: float, chg_h1: float, chg_h24: float,
                   vol_1m: float, liquidity: float, turnover: float,
                   micro: MicroFeatures | None,
                   buys: int | None, sells: int | None,
                   hour_sin: float = 0.0, hour_cos: float = 0.0,
                   sol_ret5: float = 0.0,
                   strat: tuple[float, float, float, float] = (0, 0, 0, 0),
                   tsfm: tuple[float, float] = (0.0, 0.0),
                   ) -> list[float]:
    total_tx = (buys or 0) + (sells or 0)
    return [
        chg_m5,
        chg_h1,
        max(-200.0, min(chg_h24, 200.0)) / 10.0,
        vol_1m,
        math.log10(max(liquidity, 1.0)),
        min(turnover, 100.0),
        micro.ret_60s if micro else 0.0,
        micro.ret_300s if micro else 0.0,
        (micro.uptick_ratio - 0.5) * 2.0 if micro else 0.0,
        (buys / total_tx - 0.5) * 2.0 if total_tx else 0.0,
        hour_sin, hour_cos, sol_ret5,
        strat[0], strat[1], strat[2], strat[3],
        max(-20.0, min(20.0, tsfm[0])), min(tsfm[1], 50.0),
    ]


def build_dataset(db: Database, horizon_s: float, max_rows: int = 8000):
    """(features, labels, timestamps) in ascending time order."""
    now = time.time()
    rows = db.catalog_history(now - horizon_s - 90.0, limit=max_rows)
    rows.reverse()  # catalog_history returns newest-first; train wants time order
    X: list[list[float]] = []
    y: list[int] = []
    ts_list: list[float] = []
    for r in rows:
        p0 = db.price_at_or_after(r["mint"], r["ts"])
        p1 = db.price_at_or_after(r["mint"], r["ts"] + horizon_s)
        if p0 is None or p1 is None:
            continue
        if p1[0] - (r["ts"] + horizon_s) > 60.0 or p0[1] <= 0:
            continue
        move = (p1[1] / p0[1] - 1.0) * 100.0
        if abs(move) < MIN_LABEL_MOVE or abs(move) > GLITCH_MOVE:
            continue  # quiet = noise; extreme = glitch print
        ticks = db.price_range(r["mint"], r["ts"] - 1560.0, r["ts"])
        micro = micro_features(ticks)
        strat = strategy_features(one_minute_closes(ticks))
        flow = db.flow_before(r["mint"], r["ts"])
        turnover = (r["volume_h24"] / r["liquidity_usd"]
                    if r["liquidity_usd"] > 0 else 0.0)
        p5 = db.price_at_or_after(r["mint"], r["ts"] - 300.0)
        chg_m5 = ((p0[1] / p5[1] - 1.0) * 100.0
                  if p5 is not None and p5[1] > 0 and p5[0] < r["ts"] else 0.0)
        hs, hc, sol5 = context_features(db, r["ts"])
        tsfm = db.tsfm_before(r["mint"], r["ts"]) or (0.0, 0.0)
        X.append(feature_vector(
            chg_m5, r["price_change_h1"], r["price_change_h24"],
            r["volatility_pct"], r["liquidity_usd"], turnover, micro,
            flow[0] if flow else None, flow[1] if flow else None,
            hs, hc, sol5, strat, tsfm,
        ))
        y.append(1 if move > 0 else 0)
        ts_list.append(r["ts"])
    return X, y, ts_list


# --- EvoML: self-evolving architecture -----------------------------------------
# The champion's genome (model kind + hyperparameters) is persisted; every
# retrain it spawns mutated offspring plus a random immigrant, all face the
# purged holdout, and the survivor's genome seeds the next generation
# (neuroevolution / population-based training, made small and local).

META_GENOME = "ml_genome"
META_GENERATION = "ml_generation"
META_HALL = "ml_hall_of_fame"
META_NET_STATE = "ml_net_state"     # the organism: weights, moments, calibration
META_GROWTH = "ml_growth"           # experience / params / growth events (for the UI)
REPLAY_SIZE = 4000                  # rows of remembered experience for online replay
REPLAY_BATCH = 128
GROW_STEPS = 2                      # Adam steps per absorbed window
PENDING_MAX_AGE_S = 2 * 3600.0
GROW_EVERY_ROWS = 1500              # widen the body every N rows of experience
GROW_FRACTION = 0.25                # by this share of the smallest hidden layer
MAX_PARAMS = 50_000                 # hard ceiling so growth stays honest
HALL_SIZE = 5
SUCCESSION_SE_MULT = 1.0   # challenger must beat incumbent by > 1 SE
NUM_FEATURES = len(FEATURE_NAMES)
MIN_FEATURE_GENES = 6
MAX_SYNTH_GENES = 8
DEFAULT_GENOME = {"kind": "logreg", "C": 0.5}

# --- invented genes: genetic programming for feature construction (Koza 1992)
# A synth gene is a tiny expression over base features, e.g. ("mul", 9, 6)
# = buy_ratio * ret_60s. The model invents them, tests them on itself, and
# keeps the ones whose lineage survives the purged holdout.
UNARY_OPS = ("neg", "abs", "sq", "sign")
BINARY_OPS = ("mul", "sub", "div", "max", "min")


def eval_synth(expr: list | tuple, base: list[float]) -> float:
    op = expr[0]
    a = base[int(expr[1])] if len(expr) > 1 else 0.0
    if op in UNARY_OPS:
        v = {"neg": -a, "abs": abs(a), "sq": a * a,
             "sign": (1.0 if a > 0 else -1.0 if a < 0 else 0.0)}[op]
    else:
        b = base[int(expr[2])]
        if op == "mul":
            v = a * b
        elif op == "sub":
            v = a - b
        elif op == "div":
            v = a / (abs(b) + 1e-6)
        elif op == "max":
            v = max(a, b)
        else:
            v = min(a, b)
    return max(-1e3, min(1e3, float(v)))


def synth_str(expr: list | tuple) -> str:
    names = [FEATURE_NAMES[int(i)] for i in expr[1:]]
    return f"{expr[0]}({', '.join(names)})"


def apply_genome(g: dict, base: list[float]) -> list[float]:
    """The genome's sensory chromosome + its invented genes -> model input."""
    return ([base[i] for i in g["features"]]
            + [eval_synth(e, base) for e in g.get("synth", [])])


def gene_report(X: list[list[float]], y: list[int]) -> list[float]:
    """Characteristics of each base gene: point-biserial correlation with
    the label. Evolution reads this to decide what to combine and invent."""
    n = len(X)
    if n < 10:
        return [0.0] * NUM_FEATURES
    ym = sum(y) / n
    ysd = math.sqrt(sum((v - ym) ** 2 for v in y) / n) or 1.0
    out = []
    for j in range(NUM_FEATURES):
        col = [row[j] for row in X]
        xm = sum(col) / n
        xsd = math.sqrt(sum((v - xm) ** 2 for v in col) / n)
        if xsd == 0:
            out.append(0.0)
            continue
        cov = sum((a - xm) * (b - ym) for a, b in zip(col, y)) / n
        out.append(cov / (xsd * ysd))
    return out


def _ensure_genome_defaults(g: dict) -> dict:
    """Self-modification genes (self-adaptive ES, Rechenberg/Schwefel):
    - features: which inputs this lineage uses (its sensory chromosome)
    - mut_rate: how aggressively its own genome mutates (mutation of mutation)
    - thr: the call threshold governing how the model is USED, not just built
    """
    g.setdefault("features", list(range(NUM_FEATURES)))
    g["features"] = sorted(int(i) for i in g["features"] if 0 <= int(i) < NUM_FEATURES)
    g.setdefault("mut_rate", 1.0)
    g.setdefault("thr", MIN_CALL_PROB)
    g.setdefault("synth", [])
    # Epistemic-humility genes: a probability is only as good as the evidence
    # density behind it. max_conf refuses calls that are "too sure" (usually
    # extrapolation); min_fam refuses calls on unfamiliar inputs.
    g.setdefault("max_conf", 0.95)
    g.setdefault("min_fam", 0.0)
    # Polymorphic genome: one genotype, three phenotypes expressed by regime.
    g.setdefault("poly", False)
    return g


# --- epistemic humility: familiarity of an input relative to training data --
FAM_Z = 2.5
REGIME_VR_INDEX = FEATURE_NAMES.index("vr2c")
REGIME_NAMES = ("trend", "revert", "walk")


def fit_familiarity(X: list[list[float]]) -> tuple[list[float], list[float]]:
    n = max(1, len(X))
    mu = [sum(r[j] for r in X) / n for j in range(NUM_FEATURES)]
    sd = [math.sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / n)
          for j in range(NUM_FEATURES)]
    return mu, sd


def familiarity(x: list[float], mu: list[float], sd: list[float]) -> float:
    """Share of base genes lying within FAM_Z sigma of the training bulk."""
    inside = 0
    for j in range(NUM_FEATURES):
        if sd[j] == 0 or abs((x[j] - mu[j]) / sd[j]) <= FAM_Z:
            inside += 1
    return inside / NUM_FEATURES


def regime_of(x: list[float]) -> str:
    """Environment the phenotype is expressed in, from the variance-ratio gene."""
    vr = x[REGIME_VR_INDEX]
    if vr >= 0.05:
        return "trend"
    if vr <= -0.05:
        return "revert"
    return "walk"


def genome_label(g: dict) -> str:
    kind = g["kind"]
    if kind == "logreg":
        return (f"logreg(C={g.get('C', 0.5):g})"
                + _genome_suffix(g))
    if kind == "net":
        return (f"net{'x'.join(str(h) for h in g.get('hidden', [16, 8]))}"
                f"({g.get('act', 'tanh')},lr={g.get('lr', 3e-3):g}"
                f"{',p=' + format(g['prune'], '.1f') if g.get('prune') else ''})"
                + _genome_suffix(g))
    if kind == "mlp":
        return (f"mlp{'x'.join(str(h) for h in g.get('hidden', [16, 8]))}"
                f"(a={g.get('alpha', 1e-4):g})" + _genome_suffix(g))
    return (f"hgb(d={g.get('depth', 3)},lr={g.get('lr', 0.08):g},"
            f"i={g.get('iters', 120)})" + _genome_suffix(g))


def _genome_suffix(g: dict) -> str:
    nf = len(g.get("features", [])) or NUM_FEATURES
    ns = len(g.get("synth", []))
    extra = ""
    if g.get("max_conf", 0.95) < 0.95:
        extra += f",cap={g['max_conf']:.2f}"
    if g.get("min_fam", 0.0) > 0:
        extra += f",fam={g['min_fam']:.1f}"
    if g.get("poly"):
        extra += ",POLY"
    return f"[{nf}f+{ns}s,thr={g.get('thr', MIN_CALL_PROB):.2f}{extra}]"


def genome_to_pipeline(g: dict):
    """Build (pipeline, sample_weight_kwarg_or_None) from a genome."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    kind = g["kind"]
    if kind == "net":
        # Our own network, written from first principles (see evonet.py).
        n_in = (len(g.get("features", list(range(NUM_FEATURES))))
                + len(g.get("synth", [])))
        net = EvoNet(n_in,
                     hidden=[int(h) for h in g.get("hidden", [16, 8])],
                     act=g.get("act", "tanh"), lr=float(g.get("lr", 3e-3)),
                     l2=float(g.get("l2", 1e-4)), epochs=int(g.get("epochs", 40)),
                     prune=float(g.get("prune", 0.0)))
        return NetPipeline(net), "sample_weight"
    if kind == "logreg":
        model = LogisticRegression(max_iter=1000, C=float(g.get("C", 0.5)))
        wkey = "logisticregression__sample_weight"
    elif kind == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=tuple(int(h) for h in g.get("hidden", [16, 8])),
            alpha=float(g.get("alpha", 1e-4)), max_iter=600, random_state=7)
        wkey = None  # MLP has no sample_weight support
    else:
        model = HistGradientBoostingClassifier(
            max_iter=int(g.get("iters", 120)), max_depth=int(g.get("depth", 3)),
            learning_rate=float(g.get("lr", 0.08)), random_state=7)
        wkey = "histgradientboostingclassifier__sample_weight"
    return make_pipeline(StandardScaler(), model), wkey


def _mutate_hyper(g: dict, rng: _random.Random) -> None:
    kind = g["kind"]
    if kind == "net":
        hidden = [int(h) for h in g.get("hidden", [16, 8])]
        op = rng.choice(["grow", "shrink", "add", "drop", "lr", "l2", "act",
                         "prune", "epochs"])
        if op == "grow":
            i = rng.randrange(len(hidden)); hidden[i] = min(64, hidden[i] * 2)
        elif op == "shrink":
            i = rng.randrange(len(hidden)); hidden[i] = max(4, hidden[i] // 2)
        elif op == "add" and len(hidden) < 3:
            hidden.append(max(4, hidden[-1] // 2))
        elif op == "drop" and len(hidden) > 1:
            hidden.pop()
        elif op == "lr":
            g["lr"] = min(3e-2, max(1e-4, g.get("lr", 3e-3) * rng.choice([0.5, 2.0])))
        elif op == "l2":
            g["l2"] = min(1e-1, max(1e-6, g.get("l2", 1e-4) * rng.choice([0.3, 3.0])))
        elif op == "act":
            g["act"] = "relu" if g.get("act", "tanh") == "tanh" else "tanh"
        elif op == "prune":
            g["prune"] = min(0.5, max(0.0, g.get("prune", 0.0) + rng.choice([-0.1, 0.1])))
        else:
            g["epochs"] = min(120, max(10, int(g.get("epochs", 40) * rng.choice([0.5, 1.5]))))
        g["hidden"] = hidden
        return
    if kind == "logreg":
        g["C"] = min(10.0, max(0.01, g.get("C", 0.5) * rng.choice([0.5, 2.0])))
    elif kind == "mlp":
        hidden = [int(h) for h in g.get("hidden", [16, 8])]
        op = rng.choice(["grow", "shrink", "add", "drop"])
        if op == "grow":
            i = rng.randrange(len(hidden)); hidden[i] = min(64, hidden[i] * 2)
        elif op == "shrink":
            i = rng.randrange(len(hidden)); hidden[i] = max(4, hidden[i] // 2)
        elif op == "add" and len(hidden) < 3:
            hidden.append(max(4, hidden[-1] // 2))
        elif op == "drop" and len(hidden) > 1:
            hidden.pop()
        g["hidden"] = hidden
        g["alpha"] = min(0.1, max(1e-6,
                                  g.get("alpha", 1e-4) * rng.choice([0.3, 3.0])))
    else:
        g["depth"] = min(6, max(2, int(g.get("depth", 3)) + rng.choice([-1, 1])))
        g["lr"] = min(0.3, max(0.01, g.get("lr", 0.08) * rng.choice([2 / 3, 1.5])))
        g["iters"] = min(400, max(40, int(g.get("iters", 120) * rng.choice([0.5, 1.5]))))


def _pick_gene(rng: _random.Random, report: list[float] | None) -> int:
    """Choose a base gene to build with: usually one whose characteristics
    (|correlation with outcome|) look informative, sometimes at random."""
    if report and rng.random() < 0.7:
        ranked = sorted(range(NUM_FEATURES), key=lambda i: -abs(report[i]))
        return rng.choice(ranked[:6])
    return rng.randrange(NUM_FEATURES)


def invent_gene(rng: _random.Random, report: list[float] | None) -> list:
    if rng.random() < 0.35:
        return [rng.choice(UNARY_OPS), _pick_gene(rng, report)]
    a, b = _pick_gene(rng, report), _pick_gene(rng, report)
    return [rng.choice(BINARY_OPS), a, b]


def mutate_genome(g: dict, rng: _random.Random,
                  report: list[float] | None = None) -> dict:
    """Self-adaptive mutation: the genome's own mut_rate gene decides how
    many edits it receives, and every gene class — hyperparameters, the
    feature chromosome, the call threshold, mut_rate itself, and the
    invention or removal of synthetic genes — is fair game."""
    g = _ensure_genome_defaults(json.loads(json.dumps(g)))
    for _ in range(max(1, round(g["mut_rate"]))):
        ops = ["hyper", "feat", "thr", "rate", "invent", "kill",
               "humility", "poly"]
        weights = [4, 2, 1, 1, 2 if len(g["synth"]) < MAX_SYNTH_GENES else 0,
                   1 if g["synth"] else 0, 2, 1]
        op = rng.choices(ops, weights=weights)[0]
        if op == "hyper":
            _mutate_hyper(g, rng)
        elif op == "humility":
            if rng.random() < 0.5:
                g["max_conf"] = min(0.95, max(0.62,
                                              g["max_conf"] + rng.choice([-0.03, 0.03])))
            else:
                g["min_fam"] = min(0.9, max(0.0,
                                            g["min_fam"] + rng.choice([-0.1, 0.1])))
        elif op == "poly":
            g["poly"] = not g["poly"]
        elif op == "invent":
            g["synth"] = g["synth"] + [invent_gene(rng, report)]
        elif op == "kill":
            g["synth"] = [s for k, s in enumerate(g["synth"])
                          if k != rng.randrange(len(g["synth"]))]
        elif op == "feat":
            feats = set(g["features"])
            i = rng.randrange(NUM_FEATURES)
            if i in feats and len(feats) > MIN_FEATURE_GENES:
                feats.remove(i)
            else:
                feats.add(i)
            g["features"] = sorted(feats)
        elif op == "thr":
            g["thr"] = min(0.70, max(0.54,
                                     g["thr"] + rng.choice([-0.02, 0.02])))
        else:
            g["mut_rate"] = min(3.0, max(0.5,
                                         g["mut_rate"] * rng.choice([1 / 1.3, 1.3])))
    return g


def random_genome(rng: _random.Random, exclude_kind: str | None = None) -> dict:
    kinds = [k for k in ("logreg", "mlp", "hgb", "net") if k != exclude_kind]
    kind = rng.choice(kinds)
    if kind == "net":
        g = {"kind": "net", "hidden": rng.choice([[16, 8], [32, 16], [24, 12, 6]]),
             "act": rng.choice(["tanh", "relu"]), "lr": rng.choice([1e-3, 3e-3, 1e-2]),
             "l2": rng.choice([1e-5, 1e-4, 1e-3]), "epochs": 40, "prune": 0.0}
    elif kind == "logreg":
        g = {"kind": "logreg", "C": rng.choice([0.1, 0.5, 2.0])}
    elif kind == "mlp":
        g = {"kind": "mlp",
             "hidden": rng.choice([[8], [16, 8], [32, 16], [24, 12, 6]]),
             "alpha": rng.choice([1e-5, 1e-4, 1e-3])}
    else:
        g = {"kind": "hgb", "depth": rng.choice([2, 3, 4]),
             "lr": rng.choice([0.05, 0.08, 0.12]),
             "iters": rng.choice([80, 120, 200])}
    return _ensure_genome_defaults(g)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def count_parameters(pipeline) -> int:
    """Best-effort learned-parameter count for the fitted champion."""
    try:
        model = pipeline[-1]
        if hasattr(model, "coefs_"):           # MLP
            return int(sum(w.size for w in model.coefs_)
                       + sum(b.size for b in model.intercepts_))
        if hasattr(model, "coef_"):            # logistic regression
            return int(model.coef_.size + model.intercept_.size)
        if hasattr(model, "_predictors"):      # HistGradientBoosting
            return int(sum(p.nodes.shape[0] for stage in model._predictors
                           for p in stage))
    except Exception:
        pass
    return 0


class MLForecaster:
    def __init__(self):
        self._pipeline = None
        self._platt = None            # (a, b): p' = sigmoid(a * logit(p) + b)
        self._genome: dict | None = None
        self._pipes: dict | None = None
        self._parent_net: EvoNet | None = None
        self._fam: tuple[list[float], list[float]] | None = None
        self._base_thr: float = MIN_CALL_PROB
        self.n_train = 0
        self.n_params = 0
        self.holdout_accuracy: float | None = None
        self.last_trained: float | None = None
        self.champion: str = "none"
        self.candidate_scores: dict[str, float] = {}
        # Continual growth: base feature vectors stashed per prediction window
        # until their outcome exists, a replay memory of absorbed rows, and a
        # running count of experience folded into the weights online.
        self._pending: dict[float, list[tuple[str, list[float]]]] = {}
        self._replay: _deque = _deque(maxlen=REPLAY_SIZE)
        self.experience = 0
        self.growth_events: list[str] = []
        self.last_absorbed: float | None = None

    # --- the organism: persistence and continual learning ------------------------
    def _net(self) -> EvoNet | None:
        return (self._pipeline.net
                if isinstance(self._pipeline, NetPipeline) else None)

    def _save_state(self, db: Database) -> None:
        """Persist the champion network with its optimiser moments, calibration
        and familiarity statistics, so a restart resumes the same organism."""
        net = self._net()
        if net is None or self._genome is None:
            return
        state = {
            "net": net.state_dict(), "platt": self._platt, "fam": self._fam,
            "genome": self._genome, "champion": self.champion,
            "holdout": self.holdout_accuracy, "n_train": self.n_train,
            "saved_at": time.time(),
        }
        db.set_meta(META_NET_STATE, json.dumps(state))
        db.set_meta(META_GROWTH, json.dumps({
            "experience": net.experience, "params": net.n_params(),
            "hidden": net.hidden, "acts": net.acts, "growth": net.growth,
            "champion": self.champion, "updated": time.time(),
            "last_absorbed": self.last_absorbed,
        }))

    def load_state(self, db: Database) -> bool:
        """Resume the persisted organism instead of starting from zero."""
        raw = db.get_meta(META_NET_STATE)
        if not raw:
            return False
        try:
            state = json.loads(raw)
            net = EvoNet.from_state(state["net"])
        except Exception:
            log.exception("could not restore the persisted network")
            return False
        self._pipeline = NetPipeline(net)
        self._pipes = {"global": self._pipeline}
        self._parent_net = net
        self._platt = tuple(state["platt"]) if state.get("platt") else None
        self._fam = tuple(state["fam"]) if state.get("fam") else None
        self._genome = _ensure_genome_defaults(state["genome"])
        self._base_thr = self._genome["thr"]
        self.champion = state.get("champion", genome_label(self._genome))
        self.holdout_accuracy = state.get("holdout")
        self.n_train = int(state.get("n_train", 0))
        self.n_params = net.n_params()
        self.experience = net.experience
        self.growth_events = list(net.growth)
        self.last_trained = state.get("saved_at")
        return True

    def remember_window(self, ts: float, rows: list[tuple[str, list[float]]]) -> None:
        if rows:
            self._pending[ts] = rows
        stale = [t for t in self._pending if ts - t > PENDING_MAX_AGE_S]
        for t in stale:
            del self._pending[t]

    def grow_step(self, db: Database, horizon_s: float) -> dict:
        """Fold every newly resolved window into the champion's weights: label
        the stashed feature vectors against the real price at the horizon,
        take a small Adam step with a replay of remembered rows, persist.
        The organism learns from every outcome as it arrives; the tournament
        only decides whether a grown or mutated body would learn better."""
        now = time.time()
        ready = [t for t in self._pending
                 if t + horizon_s + 90.0 <= now]
        out = {"windows": len(ready), "rows": 0, "absorbed": 0, "loss": None}
        if not ready:
            return out
        Xn: list[list[float]] = []
        yn: list[int] = []
        for t in sorted(ready):
            rows = self._pending.pop(t)
            for mint, xb in rows:
                p0 = db.price_at_or_after(mint, t)
                p1 = db.price_at_or_after(mint, t + horizon_s)
                if p0 is None or p1 is None or p0[1] <= 0:
                    continue
                if p1[0] - (t + horizon_s) > 60.0:
                    continue
                move = (p1[1] / p0[1] - 1.0) * 100.0
                if abs(move) < MIN_LABEL_MOVE or abs(move) > GLITCH_MOVE:
                    continue
                Xn.append(apply_genome(self._genome, xb) if self._genome else xb)
                yn.append(1 if move > 0 else 0)
        out["rows"] = len(Xn)
        net = self._net()
        if net is None or not Xn:
            for xm, yi in zip(Xn, yn):
                self._replay.append((xm, yi))
            return out
        replay = None
        if self._replay:
            rng = _random.Random(int(now))
            sample = rng.sample(list(self._replay), min(REPLAY_BATCH, len(self._replay)))
            replay = ([r[0] for r in sample], [r[1] for r in sample],
                      [1.0] * len(sample))
        try:
            out["loss"] = net.absorb(Xn, yn, replay=replay, steps=GROW_STEPS)
        except Exception:
            log.exception("online absorb failed; the organism keeps its weights")
            return out
        for xm, yi in zip(Xn, yn):
            self._replay.append((xm, yi))
        out["absorbed"] = len(Xn)
        # Growth with experience: every GROW_EVERY_ROWS rows the body widens its
        # smallest hidden layer without changing what it computes. Capacity
        # rises as a matter of course; the tournament prunes it back only if a
        # smaller body is measurably better.
        before_bucket = (net.experience - len(Xn)) // GROW_EVERY_ROWS
        after_bucket = net.experience // GROW_EVERY_ROWS
        if after_bucket > before_bucket and net.n_params() < MAX_PARAMS:
            layer = min(range(len(net.hidden)), key=lambda i: net.hidden[i])
            extra = max(2, int(net.hidden[layer] * GROW_FRACTION))
            net.widen(layer, extra)
            if self._genome is not None:
                self._genome["hidden"] = list(net.hidden)
            self.growth_events = list(net.growth)
            out["grew"] = net.growth[-1]
            log.info("[ml] organism grew: %s -> hidden=%s params=%d",
                     net.growth[-1], net.hidden, net.n_params())
        self.experience = net.experience
        self.n_params = net.n_params()
        self.last_absorbed = now
        self._save_state(db)
        return out

    @property
    def trained(self) -> bool:
        return self._pipeline is not None

    def _calibrated(self, raw_p: float) -> float:
        if self._platt is None:
            return raw_p
        a, b = self._platt
        z = a * _logit(raw_p) + b
        return 1.0 / (1.0 + math.exp(-z))

    def retrain(self, db: Database, horizon_s: float) -> dict:
        """EvoML generation: the persisted champion genome spawns mutated
        offspring plus a random immigrant; all fit with recency weights, get
        Platt-calibrated, and face the purged forward holdout. The survivor
        becomes the champion and seeds the next generation. Worker-thread."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        X, y, ts = build_dataset(db, horizon_s)
        result = {"rows": len(X), "trained": False, "holdout_accuracy": None,
                  "champion": None, "scores": {}, "n_params": 0,
                  "generation": None}
        if len(X) < MIN_TRAIN_ROWS or len(set(y)) < 2:
            return result

        split = max(1, int(len(X) * (1.0 - HOLDOUT_FRACTION)))
        if split >= len(X):
            return result
        # Purge: train rows whose label window reaches into the holdout leak.
        purge_before = ts[split] - (horizon_s + 60.0)
        train_idx = [i for i in range(split) if ts[i] <= purge_before]
        if len(train_idx) < MIN_TRAIN_ROWS // 2:
            return result
        cal_n = max(20, int(len(train_idx) * CAL_FRACTION))
        fit_idx, cal_idx = train_idx[:-cal_n], train_idx[-cal_n:]
        if (len(fit_idx) < 50 or len(set(y[i] for i in fit_idx)) < 2
                or len(set(y[i] for i in cal_idx)) < 2):
            return result

        Xf = [X[i] for i in fit_idx]
        yf = [y[i] for i in fit_idx]
        t_max = ts[fit_idx[-1]]
        weights = np.array([math.exp(-(t_max - ts[i]) / RECENCY_TAU_S)
                            for i in fit_idx])
        Xc, yc = [X[i] for i in cal_idx], [y[i] for i in cal_idx]
        Xh, yh = X[split:], y[split:]

        # --- assemble this generation's population ---------------------------
        generation = int(db.get_meta(META_GENERATION) or 0)
        raw = db.get_meta(META_GENOME)
        champion_genome = _ensure_genome_defaults(
            json.loads(raw) if raw else dict(DEFAULT_GENOME))
        rng = _random.Random(generation * 7919 + 17)
        report = gene_report(Xf, yf)   # read the genes before editing them
        # Seed bank: half the time the immigrant is a resurrected past champion
        # from the hall of fame, so a strong lineage lost to one noisy
        # tournament can return and compete again.
        hall = json.loads(db.get_meta(META_HALL) or "[]")
        if hall and rng.random() < 0.5:
            immigrant = _ensure_genome_defaults(dict(rng.choice(hall)["genome"]))
        else:
            immigrant = random_genome(rng, exclude_kind=champion_genome["kind"])
        population = [champion_genome,
                      mutate_genome(champion_genome, rng, report),
                      mutate_genome(champion_genome, rng, report),
                      immigrant]
        # Function-preserving growth children: the living champion's body,
        # widened or deepened without changing what it computes, then trained
        # on. Growth only sticks if the bigger body learns measurably better.
        parent = self._parent_net
        if parent is not None and champion_genome["kind"] == "net":
            layer = rng.randrange(len(parent.hidden))
            extra = max(2, parent.hidden[layer] // 2)
            wide = json.loads(json.dumps(champion_genome))
            wide["hidden"] = list(parent.hidden)
            wide["hidden"][layer] += extra
            wide["grow"] = ["widen", layer, extra]
            deep = json.loads(json.dumps(champion_genome))
            deep["hidden"] = list(parent.hidden)
            deep["hidden"].insert(layer + 1, parent.hidden[layer] * (2 if parent.act == "relu" else 1))
            deep["grow"] = ["deepen", layer]
            population += [wide, deep]
        seen: set[str] = set()
        candidates: dict[str, tuple] = {}
        genomes: dict[str, dict] = {}
        for g in population:
            key = json.dumps(g, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            label = genome_label(g)
            pipe, wkey = genome_to_pipeline(g)
            candidates[label] = (pipe, wkey)
            genomes[label] = g
        scores: dict[str, float] = {}
        fitted: dict[str, tuple] = {}
        fam_mu, fam_sd = fit_familiarity(Xf)
        Xh_fam = [familiarity(r, fam_mu, fam_sd) for r in Xh]
        Xh_reg = [regime_of(r) for r in Xh]
        for name, (pipe, weight_key) in candidates.items():
            try:
                gen = genomes[name]
                Xf_m = [apply_genome(gen, r) for r in Xf]
                Xc_m = [apply_genome(gen, r) for r in Xc]
                Xh_m = [apply_genome(gen, r) for r in Xh]
                kwargs = {weight_key: weights} if weight_key else {}
                # The organism continues: the incumbent net is its own clone
                # (weights, moments, experience); growth children are exact
                # function-preserving copies that gained capacity; other net
                # children inherit overlapping weight blocks.
                if isinstance(pipe, NetPipeline) and self._parent_net is not None:
                    grow = gen.get("grow")
                    same_body = (gen["hidden"] == self._parent_net.hidden
                                 and gen.get("features") == champion_genome.get("features")
                                 and gen.get("synth") == champion_genome.get("synth"))
                    if grow and same_body is False and gen.get("features") == champion_genome.get("features"):
                        body = self._parent_net.clone()
                        if grow[0] == "widen":
                            body.widen(int(grow[1]), int(grow[2]))
                        else:
                            body.deepen(int(grow[1]))
                        body.epochs = max(4, min(body.epochs, 12))
                        pipe = NetPipeline(body)
                    elif same_body and not grow:
                        body = self._parent_net.clone()
                        body.epochs = max(4, min(body.epochs, 12))
                        pipe = NetPipeline(body)
                    else:
                        pipe.net.inherit(self._parent_net)
                pipe.fit(Xf_m, yf, **kwargs)
                pipes = {"global": pipe}
                if gen.get("poly"):
                    # Express a separate phenotype per regime where data allows.
                    regs = [regime_of(r) for r in Xf]
                    for reg in REGIME_NAMES:
                        idx = [i for i, rg in enumerate(regs) if rg == reg]
                        ys = [yf[i] for i in idx]
                        if len(idx) >= 100 and len(set(ys)) == 2:
                            sub, _ = genome_to_pipeline(gen)
                            sub_kw = ({weight_key: weights[idx]}
                                      if weight_key else {})
                            sub.fit([Xf_m[i] for i in idx], ys, **sub_kw)
                            pipes[reg] = sub

                def proba(xm, xb):
                    p = pipes.get(regime_of(xb), pipes["global"])
                    return float(p.predict_proba([xm])[0][1])

                # Platt calibration on the reserved slice.
                cal_probs = [proba(xm, xb) for xm, xb in zip(Xc_m, Xc)]
                lr = LogisticRegression(max_iter=1000)
                lr.fit([[_logit(p)] for p in cal_probs], yc)
                a = float(lr.coef_[0][0])
                b = float(lr.intercept_[0])
                # Policy-aware fitness: judge the genome by the calls it would
                # actually make under its own threshold / humility genes.
                overall = called = called_ok = 0
                for xm, xb, yi, fam in zip(Xh_m, Xh, yh, Xh_fam):
                    z = a * _logit(proba(xm, xb)) + b
                    p_cal = 1.0 / (1.0 + math.exp(-z))
                    hit = (p_cal >= 0.5) == bool(yi)
                    overall += hit
                    conf = max(p_cal, 1.0 - p_cal)
                    if (conf >= gen["thr"] and conf <= gen["max_conf"]
                            and fam >= gen["min_fam"]):
                        called += 1
                        called_ok += hit
                scores[name] = (called_ok / called if called >= 30
                                else overall / len(Xh))
                fitted[name] = (pipes, (a, b))
            except Exception:
                log.exception("candidate %s failed to train", name)
        if not scores:
            return result

        # Significance-gated succession: a challenger must beat the incumbent
        # by more than the holdout's noise floor, or incumbency holds.
        champion = max(scores, key=scores.get)
        incumbent = genome_label(champion_genome)
        succession = "new lineage"
        noise = SUCCESSION_SE_MULT * math.sqrt(0.25 / max(1, len(Xh)))
        if incumbent in scores and champion != incumbent:
            if scores[champion] - scores[incumbent] <= noise:
                champion = incumbent
                succession = "held (challenger within noise)"
            else:
                succession = "dethroned"
        elif champion == incumbent:
            succession = "held"
        # Growth bias: a bigger body that is not measurably worse than the
        # incumbent takes over, so capacity ratchets upward with evidence
        # that it did no harm, and comes down only when smaller is better.
        if champion == incumbent:
            grown = [n for n, g in genomes.items()
                     if g.get("grow") and n in scores
                     and scores[n] >= scores[incumbent] - noise]
            if grown:
                champion = max(grown, key=scores.get)
                succession = "grew (bigger body within noise)"
        self._pipes, self._platt = fitted[champion]
        self._pipeline = self._pipes["global"]
        self._parent_net = (self._pipeline.net
                            if isinstance(self._pipeline, NetPipeline) else None)
        self._fam = (fam_mu, fam_sd)
        genomes[champion] = {k: v for k, v in genomes[champion].items() if k != "grow"}
        if self._parent_net is not None:
            genomes[champion]["hidden"] = list(self._parent_net.hidden)
            self.experience = self._parent_net.experience
            self.growth_events = list(self._parent_net.growth)
        self._genome = genomes[champion]
        self._base_thr = genomes[champion]["thr"]
        self.champion = champion
        self.candidate_scores = scores
        self.n_train = len(fit_idx)
        self.n_params = count_parameters(self._pipeline)
        self.holdout_accuracy = scores[champion]
        self.last_trained = time.time()
        # The survivor's genome seeds the next generation.
        db.set_meta(META_GENOME, json.dumps(genomes[champion]))
        db.set_meta(META_GENERATION, str(generation + 1))
        # Hall of fame: remember the best genomes ever seen (deduped by label).
        hall = [h for h in hall if h["label"] != champion]
        hall.append({"label": champion, "score": scores[champion],
                     "genome": genomes[champion], "generation": generation + 1})
        hall.sort(key=lambda h: -h["score"])
        db.set_meta(META_HALL, json.dumps(hall[:HALL_SIZE]))
        result["succession"] = succession
        db.set_meta("ml_gene_report", json.dumps(
            {FEATURE_NAMES[i]: round(c, 4) for i, c in enumerate(report)}))
        # Lab notebook: every self-experiment is recorded, append-only.
        db.insert_evolution(time.time(), generation + 1, champion,
                            json.dumps(genomes[champion]), json.dumps(scores))
        self._save_state(db)
        result["growth"] = self.growth_events[-3:]
        result["experience"] = self.experience
        result["synth"] = [synth_str(e) for e in genomes[champion]["synth"]]
        result["top_genes"] = sorted(
            ((FEATURE_NAMES[i], round(c, 3)) for i, c in enumerate(report)),
            key=lambda kv: -abs(kv[1]))[:3]
        result.update({"trained": True, "champion": champion,
                       "holdout_accuracy": scores[champion],
                       "scores": scores, "n_params": self.n_params,
                       "generation": generation + 1,
                       "genome": genomes[champion]})
        return result

    def _rolling_accuracy(self, db: Database) -> float | None:
        rows = [r for r in db.resolved_predictions("ml")
                if r["status"] == "resolved" and r["return_pct"] != 0]
        rows = rows[-ROLLING_WINDOW:]
        if len(rows) < ROLLING_MIN_N:
            return None
        return sum(1 for r in rows if r["correct"]) / len(rows)

    def make_prediction(self, catalog: list[CatalogEntry],
                        ticks_by_mint: dict[str, list[tuple[float, float]]],
                        horizon_minutes: float, db: Database) -> Prediction:
        now = time.time()
        rolling = self._rolling_accuracy(db)
        threshold = adaptive_threshold(rolling, self.holdout_accuracy,
                                       base=self._base_thr)
        meta = (f"{self.champion} n={self.n_train} params={self.n_params}"
                + (f" holdout={self.holdout_accuracy:.2f}"
                   if self.holdout_accuracy is not None else "")
                + (f" roll30={rolling:.2f}" if rolling is not None else "")
                + (f" thr={threshold:.2f}" if threshold is not None
                   else " SELF-BENCHED"))
        if not self.trained:
            return self._skip(now, horizon_minutes,
                              f"untrained (needs {MIN_TRAIN_ROWS} rows)", meta)
        if threshold is None:
            return self._skip(
                now, horizon_minutes,
                f"self-benched: holdout {self.holdout_accuracy:.1%} < "
                f"{SELF_BENCH_HOLDOUT:.0%}, waiting for a better retrain", meta)

        hs, hc, sol5 = context_features(db, now)
        best = None
        lines = []
        stash: list[tuple[str, list[float]]] = []
        for e in catalog:
            ticks = ticks_by_mint.get(e.mint, [])
            micro = micro_features(ticks)
            # Movement gate: direction is only predictable on coins that move.
            if micro is None or (micro.micro_vol < MOVE_GATE_VOL
                                 and abs(micro.ret_300s) < MOVE_GATE_RET):
                lines.append(f"{e.symbol}: gated (not moving)")
                continue
            flow = db.flow_before(e.mint, now)
            turnover = e.volume_h24 / e.liquidity_usd if e.liquidity_usd > 0 else 0.0
            x = feature_vector(e.chg_m5, e.price_change_h1, e.price_change_h24,
                               e.volatility_pct, e.liquidity_usd, turnover,
                               micro,
                               flow[0] if flow else e.buys_m5 or None,
                               flow[1] if flow else e.sells_m5 or None,
                               hs, hc, sol5,
                               strategy_features(one_minute_closes(ticks)),
                               db.tsfm_before(e.mint, now) or (0.0, 0.0))
            xb = x
            stash.append((e.mint, list(xb)))
            gen = self._genome or {}
            if self._genome is not None:
                x = apply_genome(self._genome, x)
            fam = (familiarity(xb, *self._fam) if self._fam else 1.0)
            regime = regime_of(xb)
            pipe = (self._pipes or {}).get(regime) or self._pipeline
            p_up = self._calibrated(float(pipe.predict_proba([x])[0][1]))
            conf = max(p_up, 1.0 - p_up)
            # Epistemic-humility genes: too sure, or too unfamiliar -> no call.
            if conf > gen.get("max_conf", 0.95):
                lines.append(f"{e.symbol}: P(up)={p_up:.3f} humility-capped")
                continue
            if fam < gen.get("min_fam", 0.0):
                lines.append(f"{e.symbol}: P(up)={p_up:.3f} unfamiliar ({fam:.2f})")
                continue
            lines.append(f"{e.symbol}[{regime}]: P(up)={p_up:.3f} fam={fam:.2f}")
            if best is None or conf > best[1]:
                best = (e, conf, "UP" if p_up >= 0.5 else "DOWN")

        detail = "calibrated model probabilities:\n" + "\n".join(lines)
        self.remember_window(now, stash)
        if best is None or best[1] < threshold:
            why = ("all candidates gated or none scored" if best is None else
                   f"max P={best[1]:.3f} < thr={threshold:.2f}")
            return self._skip(now, horizon_minutes, why, meta, detail)

        entry, conf, direction = best
        return Prediction(
            ts=now, arm="ml", mint=entry.mint, symbol=entry.symbol,
            direction=direction, confidence=conf,
            horizon_end=now + horizon_minutes * 60.0,
            price_at=entry.price_usd, prompt=detail,
            response=f"{entry.symbol} {direction} P={conf:.3f}",
            model=meta, backend="sklearn",
        )

    def _skip(self, now: float, horizon_minutes: float, why: str,
              meta: str, detail: str = "") -> Prediction:
        return Prediction(
            ts=now, arm="ml", mint="", symbol="SKIP", direction="SKIP",
            confidence=0.0, horizon_end=now + horizon_minutes * 60.0,
            price_at=0.0, prompt=detail or "n/a",
            response=f"SKIP [{why}]", model=meta, backend="sklearn",
        )

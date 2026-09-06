"""Second test bench: the EvoML loop on public credit-card fraud data.

Same ingredients as the live crypto run, applied to a fintech problem:
  * a genome (learner kind + hyper-parameters + feature chromosome),
  * a tournament per generation (champion, two mutants, one immigrant),
  * significance-gated succession (win by more than one bootstrap SE),
  * a random control and a standard baseline scored on the same held-out
    window,
  * gates written down here, before the run.

Data: the ULB/Worldline credit-card fraud set (284,807 transactions, 492
frauds, 0.17 %), fetched from OpenML (id 1597), ordered by time. Splits are
time-ordered with purge gaps so the test window is strictly in the future.

Pre-registered gates (declared before running):
  G1  EvoML average precision (PR-AUC) on the test window beats the random
      control with a bootstrap 95 % interval that excludes the control.
  G2  Recall at a fixed alert budget of 0.5 % of test transactions >= 0.70.
  G3  Non-inferiority to a standard balanced logistic regression:
      EvoML AP >= baseline AP - 0.02. Whether it also beats the baseline is
      reported either way.

Run:  python bench/fraud_bench.py [--generations 8] [--out bench/results]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memescalp.evonet import EvoNet, NetPipeline  # noqa: E402

ALERT_BUDGET = 0.005          # fixed a priori: flag the top 0.5 % of transactions
PURGE_FRAC = 0.01             # gap between splits so labels cannot leak across
MIN_FEATURES = 12
MAX_SYNTH = 6                 # invented genes per genome
HALL_SIZE = 5
SUCCESSION_SE_MULT = 1.0
BOOT = 200
UNARY = ("neg", "abs", "sq", "sign")
BINARY = ("mul", "sub", "div", "max", "min")


def eval_synth(expr: list, X: np.ndarray) -> np.ndarray:
    """Vectorised evaluation of one invented gene over base columns."""
    op, *args = expr
    a = X[:, args[0]]
    if op == "neg":
        return -a
    if op == "abs":
        return np.abs(a)
    if op == "sq":
        return a * a
    if op == "sign":
        return np.sign(a)
    b = X[:, args[1]]
    if op == "mul":
        return a * b
    if op == "sub":
        return a - b
    if op == "div":
        return np.clip(a / np.where(np.abs(b) < 1e-3, 1e-3, b), -1e3, 1e3)
    if op == "max":
        return np.maximum(a, b)
    return np.minimum(a, b)


def synth_str(expr: list, cols: list[str]) -> str:
    op, *args = expr
    return f"{op}({', '.join(cols[i] for i in args)})"


def design(g: dict, X: np.ndarray) -> np.ndarray:
    """Feature chromosome + invented genes -> the matrix a genome actually sees."""
    parts = [X[:, g["features"]]]
    for expr in g.get("synth", []):
        parts.append(eval_synth(expr, X)[:, None].astype(np.float32))
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def invent_gene(rng: random.Random, n_feat: int) -> list:
    if rng.random() < 0.35:
        return [rng.choice(UNARY), rng.randrange(n_feat)]
    i, j = rng.sample(range(n_feat), 2)
    return [rng.choice(BINARY), i, j]


# --- data -----------------------------------------------------------------------

def load() -> tuple[np.ndarray, np.ndarray, list[str]]:
    ds = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    df = ds.frame.copy()
    df["Class"] = df["Class"].astype(int)
    # The OpenML copy drops the Time column but keeps the original row order,
    # which is chronological; sort by Time when it is present.
    if "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)
    df["Amount"] = np.log1p(df["Amount"].astype(float))
    cols = [c for c in df.columns if c not in ("Class", "Time")]
    return df[cols].to_numpy(np.float32), df["Class"].to_numpy(np.int8), cols


def splits(n: int) -> tuple[slice, slice, slice]:
    gap = int(n * PURGE_FRAC)
    a = int(n * 0.60)
    b = int(n * 0.80)
    return slice(0, a), slice(a + gap, b), slice(b + gap, n)


# --- genome ---------------------------------------------------------------------

def random_genome(rng: random.Random, n_feat: int, kind: str | None = None) -> dict:
    kind = kind or rng.choice(["net", "net", "hgb", "logreg"])
    g = {"kind": kind, "features": list(range(n_feat)), "synth": []}
    if kind == "net":
        g.update(hidden=[rng.choice([16, 24, 32]), rng.choice([8, 12])],
                 act=rng.choice(["tanh", "relu"]), lr=rng.choice([1e-3, 3e-3, 1e-2]),
                 l2=rng.choice([1e-5, 1e-4, 1e-3]), epochs=rng.choice([8, 12]),
                 prune=rng.choice([0.0, 0.1, 0.2]))
    elif kind == "hgb":
        g.update(depth=rng.choice([3, 4, 6]), lr=rng.choice([0.05, 0.1]), iters=rng.choice([100, 200]))
    else:
        g.update(C=rng.choice([0.1, 0.5, 1.0]))
    return g


def mutate(g: dict, rng: random.Random, n_feat: int) -> dict:
    g = json.loads(json.dumps(g))
    g.setdefault("synth", [])
    op = rng.choice(["hyper", "hyper", "feat", "feat", "invent", "invent", "kill"])
    if op == "invent":
        if len(g["synth"]) < MAX_SYNTH:
            g["synth"].append(invent_gene(rng, n_feat))
        return g
    if op == "kill":
        if g["synth"]:
            g["synth"].pop(rng.randrange(len(g["synth"])))
        return g
    if op == "feat":
        feats = set(g["features"])
        if rng.random() < 0.5 and len(feats) > MIN_FEATURES:
            feats.discard(rng.choice(sorted(feats)))
        else:
            feats.add(rng.randrange(n_feat))
        g["features"] = sorted(feats)
        return g
    k = g["kind"]
    if k == "net":
        which = rng.choice(["hidden", "lr", "l2", "act", "prune", "epochs"])
        if which == "hidden":
            h = g["hidden"]
            i = rng.randrange(len(h))
            h[i] = int(min(64, max(4, h[i] * rng.choice([0.5, 1.5]))))
        elif which == "lr":
            g["lr"] = float(min(3e-2, max(1e-4, g["lr"] * rng.choice([0.5, 2.0]))))
        elif which == "l2":
            g["l2"] = float(min(1e-2, max(1e-6, g["l2"] * rng.choice([1 / 3, 3.0]))))
        elif which == "act":
            g["act"] = "relu" if g["act"] == "tanh" else "tanh"
        elif which == "prune":
            g["prune"] = float(min(0.5, max(0.0, g["prune"] + rng.choice([-0.1, 0.1]))))
        else:
            g["epochs"] = int(min(24, max(4, g["epochs"] + rng.choice([-4, 4]))))
    elif k == "hgb":
        which = rng.choice(["depth", "lr", "iters"])
        if which == "depth":
            g["depth"] = int(min(8, max(2, g["depth"] + rng.choice([-1, 1]))))
        elif which == "lr":
            g["lr"] = float(min(0.3, max(0.01, g["lr"] * rng.choice([0.5, 2.0]))))
        else:
            g["iters"] = int(min(400, max(50, g["iters"] + rng.choice([-50, 100]))))
    else:
        g["C"] = float(min(10.0, max(0.01, g["C"] * rng.choice([0.3, 3.0]))))
    return g


def label(g: dict) -> str:
    k = g["kind"]
    tail = f"[{len(g['features'])}f+{len(g.get('synth', []))}s]"
    if k == "net":
        return f"net{'x'.join(map(str, g['hidden']))}({g['act']},lr={g['lr']:g},p={g['prune']:.1f}){tail}"
    if k == "hgb":
        return f"hgb(d={g['depth']},lr={g['lr']:g},i={g['iters']}){tail}"
    return f"logreg(C={g['C']:g}){tail}"


def build(g: dict, parent_net: EvoNet | None):
    if g["kind"] == "net":
        net = EvoNet(len(g["features"]) + len(g.get("synth", [])), g["hidden"], act=g["act"], lr=g["lr"], l2=g["l2"],
                     epochs=g["epochs"], prune=g["prune"], seed=7)
        if parent_net is not None:
            net.inherit(parent_net)
        return NetPipeline(net)
    if g["kind"] == "hgb":
        return HistGradientBoostingClassifier(max_depth=g["depth"], learning_rate=g["lr"],
                                              max_iter=g["iters"], random_state=7)
    return make_pipeline(StandardScaler(), LogisticRegression(C=g["C"], max_iter=2000))


def balanced_weights(y: np.ndarray) -> np.ndarray:
    pos = max(1, int(y.sum()))
    neg = len(y) - pos
    w = np.where(y == 1, neg / pos, 1.0).astype(np.float32)
    return w / w.mean()


def fit_score(g: dict, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, parent_net: EvoNet | None):
    model = build(g, parent_net)
    w = balanced_weights(y)
    Xd, Xvd = design(g, X), design(g, Xv)
    if g["kind"] == "logreg":
        model.fit(Xd, y, logisticregression__sample_weight=w)
    else:
        model.fit(Xd, y, sample_weight=w)
    return model, model.predict_proba(Xvd)[:, 1]


# --- metrics ----------------------------------------------------------------------

def ap_and_se(y: np.ndarray, s: np.ndarray, rng: np.random.Generator, boot: int = 60) -> tuple[float, float]:
    ap = float(average_precision_score(y, s))
    n = len(y)
    aps = []
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        aps.append(average_precision_score(y[idx], s[idx]))
    return ap, float(np.std(aps)) if aps else 0.0


def at_budget(y: np.ndarray, s: np.ndarray, budget: float) -> dict:
    k = max(1, int(len(y) * budget))
    top = np.argsort(-s)[:k]
    tp = int(y[top].sum())
    return {"alerts": k, "true_positives": tp, "precision": tp / k, "recall": tp / max(1, int(y.sum()))}


def boot_ci_diff(y, s_a, s_b, rng, boot=BOOT) -> tuple[float, float]:
    n = len(y)
    diffs = []
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], s_a[idx]) - average_precision_score(y[idx], s_b[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --- run ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    t0 = time.time()

    X, y, cols = load()
    tr, va, te = splits(len(y))
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    print(f"rows train={len(ytr)} val={len(yva)} test={len(yte)} | frauds train={int(ytr.sum())} val={int(yva.sum())} test={int(yte.sum())}", flush=True)

    # Controls, fitted once, scored on validation and test.
    rand_va, rand_te = nrng.random(len(yva)), nrng.random(len(yte))
    base = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    base.fit(Xtr, ytr)
    base_va, base_te = base.predict_proba(Xva)[:, 1], base.predict_proba(Xte)[:, 1]

    journal = []
    champion = random_genome(rng, X.shape[1], kind="net")
    champ_model, champ_va = fit_score(champion, Xtr, ytr, Xva, None)
    champ_ap, champ_se = ap_and_se(yva, champ_va, nrng)
    champ_net = champ_model.net if isinstance(champ_model, NetPipeline) else None
    print(f"gen 0 champion {label(champion)} val AP {champ_ap:.3f} ± {champ_se:.3f}", flush=True)
    journal.append({"generation": 0, "champion": label(champion), "val_ap": champ_ap, "population": {label(champion): champ_ap}})
    hall: list[tuple[float, dict]] = [(champ_ap, champion)]

    def remember(a: float, g: dict) -> None:
        if any(json.dumps(g, sort_keys=True) == json.dumps(h, sort_keys=True) for _, h in hall):
            return
        hall.append((a, g))
        hall.sort(key=lambda t: -t[0])
        del hall[HALL_SIZE:]

    for gen in range(1, args.generations + 1):
        immigrant = (mutate(rng.choice(hall)[1], rng, X.shape[1]) if hall and rng.random() < 0.5
                     else random_genome(rng, X.shape[1]))
        pop = [mutate(champion, rng, X.shape[1]), mutate(champion, rng, X.shape[1]), immigrant]
        scores = {label(champion): champ_ap}
        best = (champ_ap, champion, champ_model, champ_va, champ_se, champ_net, False)
        for g in pop:
            try:
                parent = champ_net if (g["kind"] == "net" and champ_net is not None
                                       and g["features"] == champion["features"]
                                       and g.get("synth", []) == champion.get("synth", [])) else None
                model, s_va = fit_score(g, Xtr, ytr, Xva, parent)
            except Exception as exc:  # noqa: BLE001 - a broken genome just loses
                print(f"  {label(g)} failed: {exc}", flush=True)
                continue
            a, se = ap_and_se(yva, s_va, nrng)
            scores[label(g)] = a
            remember(a, g)
            if a > best[0] + SUCCESSION_SE_MULT * max(best[4], se):
                best = (a, g, model, s_va, se, model.net if isinstance(model, NetPipeline) else None, True)
        dethroned = best[6]
        champ_ap, champion, champ_model, champ_va, champ_se, champ_net, _ = best
        journal.append({"generation": gen, "champion": label(champion), "val_ap": champ_ap,
                        "succession": "dethroned" if dethroned else "held", "population": scores})
        print(f"gen {gen} champion {label(champion)} val AP {champ_ap:.3f} ± {champ_se:.3f} "
              f"({'dethroned' if dethroned else 'held'}) | " +
              ", ".join(f"{k}={v:.3f}" for k, v in scores.items()), flush=True)

    # Held-out test window, scored once, after evolution is frozen.
    evo_te = champ_model.predict_proba(design(champion, Xte))[:, 1]
    res = {}
    for name, s in (("evoml", evo_te), ("logreg_balanced", base_te), ("random", rand_te)):
        res[name] = {"ap": float(average_precision_score(yte, s)), **at_budget(yte, s, ALERT_BUDGET)}
    lo_r, hi_r = boot_ci_diff(yte, evo_te, rand_te, nrng)
    lo_b, hi_b = boot_ci_diff(yte, evo_te, base_te, nrng)
    gates = {
        "G1_beats_random": lo_r > 0,
        "G2_recall_at_budget": res["evoml"]["recall"] >= 0.70,
        "G3_noninferior_to_logreg": res["evoml"]["ap"] >= res["logreg_balanced"]["ap"] - 0.02,
        "beats_logreg_ci_excludes_zero": lo_b > 0,
    }
    out = {
        "dataset": "openml creditcard (id 1597), time-ordered, purged splits",
        "alert_budget": ALERT_BUDGET, "test_rows": int(len(yte)), "test_frauds": int(yte.sum()),
        "champion": label(champion), "champion_genome": champion,
        "invented_genes": [synth_str(e, cols) for e in champion.get("synth", [])],
        "hall_of_fame": [{"val_ap": a, "genome": label(g)} for a, g in hall],
        "results": res,
        "ap_diff_ci95": {"vs_random": [lo_r, hi_r], "vs_logreg": [lo_b, hi_b]},
        "gates": gates, "all_gates_pass": all(v for k, v in gates.items() if k.startswith("G")),
        "journal": journal, "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "fraud_bench.json").write_text(json.dumps(out, indent=2))
    md = ["| Scorer | PR-AUC | Precision @0.5% | Recall @0.5% |", "|---|---:|---:|---:|"]
    for name in ("evoml", "logreg_balanced", "random"):
        r = res[name]
        md.append(f"| {name} | {r['ap']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |")
    md.append("")
    md.append(f"Champion: `{label(champion)}` · test rows {len(yte)} · test frauds {int(yte.sum())}")
    if champion.get("synth"):
        md.append("Invented genes: " + ", ".join(f"`{synth_str(e, cols)}`" for e in champion["synth"]))
    md.append(f"AP difference 95% CI: vs random [{lo_r:.3f}, {hi_r:.3f}] · vs logreg [{lo_b:.3f}, {hi_b:.3f}]")
    md.append("Gates: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()))
    Path(args.out, "fraud_bench.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()

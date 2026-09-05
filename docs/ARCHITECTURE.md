# Architecture

One Python process (`run.py`) runs four asyncio loops and a FastAPI app.

| Loop | Module | Cadence | Notes |
|---|---|---|---|
| feed | `feed.py`, `jupiter.py` | 5 s | chunked price polls, cached to SQLite |
| catalog | `catalog.py` | 60 s | DexScreener liquidity, volume, buys/sells, discovery feeds |
| predict | `run.py::predict_loop` | 5 min | catalog → Chronos features (thread) → random → math → EvoML → hedge |
| resolve | `run.py::resolve_loop` | continuous | scores calls at horizon, voids after a grace period |
| retrain | `run.py::ml_retrain_loop` | 30 min | genome tournament, hall of fame, evolution journal |

## Data model (SQLite, WAL, append-only)

`prices`, `liquidity`, `flow`, `catalog`, `tsfm`, `predictions`,
`resolutions`, `evolution`, `meta`, plus legacy trade tables. Rows are never
updated or deleted; restarts resume from the log.

## Feature vector (19)

`chg_m5, chg_h1, chg_h24, vol_1m, log_liq, turnover, ret_60s, ret_300s,
uptick, buy_ratio, hour_sin, hour_cos, sol_ret5, z20, vr2c, ema_gap,
strat_sig, tsfm_ret, tsfm_spread` plus up to 8 invented genes appended by the
genome.

## EvoML pipeline

genome → feature chromosome + invented genes → learner (`net` from
`evonet.py`, or sklearn `mlp` / `hgb` / `logreg`) → Platt calibration →
adaptive threshold (rolling live accuracy vs holdout) → familiarity and
regime gates → humility cap → UP / DOWN / SKIP.

## From-scratch network (`evonet.py`)

- float32 weights, He/Glorot init, tanh or relu hidden layers, sigmoid output.
- Loss: weighted binary cross-entropy + L2. Gradients derived by hand:
  `δ_out = (p − y)·w/Σw`, `∂W_i = a_iᵀ δ + λ W_i`, `δ_{i−1} = (δ W_iᵀ ⊙ mask) ⊙ act′`.
- Adam with bias correction; mini-batches of 256 chosen so a batch plus the
  weights fit in L2 cache; matrix products go through BLAS.
- Magnitude pruning after training silences the smallest weights permanently.
- `inherit(parent)` copies overlapping weight blocks across grown or shrunk
  layers and carries the parent's standardisation statistics.
- `tests/test_evonet.py` checks every gradient against central finite
  differences.

## Public site

`tools/push_snapshot.py` POSTs a compact JSON snapshot (summary, lineage,
genome, hall of fame, accuracy curves, latest calls) every 20 s to the site's
`/api/ingest` route; the site stores it in KV and renders it as the lineage
constellation, genome ring, accuracy race and live ledger.

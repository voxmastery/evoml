# EvoML — a self-evolving prediction model with an honest scoreboard

EvoML is a self-evolving machine-learning system that predicts short-horizon
price direction (Binance-style UP / DOWN calls at a fixed horizon) on live
Solana token prices, and — more importantly — **measures itself honestly
against a random control, in public, with an append-only audit trail.**

It runs on a laptop CPU, needs **no AI API key**, and writes its own neural
network from first principles in NumPy.

**This repository is measurement-only.** There is no wallet integration, no
private-key handling, and no code path that can sign or submit a transaction.
Requests to add live trading are declined by design.

## Results (live run, 14.5 days, as of 2026-09-06)

| Arm | Resolved | Accuracy | Note |
|---|---|---|---|
| **EvoML (subject)** | 562 | **56.6 %** | skill z = 3.12 vs 50 % |
| Random control | ~1 400 | 50.6 % | same coins, same windows |
| Classic math (OU z-score + variance ratio) | ~1 200 | 49.1 % | |
| Chronos-Bolt (Amazon TSFM, standalone) | ~930 | 51.5 % | coin flip on its own |
| Regime hedge (6 experts incl. Chronos) | ~1 100 | 52.4 % | |

Pre-registered gates, all passed: skill z ≥ 1.64 · ≥ 200 resolved · ≥ 14 days ·
beats random with two-proportion z ≥ 1.64 (actual **2.38**).

Honest caveats: 56.6 % direction accuracy is a *statistical* edge, not a money
machine. Fee-inclusive paper capital is flat-to-negative; Kelly-sized capital
is roughly flat. The claim we make is "measurably better than random, with an
audit trail", nothing more.

## What is new here

- **Self-modifying genome.** A model is a genome: kind (`logreg` / `mlp` /
  `hgb` / `net`), hyper-parameters, a *feature chromosome* (which of 19 base
  features it reads), *invented genes* (genetic-programming expressions over
  base features such as `max(ema_gap, log_liq)`), its own confidence threshold,
  its own mutation rate, a *humility cap* on confidence, a *familiarity floor*
  (refuse inputs far outside training distribution), and a polymorphic flag.
- **Tournament every 30 min.** Champion + 2 mutants + 1 immigrant (50 % from a
  hall of fame). Fitness is policy-aware (accuracy on the rows it would
  actually call). **Succession is significance-gated**: a challenger must beat
  the incumbent by more than one standard error, so noise cannot dethrone
  skill. Every generation is written to an evolution journal (`/api/evolution`).
- **From-scratch network (`memescalp/evonet.py`).** Forward pass, hand-derived
  back-propagation, Adam, weighted BCE + L2, magnitude pruning, float32 for
  AVX2. Verified by finite-difference gradient checks in `tests/test_evonet.py`.
  **Lamarckian inheritance**: a child copies the overlapping weight blocks of
  its parent, so learned weights persist and grow across generations instead
  of restarting from random.
- **Foundation-model ingestion.** Amazon `chronos-bolt-small` (47.7 M params,
  open weights) runs on CPU as a feature (`tsfm_ret`, `tsfm_spread`) and as an
  expert inside the hedge arm. Its standalone score is published, not hidden.
- **Honest measurement.** Predictions are logged before outcomes exist,
  resolved against real prices, voided on flat/stale quotes, winsorised
  against glitches, and evaluated with pre-registered pass/fail gates.

## Architecture

```
Jupiter Price API (5 s)  ─┐
DexScreener (liq/vol/flow)┼─► catalog (30 tokens) ─► feature vector (19 + invented genes)
Chronos-Bolt (CPU)       ─┘                                    │
                                                               ▼
   random ── math ── hedge ── EvoML (genome → pipeline → Platt calibration)
                                                               │
   predictions (append-only SQLite) ◄──── resolver (real price @ +15 min)
                                                               │
   /api/predict/summary · /api/evolution · dashboard (FastAPI + vanilla JS)
```

Key modules: `memescalp/mlpred.py` (genome, mutation, tournament, hall of fame),
`memescalp/evonet.py` (from-scratch network), `memescalp/hedgepred.py`
(regime-conditioned multiplicative-weights hedge), `memescalp/tsfm.py`
(Chronos), `memescalp/metrics.py` (z-tests, Brier, Kelly), `run.py` (loops).

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # MODE=predict, LLM_ARM=off, add JUPITER_API_KEY
python run.py                # dashboard at http://127.0.0.1:8765
pytest                       # 103 tests incl. gradient checks
```

No Anthropic / OpenAI key is required. The optional Claude arm (`LLM_ARM=on`)
used the Claude Code CLI on a subscription and is off by default.

## Why this matters for fintech

Risk, fraud and pricing models decay. The hard part is not training a model
once; it is knowing, continuously and honestly, whether the model in
production still beats a trivial baseline. EvoML is that loop: a model that
keeps rewriting itself, and a harness that refuses to believe it until the
numbers clear pre-registered gates.

## License

MIT.

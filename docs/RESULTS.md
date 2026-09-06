# Results and methodology

Snapshot: 2026-09-06 01:20 IST, 14.6 days into the live run. The dashboard
(`/api/predict/summary`) and the live site show current values.

## Setup

- Universe: 30 Solana tokens (watchlist + DexScreener discovery, liquidity
  floor $20 k for discovered tokens), refreshed continuously.
- Prices: Jupiter Price API v3, polled every 5 s. Flow and liquidity from
  DexScreener.
- Prediction window: every 5 minutes each arm may call UP, DOWN, or SKIP for
  each token, for a horizon of 15 minutes.
- Resolution: the real Jupiter price at the horizon end. A call is scored
  correct if the sign of the return matches the direction. Flat or stale
  quotes are **voided** (not scored). Returns are winsorised at ±50 % to
  neutralise data glitches; a movement gate ignores dead markets.
- Labels for training require a real move (≥ 0.15 %), so the model does not
  learn noise.

## Arms

| Arm | What it is |
|---|---|
| EvoML | the subject: self-evolving genome, tournament every 30 min |
| random | uniform coin flip on the same tokens and windows (the control) |
| math | OU z-score (1930), Lo-MacKinlay variance ratio (1988), EMA gap |
| hedge | regime-conditioned multiplicative-weights hedge over 6 experts |
| tsfm | Amazon Chronos-Bolt forecast used directly (standalone score) |
| llm | Claude Haiku via CLI (stopped; rows retained) |

## Pre-registered gates

Fixed in `.env` / `memescalp/metrics.py` before the run:

1. Skill: one-sample z against 50 % ≥ 1.64.
2. Sample: ≥ 200 resolved calls.
3. Duration: ≥ 14 calendar days.
4. Beats control: two-proportion z against the random arm ≥ 1.64.

## Numbers

| Arm | Resolved | Accuracy | z vs 50 % |
|---|---:|---:|---:|
| EvoML | 572 | 56.3 % | 3.01 |
| random | 1 387 | 50.5 % | 0.4 |
| math | 1 622 | 49.3 % | −0.6 |
| hedge | 1 635 | 52.6 % | 2.1 |
| tsfm (standalone, earlier window) | ~930 | 51.5 % | 0.9 |

EvoML vs random: two-proportion z = **2.32** (p ≈ 0.01, one-sided).

## What the numbers do not say

- They do not say EvoML makes money. Frictionless paper capital compounding
  each call is roughly flat once fees and slippage are considered.
- They do not say the edge is stable across regimes. The run spans one
  fortnight; the hall of fame and journal exist precisely so drift is visible.
- Calibration matters more than accuracy for sizing: Brier ≈ 0.26 for EvoML
  vs 0.25 for random, i.e. the model is only mildly informative per call.

## Reproduce

```bash
sqlite3 data/memescalp.db "SELECT p.arm, COUNT(*), AVG(r.correct)
  FROM predictions p JOIN resolutions r ON r.prediction_id=p.id
  WHERE r.status='scored' GROUP BY p.arm;"
```

`memescalp/metrics.py::evaluate_predictions` implements the gates; the tests
in `tests/test_metrics.py` and `tests/test_predict.py` pin the definitions.

## Second bench: credit-card fraud

`bench/fraud_bench.py`, OpenML `creditcard` (id 1597), rows in chronological
order, 60 / 20 / 20 split with 1 % purge gaps, balanced sample weights,
8 generations, fitness = validation PR-AUC with bootstrap SE, succession
gated at one SE, test window scored once after evolution is frozen.

| Scorer | PR-AUC | Precision @0.5% | Recall @0.5% |
|---|---:|---:|---:|
| evoml | 0.776 | 0.222 | 0.845 |
| logreg_balanced | 0.770 | 0.222 | 0.845 |
| random | 0.002 | 0.004 | 0.014 |

Champion: `net24x12(relu,lr=0.01,p=0.0)[29f]` · test rows 54114 · test frauds 71
AP difference 95% CI: vs random [0.698, 0.858] · vs logreg [-0.032, 0.052]
Gates: G1_beats_random=PASS, G2_recall_at_budget=PASS, G3_noninferior_to_logreg=PASS, beats_logreg_ci_excludes_zero=FAIL

### Longer run: 30 generations, invented genes, hall of fame

| Scorer | PR-AUC | Precision @0.5% | Recall @0.5% |
|---|---:|---:|---:|
| evoml | 0.764 | 0.219 | 0.831 |
| logreg_balanced | 0.770 | 0.222 | 0.845 |
| random | 0.002 | 0.004 | 0.014 |

Champion: `net24x12(relu,lr=0.01,p=0.0)[29f+1s]` · test rows 54114 · test frauds 71
Invented genes: `div(V4, V15)`
AP difference 95% CI: vs random [0.664, 0.843] · vs logreg [-0.071, 0.056]
Gates: G1_beats_random=PASS, G2_recall_at_budget=PASS, G3_noninferior_to_logreg=PASS, beats_logreg_ci_excludes_zero=FAIL

Reading: the invented gene raised validation PR-AUC (0.747 → 0.774) but the
test score (0.764) sits inside the noise of the shorter run (0.776). With 71
frauds in the test window the PR-AUC has a standard error near 0.06, so
neither run can be called better than the other, or than the logistic
baseline. What the run demonstrates is the mechanism working end to end on
fintech data: invention, gated succession, a hall of fame, and an honest
held-out score.

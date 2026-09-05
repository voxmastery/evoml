# How evolution works

## Genome

```json
{
  "kind": "net", "hidden": [16, 8], "act": "tanh", "lr": 0.003, "l2": 1e-4,
  "epochs": 40, "prune": 0.2,
  "features": [0, 1, 2, 4, 5, 6, 7, 9, 13, 14, 15, 17, 18],
  "synth": [["max", 15, 4], ["mul", 9, 6]],
  "thr": 0.58, "mut_rate": 1.4, "max_conf": 0.80, "min_fam": 0.35, "poly": true
}
```

| Gene | Meaning | Bounds |
|---|---|---|
| kind + hyper-parameters | learner family and its settings | per kind |
| features | which base features are read (chromosome) | ≥ 6 of 19 |
| synth | invented genes: expressions over base features | ≤ 8 |
| thr | confidence needed to call | 0.54 – 0.70 |
| mut_rate | how aggressively the genome mutates itself | 0.5 – 3.0 |
| max_conf | humility cap on stated confidence | 0.62 – 0.95 |
| min_fam | familiarity floor: abstain on alien inputs | 0 – 0.9 |
| poly | polymorphic fitting (kind may switch mid-line) | bool |

## Mutation operators

`hyper` (kind-specific), `feat` (add/drop a feature, biased by gene report),
`thr`, `rate`, `invent` (genetic programming with ops `neg abs sq sign mul sub
div max min`), `kill` (drop an invented gene), `humility`, `poly`.

## Tournament (every 30 minutes)

1. Population: champion, two mutants of the champion, one immigrant
   (50 % hall of fame, 50 % random genome).
2. Each candidate is fitted on recency-weighted rows (τ = 6 h) with purged
   forward validation and Platt calibration. `net` children inherit the
   champion's weights before fitting.
3. Fitness is policy-aware: accuracy on holdout rows the candidate would
   actually call (if ≥ 30 such rows), else plain holdout accuracy.
4. Succession is significance-gated: a challenger replaces the champion only
   if it wins by more than one standard error of the holdout estimate.
5. The hall of fame keeps the five best genomes ever seen; the evolution
   journal records the generation, champion, genome and all scores.

## Self-benching and drift

If holdout accuracy falls below 52 % the model refuses to call. The live
threshold adapts: cold live accuracy raises it to 0.65, hot live and holdout
lower it to 0.55.

## What the lineage taught us

- Gen 363 invented `max(ema_gap, log_liq)` and held the crown to gen 370
  with holdouts up to 58.6 %.
- Gen 371 lost that lineage to a 0.5-point noise swing. That is why
  succession is now significance-gated and why the hall of fame exists.
- Calls above 0.70 confidence were only 48 % right early on; the humility cap
  and Platt calibration removed the inversion.

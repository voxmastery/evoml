# Razorpay AI Buildathon submission (Open Track)

| Field | Value |
|---|---|
| Project | EvoML: a self-evolving prediction model with an honest scoreboard |
| Repository | https://github.com/voxmastery/evoml |
| Live site | https://evoml-lab.higgsfield.app (live growth visualisation, pitch, audit trail) |
| Pitch video (5:00, cartoon, narrated, subtitled) | see README link |
| Track | Open Track |

## Objectives

Every trading or risk model claims an edge; almost none can prove it, and
the ones that work decay within weeks because markets are non-stationary.
EvoML attacks both problems: a model that rewrites its own genome every 30
minutes, and a harness that refuses to believe it until pre-registered gates
are cleared against a random control on identical data, with an append-only
audit trail. It runs on a laptop CPU with no AI API key, and the neural
network inside is written from first principles.

## Build challenges

1. Future prices leaked across overlapping horizons: purged walk-forward
   holdout so training labels cannot touch the test window.
2. Quiet coins are noise: direction is trained only on real moves and live
   calls are gated on movement.
3. A noisy tournament can crown a lucky genome: the challenger must beat the
   incumbent by more than holdout noise, with a hall of fame so a strong
   lineage can return.
4. Stale flat quotes looked like losses: voided, not scored.
5. High-confidence calls were over-confident (75 %+ stated, 46 % correct):
   Platt calibration, confidence caps, and a self-bench when holdout falls to
   chance.
6. Writing back-propagation by hand through pruning masks and proving it with
   finite-difference gradient checks; making weight inheritance survive grown
   or shrunk layers.
7. Running a 47.7 M-parameter foundation model inside a 5-second polling loop
   on CPU without stalling the event loop.

## Confirmation

Final submission for the Open Track. Public repository, 5-minute pitch
video, documented architecture and audit trail. Measurement-only: no real
funds, no wallet, no live trading.

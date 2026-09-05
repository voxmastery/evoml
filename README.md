# memescalp — Solana memecoin scalping, paper-trading experiment

A measurement harness for one hypothesis: *does having Claude pick which
memecoin to scalp every 30 minutes beat picking at random, after all real-world
costs?*

**This repository is simulation-only and measurement-only.**
There is no wallet integration, no private-key handling, and no code path that
can sign or submit a mainnet transaction. Requests to add live trading will be
declined — the point of this repo is to measure whether the strategy would
have worked, not to run it.

## How it works

- **Price feed** — polls the Jupiter Price API every 5 s for every watched
  token; pool liquidity, volume, and price-change data come from DexScreener
  (refreshed every 60 s). Every observation is cached to SQLite so a run is
  reproducible.
- **Coin picker (the hypothesis)** — every 30 minutes the top trending Solana
  tokens are fetched from DexScreener and handed to Claude, which picks ONE
  token for the next window. The full prompt and response are logged verbatim.
- **Control arm** — a second identical paper account picks uniformly at random
  from the *same* candidate list, with the same capital, costs, and loop.
- **Execution simulator** — each account starts at ₹2,000 (≈$23). Fills happen
  at the feed price with four separately-stored cost components per side:
  LP fee 0.25 %, slippage `size / (pool_liquidity + size)`, priority fee
  $0.04 flat, and India TDS 1 % (Section 194S).
- **Strategy loop** — enter with the full balance, exit at +$1.50 net profit
  (configurable), optional stop-loss including the original "no stop" mode
  (each trade is flagged with the active mode), re-enter after each exit,
  rotate when a new window picks a different token.
- **Logging** — every decision and fill goes to append-only SQLite plus a CSV
  mirror (`data/csv/`). Rows are never edited or deleted; restarting the
  process resumes balance and any open position from the log.

## Pass / fail

Displayed at the top of the dashboard. **PASS requires all of:**

1. Net positive PnL after ALL modeled costs,
2. over ≥ 200 completed round trips **and** ≥ 14 calendar days,
3. meaningfully above the random control (higher net PnL *and* a Welch
   z-score ≥ 1.64 on per-trade PnL).

No other definition of "working" counts.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py          # dashboard at http://127.0.0.1:8765
```

### LLM access

Two backends, chosen by `PICKER_BACKEND` in `.env`:

- `claude_cli` (default) — shells out to headless `claude -p`, so the picker
  runs on your existing **Claude subscription** through the official Claude
  Code client. No API key required; just be logged in (`claude` works in your
  terminal). The harness never touches the subscription's OAuth token itself —
  calling the Anthropic API directly with that token is not a supported use.
- `api` — uses the official `anthropic` SDK with `ANTHROPIC_API_KEY`.

## Tests

```bash
pytest
```

Covers the fee model, execution math, strategy state machine (target / stop /
no-stop / rotate / re-enter / resume), metrics + pass-fail evaluation,
append-only DB behavior, and picker response parsing.

## Notes and modeling caveats

- Long-only, spot-only: you can't short a memecoin in a spot wallet, so both
  arms only buy. The control arm therefore randomizes the *coin*; direction is
  always long.
- Slippage uses a constant-product approximation against the deepest pool's
  liquidity; real fills on thin pools can be worse (MEV, sandwich, latency).
  If anything the simulator is optimistic — a FAIL here is strong evidence,
  a PASS is grounds for more careful measurement, not for going live.
- TDS is modeled as 1 % of transaction value on every fill per Section 194S;
  30 % tax on gains under Section 115BBH is *not* modeled and would apply on
  top of any profit.

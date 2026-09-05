"""Regime-Conditioned Hedge (RCH): a custom online-learning forecaster.

Designed from first principles for this experiment (no off-the-shelf
equivalent), on two classical foundations:
- Multiplicative-weights prediction with expert advice (Littlestone &
  Warmuth's Hedge, 1994): a committee of simple experts votes; weights
  update after EVERY resolved outcome, not in periodic batches.
- Log-wealth updates (in the spirit of Cover's universal portfolios, 1991):
  an expert's weight compounds by the money its advice would have made, so
  influence tracks hypothetical wealth, which is exactly what Kelly rewards.

The custom part: weights are maintained SEPARATELY per market regime
(trending / reverting / random-walk, detected live by the variance ratio),
so the committee can trust momentum in trends and fade it in chop. Fully
local, fully interpretable, no API of any kind.
"""
from __future__ import annotations

import json
import logging
import time

from .catalog import CatalogEntry
from .db import Database
from .indicators import one_minute_closes
from .mathpred import signal_for, variance_ratio_2, zscore_20
from .micro import micro_features
from .predictor import Prediction

log = logging.getLogger(__name__)

ARM = "hedge"
WEALTH_FRACTION = 0.25    # f in w *= (1 + f * d * r): update aggressiveness
RETURN_CAP = 0.50         # cap |r| per outcome (glitch guard), as a fraction
MIN_WEIGHT = 1e-4         # floor so no expert is ever permanently dead
CALL_MARGIN = 0.20        # |committee score| needed to call (else SKIP)
MOVE_GATE_VOL = 0.03
MOVE_GATE_RET = 0.15

REGIMES = ("trend", "revert", "walk")
META_WEIGHTS = "hedge_weights"
META_LAST_ID = "hedge_last_processed_id"


def detect_regime(closes: list[float]) -> str:
    vr = variance_ratio_2(closes)
    if vr is None:
        return "walk"
    if vr >= 1.05:
        return "trend"
    if vr <= 0.95:
        return "revert"
    return "walk"


TSFM_VOTE_MIN = 0.05  # % forecast move needed for the tsfm expert to vote


def expert_votes(closes: list[float], micro, buys: int | None,
                 sells: int | None,
                 tsfm: tuple[float, float] | None = None) -> dict[str, int]:
    """Each expert votes +1 (UP), -1 (DOWN), or 0 (no opinion)."""
    votes: dict[str, int] = {}
    ret5 = micro.ret_300s if micro else 0.0
    votes["momentum"] = 1 if ret5 > 0 else (-1 if ret5 < 0 else 0)
    votes["meanrevert"] = -votes["momentum"]
    total = (buys or 0) + (sells or 0)
    if total >= 6:
        pressure = (buys / total) - 0.5
        votes["flow"] = 1 if pressure > 0.1 else (-1 if pressure < -0.1 else 0)
    else:
        votes["flow"] = 0
    sig = signal_for(closes)
    votes["classic"] = (0 if sig is None
                        else (1 if sig[0] == "UP" else -1))
    z = zscore_20(closes)
    votes["oufade"] = 0 if z is None or abs(z) < 1.5 else (-1 if z > 0 else 1)
    # Chronos-Bolt (open weights, 47.7M params) as one more fallible expert.
    if tsfm is not None and abs(tsfm[0]) >= TSFM_VOTE_MIN:
        votes["tsfm"] = 1 if tsfm[0] > 0 else -1
    else:
        votes["tsfm"] = 0
    return votes


class HedgeForecaster:
    """Continuously-learning committee. Weights live in the meta table and
    update from every resolved hedge prediction, one outcome at a time."""

    def __init__(self, db: Database):
        self._db = db
        raw = db.get_meta(META_WEIGHTS)
        experts = ("momentum", "meanrevert", "flow", "classic", "oufade",
                   "tsfm")
        if raw:
            self.weights = json.loads(raw)
            for reg in REGIMES:  # migrate: new experts join at neutral weight
                for e in experts:
                    self.weights.setdefault(reg, {}).setdefault(e, 1.0)
        else:
            self.weights = {reg: {e: 1.0 for e in experts} for reg in REGIMES}
        self._last_id = int(db.get_meta(META_LAST_ID) or 0)

    # --- online learning ------------------------------------------------------

    def absorb_outcomes(self) -> int:
        """Update weights from every newly resolved hedge prediction."""
        rows = self._db.hedge_unprocessed(self._last_id)
        updated = 0
        for r in rows:
            pid = r["prediction_id"]
            if r["status"] != "resolved":
                self._last_id = max(self._last_id, pid)
                continue
            try:
                detail = json.loads(r["prompt"] or "{}")
            except (json.JSONDecodeError, TypeError):
                detail = {}
            if not detail or "votes" not in detail:
                self._last_id = max(self._last_id, pid)
                continue
            regime = detail.get("regime", "walk")
            votes = detail.get("votes", {})
            ret = max(-RETURN_CAP, min(RETURN_CAP, r["return_pct"] / 100.0))
            if ret != 0.0 and regime in self.weights:
                w = self.weights[regime]
                for expert, vote in votes.items():
                    if expert in w and vote:
                        w[expert] = max(MIN_WEIGHT,
                                        w[expert]
                                        * (1.0 + WEALTH_FRACTION * vote * ret))
                total = sum(w.values())
                for expert in w:
                    w[expert] = w[expert] / total * len(w)
                updated += 1
            self._last_id = max(self._last_id, pid)
        if updated:
            self._db.set_meta(META_WEIGHTS, json.dumps(self.weights))
            self._db.set_meta(META_LAST_ID, str(self._last_id))
        return updated

    # --- prediction -------------------------------------------------------------

    def make_prediction(self, catalog: list[CatalogEntry],
                        ticks_by_mint: dict[str, list[tuple[float, float]]],
                        horizon_minutes: float,
                        tsfm_by_mint: dict[str, tuple[float, float]] | None = None,
                        ) -> Prediction:
        now = time.time()
        self.absorb_outcomes()
        best = None
        lines = []
        for e in catalog:
            ticks = ticks_by_mint.get(e.mint, [])
            micro = micro_features(ticks)
            if micro is None or (micro.micro_vol < MOVE_GATE_VOL
                                 and abs(micro.ret_300s) < MOVE_GATE_RET):
                continue
            closes = one_minute_closes(ticks)
            regime = detect_regime(closes)
            votes = expert_votes(closes, micro, e.buys_m5, e.sells_m5,
                                 (tsfm_by_mint or {}).get(e.mint))
            w = self.weights[regime]
            total_w = sum(w.values())
            score = sum(w[k] * v for k, v in votes.items()) / total_w
            lines.append(f"{e.symbol}[{regime}] score={score:+.3f} {votes}")
            if best is None or abs(score) > abs(best[1]):
                best = (e, score, regime, votes)

        if best is None or abs(best[1]) < CALL_MARGIN:
            why = ("all candidates gated" if best is None
                   else f"committee margin {abs(best[1]):.2f} < {CALL_MARGIN}")
            return Prediction(
                ts=now, arm=ARM, mint="", symbol="SKIP", direction="SKIP",
                confidence=0.0, horizon_end=now + horizon_minutes * 60.0,
                price_at=0.0, prompt=json.dumps({"skipped": why}),
                response=f"SKIP [{why}]", model="rch-mwu", backend="hedge")

        entry, score, regime, votes = best
        direction = "UP" if score > 0 else "DOWN"
        confidence = min(0.5 + abs(score) / 2.0, 0.85)
        detail = {"regime": regime, "votes": votes,
                  "weights": self.weights[regime], "score": score}
        return Prediction(
            ts=now, arm=ARM, mint=entry.mint, symbol=entry.symbol,
            direction=direction, confidence=confidence,
            horizon_end=now + horizon_minutes * 60.0,
            price_at=entry.price_usd, prompt=json.dumps(detail),
            response=f"{entry.symbol} {direction} score={score:+.2f} [{regime}]",
            model="rch-mwu", backend="hedge",
        )

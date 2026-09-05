"""CSV mirror of the SQLite log. Append-only, survives restarts."""
from __future__ import annotations

import csv
from pathlib import Path

from .models import Decision, Fill

_FILL_FIELDS = [
    "ts", "arm", "trade_id", "side", "mint", "symbol", "feed_price", "exec_price",
    "size_usd", "token_qty", "fee_lp", "fee_slippage", "fee_priority", "fee_tds",
    "realized_pnl", "balance_after", "stop_mode", "note",
]
_DECISION_FIELDS = [
    "ts", "arm", "window_start", "window_end", "mint", "symbol", "direction",
    "model", "backend", "prompt", "response",
]
_PREDICTION_FIELDS = [
    "id", "ts", "arm", "mint", "symbol", "direction", "confidence",
    "horizon_end", "price_at", "model", "backend", "prompt", "response",
]
_RESOLUTION_FIELDS = [
    "prediction_id", "ts", "price_end", "return_pct", "correct", "status",
]
_ANALYSIS_FIELDS = [
    "ts", "arm", "mint", "symbol", "action", "confidence", "model", "backend",
    "prompt", "response",
]


class CsvMirror:
    def __init__(self, directory: Path | str):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fills_path = self._dir / "fills.csv"
        self._decisions_path = self._dir / "decisions.csv"
        self._analyses_path = self._dir / "analyses.csv"
        self._predictions_path = self._dir / "predictions.csv"
        self._resolutions_path = self._dir / "resolutions.csv"

    def _append(self, path: Path, fields: list[str], row: dict) -> None:
        new_file = not path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def append_fill(self, f: Fill) -> None:
        self._append(self._fills_path, _FILL_FIELDS, {
            "ts": f.ts, "arm": f.arm, "trade_id": f.trade_id, "side": f.side,
            "mint": f.mint, "symbol": f.symbol, "feed_price": f.feed_price,
            "exec_price": f.exec_price, "size_usd": f.size_usd,
            "token_qty": f.token_qty, "fee_lp": f.fees.lp,
            "fee_slippage": f.fees.slippage, "fee_priority": f.fees.priority,
            "fee_tds": f.fees.tds, "realized_pnl": f.realized_pnl,
            "balance_after": f.balance_after, "stop_mode": f.stop_mode,
            "note": f.note,
        })

    def append_prediction(self, prediction_id: int, p) -> None:
        self._append(self._predictions_path, _PREDICTION_FIELDS, {
            "id": prediction_id, "ts": p.ts, "arm": p.arm, "mint": p.mint,
            "symbol": p.symbol, "direction": p.direction,
            "confidence": p.confidence, "horizon_end": p.horizon_end,
            "price_at": p.price_at, "model": p.model, "backend": p.backend,
            "prompt": p.prompt, "response": p.response,
        })

    def append_resolution(self, prediction_id: int, ts: float,
                          price_end: float, return_pct: float, correct: bool,
                          status: str) -> None:
        self._append(self._resolutions_path, _RESOLUTION_FIELDS, {
            "prediction_id": prediction_id, "ts": ts, "price_end": price_end,
            "return_pct": return_pct, "correct": int(correct),
            "status": status,
        })

    def append_analysis(self, a) -> None:
        self._append(self._analyses_path, _ANALYSIS_FIELDS, {
            "ts": a.ts, "arm": a.arm, "mint": a.mint, "symbol": a.symbol,
            "action": a.action, "confidence": a.confidence, "model": a.model,
            "backend": a.backend, "prompt": a.prompt, "response": a.response,
        })

    def append_decision(self, d: Decision) -> None:
        self._append(self._decisions_path, _DECISION_FIELDS, {
            "ts": d.ts, "arm": d.arm, "window_start": d.window_start,
            "window_end": d.window_end, "mint": d.mint, "symbol": d.symbol,
            "direction": d.direction, "model": d.model, "backend": d.backend,
            "prompt": d.prompt, "response": d.response,
        })

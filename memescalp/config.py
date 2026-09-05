"""Settings loaded from .env at startup. Fails fast on invalid values."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    mode: str              # "predict" (directional forecasts) | "trade" (paper fills)
    llm_arm: bool          # LLM_ARM=off stops the Claude arm entirely
    picker_backend: str
    claude_model: str
    anthropic_api_key: str

    predict_minutes: float   # cadence of prediction windows
    horizon_minutes: float   # how far ahead each prediction looks
    min_predictions: int     # pass/fail sample-size gate (predict mode)

    fill_model: str        # "quote" (Jupiter quote API) | "model" (formula)
    jupiter_api_key: str

    start_balance_inr: float
    inr_per_usd: float

    target_profit_usd: float
    stop_loss_pct: float | None  # None => "no stop" mode
    poll_seconds: float
    pick_minutes: float
    analysis_minutes: float  # cadence of chart-analysis (HOLD/EXIT) calls

    lp_fee_rate: float
    priority_fee_usd: float
    tds_rate: float

    watchlist: tuple[str, ...]
    db_path: Path
    csv_dir: Path

    dash_port: int
    min_round_trips: int
    min_days: int
    edge_z_min: float

    @property
    def start_balance_usd(self) -> float:
        return self.start_balance_inr / self.inr_per_usd

    @property
    def stop_mode(self) -> str:
        if self.stop_loss_pct is None:
            return "no_stop"
        return f"sl_{self.stop_loss_pct:g}pct"


def _optional_float(raw: str) -> float | None:
    raw = raw.strip()
    return float(raw) if raw else None


def load_settings(env_file: str | os.PathLike | None = None) -> Settings:
    load_dotenv(env_file or ".env")
    env = os.environ

    backend = env.get("PICKER_BACKEND", "claude_cli").strip()
    if backend not in ("claude_cli", "api"):
        raise ValueError(f"PICKER_BACKEND must be claude_cli or api, got {backend!r}")
    if backend == "api" and not env.get("ANTHROPIC_API_KEY", "").strip():
        raise ValueError("PICKER_BACKEND=api requires ANTHROPIC_API_KEY to be set")

    mode = env.get("MODE", "predict").strip()
    if mode not in ("predict", "trade"):
        raise ValueError(f"MODE must be predict or trade, got {mode!r}")

    fill_model = env.get("FILL_MODEL", "quote").strip()
    if fill_model not in ("quote", "model"):
        raise ValueError(f"FILL_MODEL must be quote or model, got {fill_model!r}")

    stop_pct = _optional_float(env.get("STOP_LOSS_PCT", ""))
    if stop_pct is not None and not 0 < stop_pct < 100:
        raise ValueError("STOP_LOSS_PCT must be between 0 and 100, or empty for no stop")

    watchlist = tuple(
        m.strip() for m in env.get("WATCHLIST", "").split(",") if m.strip()
    )

    settings = Settings(
        mode=mode,
        llm_arm=env.get("LLM_ARM", "on").strip().lower()
        not in ("off", "0", "false"),
        predict_minutes=float(env.get("PREDICT_MINUTES", "15")),
        horizon_minutes=float(env.get("HORIZON_MINUTES", "30")),
        min_predictions=int(env.get("MIN_PREDICTIONS", "200")),
        picker_backend=backend,
        claude_model=env.get("CLAUDE_MODEL", "claude-opus-5").strip(),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", "").strip(),
        fill_model=fill_model,
        jupiter_api_key=env.get("JUPITER_API_KEY", "").strip(),
        start_balance_inr=float(env.get("START_BALANCE_INR", "2000")),
        inr_per_usd=float(env.get("INR_PER_USD", "87.0")),
        target_profit_usd=float(env.get("TARGET_PROFIT_USD", "1.50")),
        stop_loss_pct=stop_pct,
        poll_seconds=float(env.get("POLL_SECONDS", "5")),
        pick_minutes=float(env.get("PICK_MINUTES", "30")),
        analysis_minutes=float(env.get("ANALYSIS_MINUTES", "10")),
        lp_fee_rate=float(env.get("LP_FEE_RATE", "0.0025")),
        priority_fee_usd=float(env.get("PRIORITY_FEE_USD", "0.04")),
        tds_rate=float(env.get("TDS_RATE", "0.01")),
        watchlist=watchlist,
        db_path=Path(env.get("DB_PATH", "data/memescalp.db")),
        csv_dir=Path(env.get("CSV_DIR", "data/csv")),
        dash_port=int(env.get("DASH_PORT", "8765")),
        min_round_trips=int(env.get("MIN_ROUND_TRIPS", "200")),
        min_days=int(env.get("MIN_DAYS", "14")),
        edge_z_min=float(env.get("EDGE_Z_MIN", "1.64")),
    )

    if settings.start_balance_inr <= 0 or settings.inr_per_usd <= 0:
        raise ValueError("START_BALANCE_INR and INR_PER_USD must be positive")
    if settings.target_profit_usd <= 0:
        raise ValueError("TARGET_PROFIT_USD must be positive")
    if settings.poll_seconds < 1 or settings.pick_minutes <= 0:
        raise ValueError("POLL_SECONDS must be >= 1 and PICK_MINUTES positive")
    if settings.analysis_minutes <= 0:
        raise ValueError("ANALYSIS_MINUTES must be positive")
    if settings.predict_minutes <= 0 or settings.horizon_minutes <= 0:
        raise ValueError("PREDICT_MINUTES and HORIZON_MINUTES must be positive")
    if settings.min_predictions <= 0:
        raise ValueError("MIN_PREDICTIONS must be positive")
    return settings

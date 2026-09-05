import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memescalp.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        mode="trade",
        llm_arm=True,
        predict_minutes=15.0,
        horizon_minutes=30.0,
        min_predictions=200,
        picker_backend="claude_cli",
        claude_model="claude-opus-5",
        anthropic_api_key="",
        fill_model="model",
        jupiter_api_key="",
        start_balance_inr=2000.0,
        inr_per_usd=87.0,
        target_profit_usd=1.50,
        stop_loss_pct=None,
        poll_seconds=5.0,
        pick_minutes=30.0,
        analysis_minutes=10.0,
        lp_fee_rate=0.0025,
        priority_fee_usd=0.04,
        tds_rate=0.01,
        watchlist=(),
        db_path=tmp_path / "test.db",
        csv_dir=tmp_path / "csv",
        dash_port=8765,
        min_round_trips=200,
        min_days=14,
        edge_z_min=1.64,
    )

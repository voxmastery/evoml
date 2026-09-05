"""Immutable value objects shared across modules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSnapshot:
    """Latest market data for one token, assembled from the feed cache."""

    mint: str
    symbol: str
    price_usd: float
    liquidity_usd: float
    volume_h24: float = 0.0
    price_change_h1: float = 0.0
    price_change_h24: float = 0.0
    decimals: int = 0  # token decimals (from Jupiter price v3); 0 = unknown
    # Order flow (DexScreener per-pool transaction data)
    buys_m5: int = 0
    sells_m5: int = 0
    vol_m5: float = 0.0
    chg_m5: float = 0.0


@dataclass(frozen=True)
class FeeBreakdown:
    """Every modeled cost component for one fill, in USD."""

    lp: float
    slippage: float
    priority: float
    tds: float

    @property
    def total(self) -> float:
        return self.lp + self.slippage + self.priority + self.tds


@dataclass(frozen=True)
class Fill:
    """One executed paper fill (one side of a round trip)."""

    ts: float
    arm: str            # "llm" | "random"
    trade_id: str
    side: str           # "buy" | "sell"
    mint: str
    symbol: str
    feed_price: float
    exec_price: float   # effective price after all costs
    size_usd: float     # gross transaction value
    token_qty: float
    fees: FeeBreakdown
    realized_pnl: float | None  # set on sells, None on buys
    balance_after: float
    stop_mode: str
    note: str = ""


@dataclass(frozen=True)
class Position:
    """An open long position for one arm."""

    trade_id: str
    mint: str
    symbol: str
    token_qty: float
    entry_cost_usd: float  # cash spent to open (gross)
    entry_ts: float
    stop_mode: str


@dataclass(frozen=True)
class Decision:
    """One picker decision (either arm), logged verbatim."""

    ts: float
    arm: str
    window_start: float
    window_end: float
    mint: str
    symbol: str
    direction: str
    prompt: str
    response: str
    model: str
    backend: str

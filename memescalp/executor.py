"""Execution simulator: paper fills only. No transaction is ever submitted."""
from __future__ import annotations

import time
import uuid

from .config import Settings
from .fees import compute_fees
from .models import FeeBreakdown, Fill, Position, TokenSnapshot


USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDC_DECIMALS = 6


class ExecutionError(Exception):
    pass


def simulate_buy(
    settings: Settings,
    arm: str,
    snapshot: TokenSnapshot,
    balance_usd: float,
    stop_mode: str,
    note: str = "",
    ts: float | None = None,
) -> tuple[Fill, Position]:
    """Spend the full available balance on the token at the current feed price."""
    if balance_usd <= settings.priority_fee_usd:
        raise ExecutionError(f"balance ${balance_usd:.2f} cannot cover the priority fee")
    if snapshot.price_usd <= 0:
        raise ExecutionError(f"no valid price for {snapshot.symbol}")

    ts = time.time() if ts is None else ts
    gross = balance_usd
    fees = compute_fees(gross, snapshot.liquidity_usd, settings.lp_fee_rate,
                        settings.priority_fee_usd, settings.tds_rate)
    net_invested = gross - fees.total
    if net_invested <= 0:
        raise ExecutionError(
            f"fees (${fees.total:.2f}) exceed trade size ${gross:.2f}"
        )
    token_qty = net_invested / snapshot.price_usd
    trade_id = uuid.uuid4().hex[:12]

    fill = Fill(
        ts=ts, arm=arm, trade_id=trade_id, side="buy",
        mint=snapshot.mint, symbol=snapshot.symbol,
        feed_price=snapshot.price_usd,
        exec_price=gross / token_qty,
        size_usd=gross, token_qty=token_qty, fees=fees,
        realized_pnl=None, balance_after=balance_usd - gross,
        stop_mode=stop_mode, note=note,
    )
    position = Position(
        trade_id=trade_id, mint=snapshot.mint, symbol=snapshot.symbol,
        token_qty=token_qty, entry_cost_usd=gross, entry_ts=ts,
        stop_mode=stop_mode,
    )
    return fill, position


def sell_proceeds(
    settings: Settings, position: Position, snapshot: TokenSnapshot
) -> tuple[float, FeeBreakdown]:
    """Net USD received if the position were closed at the current feed price."""
    gross = position.token_qty * snapshot.price_usd
    fees = compute_fees(gross, snapshot.liquidity_usd, settings.lp_fee_rate,
                        settings.priority_fee_usd, settings.tds_rate)
    return gross - fees.total, fees


def simulate_sell(
    settings: Settings,
    arm: str,
    position: Position,
    snapshot: TokenSnapshot,
    balance_usd: float,
    note: str = "",
    ts: float | None = None,
) -> Fill:
    if snapshot.mint != position.mint:
        raise ExecutionError("snapshot token does not match the open position")
    if snapshot.price_usd <= 0:
        raise ExecutionError(f"no valid price for {snapshot.symbol}")

    ts = time.time() if ts is None else ts
    net, fees = sell_proceeds(settings, position, snapshot)
    gross = position.token_qty * snapshot.price_usd

    return Fill(
        ts=ts, arm=arm, trade_id=position.trade_id, side="sell",
        mint=position.mint, symbol=position.symbol,
        feed_price=snapshot.price_usd,
        exec_price=net / position.token_qty if position.token_qty else 0.0,
        size_usd=gross, token_qty=position.token_qty, fees=fees,
        realized_pnl=net - position.entry_cost_usd,
        balance_after=balance_usd + net,
        stop_mode=position.stop_mode, note=note,
    )


# --- quote-based fills (Jupiter quote API; still simulation only) -------------
#
# The swap's combined LP-fee + price-impact cost is exact — it is whatever the
# real quote says. Splitting it into the separate lp/slippage columns uses the
# quote's priceImpactPct and is approximate; totals (and therefore PnL) are not.

def buy_swap_units(settings: Settings, balance_usd: float) -> int:
    """USDC base units to quote for a full-balance entry, after the modeled
    priority fee and TDS are set aside from cash."""
    swap_usd = balance_usd - settings.priority_fee_usd \
        - settings.tds_rate * balance_usd
    if swap_usd <= 0:
        raise ExecutionError(
            f"balance ${balance_usd:.2f} cannot cover priority fee + TDS"
        )
    return int(round(swap_usd * 10**USDC_DECIMALS))


def quote_buy(
    settings: Settings,
    arm: str,
    snapshot: TokenSnapshot,
    balance_usd: float,
    stop_mode: str,
    quote: dict,
    note: str = "",
    ts: float | None = None,
) -> tuple[Fill, Position]:
    if snapshot.price_usd <= 0:
        raise ExecutionError(f"no valid price for {snapshot.symbol}")
    if snapshot.decimals <= 0:
        raise ExecutionError(f"unknown token decimals for {snapshot.symbol}")

    ts = time.time() if ts is None else ts
    gross = balance_usd
    priority = settings.priority_fee_usd
    tds = settings.tds_rate * gross
    swap_usd = int(quote["inAmount"]) / 10**USDC_DECIMALS
    token_qty = int(quote["outAmount"]) / 10**snapshot.decimals
    if token_qty <= 0:
        raise ExecutionError("quote returned zero output tokens")

    impact = abs(float(quote.get("priceImpactPct") or 0.0))
    slippage = swap_usd * impact
    out_value_mid = token_qty * snapshot.price_usd
    lp = max(0.0, swap_usd - out_value_mid - slippage)
    fees = FeeBreakdown(lp=lp, slippage=slippage, priority=priority, tds=tds)

    trade_id = uuid.uuid4().hex[:12]
    fill = Fill(
        ts=ts, arm=arm, trade_id=trade_id, side="buy",
        mint=snapshot.mint, symbol=snapshot.symbol,
        feed_price=snapshot.price_usd,
        exec_price=gross / token_qty,
        size_usd=gross, token_qty=token_qty, fees=fees,
        realized_pnl=None, balance_after=balance_usd - gross,
        stop_mode=stop_mode, note=note,
    )
    position = Position(
        trade_id=trade_id, mint=snapshot.mint, symbol=snapshot.symbol,
        token_qty=token_qty, entry_cost_usd=gross, entry_ts=ts,
        stop_mode=stop_mode,
    )
    return fill, position


def quote_sell_net(settings: Settings, quote: dict) -> float:
    """Net USD proceeds implied by a token->USDC quote, after priority + TDS."""
    usd_out = int(quote["outAmount"]) / 10**USDC_DECIMALS
    return usd_out - settings.priority_fee_usd - settings.tds_rate * usd_out


def quote_sell(
    settings: Settings,
    arm: str,
    position: Position,
    snapshot: TokenSnapshot,
    quote: dict,
    balance_usd: float,
    note: str = "",
    ts: float | None = None,
) -> Fill:
    if snapshot.mint != position.mint:
        raise ExecutionError("snapshot token does not match the open position")

    ts = time.time() if ts is None else ts
    usd_out = int(quote["outAmount"]) / 10**USDC_DECIMALS
    priority = settings.priority_fee_usd
    tds = settings.tds_rate * usd_out
    net = usd_out - priority - tds

    mid_value = position.token_qty * snapshot.price_usd
    swap_cost = max(0.0, mid_value - usd_out)
    impact = abs(float(quote.get("priceImpactPct") or 0.0))
    slippage = min(swap_cost, mid_value * impact)
    lp = max(0.0, swap_cost - slippage)
    fees = FeeBreakdown(lp=lp, slippage=slippage, priority=priority, tds=tds)

    return Fill(
        ts=ts, arm=arm, trade_id=position.trade_id, side="sell",
        mint=position.mint, symbol=position.symbol,
        feed_price=snapshot.price_usd,
        exec_price=net / position.token_qty if position.token_qty else 0.0,
        size_usd=mid_value, token_qty=position.token_qty, fees=fees,
        realized_pnl=net - position.entry_cost_usd,
        balance_after=balance_usd + net,
        stop_mode=position.stop_mode, note=note,
    )

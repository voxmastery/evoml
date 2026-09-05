"""Strategy loop for one simulated account (arm).

As specified: enter with the full available balance in the picked token, exit
at +$X profit (after all modeled costs), optional stop-loss (including a
"no stop" mode), re-enter after each exit. Rotates tokens when a new window
picks a different one.
"""
from __future__ import annotations

import logging

from .config import Settings
from .csvlog import CsvMirror
from .db import Database
from .executor import (
    USDC_MINT, ExecutionError, buy_swap_units, quote_buy, quote_sell,
    quote_sell_net, sell_proceeds, simulate_buy, simulate_sell,
)
from .feed import PriceFeed
from .jupiter import JupiterQuoter, QuoteError
from .models import Decision, Fill, Position, TokenSnapshot

log = logging.getLogger(__name__)

MIN_TRADABLE_BALANCE_USD = 1.0


class TradingArm:
    def __init__(self, settings: Settings, db: Database, csv: CsvMirror,
                 feed: PriceFeed, arm: str,
                 quoter: JupiterQuoter | None = None):
        self._settings = settings
        self._db = db
        self._csv = csv
        self._feed = feed
        self._quoter = quoter
        if settings.fill_model == "quote" and quoter is None:
            raise ValueError("FILL_MODEL=quote requires a JupiterQuoter")
        self.arm = arm
        # Resume from the append-only log if the process restarted.
        self.balance = db.latest_balance(arm, settings.start_balance_usd)
        self.position: Position | None = db.open_position(arm)
        self.pick_mint: str | None = self.position.mint if self.position else None
        self.pending_exit: str | None = None  # analyst-requested exit reason
        self.busted = False
        if self.position:
            log.info("[%s] resumed open position in %s, balance $%.2f",
                     arm, self.position.symbol, self.balance)

    # --- wiring ---------------------------------------------------------------

    def _record(self, fill: Fill) -> None:
        self._db.insert_fill(fill)
        self._csv.append_fill(fill)
        self.balance = fill.balance_after

    def apply_decision(self, decision: Decision) -> None:
        self.pick_mint = decision.mint or None

    def request_exit(self, reason: str) -> None:
        """Ask for a market exit on the next tick (used by the chart analyst)."""
        if self.position is not None:
            self.pending_exit = reason

    # --- the loop body ----------------------------------------------------------

    async def tick(self, ts: float | None = None) -> None:
        if self.busted:
            return
        if self.position is not None:
            # Exit and re-entry never share a tick: re-buying at the exact
            # exit price would be unrealistically kind to the simulation.
            await self._maybe_exit(ts)
        elif self.pick_mint is not None:
            await self._maybe_enter(ts)

    def _exit_reason(self, pnl: float, pos: Position) -> str | None:
        if self.pending_exit is not None:
            return self.pending_exit
        if pnl >= self._settings.target_profit_usd:
            return "target"
        if (self._settings.stop_loss_pct is not None
                and pnl <= -pos.entry_cost_usd * self._settings.stop_loss_pct / 100.0):
            return "stop"
        if self.pick_mint is not None and self.pick_mint != pos.mint:
            return "rotate"
        return None

    async def _maybe_exit(self, ts: float | None) -> None:
        pos = self.position
        snap = self._feed.snapshot(pos.mint)
        if snap is None or snap.price_usd <= 0:
            return
        try:
            if self._settings.fill_model == "quote":
                fill = await self._quote_exit(pos, snap, ts)
            else:
                fill = self._model_exit(pos, snap, ts)
        except ExecutionError as e:
            log.warning("[%s] sell failed: %s", self.arm, e)
            return
        if fill is None:
            return
        self._record(fill)
        self.position = None
        self.pending_exit = None
        log.info("[%s] exit %s (%s) pnl $%+.2f balance $%.2f",
                 self.arm, pos.symbol, fill.note, fill.realized_pnl, self.balance)

    def _model_exit(self, pos: Position, snap: TokenSnapshot,
                    ts: float | None) -> Fill | None:
        net, _ = sell_proceeds(self._settings, pos, snap)
        reason = self._exit_reason(net - pos.entry_cost_usd, pos)
        if reason is None:
            return None
        return simulate_sell(self._settings, self.arm, pos, snap,
                             self.balance, note=reason, ts=ts)

    async def _quote_exit(self, pos: Position, snap: TokenSnapshot,
                          ts: float | None) -> Fill | None:
        if snap.decimals <= 0:
            return None
        units = int(pos.token_qty * 10**snap.decimals)
        if units <= 0:
            return None
        try:
            quote = await self._quoter.quote(pos.mint, USDC_MINT, units)
        except QuoteError as e:
            log.warning("[%s] sell quote failed, holding: %s", self.arm, e)
            return None
        reason = self._exit_reason(
            quote_sell_net(self._settings, quote) - pos.entry_cost_usd, pos)
        if reason is None:
            return None
        return quote_sell(self._settings, self.arm, pos, snap, quote,
                          self.balance, note=reason, ts=ts)

    async def _maybe_enter(self, ts: float | None) -> None:
        if self.balance < MIN_TRADABLE_BALANCE_USD:
            self.busted = True
            log.warning("[%s] balance $%.2f below tradable minimum — arm halted",
                        self.arm, self.balance)
            return
        snap = self._feed.snapshot(self.pick_mint)
        if snap is None or snap.price_usd <= 0:
            return
        try:
            if self._settings.fill_model == "quote":
                result = await self._quote_entry(snap, ts)
            else:
                if snap.liquidity_usd <= 0:
                    return
                result = simulate_buy(
                    self._settings, self.arm, snap, self.balance,
                    stop_mode=self._settings.stop_mode, note="enter", ts=ts,
                )
        except ExecutionError as e:
            log.warning("[%s] buy failed: %s", self.arm, e)
            return
        if result is None:
            return
        fill, position = result
        self._record(fill)
        self.position = position
        log.info("[%s] enter %s size $%.2f balance $%.2f",
                 self.arm, snap.symbol, fill.size_usd, self.balance)

    async def _quote_entry(
        self, snap: TokenSnapshot, ts: float | None
    ) -> tuple[Fill, Position] | None:
        if snap.decimals <= 0:
            return None
        units = buy_swap_units(self._settings, self.balance)
        try:
            quote = await self._quoter.quote(USDC_MINT, snap.mint, units)
        except QuoteError as e:
            log.warning("[%s] buy quote failed, staying flat: %s", self.arm, e)
            return None
        return quote_buy(self._settings, self.arm, snap, self.balance,
                         stop_mode=self._settings.stop_mode, quote=quote,
                         note="enter", ts=ts)

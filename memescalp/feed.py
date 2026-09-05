"""Market data: Jupiter prices every poll tick, DexScreener liquidity/trending.

Everything fetched is cached to SQLite so a run is reproducible after the fact.
"""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

from .db import Database
from .models import TokenSnapshot

log = logging.getLogger(__name__)

JUPITER_PRICE_URL_LITE = "https://lite-api.jup.ag/price/v3"
JUPITER_PRICE_URL_PRO = "https://api.jup.ag/price/v3"
DEXSCREENER_TOKENS_URL = "https://api.dexscreener.com/tokens/v1/solana/{mints}"
DEXSCREENER_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/top/v1"
DEXSCREENER_BOOSTS_LATEST_URL = "https://api.dexscreener.com/token-boosts/latest/v1"
DEXSCREENER_PROFILES_URL = "https://api.dexscreener.com/token-profiles/latest/v1"

LIQUIDITY_REFRESH_SECONDS = 60.0
MAX_CANDIDATES = 30
MIN_DISCOVERY_LIQUIDITY = 20_000.0  # newly discovered tokens need a real pool
PRICE_CHUNK = 50


class PriceFeed:
    """Holds the latest snapshot per mint and persists every observation."""

    def __init__(self, db: Database, watchlist: tuple[str, ...],
                 jupiter_api_key: str = ""):
        self._db = db
        self._pinned: frozenset[str] = frozenset(watchlist)  # never pruned
        self._watch: set[str] = set(watchlist)
        self._trend_seen: dict[str, float] = {}
        self._snapshots: dict[str, TokenSnapshot] = {}
        self._lock = asyncio.Lock()
        self._price_url = (JUPITER_PRICE_URL_PRO if jupiter_api_key
                           else JUPITER_PRICE_URL_LITE)
        headers = {"x-api-key": jupiter_api_key} if jupiter_api_key else {}
        self._client = httpx.AsyncClient(timeout=10.0, headers=headers)
        self._last_liquidity_fetch = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def add_to_watchlist(self, mint: str) -> None:
        async with self._lock:
            self._watch.add(mint)

    async def prune_watchlist(self, keep: set[str],
                              max_age_seconds: float = 6 * 3600.0) -> None:
        """Drop mints that stopped trending and aren't pinned, picked, or held,
        so the poll request size stays bounded over a weeks-long run."""
        now = time.time()
        async with self._lock:
            before = len(self._watch)
            self._watch = {
                m for m in self._watch
                if m in self._pinned or m in keep
                or now - self._trend_seen.get(m, 0.0) < max_age_seconds
            }
            dropped = before - len(self._watch)
        if dropped:
            log.info("pruned %d stale mints from the watchlist", dropped)

    def snapshot(self, mint: str) -> TokenSnapshot | None:
        return self._snapshots.get(mint)

    def all_snapshots(self) -> list[TokenSnapshot]:
        return list(self._snapshots.values())

    # --- polling ---------------------------------------------------------------

    async def poll_forever(self, interval_seconds: float) -> None:
        while True:
            started = time.time()
            try:
                await self.poll_once()
            except Exception:
                log.exception("price poll failed; will retry next tick")
            await asyncio.sleep(max(0.5, interval_seconds - (time.time() - started)))

    async def poll_once(self) -> None:
        async with self._lock:
            mints = sorted(self._watch)
        if not mints:
            return
        now = time.time()
        if now - self._last_liquidity_fetch >= LIQUIDITY_REFRESH_SECONDS:
            await self._refresh_dexscreener(mints)
            self._last_liquidity_fetch = now
        await self._refresh_prices(mints)

    async def _refresh_prices(self, mints: list[str]) -> None:
        entries: dict = {}
        for i in range(0, len(mints), PRICE_CHUNK):
            chunk = mints[i : i + PRICE_CHUNK]
            resp = await self._client.get(
                self._price_url, params={"ids": ",".join(chunk)}
            )
            resp.raise_for_status()
            payload = resp.json()
            # v3: {mint: {usdPrice}}; v2 wrapped it in {"data": {mint: {"price"}}}
            entries.update(payload.get("data", payload))
        ts = time.time()
        for mint, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            price = entry.get("usdPrice", entry.get("price"))
            if price is None:
                continue
            price = float(price)
            prev = self._snapshots.get(mint)
            symbol = prev.symbol if prev else mint[:6]
            decimals = int(entry.get("decimals")
                           or (prev.decimals if prev else 0))
            self._snapshots[mint] = TokenSnapshot(
                mint=mint, symbol=symbol, price_usd=price,
                liquidity_usd=prev.liquidity_usd if prev else 0.0,
                volume_h24=prev.volume_h24 if prev else 0.0,
                price_change_h1=prev.price_change_h1 if prev else 0.0,
                price_change_h24=prev.price_change_h24 if prev else 0.0,
                decimals=decimals,
            )
            self._db.insert_price(ts, mint, symbol, price, "jupiter")

    async def _refresh_dexscreener(self, mints: list[str]) -> None:
        ts = time.time()
        # DexScreener accepts up to 30 comma-separated token addresses per call.
        for i in range(0, len(mints), 30):
            chunk = mints[i : i + 30]
            resp = await self._client.get(
                DEXSCREENER_TOKENS_URL.format(mints=",".join(chunk))
            )
            resp.raise_for_status()
            pairs = resp.json() or []
            best: dict[str, dict] = {}
            for pair in pairs:
                base = pair.get("baseToken", {})
                mint = base.get("address")
                if mint not in chunk:
                    continue
                liq = float((pair.get("liquidity") or {}).get("usd") or 0.0)
                if mint not in best or liq > best[mint]["liq"]:
                    change = pair.get("priceChange") or {}
                    volume = pair.get("volume") or {}
                    txns_m5 = (pair.get("txns") or {}).get("m5") or {}
                    best[mint] = {
                        "liq": liq,
                        "symbol": base.get("symbol") or mint[:6],
                        "vol24": float(volume.get("h24") or 0.0),
                        "chg1": float(change.get("h1") or 0.0),
                        "chg24": float(change.get("h24") or 0.0),
                        "buys_m5": int(txns_m5.get("buys") or 0),
                        "sells_m5": int(txns_m5.get("sells") or 0),
                        "vol_m5": float(volume.get("m5") or 0.0),
                        "chg_m5": float(change.get("m5") or 0.0),
                    }
            for mint, info in best.items():
                prev = self._snapshots.get(mint)
                self._snapshots[mint] = TokenSnapshot(
                    mint=mint, symbol=info["symbol"],
                    price_usd=prev.price_usd if prev else 0.0,
                    liquidity_usd=info["liq"], volume_h24=info["vol24"],
                    price_change_h1=info["chg1"], price_change_h24=info["chg24"],
                    decimals=prev.decimals if prev else 0,
                    buys_m5=info["buys_m5"], sells_m5=info["sells_m5"],
                    vol_m5=info["vol_m5"], chg_m5=info["chg_m5"],
                )
                self._db.insert_liquidity(
                    ts, mint, info["symbol"], info["liq"], info["vol24"],
                    info["chg1"], info["chg24"],
                )
                self._db.insert_flow(
                    ts, mint, info["buys_m5"], info["sells_m5"],
                    info["vol_m5"], info["chg_m5"],
                )

    # --- trending candidates -----------------------------------------------------

    async def trending_candidates(self) -> list[TokenSnapshot]:
        """Solana tokens from three DexScreener discovery feeds (top boosts,
        newest boosts, latest profiles), hydrated and liquidity-filtered."""
        mints: list[str] = []
        for url in (DEXSCREENER_BOOSTS_URL, DEXSCREENER_BOOSTS_LATEST_URL,
                    DEXSCREENER_PROFILES_URL):
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
                entries = resp.json() or []
            except Exception:
                log.warning("discovery feed failed: %s", url)
                continue
            for e in entries:
                addr = e.get("tokenAddress")
                if e.get("chainId") == "solana" and addr and addr not in mints:
                    mints.append(addr)
        mints = mints[:40]

        now = time.time()
        for mint in mints:
            self._trend_seen[mint] = now
            await self.add_to_watchlist(mint)
        if mints:
            await self._refresh_dexscreener(sorted(set(mints)))
            await self._refresh_prices(sorted(self._watch))

        # Newly discovered tokens must have a real pool; anything we already
        # track keeps a lower bar. Relax if the market is quiet.
        seen: dict[str, TokenSnapshot] = {}
        for m in mints:
            s = self._snapshots.get(m)
            if (s is not None and s.price_usd > 0
                    and s.liquidity_usd >= MIN_DISCOVERY_LIQUIDITY):
                seen[m] = s
        for s in self._snapshots.values():
            if (s.mint not in seen and s.price_usd > 0
                    and s.liquidity_usd >= 1000):
                seen[s.mint] = s
        if len(seen) < 5:
            for m in mints:
                s = self._snapshots.get(m)
                if (s is not None and m not in seen and s.price_usd > 0
                        and s.liquidity_usd > 0):
                    seen[m] = s
        return list(seen.values())[:MAX_CANDIDATES]

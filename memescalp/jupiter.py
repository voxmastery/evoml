"""Read-only Jupiter swap-quote client.

Used to price simulated fills with the exact quote a real trader would
execute against (routing, price impact, and pool fees included). This module
only ever calls the GET /quote endpoint — it cannot build, sign, or submit a
transaction.
"""
from __future__ import annotations

import httpx

PRO_BASE = "https://api.jup.ag"
LITE_BASE = "https://lite-api.jup.ag"

DEFAULT_SLIPPAGE_BPS = 100


class QuoteError(Exception):
    pass


class JupiterQuoter:
    def __init__(self, api_key: str = ""):
        base = PRO_BASE if api_key else LITE_BASE
        self._url = f"{base}/swap/v1/quote"
        headers = {"x-api-key": api_key} if api_key else {}
        self._client = httpx.AsyncClient(timeout=10.0, headers=headers)

    async def close(self) -> None:
        await self._client.aclose()

    async def quote(self, input_mint: str, output_mint: str, amount: int,
                    slippage_bps: int = DEFAULT_SLIPPAGE_BPS) -> dict:
        """ExactIn quote: swap `amount` base units of input for output."""
        if amount <= 0:
            raise QuoteError("quote amount must be positive")
        try:
            resp = await self._client.get(self._url, params={
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": str(slippage_bps),
                "swapMode": "ExactIn",
            })
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise QuoteError(
                f"quote HTTP {e.response.status_code}: {e.response.text[:200]}"
            )
        except httpx.HTTPError as e:
            raise QuoteError(f"quote request failed: {e}")
        data = resp.json()
        if not isinstance(data, dict) or "outAmount" not in data:
            raise QuoteError(f"unexpected quote payload: {str(data)[:200]}")
        return data

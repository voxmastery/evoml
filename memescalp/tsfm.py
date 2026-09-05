"""Chronos-Bolt (Amazon, open weights, 47.7M params) consumed as a local
feature generator: its 15-minute probabilistic forecast becomes input for
EvoML and one expert vote in the hedge committee. Runs entirely on CPU;
weights cached locally by huggingface_hub. If torch/chronos are unavailable
the feature degrades to zeros and the experiment continues unaffected.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

MODEL_ID = "amazon/chronos-bolt-small"
CONTEXT_LEN = 64          # one-minute closes fed as context
MIN_CONTEXT = 20
QUANTILES = [0.1, 0.5, 0.9]


class ChronosFeatures:
    """Lazy-loaded, thread-safe wrapper. forecast_batch() is CPU-bound —
    call it via asyncio.to_thread from the event loop."""

    def __init__(self, prediction_length: int):
        self._prediction_length = int(prediction_length)
        self._pipe = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return not self._failed

    def _load(self):
        with self._lock:
            if self._pipe is not None or self._failed:
                return
            try:
                import torch
                from chronos import BaseChronosPipeline
                self._pipe = BaseChronosPipeline.from_pretrained(
                    MODEL_ID, device_map="cpu", torch_dtype=torch.float32)
                n = sum(p.numel() for p in self._pipe.model.parameters())
                log.info("[tsfm] %s loaded on CPU (%s params)", MODEL_ID, f"{n:,}")
            except Exception:
                self._failed = True
                log.exception("[tsfm] failed to load %s — feature disabled",
                              MODEL_ID)

    def forecast_batch(
        self, closes_by_key: dict[str, list[float]]
    ) -> dict[str, tuple[float, float]]:
        """key -> (median 15-min forecast return %, q10-q90 spread %).
        Keys with too little history are omitted."""
        self._load()
        if self._pipe is None:
            return {}
        import torch
        keys, contexts = [], []
        for key, closes in closes_by_key.items():
            usable = [c for c in closes[-CONTEXT_LEN:] if c > 0]
            if len(usable) < MIN_CONTEXT:
                continue
            keys.append(key)
            contexts.append(torch.tensor(usable, dtype=torch.float32))
        if not keys:
            return {}
        try:
            q, _ = self._pipe.predict_quantiles(
                contexts, prediction_length=self._prediction_length,
                quantile_levels=QUANTILES)
        except Exception:
            log.exception("[tsfm] inference failed this window")
            return {}
        out: dict[str, tuple[float, float]] = {}
        for i, key in enumerate(keys):
            last = float(contexts[i][-1])
            if last <= 0:
                continue
            lo = float(q[i, -1, 0])
            med = float(q[i, -1, 1])
            hi = float(q[i, -1, 2])
            out[key] = ((med / last - 1.0) * 100.0,
                        max(0.0, (hi - lo) / last * 100.0))
        return out

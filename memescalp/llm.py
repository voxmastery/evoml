"""LLM access for the coin picker.

Two backends:
- "claude_cli": headless `claude -p` — runs on the user's Claude subscription
  through the official Claude Code client. No API key involved.
- "api": the official Anthropic SDK with ANTHROPIC_API_KEY.

Note: this module never reads or forwards Claude Code's OAuth credentials
itself; subscription access only ever goes through the `claude` binary.
"""
from __future__ import annotations

import asyncio
import json
import logging

import anthropic

log = logging.getLogger(__name__)

CLI_TIMEOUT_SECONDS = 180.0


class LlmError(Exception):
    pass


class ClaudeCliBackend:
    name = "claude_cli"

    def __init__(self, model: str):
        self._model = model

    async def complete(self, prompt: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p", prompt,
                "--model", self._model,
                "--output-format", "json",
                # Skip plugin/MCP startup: cuts CLI latency ~22s -> ~4s.
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--setting-sources", "",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as e:
            raise LlmError(f"could not launch the claude CLI: {e}")
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=CLI_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise LlmError("claude CLI timed out")
        if proc.returncode != 0:
            raise LlmError(
                f"claude CLI exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}"
            )
        try:
            payload = json.loads(stdout.decode())
        except json.JSONDecodeError as e:
            raise LlmError(f"claude CLI returned non-JSON output: {e}")
        # Older CLIs emit a single result object; newer ones emit a list of
        # message objects ending with {"type": "result", "result": "..."}.
        if isinstance(payload, dict):
            payload = [payload]
        for entry in reversed(payload):
            if isinstance(entry, dict) and isinstance(entry.get("result"), str):
                return entry["result"]
        raise LlmError("claude CLI output contained no result message")


class AnthropicApiBackend:
    name = "api"

    def __init__(self, model: str, api_key: str):
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str) -> str:
        try:
            response = await self._client.beta.messages.create(
                model=self._model,
                max_tokens=2048,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.BadRequestError:
            # Fallback parameter unsupported for this model/account: plain call.
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
            except anthropic.APIStatusError as e:
                raise LlmError(f"API error {e.status_code}: {e.message}")
            except anthropic.APIConnectionError as e:
                raise LlmError(f"network error reaching the Anthropic API: {e}")
        except anthropic.RateLimitError as e:
            raise LlmError(f"rate limited: {e.message}")
        except anthropic.APIStatusError as e:
            raise LlmError(f"API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            raise LlmError(f"network error reaching the Anthropic API: {e}")

        if response.stop_reason == "refusal":
            raise LlmError("model declined the request (stop_reason=refusal)")
        return "".join(b.text for b in response.content if b.type == "text")


def make_backend(backend: str, model: str, api_key: str):
    if backend == "claude_cli":
        return ClaudeCliBackend(model)
    if backend == "api":
        return AnthropicApiBackend(model, api_key)
    raise ValueError(f"unknown picker backend: {backend!r}")

"""
OpenAI-spec chat-completions adapter.

Serves Qwen3-4B today. Any endpoint that speaks ``POST /chat/completions`` with
SSE deltas works unchanged -- point ``LLM_BASE_URL`` and ``LLM_MODEL`` at it.

Design notes:
  - One shared AsyncClient. A new TLS handshake per turn is ~100-300ms of dead
    air on the live path, so the connection pool is not optional.
  - Live and batch traffic hold separate semaphores. A queue of scorecards can
    never delay a candidate's next question.
  - Deadlines are enforced here, once, instead of at every call site.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx

from interviewer.config import LLMSettings, settings
from interviewer.ports.llm import (
    LLMFailure,
    LLMProvider,
    LLMRequest,
    LLMResult,
    LLMTask,
)

_LIVE_TASKS = {LLMTask.PHRASING}


class OpenAICompatLLM(LLMProvider):
    name = "openai_compat"

    def __init__(self, config: LLMSettings | None = None) -> None:
        self._cfg = config or settings.llm
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._live = asyncio.Semaphore(self._cfg.live_concurrency)
        self._batch = asyncio.Semaphore(self._cfg.batch_concurrency)

    # ── wiring ────────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                headers = {"Content-Type": "application/json"}
                if self._cfg.api_key:
                    headers["Authorization"] = f"Bearer {self._cfg.api_key}"
                self._client = httpx.AsyncClient(
                    base_url=self._cfg.base_url.rstrip("/"),
                    headers=headers,
                    timeout=httpx.Timeout(None, connect=self._cfg.connect_timeout_s),
                    limits=httpx.Limits(
                        max_connections=self._cfg.max_connections,
                        max_keepalive_connections=self._cfg.max_connections,
                    ),
                )
        return self._client

    def _model_for(self, task: LLMTask) -> str:
        override = {
            LLMTask.PHRASING: self._cfg.phrasing_model,
            LLMTask.JUDGE: self._cfg.judge_model,
            LLMTask.EXTRACT: self._cfg.extract_model,
        }.get(task, "")
        return override or self._cfg.model

    def _deadline_for(self, request: LLMRequest) -> float:
        if request.deadline_ms is not None:
            return max(1, request.deadline_ms) / 1000.0
        ms = {
            LLMTask.PHRASING: self._cfg.phrasing_deadline_ms,
            LLMTask.JUDGE: self._cfg.judge_deadline_ms,
            LLMTask.EXTRACT: self._cfg.extract_deadline_ms,
        }.get(request.task, self._cfg.phrasing_deadline_ms)
        return max(1, ms) / 1000.0

    def _gate(self, task: LLMTask) -> asyncio.Semaphore:
        return self._live if task in _LIVE_TASKS else self._batch

    def _payload(self, request: LLMRequest, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_for(request.task),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        return payload

    # ── calls ─────────────────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResult:
        started = time.perf_counter()
        model = self._model_for(request.task)

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000.0

        try:
            async with self._gate(request.task):
                client = await self._get_client()
                async with asyncio.timeout(self._deadline_for(request)):
                    response = await client.post("/chat/completions", json=self._payload(request, stream=False))
        except (TimeoutError, asyncio.TimeoutError):
            return LLMResult.failed(LLMFailure.TIMEOUT, elapsed(), model)
        except (httpx.HTTPError, OSError) as exc:
            return LLMResult.failed(LLMFailure.TRANSPORT, elapsed(), model, f"{type(exc).__name__}: {exc}")

        if response.status_code != 200:
            return LLMResult.failed(LLMFailure.BAD_STATUS, elapsed(), model, f"HTTP {response.status_code}")

        try:
            text = (response.json()["choices"][0]["message"]["content"] or "").strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return LLMResult.failed(LLMFailure.EMPTY, elapsed(), model, f"{type(exc).__name__}: {exc}")

        if not text:
            return LLMResult.failed(LLMFailure.EMPTY, elapsed(), model)
        return LLMResult(text=text, ok=True, latency_ms=elapsed(), model=model)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Yield content deltas. Ends quietly on deadline or transport fault --
        the caller keeps whatever arrived and falls back if that is nothing.
        """
        try:
            async with self._gate(request.task):
                client = await self._get_client()
                async with asyncio.timeout(self._deadline_for(request)):
                    async with client.stream(
                        "POST", "/chat/completions", json=self._payload(request, stream=True)
                    ) as response:
                        if response.status_code != 200:
                            return
                        async for line in response.aiter_lines():
                            delta = _parse_sse_delta(line)
                            if delta:
                                yield delta
        except (TimeoutError, asyncio.TimeoutError, httpx.HTTPError, OSError):
            return

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse_sse_delta(line: str) -> str:
    """Pull the content delta out of one ``data:`` frame. Returns '' otherwise."""
    if not line or not line.startswith("data:"):
        return ""
    body = line[5:].strip()
    if not body or body == "[DONE]":
        return ""
    try:
        choice = json.loads(body)["choices"][0]
    except (ValueError, KeyError, IndexError, TypeError):
        return ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""

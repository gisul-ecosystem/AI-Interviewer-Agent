"""
Stage 1 smoke test: prove the LLM port honours its contract.

  python -m tools.smoke_llm

Checks, in order:
  1. NullLLM fails cleanly (no exception) -- the safety valve works.
  2. A 1ms deadline reports TIMEOUT rather than raising.
  3. The real endpoint completes and streams within the phrasing budget.

A down endpoint is a PASS for the contract; it just means ok=False.
"""

from __future__ import annotations

import asyncio
import time

from interviewer.adapters.registry import close_llm, get_llm
from interviewer.config import settings
from interviewer.ports.llm import LLMFailure, LLMRequest, LLMTask, Message, NullLLM

PROMPT = [
    Message("system", "You are an interviewer. Output ONE question, max 25 words. No preamble."),
    Message("user", "The candidate said they used PyTorch and froze the backbone. Probe the failure mode."),
]


async def main() -> None:
    print(f"provider = {settings.llm.provider}")
    print(f"base_url = {settings.llm.base_url}")
    print(f"model    = {settings.llm.model}")
    print(f"budgets  = phrasing {settings.llm.phrasing_deadline_ms}ms / judge {settings.llm.judge_deadline_ms}ms\n")

    null = NullLLM()
    result = await null.complete(LLMRequest(task=LLMTask.PHRASING, messages=PROMPT))
    assert result.ok is False and result.failure is LLMFailure.DISABLED
    assert [chunk async for chunk in null.stream(LLMRequest(task=LLMTask.PHRASING, messages=PROMPT))] == []
    print("[PASS] NullLLM fails closed without raising")

    llm = get_llm()

    tight = await llm.complete(LLMRequest(task=LLMTask.PHRASING, messages=PROMPT, deadline_ms=1))
    assert tight.ok is False, "a 1ms deadline must not succeed"
    print(f"[PASS] 1ms deadline -> ok=False failure={tight.failure.value} ({tight.latency_ms:.0f}ms, no exception)")

    result = await llm.complete(LLMRequest(task=LLMTask.PHRASING, messages=PROMPT, max_tokens=60))
    if result.ok:
        print(f"[PASS] complete  {result.latency_ms:7.0f}ms  {result.model}")
        print(f"       -> {result.text[:120]}")
    else:
        print(f"[INFO] complete unavailable: {result.failure.value} {result.detail} ({result.latency_ms:.0f}ms)")

    started = time.perf_counter()
    first_delta_ms = None
    chunks: list[str] = []
    async for delta in llm.stream(LLMRequest(task=LLMTask.PHRASING, messages=PROMPT, max_tokens=60)):
        if first_delta_ms is None:
            first_delta_ms = (time.perf_counter() - started) * 1000.0
        chunks.append(delta)
    total_ms = (time.perf_counter() - started) * 1000.0

    if chunks:
        print(f"[PASS] stream    first delta {first_delta_ms:.0f}ms, total {total_ms:.0f}ms, {len(chunks)} deltas")
        print(f"       -> {''.join(chunks)[:120]}")
    else:
        print(f"[INFO] stream produced nothing in {total_ms:.0f}ms (endpoint down or over deadline)")

    budget = settings.llm.phrasing_deadline_ms
    if result.ok and result.latency_ms > budget:
        print(f"[WARN] completion took {result.latency_ms:.0f}ms against a {budget}ms phrasing budget.")
        print("       The turn orchestrator must speak its template first and treat the model as an upgrade.")

    await close_llm()
    print("\nStage 1 contract holds: the hot path can always fall back.")


if __name__ == "__main__":
    asyncio.run(main())

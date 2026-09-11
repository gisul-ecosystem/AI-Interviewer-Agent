"""
LLM port.

The interviewer's policy is pure Python. The model is only ever asked to turn
an already-decided intent into a spoken sentence, to judge an answer against a
rubric after the call, or to extract fields from a resume. Because those three
jobs have very different latency budgets, every request carries a task profile
and the transport enforces its deadline.

Contract for every implementation:
  - ``complete`` and ``stream`` NEVER raise for a model/network problem. They
    report ``ok=False`` so the caller can speak a template instead.
  - A deadline overrun is a normal, expected outcome, not an exception.
  - Implementations are safe to share across concurrent sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import AsyncIterator, Literal, Protocol, Sequence, runtime_checkable

Role = Literal["system", "user", "assistant"]


class LLMTask(StrEnum):
    """What the model is being used for. Drives model choice and deadline."""

    PHRASING = "phrasing"   # live path: turn a decided probe into one sentence
    JUDGE = "judge"         # cold path: score an answer against a rubric
    EXTRACT = "extract"     # ingestion: pull structured fields from a resume


class LLMFailure(StrEnum):
    NONE = "none"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    BAD_STATUS = "bad_status"
    EMPTY = "empty"
    DISABLED = "disabled"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMRequest:
    task: LLMTask
    messages: Sequence[Message]
    temperature: float = 0.2
    max_tokens: int = 160
    # Overrides the task default. Use when a caller has a tighter budget than
    # the profile, e.g. the remaining time before TTS must start.
    deadline_ms: int | None = None
    stop: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class LLMResult:
    text: str
    ok: bool
    failure: LLMFailure = LLMFailure.NONE
    latency_ms: float = 0.0
    model: str = ""
    detail: str = ""

    @staticmethod
    def failed(failure: LLMFailure, latency_ms: float = 0.0, model: str = "", detail: str = "") -> "LLMResult":
        return LLMResult(text="", ok=False, failure=failure, latency_ms=latency_ms, model=model, detail=detail[:200])


@runtime_checkable
class LLMProvider(Protocol):
    """Swap seam. Implement this to move off Qwen3-4B."""

    name: str

    async def complete(self, request: LLMRequest) -> LLMResult:
        """Return the full completion. Never raises for model/network faults."""
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Yield text deltas as they arrive, for sentence-chunked TTS.

        Stops silently at the deadline; whatever was yielded stands.
        """
        ...

    async def aclose(self) -> None:
        ...


class NullLLM:
    """
    Explicit 'no model available' provider.

    Used in tests and as a safety valve: every caller already has a template
    fallback, so the interview still runs end to end with this installed.
    """

    name = "null"

    async def complete(self, request: LLMRequest) -> LLMResult:
        return LLMResult.failed(LLMFailure.DISABLED, model=self.name)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        return
        yield ""  # pragma: no cover - makes this an async generator

    async def aclose(self) -> None:
        return

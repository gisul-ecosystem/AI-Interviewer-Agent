"""
Provider registry -- the single place that knows which adapter is live.

Adding a provider is two lines here plus one adapter file. Nothing in
services/ or api/ ever names a vendor.
"""

from __future__ import annotations

from typing import Callable

from interviewer.config import settings
from interviewer.ports.llm import LLMProvider, NullLLM

_BUILDERS: dict[str, Callable[[], LLMProvider]] = {}
_instance: LLMProvider | None = None


def register(name: str, builder: Callable[[], LLMProvider]) -> None:
    _BUILDERS[name] = builder


def _default_builders() -> None:
    if _BUILDERS:
        return

    def _openai_compat() -> LLMProvider:
        from interviewer.adapters.llm_openai_compat import OpenAICompatLLM

        return OpenAICompatLLM()

    register("openai_compat", _openai_compat)
    register("null", NullLLM)


def get_llm() -> LLMProvider:
    """Process-wide provider. Safe to call from any coroutine."""
    global _instance
    if _instance is None:
        _default_builders()
        builder = _BUILDERS.get(settings.llm.provider)
        if builder is None:
            raise ValueError(
                f"Unknown LLM_PROVIDER {settings.llm.provider!r}. Known: {sorted(_BUILDERS)}"
            )
        _instance = builder()
    return _instance


def set_llm(provider: LLMProvider | None) -> None:
    """Override the provider. Tests inject fakes; app shutdown passes None."""
    global _instance
    _instance = provider


async def close_llm() -> None:
    global _instance
    if _instance is not None:
        await _instance.aclose()
        _instance = None

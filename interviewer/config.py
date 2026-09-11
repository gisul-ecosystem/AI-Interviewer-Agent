"""
Single source of truth for runtime configuration.

Everything an operator might change between environments lives here and is
env-overridable. Modules import ``settings``; they never read os.environ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    path = BASE_DIR / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LLMSettings:
    """
    Provider-agnostic LLM config.

    ``provider`` selects the adapter. Any OpenAI-spec endpoint works with the
    default adapter, so upgrading Qwen3-4B to a stronger model is a URL and a
    model name -- not a code change.
    """

    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai_compat"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://llm.gisul.ai/v1"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M"))

    # Per-task model overrides. Empty means "use `model`". This is the seam for
    # running a small fast model on the live path and a stronger one for grading.
    phrasing_model: str = field(default_factory=lambda: _env("LLM_PHRASING_MODEL", ""))
    judge_model: str = field(default_factory=lambda: _env("LLM_JUDGE_MODEL", ""))
    extract_model: str = field(default_factory=lambda: _env("LLM_EXTRACT_MODEL", ""))

    # Hard deadlines in milliseconds. The live path must never block a candidate.
    phrasing_deadline_ms: int = field(default_factory=lambda: _env_int("LLM_PHRASING_DEADLINE_MS", 1200))
    judge_deadline_ms: int = field(default_factory=lambda: _env_int("LLM_JUDGE_DEADLINE_MS", 20_000))
    extract_deadline_ms: int = field(default_factory=lambda: _env_int("LLM_EXTRACT_DEADLINE_MS", 15_000))

    # Separate budgets so a backlog of cold-path work cannot starve live turns.
    live_concurrency: int = field(default_factory=lambda: _env_int("LLM_LIVE_CONCURRENCY", 16))
    batch_concurrency: int = field(default_factory=lambda: _env_int("LLM_BATCH_CONCURRENCY", 2))

    connect_timeout_s: float = field(default_factory=lambda: _env_float("LLM_CONNECT_TIMEOUT_S", 2.0))
    max_connections: int = field(default_factory=lambda: _env_int("LLM_MAX_CONNECTIONS", 32))


@dataclass(frozen=True)
class SpeechSettings:
    # whisper_api = Groq Whisper large-v3-turbo. whisper = local faster-whisper.
    # Nemotron / FastConformer is not used and is ignored if set.
    stt_provider: str = field(default_factory=lambda: _env("STT_PROVIDER", "whisper_api"))
    stt_model: str = field(default_factory=lambda: _env("STT_WHISPER_MODEL", "whisper-large-v3-turbo"))
    stt_device: str = field(default_factory=lambda: _env("STT_DEVICE", "auto"))
    stt_language: str = field(default_factory=lambda: _env("STT_LANGUAGE", "en"))
    stt_api_key: str = field(default_factory=lambda: _env("STT_API_KEY", _env("GROQ_API_KEY", "")))
    stt_api_base: str = field(
        default_factory=lambda: _env("STT_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
    )
    tts_url: str = field(default_factory=lambda: _env("TTS_URL", "https://tts.gisul.ai/synthesize"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "af_heart"))
    tts_deadline_ms: int = field(default_factory=lambda: _env_int("TTS_DEADLINE_MS", 20000))
    sample_rate: int = field(default_factory=lambda: _env_int("AUDIO_SAMPLE_RATE", 16_000))


@dataclass(frozen=True)
class StorageSettings:
    db_path: str = field(default_factory=lambda: _env("INTERVIEW_DB_PATH", str(BASE_DIR / "interviews.db")))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    use_redis: bool = field(default_factory=lambda: _env_bool("USE_REDIS", True))
    session_ttl_s: int = field(default_factory=lambda: _env_int("SESSION_TTL_S", 7200))


@dataclass(frozen=True)
class InterviewSettings:
    question_bank_path: str = field(
        default_factory=lambda: _env("QUESTION_BANK_PATH", str(BASE_DIR / "question_bank.json"))
    )
    project_questions: int = field(default_factory=lambda: _env_int("PROJECT_QUESTIONS", 4))
    skill_questions: int = field(default_factory=lambda: _env_int("SKILL_QUESTIONS", 3))
    max_projects: int = field(default_factory=lambda: _env_int("MAX_PROJECTS", 2))
    max_skills: int = field(default_factory=lambda: _env_int("MAX_SKILLS", 2))
    duration_seconds: int = field(default_factory=lambda: _env_int("INTERVIEW_DURATION_S", 900))
    wrap_up_seconds: int = field(default_factory=lambda: _env_int("INTERVIEW_WRAP_UP_S", 90))
    resume_context_chars: int = field(default_factory=lambda: _env_int("RESUME_CONTEXT_CHARS", 6000))
    api_key: str = field(default_factory=lambda: _env("INTERVIEW_API_KEY", ""))
    allowed_origins: str = field(default_factory=lambda: _env("ALLOWED_ORIGINS", "*"))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 8))
    uvicorn_reload: bool = field(default_factory=lambda: _env_bool("UVICORN_RELOAD", True))


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    interview: InterviewSettings = field(default_factory=InterviewSettings)
    base_dir: Path = BASE_DIR


settings = Settings()

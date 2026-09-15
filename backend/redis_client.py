"""
Redis Client and Session Persistence Layer for AI Interview System.

Provides robust, production-grade session persistence with seamless in-memory fallback.
If Redis is running (locally or remote), all interview state, conversation history, and
evaluation telemetry are persisted with TTLs. If Redis is unavailable, it gracefully
falls back to in-memory state without crashing the server.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Interview.Redis")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False


class SessionStore:
    """
    Hybrid Session Store: Redis-backed with in-memory fallback.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 7200  # 2 hours
    ):
        self.default_ttl = default_ttl
        self.redis_client = None
        self._memory_sessions: Dict[str, Dict[str, Any]] = {}
        self._memory_latest_cv: Dict[str, Any] = {}

        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            from interviewer.config import settings as _storage_settings
            use_redis = _storage_settings.storage.use_redis
            url = redis_url or _storage_settings.storage.redis_url or url
        except Exception:
            use_redis = os.getenv("USE_REDIS", "1").strip().lower() in ("1", "true", "yes", "on")
        if REDIS_AVAILABLE and use_redis:
            try:
                client = redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=1.5,
                    socket_timeout=1.5,
                )
                client.ping()
                self.redis_client = client
                safe_url = url.split("@")[-1] if "@" in url else url
                logger.info("[SessionStore] Successfully connected to Redis at %s", safe_url)
            except Exception as e:
                logger.warning(f"[SessionStore] Redis connection failed ({e}). Falling back to in-memory store.")
                self.redis_client = None
        else:
            logger.info("[SessionStore] Redis disabled or library not available. Using in-memory store.")

    @property
    def is_redis_active(self) -> bool:
        return self.redis_client is not None

    # ─── Session Management ───────────────────────────────────────────────────

    def save_session(self, session_id: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Persist entire session dict."""
        expire = ttl if ttl is not None else self.default_ttl
        self._memory_sessions[session_id] = data
        if self.redis_client:
            try:
                # Exclude runtime-only Python objects (ResumeRAG, compiled regexes)
                clean_data = {
                    k: v for k, v in data.items()
                    if k not in ("session_rag", "phonetic_replacements")
                }
                key = f"interview:session:{session_id}"
                self.redis_client.set(key, json.dumps(clean_data, default=str, ensure_ascii=False), ex=expire)
                return
            except Exception as e:
                logger.error(f"[SessionStore] Redis save_session error ({e}), storing in memory.")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session dict by ID. Redis first, then this process's memory."""
        if self.redis_client:
            try:
                key = f"interview:session:{session_id}"
                raw = self.redis_client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.error(f"[SessionStore] Redis get_session error ({e}), checking memory.")
        return self._memory_sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """Remove session."""
        if self.redis_client:
            try:
                self.redis_client.delete(f"interview:session:{session_id}")
            except Exception as e:
                logger.error(f"[SessionStore] Redis delete error: {e}")
        self._memory_sessions.pop(session_id, None)

    # ─── Latest Uploaded CV Features ──────────────────────────────────────────

    def save_latest_cv_features(self, features: Dict[str, Any]) -> None:
        """Cache the most recent uploaded CV features."""
        if self.redis_client:
            try:
                self.redis_client.set("interview:latest_cv", json.dumps(features, ensure_ascii=False), ex=86400)
                return
            except Exception as e:
                logger.error(f"[SessionStore] Redis save_latest_cv error: {e}")
        self._memory_latest_cv = features

    def get_latest_cv_features(self) -> Dict[str, Any]:
        """Fetch the most recent uploaded CV features."""
        if self.redis_client:
            try:
                raw = self.redis_client.get("interview:latest_cv")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.error(f"[SessionStore] Redis get_latest_cv error: {e}")
        return self._memory_latest_cv

    # ─── Per-upload resume profiles (multi-candidate isolation) ───────────────

    def save_resume_profile(self, token: str, data: Dict[str, Any], ttl: int = 86400) -> None:
        payload = json.dumps(data, default=str, ensure_ascii=False)
        if self.redis_client:
            try:
                self.redis_client.set(f"interview:resume:{token}", payload, ex=ttl)
            except Exception as e:
                logger.error(f"[SessionStore] Redis save_resume_profile error: {e}")
        self._memory_sessions[f"resume:{token}"] = data

    def get_resume_profile(self, token: str) -> Optional[Dict[str, Any]]:
        if self.redis_client:
            try:
                raw = self.redis_client.get(f"interview:resume:{token}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.error(f"[SessionStore] Redis get_resume_profile error: {e}")
        return self._memory_sessions.get(f"resume:{token}")

    # ─── Scorecards produced by the async grading pass ────────────────────────

    def save_report(self, session_id: str, report: Dict[str, Any], ttl: int = 604800) -> None:
        """Persist a grading report (default TTL 7 days so recruiters can read it)."""
        self._memory_sessions[f"report:{session_id}"] = report
        if self.redis_client:
            try:
                self.redis_client.set(
                    f"interview:report:{session_id}",
                    json.dumps(report, default=str, ensure_ascii=False),
                    ex=ttl,
                )
            except Exception as e:
                logger.error(f"[SessionStore] Redis save_report error: {e}")

    def get_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self.redis_client:
            try:
                raw = self.redis_client.get(f"interview:report:{session_id}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.error(f"[SessionStore] Redis get_report error: {e}")
        return self._memory_sessions.get(f"report:{session_id}")

    # ─── Granular History & Turns ─────────────────────────────────────────────

    def append_turn_history(self, session_id: str, turn: Dict[str, Any]) -> None:
        """Append a completed turn record to the session history list."""
        sess = self.get_session(session_id)
        if sess is not None:
            if "history" not in sess:
                sess["history"] = []
            sess["history"].append(turn)
            sess["last_turn_time"] = time.time()
            self.save_session(session_id, sess)


# Global singleton instance
session_store = SessionStore()

"""Mint LiveKit join tokens. One interview session = one room."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

from interviewer.config import _load_dotenv, settings as _settings


def _livekit_env() -> Tuple[str, str, str]:
    """Read LiveKit creds from .env at call time (not the frozen boot snapshot)."""
    _load_dotenv()
    url = (os.getenv("LIVEKIT_URL") or _settings.speech.livekit_url or "").strip()
    key = (os.getenv("LIVEKIT_API_KEY") or _settings.speech.livekit_api_key or "").strip()
    secret = (os.getenv("LIVEKIT_API_SECRET") or _settings.speech.livekit_api_secret or "").strip()
    return url, key, secret


def room_name_for_session(session_id: str) -> str:
    return f"interview-{session_id}"


def livekit_client_url(raw: Optional[str] = None) -> str:
    url = (raw if raw is not None else _livekit_env()[0]).strip()
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def is_livekit_configured() -> bool:
    url, key, secret = _livekit_env()
    return bool(livekit_client_url(url) and key and secret)


def mint_candidate_token(session_id: str, identity_name: str = "Candidate") -> Dict[str, Any]:
    if not session_id:
        raise ValueError("session_id is required")
    url, key, secret = _livekit_env()
    if not (livekit_client_url(url) and key and secret):
        raise RuntimeError("LiveKit is not configured")

    from livekit import api

    room = room_name_for_session(session_id)
    identity = f"candidate-{session_id}"
    token = (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(identity_name or "Candidate")
        .with_ttl(timedelta(seconds=max(300, int(_settings.storage.session_ttl_s or 7200))))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )
    return {
        "url": livekit_client_url(url),
        "token": token,
        "room": room,
        "identity": identity,
    }


def mint_agent_token(session_id: str) -> Dict[str, Any]:
    if not session_id:
        raise ValueError("session_id is required")
    url, key, secret = _livekit_env()
    if not (livekit_client_url(url) and key and secret):
        raise RuntimeError("LiveKit is not configured")

    from livekit import api

    room = room_name_for_session(session_id)
    identity = f"ai-interviewer-{session_id}"
    token = (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name("AI Interviewer")
        .with_ttl(timedelta(seconds=max(300, int(_settings.storage.session_ttl_s or 7200))))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )
    return {
        "url": livekit_client_url(url),
        "token": token,
        "room": room,
        "identity": identity,
    }

"""Regression checks for the live interview session contract.

Run with: ``python -m backend.test_session_contract``
"""

import asyncio

from fastapi import HTTPException

import app


async def _expect_http_error(coro, status_code: int) -> None:
    try:
        await coro
    except HTTPException as exc:
        assert exc.status_code == status_code, exc.detail
    else:
        raise AssertionError(f"Expected HTTP {status_code}")


def test_session_contract() -> None:
    asyncio.run(_session_contract())


async def _session_contract() -> None:
    features = {
        "name": "Test Candidate",
        "skills": ["Python", "Machine Learning"],
        "projects": ["Interview Test Project"],
        "role": "AI / ML Engineer",
        "degree": "B.Tech",
        "college": "Test University",
    }

    created = await app.start_interview({"role": "aiml", "features": features})
    assert created["status"] == "ok"
    session_id = created["session_id"]
    live = app.load_live_session(session_id)
    assert live is not None
    assert live["candidate"]["target_track"]

    await _expect_http_error(app.api_interview_turn_stream({"transcript": "hello"}), 400)
    await _expect_http_error(
        app.api_interview_turn_stream({"session_id": "not-a-session", "transcript": "hello"}),
        404,
    )
    await _expect_http_error(app.api_interview_turn({"transcript": "hello"}), 400)

    print("[SUCCESS] Live interview session contract checks passed.")


if __name__ == "__main__":
    test_session_contract()

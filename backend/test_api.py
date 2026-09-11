"""HTTP contract tests. Run: python -m backend.test_api"""

from fastapi.testclient import TestClient

import app
from app import _should_complete_after_turn
from backend.intent_engine import detect_candidate_intent


def test_health_and_upload_and_closing_gate():
    with TestClient(app.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        ready = client.get("/api/ready")
        assert ready.status_code == 200

        cv = (
            "Test Candidate\n\nSKILLS\nPython, FastAPI\n\nPROJECTS\n"
            "Ferrite Mesh - A service mesh sidecar.\n"
        )
        up = client.post("/api/upload-resume", files={"file": ("cv.txt", cv.encode("utf-8"), "text/plain")})
        assert up.status_code == 200
        body = up.json()
        assert body["status"] == "ok"
        assert body.get("resume_token")
        assert "Ferrite Mesh" in (body.get("extracted_features") or {}).get("projects", [])

        started = client.post(
            "/api/start-interview",
            json={"role": "auto", "resume_token": body["resume_token"]},
        )
        assert started.status_code == 200
        session_id = started.json()["session_id"]

        closing_state = {"stage": "closing", "action": "CONTINUE"}
        assert _should_complete_after_turn(closing_state, None) is False
        done_state = {"stage": "completed", "action": "END_INTERVIEW"}
        assert _should_complete_after_turn(done_state, None) is True

        turn = client.post(
            "/api/interview-turn",
            json={"session_id": session_id, "transcript": "I built Ferrite Mesh in Rust for mTLS retries."},
        )
        assert turn.status_code == 200
        payload = turn.json()
        assert payload.get("status") == "ok"
        assert payload.get("interview_state", {}).get("stage") != "completed"

        print("[SUCCESS] API health/upload/closing-gate checks passed.")


def test_intent_skip_not_false_positive():
    assert detect_candidate_intent("I move to Kubernetes after the build step") == "TECHNICAL_ANSWER"
    assert detect_candidate_intent("skip this") == "UNKNOWN_OR_SKIP"
    assert detect_candidate_intent("can we switch to Python") == "TOPIC_CHANGE"
    assert detect_candidate_intent("could you repeat the question") == "REPEAT_REQUEST"
    print("[SUCCESS] Intent skip/pivot checks passed.")


if __name__ == "__main__":
    test_intent_skip_not_false_positive()
    test_health_and_upload_and_closing_gate()

"""Durable interview store: transcript survives Redis expiry and restarts."""

import os
import tempfile

from backend.store import InterviewStore


def _session(sid="sess-1", turns=2):
    return {
        "session_id": sid,
        "created_at": 1000.0,
        "ended_at": 2000.0,
        "candidate": {"name": "Priya Sharma", "target_role": "AI / ML Engineer", "target_track": "aiml"},
        "job_requirements": {"title": "AI / ML Engineer", "intersection": ["Python"]},
        "interview_state": {"stage": "completed", "difficulty_level": 3},
        "resume_text": "MoleCheck ResNet50 dermoscopy classifier",
        "history": [
            {
                "turn": i,
                "asked_question": f"Question {i}?",
                "candidate_answer": f"Answer {i} with ResNet50 and AUC 0.91",
                "next_question": f"Question {i+1}?",
                "competency": "deep_learning" if i % 2 else "python",
                "topic": f"Skill: T{i}",
                "question_id": f"q-{i}",
                "question_difficulty": 2,
                "question_mode": "RETRIEVE",
                "live_score": 0.7,
                "expected_concepts": ["pretrained backbone"],
                "asked_at": 1500.0 + i,
            }
            for i in range(1, turns + 1)
        ],
    }


def _fresh_store():
    path = os.path.join(tempfile.mkdtemp(), "test_interviews.db")
    return InterviewStore(db_path=path), path


def test_transcript_round_trip():
    store, _ = _fresh_store()
    assert store.is_available

    assert store.save_interview(_session()) is True
    loaded = store.get_interview("sess-1")

    assert loaded is not None
    assert loaded["candidate"]["name"] == "Priya Sharma"
    assert len(loaded["history"]) == 2
    # The actual words the candidate said must come back verbatim.
    assert loaded["history"][0]["candidate_answer"] == "Answer 1 with ResNet50 and AUC 0.91"
    assert loaded["history"][0]["asked_question"] == "Question 1?"
    assert loaded["history"][0]["expected_concepts"] == ["pretrained backbone"]
    assert loaded["history"][1]["turn"] == 2
    print("[OK] Transcript round-trips with questions, answers, and rubric intact.")


def test_upsert_is_idempotent():
    store, _ = _fresh_store()
    store.save_interview(_session(turns=2))
    store.save_interview(_session(turns=2))  # replayed persist

    grown = _session(turns=4)  # interview continued
    store.save_interview(grown)

    loaded = store.get_interview("sess-1")
    assert len(loaded["history"]) == 4, len(loaded["history"])
    turn_ids = [t["turn"] for t in loaded["history"]]
    assert turn_ids == [1, 2, 3, 4], turn_ids
    print("[OK] Repeated writes do not duplicate turns; new turns append.")


def test_report_outlives_and_joins():
    store, _ = _fresh_store()
    store.save_interview(_session())
    store.save_report("sess-1", {
        "session_id": "sess-1",
        "overall_score": 3.8,
        "recommendation": "hire",
        "graded_by": "llm",
        "turns_graded": 2,
        "competencies": [{"competency": "deep_learning", "score": 4.0}],
    })

    report = store.get_report("sess-1")
    assert report["recommendation"] == "hire"
    assert report["competencies"][0]["competency"] == "deep_learning"

    rows = store.list_interviews()
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_name"] == "Priya Sharma"
    assert row["overall_score"] == 3.8
    assert row["turn_count"] == 2
    print(f"[OK] Listing joins score to interview: {row['candidate_name']} {row['overall_score']}/5 {row['recommendation']}")


def test_survives_reopen():
    store, path = _fresh_store()
    store.save_interview(_session())

    reopened = InterviewStore(db_path=path)  # simulates a server restart
    loaded = reopened.get_interview("sess-1")
    assert loaded is not None
    assert loaded["history"][0]["candidate_answer"].startswith("Answer 1")
    print("[OK] Record survives process restart.")


def test_filters_and_missing():
    store, _ = _fresh_store()
    store.save_interview(_session("a"))
    s = _session("b")
    s["candidate"]["target_role"] = "Web Developer"
    store.save_interview(s)
    store.save_report("a", {"recommendation": "hire", "overall_score": 3.5})

    assert len(store.list_interviews(target_role="Web Developer")) == 1
    assert len(store.list_interviews(recommendation="hire")) == 1
    assert store.get_interview("does-not-exist") is None
    assert store.get_report("does-not-exist") is None
    print("[OK] Role/recommendation filters work; missing ids return None.")


if __name__ == "__main__":
    test_transcript_round_trip()
    test_upsert_is_idempotent()
    test_report_outlives_and_joins()
    test_survives_reopen()
    test_filters_and_missing()
    print("\n[SUCCESS] Durable interview store verified.")

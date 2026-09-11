"""Async grading pass: rubric judging, quote verification, heuristic fallback."""

import asyncio
import json

from backend.grader import (
    aggregate_report,
    grade_session,
    grade_turn,
    heuristic_grade,
    parse_judge_output,
    verify_quote,
)


ANSWER = (
    "In MoleCheck I fine-tuned a ResNet50 backbone on about 12000 dermoscopy images. "
    "I froze the first three blocks and trained the head for 20 epochs, then unfroze gradually. "
    "Validation AUC went from 0.81 to 0.91, and I added class weighting because melanoma cases "
    "were only 8 percent of the data."
)

TURN = {
    "turn": 1,
    "asked_question": "For an image classifier with limited labelled data, when would you use transfer learning?",
    "candidate_answer": ANSWER,
    "competency": "deep_learning",
    "topic": "Skill: Deep Learning",
    "question_id": "aiml-cnn-transfer-01",
    "question_difficulty": 3,
    "expected_concepts": ["pretrained backbone", "limited data", "freeze layers", "fine tuning"],
    "strong_signals": ["explains why early features transfer"],
    "weak_signals": ["always trains from scratch"],
}

NO_ANSWER_TURN = {
    "turn": 2,
    "asked_question": "How does the JVM distinguish Stack and Heap memory?",
    "candidate_answer": "i don't know",
    "competency": "java",
    "expected_concepts": ["stack", "heap", "garbage collection"],
}

VAGUE_TURN = {
    "turn": 3,
    "asked_question": "How did you manage client state in the UI?",
    "candidate_answer": "I think we probably used some state management, maybe redux or something like that, not sure.",
    "competency": "frontend_state",
    "expected_concepts": ["state", "props", "re-render"],
}


def _stub_llm(response_body: str):
    async def _call(payload):
        assert payload["temperature"] == 0.0, "judge must be deterministic"
        assert "CANDIDATE_ANSWER" in payload["messages"][1]["content"]
        return response_body
    return _call


def test_quote_verification():
    assert verify_quote("I froze the first three blocks", ANSWER) is True
    assert verify_quote("Validation AUC went from 0.81 to 0.91", ANSWER) is True
    # A plausible-sounding quote the candidate never said must be rejected.
    assert verify_quote("I deployed the model to Kubernetes with autoscaling", ANSWER) is False
    assert verify_quote("yes", ANSWER) is False
    print("[OK] Quote verification accepts real quotes and rejects invented ones.")


def test_parse_drops_hallucinated_evidence():
    raw = json.dumps({
        "score": 4,
        "verdict": "strong",
        "covered_concepts": ["pretrained backbone", "freeze layers"],
        "missing_concepts": [],
        "evidence": [
            {"quote": "I froze the first three blocks", "why": "concrete fine-tuning strategy"},
            {"quote": "I ran a 5-fold cross validation sweep", "why": "invented"},
        ],
        "red_flags": [],
        "summary": "Concrete transfer learning experience.",
    })
    parsed = parse_judge_output(raw, ANSWER)
    assert parsed is not None
    assert parsed["score"] == 4
    assert parsed["graded_by"] == "llm"
    assert len(parsed["evidence"]) == 1, parsed["evidence"]
    assert "froze the first three blocks" in parsed["evidence"][0]["quote"]

    # Markdown-fenced output must still parse.
    assert parse_judge_output("```json\n{\"score\": 3, \"verdict\": \"adequate\"}\n```", ANSWER)["score"] == 3
    # Out-of-range and unparseable output must be rejected so we fall back.
    assert parse_judge_output('{"score": 9, "verdict": "strong"}', ANSWER) is None
    assert parse_judge_output("I would give this a solid 4 out of 5.", ANSWER) is None
    print("[OK] Judge parser strips fences, rejects bad scores, drops fake quotes.")


def test_heuristic_fallback_separates_answers():
    strong = heuristic_grade(TURN)
    vague = heuristic_grade(VAGUE_TURN)
    empty = heuristic_grade(NO_ANSWER_TURN)

    assert empty["verdict"] == "no_answer" and empty["score"] == 1
    assert vague["score"] < strong["score"], (vague["score"], strong["score"])
    assert "heavily hedged" in vague["red_flags"]
    assert all(g["graded_by"] == "heuristic" for g in (strong, vague, empty))
    print(f"[OK] Heuristic separates answers: strong={strong['score']} vague={vague['score']} none={empty['score']}")


def test_grading_survives_llm_failure():
    async def _boom(payload):
        raise RuntimeError("llm.gisul.ai timeout")

    graded = asyncio.run(grade_turn(TURN, {}, _boom))
    assert graded["graded_by"] == "heuristic"
    assert "grader_error" in graded
    assert graded["score"] >= 1

    no_llm = asyncio.run(grade_turn(TURN, {}, None))
    assert no_llm["graded_by"] == "heuristic"
    print("[OK] Judge failure downgrades to heuristic instead of losing the turn.")


def test_full_session_report():
    session = {
        "session_id": "sess-test",
        "candidate": {"name": "Priya Sharma", "target_role": "AI / ML Engineer"},
        "history": [TURN, VAGUE_TURN, NO_ANSWER_TURN, {"candidate_answer": "   "}],
    }
    body = json.dumps({
        "score": 4,
        "verdict": "strong",
        "covered_concepts": ["freeze layers"],
        "missing_concepts": [],
        "evidence": [{"quote": "I froze the first three blocks", "why": "specific"}],
        "red_flags": [],
        "summary": "Strong.",
    })
    report = asyncio.run(grade_session(session, llm_call=_stub_llm(body), concurrency=2))

    assert report["turns_graded"] == 3, "blank answers must not be graded"
    assert report["session_id"] == "sess-test"
    assert report["candidate_name"] == "Priya Sharma"
    assert 1 <= report["overall_score"] <= 5
    assert report["recommendation"] in ("strong_hire", "hire", "borderline", "no_hire")
    competencies = {c["competency"] for c in report["competencies"]}
    assert {"deep_learning", "java", "frontend_state"} <= competencies, competencies

    # "i don't know" is decided by rule, so a lenient judge cannot award it points.
    by_turn = {g["turn"]: g for g in report["turn_grades"]}
    assert by_turn[2]["graded_by"] == "heuristic"
    assert by_turn[2]["verdict"] == "no_answer" and by_turn[2]["score"] == 1
    assert by_turn[1]["graded_by"] == "llm"
    assert report["graded_by"] == "mixed"
    assert "java" in report["weaknesses"]
    print(
        f"[OK] Report: overall={report['overall_score']}/5 rec={report['recommendation']} "
        f"competencies={len(report['competencies'])}"
    )


def test_aggregate_rolls_up_per_competency():
    grades = [
        {"competency": "python", "score": 5, "verdict": "strong", "evidence": [], "missing_concepts": [], "red_flags": [], "graded_by": "llm"},
        {"competency": "python", "score": 4, "verdict": "strong", "evidence": [], "missing_concepts": [], "red_flags": [], "graded_by": "llm"},
        {"competency": "sql", "score": 2, "verdict": "weak", "evidence": [], "missing_concepts": ["indexing"], "red_flags": [], "graded_by": "llm"},
    ]
    report = aggregate_report(grades)
    by_name = {c["competency"]: c for c in report["competencies"]}
    assert by_name["python"]["score"] == 4.5 and by_name["python"]["level"] == "strong"
    assert by_name["sql"]["level"] == "weak"
    assert report["strengths"] == ["python"]
    assert report["weaknesses"] == ["sql"]
    assert report["graded_by"] == "llm"
    print("[OK] Uneven candidates surface as per-competency scores, not one average.")


def test_grade_session_heuristic_when_judge_raises():
    async def boom(_payload):
        raise RuntimeError("timeout")

    report = asyncio.run(grade_session({
        "session_id": "sess-fail",
        "candidate": {"name": "A", "target_role": "Backend"},
        "history": [TURN],
    }, llm_call=boom))
    assert report["overall_score"] is not None
    assert report["overall_score"] >= 1
    assert report["graded_by"] == "heuristic"
    print("[OK] Session still receives a scorecard when the judge raises.")


if __name__ == "__main__":
    test_quote_verification()
    test_parse_drops_hallucinated_evidence()
    test_heuristic_fallback_separates_answers()
    test_grading_survives_llm_failure()
    test_full_session_report()
    test_aggregate_rolls_up_per_competency()
    test_grade_session_heuristic_when_judge_raises()
    print("\n[SUCCESS] Async grading pass verified.")

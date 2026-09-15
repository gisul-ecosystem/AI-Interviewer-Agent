"""
Unit tests for the 4 core architecture fixes + contract guarantees:
1. extract_resume_features_llm runs on the active asyncio loop and falls back on error.
2. Hard cap on project quotas: PROBE_DEEPER cannot stall advancement once quota is reached.
3. TOPIC_CHANGE and skips do not hijack spoken questions away from planned agenda.
4. 15-minute contract: Clock reaching 0 during projects cuts to OOP/DSA instead of closing.
5. Empty project list skips project_deep_dive directly to skills without phantom project.
6. INVALID_FIELD_WORDS does not filter out legitimate compound skills (e.g. iOS Development, User Experience).
"""

import inspect
import pytest
from unittest.mock import AsyncMock, patch

from backend.interview_fsm import InterviewFSM
from backend.interview_plan import build_interview_plan
from interviewer.services.interview_structure import planned_question_count, opening_greeting


def test_extract_resume_features_llm_is_async():
    """Finding 1: LLM extract must be an async coroutine running on FastAPI loop."""
    import app
    assert inspect.iscoroutinefunction(app.extract_resume_features_llm)


@pytest.mark.anyio
async def test_extract_resume_features_llm_fallback_on_error():
    """Finding 1: LLM extract falls back cleanly on error without dying or throwing."""
    import app
    with patch("interviewer.adapters.registry.get_llm") as mock_get_llm:
        mock_adapter = AsyncMock()
        mock_adapter.complete.side_effect = RuntimeError("Transport timeout")
        mock_get_llm.return_value = mock_adapter

        result = await app.extract_resume_features_llm("Name: John Doe\nSkills: Python, Go")
        assert isinstance(result, dict)
        assert "name" in result or "skills" in result


def test_probe_deeper_cannot_stall_project_advance():
    """Finding 4: PROBE_DEEPER must not block project advancement once quota is met."""
    candidate = {
        "projects": ["AlphaProject", "BetaProject"],
        "skills": ["Object-Oriented Programming", "Data Structures and Algorithms", "Python"],
        "interview_plan": {
            "questions_per_project": 2,
            "questions_per_skill": 2,
            "wrap_up_seconds": 90,
            "duration_seconds": 900,
        },
    }
    fsm = InterviewFSM({
        "stage": "project_deep_dive",
        "questions_asked": 2,
        "questions_remaining": 8,
        "time_remaining_seconds": 700,
        "difficulty_level": 2,
        "current_project_index": 0,
        "project_question_count": 1,
        "current_topic": "Project: AlphaProject",
    })
    # Candidate gives a very weak answer that triggers PROBE_DEEPER
    eval_weak = {"score": 0.2, "depth": "low", "is_skip": False, "missing_concepts": ["design"]}
    state = fsm.update_from_evaluation(eval_weak, "TECHNICAL_ANSWER", candidate)
    
    assert state["action"] == "PROBE_DEEPER"
    # Project 1 had quota 2. After 2 answers (Q1 + Q2), it MUST advance to Project 2 (index 1)
    assert state["stage"] == "project_deep_dive"
    assert state["current_project_index"] == 1
    assert "BetaProject" in state["current_topic"]


def test_topic_change_does_not_hijack_spoken_question():
    """Finding 3: Explicit topic request in structured stages falls through to planned agenda."""
    from backend.question_engine import question_engine
    from backend.evaluator import evaluate_turn_answer
    from backend.intent_engine import detect_candidate_intent

    candidate = {
        "name": "Jane",
        "projects": ["DistributedDB"],
        "skills": ["Object-Oriented Programming", "Data Structures and Algorithms", "Java", "Python"],
        "interview_plan": {
            "questions_per_project": 4,
            "questions_per_skill": 2,
            "duration_seconds": 900,
        },
    }
    answer = "I don't know that. Can we switch to Python?"
    intent = detect_candidate_intent(answer)
    eval_res = evaluate_turn_answer(answer, {"question": "How did you handle consistency?"}, intent)
    
    fsm_state = {
        "stage": "project_deep_dive",
        "difficulty_level": 2,
        "current_project_index": 0,
        "project_question_count": 2,
        "questions_already_asked": ["Tell me about DistributedDB."],
        "action": "PIVOT_TOPIC",
        "current_topic": "Project: DistributedDB",
    }
    
    decision = question_engine.decide(
        candidate_answer=answer,
        intent=intent,
        fsm_state=fsm_state,
        eval_res=eval_res,
        candidate_dict=candidate,
    )
    spoken = (decision.spoken_question or "").lower()
    assert "let's switch to python" not in spoken
    assert "sure, let's switch" not in spoken
    assert "distributeddb" in spoken


def test_clock_zero_cuts_to_foundation_not_closing():
    """Finding 2: 15-min contract ensures OOP/DSA are reached even if time runs out on projects."""
    candidate = {
        "projects": ["OldProject"],
        "skills": ["Object-Oriented Programming", "Data Structures and Algorithms", "Python"],
        "interview_plan": {
            "questions_per_project": 4,
            "questions_per_skill": 2,
            "wrap_up_seconds": 60,
            "duration_seconds": 900,
        },
    }
    fsm = InterviewFSM({
        "stage": "project_deep_dive",
        "questions_asked": 6,
        "questions_remaining": 6,
        "time_remaining_seconds": 0,  # Clock hit 0 during projects
        "difficulty_level": 2,
        "current_project_index": 0,
        "project_question_count": 2,
        "current_topic": "Project: OldProject",
    })
    state = fsm.update_from_evaluation(
        {"score": 0.8, "is_skip": False, "missing_concepts": []},
        "TECHNICAL_ANSWER",
        candidate,
    )
    # Must cut to skills_assessment (OOPs / DSA) rather than closing immediately
    assert state["stage"] == "skills_assessment"
    assert "Object-Oriented" in state["current_topic"]


def test_zero_projects_skips_project_stage():
    """Finding 6: Empty projects list must not produce phantom 'a project on your resume'."""
    candidate = {
        "name": "Sam",
        "projects": [],
        "skills": ["Object-Oriented Programming", "Data Structures and Algorithms", "Python"],
        "interview_plan": {
            "questions_per_project": 4,
            "questions_per_skill": 2,
            "duration_seconds": 900,
        },
    }
    fsm = InterviewFSM({
        "stage": "warmup",
        "questions_asked": 1,
        "questions_remaining": 6,
        "time_remaining_seconds": 850,
        "difficulty_level": 2,
        "current_topic": "Introduction",
    })
    state = fsm.update_from_evaluation(
        {"score": 0.8, "is_skip": False, "missing_concepts": []},
        "SELF_INTRO",
        candidate,
    )
    # Must jump directly to skills_assessment
    assert state["stage"] == "skills_assessment"
    assert "a project on your resume" not in (state.get("current_topic") or "")
    
    # Check greeting does not mention "questions on each of"
    greeting = opening_greeting("Sam", "Software Engineer", 900, [], candidate["skills"])
    assert "a project on your resume" not in greeting
    assert "questions on each of" not in greeting


def test_skill_filter_word_boundary():
    """Finding 8: Word boundary matching does not filter legitimate compound skills."""
    from interviewer.services.resume import extract_skills
    text = """
    TECHNICAL SKILLS
    - iOS Development, Web Development
    - User Experience Design
    - HttpClient, Httpx, FastAPI
    """
    skills = extract_skills(text)
    lower_skills = [s.lower() for s in skills]
    assert any("development" in s for s in lower_skills)
    assert any("experience" in s for s in lower_skills)

"""15-minute project-then-skills structure and clean close."""

from backend.interview_fsm import InterviewFSM
from backend.interview_plan import build_interview_plan
from interviewer.services.interview_structure import planned_question_count


def test_plan_is_fifteen_minutes():
    plan = build_interview_plan(
        {
            "skills": ["Python", "PyTorch"],
            "projects": ["MoleCheck", "AlgoSync"],
            "bank_track": "aiml",
            "target_track": "aiml",
        }
    )
    assert plan["duration_seconds"] == 900
    assert plan["questions_per_project"] == 4
    assert plan["questions_per_skill"] == 3
    assert plan["structure"].startswith("warmup")
    assert "Object-Oriented Programming" in plan["skill_names"]
    assert plan["total_questions"] == planned_question_count(2, len(plan["skill_names"]))
    print("[ok] 15-min plan", plan["total_questions"], "questions,", plan["duration_seconds"], "seconds")


def test_projects_then_skills_then_close():
    plan = build_interview_plan(
        {"projects": ["Ferrite Mesh", "Orchestra"], "skills": ["Rust", "Go"], "bank_track": "backend"}
    )
    candidate = {
        "projects": ["Ferrite Mesh", "Orchestra"],
        "skills": plan["skill_names"],
        "interview_plan": plan,
    }
    fsm = InterviewFSM({
        "stage": "warmup",
        "questions_asked": 0,
        "questions_remaining": candidate["interview_plan"]["total_questions"],
        "time_remaining_seconds": 900,
        "difficulty_level": 2,
    })

    stages = []
    eval_ok = {"score": 0.8, "depth": "high", "is_skip": False, "missing_concepts": []}
    for _ in range(40):
        state = fsm.update_from_evaluation(eval_ok, "TECHNICAL_ANSWER", candidate)
        stages.append(state["stage"])
        if state["stage"] == "completed":
            break

    assert "project_deep_dive" in stages
    assert "skills_assessment" in stages
    assert stages[-1] == "completed"
    first_skill = next(i for i, s in enumerate(stages) if s == "skills_assessment")
    last_project = max(i for i, s in enumerate(stages) if s == "project_deep_dive")
    assert last_project < first_skill, "skills must not start before projects finish"
    print("[ok] stage order", stages)


def test_time_up_closes_instead_of_running_forever():
    candidate = {
        "projects": ["MoleCheck"],
        "skills": ["Python", "PyTorch", "SQL"],
        "interview_plan": {
            "questions_per_project": 2,
            "questions_per_skill": 1,
            "wrap_up_seconds": 90,
            "duration_seconds": 900,
        },
    }
    fsm = InterviewFSM({
        "stage": "project_deep_dive",
        "questions_asked": 3,
        "questions_remaining": 6,
        "time_remaining_seconds": 40,
        "difficulty_level": 2,
        "current_project_index": 0,
        "project_question_count": 1,
    })
    state = fsm.update_from_evaluation(
        {"score": 0.8, "is_skip": False, "missing_concepts": []},
        "TECHNICAL_ANSWER",
        candidate,
    )
    assert state["stage"] == "closing"
    assert state.get("action") != "END_INTERVIEW"
    finished = fsm.update_from_evaluation(
        {"score": 0.5, "is_skip": False, "missing_concepts": []},
        "TECHNICAL_ANSWER",
        candidate,
    )
    assert finished["stage"] == "completed"
    assert finished["action"] == "END_INTERVIEW"
    print("[ok] low time wraps to closing, then completes after candidate Q&A")


def test_agenda_asks_oop_then_dsa():
    plan = build_interview_plan(
        {
            "skills": ["Python", "TensorFlow"],
            "projects": ["MoleCheck", "Mental Health Predictor"],
            "bank_track": "aiml",
            "target_track": "aiml",
        }
    )
    candidate = {
        "projects": ["MoleCheck", "Mental Health Predictor"],
        "skills": plan["skill_names"],
        "interview_plan": plan,
    }
    fsm = InterviewFSM({
        "stage": "warmup",
        "questions_asked": 0,
        "questions_remaining": plan["total_questions"],
        "time_remaining_seconds": 900,
        "difficulty_level": 2,
    })
    topics = []
    eval_ok = {"score": 0.8, "depth": "high", "is_skip": False, "missing_concepts": []}
    for _ in range(50):
        state = fsm.update_from_evaluation(eval_ok, "TECHNICAL_ANSWER", candidate)
        topics.append(state.get("current_topic") or "")
        if state["stage"] == "completed":
            break
    blob = " | ".join(topics)
    oop_at = next(i for i, t in enumerate(topics) if "Object-Oriented" in t)
    dsa_at = next(i for i, t in enumerate(topics) if "Data Structures" in t)
    py_at = next((i for i, t in enumerate(topics) if t.startswith("Skill: Python")), None)
    assert oop_at < dsa_at, blob
    if py_at is not None:
        assert dsa_at < py_at, blob
    print("[ok] agenda asks OOP then DSA before CV skills")


def test_skip_does_not_leave_foundation_skill():
    plan = build_interview_plan(
        {"projects": ["MoleCheck"], "skills": ["Python"], "bank_track": "aiml"}
    )
    candidate = {
        "projects": ["MoleCheck"],
        "skills": plan["skill_names"],
        "interview_plan": plan,
    }
    fsm = InterviewFSM({
        "stage": "skills_assessment",
        "questions_asked": 8,
        "questions_remaining": 12,
        "time_remaining_seconds": 400,
        "difficulty_level": 2,
        "current_skill_index": 0,
        "skill_question_count": 0,
        "current_topic": "Skill: Object-Oriented Programming",
    })
    state = fsm.update_from_evaluation(
        {"score": 0.0, "is_skip": True, "missing_concepts": []},
        "UNKNOWN_OR_SKIP",
        candidate,
    )
    assert state["stage"] == "skills_assessment"
    assert state["current_skill_index"] == 0
    assert "Object-Oriented" in (state.get("current_topic") or "")
    assert state["skill_question_count"] == 1
    print("[ok] skip stays on OOP instead of jumping to DSA")


def test_low_time_still_asks_dsa():
    plan = build_interview_plan(
        {"projects": ["MoleCheck"], "skills": ["Python"], "bank_track": "aiml"}
    )
    candidate = {
        "projects": ["MoleCheck"],
        "skills": plan["skill_names"],
        "interview_plan": plan,
    }
    fsm = InterviewFSM({
        "stage": "skills_assessment",
        "questions_asked": 12,
        "questions_remaining": 0,
        "time_remaining_seconds": 40,
        "difficulty_level": 2,
        "current_skill_index": 0,
        "skill_question_count": 2,
        "current_topic": "Skill: Object-Oriented Programming",
    })
    state = fsm.update_from_evaluation(
        {"score": 0.8, "is_skip": False, "missing_concepts": []},
        "TECHNICAL_ANSWER",
        candidate,
    )
    assert state["stage"] == "skills_assessment"
    assert "Data Structures" in (state.get("current_topic") or "")
    print("[ok] low remaining time still moves OOP -> DSA")


def test_clock_zero_during_projects_still_asks_oop():
    plan = build_interview_plan(
        {"projects": ["MoleCheck", "Mental Health Predictor"], "skills": ["Python"], "bank_track": "aiml"}
    )
    candidate = {
        "projects": ["MoleCheck", "Mental Health Predictor"],
        "skills": plan["skill_names"],
        "interview_plan": plan,
    }
    fsm = InterviewFSM({
        "stage": "project_deep_dive",
        "questions_asked": 4,
        "questions_remaining": 10,
        "time_remaining_seconds": 0,
        "difficulty_level": 2,
        "current_project_index": 0,
        "project_question_count": 2,
        "current_topic": "Project: MoleCheck",
    })
    state = fsm.update_from_evaluation(
        {"score": 0.8, "is_skip": False, "missing_concepts": []},
        "TECHNICAL_ANSWER",
        candidate,
    )
    assert state["stage"] == "skills_assessment"
    assert "Object-Oriented" in (state.get("current_topic") or "")
    print("[ok] clock 0 during projects cuts to OOP instead of closing")


def test_probe_deeper_does_not_extend_project_quota():
    plan = build_interview_plan(
        {"projects": ["MoleCheck", "Mental Health Predictor"], "skills": ["Python"], "bank_track": "aiml"}
    )
    candidate = {
        "projects": ["MoleCheck", "Mental Health Predictor"],
        "skills": plan["skill_names"],
        "interview_plan": plan,
    }
    fsm = InterviewFSM({
        "stage": "project_deep_dive",
        "questions_asked": 4,
        "questions_remaining": 12,
        "time_remaining_seconds": 700,
        "difficulty_level": 2,
        "current_project_index": 0,
        "project_question_count": 3,
        "current_topic": "Project: MoleCheck",
    })
    state = fsm.update_from_evaluation(
        {"score": 0.2, "depth": "low", "is_skip": False, "missing_concepts": ["regularization"]},
        "TECHNICAL_ANSWER",
        candidate,
    )
    assert state["action"] == "PROBE_DEEPER"
    assert state["stage"] == "project_deep_dive"
    assert state["current_project_index"] == 1
    assert "Mental Health" in (state.get("current_topic") or "")
    print("[ok] weak last project answer still advances after the quota")


if __name__ == "__main__":
    test_plan_is_fifteen_minutes()
    test_projects_then_skills_then_close()
    test_time_up_closes_instead_of_running_forever()
    test_agenda_asks_oop_then_dsa()
    test_skip_does_not_leave_foundation_skill()
    test_low_time_still_asks_dsa()
    test_clock_zero_during_projects_still_asks_oop()
    test_probe_deeper_does_not_extend_project_quota()
    print("[SUCCESS] Interview structure checks passed.")

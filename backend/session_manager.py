"""
Session Manager and Store for Interview Sessions.
"""

import time
import uuid
from typing import Dict, Any
from backend.role_sift import sift_skills_for_role
from backend.redis_client import session_store
from backend.interview_plan import build_interview_plan

# Retain dict proxy reference for tests / modules that inspect INTERVIEW_SESSIONS directly
INTERVIEW_SESSIONS: Dict[str, Dict[str, Any]] = session_store._memory_sessions


def get_latest_cv_features() -> Dict[str, Any]:
    return session_store.get_latest_cv_features()


def set_latest_cv_features(features: Dict[str, Any]):
    session_store.save_latest_cv_features(features)


def save_session(session: Dict[str, Any]):
    sid = session.get("session_id")
    if sid:
        session_store.save_session(sid, session)


def create_session(cv_features: Dict[str, Any] = None, role_override: str = None, job_description: str = None) -> Dict[str, Any]:
    """Test helper. Live interviews use app._make_session so quotas and RAG match production."""
    sid = str(uuid.uuid4())[:8]

    if not cv_features:
        cv_features = {}

    sift = sift_skills_for_role(cv_features or {}, role_override=role_override, job_description=job_description)
    c_name = cv_features.get("name", "Candidate")
    c_skills = list(sift.get("intersection") or [])
    c_projects = list(sift.get("interview_projects") or cv_features.get("projects") or [])
    c_degree = cv_features.get("degree", "B.Tech Computer Science")
    c_college = cv_features.get("college", "University")
    target_role = sift["label"]

    try:
        from rag_engine import question_bank_rag
        if question_bank_rag and not question_bank_rag.is_ready:
            question_bank_rag.load()
        interview_plan = build_interview_plan(
            {
                "name": c_name,
                "skills": c_skills,
                "projects": c_projects,
                "resume_text": cv_features.get("raw_text") or cv_features.get("resume_context") or "",
                "bank_track": sift["bank_track"],
                "target_track": sift["track"],
                "interview_style": sift["style"],
            },
            question_bank_rag=question_bank_rag,
        )
    except Exception:
        interview_plan = {
            "track": sift["track"],
            "opener_id": None,
            "project_names": c_projects,
            "skill_names": c_skills,
            "skill_slots": [],
            "duration_seconds": 900,
            "total_questions": 8,
        }

    c_skills = list(interview_plan.get("skill_names") or c_skills)

    est_total_questions = interview_plan.get("total_questions") or 8

    now = time.time()
    session = {
        "session_id": sid,
        "created_at": now,
        "started_at": now,
        "ended_at": None,
        "duration_seconds": None,
        "last_turn_time": now,
        "candidate": {
            "name": c_name,
            "skills": c_skills,
            "projects": c_projects,
            "degree": c_degree,
            "college": c_college,
            "resume_context": cv_features.get("raw_text", "")[:500],
            "target_role": target_role,
            "target_track": sift["track"],
            "bank_track": sift["bank_track"],
            "interview_style": sift["style"],
            "skill_sift": {
                "intersection": sift["intersection"],
                "role_gaps": sift["role_gaps"],
                "cv_only": sift["cv_only"],
            },
            "interview_plan": interview_plan,
        },
        "interview_plan": interview_plan,
        "job_requirements": {
            "title": target_role,
            "track": sift["track"],
            "required_skills": sift["required_skills"],
            "interview_skills": c_skills,
            "intersection": sift["intersection"],
            "role_gaps": sift["role_gaps"],
            "difficulty_bias": "mid"
        },
        "interview_state": {
            "stage": "warmup",
            "difficulty_level": 2,  # Bounded strictly within [1, 3]
            "current_topic": "Introduction",
            "questions_asked": 0,
            "questions_remaining": est_total_questions,
            "time_remaining_seconds": interview_plan.get("duration_seconds") or 900,
            "topics_covered": [],
            "questions_already_asked": [],
            "candidate_weaknesses": [],
            "topic_scores": {},
            "current_project_index": 0,
            "project_question_count": 0,
            "current_skill_index": 0,
            "skill_question_count": 0,
            "project_thread": {},
        },
        "current_question": {
            "question": (
                f"Hi {c_name}, I'm your interviewer. Please introduce yourself "
                f"and the work you're proudest of."
            ),
            "skill": "Introduction",
            "difficulty": 2,
            "expected_concepts": ["Name", "Background", "Primary Tech Stack"],
            "evaluation_rubric": []
        },
        "history": []
    }

    session_store.save_session(sid, session)
    return session


def get_session(session_id: str) -> Dict[str, Any]:
    return session_store.get_session(session_id)

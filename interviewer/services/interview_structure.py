"""
15-minute interview structure.

  Opening        1 spoken turn   intro, then a bridge into project 1
  Projects       4 questions each, up to 2 CV projects
  Fundamentals   OOPs + DSA — 3 questions each, locked at resume upload
  Skills         3 questions each, locked at resume upload from bank + CV terms
  Closing        1 wrap-up turn

The clock may drop leftover projects or extra CV skills. OOP and DSA still run.
"""

from __future__ import annotations

from typing import Any, Mapping

from interviewer.config import settings

DURATION_SECONDS = settings.interview.duration_seconds
WRAP_UP_SECONDS = settings.interview.wrap_up_seconds
QUESTIONS_PER_PROJECT = settings.interview.project_questions
QUESTIONS_PER_SKILL = settings.interview.skill_questions
MAX_PROJECTS = settings.interview.max_projects
MAX_SKILLS = settings.interview.max_skills


def planned_question_count(project_count: int, skill_count: int) -> int:
    """Opening is counted inside the first project's quota. Plus one closing turn."""
    projects = min(MAX_PROJECTS, max(0, project_count))
    skills = max(0, min(MAX_SKILLS + 2, skill_count))
    per_skill = max(3, QUESTIONS_PER_SKILL)
    return projects * QUESTIONS_PER_PROJECT + skills * per_skill + 1


def opening_greeting(
    name: str,
    role_label: str,
    duration_seconds: int,
    projects: list[str] | None,
    skills: list[str] | None,
) -> str:
    mins = max(1, int(round((duration_seconds or DURATION_SECONDS) / 60)))
    who = name if name and name.lower() not in ("", "null", "the candidate") else "there"
    proj_list = [p for p in (projects or []) if p and str(p).strip()]
    proj = ", ".join(proj_list[:2])
    foundation = {
        "object-oriented programming",
        "data structures and algorithms",
        "oop",
        "oops",
        "dsa",
        "data structures",
        "algorithms",
    }
    cv_skills = [
        s for s in (skills or [])
        if str(s).strip().lower() not in foundation and "object-oriented" not in str(s).lower()
    ]
    skill = ", ".join(cv_skills[:2]) or "the skills on your resume"
    if proj:
        agenda = f"{QUESTIONS_PER_PROJECT} questions on each of {proj}, then OOPs and DSA, and finish with {skill}"
    else:
        agenda = f"OOPs and DSA, and finish with {skill}"
    return (
        f"Hi {who}, I'm your interviewer for this {mins}-minute {role_label} conversation. "
        f"We'll start with a short introduction, then {agenda}. "
        f"Please introduce yourself and the work you're proudest of."
    )


def quotas_from_plan(plan: Mapping[str, Any] | None) -> dict[str, int]:
    plan = plan or {}
    return {
        "questions_per_project": int(plan.get("questions_per_project") or QUESTIONS_PER_PROJECT),
        "questions_per_skill": int(plan.get("questions_per_skill") or QUESTIONS_PER_SKILL),
        "duration_seconds": int(plan.get("duration_seconds") or DURATION_SECONDS),
        "wrap_up_seconds": int(plan.get("wrap_up_seconds") or WRAP_UP_SECONDS),
    }

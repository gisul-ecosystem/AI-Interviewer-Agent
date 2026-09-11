"""
Job-role skill sifting for industry interviews.

Interview topics come from three sets:
  1. Intersection — skills on the CV that the role also requires (primary).
  2. Role gaps     — role-required skills the CV does not mention (probe lightly).
  3. CV-only       — extra CV skills that are still in-family for the role (optional).

No vector DB is required: profiles are small, in-memory, and per session.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from interviewer.services.resume import INVALID_FIELD_WORDS, is_resume_metadata_title, looks_like_job_line


ROLE_ALIASES = {
    "auto": "auto",
    "aiml": "aiml",
    "ai": "aiml",
    "ml": "aiml",
    "ai/ml": "aiml",
    "ai-ml": "aiml",
    "machine learning": "aiml",
    "webd": "webd",
    "web": "webd",
    "web dev": "webd",
    "web development": "webd",
    "fullstack": "webd",
    "full-stack": "webd",
    "full stack": "webd",
    "frontend": "frontend",
    "front-end": "frontend",
    "backend": "backend",
    "back-end": "backend",
    "hr": "hr",
    "human resources": "hr",
    "people": "hr",
    "ai / ml engineer": "aiml",
    "web developer": "webd",
    "web developer (full stack)": "webd",
    "frontend engineer": "frontend",
    "backend engineer": "backend",
    "hr / people operations": "hr",
}

SYNONYMS = {
    "pytorch": "pytorch",
    "py torch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "deep learning": "deep learning",
    "dl": "deep learning",
    "neural networks": "deep learning",
    "llm": "llms",
    "llms": "llms",
    "large language models": "llms",
    "nlp": "nlp",
    "rag": "rag",
    "langchain": "rag",
    "python": "python",
    "rust": "rust",
    "go": "go",
    "golang": "go",
    "java": "java",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "terraform": "terraform",
    "grpc": "grpc",
    "langgraph": "langgraph",
    "vllm": "vllm",
    "duckdb": "duckdb",
    "react": "react",
    "react.js": "react",
    "reactjs": "react",
    "node": "node.js",
    "node.js": "node.js",
    "nodejs": "node.js",
    "express": "node.js",
    "express.js": "node.js",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "html": "html/css",
    "css": "html/css",
    "html/css": "html/css",
    "mongodb": "mongodb",
    "mongo": "mongodb",
    "sql": "sql",
    "mysql": "sql",
    "postgresql": "sql",
    "postgres": "sql",
    "rest": "rest api",
    "rest api": "rest api",
    "fastapi": "backend apis",
    "django": "backend apis",
    "flask": "backend apis",
    "docker": "docker",
    "git": "git",
    "dsa": "dsa",
    "data structures": "dsa",
    "algorithms": "dsa",
    "communication": "communication",
    "conflict": "conflict resolution",
    "conflict resolution": "conflict resolution",
    "hiring": "hiring",
    "recruitment": "hiring",
    "stakeholder": "stakeholder management",
    "stakeholder management": "stakeholder management",
}

ROLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "aiml": {
        "label": "AI / ML Engineer",
        "style": "technical",
        "bank_track": "aiml",
        "required": [
            "python",
            "machine learning",
            "deep learning",
            "pytorch",
            "llms",
            "nlp",
            "rag",
            "sql",
        ],
    },
    "webd": {
        "label": "Web Developer",
        "style": "technical",
        "bank_track": "webd",
        "required": [
            "javascript",
            "react",
            "node.js",
            "html/css",
            "sql",
            "mongodb",
            "rest api",
            "git",
        ],
    },
    "frontend": {
        "label": "Frontend Engineer",
        "style": "technical",
        "bank_track": "frontend",
        "required": ["javascript", "typescript", "react", "html/css", "git"],
    },
    "backend": {
        "label": "Backend Engineer",
        "style": "technical",
        "bank_track": "backend",
        "required": [
            "python",
            "go",
            "rust",
            "java",
            "node.js",
            "sql",
            "rest api",
            "grpc",
            "docker",
            "kubernetes",
            "dsa",
        ],
    },
    "hr": {
        "label": "HR / People Operations",
        "style": "behavioral",
        "bank_track": "hr",
        "required": [
            "communication",
            "hiring",
            "conflict resolution",
            "stakeholder management",
        ],
    },
    "resume": {
        "label": "Technical interview (from your resume)",
        "style": "technical",
        "bank_track": None,
        "required": [],
    },
}


def normalize_track(raw: str | None) -> str:
    key = (raw or "auto").strip().lower()
    if key in ROLE_ALIASES:
        return ROLE_ALIASES[key]
    for alias, track in sorted(ROLE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias == "auto" or len(alias) < 2:
            continue
        if alias in key:
            return track
    return "auto"


def canonicalize_skill(raw: str) -> str:
    text = " ".join(str(raw or "").strip().lower().replace("_", " ").split())
    return SYNONYMS.get(text, text)


def _as_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _detect_track_from_cv(skills: Iterable[str], domains: Iterable[str], cv_role: str) -> str:
    blob = " ".join([*skills, *domains, cv_role]).lower()
    if any(term in blob for term in ("hr", "recruit", "people operations", "talent")):
        return "hr"
    if any(term in blob for term in ("pytorch", "tensorflow", "machine learning", "llm", "nlp", "deep learning")):
        return "aiml"
    if any(term in blob for term in ("rust", "golang", "kubernetes", "terraform", "grpc")):
        return "backend"
    if any(term in blob for term in ("react", "javascript", "frontend", "html")):
        return "webd"
    if any(term in blob for term in ("node", "backend", "fastapi", "django", "flask")):
        return "backend"
    return "resume"


def clean_interview_projects(raw_projects: Iterable[str] | None) -> List[str]:
    """Drop resume section headings, fragments, and metric phrases that leaked into the project list."""
    cleaned: List[str] = []
    seen: set[str] = set()
    for item in raw_projects or []:
        title = str(item or "").strip()
        if not title or is_resume_metadata_title(title) or looks_like_job_line(title):
            continue
        low = title.lower()
        if low in INVALID_FIELD_WORDS:
            continue
        first_word = low.split()[0] if low.split() else ""
        if first_word in ("and", "or", "the", "a", "an", "with", "in", "for", "using", "by"):
            continue
        if any(frag in low for frag in ("confusion matrix", "roc curve", "precision recall", "sensitivity specificity")):
            continue
        key = low
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(title)
    return cleaned[:3]


def extract_skills_from_jd(job_description: str) -> List[str]:
    """Pull canonical skills mentioned in a pasted job description."""
    if not job_description or not str(job_description).strip():
        return []
    blob = " ".join(str(job_description).lower().split())
    found: List[str] = []
    seen = set()
    for key in sorted(SYNONYMS.keys(), key=len, reverse=True):
        if len(key) < 2:
            continue
        if key in blob:
            canon = SYNONYMS[key]
            if canon not in seen:
                seen.add(canon)
                found.append(canon)
    return found


def sift_skills_for_role(
    cv_features: Dict[str, Any],
    role_override: str | None = None,
    job_description: str | None = None,
) -> Dict[str, Any]:
    """
    Role profile plus the interview agenda.

    Projects are mandatory and come only from the CV. Skills asked live are
    strictly CV ∩ role (or CV ∩ pasted JD). Role gaps are reported, never
    promoted into the question plan.
    """
    cv_skills_raw = _as_list(cv_features.get("skills")) + _as_list(cv_features.get("domains"))
    projects = clean_interview_projects(_as_list(cv_features.get("projects")))
    cv_role = str(cv_features.get("role") or cv_features.get("target_role") or "")
    jd_text = (job_description or "").strip()
    jd_skills = extract_skills_from_jd(jd_text)

    requested = normalize_track(role_override)
    detect_blob_skills = cv_skills_raw + jd_skills
    if requested != "auto":
        track = requested
        track_source = "role"
    elif jd_text:
        track = _detect_track_from_cv(detect_blob_skills, [], f"{cv_role} {jd_text[:400]}")
        track_source = "jd"
    else:
        track = _detect_track_from_cv(detect_blob_skills, [], cv_role)
        track_source = "cv" if track != "resume" else "resume"
    profile = ROLE_PROFILES.get(track, ROLE_PROFILES["resume"])

    cv_canon = []
    seen = set()
    display_by_canon: Dict[str, str] = {}
    for skill in cv_skills_raw:
        canon = canonicalize_skill(skill)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        cv_canon.append(canon)
        display_by_canon[canon] = skill

    if jd_skills:
        required = list(dict.fromkeys(jd_skills))
        for skill in profile["required"]:
            if skill not in required:
                required.append(skill)
    else:
        required = list(profile["required"])

    # No JD and no explicit role: interview the CV, don't inject a guessed role's gaps.
    resume_grounded = track_source in ("resume",) or (track_source == "cv" and requested == "auto" and not jd_text and not profile["required"])

    intersection = [skill for skill in required if skill in seen] if required else list(cv_canon)
    if resume_grounded or not required:
        intersection = list(cv_canon)
        role_gaps = []
        cv_only = []
        interview_skills = [display_by_canon.get(skill, skill.title()) for skill in intersection[:5]]
        required_display = interview_skills
    else:
        role_gaps = [skill for skill in required if skill not in seen]
        cv_only = [skill for skill in cv_canon if skill not in required]
        interview_skills = [display_by_canon.get(skill, skill.title()) for skill in intersection[:5]]
        required_display = [display_by_canon.get(skill, skill.title()) for skill in required]

    return {
        "track": track,
        "track_source": track_source,
        "label": profile["label"],
        "style": profile["style"],
        "bank_track": profile["bank_track"],
        "required_skills": required_display,
        "intersection": [display_by_canon.get(skill, skill.title()) for skill in intersection],
        "role_gaps": [skill.title() for skill in role_gaps],
        "cv_only": [display_by_canon.get(skill, skill.title()) for skill in cv_only],
        "interview_skills": interview_skills,
        "interview_projects": projects[:3],
        "job_description": jd_text[:4000],
    }

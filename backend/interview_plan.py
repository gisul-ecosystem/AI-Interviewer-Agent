"""
Deterministic interview plan built once at resume upload / session start.

Live turns only advance this plan. Every skill question — base and follow-ups —
is locked here from the question bank plus technical terms on the CV.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence

from interviewer.services.interview_structure import (
    DURATION_SECONDS,
    MAX_PROJECTS,
    MAX_SKILLS,
    QUESTIONS_PER_PROJECT,
    QUESTIONS_PER_SKILL,
    WRAP_UP_SECONDS,
    planned_question_count,
)
from interviewer.services.resume import TECH_VOCAB


OPENER_BY_TRACK = {
    "aiml": "aiml-project-ownership-01",
    "webd": "webd-project-ownership-01",
    "frontend": "webd-project-ownership-01",
    "backend": "aiml-project-ownership-01",
    "hr": "hr-intro-01",
    "resume": "aiml-project-ownership-01",
    "general_cs": "aiml-project-ownership-01",
    "software_engineering": "aiml-project-ownership-01",
}

SKIP_COMPETENCIES = frozenset({
    "project_ownership",
    "communication",
    "technical_communication",
    "hiring",
    "conflict_resolution",
})

FOUNDATION_SLOTS = (
    {
        "skill": "Object-Oriented Programming",
        "kind": "foundation",
        "aliases": ("oop", "oops", "object-oriented programming", "object oriented programming"),
        "preferred_ids": ("cs-oop-polymorphism-01", "cs-oop-composition-02", "cs-oop-solid-03"),
        "queries": (
            "object oriented polymorphism encapsulation inheritance",
            "composition vs inheritance solid",
            "liskov open closed principle",
        ),
        "fallbacks": (
            "Can you explain the difference between compile-time and runtime polymorphism with a concrete example?",
            "Why is composition over inheritance widely recommended, and when is inheritance a poor choice?",
            "How does the Liskov Substitution Principle stop a subclass from breaking callers of the base type?",
        ),
    },
    {
        "skill": "Data Structures and Algorithms",
        "kind": "foundation",
        "aliases": ("dsa", "data structures", "data structures and algorithms", "algorithms"),
        "preferred_ids": ("cs-dsa-hash-table-01", "cs-dsa-lru-cache-02", "cs-dsa-two-pointers-03"),
        "queries": (
            "hash table collision load factor",
            "lru cache hashmap doubly linked list",
            "linked list cycle two pointers",
        ),
        "fallbacks": (
            "Why is hash table lookup O(1) on average, and what makes it degrade to O(N)?",
            "How would you design an LRU cache so get and put are both O(1)?",
            "How would you detect a cycle in a linked list in linear time and constant extra memory?",
        ),
    },
)

_CV_TERM_RE = re.compile(
    r"\b(?:MobileNetV2|MobileNet|ResNet50|ResNet|EfficientNet|YOLO|ISIC|"
    r"class[- ]weights?|data augmentation|transfer learning|confusion matrix|"
    r"sensitivity|specificity|precision[- ]recall|false negatives?|"
    r"Keras|TensorFlow|PyTorch|NumPy|Pandas|OpenCV|scikit[- ]learn|"
    r"GIL|asyncio|multiprocessing|generator|decorator|type hints?|"
    r"gRPC|mTLS|LangGraph|vLLM|DuckDB|FastAPI|Docker|Kubernetes|"
    r"hash table|linked list|binary search|heap|LRU|"
    r"polymorphism|inheritance|composition|encapsulation)\b",
    re.IGNORECASE,
)

_SKILL_QUERY_HINTS = {
    "python": ("python GIL multiprocessing", "python generators yield", "numpy pandas pipeline"),
    "tensorflow": ("transfer learning CNN freeze layers", "keras class imbalance", "model evaluation threshold"),
    "keras": ("keras transfer learning", "class weights augmentation", "confusion matrix sensitivity"),
    "pytorch": ("pytorch dataloader autograd", "cnn transfer learning", "training loop overfitting"),
    "java": ("java hashmap equals hashcode", "java memory garbage collection", "java concurrency synchronized"),
    "sql": ("sql indexes b-tree", "sql acid isolation", "query plan"),
    "javascript": ("javascript event loop", "closures this", "promises async"),
    "react": ("react state rendering", "hooks reconciliation", "controlled components"),
    "docker": ("docker layers image", "container networking", "multi-stage build"),
}

_SKILL_COMPETENCIES = {
    "python": {"python"},
    "tensorflow": {"deep_learning", "model_training", "model_evaluation", "ml_deployment", "data_validation"},
    "keras": {"deep_learning", "model_training", "model_evaluation", "data_validation"},
    "pytorch": {"deep_learning", "model_training", "model_evaluation", "ml_deployment"},
    "java": {"java"},
    "sql": {"database_systems"},
    "javascript": {"frontend_state"},
    "react": {"frontend_state"},
    "docker": {"ml_deployment", "reliability"},
}

_SKILL_TERM_ALLOW = {
    "python": ("gil", "asyncio", "multiprocessing", "generator", "numpy", "pandas", "decorator", "typing"),
    "tensorflow": ("keras", "mobilenet", "resnet", "efficientnet", "transfer learning", "class weight", "augmentation", "cnn"),
    "keras": ("mobilenet", "resnet", "transfer learning", "class weight", "augmentation", "callback"),
    "pytorch": ("dataloader", "autograd", "cuda", "mobilenet", "resnet", "transfer learning"),
    "java": ("hashmap", "garbage", "thread", "synchronized", "jvm"),
    "sql": ("index", "acid", "isolation", "join", "query"),
}


def _record_id(record: Optional[Dict[str, Any]]) -> Optional[str]:
    if not record:
        return None
    return record.get("id")


def _record_text(record: Optional[Dict[str, Any]]) -> str:
    if not record:
        return ""
    return str(record.get("question") or "").strip()


def _usable_skill_record(record: Optional[Dict[str, Any]], skill: str = "", kind: str = "") -> bool:
    if not record or not _record_text(record):
        return False
    competency = str(record.get("competency") or "").strip().lower()
    topic = str(record.get("topic") or "").strip().lower()
    question = _record_text(record).lower()
    if competency in SKIP_COMPETENCIES:
        return False
    if "project" in topic or competency == "project_ownership":
        return False
    skill_l = str(skill or "").strip().lower()
    if kind == "foundation":
        if "object" in skill_l:
            return "object" in competency or "oop" in competency
        if "data" in skill_l or "algorithm" in skill_l or skill_l == "dsa":
            return "data_structure" in competency or "algorithm" in competency
        return True
    if not skill_l:
        return True
    allowed = _SKILL_COMPETENCIES.get(skill_l)
    if allowed:
        return competency in allowed or skill_l in competency or skill_l in question
    tokens = {t for t in re.findall(r"[a-z0-9]+", skill_l) if len(t) > 2}
    blob = f"{competency} {topic} {question}"
    return any(token in blob for token in tokens)


def _norm_question(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _stable_pick(items: Sequence[Any], seed: str) -> List[Any]:
    """Deterministic order per CV so the same resume is stable, different CVs differ."""
    if not items:
        return []
    digest = hashlib.sha256((seed or "cv").encode("utf-8", errors="replace")).digest()

    def key(item: Any) -> bytes:
        raw = item if isinstance(item, str) else str(item)
        return hashlib.sha256(digest + raw.encode("utf-8", errors="replace")).digest()

    return sorted(items, key=key)


def _cv_terms(resume_text: str, skill: str) -> List[str]:
    blob = resume_text or ""
    found: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        clean = re.sub(r"\s+", " ", str(term or "")).strip()
        key = clean.lower()
        if not clean or key in seen or key == str(skill).lower():
            return
        seen.add(key)
        found.append(clean)

    skill_l = str(skill or "").lower()
    for match in _CV_TERM_RE.findall(blob):
        add(match)
    for tech in TECH_VOCAB:
        if tech.lower() == skill_l:
            continue
        if re.search(rf"\b{re.escape(tech)}\b", blob, flags=re.IGNORECASE):
            add(tech)
    allow = _SKILL_TERM_ALLOW.get(skill_l)
    if allow:
        found = [term for term in found if any(key in term.lower() for key in allow)]
    elif "object" in skill_l:
        found = [term for term in found if any(k in term.lower() for k in ("polymorphism", "inheritance", "composition", "encapsulation"))]
    elif "data structure" in skill_l or skill_l in {"dsa", "algorithms"}:
        found = [term for term in found if any(k in term.lower() for k in ("hash", "linked list", "lru", "heap", "binary search"))]
    return found[:10]


def _skill_queries(skill: str, terms: Sequence[str]) -> List[str]:
    queries = [str(skill)]
    key = str(skill or "").strip().lower()
    queries.extend(_SKILL_QUERY_HINTS.get(key, ()))
    if terms:
        queries.append(f"{skill} " + " ".join(list(terms)[:3]))
    # de-dupe, keep order
    out, seen = [], set()
    for q in queries:
        n = q.strip().lower()
        if n and n not in seen:
            seen.add(n)
            out.append(q.strip())
    return out


def _grounded_fallbacks(skill: str, kind: str, terms: Sequence[str]) -> List[str]:
    skill = str(skill or "this skill").strip()
    a = terms[0] if terms else ""
    b = terms[1] if len(terms) > 1 else ""
    out: list[str] = []
    if kind != "foundation":
        if a and b:
            out.append(
                f"In {skill}, when do you choose {a} over {b}, and what is the failure mode of the wrong pick?"
            )
        if a:
            out.append(
                f"How does {skill} use {a} under the hood, and what breaks if that assumption is wrong?"
            )
            out.append(
                f"How would you debug a silent failure in a {skill} pipeline that involves {a}?"
            )
        out.append(
            f"What {skill} API or data structure did you rely on for correctness, and why not a simpler alternative?"
        )
        out.append(
            f"How do you test a {skill} change so a regression in types, shapes, or latency is caught before production?"
        )
    return out


def _pick_bank_questions(
    question_bank_rag: Any,
    *,
    skill: str,
    kind: str,
    preferred_ids: tuple[str, ...] = (),
    queries: Sequence[str] = (),
    used_ids: set[str],
    used_questions: List[str],
    track: str,
    limit: int,
) -> List[Dict[str, Any]]:
    found: list[Dict[str, Any]] = []
    if not question_bank_rag or limit <= 0:
        return found

    def take(record: Optional[Dict[str, Any]]) -> bool:
        if not _usable_skill_record(record, skill, kind):
            return False
        qid = _record_id(record)
        text = _record_text(record)
        if not text:
            return False
        if qid and qid in used_ids:
            return False
        if _norm_question(text) in {_norm_question(q) for q in used_questions}:
            return False
        found.append(record)
        if qid:
            used_ids.add(qid)
        used_questions.append(text)
        return True

    for qid in preferred_ids:
        if len(found) >= limit:
            break
        if qid in used_ids or not hasattr(question_bank_rag, "get_record"):
            continue
        take(question_bank_rag.get_record(qid))

    if found and hasattr(question_bank_rag, "get_record"):
        try:
            from interviewer.services.question_bank import get_question_bank
            bank = get_question_bank()
            first = bank.get(_record_id(found[0]))
            if first is not None:
                for follow in bank.follow_ups(first, exclude_ids=used_ids):
                    if len(found) >= limit:
                        break
                    take(follow.as_dict())
        except Exception:
            pass

    if hasattr(question_bank_rag, "retrieve_scored_records"):
        bank_filter = None if str(track).lower() in ("resume", "none", "") else track
        for query in queries:
            if len(found) >= limit:
                break
            hits = question_bank_rag.retrieve_scored_records(
                query=query,
                top_k=6,
                exclude_questions=used_questions,
                track=bank_filter,
                target_difficulty=2,
            ) or []
            for hit in hits:
                if len(found) >= limit:
                    break
                take(hit)
    return found


def _slot_questions(
    *,
    skill: str,
    kind: str,
    preferred_ids: tuple[str, ...] = (),
    queries: Sequence[str] = (),
    fallbacks: Sequence[str] = (),
    resume_text: str,
    seed: str,
    used_ids: set[str],
    used_questions: List[str],
    question_bank_rag: Any,
    track: str,
    count: int,
) -> Dict[str, Any]:
    terms = _cv_terms(resume_text, skill)
    query_list = list(queries) + _skill_queries(skill, terms)
    records = _pick_bank_questions(
        question_bank_rag,
        skill=skill,
        kind=kind,
        preferred_ids=preferred_ids,
        queries=query_list,
        used_ids=used_ids,
        used_questions=used_questions,
        track=track,
        limit=count,
    )
    texts = [_record_text(r) for r in records if _record_text(r)]
    ids = [_record_id(r) for r in records if _record_id(r)]

    extras = [t for t in list(fallbacks) + _grounded_fallbacks(skill, kind, terms) if t]
    extras = _stable_pick(extras, f"{seed}|{skill}|{kind}")
    have = {_norm_question(t) for t in texts}
    for extra in extras:
        if len(texts) >= count:
            break
        key = _norm_question(extra)
        if not key or key in have:
            continue
        have.add(key)
        texts.append(extra)
        used_questions.append(extra)

    texts = texts[:count]
    if not texts:
        texts = [f"Explain a concrete {skill} failure you have seen, including the data structure or API involved."]
    return {
        "skill": skill,
        "kind": kind,
        "question_ids": ids,
        "base_question": texts[0],
        "questions": texts,
        "cv_terms": terms[:6],
    }


def build_interview_plan(
    candidate_dict: Dict[str, Any],
    question_bank_rag: Any = None,
    max_projects: int = MAX_PROJECTS,
    max_skills: int = MAX_SKILLS,
) -> Dict[str, Any]:
    """Pick opener, OOPs/DSA, and 3 locked questions per skill before the first spoken turn."""
    projects = list(candidate_dict.get("projects") or [])[:max_projects]
    cv_skills = list(candidate_dict.get("skills") or [])[:max_skills]
    track = (
        candidate_dict.get("bank_track")
        or candidate_dict.get("target_track")
        or "resume"
    )
    if not track or str(track).lower() in ("none", "null"):
        track = "resume"
    style = str(candidate_dict.get("interview_style") or "").lower()
    opener_id = OPENER_BY_TRACK.get(str(track).lower(), OPENER_BY_TRACK["resume"])
    resume_text = str(
        candidate_dict.get("resume_text")
        or candidate_dict.get("raw_text")
        or candidate_dict.get("resume_context")
        or ""
    )
    seed = "|".join(
        [
            str(candidate_dict.get("name") or ""),
            ",".join(projects),
            ",".join(cv_skills),
            resume_text[:400],
        ]
    )

    skill_slots: List[Dict[str, Any]] = []
    used_ids = {opener_id} if opener_id else set()
    used_questions: List[str] = []
    per_skill = max(3, int(QUESTIONS_PER_SKILL or 3))

    if question_bank_rag and hasattr(question_bank_rag, "get_record") and opener_id:
        opener = question_bank_rag.get_record(opener_id)
        if opener and opener.get("question"):
            used_questions.append(opener["question"])

    foundation_names = set()
    if style != "behavioral" and str(track).lower() != "hr":
        for foundation in FOUNDATION_SLOTS:
            foundation_names.add(foundation["skill"].strip().lower())
            foundation_names.update(foundation["aliases"])
            skill_slots.append(
                _slot_questions(
                    skill=foundation["skill"],
                    kind="foundation",
                    preferred_ids=foundation["preferred_ids"],
                    queries=foundation["queries"],
                    fallbacks=foundation["fallbacks"],
                    resume_text=resume_text,
                    seed=seed,
                    used_ids=used_ids,
                    used_questions=used_questions,
                    question_bank_rag=question_bank_rag,
                    track=track,
                    count=per_skill,
                )
            )

    for skill in cv_skills:
        skill_l = str(skill).strip().lower()
        if not skill_l or skill_l in foundation_names:
            continue
        skill_slots.append(
            _slot_questions(
                skill=skill,
                kind="cv",
                queries=_skill_queries(skill, _cv_terms(resume_text, skill)),
                resume_text=resume_text,
                seed=seed,
                used_ids=used_ids,
                used_questions=used_questions,
                question_bank_rag=question_bank_rag,
                track=track,
                count=per_skill,
            )
        )

    skill_names = [slot["skill"] for slot in skill_slots]
    return {
        "track": track,
        "opener_id": opener_id,
        "project_names": projects,
        "skill_names": skill_names,
        "skill_slots": skill_slots,
        "structure": "warmup -> projects -> oops/dsa -> cv skills -> closing",
        "duration_seconds": DURATION_SECONDS,
        "wrap_up_seconds": WRAP_UP_SECONDS,
        "questions_per_project": QUESTIONS_PER_PROJECT,
        "questions_per_skill": per_skill,
        "total_questions": planned_question_count(len(projects), len(skill_names)),
    }

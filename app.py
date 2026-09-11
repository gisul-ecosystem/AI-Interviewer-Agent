"""
FastAPI Server for Live Real-Time Voice Interviewing.

Features:
    - Serves the live interview web interface
    - Mic PCM → Groq Whisper (or local Whisper) captions
    - Qwen3-4B phrasing and Kokoro TTS
"""

import time
import uuid
import difflib
import io
import os
import asyncio
import json
import re
import sys
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rag_engine import build_rag_context, init_rag, resume_rag
from interviewer.services.resume import ingest_resume, normalize_resume_text, parse_resume, sanitize_features

TTS_SYNTHESIZE_URL = "https://tts.gisul.ai/synthesize"

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

LLM_COMPLETIONS_URL = "https://llm.gisul.ai/v1/chat/completions"
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LLM_CONCURRENCY = asyncio.Semaphore(12)
# Grading runs after the call ends. Keep its budget small and separate so a
# backlog of scorecards can never slow down a live candidate's next question.
GRADING_CONCURRENCY = asyncio.Semaphore(2)
GRADING_QUEUE: "asyncio.Queue[str]" = asyncio.Queue()
GRADING_WORKERS = 2
_BACKGROUND_TASKS: list[asyncio.Task] = []
# Durable writes are queued so a live turn never waits on disk I/O.
PERSIST_QUEUE: "asyncio.Queue[str]" = asyncio.Queue()
_PENDING_PERSIST: set[str] = set()
_RUNTIME_RAG: dict[str, Any] = {}

# Do not store the live candidate's CV in process globals. Two concurrent
# uploads would otherwise cross-contaminate hotwords and interview plans.


from interviewer.services.stt_lexicon import PROTECTED_WORDS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _safe_log(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"))


HOTWORD_STOPWORDS = {
    "and", "with", "from", "using", "for", "the", "that", "this", "demonstrated", "through",
    "hands-on", "project", "development", "problems", "500+", "guna", "2023", "2024", "2025",
    "2026", "2027", "2028", "candidate", "name", "degree", "college", "university", "role"
}


def build_cv_hotwords(features: dict) -> list[str]:
    """Builds a flat priority list of all important proper nouns from CV features."""
    hotwords: list[str] = []

    def _add(val):
        if not val:
            return
        if isinstance(val, list):
            for v in val:
                _add(v)
        else:
            v = str(val).strip()
            if v and v.lower() not in ("null", "none"):
                s_low = v.lower()
                if any(w in s_low for w in INVALID_NAME_WORDS | INVALID_DEGREE_WORDS):
                    return
                hotwords.append(v)
                # Only keep long tokens. Short splits ("Mole", "Net") fuzzy-match
                # ordinary words like "model" / "movile" and corrupt the transcript.
                for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.]{4,}", v):
                    if token.lower() not in HOTWORD_STOPWORDS and token.lower() not in PROTECTED_WORDS:
                        hotwords.append(token)

    _add(features.get("name"))
    _add(features.get("skills"))
    _add(features.get("projects"))
    _add(features.get("college"))
    _add(features.get("degree"))
    _add(features.get("company"))
    _add(features.get("certifications"))
    _add(features.get("domains"))

    seen = set()
    result = []
    for w in hotwords:
        key = w.lower()
        if key not in seen and key not in HOTWORD_STOPWORDS:
            seen.add(key)
            result.append(w)
    return result


def fuzzy_correct_transcript(text: str, hotwords: list[str], cutoff: float = 0.78) -> str:
    """Word-level fuzzy correction: replaces STT words that are phonetically close to CV hotwords."""
    if not text or not hotwords:
        return text

    # Build a lowercased lookup map: lower -> original
    hw_lower = {w.lower(): w for w in hotwords}
    hw_keys = list(hw_lower.keys())

    words = text.split()
    corrected = []
    for word in words:
        clean_word = word.strip('.,!?;:\'"').lower()
        # Skip very short words (articles, prepositions, etc.)
        if len(clean_word) < 4:
            corrected.append(word)
            continue
        # Check if already a known hotword (exact match)
        if clean_word in hw_lower:
            corrected.append(word)
            continue
        # Fuzzy match against hotword list
        matches = difflib.get_close_matches(clean_word, hw_keys, n=1, cutoff=cutoff)
        if matches:
            # Preserve leading/trailing punctuation
            prefix = word[:len(word) - len(word.lstrip('.,!?;:\'"'))]
            suffix = word[len(word.rstrip('.,!?;:\'"')):]
            corrected.append(prefix + hw_lower[matches[0]] + suffix)
        else:
            corrected.append(word)
    return ' '.join(corrected)


def _stt_active_topic(session: dict | None) -> str:
    """Current project/question text so STT can restore names like MoleCheck."""
    if not session:
        return ""
    state = session.get("interview_state") or {}
    cand = session.get("candidate") or {}
    current_q = session.get("current_question") or {}
    stage = str(state.get("stage") or "")
    pieces = [str(state.get("current_topic") or ""), str(current_q.get("question") or "")]
    if stage in ("warmup", "project_deep_dive", "technical", ""):
        projects = cand.get("projects") or []
        idx = int(state.get("current_project_index") or 0)
        if projects and 0 <= idx < len(projects):
            pieces.append(str(projects[idx]))
    return " ".join(p for p in pieces if p)


def _persist_followup_thread(fsm, q_decision) -> None:
    spec = getattr(q_decision, "followup_spec", None) or {}
    thread = spec.get("project_thread")
    if thread:
        fsm.set_project_thread(thread)


async def _resolve_followup_anchor_for_turn(
    session: dict,
    answer: str,
    fsm_state: dict,
    intent: str,
) -> str:
    """Map this utterance onto a term from THIS resume. No global tech list."""
    if intent in ("REPEAT_REQUEST", "UNKNOWN_OR_SKIP", "TOPIC_CHANGE", "SELF_INTRO"):
        return ""
    stage = str((fsm_state or {}).get("stage") or "")
    if stage not in ("project_deep_dive", "technical", "skills_assessment", "behavioral"):
        return ""
    from interviewer.services.anchor import resolve_followup_anchor

    thread = (fsm_state or {}).get("project_thread") or {}
    candidate = session.get("candidate") or {}
    projects = candidate.get("projects") or []
    p_idx = int((fsm_state or {}).get("current_project_index") or 0)
    project = str(thread.get("project") or "")
    if not project and projects:
        project = str(projects[p_idx] if 0 <= p_idx < len(projects) else projects[0])
    last_q = str((session.get("current_question") or {}).get("question") or "")
    try:
        return await resolve_followup_anchor(
            answer,
            candidate,
            last_question=last_q,
            last_anchor=str(thread.get("last_anchor") or ""),
            project=project,
        )
    except Exception:
        return str(thread.get("last_anchor") or project or "")


_DYNAMIC_PHONETIC_REPLACEMENTS: list[tuple[str, str]] = []


KNOWN_TECH_VOCAB = [
    "Python", "Java", "C++", "C#", "C", "JavaScript", "TypeScript", "HTML", "CSS", "SQL",
    "React.js", "React", "Node.js", "Node", "Express.js", "Express", "Next.js", "Vue.js",
    "MongoDB", "MySQL", "PostgreSQL", "SQLite", "Redis", "Firebase",
    "Machine Learning", "Deep Learning", "Neural Networks", "LLMs", "Large Language Models",
    "TensorFlow", "PyTorch", "Scikit-Learn", "OpenCV", "NumPy", "Pandas", "Keras",
    "FastAPI", "Flask", "Django", "REST API", "RESTful API", "GraphQL",
    "Git", "GitHub", "Docker", "Kubernetes", "AWS", "Linux",
    "Data Structures", "Algorithms", "Data Structures and Algorithms", "DSA",
    "Object-Oriented Programming", "OOP", "MERN Stack", "MERN", "Artificial Intelligence", "AI"
]

INVALID_NAME_WORDS = {
    "skill", "skills", "leetcode", "demonstrated", "b.tech", "education", "experience",
    "projects", "summary", "curriculum", "vitae", "resume", "page", "http", "github",
    "email", "phone", "profile", "overview", "developer", "engineer", "intern", "stack", "domain",
    "problems", "development", "hands-on", "experienced"
}

INVALID_DEGREE_WORDS = {
    "skill", "skills", "leetcode", "500+", "demonstrated", "hands-on", "experienced",
    "problem", "problems", "project", "projects", "development", "developer", "proficient",
    "applications", "design", "curriculum"
}


def sanitize_resume_features(features: dict, raw_text: str) -> dict:
    """Keep LLM fields, then re-derive projects and skills from the source resume."""
    return sanitize_features(features if isinstance(features, dict) else {}, raw_text or "")


def build_cv_phonetic_replacements(features: dict) -> list:
    """Builds regex patterns for dynamic STT correction from candidate CV features."""
    replacements = []
    name = str(features.get("name") or "").strip()
    degree = str(features.get("degree") or "").strip()
    college = str(features.get("college") or "").strip()
    company_raw = features.get("company", features.get("companies"))
    company = ", ".join(company_raw) if isinstance(company_raw, list) else str(company_raw or "")
    skills = features.get("skills", [])
    projects = features.get("projects", [])

    # 1. DYNAMIC CANDIDATE NAME STT CORRECTION
    if name and name.lower() not in ("null", "general candidate", "none"):
        name_clean = re.sub(r"^(candidate name:|name:)\s*", "", name, flags=re.IGNORECASE).strip()
        parts = name_clean.split()
        if len(parts) >= 1:
            first_name = parts[0]
            last_name = parts[-1] if len(parts) > 1 else ""

            fn_low = first_name.lower()
            fn_variants = [re.escape(first_name)]
            if fn_low in ("aditya", "aditia", "adithya"):
                fn_variants.extend(["aditia", "adithya", "aditiya", "adittya", "additya", "a ditya", "adity", "adityo"])
            elif fn_low in ("ashutosh", "asutosh", "ashutoss"):
                fn_variants.extend(["asutosh", "ashutoss", "ashutos", "a shutosh", "ashutoshi", "asutose"])
            elif fn_low in ("rahul", "rahool"):
                fn_variants.extend(["raahul", "rahool", "rahuul"])

            ln_variants = [re.escape(last_name)] if last_name else []
            if last_name:
                ln_low = last_name.lower()
                if ln_low in ("bargujar", "barguzar", "bargoojar"):
                    ln_variants.extend(["barguzar", "bargoojar", "bar gujar", "bar gujjar", "bar goojar", "bargujar", "bar gujjar"])
                elif ln_low in ("soni", "sony"):
                    ln_variants.extend(["sony", "soney", "soani", "sooni"])
                elif ln_low in ("sharma", "sarma"):
                    ln_variants.extend(["sarma", "sharmma"])

            fn_pattern = "|".join(fn_variants)
            ln_pattern = "|".join(ln_variants) if ln_variants else ""

            if ln_pattern:
                intro_regex = rf"\b(my name is|i am|i'm|this is)\s+(?:{fn_pattern})\s+(?:{ln_pattern})\b"
                replacements.append((intro_regex, f"my name is {name_clean}"))

                full_name_regex = rf"\b(?:{fn_pattern})\s+(?:{ln_pattern})\b"
                replacements.append((full_name_regex, name_clean))

                replacements.append((rf"\b(?:{ln_pattern})\b", last_name))

            replacements.append((rf"\b(?:{fn_pattern})\b", first_name))

    # 2. Degree / Qualification fixes
    if "b.tech" in degree.lower() or "bachelor" in degree.lower():
        replacements.extend([
            (r"\bseeing my ABT\b", "pursuing my B.Tech"),
            (r"\bABT\b", "B.Tech"),
            (r"\bvideo\.? Tag\b", "B.Tech"),
            (r"\bB Tech\b", "B.Tech"),
            (r"\bB\. Tech\b", "B.Tech"),
            (r"\bbittech\b", "B.Tech"),
        ])
    elif "m.tech" in degree.lower() or "master" in degree.lower():
        replacements.extend([
            (r"\bAMT\b", "M.Tech"),
            (r"\bM Tech\b", "M.Tech"),
            (r"\bmittech\b", "M.Tech"),
        ])

    # 3. College Name fixes
    col_low = college.lower()
    if "jaypee" in col_low or "juet" in col_low:
        replacements.extend([
            (r"\bChrome GP\b", "Jaypee"),
            (r"\bGP Institute\b", "Jaypee Institute"),
            (r"\bJP Institute\b", "Jaypee Institute"),
            (r"\bJay pees\b", "Jaypee"),
            (r"\bJP University\b", "Jaypee University"),
            (r"\bJ U E T\b", "JUET"),
            (r"\bJUET University\b", "Jaypee University of Engineering and Technology"),
        ])

    # 4. Company & Internship fixes
    if "gisul" in company.lower():
        replacements.extend([
            (r"\bGisol\b", "GISUL"),
            (r"\bGisel\b", "GISUL"),
            (r"\bGisul AI\b", "GISUL"),
        ])
    if "labmentix" in company.lower():
        replacements.extend([
            (r"\bLab mentix\b", "Labmentix"),
            (r"\bLabmentics\b", "Labmentix"),
        ])

    from interviewer.services.stt_lexicon import build_term_patterns, collect_cv_terms

    for pattern, replacement in build_term_patterns(collect_cv_terms(features)):
        replacements.append((pattern, replacement))

    return replacements


def clean_transcript(text: str) -> str:
    """Corrects Indian English ASR phonetic distortions using CV-aware dynamic regex replacements."""
    if not text:
        return text
    cleaned = text
    for pattern, replacement in _DYNAMIC_PHONETIC_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extracts raw text from PDF, TXT, or MD resume file."""
    if filename.lower().endswith(".pdf"):
        if not HAS_PYPDF:
            raise ValueError("pypdf is not installed on the server to parse PDF files.")
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in pdf_reader.pages:
            t = ""
            try:
                t = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                t = ""
            if not t.strip():
                t = page.extract_text() or ""
            if t:
                pages_text.append(t)
        return normalize_resume_text("\n".join(pages_text))
    return normalize_resume_text(file_bytes.decode("utf-8", errors="replace"))


def parse_resume_heuristics(text: str) -> dict:
    """Fallback heuristic parser if LLM extraction encounters an error."""
    return parse_resume(text or "")


def extract_resume_features_llm(resume_text: str) -> dict:
    """Heuristic parse is authoritative. Qwen may enrich name/college/role; timeouts never fail ingest."""
    resume_text = normalize_resume_text(resume_text)
    fallback = ingest_resume(resume_text)
    if os.environ.get("RESUME_LLM_EXTRACT", "").strip().lower() not in ("1", "true", "yes", "on"):
        return fallback
    prompt_text = (
        "You are an AI resume analyzer. Extract the candidate's key terms and features from the resume text into JSON.\n"
        "STRICT MANDATORY RULES:\n"
        "1. Extract ONLY facts explicitly stated in the resume text. If a field (e.g. experience, company, degree, college, role, certifications, domains) is NOT explicitly mentioned in the CV text, set its value to null or empty list []. DO NOT invent, guess, or assume missing details.\n"
        "2. ROLE VS DOMAIN RULE: Only set \"role\" if the candidate has formal work experience/employment listed in that position with a duration or company. If a title like \"Software Developer\" or \"Machine Learning Intern\" is written in the CV without explicit work experience duration or company, set \"role\": null and place that title in \"domains\" as their specialization.\n"
        "4. PROJECTS: return only software project TITLES (e.g. \"Ferrite Mesh\", \"MoleCheck\"). Never descriptions, never action verbs, never skills, never languages/frameworks such as Python or TensorFlow.\n"
        "5. SKILLS and PROJECTS are disjoint. A skill name must not appear in projects.\n\n"
        "Return ONLY a single valid JSON object with these exact keys:\n"
        '  "name": full name of candidate (string or null)\n'
        '  "college": university / college / institute name (string or null)\n'
        '  "degree": degree / qualification (string or null)\n'
        '  "role": formal job role ONLY IF accompanied by stated work experience duration or company (string or null)\n'
        '  "experience": explicitly stated years of experience or work duration (string or null)\n'
        '  "company": company, organization, or employer name(s) where candidate worked (string, list of strings, or null)\n'
        '  "skills": core technical skills, programming languages, tools & frameworks explicitly mentioned (list of strings). NEVER include section headers such as Certifications, Skills, Projects, Education, Experience.\n'
        '  "projects": short project TITLES only, 2-6 words each, not full descriptions, never the word Certifications or other resume headings (list of strings)\n'
        '  "certifications": explicit certifications, achievements, or honors (list of strings)\n'
        '  "domains": specialized technical domain focus or non-employed specializations (list of strings)\n\n'
        f'Resume Text:\n"""\n{resume_text[:4000]}\n"""\n\n'
        "Do not include any explanation or markdown formatting outside the JSON object."
    )

    try:
        from interviewer.adapters.registry import get_llm
        from interviewer.ports.llm import LLMRequest, LLMTask, Message

        async def _extract():
            return await get_llm().complete(
                LLMRequest(
                    task=LLMTask.EXTRACT,
                    messages=[
                        Message(role="system", content="You are a high-precision JSON resume extraction engine."),
                        Message(role="user", content=prompt_text),
                    ],
                    temperature=0.1,
                    max_tokens=600,
                )
            )

        result = asyncio.run(_extract())
        if not result.ok:
            raise RuntimeError(result.detail or result.failure)
        content = result.text
        cleaned_json = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
        cleaned_json = re.sub(r"\s*```$", "", cleaned_json).strip()

        parsed = json.loads(cleaned_json)
        if isinstance(parsed, dict):
            return ingest_resume(resume_text, parsed)
    except Exception as e:
        _safe_log(f"[Resume LLM Extract Exception] {e}")

    return fallback


def generate_system_prompt_from_features(features: dict) -> str:
    """Returns the adaptive JSON interviewer system prompt from system_prompt.txt.
    The new prompt is a JSON-in/JSON-out adaptive engine — it does not need to embed
    candidate details because those are injected at runtime via the structured JSON payload."""
    prompt_file = BASE_DIR / "system_prompt.txt"
    if prompt_file.is_file():
        return prompt_file.read_text(encoding="utf-8")
    return "You are an expert AI Technical Interviewer. Return only valid JSON per the output format."


def _estimate_experience_years(features: dict) -> int:
    """Infer rough years of experience from resume features."""
    exp = str(features.get("experience") or "")
    import re as _re
    nums = _re.findall(r"\d+", exp)
    return int(nums[0]) if nums else 0


def _infer_experience_level(years: int) -> str:
    if years == 0:
        return "fresher"
    if years <= 2:
        return "junior"
    if years <= 5:
        return "mid"
    return "senior"


def _important_topics_from_features(features: dict) -> list[str]:
    """Derive the top important interview topics from resume skills and domains."""
    raw = []
    for key in ("skills", "domains", "projects"):
        val = features.get(key, [])
        if isinstance(val, list):
            raw.extend(val)
        elif isinstance(val, str) and val.strip():
            raw.extend([v.strip() for v in val.split(",")])
    # deduplicate, lowercase, take first 8
    seen: set = set()
    result = []
    for t in raw:
        tl = t.strip().lower()
        if tl and tl not in seen:
            seen.add(tl)
            result.append(t.strip())
        if len(result) >= 8:
            break
    return result


def _build_resume_context_str(features: dict) -> str:
    """Build a compact resume context string for the JSON payload."""
    lines = []
    for key, label in [
        ("name", "Name"), ("college", "College"), ("degree", "Degree"),
        ("role", "Role"), ("experience", "Experience"), ("company", "Company"),
    ]:
        val = features.get(key)
        if val and str(val).strip() and str(val).lower() != "null":
            lines.append(f"{label}: {val}")
    for key, label in [("skills", "Skills"), ("projects", "Projects"),
                       ("certifications", "Certifications"), ("domains", "Domains")]:
        val = features.get(key, [])
        if isinstance(val, list) and val:
            lines.append(f"{label}: {', '.join(str(v) for v in val)}")
        elif isinstance(val, str) and val.strip():
            lines.append(f"{label}: {val}")
    return "\n".join(lines)





# ── Interview Session Store (Redis + In-Memory Fallback) ──────────────────────
# Powered by backend.redis_client.session_store for persistence across restarts
from backend.role_sift import sift_skills_for_role
from backend.redis_client import session_store
INTERVIEW_SESSIONS: dict[str, dict] = session_store._memory_sessions

from interviewer.services.interview_structure import (
    DURATION_SECONDS as INTERVIEW_DURATION_SECONDS,
    opening_greeting,
)


def load_live_session(session_id: str) -> Optional[dict]:
    """Load a session by id, rehydrating per-session ResumeRAG from stored resume text."""
    if not session_id:
        return None
    sess = INTERVIEW_SESSIONS.get(session_id)
    if sess is None:
        sess = session_store.get_session(session_id)
        if sess is None:
            return None
        INTERVIEW_SESSIONS[session_id] = sess
    if sess.get("session_rag") is None:
        from rag_engine import ResumeRAG
        rag = _RUNTIME_RAG.get(session_id)
        if rag is None:
            rag = ResumeRAG(chunk_size=120, overlap=30)
            text = sess.get("resume_text") or (sess.get("candidate") or {}).get("resume_context") or ""
            if text and str(text).strip():
                rag.index(str(text), session_id=session_id)
            _RUNTIME_RAG[session_id] = rag
        sess["session_rag"] = rag
    cand = sess.setdefault("candidate", {})
    if sess.get("interview_plan") and not cand.get("interview_plan"):
        cand["interview_plan"] = sess["interview_plan"]
    return sess


def _spoken_from_decision(q_decision, candidate_dict: dict) -> str:
    if getattr(q_decision, "spoken_question", None):
        return q_decision.spoken_question
    if getattr(q_decision, "seed_question", None):
        return q_decision.seed_question
    projects = candidate_dict.get("projects") or ["your recent work"]
    return f"Could you walk me through {projects[0]} and what you personally owned?"


def _build_current_question(q_decision, question_text: str, topic: str, difficulty) -> dict:
    """Bind the spoken question to the bank rubric that will grade its answer."""
    record = getattr(q_decision, "selected_question", None) or {}
    return {
        "question": question_text,
        "skill": topic,
        "difficulty": difficulty,
        "competency": record.get("competency") or topic,
        "expected_concepts": record.get("expected_concepts", []),
        "strong_signals": record.get("strong_signals", []),
        "weak_signals": record.get("weak_signals", []),
        "evaluation_rubric": record.get("strong_signals", []),
        "question_id": record.get("id"),
        "question_mode": getattr(q_decision, "mode", None),
    }


def _build_turn_record(asked_q: dict, candidate_answer: str, next_question: str, eval_res: dict, turn_index: int = 0) -> dict:
    """One transcript row: the question asked, its rubric, and the answer given."""
    asked_q = asked_q or {}
    return {
        "turn": turn_index,
        "asked_question": asked_q.get("question", ""),
        "candidate_answer": candidate_answer,
        "next_question": next_question,
        "topic": asked_q.get("skill") or asked_q.get("competency") or "General",
        "competency": asked_q.get("competency") or asked_q.get("skill") or "General",
        "question_id": asked_q.get("question_id"),
        "question_difficulty": asked_q.get("difficulty"),
        "question_mode": asked_q.get("question_mode"),
        "expected_concepts": asked_q.get("expected_concepts", []),
        "strong_signals": asked_q.get("strong_signals", []),
        "weak_signals": asked_q.get("weak_signals", []),
        "live_score": eval_res.get("score", 0.6),
        "live_depth": eval_res.get("depth"),
        "asked_at": time.time(),
    }


def _append_turn(session: dict, asked_q: dict, candidate_answer: str, next_question: str, eval_res: dict) -> dict:
    """Append a transcript row and schedule a durable write."""
    history = session.setdefault("history", [])
    record = _build_turn_record(asked_q, candidate_answer, next_question, eval_res, len(history) + 1)
    history.append(record)
    _schedule_persist(session.get("session_id"))
    return record


def _tick_session_clock(session: dict) -> None:
    """Count the 15-minute budget down on every turn, including the streaming path."""
    now = time.time()
    last = session.get("last_turn_time") or session.get("created_at") or now
    elapsed = max(0, int(now - last))
    session["last_turn_time"] = now
    state = session.setdefault("interview_state", {})
    remaining = int(state.get("time_remaining_seconds") or INTERVIEW_DURATION_SECONDS)
    state["time_remaining_seconds"] = max(0, remaining - elapsed)
    started = session.get("started_at") or session.get("created_at") or now
    state["elapsed_seconds"] = max(0, int(now - started))


async def _finish_interview(session: dict) -> dict:
    """
    Mark the session complete, write the transcript to Redis + SQLite, queue grading.

    Safe to call more than once. Redis is the hot copy (TTL ~2h). SQLite is the
    durable transcript that survives Redis expiry and process restart.
    """
    from backend.store import interview_store

    session_id = session.get("session_id")
    state = session.setdefault("interview_state", {})
    state["stage"] = "completed"
    state["action"] = "END_INTERVIEW"
    if not session.get("started_at"):
        session["started_at"] = session.get("created_at") or time.time()
    if not session.get("ended_at"):
        session["ended_at"] = time.time()
    session["duration_seconds"] = max(
        0, int(session["ended_at"] - session["started_at"])
    )
    if session_id:
        session_store.save_session(session_id, session)
        persisted = await asyncio.to_thread(interview_store.save_interview, session)
        existing = session_store.get_report(session_id) or {}
        if existing.get("status") not in ("queued", "running", "done"):
            session_store.save_report(session_id, {"session_id": session_id, "status": "queued"})
            await GRADING_QUEUE.put(session_id)
            grading = "queued"
        else:
            grading = existing.get("status")
    else:
        persisted = False
        grading = "skipped"
    return {
        "interview_complete": True,
        "persisted": persisted,
        "grading": grading,
        "turns": len(session.get("history") or []),
        "stage": "completed",
        "time_remaining_seconds": state.get("time_remaining_seconds", 0),
    }


def _should_complete_after_turn(state: dict, q_decision) -> bool:
    """Finish only after the wrap-up turn is answered (stage completed), not when wrap-up is spoken."""
    stage = str((state or {}).get("stage") or "")
    action = str((state or {}).get("action") or "")
    if stage == "completed" or action == "END_INTERVIEW":
        return True
    return False


def _schedule_persist(session_id: Optional[str]) -> None:
    """
    Queue a durable write without blocking the turn.

    Candidates disconnect without ever calling /api/end-interview, so the
    transcript is flushed to disk as the interview happens, not only at the end.
    """
    if not session_id or session_id in _PENDING_PERSIST:
        return
    try:
        _PENDING_PERSIST.add(session_id)
        PERSIST_QUEUE.put_nowait(session_id)
    except Exception:
        _PENDING_PERSIST.discard(session_id)


async def _persist_worker():
    """Drains the durable-write queue. Never blocks a live turn."""
    from backend.store import interview_store

    while True:
        session_id = await PERSIST_QUEUE.get()
        _PENDING_PERSIST.discard(session_id)
        try:
            session = INTERVIEW_SESSIONS.get(session_id) or session_store.get_session(session_id)
            if session:
                await asyncio.to_thread(interview_store.save_interview, session)
        except Exception as exc:
            print(f"[Persist] failed session={session_id}: {exc}")
        finally:
            PERSIST_QUEUE.task_done()


async def _call_judge_llm(payload: dict) -> str:
    """LLM call for the grading pass. Uses the shared adapter, never a hardcoded URL."""
    from interviewer.adapters.registry import get_llm
    from interviewer.ports.llm import LLMRequest, LLMTask, Message

    msgs = payload.get("messages") or []
    messages = [Message(role=m["role"], content=m["content"]) for m in msgs]
    async with GRADING_CONCURRENCY:
        result = await get_llm().complete(
            LLMRequest(
                task=LLMTask.JUDGE,
                messages=messages,
                temperature=float(payload.get("temperature") or 0.0),
                max_tokens=int(payload.get("max_tokens") or 420),
            )
        )
    if not result.ok:
        raise RuntimeError(f"judge {result.failure}: {result.detail}")
    return result.text


async def persist_graded_report(session_id: str, session: Optional[dict] = None) -> dict:
    """Always write a scorecard to Redis and SQLite. Heuristic if the judge path dies."""
    from backend.grader import grade_session
    from backend.store import interview_store

    if session is None:
        session = (
            session_store.get_session(session_id)
            or INTERVIEW_SESSIONS.get(session_id)
            or await asyncio.to_thread(interview_store.get_interview, session_id)
        )
    if not session:
        report = {
            "session_id": session_id,
            "status": "failed",
            "error": "session not found or expired",
            "overall_score": 0,
            "overall_out_of": 5,
            "recommendation": "insufficient_data",
            "graded_by": "none",
        }
        session_store.save_report(session_id, report)
        await asyncio.to_thread(interview_store.save_report, session_id, report)
        return report

    session_store.save_report(session_id, {"session_id": session_id, "status": "running"})
    try:
        report = await grade_session(session, llm_call=_call_judge_llm, model=LLM_MODEL)
    except Exception as exc:
        print(f"[Grader] LLM path failed session={session_id}: {exc}; using heuristic")
        report = await grade_session(session, llm_call=None)
        report["grader_error"] = str(exc)[:300]
    report["status"] = "done"
    report["graded_at"] = time.time()
    session_store.save_report(session_id, report)
    await asyncio.to_thread(interview_store.save_report, session_id, report)
    return report


async def _grading_worker(worker_id: int):
    """Drains the grading queue. Never touches the live interview path."""
    while True:
        session_id = await GRADING_QUEUE.get()
        try:
            report = await persist_graded_report(session_id)
            print(
                f"[Grader:{worker_id}] session={session_id} score={report.get('overall_score')} "
                f"rec={report.get('recommendation')} via={report.get('graded_by')} status={report.get('status')}"
            )
        except Exception as exc:
            print(f"[Grader:{worker_id}] persist failed session={session_id}: {exc}")
            try:
                from backend.grader import grade_session
                from backend.store import interview_store
                session = (
                    session_store.get_session(session_id)
                    or INTERVIEW_SESSIONS.get(session_id)
                    or await asyncio.to_thread(interview_store.get_interview, session_id)
                )
                report = await grade_session(session or {"session_id": session_id, "history": []}, llm_call=None)
                report["status"] = "done"
                report["graded_at"] = time.time()
                report["grader_error"] = str(exc)[:300]
                session_store.save_report(session_id, report)
                await asyncio.to_thread(interview_store.save_report, session_id, report)
            except Exception as inner:
                failed = {
                    "session_id": session_id,
                    "status": "failed",
                    "error": str(inner)[:300],
                    "overall_score": 0,
                    "recommendation": "insufficient_data",
                    "graded_by": "none",
                }
                session_store.save_report(session_id, failed)
                try:
                    from backend.store import interview_store
                    await asyncio.to_thread(interview_store.save_report, session_id, failed)
                except Exception:
                    pass
        finally:
            GRADING_QUEUE.task_done()


async def _generate_followup_question(llm_payload: dict) -> str:
    """Call Qwen to phrase the already-chosen question. Empty string on timeout/error."""
    if not llm_payload:
        return ""
    from interviewer.adapters.registry import get_llm
    from interviewer.ports.llm import LLMRequest, LLMTask, Message

    msgs = llm_payload.get("messages") or []
    messages = [Message(role=m["role"], content=m["content"]) for m in msgs]
    try:
        async with LLM_CONCURRENCY:
            result = await get_llm().complete(
                LLMRequest(
                    task=LLMTask.PHRASING,
                    messages=messages,
                    temperature=float(llm_payload.get("temperature") or 0.2),
                    max_tokens=int(llm_payload.get("max_tokens") or 160),
                )
            )
        if result.ok:
            return clean_interviewer_speech(result.text.strip())
    except Exception as e:
        print(f"[Interview Turn LLM Error] {e}")
    return ""


async def _phrase_with_qwen(q_decision, intent: str, fallback: str, llm_payload: dict | None) -> str:
    """NLG polish. Policy seed is spoken if Qwen is down or goes off-brief."""
    if not getattr(q_decision, "wants_qwen_phrasing", lambda _intent: False)(intent):
        return fallback
    from backend.followup_engine import finalize_spoken_question
    llm_question = await _generate_followup_question(llm_payload or {})
    return finalize_spoken_question(llm_question, q_decision, fallback) or fallback


_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


async def _iter_phrased_sentences(q_decision, intent: str, fallback: str, llm_payload: dict | None):
    """
    Yield the spoken question in TTS-sized sentences.

    Qwen only phrases the question policy already chose. The whole reply is
    buffered and passed through the quality gate before anything is spoken:
    streaming a first clause let ungated, off-topic wording reach Kokoro.
    """
    if not getattr(q_decision, "wants_qwen_phrasing", lambda _intent: False)(intent) or not llm_payload:
        if fallback:
            yield fallback
        return

    from interviewer.adapters.registry import get_llm
    from interviewer.ports.llm import LLMRequest, LLMTask, Message
    from backend.followup_engine import finalize_spoken_question

    messages = []
    for item in llm_payload.get("messages") or []:
        role = item.get("role")
        if role in ("system", "user", "assistant") and item.get("content"):
            messages.append(Message(role=role, content=str(item["content"])))
    if not messages:
        if fallback:
            yield fallback
        return

    buf = ""
    try:
        async with asyncio.timeout(3.5):
            async for delta in get_llm().stream(
                LLMRequest(
                    task=LLMTask.PHRASING,
                    messages=messages,
                    temperature=float(llm_payload.get("temperature") or 0.4),
                    max_tokens=int(llm_payload.get("max_tokens") or 160),
                    deadline_ms=3200,
                )
            ):
                if delta:
                    buf += delta
    except Exception as exc:
        print(f"[Phrasing stream] {exc}")

    phrased = clean_interviewer_speech(buf).strip()
    final = finalize_spoken_question(phrased, q_decision, fallback) if phrased else (fallback or "")
    for sentence in _SENTENCE_SPLIT.split(final):
        text = sentence.strip()
        if text:
            yield text


def _make_session(features: dict, role_override: str | None = None, raw_resume_text: str | None = None, job_description: str | None = None) -> dict:
    """Build a fresh interview session from extracted resume features with isolated ResumeRAG."""
    import uuid, time
    from rag_engine import ResumeRAG

    years = _estimate_experience_years(features)
    exp_level = _infer_experience_level(years)
    resume_ctx = _build_resume_context_str(features)

    name_raw = features.get("name") or ""
    name = name_raw if name_raw.lower() not in ("", "null") else "the Candidate"

    skills_raw = features.get("skills", [])
    cv_skills = skills_raw if isinstance(skills_raw, list) else [skills_raw] if skills_raw else []
    sift = sift_skills_for_role(features, role_override=role_override, job_description=job_description)
    # Live skills are CV ∩ role only. Gaps stay on the report, never in the plan.
    skills = list(sift.get("intersection") or [])
    projects = list(sift.get("interview_projects") or features.get("projects") or [])
    target_role = sift["label"]
    topics = skills or _important_topics_from_features(features)

    sid = str(uuid.uuid4())
    # Dedicated ResumeRAG instance for complete multi-interview concurrency
    session_rag = ResumeRAG(chunk_size=120, overlap=30)
    text_to_index = raw_resume_text or features.get("raw_text") or resume_ctx
    if text_to_index and text_to_index.strip():
        session_rag.index(text_to_index, session_id=sid)
    _RUNTIME_RAG[sid] = session_rag

    session_phonetics = build_cv_phonetic_replacements(features)
    phonetic_patterns = [[pat, repl] for pat, repl in session_phonetics]
    cv_hotwords = build_cv_hotwords(features)

    from backend.interview_plan import build_interview_plan
    from rag_engine import question_bank_rag as _qbank
    interview_plan = build_interview_plan(
        {
            "name": name,
            "skills": skills,
            "projects": projects,
            "resume_text": text_to_index or resume_ctx or "",
            "bank_track": sift["bank_track"],
            "target_track": sift["track"],
            "interview_style": sift["style"],
        },
        question_bank_rag=_qbank if getattr(_qbank, "is_ready", False) else None,
    )
    planned_skills = list(interview_plan.get("skill_names") or skills)
    greeting = opening_greeting(
        name,
        target_role,
        interview_plan.get("duration_seconds") or INTERVIEW_DURATION_SECONDS,
        projects,
        planned_skills,
    )

    now = time.time()
    return {
        "session_id": sid,
        "created_at": now,
        "started_at": now,
        "ended_at": None,
        "duration_seconds": None,
        "last_turn_time": now,

        "session_rag": session_rag,
        "phonetic_replacements": session_phonetics,
        "phonetic_patterns": phonetic_patterns,
        "cv_hotwords": cv_hotwords,
        "resume_text": text_to_index or "",
        "interview_plan": interview_plan,
        "resume_token": features.get("resume_token") or "",
        "opening_greeting": greeting,
        # ── candidate block (sent to LLM each turn) ──
        "candidate": {
            "name": name,
            "college": features.get("college"),
            "degree": features.get("degree"),
            "role": features.get("role"),
            "experience_years": years,
            "skills": planned_skills,
            "cv_skills": cv_skills,
            "projects": projects,
            "certifications": features.get("certifications", []),
            "domains": features.get("domains", []),
            "resume_context": resume_ctx,
            "target_role": target_role,
            "target_track": sift["track"],
            "bank_track": sift["bank_track"],
            "interview_style": sift["style"],
            "skill_sift": {
                "intersection": sift["intersection"],
                "role_gaps": sift["role_gaps"],
                "cv_only": sift["cv_only"],
            },
            "job_description": (job_description or "")[:2000],
            "interview_plan": interview_plan,
        },
        # ── job requirements (role ∩ CV) ──
        "job_requirements": {
            "title": target_role,
            "track": sift["track"],
            "skills": sift["required_skills"],
            "interview_skills": planned_skills,
            "intersection": sift["intersection"],
            "role_gaps": sift["role_gaps"],
            "experience_level": exp_level,
            "important_topics": topics,
            "job_description": (job_description or "")[:4000],
        },
        # ── live interview state (updated every turn) ──
        "interview_state": {
            "stage": "warmup",
            "current_topic": (projects or topics or ["Resume Technical Background"])[0],
            "difficulty": 1,
            "difficulty_level": 1,
            "questions_asked": 0,
            "questions_remaining": interview_plan.get("total_questions") or 8,
            "time_remaining_seconds": interview_plan.get("duration_seconds") or INTERVIEW_DURATION_SECONDS,
            "topics_covered": [],
            "questions_already_asked": [],
            "candidate_weaknesses": [],
            "topic_scores": {},
            "action": "CONTINUE",
            "current_project_index": 0,
            "project_question_count": 0,
            "current_skill_index": 0,
            "skill_question_count": 0,
            "project_thread": {},
        },
        # ── last question sent to candidate ──
        "current_question": {
            "question": greeting,
            "skill": "Introduction",
            "difficulty": 1,
            "expected_concepts": ["Name", "Background", "Primary Tech Stack"],
            "evaluation_rubric": [],
        },
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the question bank, start background workers, and close the LLM pool."""
    await asyncio.to_thread(init_rag)
    print("[Startup] RAG engine initialised.")
    from backend.store import interview_store
    print(f"[Startup] Durable store: {'ready at ' + interview_store.db_path if interview_store.is_available else 'UNAVAILABLE'}")
    for i in range(GRADING_WORKERS):
        _BACKGROUND_TASKS.append(asyncio.create_task(_grading_worker(i + 1)))
    _BACKGROUND_TASKS.append(asyncio.create_task(_persist_worker()))
    print(f"[Startup] {GRADING_WORKERS} grading workers + 1 persist worker running.")
    from interviewer.config import settings as _settings
    from interviewer.services.whisper_stt import uses_whisper_api, warmup_whisper
    provider = _settings.speech.stt_provider.lower()
    if provider in ("nemotron", "fastconformer"):
        print("[Startup] STT_PROVIDER=nemotron is disabled. Using Whisper instead.")
    if uses_whisper_api():
        if _settings.speech.stt_api_key:
            print("[Startup] Whisper large-v3-turbo via Groq API (STT_API_KEY / GROQ_API_KEY).")
        else:
            print("[Startup] WARNING: STT_PROVIDER=whisper_api but no STT_API_KEY or GROQ_API_KEY.")
    elif provider == "whisper":
        _BACKGROUND_TASKS.append(asyncio.create_task(asyncio.to_thread(warmup_whisper)))
        print("[Startup] Local Whisper large-v3-turbo loading in background.")
    try:
        yield
    finally:
        for task in list(_BACKGROUND_TASKS):
            task.cancel()
        for task in list(_BACKGROUND_TASKS):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        _BACKGROUND_TASKS.clear()
        from interviewer.adapters.registry import close_llm
        await close_llm()
        print("[Shutdown] Background workers stopped.")


app = FastAPI(title="Live Voice AI Interviewer", version="2.0", lifespan=lifespan)


from interviewer.config import settings as _app_settings

_cors_origins = [o.strip() for o in (_app_settings.interview.allowed_origins or "*").split(",") if o.strip()]
_cors_wildcard = not _cors_origins or _cors_origins == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_wildcard else _cors_origins,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _api_key_gate(request, call_next):
    required = (_app_settings.interview.api_key or "").strip()
    path = request.url.path
    public = path in ("/", "/api/health", "/api/ready") or path.startswith("/static")
    if not required or public:
        return await call_next(request)
    provided = (request.headers.get("x-api-key") or "").strip()
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        provided = provided or auth[7:].strip()
    if provided != required:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _load_cv_hotwords_from_disk() -> None:
    """Legacy disk cache is not used for live interviews (session isolation)."""
    return



# Load existing hotwords and phonetic rules immediately when server starts
_load_cv_hotwords_from_disk()


@app.get("/api/health")
async def api_health():
    return {"status": "ok"}


@app.get("/api/ready")
async def api_ready():
    from backend.store import interview_store
    from rag_engine import question_bank_rag
    from interviewer.config import settings as ready_cfg
    return {
        "status": "ok",
        "store": bool(interview_store.is_available),
        "redis": bool(session_store.is_redis_active),
        "question_bank": bool(getattr(question_bank_rag, "is_ready", False)),
        "stt": ready_cfg.speech.stt_provider,
        "stt_key": bool(ready_cfg.speech.stt_api_key) if ready_cfg.speech.stt_provider in ("whisper_api", "groq", "api") else True,
    }


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Live Voice Interview App</h1><p>UI loading...</p>")


@app.get("/api/system-prompt")
async def get_system_prompt():
    prompt_file = BASE_DIR / "system_prompt.txt"
    if prompt_file.is_file():
        return {"prompt": prompt_file.read_text(encoding="utf-8")}
    return {"prompt": "You are an expert AI Technical Interviewer."}


@app.get("/api/cv-hotwords")
async def get_cv_hotwords(session_id: str | None = None):
    """Session-scoped CV hotwords. Empty unless a live session_id is provided."""
    session = load_live_session(session_id) if session_id else None
    hotwords = (session or {}).get("cv_hotwords") or []
    return {"hotwords": hotwords, "count": len(hotwords)}


from backend.transcript_corrector import correct_transcript_fast


async def build_rag_context_async(candidate_answer: str) -> str:
    """Run TF-IDF RAG retrieval in a thread to avoid blocking the async event loop."""
    return await asyncio.to_thread(build_rag_context, candidate_answer)


@app.post("/api/correct-transcript")
async def api_correct_transcript(payload: dict):
    raw_text = payload.get("transcript", "")
    session_id = payload.get("session_id")
    session = load_live_session(session_id) if session_id else None
    corrected = correct_transcript_fast(
        raw_text,
        candidate_dict=(session or {}).get("candidate") or {},
        hotwords=(session or {}).get("cv_hotwords") or [],
        phonetic_patterns=(session or {}).get("phonetic_patterns") or [],
        active_topic=_stt_active_topic(session),
    )
    rag_context = ""
    if session and session.get("session_rag") is not None:
        rag_context = "\n".join(session["session_rag"].retrieve(corrected or raw_text, top_k=2) or [])
    return {"raw": raw_text, "corrected": corrected, "rag_context": rag_context}


def _plan_candidate_payload(features: dict, plan: dict, sift: dict | None = None, raw_text: str = "") -> dict:
    sift = sift or {}
    return {
        "name": features.get("name") or "",
        "role": sift.get("label") or features.get("role"),
        "target_role": sift.get("label") or features.get("target_role") or features.get("role"),
        "projects": plan.get("project_names") or features.get("projects") or [],
        "resume_text": raw_text or features.get("raw_text") or "",
    }


async def _qwen_fill_skill_plan(plan: dict, features: dict, sift: dict | None = None, raw_text: str = "") -> dict:
    """Generate skill questions at upload/preview so the live path only phrases them."""
    from backend.interview_plan import generate_skill_questions_with_qwen
    return await generate_skill_questions_with_qwen(
        plan,
        _plan_candidate_payload(features, plan, sift, raw_text),
    )


def _apply_plan_to_session(session: dict, plan: dict) -> None:
    session["interview_plan"] = plan
    candidate = session.setdefault("candidate", {})
    candidate["interview_plan"] = plan
    skills = list(plan.get("skill_names") or [])
    if skills:
        candidate["skills"] = skills
        session.setdefault("job_requirements", {})["interview_skills"] = skills


@app.post("/api/preview-agenda")
async def preview_agenda(payload: dict = None):
    """Show the recruiter/candidate what this interview will cover before the mic opens."""
    payload = payload or {}
    resume_token = payload.get("resume_token")
    if not resume_token:
        raise HTTPException(status_code=400, detail="Upload a resume first")
    profile = session_store.get_resume_profile(str(resume_token))
    if not profile:
        raise HTTPException(status_code=404, detail="Resume token was not found; upload the resume again")
    features = profile.get("features") or {}
    role_override = payload.get("role")
    if role_override == "auto":
        role_override = None
    job_description = (payload.get("job_description") or payload.get("jd") or "").strip()
    sift = sift_skills_for_role(features, role_override=role_override, job_description=job_description)
    from rag_engine import question_bank_rag as _qbank
    from backend.interview_plan import build_interview_plan
    if _qbank and not getattr(_qbank, "is_ready", False):
        try:
            _qbank.load()
        except Exception:
            pass
    cached = profile.get("interview_plan") or {}
    cached_names = list(cached.get("skill_names") or [])
    plan = build_interview_plan(
        {
            "name": features.get("name") or "",
            "skills": list(sift.get("intersection") or []),
            "projects": list(sift.get("interview_projects") or features.get("projects") or []),
            "resume_text": features.get("raw_text") or "",
            "bank_track": sift["bank_track"],
            "target_track": sift["track"],
            "interview_style": sift["style"],
        },
        question_bank_rag=_qbank if getattr(_qbank, "is_ready", False) else None,
    )
    if (
        cached.get("skill_questions_source") == "qwen"
        and cached_names == list(plan.get("skill_names") or [])
        and cached.get("skill_slots")
    ):
        plan = cached
    else:
        plan = await _qwen_fill_skill_plan(plan, features, sift, features.get("raw_text") or "")
        profile["interview_plan"] = plan
        session_store.save_resume_profile(str(resume_token), profile)
    duration = int(plan.get("duration_seconds") or INTERVIEW_DURATION_SECONDS)
    name = features.get("name") or "Candidate"
    greeting = opening_greeting(
        name,
        sift["label"],
        duration,
        plan.get("project_names") or [],
        plan.get("skill_names") or [],
    )
    return {
        "status": "ok",
        "name": name,
        "role_label": sift["label"],
        "track": sift["track"],
        "track_source": sift.get("track_source"),
        "duration_seconds": duration,
        "duration_minutes": max(1, int(round(duration / 60))),
        "projects": plan.get("project_names") or [],
        "skills": plan.get("skill_names") or [],
        "questions_per_project": plan.get("questions_per_project") or 4,
        "opening_greeting": greeting,
    }


@app.post("/api/start-interview")
async def start_interview(payload: dict = None):
    """
    Initializes a new adaptive interview session bound to one resume_token.
    """
    payload = payload or {}
    role_override = payload.get("role")
    if role_override == "auto":
        role_override = None

    custom_features = payload.get("features")
    raw_text = payload.get("resume_text")
    job_description = (payload.get("job_description") or payload.get("jd") or "").strip()
    resume_token = payload.get("resume_token")

    if resume_token:
        profile = session_store.get_resume_profile(str(resume_token))
        if not profile:
            raise HTTPException(status_code=404, detail="Resume token was not found; upload the resume again")
        custom_features = custom_features or profile.get("features")
        raw_text = raw_text or profile.get("raw_text")

    if not custom_features:
        raise HTTPException(
            status_code=400,
            detail="Upload a resume and confirm the agenda before starting the interview",
        )

    if resume_token:
        custom_features = dict(custom_features)
        custom_features["resume_token"] = str(resume_token)

    session = _make_session(custom_features, role_override=role_override, raw_resume_text=raw_text, job_description=job_description)
    profile = session_store.get_resume_profile(str(resume_token)) if resume_token else None
    cached_plan = (profile or {}).get("interview_plan") or {}
    live_names = list((session.get("interview_plan") or {}).get("skill_names") or [])
    if (
        cached_plan.get("skill_questions_source") == "qwen"
        and list(cached_plan.get("skill_names") or []) == live_names
        and cached_plan.get("skill_slots")
    ):
        _apply_plan_to_session(session, cached_plan)
    else:
        filled = await _qwen_fill_skill_plan(
            session.get("interview_plan") or {},
            custom_features,
            None,
            raw_text or "",
        )
        _apply_plan_to_session(session, filled)
        if profile is not None and resume_token:
            profile["interview_plan"] = filled
            session_store.save_resume_profile(str(resume_token), profile)
    session_id = session["session_id"]
    if not session.get("started_at"):
        session["started_at"] = session.get("created_at") or time.time()
    session_store.save_session(session_id, session)
    _schedule_persist(session_id)
    jr = session.get("job_requirements") or {}
    _safe_log(
        f"[Start Interview] session={session_id} role={jr.get('title')} "
        f"intersection={jr.get('intersection')} gaps={jr.get('role_gaps')}"
    )
    cand = session["candidate"]
    clean_session = {
        "session_id": session_id,
        "created_at": session["created_at"],
        "started_at": session.get("started_at") or session["created_at"],
        "candidate": cand,
        "job_requirements": session["job_requirements"],
        "interview_state": session["interview_state"],
        "current_question": session["current_question"],
        "opening_greeting": session.get("opening_greeting"),
        "agenda": {
            "projects": cand.get("projects") or [],
            "skills": cand.get("skills") or [],
            "role_label": cand.get("target_role"),
            "duration_seconds": session["interview_state"].get("time_remaining_seconds"),
        },
    }
    return {
        "status": "ok",
        "session_id": session_id,
        "session": clean_session,
        "opening_greeting": session.get("opening_greeting"),
        "agenda": clean_session["agenda"],
    }


@app.get("/api/interview-history/{session_id}")
async def get_interview_history(session_id: str):
    """Full transcript. Reads Redis while hot, then falls back to durable storage."""
    from backend.store import interview_store

    session = session_store.get_session(session_id)
    if not session:
        session = await asyncio.to_thread(interview_store.get_interview, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    from backend.store import interview_timing

    timing = interview_timing(
        session.get("started_at") or session.get("created_at"),
        session.get("ended_at"),
        session.get("duration_seconds"),
    )
    return {
        "status": "ok",
        "session_id": session_id,
        "candidate": session.get("candidate", {}),
        "stage": session.get("interview_state", {}).get("stage"),
        "difficulty": session.get("interview_state", {}).get("difficulty_level"),
        "total_turns": len(session.get("history", [])),
        "history": session.get("history", []),
        **timing,
    }


@app.post("/api/end-interview")
async def end_interview(payload: dict):
    """
    Close the interview and queue the scorecard.

    Returns immediately (202-style): grading happens on the cold path so the
    candidate's browser never waits for the judge model.
    """
    session_id = (payload or {}).get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    session = load_live_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session was not found")

    finish = await _finish_interview(session)
    from backend.store import interview_timing

    timing = interview_timing(
        session.get("started_at"),
        session.get("ended_at"),
        session.get("duration_seconds"),
    )
    return {
        "status": "ok",
        "session_id": session_id,
        "grading": finish.get("grading"),
        "persisted": finish.get("persisted"),
        "turns": finish.get("turns"),
        "interview_complete": True,
        **timing,
    }


@app.get("/api/interview-report/{session_id}")
async def get_interview_report(session_id: str):
    """Fetch the scorecard. Poll until status is 'done'. Served from disk once Redis expires."""
    from backend.store import interview_store

    report = session_store.get_report(session_id)
    if not report or report.get("status") not in ("done", None):
        durable = await asyncio.to_thread(interview_store.get_report, session_id)
        if durable:
            report = durable
    if not report:
        raise HTTPException(status_code=404, detail="No report for this session; call /api/end-interview first")
    return {"status": "ok", "report": report, "grading": report.get("status", "done")}


@app.get("/api/interviews")
async def list_interviews(limit: int = 50, role: Optional[str] = None, recommendation: Optional[str] = None):
    """Recruiter listing of completed interviews, newest first."""
    from backend.store import interview_store

    rows = await asyncio.to_thread(
        interview_store.list_interviews, limit, role, recommendation
    )
    return {"status": "ok", "count": len(rows), "interviews": rows}


@app.post("/api/interview-turn")
async def api_interview_turn(payload: dict):
    """
    Core Turn Endpoint:
    Takes candidate answer + session_id.
    Builds structured JSON input required by the Qwen3-4B adaptive interviewer prompt.
    Sends to LLM and receives structured evaluation + next action + question JSON.
    Updates in-memory interview state and returns JSON response.
    """
    session_id = payload.get("session_id")
    candidate_answer = payload.get("candidate_answer") or payload.get("transcript") or ""
    role_val = payload.get("role")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required; start an interview first")

    session = load_live_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session was not found; start a new interview")

    state = session["interview_state"]
    if state.get("stage") == "completed" or session.get("ended_at"):
        finish = await _finish_interview(session)
        return {
            "status": "ok",
            "session_id": session_id,
            "interview_complete": True,
            "interview_state": state,
            "llm_response": {
                "question": "The interview is already complete. Thank you.",
                "question_mode": "TEMPLATE",
            },
            **finish,
        }

    # 1. Fast session-scoped STT correction (no LLM on the live path)
    candidate_answer_final = correct_transcript_fast(
        candidate_answer,
        candidate_dict=session.get("candidate") or {},
        hotwords=session.get("cv_hotwords") or [],
        phonetic_patterns=session.get("phonetic_patterns") or [],
        active_topic=_stt_active_topic(session),
    ) or candidate_answer

    # 2. Intent Engine, Evaluation & Question Engine
    from backend.intent_engine import detect_candidate_intent
    from backend.evaluator import evaluate_turn_answer
    from backend.interview_fsm import InterviewFSM
    from backend.question_engine import question_engine
    from rag_engine import question_bank_rag

    intent = detect_candidate_intent(
        candidate_answer_final,
        stage=state.get("stage"),
        questions_asked=state.get("questions_asked")
    )
    if intent != "REPEAT_REQUEST":
        _tick_session_clock(session)
        state = session["interview_state"]
    current_q_obj = session.get("current_question", {})
    eval_res = evaluate_turn_answer(
        candidate_answer_final,
        current_q_obj,
        intent,
        expected_concepts=current_q_obj.get("expected_concepts", [])
    )
    candidate_dict = dict(session["candidate"])
    fsm = InterviewFSM(state)
    fsm.update_from_evaluation(eval_res, intent, candidate_dict=candidate_dict)
    llm_anchor = await _resolve_followup_anchor_for_turn(
        session, candidate_answer_final, fsm.get_dict(), intent
    )
    q_decision = question_engine.decide(
        candidate_answer=candidate_answer_final,
        intent=intent,
        fsm_state=fsm.get_dict(),
        eval_res=eval_res,
        candidate_dict=candidate_dict,
        question_bank_rag=question_bank_rag,
        exclude_questions=fsm.get_dict().get("questions_already_asked", []),
        interview_plan=session.get("interview_plan"),
        llm_anchor=llm_anchor,
    )

    fsm.set_current_topic(q_decision.target_topic)
    _persist_followup_thread(fsm, q_decision)
    session["interview_state"] = fsm.get_dict()
    state = session["interview_state"]

    from backend.prompt_builder import build_interviewer_prompt
    last_q_text = current_q_obj.get("question", "")
    session_rag_instance = session.get("session_rag")
    resume_chunks = []
    phrase = q_decision.wants_qwen_phrasing(intent)
    if phrase and session_rag_instance is not None:
        resume_chunks = session_rag_instance.retrieve(
            f"{candidate_answer_final} {q_decision.target_topic}".strip(),
            top_k=2,
            session_id=session_id
        )

    generated_question = _spoken_from_decision(q_decision, candidate_dict)
    llm_payload = None
    if phrase:
        llm_payload = build_interviewer_prompt(
            fsm_state=fsm.get_dict(),
            candidate_dict=candidate_dict,
            candidate_answer=candidate_answer_final,
            question_bank_context=q_decision.probes,
            last_question=last_q_text,
            intent=intent,
            resume_chunks=resume_chunks,
            question_decision=q_decision
        )
        generated_question = await _phrase_with_qwen(
            q_decision, intent, generated_question, llm_payload
        )

    llm_response_json = {
        "action": "ASK_FOLLOWUP",
        "question": generated_question,
        "question_mode": q_decision.mode,
        "evaluation": {
            "correctness": round(eval_res.get("score", 0.6) * 10, 1),
            "depth": 7 if eval_res.get("depth") == "high" else 5,
            "reasoning": 7,
            "practical_application": 7,
            "communication": 8,
            "overall": round(eval_res.get("score", 0.6) * 10, 1),
        },
        "answer_analysis": {
            "demonstrated_concepts": eval_res.get("concepts_covered", []),
            "missing_concepts": eval_res.get("missing_concepts", []),
            "strengths": [],
            "weaknesses": eval_res.get("missing_concepts", []),
        },
        "next_difficulty": state.get("difficulty_level", 3),
        "topic": q_decision.target_topic,
        "confidence": 0.9
    }

    # 6. Update interview state
    next_diff = llm_response_json.get("next_difficulty", state.get("difficulty_level", state.get("difficulty", 2)))
    topic = llm_response_json.get("topic", state["current_topic"])
    next_question = llm_response_json.get("question", "")
    overall_score = llm_response_json.get("evaluation", {}).get("overall", 5)

    state["difficulty"] = next_diff
    state["difficulty_level"] = next_diff
    state["current_topic"] = topic
    if topic and topic not in state["topics_covered"]:
        state["topics_covered"].append(topic)
    canonical = (next_question or q_decision.seed_question or "").strip()
    if intent != "REPEAT_REQUEST" and canonical:
        if canonical not in state["questions_already_asked"]:
            state["questions_already_asked"].append(canonical)
        if q_decision.seed_question:
            seed = q_decision.seed_question.strip()
            if seed and seed not in state["questions_already_asked"]:
                state["questions_already_asked"].append(seed)
        state["last_canonical_question"] = canonical
    state["topic_scores"][topic] = overall_score

    if state["questions_remaining"] <= 0 or state["time_remaining_seconds"] <= 0:
        if state.get("stage") not in ("closing", "completed"):
            state["stage"] = "closing"

    session["current_question"] = _build_current_question(q_decision, next_question, topic, next_diff)

    # History pairs the answer with the question it answered, plus that
    # question's rubric, so the async grader has something to judge against.
    _append_turn(session, current_q_obj, candidate_answer_final, next_question, eval_res)

    # Persist updated session state to Redis / in-memory store
    session_store.save_session(session_id, session)

    finish = {}
    if _should_complete_after_turn(state, q_decision):
        finish = await _finish_interview(session)
        state = session["interview_state"]

    return {
        "status": "ok",
        "session_id": session_id,
        "corrected_transcript": candidate_answer_final,
        "llm_response": llm_response_json,
        "interview_state": state,
        "interview_complete": bool(finish.get("interview_complete")),
        "time_remaining_seconds": state.get("time_remaining_seconds"),
        **({k: v for k, v in finish.items() if k != "interview_complete"} if finish else {}),
    }



def clean_interviewer_speech(text: str) -> str:
    """
    Cleans raw LLM interviewer output to ensure pure natural spoken text.
    Strips accidental JSON wrappers, markdown code blocks, and quote artifacts.
    """
    if not text:
        return ""
    t = text.strip()
    if t.startswith("{") and ("}" in t or '"question"' in t):
        try:
            parsed = json.loads(t)
            if isinstance(parsed, dict) and "question" in parsed:
                return str(parsed["question"]).strip()
        except Exception:
            m = re.search(r'"question"\s*:\s*"([^"]+)"', t)
            if m:
                return m.group(1).strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t).strip()
    t = re.sub(r'^\{\s*"question"\s*:\s*"?', '', t, flags=re.IGNORECASE)
    t = re.sub(r'"\s*,\s*"action"\s*:.*$', '', t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r'"\s*\}$', '', t)
    return t.strip().strip('"')


@app.post("/api/interview-turn-stream")
async def api_interview_turn_stream(payload: dict):
    """
    Streaming Server-Sent Events (SSE) endpoint for interview turns.
    Yields sentence chunks in real-time as LLM tokens generate for sub-200ms TTFA playback.
    """
    candidate_answer = payload.get("transcript", "").strip()
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required; start an interview first")

    session = load_live_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session was not found; start a new interview")

    state = session["interview_state"]
    if state.get("stage") == "completed" or session.get("ended_at"):
        finish = await _finish_interview(session)
        already = "The interview is already complete. Thank you."

        async def _already_done():
            yield f"event: sentence_chunk\ndata: {json.dumps({'text': already})}\n\n"
            yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'full_question': already, 'question_mode': 'TEMPLATE', 'interview_complete': True, 'stage': 'completed', 'time_remaining_seconds': 0})}\n\n"

        return StreamingResponse(_already_done(), media_type="text/event-stream")

    from backend.intent_engine import detect_candidate_intent
    from backend.evaluator import evaluate_turn_answer
    from backend.interview_fsm import InterviewFSM
    from backend.prompt_builder import build_interviewer_prompt

    candidate_answer_final = correct_transcript_fast(
        candidate_answer,
        candidate_dict=session.get("candidate") or {},
        hotwords=session.get("cv_hotwords") or [],
        phonetic_patterns=session.get("phonetic_patterns") or [],
        active_topic=_stt_active_topic(session),
    ) or candidate_answer

    # 2. Intent Detection & Turn Answer Evaluation
    intent = detect_candidate_intent(
        candidate_answer_final,
        stage=state.get("stage"),
        questions_asked=state.get("questions_asked")
    )
    if intent != "REPEAT_REQUEST":
        _tick_session_clock(session)
        state = session["interview_state"]
    current_q_obj = session.get("current_question", {})
    eval_res = evaluate_turn_answer(
        candidate_answer_final,
        current_q_obj,
        intent,
        expected_concepts=current_q_obj.get("expected_concepts", [])
    )

    from backend.question_engine import question_engine
    from rag_engine import question_bank_rag

    candidate_dict = dict(session["candidate"])
    fsm = InterviewFSM(state)
    updated_state = fsm.update_from_evaluation(eval_res, intent, candidate_dict=candidate_dict)
    session["interview_state"] = updated_state
    llm_anchor = await _resolve_followup_anchor_for_turn(
        session, candidate_answer_final, fsm.get_dict(), intent
    )
    q_decision = question_engine.decide(
        candidate_answer=candidate_answer_final,
        intent=intent,
        fsm_state=fsm.get_dict(),
        eval_res=eval_res,
        candidate_dict=candidate_dict,
        question_bank_rag=question_bank_rag,
        exclude_questions=updated_state.get("questions_already_asked", []),
        interview_plan=session.get("interview_plan"),
        llm_anchor=llm_anchor,
    )

    fsm.set_current_topic(q_decision.target_topic)
    _persist_followup_thread(fsm, q_decision)
    active_target = q_decision.target_topic or ""
    rag_query = f"{candidate_answer_final} {active_target}".strip()
    session_rag_instance = session.get("session_rag")
    resume_chunks = []
    phrase = q_decision.wants_qwen_phrasing(intent)
    if phrase and session_rag_instance is not None:
        resume_chunks = session_rag_instance.retrieve(rag_query, top_k=2)
        if not resume_chunks:
            resume_chunks = session_rag_instance.retrieve(" ".join(candidate_dict.get("projects", [])[:2]), top_k=2)
    last_q_text = current_q_obj.get("question", "")

    llm_payload = None
    if phrase:
        llm_payload = build_interviewer_prompt(
            fsm_state=fsm.get_dict(),
            candidate_dict=candidate_dict,
            candidate_answer=candidate_answer_final,
            question_bank_context=q_decision.probes,
            last_question=last_q_text,
            intent=intent,
            resume_chunks=resume_chunks,
            question_decision=q_decision
        )
        llm_payload["stream"] = True

    async def event_generator():
        yield f"event: corrected_transcript\ndata: {json.dumps({'text': candidate_answer_final})}\n\n"

        retrieved = _spoken_from_decision(q_decision, candidate_dict)
        spoken_parts: list[str] = []

        async for sentence in _iter_phrased_sentences(q_decision, intent, retrieved, llm_payload):
            text = (sentence or "").strip()
            if not text:
                continue
            spoken_parts.append(text)
            yield f"event: sentence_chunk\ndata: {json.dumps({'text': text})}\n\n"

        cleaned_final_question = " ".join(spoken_parts).strip()
        if not cleaned_final_question:
            cleaned_final_question = retrieved or "You mentioned that approach — what failed first when it didn't work?"
            yield f"event: sentence_chunk\ndata: {json.dumps({'text': cleaned_final_question})}\n\n"

        # Event 3: Done & Log Question Deduplication & Session state update
        canonical = (cleaned_final_question or q_decision.seed_question or "").strip()
        if intent != "REPEAT_REQUEST":
            fsm.add_asked_question(canonical)
            if q_decision.seed_question and q_decision.seed_question.strip() != canonical:
                fsm.add_asked_question(q_decision.seed_question.strip())
        session["interview_state"] = fsm.get_dict()
        _append_turn(session, current_q_obj, candidate_answer_final, cleaned_final_question, eval_res)
        session["current_question"] = _build_current_question(
            q_decision, cleaned_final_question, q_decision.target_topic, q_decision.target_difficulty
        )

        # Persist updated session state to Redis / in-memory store
        session_store.save_session(session_id, session)

        finish = {}
        if _should_complete_after_turn(session["interview_state"], q_decision):
            finish = await _finish_interview(session)

        done_payload = {
            "session_id": session_id,
            "full_question": cleaned_final_question,
            "question_mode": q_decision.mode,
            "stage": session["interview_state"].get("stage"),
            "time_remaining_seconds": session["interview_state"].get("time_remaining_seconds"),
            "interview_complete": bool(finish.get("interview_complete")),
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"


    return StreamingResponse(event_generator(), media_type="text/event-stream")



@app.post("/api/system-prompt")
async def save_system_prompt(payload: dict):
    new_prompt = payload.get("prompt", "")
    prompt_file = BASE_DIR / "system_prompt.txt"
    prompt_file.write_text(new_prompt, encoding="utf-8")
    return {"status": "ok", "message": "System prompt saved"}


@app.get("/api/candidate-profile")
async def get_candidate_profile(resume_token: str | None = None, session_id: str | None = None):
    """Returns this browser's resume — never another candidate's global cache."""
    features: dict = {}
    hotwords: list = []
    if resume_token:
        profile = session_store.get_resume_profile(str(resume_token))
        if profile:
            features = profile.get("features") or {}
            hotwords = build_cv_hotwords(features)
    elif session_id:
        session = load_live_session(session_id)
        if session:
            features = session.get("candidate") or {}
            hotwords = session.get("cv_hotwords") or []
    return {
        "status": "ok",
        "features": features,
        "hotwords": hotwords[:25],
    }



@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(None), text_content: str = Form(None)):
    """Uploads resume (PDF, TXT, MD) or pasted text and extracts interview features."""
    try:
        raw_text = ""
        filename = "resume.txt"
        if file is not None and getattr(file, "filename", None):
            filename = file.filename or "resume.txt"
            content = await file.read()
            max_bytes = max(1, _app_settings.interview.max_upload_mb) * 1024 * 1024
            if len(content) > max_bytes:
                return {"status": "error", "message": f"Resume file exceeds { _app_settings.interview.max_upload_mb } MB"}
            if not content:
                return {"status": "error", "message": "Resume file was empty"}
            raw_text = extract_text_from_file(content, filename)
        elif text_content:
            raw_text = normalize_resume_text(text_content)
            filename = "pasted_resume.txt"
        else:
            return {"status": "error", "message": "No resume file or text content provided"}

        if not raw_text.strip():
            return {"status": "error", "message": "Resume file contained no readable text. Try a text-based PDF or .txt/.md file."}

        features = await asyncio.to_thread(extract_resume_features_llm, raw_text)
        features["raw_text"] = raw_text[:6000]

        resume_token = str(uuid.uuid4())
        from rag_engine import question_bank_rag as _qbank
        from backend.interview_plan import build_interview_plan
        if _qbank and not getattr(_qbank, "is_ready", False):
            try:
                _qbank.load()
            except Exception:
                pass
        sift = sift_skills_for_role(features)
        plan = build_interview_plan(
            {
                "name": features.get("name") or "",
                "skills": list(sift.get("intersection") or features.get("skills") or []),
                "projects": list(sift.get("interview_projects") or features.get("projects") or []),
                "resume_text": raw_text,
                "bank_track": sift["bank_track"],
                "target_track": sift["track"],
                "interview_style": sift["style"],
            },
            question_bank_rag=_qbank if getattr(_qbank, "is_ready", False) else None,
        )
        plan = await _qwen_fill_skill_plan(plan, features, sift, raw_text)
        session_store.save_resume_profile(resume_token, {
            "features": features,
            "raw_text": raw_text,
            "filename": filename,
            "interview_plan": plan,
        })

        new_prompt = generate_system_prompt_from_features(features)
        _safe_log(f"[Upload Resume] token={resume_token[:8]}… projects={len(features.get('projects') or [])} skills={len(features.get('skills') or [])}")

        try:
            raw_resume_for_rag = raw_text if raw_text.strip() else ""
            await asyncio.to_thread(resume_rag.index, raw_resume_for_rag)
        except Exception as exc:
            _safe_log(f"[Upload Resume] RAG index skipped: {exc}")

        return {
            "status": "ok",
            "message": "Resume features extracted successfully",
            "filename": filename,
            "extracted_features": features,
            "prompt": new_prompt,
            "hotword_count": len(build_cv_hotwords(features)),
            "resume_token": resume_token,
        }

    except Exception as exc:
        _safe_log(f"[Upload Resume Endpoint Error] {exc}")
        return {"status": "error", "message": f"Resume processing error: {exc}"}


DEFAULT_KOKORO_VOICE = "af_heart"


@app.post("/api/tts")
async def text_to_speech(payload: dict):
    """Kokoro only. Never falls back to a browser voice."""
    text = payload.get("text", "").strip()
    if not text:
        return Response(content=b"", media_type="audio/wav", status_code=400)

    from interviewer.config import settings as speech_settings

    voice = payload.get("voice", payload.get("description", speech_settings.speech.tts_voice or DEFAULT_KOKORO_VOICE))
    if not isinstance(voice, str) or not voice.strip() or len(voice) > 50:
        voice = DEFAULT_KOKORO_VOICE
    speed = float(payload.get("speed", 1.0))
    url = speech_settings.speech.tts_url or TTS_SYNTHESIZE_URL
    timeout_s = max(8.0, speech_settings.speech.tts_deadline_ms / 1000.0)
    last_error = ""

    def fetch_tts():
        resp = requests.post(
            url,
            json={"text": text, "voice": voice, "speed": speed},
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Interview-Kokoro/1.0",
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.content

    for attempt in range(3):
        try:
            audio_bytes = await asyncio.to_thread(fetch_tts)
            if audio_bytes and len(audio_bytes) > 64:
                return Response(content=audio_bytes, media_type="audio/wav")
            last_error = "empty audio"
        except Exception as exc:
            last_error = str(exc)
            await asyncio.sleep(0.35 * (attempt + 1))

    print(f"[Kokoro TTS] failed after retries: {last_error}")
    return Response(content=b"", media_type="audio/wav", status_code=502)


async def _authorize_interview_ws(client_ws: WebSocket) -> bool:
    required = (_app_settings.interview.api_key or "").strip()
    if not required:
        return True
    provided = (
        (client_ws.query_params.get("api_key") or "")
        or (client_ws.headers.get("x-api-key") or "")
    ).strip()
    if provided != required:
        await client_ws.close(code=4401)
        return False
    return True


def _apply_stt_lexicon(text: str, mode: str, live_session: dict | None) -> str:
    if not text:
        return text
    if mode != "interview":
        return text
    if live_session:
        return correct_transcript_fast(
            text,
            candidate_dict=live_session.get("candidate") or {},
            hotwords=live_session.get("cv_hotwords") or [],
            phonetic_patterns=live_session.get("phonetic_patterns") or [],
            active_topic=_stt_active_topic(live_session),
        ) or text
    return clean_transcript(text)


async def _whisper_live_interview(client_ws: WebSocket, mode: str, session_id: Optional[str]) -> None:
    """Browser PCM16 → Whisper large-v3-turbo. No VAD; the client gates the mic."""
    if not await _authorize_interview_ws(client_ws):
        return
    await client_ws.accept()
    live_session = load_live_session(session_id) if session_id else None
    prompt_bits = []
    if live_session:
        cand = live_session.get("candidate") or {}
        from interviewer.services.resume import is_resume_metadata_title
        for proj in cand.get("projects") or []:
            title = str(proj or "").strip()
            if title and len(title.split()) <= 8 and not is_resume_metadata_title(title):
                prompt_bits.append(title)
        prompt_bits.extend(list(cand.get("skills") or [])[:8])
    from interviewer.services.whisper_stt import WhisperStreamSession, get_whisper_engine

    try:
        await asyncio.to_thread(get_whisper_engine)
    except Exception as exc:
        await client_ws.send_json({"type": "error", "message": f"Whisper STT failed to load: {exc}"})
        await client_ws.close()
        return

    stream = WhisperStreamSession(prompt=" ".join(str(p) for p in prompt_bits if p)[:220])
    stop = asyncio.Event()

    async def caption_loop():
        from interviewer.services.whisper_stt import uses_whisper_api
        tick = 3.2 if uses_whisper_api() else 0.7
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=tick)
                break
            except asyncio.TimeoutError:
                pass
            if not stream.has_enough():
                continue
            try:
                raw = await stream.transcribe_latest()
                text = _apply_stt_lexicon(raw, mode, live_session)
                if text:
                    await client_ws.send_json({"type": "transcript", "text": text})
            except Exception as exc:
                print(f"[Whisper STT] caption error: {exc}")
                try:
                    await client_ws.send_json({"type": "error", "message": "Captions delayed — keep speaking"})
                except Exception:
                    pass

    pump = asyncio.create_task(caption_loop())
    try:
        while True:
            data = await client_ws.receive()
            msg_type = data.get("type", "")
            if msg_type == "websocket.disconnect":
                break
            if "bytes" in data and data["bytes"]:
                stream.add_pcm(data["bytes"])
            elif "text" in data and data["text"]:
                try:
                    payload = json.loads(data["text"])
                except Exception:
                    continue
                action = payload.get("action")
                if action == "reset":
                    stream.reset()
                    await client_ws.send_json({"type": "reset_ack"})
                elif action == "flush":
                    raw = await stream.transcribe_latest(force=True)
                    text = _apply_stt_lexicon(raw, mode, live_session)
                    if text:
                        await client_ws.send_json({"type": "transcript", "text": text})
                elif action == "stop":
                    break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:
        print(f"[Whisper STT] {exc}")
        try:
            await client_ws.send_json({"type": "error", "message": f"STT error: {exc}"})
        except Exception:
            pass
    finally:
        stop.set()
        pump.cancel()
        try:
            await client_ws.close()
        except Exception:
            pass


@app.websocket("/ws/live-interview")
async def websocket_live_interview(client_ws: WebSocket, mode: str = "interview", session_id: Optional[str] = None):
    """Live mic PCM16 → Whisper captions. Nemotron is not used."""
    if not await _authorize_interview_ws(client_ws):
        return
    await _whisper_live_interview(client_ws, mode, session_id)


if __name__ == "__main__":
    print("=" * 70)
    print("LIVE VOICE AI INTERVIEW SERVER RUNNING")
    print("=" * 70)
    print("Open your browser at: http://localhost:8000")
    print("=" * 70)
    from interviewer.config import settings as run_cfg
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=bool(run_cfg.interview.uvicorn_reload),
        reload_dirs=[str(BASE_DIR / "backend"), str(BASE_DIR / "static"), str(BASE_DIR / "interviewer")],
        reload_includes=["app.py", "rag_engine.py"],
    )


"""
Follow-up anchors from THIS resume + Qwen, not a global tech word list.

Any CV works: MoleCheck, Ferrite Mesh, a campus hiring drive. The dictionary
is the candidate's own skills/projects/raw text. If they only said "yeah I
used this", Qwen maps that onto the last real term or the current project.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from interviewer.ports.llm import LLMRequest, LLMTask, Message

CAMELISH = re.compile(r"[A-Z][a-zA-Z]*[A-Z][A-Za-z0-9]*|[A-Za-z]+(?:v|V)\d+")
# Spoken English — not a per-product catalog. Yeah/Sorry never become tools.
ENGLISH_FUNCTION = frozenset({
    "the", "and", "with", "from", "that", "this", "were", "was", "using", "used",
    "into", "then", "have", "been", "they", "them", "for", "our", "you", "your",
    "about", "just", "like", "also", "very", "some", "when", "what", "which",
    "it", "those", "yeah", "yea", "yep", "yup", "yes", "nah", "nope", "ok",
    "okay", "okey", "sure", "right", "well", "so", "um", "uh", "ah", "oh",
    "hmm", "huh", "sorry", "thanks", "thank", "please", "actually", "basically",
    "literally", "weather", "whether", "hello", "hi", "hey", "alright", "fine",
})


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _blocked_labels() -> set[str]:
    try:
        from interviewer.services.resume import INVALID_FIELD_WORDS
        return set(INVALID_FIELD_WORDS)
    except Exception:
        return set()


def _term_in_answer(term: str, answer: str) -> bool:
    """Whole-token match so 'Java' does not hit 'JavaScript'."""
    token = str(term or "").strip()
    if not token or not answer:
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
            answer,
            flags=re.IGNORECASE,
        )
    )


def headers_from_resume(raw: str) -> set[str]:
    """Section titles that appear on this resume (Skills:, CERTIFICATIONS, ...)."""
    found: set[str] = set()
    for line in (raw or "").splitlines():
        t = re.sub(r"^[\s\-\*•]+", "", line).strip()
        if not t or len(t) > 48:
            continue
        words = t.rstrip(":").split()
        if not (1 <= len(words) <= 4):
            continue
        if t.isupper() or t.endswith(":"):
            found.add(t.rstrip(":").strip().lower())
    return found


def cv_terms(candidate: dict[str, Any] | None) -> list[str]:
    """Proper nouns for THIS candidate only."""
    cand = candidate or {}
    found: list[str] = []
    seen: set[str] = set()
    headers = headers_from_resume(str(cand.get("raw_text") or cand.get("resume_context") or ""))
    blocked = headers | _blocked_labels()

    def add(raw: Any) -> None:
        text = re.sub(r"\s+", " ", str(raw or "")).strip(" -–—,")
        if not text or len(text) < 3:
            return
        key = text.lower()
        if key in seen or key in blocked or key in ENGLISH_FUNCTION:
            return
        seen.add(key)
        found.append(text)

    for key in ("name", "college", "company"):
        add(cand.get(key))
    for key in ("skills", "cv_skills", "projects", "certifications", "domains"):
        val = cand.get(key) or []
        if isinstance(val, str):
            val = [val]
        for item in val:
            add(item)

    try:
        from interviewer.services.stt_lexicon import collect_cv_terms
        for term in collect_cv_terms(cand):
            add(term)
    except Exception:
        pass
    return found


def resume_mentions(
    answer: str,
    candidate: dict[str, Any] | None,
    extra: Iterable[str] | None = None,
    limit: int = 5,
) -> list[str]:
    """CV terms that actually appear in what they said. Longest match first."""
    if not answer:
        return []
    terms = list(cv_terms(candidate))
    for item in extra or ():
        t = str(item).strip()
        if t and t.lower() not in {x.lower() for x in terms} and t.lower() not in ENGLISH_FUNCTION:
            terms.append(t)
    terms.sort(key=lambda t: len(t), reverse=True)
    found: list[str] = []
    seen: set[str] = set()
    blocked = _blocked_labels()
    for term in terms:
        key = term.lower()
        if key in seen or key in ENGLISH_FUNCTION or key in blocked:
            continue
        if _term_in_answer(term, answer):
            seen.add(key)
            found.append(term)
        if len(found) >= limit:
            break
    return found


def accept_anchor(term: str, candidate: dict[str, Any] | None, fallbacks: Iterable[str]) -> str:
    raw = str(term or "").strip().strip("\"'")
    if len(raw) < 3 or raw.lower() in ENGLISH_FUNCTION:
        return ""
    headers = headers_from_resume(str((candidate or {}).get("raw_text") or (candidate or {}).get("resume_context") or ""))
    if raw.lower() in headers or raw.lower() in _blocked_labels():
        return ""
    allowed = {_fold(t) for t in list(cv_terms(candidate)) + [str(x) for x in fallbacks if x]}
    if _fold(raw) in allowed:
        return raw
    # Prefer the canonical CV spelling when the model returned a close variant.
    for t in cv_terms(candidate):
        if _fold(t) == _fold(raw) or raw.lower() in t.lower() or t.lower() in raw.lower():
            return t
    return ""


def _parse_anchor_payload(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return str(data.get("anchor") or data.get("term") or "").strip()
        except json.JSONDecodeError:
            pass
    return raw.split("\n")[0].strip().strip("\"'")


async def resolve_followup_anchor(
    answer: str,
    candidate: dict[str, Any] | None,
    last_question: str = "",
    last_anchor: str = "",
    project: str = "",
) -> str:
    """
    One grounded term for the next follow-up.

    Fast path: a resume term they actually named.
    Slow path: Qwen maps vague speech ("yeah I used this") onto that resume.
    """
    fallbacks = [x for x in (last_anchor, project) if x]
    grounded = resume_mentions(answer, candidate, extra=fallbacks)
    # A CV hit that isn't only the project name is enough — no model call.
    project_fold = _fold(project)
    specific = [t for t in grounded if _fold(t) != project_fold]
    if specific:
        return specific[0]
    if grounded and not (last_anchor and _fold(last_anchor) != project_fold):
        return grounded[0]

    terms = cv_terms(candidate)[:24]
    if last_anchor and last_anchor not in terms:
        terms = [last_anchor] + terms
    if project and project not in terms:
        terms = [project] + terms

    prompt = (
        "Pick the ONE thing this candidate referred to. Return JSON {\"anchor\": \"...\"} only.\n"
        "Rules:\n"
        "- Prefer a name from RESUME_TERMS.\n"
        "- Yeah/yes/ok/sorry/well and pronouns (this/that/it) are not names.\n"
        "- Resume section titles (Skills, Certifications, Education) are not names.\n"
        "- If they only agreed or said 'I used this', return LAST_ANCHOR or PROJECT.\n"
        "- Never invent a tool that is not in RESUME_TERMS, LAST_ANCHOR, or PROJECT.\n"
        f"PROJECT: {project or 'unknown'}\n"
        f"LAST_ANCHOR: {last_anchor or 'none'}\n"
        f"LAST_QUESTION: {(last_question or '')[:300]}\n"
        f"RESUME_TERMS: {', '.join(terms) or 'none'}\n"
        f"ANSWER: {(answer or '')[:600]}"
    )

    from interviewer.adapters.registry import get_llm
    from interviewer.ports.llm import LLMFailure, LLMResult

    try:
        result = await get_llm().complete(
            LLMRequest(
                task=LLMTask.PHRASING,
                messages=[
                    Message(role="system", content="You extract one resume-grounded interview term. JSON only."),
                    Message(role="user", content=prompt),
                ],
                temperature=0.0,
                max_tokens=40,
                deadline_ms=900,
            )
        )
    except Exception:
        result = LLMResult.failed(LLMFailure.TRANSPORT)

    if result.ok:
        picked = accept_anchor(_parse_anchor_payload(result.text), candidate, fallbacks)
        if picked:
            return picked
    if last_anchor:
        return last_anchor
    if project:
        return project
    return grounded[0] if grounded else ""

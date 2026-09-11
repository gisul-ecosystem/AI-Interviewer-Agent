"""
DEPRECATED compatibility shim -- delete once Stage 5 lands.

The TF-IDF retrieval layer is gone. Two things replaced it:

  * Question lookup  -> interviewer.services.question_bank (metadata filtering)
  * Resume grounding -> the whole resume goes into the prompt (Stage 3)

This module only exists so the not-yet-migrated ``app.py`` and
``backend/*`` modules keep importing successfully during the migration.
New code must import from ``interviewer.*`` directly.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Tuple

from interviewer.domain.models import BankQuery
from interviewer.services.question_bank import get_question_bank

__all__ = [
    "question_bank_rag",
    "resume_rag",
    "ResumeRAG",
    "init_rag",
    "build_rag_context",
]


class _QuestionBankShim:
    """Old ``question_bank_rag`` surface, backed by metadata filtering."""

    @property
    def is_ready(self) -> bool:
        return get_question_bank().is_loaded

    def load(self, path: Any = None) -> int:
        return get_question_bank().load(path)

    def get_record(self, question_id: str) -> dict | None:
        question = get_question_bank().get(question_id)
        return question.as_dict() if question else None

    def retrieve_scored_records(
        self,
        query: str,
        top_k: int = 5,
        exclude_questions: List[str] | None = None,
        track: str | None = None,
        target_difficulty: int | None = None,
    ) -> List[dict]:
        """
        Metadata selection behind the old signature.

        ``query`` is treated as a topic hint, and a hint that matches nothing
        returns [] -- a real bank miss, so callers fall back to a grounded
        template instead of speaking a weakly related question.
        """
        hint = (query or "").strip()
        questions = get_question_bank().select(
            BankQuery(
                track=(track or "general_cs"),
                difficulty=target_difficulty if target_difficulty is not None else 2,
                hint=hint,
                exclude_texts=frozenset(exclude_questions or ()),
                require_hint_match=bool(hint),
                limit=top_k,
            )
        )
        records = []
        for rank, question in enumerate(questions):
            record = question.as_dict()
            # Legacy callers threshold on this. Metadata matches are exact by
            # construction, so the value only preserves ordering.
            record["similarity_score"] = round(max(0.5, 1.0 - 0.05 * rank), 3)
            records.append(record)
        return records

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        return [(r["topic"], r["question"]) for r in self.retrieve_scored_records(query, top_k=top_k)]


class ResumeRAG:
    """
    No-op stand-in. Resume chunking and retrieval were removed: resumes are
    short enough to pass whole into the prompt, which is both faster and
    strictly more faithful than top-k chunks.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._text = ""

    def index(self, resume_text: str, session_id: str = "default_session") -> int:
        self._text = resume_text or ""
        return 0

    def retrieve(self, query: str, top_k: int = 2, session_id: str | None = None) -> List[str]:
        text = (self._text or "").strip()
        if not text:
            return []
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        if not paras:
            return [text[:400]]
        q_tokens = {t.lower() for t in (query or "").split() if len(t) > 3}
        scored: list[tuple[int, str]] = []
        for para in paras:
            hits = sum(1 for t in q_tokens if t in para.lower()) if q_tokens else 0
            scored.append((hits, para))
        scored.sort(key=lambda item: item[0], reverse=True)
        out = [p for hits, p in scored if hits > 0][:top_k]
        if not out:
            out = paras[:top_k]
        return out

    @property
    def is_ready(self) -> bool:
        return bool(self._text)


question_bank_rag = _QuestionBankShim()
resume_rag = ResumeRAG()


def init_rag(resume_text: str | None = None) -> None:
    warnings.warn(
        "rag_engine is a migration shim; import interviewer.services.question_bank instead",
        DeprecationWarning,
        stacklevel=2,
    )
    get_question_bank()


def build_rag_context(candidate_answer: str) -> str:
    """Always empty. Resume context is injected whole by the prompt builder."""
    return ""

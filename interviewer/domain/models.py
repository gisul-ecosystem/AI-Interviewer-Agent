"""
Core domain types.

Pure data. No I/O, no framework imports, no LLM. Everything else in the system
depends on this module; this module depends on nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tokens that carry no discriminating signal when matching a topic hint.
_HINT_STOPWORDS = frozenset({"the", "and", "of", "for", "with", "a", "an", "js", "dev"})


def _stem(token: str) -> str:
    """Fold a trailing plural so 'llms' matches 'llm'. Leaves 'class', 'process'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> frozenset[str]:
    """
    Lowercase, plural-folded word tokens used for topic matching.

    Token sets -- never substrings. 'java' must not match 'javascript', which
    is exactly the bug substring matching produced.
    """
    if not text:
        return frozenset()
    tokens = _TOKEN_RE.findall(str(text).lower())
    return frozenset(_stem(t) for t in tokens if len(t) > 1 and t not in _HINT_STOPWORDS)


class Track(StrEnum):
    """
    StrEnum, not ``(str, Enum)``: the latter makes ``str(Track.HR)`` return
    'Track.HR', which silently broke track filtering and let a DSA question
    into an HR interview.
    """

    AIML = "aiml"
    WEBD = "webd"
    FRONTEND = "frontend"
    BACKEND = "backend"
    HR = "hr"
    GENERAL_CS = "general_cs"
    SOFTWARE_ENGINEERING = "software_engineering"

    @classmethod
    def parse(cls, raw: Any, default: "Track | None" = None) -> "Track":
        try:
            return cls(str(raw or "").strip().lower())
        except ValueError:
            return default or cls.GENERAL_CS


# Which banks a track may borrow from, in preference order handled by scoring.
# HR is deliberately closed: a behavioural interview must never surface a
# coding question.
TRACK_FAMILY: Mapping[str, frozenset[str]] = {
    Track.AIML: frozenset({"aiml", "general_cs", "software_engineering"}),
    Track.WEBD: frozenset({"webd", "frontend", "backend", "general_cs", "software_engineering"}),
    Track.FRONTEND: frozenset({"frontend", "webd", "general_cs"}),
    Track.BACKEND: frozenset({"backend", "webd", "general_cs", "software_engineering"}),
    Track.HR: frozenset({"hr"}),
    Track.GENERAL_CS: frozenset({"general_cs", "software_engineering"}),
    Track.SOFTWARE_ENGINEERING: frozenset({"software_engineering", "general_cs"}),
}

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 3


def clamp_difficulty(value: Any, default: int = 2) -> int:
    try:
        return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, int(value)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Rubric:
    """How an answer to one question is judged. Travels with the question."""

    expected_concepts: tuple[str, ...] = ()
    strong_signals: tuple[str, ...] = ()
    weak_signals: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.expected_concepts or self.strong_signals or self.weak_signals)


@dataclass(frozen=True)
class Question:
    """A curated bank question. Immutable and always carries its rubric."""

    id: str
    text: str
    tracks: frozenset[str]
    competency: str
    topic: str
    difficulty: int
    rubric: Rubric = field(default_factory=Rubric)
    follow_up_ids: tuple[str, ...] = ()
    # Precomputed so selection stays allocation-free in the hot path.
    match_tokens: frozenset[str] = field(default_factory=frozenset)

    @staticmethod
    def from_record(record: Mapping[str, Any]) -> "Question | None":
        text = str(record.get("question") or "").strip()
        qid = str(record.get("id") or "").strip()
        if not text or not qid:
            return None

        competency = str(record.get("competency") or "general").strip()
        topic = str(record.get("topic") or competency).strip()
        tracks = frozenset(
            str(t).strip().lower() for t in (record.get("tracks") or ()) if str(t).strip()
        )

        return Question(
            id=qid,
            text=text,
            tracks=tracks,
            competency=competency,
            topic=topic,
            difficulty=clamp_difficulty(record.get("difficulty"), default=2),
            rubric=Rubric(
                expected_concepts=tuple(str(c) for c in (record.get("expected_concepts") or ()) if str(c).strip()),
                strong_signals=tuple(str(c) for c in (record.get("strong_signals") or ()) if str(c).strip()),
                weak_signals=tuple(str(c) for c in (record.get("weak_signals") or ()) if str(c).strip()),
            ),
            follow_up_ids=tuple(str(f) for f in (record.get("follow_up_ids") or ()) if str(f).strip()),
            match_tokens=tokenize(f"{topic} {competency}"),
        )

    def as_dict(self) -> dict[str, Any]:
        """Legacy-shaped dict for modules not yet migrated off the old bank."""
        return {
            "id": self.id,
            "question": self.text,
            "tracks": sorted(self.tracks),
            "competency": self.competency,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "expected_concepts": list(self.rubric.expected_concepts),
            "strong_signals": list(self.rubric.strong_signals),
            "weak_signals": list(self.rubric.weak_signals),
            "follow_up_ids": list(self.follow_up_ids),
            "source": "curated",
        }


@dataclass(frozen=True)
class BankQuery:
    """
    A selection request. Purely structural -- no free text is searched.

    ``hint`` is a topic or skill name matched by token overlap against the
    question's topic and competency. It narrows; it never invents a match.
    """

    track: str = Track.AIML
    difficulty: int = 2
    hint: str = ""
    exclude_ids: frozenset[str] = frozenset()
    exclude_texts: frozenset[str] = frozenset()
    require_hint_match: bool = False
    limit: int = 1


class QuestionSource(StrEnum):
    """Where a spoken question came from. Recorded on every turn."""

    BANK = "bank"           # curated record, has a rubric
    TEMPLATE = "template"   # grounded template, bank miss or control flow
    PHRASED = "phrased"     # template intent, LLM wrote the sentence

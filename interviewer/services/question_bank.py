"""
Question bank: deterministic metadata lookup.

There is no similarity search here, by design. A bank of this size is a
dictionary, and TF-IDF over a spoken answer is what caused a LangGraph
candidate to be asked about Python generators.

Selection is a total order over four integer keys:
    1. track tier       exact track > same family > (rejected)
    2. hint tier        exact topic/competency > token overlap > none
    3. difficulty gap   |question.difficulty - requested|
    4. id               stable tie-break, so runs are reproducible

Returning nothing is a valid, expected answer: it means "bank miss", and the
policy layer speaks a resume-grounded template instead.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from interviewer.config import settings
from interviewer.domain.models import (
    TRACK_FAMILY,
    BankQuery,
    Question,
    Track,
    clamp_difficulty,
    tokenize,
)

logger = logging.getLogger("interviewer.question_bank")

_TRACK_EXACT, _TRACK_FAMILY_TIER = 0, 1
_HINT_EXACT, _HINT_OVERLAP, _HINT_NONE = 0, 1, 2


class QuestionBank:
    """Immutable after ``load``. Safe to share across all sessions."""

    def __init__(self) -> None:
        self._by_id: dict[str, Question] = {}
        self._by_track: dict[str, list[Question]] = {}
        self._loaded = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def load(self, path: str | Path | None = None) -> int:
        """
        Read curated questions. Returns the count loaded.

        Only the ``questions`` array is read. Older top-level ``topic: [str]``
        lists are ignored on purpose: they carry no rubric, so an answer to one
        cannot be graded.
        """
        source = Path(path or settings.interview.question_bank_path)
        if not source.is_file():
            logger.error("Question bank not found at %s", source)
            self._loaded = True
            return 0

        try:
            raw: Mapping[str, Any] = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.error("Question bank unreadable at %s: %s", source, exc)
            self._loaded = True
            return 0

        by_id: dict[str, Question] = {}
        skipped = 0
        for record in raw.get("questions") or ():
            if not isinstance(record, Mapping):
                skipped += 1
                continue
            question = Question.from_record(record)
            if question is None:
                skipped += 1
                continue
            if question.id in by_id:
                logger.warning("Duplicate question id %s ignored", question.id)
                skipped += 1
                continue
            by_id[question.id] = question

        by_track: dict[str, list[Question]] = {}
        for question in by_id.values():
            for track in question.tracks or {"general_cs"}:
                by_track.setdefault(track, []).append(question)
        for bucket in by_track.values():
            bucket.sort(key=lambda q: (q.difficulty, q.id))

        self._by_id = by_id
        self._by_track = by_track
        self._loaded = True

        dangling = sorted(
            f"{q.id} -> {fid}"
            for q in by_id.values()
            for fid in q.follow_up_ids
            if fid not in by_id
        )
        if dangling:
            logger.warning("Dangling follow_up_ids: %s", ", ".join(dangling))

        logger.info("Loaded %d curated questions (%d skipped) from %s", len(by_id), skipped, source)
        return len(by_id)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __len__(self) -> int:
        return len(self._by_id)

    # ── lookup ────────────────────────────────────────────────────────────

    def get(self, question_id: str | None) -> Question | None:
        if not question_id:
            return None
        return self._by_id.get(str(question_id))

    def select(self, query: BankQuery) -> list[Question]:
        """Best matches for a structural query. Empty list means bank miss."""
        if not self._by_id:
            return []

        track = Track.parse(query.track, default=Track.GENERAL_CS)
        family = TRACK_FAMILY.get(track, frozenset({str(track)}))
        difficulty = clamp_difficulty(query.difficulty)
        hint_tokens = tokenize(query.hint)
        hint_norm = " ".join(sorted(tokenize(query.hint)))
        excluded_texts = {t.strip().lower() for t in query.exclude_texts if t and t.strip()}

        scored: list[tuple[int, int, int, str, Question]] = []
        for question in self._candidates(family):
            if question.id in query.exclude_ids:
                continue
            if question.text.strip().lower() in excluded_texts:
                continue

            track_tier = _TRACK_EXACT if str(track) in question.tracks else _TRACK_FAMILY_TIER
            hint_tier = self._hint_tier(question, hint_tokens, hint_norm)
            if query.require_hint_match and hint_tier == _HINT_NONE:
                continue

            scored.append(
                (track_tier, hint_tier, abs(question.difficulty - difficulty), question.id, question)
            )

        scored.sort(key=lambda row: row[:4])
        limit = max(1, query.limit)
        return [row[4] for row in scored[:limit]]

    def select_one(self, query: BankQuery) -> Question | None:
        found = self.select(query)
        return found[0] if found else None

    def follow_ups(self, question: Question, exclude_ids: Iterable[str] = ()) -> list[Question]:
        """Authored follow-ups for a question, in declared order."""
        skip = set(exclude_ids)
        return [
            q
            for fid in question.follow_up_ids
            if fid not in skip and (q := self._by_id.get(fid)) is not None
        ]

    # ── introspection ─────────────────────────────────────────────────────

    def coverage(self) -> dict[str, Any]:
        """Where the bank is thin. Surfaced by tools, not used at runtime."""
        by_track = Counter(t for q in self._by_id.values() for t in q.tracks)
        by_difficulty = Counter(q.difficulty for q in self._by_id.values())
        return {
            "total": len(self._by_id),
            "by_track": dict(sorted(by_track.items())),
            "by_difficulty": {d: by_difficulty.get(d, 0) for d in (1, 2, 3)},
            "no_rubric": sorted(q.id for q in self._by_id.values() if q.rubric.is_empty),
        }

    # ── internals ─────────────────────────────────────────────────────────

    def _candidates(self, family: frozenset[str]) -> Iterable[Question]:
        seen: set[str] = set()
        for track in sorted(family):
            for question in self._by_track.get(track, ()):
                if question.id not in seen:
                    seen.add(question.id)
                    yield question

    @staticmethod
    def _hint_tier(question: Question, hint_tokens: frozenset[str], hint_norm: str) -> int:
        if not hint_tokens:
            return _HINT_NONE
        if hint_norm and hint_norm in (
            " ".join(sorted(tokenize(question.topic))),
            " ".join(sorted(tokenize(question.competency))),
        ):
            return _HINT_EXACT
        return _HINT_OVERLAP if hint_tokens & question.match_tokens else _HINT_NONE


_bank: QuestionBank | None = None


def get_question_bank() -> QuestionBank:
    """Process-wide bank, loaded once on first use."""
    global _bank
    if _bank is None:
        bank = QuestionBank()
        bank.load()
        _bank = bank
    return _bank


def set_question_bank(bank: QuestionBank | None) -> None:
    """Test seam."""
    global _bank
    _bank = bank

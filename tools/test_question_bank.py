"""
Stage 2 checks: metadata selection is exact, ordered, and honest about misses.

  python -m tools.test_question_bank
"""

from __future__ import annotations

from interviewer.domain.models import BankQuery, Question, Track, tokenize
from interviewer.services.question_bank import QuestionBank, get_question_bank


def test_tokenizer_does_not_substring_match() -> None:
    assert "java" in tokenize("Java")
    assert "java" not in tokenize("JavaScript"), "substring matching is the Java/JavaScript bug"
    assert tokenize("LLMs") == tokenize("LLM"), "plural folding keeps llms/llm together"
    assert "class" in tokenize("class design"), "-ss words must not be stemmed"
    print("[PASS] tokenizer: exact tokens, plural folded, no substring bleed")


def test_loads_only_curated_records() -> None:
    bank = get_question_bank()
    assert bank.is_loaded
    assert len(bank) == 36, f"expected 36 curated questions, got {len(bank)}"
    assert not any(q.id.startswith("legacy-") for q in bank._by_id.values()), "legacy records must be gone"
    for question in bank._by_id.values():
        assert question.rubric.expected_concepts, f"{question.id} has no rubric"
    print(f"[PASS] loaded {len(bank)} curated questions, every one has a rubric, zero legacy")


def test_bank_miss_returns_nothing() -> None:
    bank = get_question_bank()
    for unknown in ("LangGraph", "vLLM", "DuckDB"):
        hit = bank.select_one(BankQuery(track=Track.AIML, hint=unknown, require_hint_match=True))
        assert hit is None, f"{unknown} should be a bank miss, got {hit.id}"
    print("[PASS] LangGraph / vLLM / DuckDB are honest misses -> policy templates instead")


def test_hint_narrows_to_the_right_topic() -> None:
    bank = get_question_bank()
    hit = bank.select_one(BankQuery(track=Track.AIML, hint="Python", require_hint_match=True))
    assert hit is not None and "python" in tokenize(f"{hit.topic} {hit.competency}")
    print(f"[PASS] hint 'Python' -> {hit.id} ({hit.topic})")


def test_hr_never_gets_a_coding_question() -> None:
    bank = get_question_bank()
    for question in bank.select(BankQuery(track=Track.HR, limit=20)):
        assert question.tracks == {"hr"}, f"{question.id} leaked into an HR interview"
    print("[PASS] HR track is closed; no coding question can surface")


def test_selection_is_deterministic_and_excludes() -> None:
    bank = get_question_bank()
    query = BankQuery(track=Track.AIML, difficulty=2, limit=3)
    first = [q.id for q in bank.select(query)]
    assert first == [q.id for q in bank.select(query)], "selection must be reproducible"

    excluded = bank.select(
        BankQuery(track=Track.AIML, difficulty=2, limit=3, exclude_ids=frozenset({first[0]}))
    )
    assert first[0] not in [q.id for q in excluded], "exclude_ids must be honoured"
    print(f"[PASS] deterministic order {first}, exclusions honoured")


def test_difficulty_preference() -> None:
    bank = get_question_bank()
    easy = bank.select_one(BankQuery(track=Track.AIML, difficulty=1))
    hard = bank.select_one(BankQuery(track=Track.AIML, difficulty=3))
    assert easy is not None and hard is not None
    assert easy.difficulty <= hard.difficulty
    print(f"[PASS] difficulty dial moves the pick: L1 -> {easy.id} (d{easy.difficulty}), "
          f"L3 -> {hard.id} (d{hard.difficulty})")


def test_empty_bank_is_survivable() -> None:
    empty = QuestionBank()
    empty.load("does-not-exist.json")
    assert empty.select(BankQuery(track=Track.AIML)) == []
    assert empty.get("anything") is None
    print("[PASS] a missing bank file degrades to total bank-miss, never crashes")


def report_coverage() -> None:
    coverage = get_question_bank().coverage()
    print("\n--- bank coverage ---")
    print(f"total: {coverage['total']}")
    print(f"by track: {coverage['by_track']}")
    print(f"by difficulty (1-3): {coverage['by_difficulty']}")
    thin = [t for t, n in coverage["by_track"].items() if n < 6]
    if thin:
        print(f"THIN -> {thin}: these tracks will template-fallback often.")


if __name__ == "__main__":
    test_tokenizer_does_not_substring_match()
    test_loads_only_curated_records()
    test_bank_miss_returns_nothing()
    test_hint_narrows_to_the_right_topic()
    test_hr_never_gets_a_coding_question()
    test_selection_is_deterministic_and_excludes()
    test_difficulty_preference()
    test_empty_bank_is_survivable()
    report_coverage()
    print("\nStage 2 contract holds: exact metadata selection, misses are explicit.")

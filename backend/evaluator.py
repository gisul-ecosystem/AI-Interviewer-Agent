"""
Per-Turn Candidate Answer Evaluation Engine.
Evaluates correctness, depth, concepts covered, and missing concepts for each candidate turn.
Uses the question bank rubric (strong_signals / weak_signals) when present.
"""

from typing import Dict, List, Any


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _match_signal(answer: str, signal: str) -> bool:
    blob = _normalize(signal)
    if not blob or blob in ("n/a", "none"):
        return False
    if blob in answer:
        return True
    tokens = [t for t in blob.replace("/", " ").split() if len(t) > 3]
    if tokens and sum(1 for t in tokens if t in answer) >= max(1, (len(tokens) + 1) // 2):
        return True
    return False


def evaluate_turn_answer(
    candidate_answer: str,
    current_question: Dict[str, Any],
    intent: str,
    expected_concepts: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluates candidate answer per turn.
    Returns metric dictionary:
    {
      "score": float (0.0 - 1.0),
      "correctness": float (0.0 - 1.0),
      "depth": str ("low", "med", "high"),
      "concepts_covered": list,
      "missing_concepts": list,
      "is_skip": bool
    }
    """
    if expected_concepts is None:
        expected_concepts = []

    if intent in ("UNKNOWN_OR_SKIP", "TOPIC_CHANGE"):
        return {
            "score": 0.0,
            "correctness": 0.0,
            "depth": "low",
            "concepts_covered": [],
            "missing_concepts": expected_concepts,
            "is_skip": True
        }

    if intent == "REPEAT_REQUEST":
        return {
            "score": 0.0,
            "correctness": 0.0,
            "depth": "low",
            "concepts_covered": [],
            "missing_concepts": [],
            "is_skip": False,
        }

    clean_answer = _normalize(candidate_answer)
    words = clean_answer.split()
    word_count = len(words)

    strong = list(current_question.get("strong_signals") or [])
    weak = list(current_question.get("weak_signals") or [])
    rubric_concepts = list(expected_concepts)
    for item in strong + weak:
        if item and item not in rubric_concepts:
            rubric_concepts.append(item)

    CONCEPT_ALIASES = {
        "garbage collection": ["gc", "garbage collector", "garbage collection"],
        "hashmap": ["hash map", "hash table", "hashmap", "hashtable"],
        "polymorphism": ["polymorphic", "polymorphism", "poly"],
        "dynamic dispatch": ["dynamic dispatch", "late binding", "runtime dispatch"],
        "b-tree": ["b tree", "b-tree", "btree"],
        "o(1)": ["constant time", "o(1)", "o 1"],
        "stack": ["call stack", "stack frame", "stack"],
        "heap": ["heap space", "heap memory", "heap"],
        "open closed": ["open-closed", "open closed", "ocp"],
        "liskov substitution": ["liskov", "lsp", "subtyping"],
        "gil": ["gil", "global interpreter lock"]
    }

    covered = []
    missing = []
    for concept in rubric_concepts:
        c_clean = _normalize(concept)
        matched = _match_signal(clean_answer, concept)

        if not matched:
            for canon, aliases in CONCEPT_ALIASES.items():
                if c_clean == canon and any(a in clean_answer for a in aliases):
                    matched = True
                    break

        if matched:
            covered.append(concept)
        else:
            missing.append(concept)

    strong_hits = sum(1 for s in strong if _match_signal(clean_answer, s))
    weak_hits = sum(1 for s in weak if _match_signal(clean_answer, s))

    technical_stems = {
        "class", "object", "method", "interface", "override", "overload",
        "stack", "heap", "memory", "thread", "concurrency", "lock", "mutex",
        "index", "query", "database", "table", "complexity", "pointer", "tree",
        "node", "cache", "hash", "bucket", "latency", "throughput", "runtime"
    }
    tech_hits = sum(1 for w in words if any(stem in w for stem in technical_stems))

    if strong and strong_hits >= max(1, (len(strong) + 1) // 2):
        depth = "high"
        base_score = 0.88
    elif (len(covered) >= 2 or (len(covered) >= 1 and word_count >= 25)) or (word_count >= 40 and tech_hits >= 3):
        depth = "high"
        base_score = 0.85
    elif strong_hits >= 1 or weak_hits >= 1 or len(covered) >= 1 or word_count >= 18 or tech_hits >= 2:
        depth = "med"
        base_score = 0.65
    else:
        depth = "low"
        base_score = 0.40

    if rubric_concepts:
        coverage_ratio = len(covered) / len(rubric_concepts)
        final_score = round(0.4 * base_score + 0.6 * coverage_ratio, 2)
    else:
        final_score = base_score

    if strong and strong_hits == 0 and word_count >= 12:
        final_score = min(final_score, 0.48)
        depth = "low"

    return {
        "score": final_score,
        "correctness": final_score,
        "depth": depth,
        "concepts_covered": covered,
        "missing_concepts": missing,
        "is_skip": False
    }

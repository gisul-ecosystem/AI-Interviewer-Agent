"""
Async grading pass (cold path).

Runs AFTER the interview ends, so it never competes with the live turn loop.
For each turn it judges the candidate's answer against the rubric that was
attached to the question they were actually asked, and returns evidence quotes
taken verbatim from the transcript.

Design rules:
  - Pure module: no network. Callers inject an async ``llm_call(payload) -> str``.
  - Every quote is verified against the transcript; hallucinated quotes are dropped.
  - If the model is unavailable or returns garbage, a rule-based grade is used
    and the turn is marked ``graded_by: "heuristic"`` so reports stay auditable.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

JUDGE_MODEL_DEFAULT = "qwen3:4b-instruct-2507-q4_K_M"

VERDICTS = ("strong", "adequate", "weak", "no_answer")

HEDGE_PATTERNS = re.compile(
    r"\b(i think|i guess|maybe|probably|not sure|kind of|sort of|something like|"
    r"i believe|might be|i suppose)\b",
    re.IGNORECASE,
)

OWNERSHIP_SELF = re.compile(r"\b(i built|i wrote|i implemented|i designed|i owned|i added|my job|i handled)\b", re.IGNORECASE)
OWNERSHIP_TEAM = re.compile(r"\b(we|our team|the team|they)\b", re.IGNORECASE)

# Numbers, units, versions — the strongest cheap signal of real experience.
SPECIFIC_PATTERN = re.compile(
    r"(\b\d+(\.\d+)?\s?(ms|s|sec|seconds|hz|mb|gb|kb|%|x|k|m|rps|qps|epochs?|layers?)\b"
    r"|\b\d+(\.\d+)?%|\bv?\d+\.\d+(\.\d+)?\b)",
    re.IGNORECASE,
)

NO_ANSWER_PATTERN = re.compile(
    r"^\s*(i don'?t know|no idea|not sure|skip|pass|next|dunno|idk|nothing|n/?a)\s*[.!]?\s*$",
    re.IGNORECASE,
)

JUDGE_SYSTEM = """You are a strict technical interview grader. You grade ONE answer.

You are given the question, the rubric, and the candidate's verbatim answer.

Scoring (1-5):
5 = correct, specific, explains WHY, names real trade-offs or numbers
4 = correct and concrete, minor gaps
3 = broadly correct but generic; little evidence of hands-on work
2 = partially correct or heavily hedged; cannot explain mechanism
1 = incorrect, off-topic, or no real content

HARD RULES:
- Judge ONLY the answer text given. Never assume experience that is not stated.
- Length is not quality. A long vague answer scores 2, not 4.
- Every evidence quote MUST be copied verbatim from the answer. Never paraphrase a quote.
- If the answer contains no substantive content, use verdict "no_answer" and score 1.

Return ONLY this JSON object, no markdown:
{"score": <1-5>, "verdict": "strong|adequate|weak|no_answer",
 "covered_concepts": ["..."], "missing_concepts": ["..."],
 "evidence": [{"quote": "<verbatim from answer>", "why": "<what it shows>"}],
 "red_flags": ["..."], "summary": "<one sentence>"}"""


# ─── Quote verification ───────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())


def verify_quote(quote: str, answer: str) -> bool:
    """A quote counts only if it really appears in the transcript."""
    if not quote or not answer:
        return False
    q = " ".join(_normalize(quote).split())
    a = " ".join(_normalize(answer).split())
    if len(q) < 8:
        return False
    if q in a:
        return True
    # Tolerate small STT/model drift: most quote tokens must appear in sequence-free form.
    q_tokens = [t for t in q.split() if len(t) > 2]
    if not q_tokens:
        return False
    hits = sum(1 for t in q_tokens if t in a)
    return hits / len(q_tokens) >= 0.85


# ─── Heuristic fallback ───────────────────────────────────────────────────────

def heuristic_grade(turn: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based grade used when the judge model is unavailable or invalid."""
    answer = (turn.get("candidate_answer") or "").strip()
    expected = [str(c) for c in (turn.get("expected_concepts") or []) if c]

    if not answer or NO_ANSWER_PATTERN.match(answer) or len(answer.split()) < 4:
        return {
            "score": 1,
            "verdict": "no_answer",
            "covered_concepts": [],
            "missing_concepts": expected,
            "evidence": [],
            "red_flags": ["no substantive answer"],
            "summary": "Candidate did not answer the question.",
            "graded_by": "heuristic",
        }

    lowered = answer.lower()
    covered = [c for c in expected if c.lower() in lowered]
    coverage = (len(covered) / len(expected)) if expected else 0.0

    specific = len(SPECIFIC_PATTERN.findall(answer))
    hedges = len(HEDGE_PATTERNS.findall(answer))
    self_owned = bool(OWNERSHIP_SELF.search(answer))

    score = 2.0
    score += coverage * 2.0
    score += min(specific, 3) * 0.35
    score += 0.4 if self_owned else 0.0
    score -= min(hedges, 3) * 0.3
    score = max(1, min(5, round(score)))

    red_flags = []
    if hedges >= 3:
        red_flags.append("heavily hedged")
    if not self_owned and OWNERSHIP_TEAM.search(answer):
        red_flags.append("describes team work without personal ownership")

    verdict = "strong" if score >= 4 else "adequate" if score == 3 else "weak"
    return {
        "score": int(score),
        "verdict": verdict,
        "covered_concepts": covered,
        "missing_concepts": [c for c in expected if c not in covered],
        "evidence": [],
        "red_flags": red_flags,
        "summary": f"Rule-based grade: {len(covered)}/{len(expected) or 0} rubric concepts, {specific} concrete details.",
        "graded_by": "heuristic",
    }


# ─── LLM judge ────────────────────────────────────────────────────────────────

def build_judge_payload(
    turn: Dict[str, Any],
    candidate: Optional[Dict[str, Any]] = None,
    model: str = JUDGE_MODEL_DEFAULT,
) -> Dict[str, Any]:
    candidate = candidate or {}
    user_payload = {
        "QUESTION_ASKED": turn.get("asked_question") or turn.get("interviewer_question") or "",
        "COMPETENCY": turn.get("competency") or turn.get("topic") or "",
        "QUESTION_DIFFICULTY": turn.get("question_difficulty") or turn.get("difficulty"),
        "RUBRIC": {
            "expected_concepts": turn.get("expected_concepts") or [],
            "strong_signals": turn.get("strong_signals") or [],
            "weak_signals": turn.get("weak_signals") or [],
        },
        "CANDIDATE_ANSWER": turn.get("candidate_answer") or "",
        "TARGET_ROLE": candidate.get("target_role") or "",
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(user_payload, indent=2, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": 420,
    }


def parse_judge_output(raw: str, answer: str) -> Optional[Dict[str, Any]]:
    """Parse the judge JSON and drop any quote that is not in the transcript."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                return None
    if not isinstance(parsed, dict):
        return None

    try:
        score = int(round(float(parsed.get("score", 0))))
    except Exception:
        return None
    if not 1 <= score <= 5:
        return None

    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        verdict = "strong" if score >= 4 else "adequate" if score == 3 else "weak"

    evidence = []
    for item in parsed.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "").strip()
        if verify_quote(quote, answer):
            evidence.append({"quote": quote, "why": str(item.get("why") or "").strip()})

    def _strlist(key: str) -> List[str]:
        return [str(v).strip() for v in (parsed.get(key) or []) if str(v).strip()][:8]

    return {
        "score": score,
        "verdict": verdict,
        "covered_concepts": _strlist("covered_concepts"),
        "missing_concepts": _strlist("missing_concepts"),
        "evidence": evidence[:4],
        "red_flags": _strlist("red_flags"),
        "summary": str(parsed.get("summary") or "").strip()[:300],
        "graded_by": "llm",
    }


async def grade_turn(
    turn: Dict[str, Any],
    candidate: Optional[Dict[str, Any]],
    llm_call: Optional[Callable[[Dict[str, Any]], Awaitable[str]]],
    model: str = JUDGE_MODEL_DEFAULT,
) -> Dict[str, Any]:
    answer = (turn.get("candidate_answer") or "").strip()
    base = {
        "turn": turn.get("turn"),
        "competency": turn.get("competency") or turn.get("topic"),
        "topic": turn.get("topic"),
        "question_id": turn.get("question_id"),
        "question_difficulty": turn.get("question_difficulty") or turn.get("difficulty"),
        "asked_question": turn.get("asked_question") or turn.get("interviewer_question"),
    }

    # Obvious non-answers are decided by rule: it saves a judge call and stops a
    # weak model from awarding points for "i don't know".
    needs_judge = bool(answer) and not NO_ANSWER_PATTERN.match(answer) and len(answer.split()) >= 4

    if llm_call is not None and needs_judge:
        try:
            raw = await llm_call(build_judge_payload(turn, candidate, model))
            parsed = parse_judge_output(raw, answer)
            if parsed:
                return {**base, **parsed}
        except Exception as exc:  # judge must never break the report
            base["grader_error"] = str(exc)[:200]

    return {**base, **heuristic_grade(turn)}


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_report(turn_grades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll per-turn grades into per-competency scores and an overall picture."""
    by_competency: Dict[str, Dict[str, Any]] = {}
    for grade in turn_grades:
        key = str(grade.get("competency") or "General").strip() or "General"
        bucket = by_competency.setdefault(
            key, {"competency": key, "scores": [], "evidence": [], "missing": [], "red_flags": []}
        )
        bucket["scores"].append(grade.get("score", 1))
        bucket["evidence"].extend(grade.get("evidence") or [])
        bucket["missing"].extend(grade.get("missing_concepts") or [])
        bucket["red_flags"].extend(grade.get("red_flags") or [])

    competencies = []
    for bucket in by_competency.values():
        scores = bucket["scores"] or [1]
        avg = round(sum(scores) / len(scores), 2)
        competencies.append(
            {
                "competency": bucket["competency"],
                "score": avg,
                "turns": len(scores),
                "level": "strong" if avg >= 4 else "adequate" if avg >= 3 else "weak",
                "evidence": bucket["evidence"][:3],
                "gaps": list(dict.fromkeys(bucket["missing"]))[:5],
                "red_flags": list(dict.fromkeys(bucket["red_flags"]))[:3],
            }
        )
    competencies.sort(key=lambda c: c["score"], reverse=True)

    all_scores = [g.get("score", 1) for g in turn_grades] or [1]
    overall = round(sum(all_scores) / len(all_scores), 2)
    answered = sum(1 for g in turn_grades if g.get("verdict") != "no_answer")

    if overall >= 4.0:
        recommendation = "strong_hire"
    elif overall >= 3.2:
        recommendation = "hire"
    elif overall >= 2.4:
        recommendation = "borderline"
    else:
        recommendation = "no_hire"

    return {
        "overall_score": overall,
        "overall_out_of": 5,
        "recommendation": recommendation,
        "turns_graded": len(turn_grades),
        "turns_answered": answered,
        "strengths": [c["competency"] for c in competencies if c["score"] >= 4][:5],
        "weaknesses": [c["competency"] for c in competencies if c["score"] <= 2][:5],
        "competencies": competencies,
        "graded_by": (
            "llm" if all(g.get("graded_by") == "llm" for g in turn_grades)
            else "heuristic" if all(g.get("graded_by") == "heuristic" for g in turn_grades)
            else "mixed"
        ),
    }


async def grade_session(
    session: Dict[str, Any],
    llm_call: Optional[Callable[[Dict[str, Any]], Awaitable[str]]] = None,
    model: str = JUDGE_MODEL_DEFAULT,
    concurrency: int = 3,
) -> Dict[str, Any]:
    """Grade every answered turn of a finished session."""
    history = [t for t in (session.get("history") or []) if (t.get("candidate_answer") or "").strip()]
    candidate = session.get("candidate") or {}

    if not history:
        return {
            "session_id": session.get("session_id"),
            "candidate_name": candidate.get("name"),
            "target_role": candidate.get("target_role"),
            "overall_score": 0,
            "overall_out_of": 5,
            "recommendation": "insufficient_data",
            "turns_graded": 0,
            "turns_answered": 0,
            "strengths": [],
            "weaknesses": [],
            "competencies": [],
            "turn_grades": [],
            "graded_by": "none",
        }

    limiter = asyncio.Semaphore(max(1, concurrency))

    async def _one(turn: Dict[str, Any]) -> Dict[str, Any]:
        async with limiter:
            return await grade_turn(turn, candidate, llm_call, model)

    turn_grades = await asyncio.gather(*[_one(t) for t in history])
    turn_grades = list(turn_grades)

    report = aggregate_report(turn_grades)
    report.update(
        {
            "session_id": session.get("session_id"),
            "candidate_name": candidate.get("name"),
            "target_role": candidate.get("target_role"),
            "turn_grades": turn_grades,
        }
    )
    return report

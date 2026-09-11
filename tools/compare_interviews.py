"""
Run 3 scripted interviews through the real turn API.

Measures per-turn latency (hot TEMPLATE/RETRIEVE vs warm GENERATE+LLM)
and records which fallback fired. Writes transcripts into SQLite.

  python -m backend.compare_interviews
  python -m backend.inspect_db
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

from rag_engine import init_rag
from backend.store import interview_store
from backend.grader import grade_session
import app


PERSONAS: List[Dict[str, Any]] = [
    {
        "id": "aiml_bank_rich",
        "role": "aiml",
        "why": "Bank-rich AIML CV. Expect RETRIEVE openers, GENERATE only after a real project answer.",
        "resume": """
Candidate Name: Priya Sharma
College: IIT Delhi
Degree: B.Tech Computer Science
Role: Machine Learning Intern
Experience: 1 year
Company: HealthAI
Core Skills: Python, PyTorch, Machine Learning, Deep Learning, SQL
Projects: MoleCheck Image Classifier, Credit Risk Scorer
""",
        "answers": [
            "Hi, I am Priya Sharma. I interned at HealthAI and built MoleCheck, a skin-lesion classifier.",
            "I owned the training loop in PyTorch. I froze the backbone, trained the head for 12 epochs, and measured AUC 0.91 on a held-out set.",
            "When the class imbalance skewed recall, I added weighted sampling and a lower decision threshold. Latency stayed under 80ms on CPU.",
            "I don't know, can we skip this?",
            "sorry could you repeat the question please",
            "On Credit Risk Scorer I used SQL to join bureau files and trained a gradient boosting model. I implemented the feature pipeline myself.",
        ],
    },
    {
        "id": "webd_thin_bank",
        "role": "webd",
        "why": "Web CV against a thin question bank. Expect more TEMPLATE fills and weak RETRIEVE.",
        "resume": """
Candidate Name: Rohan Mehta
College: VIT Vellore
Degree: B.E. Information Technology
Role: Full Stack Developer
Experience: 2 years
Company: CampusHub
Core Skills: React.js, Node.js, JavaScript, HTML, CSS, MongoDB
Projects: Campus Portal, Realtime Chat App
""",
        "answers": [
            "Hi I am Rohan. I am a full stack developer at CampusHub and I built the Campus Portal.",
            "I wrote the React dashboard and the Node.js REST API. Students hit /api/courses; I added JWT auth and MongoDB indexes.",
            "The chat app used WebSockets. When the socket dropped I queued messages in memory and flushed on reconnect.",
            "I don't know",
            "Can we switch to Java?",
            "On the portal I handled file uploads with multer, 10MB limit, and stored objects on disk because S3 was not approved.",
        ],
    },
    {
        "id": "aiml_bank_miss",
        "role": "aiml",
        "why": "Novel CV terms (LangGraph, vLLM). Plan should miss the bank; live turn should TEMPLATE, not a weakly similar ML question.",
        "resume": """
Candidate Name: Kabir Singh
College: BITS Pilani
Degree: B.Tech CSE
Role: AI Engineer
Experience: 1 year
Company: AgentLabs
Core Skills: Python, LangGraph, vLLM, DuckDB, Machine Learning
Projects: Invoice Agent with LangGraph
""",
        "answers": [
            "Hi I am Kabir. I built an invoice agent with LangGraph at AgentLabs.",
            "The graph had extract, tool-call, and merge nodes. I wrote the reducer that merged line items into graph state.",
            "vLLM served the tool-calling model. When a tool timed out I retried twice then returned a partial invoice.",
            "I don't know, skip this.",
            "sorry can you repeat that",
            "DuckDB held the invoice tables. I used it for aggregations because Postgres was overkill for the demo.",
        ],
    },
]


def _classify_fallback(mode: str, needs_llm: bool, llm_ms: float, spoken: str) -> str:
    if mode == "TEMPLATE":
        return "TEMPLATE spoken fallback (no live LLM)"
    if mode == "GENERATE":
        if llm_ms >= 7500:
            return "GENERATE LLM timeout/slow → spoken_fallback likely"
        if needs_llm and llm_ms < 1:
            return "GENERATE skipped LLM unexpectedly"
        return "GENERATE warm path (LLM follow-up)"
    if mode == "RETRIEVE":
        return "RETRIEVE bank question (hot path)"
    return f"unknown mode={mode}"


async def run_one(persona: Dict[str, Any]) -> Dict[str, Any]:
    features = app.parse_resume_heuristics(persona["resume"])
    started = await app.start_interview({
        "role": persona["role"],
        "features": features,
        "resume_text": persona["resume"],
    })
    session_id = started["session_id"]
    session_preview = started.get("session") or {}
    jr = session_preview.get("job_requirements") or {}
    first_q = ((session_preview.get("current_question") or {}).get("question") or "")

    turns: List[Dict[str, Any]] = []
    for i, answer in enumerate(persona["answers"], start=1):
        t0 = time.perf_counter()
        result = await app.api_interview_turn({
            "session_id": session_id,
            "candidate_answer": answer,
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000
        llm = result.get("llm_response") or {}
        state = result.get("interview_state") or {}
        mode = llm.get("question_mode") or ""
        spoken = llm.get("question") or ""
        plane = "warm" if mode == "GENERATE" else "hot"
        turns.append({
            "n": i,
            "answer_preview": answer[:90],
            "intent_hint": (
                "skip" if "don't know" in answer.lower() or answer.lower().strip() in ("i don't know",)
                else "repeat" if "repeat" in answer.lower()
                else "topic" if "switch to" in answer.lower()
                else "answer"
            ),
            "mode": mode,
            "plane": plane,
            "latency_ms": round(elapsed_ms, 1),
            "stage": state.get("stage"),
            "topic": llm.get("topic"),
            "diff": state.get("difficulty_level"),
            "spoken": spoken[:160],
            "fallback": _classify_fallback(mode, mode == "GENERATE", elapsed_ms, spoken),
            "stt_corrected": result.get("corrected_transcript"),
        })

    live = app.load_live_session(session_id)
    live["ended_at"] = time.time()
    live["interview_state"]["stage"] = "completed"
    persisted = interview_store.save_interview(live)

    t_grade = time.perf_counter()
    report = await grade_session(live, llm_call=None)
    grade_ms = (time.perf_counter() - t_grade) * 1000
    report["graded_at"] = time.time()
    report["status"] = "done"
    report["graded_by"] = "heuristic"
    interview_store.save_report(session_id, report)

    latencies = [t["latency_ms"] for t in turns]
    hot = [t["latency_ms"] for t in turns if t["plane"] == "hot"]
    warm = [t["latency_ms"] for t in turns if t["plane"] == "warm"]
    return {
        "persona": persona["id"],
        "why": persona["why"],
        "session_id": session_id,
        "parsed": {
            "name": features.get("name"),
            "skills": features.get("skills"),
            "projects": features.get("projects"),
        },
        "sift": {
            "role": jr.get("title"),
            "track": jr.get("track"),
            "interview_skills": jr.get("interview_skills"),
            "intersection": jr.get("intersection"),
            "gaps": jr.get("role_gaps"),
        },
        "opener": first_q[:160],
        "persisted": persisted,
        "db_path": interview_store.db_path,
        "turns": turns,
        "latency": {
            "n": len(latencies),
            "avg_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "max_ms": max(latencies) if latencies else 0,
            "hot_avg_ms": round(sum(hot) / len(hot), 1) if hot else None,
            "warm_avg_ms": round(sum(warm) / len(warm), 1) if warm else None,
            "hot_n": len(hot),
            "warm_n": len(warm),
            "grade_heuristic_ms": round(grade_ms, 1),
        },
        "report": {
            "overall_score": report.get("overall_score"),
            "recommendation": report.get("recommendation"),
            "graded_by": report.get("graded_by"),
            "turns_graded": report.get("turns_graded") or len(report.get("turn_grades") or []),
        },
    }


async def main_async() -> None:
    init_rag()
    print("Database:", interview_store.db_path)
    print("SQLite ready:", interview_store.is_available)
    print("No Postgres. Python already includes sqlite3 — nothing to install.\n")

    results = []
    for persona in PERSONAS:
        print("=" * 72)
        print(f"INTERVIEW {persona['id']}  ({persona['role']})")
        print(persona["why"])
        print("=" * 72)
        result = await run_one(persona)
        results.append(result)
        print("session:", result["session_id"])
        print("parsed:", json.dumps(result["parsed"], ensure_ascii=False))
        print("sift:", json.dumps(result["sift"], ensure_ascii=False))
        print("opener:", result["opener"])
        for t in result["turns"]:
            print(
                f"  T{t['n']} {t['plane']:4} {t['latency_ms']:8.1f}ms  "
                f"mode={t['mode']:9} stage={t['stage']:20} {t['fallback']}"
            )
            print(f"     Q: {t['spoken']}")
        print("latency:", result["latency"])
        print("grade:", result["report"], "persisted=", result["persisted"])
        print()

    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(f"{'persona':20} {'hot_n':6} {'hot_avg':8} {'warm_n':7} {'warm_avg':9} {'max_ms':8} {'rec':12}")
    for r in results:
        L = r["latency"]
        print(
            f"{r['persona']:20} {L['hot_n']:<6} {str(L['hot_avg_ms']):8} "
            f"{L['warm_n']:<7} {str(L['warm_avg_ms']):9} {L['max_ms']:<8} "
            f"{r['report'].get('recommendation')}"
        )

    print("\nInspect the rows with:")
    print("  python -m backend.inspect_db")
    out = {"db_path": interview_store.db_path, "interviews": results}
    Path = __import__("pathlib").Path
    dump = Path(__file__).resolve().parent.parent / "dry_run_compare.json"
    dump.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", dump)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

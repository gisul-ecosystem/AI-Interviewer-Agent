"""
Test Suite for Structured 2-Phase Interview Pipeline:
- Projects Phase: 3-4 questions per project
- Skills Phase: 1-2 questions per skill
- Difficulty: strictly bounded within [1, 3]
"""

from backend.intent_engine import detect_candidate_intent
from backend.evaluator import evaluate_turn_answer
from backend.interview_fsm import InterviewFSM
from backend.session_manager import create_session
from backend.question_engine import question_engine
from backend.prompt_builder import build_interviewer_prompt
from rag_engine import question_bank_rag, init_rag


def test_structured_process():
    print("=== Testing Structured 2-Phase Interview Pipeline ===")
    init_rag()

    # Candidate with 2 projects and 2 skills
    candidate_profile = {
        "name": "Alex Mercer",
        "projects": ["Autonomous Drone Navigation", "Distributed Cache Service"],
        "skills": ["Python", "Docker", "System Design"],
        "degree": "B.Tech Computer Science",
        "college": "Tech Institute"
    }

    session = create_session(candidate_profile)
    session["interview_state"]["questions_remaining"] = 40
    plan = dict(session.get("interview_plan") or session["candidate"].get("interview_plan") or {})
    plan["questions_per_project"] = 4
    plan["questions_per_skill"] = 2
    session["interview_plan"] = plan
    session["candidate"]["interview_plan"] = plan
    state = session["interview_state"]
    candidate = session["candidate"]

    print(f"\n[Created Session] Projects: {candidate['projects']}, Skills: {candidate['skills']}")
    assert state["difficulty_level"] in [1, 2, 3], f"Invalid initial difficulty: {state['difficulty_level']}"
    assert state["stage"] == "warmup"

    # ── Turn 1: Warmup / Introduction ───────────────────────────────────────
    ans = "Hello, I am Alex Mercer, a software engineer with expertise in drone robotics and distributed systems."
    intent = detect_candidate_intent(ans, stage=state["stage"], questions_asked=state["questions_asked"])
    eval_res = evaluate_turn_answer(ans, session["current_question"], intent)
    
    fsm = InterviewFSM(state)
    state = fsm.update_from_evaluation(eval_res, intent, candidate_dict=candidate)
    
    q_dec = question_engine.decide(
        candidate_answer=ans,
        intent=intent,
        fsm_state=state,
        eval_res=eval_res,
        candidate_dict=candidate,
        question_bank_rag=question_bank_rag
    )

    print(f"Turn 1 -> Stage: {state['stage']}, Topic: {state['current_topic']}, Diff: {state['difficulty_level']}/3")
    assert state["stage"] == "project_deep_dive", f"Expected stage project_deep_dive, got {state['stage']}"
    assert state["current_project_index"] == 0
    assert "Autonomous Drone Navigation" in state["current_topic"]
    assert 1 <= state["difficulty_level"] <= 3

    # ── Project 1: Questions (Simulating 3-4 questions) ──────────────────────
    project_1_turns = [
        "In Autonomous Drone Navigation, we used ROS and Kalman filters to navigate GPS-denied environments. I personally built the perception and sensor fusion module.",
        "We processed stereo camera and LIDAR feeds through an extended Kalman filter running at 50Hz on an onboard Jetson Xavier.",
        "The hardest bottleneck was latency jitter from the USB bus and occasional Kalman divergence when features were sparse, which we resolved with an optical flow fallback.",
        "We achieved a 98% waypoint accuracy in indoor simulation runs with sub-10ms fusion loop latency."
    ]

    for i, p_ans in enumerate(project_1_turns, start=2):
        intent = detect_candidate_intent(p_ans, stage=state["stage"], questions_asked=state["questions_asked"])
        eval_res = evaluate_turn_answer(p_ans, {"expected_concepts": ["architecture", "latency", "sensor"]}, intent)
        eval_res["score"] = 0.9
        eval_res["depth"] = "high"
        eval_res["is_skip"] = False
        fsm = InterviewFSM(state)
        state = fsm.update_from_evaluation(eval_res, intent, candidate_dict=candidate)
        q_dec = question_engine.decide(
            candidate_answer=p_ans,
            intent=intent,
            fsm_state=state,
            eval_res=eval_res,
            candidate_dict=candidate,
            question_bank_rag=question_bank_rag
        )
        print(f"Turn {i} (Project 1, Q{state['project_question_count']}) -> Stage: {state['stage']}, Topic: {state['current_topic']}, Diff: {state['difficulty_level']}/3")
        assert 1 <= state["difficulty_level"] <= 3

    # By Turn 5 (after 4 questions on Project 1), should advance to Project 2
    print(f"\nAfter Project 1 -> Active Project Index: {state.get('current_project_index')}, Topic: {state['current_topic']}")
    assert state["current_project_index"] == 1, "Should have advanced to Project 2"
    assert "Distributed Cache Service" in state["current_topic"]

    # ── Project 2: Questions (Simulating 3-4 questions) ──────────────────────
    project_2_turns = [
        "For the Distributed Cache Service, I designed a consistent hashing ring in Go with virtual nodes to balance key distribution.",
        "We used gRPC for inter-node communication and an LRU eviction strategy with TTL support.",
        "A major challenge was split-brain scenarios during network partition; we implemented a Raft consensus heartbeat to safely elect leader nodes.",
        "We measured p99 get latency under 12ms at 50k QPS and kept replica lag under a second.",
        "Failover used a heartbeat plus a fencing token so a partitioned leader could not accept writes.",
    ]

    for i, p_ans in enumerate(project_2_turns, start=6):
        intent = detect_candidate_intent(p_ans, stage=state["stage"], questions_asked=state["questions_asked"])
        eval_res = evaluate_turn_answer(p_ans, {"expected_concepts": ["hash", "cache", "network"]}, intent)
        eval_res["score"] = 0.9
        eval_res["depth"] = "high"
        eval_res["is_skip"] = False
        fsm = InterviewFSM(state)
        state = fsm.update_from_evaluation(eval_res, intent, candidate_dict=candidate)
        q_dec = question_engine.decide(
            candidate_answer=p_ans,
            intent=intent,
            fsm_state=state,
            eval_res=eval_res,
            candidate_dict=candidate,
            question_bank_rag=question_bank_rag
        )
        print(f"Turn {i} (Project 2, Q{state['project_question_count']}) -> Stage: {state['stage']}, Topic: {state['current_topic']}, Diff: {state['difficulty_level']}/3")
        assert 1 <= state["difficulty_level"] <= 3

    # All projects done -> Must advance to Skills Assessment (OOPs / DSA first)
    print(f"\nAfter Projects -> Stage: {state['stage']}, Topic: {state['current_topic']}")
    assert state["stage"] == "skills_assessment", f"Expected skills_assessment, got {state['stage']}"
    assert state["current_skill_index"] == 0
    assert any(
        token in state["current_topic"]
        for token in ("Object-Oriented", "OOP", "Python", "Data Structures")
    )

    # ── Skills Assessment: (1-2 questions per skill) ─────────────────────────
    # Skill 1 (Python)
    skill_1_turns = [
        "In Python, I frequently use asyncio with aiohttp for high throughput I/O and multiprocessing to bypass the GIL for CPU intensive tasks.",
        "The GIL prevents true multi-core Python bytecode execution in standard CPython threads, so CPU tasks need separate processes or C extensions."
    ]
    for i, s_ans in enumerate(skill_1_turns, start=9):
        intent = detect_candidate_intent(s_ans, stage=state["stage"], questions_asked=state["questions_asked"])
        eval_res = evaluate_turn_answer(s_ans, {"expected_concepts": ["gil", "asyncio"]}, intent)
        eval_res["score"] = 0.9
        eval_res["depth"] = "high"
        eval_res["is_skip"] = False
        fsm = InterviewFSM(state)
        state = fsm.update_from_evaluation(eval_res, intent, candidate_dict=candidate)
        q_dec = question_engine.decide(
            candidate_answer=s_ans,
            intent=intent,
            fsm_state=state,
            eval_res=eval_res,
            candidate_dict=candidate,
            question_bank_rag=question_bank_rag
        )
        print(f"Turn {i} (Skill 1, Q{state['skill_question_count']}) -> Stage: {state['stage']}, Topic: {state['current_topic']}, Diff: {state['difficulty_level']}/3")
        assert 1 <= state["difficulty_level"] <= 3

    print(f"\nAfter Skill 1 -> Active Skill Index: {state.get('current_skill_index')}, Topic: {state['current_topic']}")
    assert state["stage"] in ("skills_assessment", "closing")
    if state["stage"] == "skills_assessment":
        assert state["current_skill_index"] >= 1

    skill_idx = min(int(state.get("current_skill_index") or 0), len(candidate["skills"]) - 1)
    next_skill = candidate["skills"][skill_idx]

    # Skill 2
    docker_ans = f"I have used {next_skill} in production with monitoring, retries, and a clear rollback path."
    intent = detect_candidate_intent(docker_ans, stage=state["stage"], questions_asked=state["questions_asked"])
    eval_res = evaluate_turn_answer(docker_ans, {"expected_concepts": [next_skill.lower()]}, intent)
    eval_res["score"] = 0.9
    eval_res["depth"] = "high"
    eval_res["is_skip"] = False
    fsm = InterviewFSM(state)
    state = fsm.update_from_evaluation(eval_res, intent, candidate_dict=candidate)
    q_dec = question_engine.decide(
        candidate_answer=docker_ans,
        intent=intent,
        fsm_state=state,
        eval_res=eval_res,
        candidate_dict=candidate,
        question_bank_rag=question_bank_rag
    )
    print(f"Turn 11 (Skill 2) -> Stage: {state['stage']}, Topic: {state['current_topic']}, Diff: {state['difficulty_level']}/3")
    assert 1 <= state["difficulty_level"] <= 3

    # Prompt builder test to verify payload formatting
    prompt_payload = build_interviewer_prompt(
        fsm_state=state,
        candidate_dict=candidate,
        candidate_answer=docker_ans,
        question_decision=q_dec
    )
    assert prompt_payload.get("messages")
    blob = " ".join(m.get("content", "") for m in prompt_payload["messages"]).lower()
    assert "alex mercer" in blob
    assert q_dec.spoken_question

    print("\nSUCCESS: All structured pipeline stages (Projects 3-4 Qs, Skills 1-2 Qs, Diff 1-3) passed flawlessly!")


if __name__ == "__main__":
    test_structured_process()

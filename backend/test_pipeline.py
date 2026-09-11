"""
Test Suite for Modular Decoupled Backend Pipeline.
Verifies Intent Engine, Evaluator, Interview FSM, and Prompt Builder.
"""

from backend.intent_engine import detect_candidate_intent
from backend.evaluator import evaluate_turn_answer
from backend.interview_fsm import InterviewFSM
from backend.prompt_builder import build_interviewer_prompt
from backend.session_manager import create_session, get_session
from backend.question_engine import question_engine, QuestionDecision
from rag_engine import question_bank_rag, init_rag


def test_modular_pipeline():
    print("=== Testing Modular Decoupled Backend Architecture ===")
    init_rag()

    # 1. Test Session Creation
    session = create_session({
        "name": "Abhijeet Singh",
        "skills": ["PyTorch", "Machine Learning", "FastAPI", "Java", "Object-Oriented Programming"],
        "projects": ["MoleCheck Image Classifier"],
        "degree": "B.Tech Computer Science",
        "college": "Jaypee University of Engineering and Technology"
    })
    sid = session["session_id"]
    print(f"[Session Created] ID: {sid}, Candidate: {session['candidate']['name']}")

    # 2. Turn 1: Self Introduction -> structured opening question with a rubric.
    ans1 = "Hi, my name is Abhijeet Singh, studying CS at Jaypee University."
    intent1 = detect_candidate_intent(ans1)
    eval1 = evaluate_turn_answer(ans1, session["current_question"], intent1, ["Name", "Background"])
    fsm1 = InterviewFSM(session["interview_state"])
    s1 = fsm1.update_from_evaluation(eval1, intent1, candidate_dict=session["candidate"])
    session["interview_state"] = s1

    dec1 = question_engine.decide(
        candidate_answer=ans1,
        intent=intent1,
        fsm_state=s1,
        eval_res=eval1,
        candidate_dict=session["candidate"],
        question_bank_rag=question_bank_rag
    )
    print(f"[Turn 1 Self-Intro] Mode: {dec1.mode}, Target Topic: {dec1.target_topic}")
    assert intent1 == "SELF_INTRO"
    assert dec1.mode == "TEMPLATE"
    assert dec1.scripted is True
    assert "molecheck" in (dec1.spoken_question or "").lower()
    assert "first project" in (dec1.spoken_question or "").lower()

    # 3. Turn 2: Deep Technical Answer -> Matches Question Bank (Mode A: RETRIEVE or Mode B: ADAPT)
    ans2 = "How did you handle overfitting — dropout, weight decay, early stopping? For overfitting in PyTorch, I use dropout layers with 0.3 probability, weight decay L2 regularization, and early stopping on validation loss."
    intent2 = detect_candidate_intent(ans2)
    eval2 = evaluate_turn_answer(ans2, {"expected_concepts": ["dropout", "regularization", "validation"]}, intent2, ["dropout", "regularization", "validation"])
    fsm2 = InterviewFSM(session["interview_state"])
    s2 = fsm2.update_from_evaluation(eval2, intent2, candidate_dict=session["candidate"])
    session["interview_state"] = s2

    dec2 = question_engine.decide(
        candidate_answer=ans2,
        intent=intent2,
        fsm_state=s2,
        eval_res=eval2,
        candidate_dict=session["candidate"],
        question_bank_rag=question_bank_rag
    )
    print(f"[Turn 2 Technical Answer] Mode: {dec2.mode}, Score: {dec2.similarity_score:.2f}, Seed Q: {dec2.seed_question}")
    assert intent2 == "TECHNICAL_ANSWER"
    # When similarity < 0.75, question engine delegates to LLM Mode C: GENERATE
    assert dec2.mode == "GENERATE"
    assert "target_topic" in dec2.__dict__

    # 4. Turn 3: Skip / Don't Know -> Question Engine Mode C (GENERATE Pivot)
    ans3 = "Sorry, I haven't read about graph neural networks yet."
    intent3 = detect_candidate_intent(ans3)
    eval3 = evaluate_turn_answer(ans3, {}, intent3, ["GNN", "Adjacency Matrix"])
    fsm3 = InterviewFSM(session["interview_state"])
    s3 = fsm3.update_from_evaluation(eval3, intent3, candidate_dict=session["candidate"])
    session["interview_state"] = s3

    dec3 = question_engine.decide(
        candidate_answer=ans3,
        intent=intent3,
        fsm_state=s3,
        eval_res=eval3,
        candidate_dict=session["candidate"],
        question_bank_rag=question_bank_rag
    )
    print(f"[Turn 3 Skip Pivot] Mode: {dec3.mode}, Target Topic: {dec3.target_topic}, Action: {s3.get('action')}")
    assert intent3 == "UNKNOWN_OR_SKIP"
    assert dec3.mode in ("TEMPLATE", "GENERATE")
    assert s3.get("action") == "PIVOT_TOPIC"

    # 5. Turn 4: Explicit "i dont know" test
    ans4 = "i dont know"
    intent4 = detect_candidate_intent(ans4)
    eval4 = evaluate_turn_answer(ans4, {}, intent4, [])
    fsm4 = InterviewFSM(s3)
    s4 = fsm4.update_from_evaluation(eval4, intent4, candidate_dict=session["candidate"])
    fsm4.set_current_topic(dec3.target_topic)
    dec4 = question_engine.decide(
        candidate_answer=ans4,
        intent=intent4,
        fsm_state=fsm4.get_dict(),
        eval_res=eval4,
        candidate_dict=session["candidate"],
        question_bank_rag=question_bank_rag
    )
    print(f"[Turn 4 'i dont know'] Intent: {intent4}, Mode: {dec4.mode}, Pivot Target: {dec4.target_topic}")
    assert intent4 == "UNKNOWN_OR_SKIP"
    assert dec4.mode in ("TEMPLATE", "GENERATE")
    topic = dec4.target_topic
    projects = session["candidate"]["projects"]
    skills = session["candidate"]["skills"]
    assert topic in skills or topic in projects or any(p in topic for p in projects) or any(s in str(topic) for s in skills)

    # 6. Turn 5: Candidate asks to repeat ("sorry could you repeat the question")
    ans5 = "sorry could you repeat the question please"
    intent5 = detect_candidate_intent(ans5)
    eval5 = evaluate_turn_answer(ans5, {}, intent5, [])
    dec5 = question_engine.decide(
        candidate_answer=ans5,
        intent=intent5,
        fsm_state=s4,
        eval_res=eval5,
        candidate_dict=session["candidate"],
        question_bank_rag=question_bank_rag,
        exclude_questions=["How did you handle overfitting — dropout, weight decay, early stopping?"]
    )
    print(f"[Turn 5 'repeat question'] Intent: {intent5}, Mode: {dec5.mode}, Directive: {dec5.directive[:60]}...")
    assert intent5 == "REPEAT_REQUEST"
    assert "How did you handle overfitting" in dec5.seed_question

    # 7. Test Explicit Topic Switch: Candidate says "Can you change after that? Can we switch to Java?"
    ans6 = "Can you change after that? Can we switch to Java?"
    intent6 = detect_candidate_intent(ans6)
    eval6 = evaluate_turn_answer(ans6, {}, intent6, [])
    fsm6 = InterviewFSM(s4)
    s6 = fsm6.update_from_evaluation(eval6, intent6, candidate_dict=session["candidate"])
    dec6 = question_engine.decide(
        candidate_answer=ans6,
        intent=intent6,
        fsm_state=s6,
        eval_res=eval6,
        candidate_dict=session["candidate"],
        question_bank_rag=question_bank_rag
    )
    print(f"\n[Turn 6 Candidate requests 'Java'] Target Topic: {dec6.target_topic}, Directive: {dec6.directive[:65]}...")
    assert dec6.target_topic == "Java"
    assert "project" not in dec6.target_topic.lower()
    assert dec6.mode == "GENERATE"
    if dec6.selected_question:
        assert "expected_concepts" in dec6.selected_question

    # 8. Test Correctness & Adaptive Difficulty Increase on Strong Technical Answer
    # Candidate answers accurately based on the question asked
    bank_q = (dec6.selected_question or {}).get("question") or ""
    if "volatile" in bank_q.lower():
        ans7 = "Volatile guarantees memory visibility across threads by bypassing CPU cache and using memory barriers, but it does not provide atomicity. Synchronized provides mutual exclusion locks so only one thread executes the critical section, preventing race conditions."
        expected_jvm = dec6.selected_question.get("expected_concepts", [])
    else:
        ans7 = "In the JVM, Stack memory stores method stack frames and local variables, whereas Heap memory stores objects. StackOverflowError happens with deep or infinite recursion, while OutOfMemoryError occurs when the garbage collection cannot reclaim heap memory."
        expected_jvm = (dec6.selected_question or {}).get("expected_concepts") or ["stack", "heap"]
    intent7 = detect_candidate_intent(ans7)
    eval7 = evaluate_turn_answer(ans7, dec6.selected_question or {}, intent7, expected_concepts=expected_jvm)
    initial_diff = s6["difficulty_level"]
    fsm7 = InterviewFSM(s6)
    s7 = fsm7.update_from_evaluation(eval7, intent7, candidate_dict=session["candidate"])
    print(f"[Turn 7 Strong Technical Answer] Score: {eval7['score']}, Depth: {eval7['depth']}, Diff: {initial_diff} -> {s7['difficulty_level']}")
    assert eval7["score"] >= 0.75
    assert 1 <= s7["difficulty_level"] <= 3
    if initial_diff < 3:
        assert s7["difficulty_level"] > initial_diff
    else:
        assert s7["difficulty_level"] == 3

    # 9. Test Explicit Topic Switch to OOP: "Can we do oops?"
    ans8 = "Can we change topic to oops?"
    intent8 = detect_candidate_intent(ans8)
    eval8 = evaluate_turn_answer(ans8, {}, intent8, [])
    dec8 = question_engine.decide(
        candidate_answer=ans8,
        intent=intent8,
        fsm_state=s7,
        eval_res=eval8,
        candidate_dict=session["candidate"],
        question_bank_rag=question_bank_rag
    )
    print(f"[Turn 8 Candidate requests 'OOP'] Target Topic: {dec8.target_topic}, Probes: {len(dec8.probes)}")
    assert "Object-Oriented Programming" in dec8.target_topic or "OOP" in dec8.target_topic
    assert "project" not in dec8.target_topic.lower()

    # 10. Test Prompt Builder with QuestionDecision
    prompt_payload = build_interviewer_prompt(
        fsm_state=s7,
        candidate_dict=session["candidate"],
        candidate_answer=ans7,
        question_bank_context=dec6.probes,
        last_question="How does the JVM distinguish between Stack and Heap memory?",
        intent=intent7,
        question_decision=dec6
    )

    sys_content = prompt_payload["messages"][0]["content"]
    user_content = prompt_payload["messages"][1]["content"]

    print("\n[Prompt Builder Output Strategy Header]:")
    print("\n".join(sys_content.split("\n")[:7]))
    assert f"[MODE {dec6.mode[:1]}" in sys_content or dec6.mode in sys_content
    assert "QUESTION_MODE" in user_content

    print("\n[SUCCESS] All Question Engine, Dynamic Difficulty, and Topic Pivot Tests Passed Successfully!")


if __name__ == "__main__":
    test_modular_pipeline()


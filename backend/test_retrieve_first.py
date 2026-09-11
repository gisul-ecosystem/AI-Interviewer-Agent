"""Qwen writes live questions. Python still owns stage, project, skill, and probe."""

from backend.intent_engine import detect_candidate_intent
from backend.evaluator import evaluate_turn_answer
from backend.interview_fsm import InterviewFSM
from backend.session_manager import create_session
from backend.question_engine import question_engine
from rag_engine import question_bank_rag, init_rag


def test_retrieve_first_modes():
    init_rag()
    session = create_session({
        "name": "Priya Sharma",
        "skills": ["PyTorch", "Machine Learning", "Python"],
        "projects": ["MoleCheck Image Classifier"],
        "degree": "B.Tech",
        "college": "Test University",
        "role": "AI / ML Engineer",
    })
    candidate = session["candidate"]
    state = session["interview_state"]
    plan = session.get("interview_plan") or {}
    assert plan.get("opener_id") == "aiml-project-ownership-01"

    intro = "Hi, my name is Priya Sharma, I am a machine learning student."
    intent = detect_candidate_intent(intro, stage=state["stage"], questions_asked=0)
    eval_res = evaluate_turn_answer(intro, session["current_question"], intent)
    fsm = InterviewFSM(state)
    state = fsm.update_from_evaluation(eval_res, intent, candidate_dict=candidate)
    dec = question_engine.decide(
        candidate_answer=intro,
        intent=intent,
        fsm_state=state,
        eval_res=eval_res,
        candidate_dict=candidate,
        question_bank_rag=question_bank_rag,
        interview_plan=plan,
    )
    assert dec.mode == "TEMPLATE", dec.mode
    assert dec.needs_llm is False
    assert dec.scripted is True
    assert dec.wants_qwen_phrasing(intent) is False
    assert "molecheck" in (dec.spoken_question or "").lower()
    assert "first project" in (dec.spoken_question or "").lower()

    repeat = question_engine.decide(
        candidate_answer="sorry could you repeat the question please",
        intent="REPEAT_REQUEST",
        fsm_state=state,
        eval_res={"is_skip": False},
        candidate_dict=candidate,
        exclude_questions=[dec.spoken_question],
        interview_plan=plan,
    )
    assert repeat.mode == "TEMPLATE"
    assert repeat.needs_llm is False
    assert dec.spoken_question in repeat.spoken_question

    print("[SUCCESS] Warmup uses a scripted project bridge; repeat stays scripted.")


def test_grounded_followup_quality():
    from backend.followup_engine import plan_followup, finalize_followup, is_quality_followup
    from backend.question_engine import question_engine

    answer = (
        "In Autonomous Drone Navigation we used ROS and Kalman filters to navigate GPS-denied "
        "environments. I personally built the perception and sensor fusion module running at 50Hz."
    )
    spec = plan_followup(
        answer,
        eval_res={"depth": "med", "missing_concepts": ["failure handling"], "score": 0.65},
        candidate_dict={"skills": ["ROS", "Python"], "projects": ["Autonomous Drone Navigation"]},
        topic="Project: Autonomous Drone Navigation",
        difficulty=2,
        focus="implementation",
    )
    assert spec["anchor_term"]
    assert "Kalman" in spec["anchor_term"] or "ROS" in spec["anchor_term"] or "Autonomous" in spec["anchor_term"]
    assert spec["spoken_fallback"]
    assert spec["anchor_term"].lower() in spec["spoken_fallback"].lower() or spec["must_probe"].lower() in spec["spoken_fallback"].lower()

    generic = "That's great, can you tell me more about your project?"
    assert is_quality_followup(generic, spec) is False
    assert finalize_followup(generic, spec) == spec["spoken_fallback"]

    good = (
        f"On Autonomous Drone Navigation you used {spec['anchor_term']} — "
        f"what did you implement yourself, and how did sensor data move through it?"
    )
    assert is_quality_followup(good, spec) is True
    assert finalize_followup(good, spec) == good
    canned_overfit = (
        "On Autonomous Drone Navigation, what did you do about overfitting "
        "and did validation actually improve?"
    )
    assert is_quality_followup(canned_overfit, spec) is False

    state = {
        "stage": "project_deep_dive",
        "difficulty_level": 2,
        "current_project_index": 0,
        "project_question_count": 1,
        "questions_already_asked": [],
    }
    dec = question_engine.decide(
        candidate_answer=answer,
        intent="TECHNICAL_ANSWER",
        fsm_state=state,
        eval_res={"depth": "med", "missing_concepts": ["latency"], "score": 0.65, "is_skip": False},
        candidate_dict={
            "name": "Alex",
            "skills": ["Python", "ROS"],
            "projects": ["Autonomous Drone Navigation"],
            "target_track": "aiml",
            "bank_track": "aiml",
        },
    )
    assert dec.mode == "GENERATE"
    assert dec.followup_spec
    assert dec.spoken_question
    assert dec.spoken_question == dec.followup_spec["spoken_fallback"]
    assert "Autonomous Drone Navigation" in dec.spoken_question
    assert spec["anchor_term"] in dec.spoken_question or "ROS" in dec.spoken_question or "Kalman" in dec.spoken_question

    vague = "yeah it mostly worked, we handled the usual issues."
    second = plan_followup(
        vague,
        eval_res={"depth": "low", "missing_concepts": [], "score": 0.4},
        candidate_dict={"skills": ["ROS", "Python"], "projects": ["Autonomous Drone Navigation"]},
        topic="Project: Autonomous Drone Navigation",
        difficulty=2,
        focus="implementation",
        project_thread=spec["project_thread"],
        project_name="Autonomous Drone Navigation",
    )
    assert second["project_name"] == "Autonomous Drone Navigation"
    assert "Autonomous Drone Navigation" in second["spoken_fallback"]
    assert second["probe_type"] != spec["probe_type"]
    assert spec["anchor_term"].lower() in second["spoken_fallback"].lower() or spec["anchor_term"].lower() in (second["anchor_term"] or "").lower()

    print("[SUCCESS] Grounded follow-up planner rejects generic LLM questions.")


def test_skip_does_not_repeat_same_question():
    init_rag()
    session = create_session({
        "name": "Priya Sharma",
        "skills": ["PyTorch", "Python"],
        "projects": ["MoleCheck Image Classifier", "Credit Risk Scorer"],
        "role": "AI / ML Engineer",
    })
    candidate = session["candidate"]
    state = session["interview_state"]
    plan = session.get("interview_plan") or {}

    intro = "Hi I am Priya, I built MoleCheck with PyTorch."
    intent = detect_candidate_intent(intro, stage=state["stage"], questions_asked=0)
    eval_res = evaluate_turn_answer(intro, session["current_question"], intent)
    fsm = InterviewFSM(state)
    state = fsm.update_from_evaluation(eval_res, intent, candidate_dict=candidate)
    first = question_engine.decide(
        candidate_answer=intro, intent=intent, fsm_state=state, eval_res=eval_res,
        candidate_dict=candidate, question_bank_rag=question_bank_rag, interview_plan=plan,
    )
    fsm.add_asked_question(first.spoken_question)
    state = fsm.get_dict()

    skip_intent = detect_candidate_intent("please skip this question", stage=state["stage"], questions_asked=1)
    assert skip_intent == "UNKNOWN_OR_SKIP"
    skip_eval = evaluate_turn_answer("please skip this question", {"question": first.spoken_question}, skip_intent)
    fsm = InterviewFSM(state)
    state = fsm.update_from_evaluation(skip_eval, skip_intent, candidate_dict=candidate)
    skipped = question_engine.decide(
        candidate_answer="please skip this question",
        intent=skip_intent,
        fsm_state=state,
        eval_res=skip_eval,
        candidate_dict=candidate,
        question_bank_rag=question_bank_rag,
        interview_plan=plan,
        exclude_questions=state.get("questions_already_asked") or [first.spoken_question],
    )
    assert skipped.mode == "GENERATE"
    assert first.spoken_question not in (skipped.spoken_question or "")
    spoken_skip = (skipped.spoken_question or "").lower()
    assert "molecheck" in spoken_skip
    assert state["stage"] == "project_deep_dive"
    assert state.get("current_project_index", 0) == 0

    skip2_eval = evaluate_turn_answer("skip", {"question": skipped.spoken_question}, "UNKNOWN_OR_SKIP")
    fsm = InterviewFSM(state)
    fsm.add_asked_question(skipped.spoken_question)
    state = fsm.update_from_evaluation(skip2_eval, "UNKNOWN_OR_SKIP", candidate_dict=candidate)
    skipped2 = question_engine.decide(
        candidate_answer="skip",
        intent="UNKNOWN_OR_SKIP",
        fsm_state=state,
        eval_res=skip2_eval,
        candidate_dict=candidate,
        question_bank_rag=question_bank_rag,
        interview_plan=plan,
        exclude_questions=state.get("questions_already_asked") or [],
    )
    assert skipped2.spoken_question != skipped.spoken_question
    assert state.get("current_project_index", 0) == 0
    print("[SUCCESS] Skip moves on instead of repeating the same question.")


def test_java_is_not_javascript():
    from backend.intent_engine import detect_requested_topic
    topic = detect_requested_topic(
        "Can we switch to Java?",
        candidate_skills=["JavaScript", "React.js", "Node.js"],
    )
    assert topic == "Java"
    print("[SUCCESS] Java is not mapped onto JavaScript.")


def test_qwen_phrasing_keeps_policy():
    from backend.followup_engine import (
        finalize_spoken_question,
        is_quality_phrasing,
        mentions_project,
        plan_followup,
    )
    from backend.prompt_builder import build_interviewer_prompt
    from backend.question_engine import QuestionDecision

    assert mentions_project("On MoleCheck you used MobileNetV2 — what failed first?", "MoleCheck Image Classifier")

    retrieve = QuestionDecision(
        mode="RETRIEVE",
        seed_question="Walk me through one project on your resume. What problem did it solve?",
        seed_topic="Project: MoleCheck",
        similarity_score=1.0,
        target_topic="Project: MoleCheck",
        target_difficulty=1,
        directive="Speak the selected bank opener.",
        probes=[],
        spoken_question="Thanks Priya. On MoleCheck — Walk me through one project on your resume. What problem did it solve?",
    )
    assert retrieve.needs_llm is False
    assert retrieve.wants_qwen_phrasing("SELF_INTRO") is False
    assert retrieve.wants_qwen_phrasing("REPEAT_REQUEST") is False

    polished = "Thanks Priya. On MoleCheck, what problem did that project solve, and what did you personally own?"
    assert is_quality_phrasing(polished, retrieve.spoken_question)
    assert "MoleCheck" in finalize_spoken_question(polished, retrieve)

    generic = "That's great, can you tell me more about your project?"
    assert finalize_spoken_question(generic, retrieve) == retrieve.spoken_question

    spec = plan_followup(
        "I mediated a conflict between a hiring manager and a rejected candidate.",
        eval_res={"depth": "med", "missing_concepts": [], "score": 0.7},
        candidate_dict={"projects": ["Campus Hiring Drive"], "skills": ["communication"]},
        topic="Campus Hiring Drive",
        focus="implementation",
        project_name="Campus Hiring Drive",
        interview_style="behavioral",
    )
    assert "situation" in spec["spoken_fallback"].lower() or "result" in spec["spoken_fallback"].lower()

    hr_prompt = build_interviewer_prompt(
        fsm_state={"stage": "project_deep_dive", "questions_already_asked": []},
        candidate_dict={
            "name": "Priya",
            "target_role": "HR / People Operations",
            "interview_style": "behavioral",
            "projects": ["Campus Hiring Drive"],
            "skills": ["communication"],
        },
        candidate_answer="I told the manager we needed evidence, not a gut call.",
        question_decision=QuestionDecision(
            mode="GENERATE",
            seed_question=spec["spoken_fallback"],
            seed_topic="Project: Campus Hiring Drive",
            similarity_score=0.0,
            target_topic="Project: Campus Hiring Drive",
            target_difficulty=2,
            directive="Stay on Campus Hiring Drive.",
            probes=[],
            spoken_question=spec["spoken_fallback"],
            followup_spec=spec,
        ),
    )
    sys_msg = hr_prompt["messages"][0]["content"]
    assert "STAR" in sys_msg
    assert "Do not ask coding" in sys_msg
    print("[SUCCESS] Qwen phrases bank/follow-up questions without becoming policy.")


def test_project_thread_climbs_one_ladder():
    """Three turns on one project must be one conversation, not three cold starts."""
    from backend.followup_engine import plan_followup
    from backend.prompt_builder import build_interviewer_prompt
    from backend.question_engine import QuestionDecision

    cv = {"name": "Aditya", "skills": ["TensorFlow", "Keras"], "projects": ["MoleCheck"]}
    answers = [
        "On MoleCheck I fine-tuned MobileNetV2 in Keras for malignant versus benign lesions.",
        "We used data augmentation because the malignant class was much smaller.",
        "I tracked sensitivity and specificity instead of raw accuracy.",
    ]

    thread = {}
    probes, spoken = [], []
    for turn, answer in enumerate(answers):
        spec = plan_followup(
            answer,
            eval_res={"depth": "med", "missing_concepts": [], "score": 0.65},
            candidate_dict=cv,
            topic="Project: MoleCheck",
            focus=("what_used", "how_built", "regularization")[turn],
            project_thread=thread,
            project_name="MoleCheck",
        )
        probes.append(spec["probe_type"])
        spoken.append(spec["spoken_fallback"])
        thread = spec["project_thread"]

    assert len(set(probes)) == 3, probes
    assert probes == ["what_used", "how_built", "regularization"], probes
    assert all("MoleCheck" in text for text in spoken)
    deep = ("bottleneck", "drift", "roll back", "rollback", "broke first")
    assert not any(word in text.lower() for text in spoken for word in deep), spoken
    assert any("overfit" in text.lower() or "augment" in text.lower() or "implement" in text.lower() for text in spoken), spoken

    last = plan_followup(
        answers[-1],
        eval_res={"depth": "med", "missing_concepts": [], "score": 0.65},
        candidate_dict=cv,
        topic="Project: MoleCheck",
        focus="metric",
        project_thread=thread,
        project_name="MoleCheck",
    )
    assert last["previous_probe"] == probes[-1]
    assert last["ladder_step"] == 4

    prompt = build_interviewer_prompt(
        fsm_state={"stage": "project_deep_dive", "questions_already_asked": spoken},
        candidate_dict=cv,
        candidate_answer=answers[-1],
        question_decision=QuestionDecision(
            mode="GENERATE",
            seed_question=last["spoken_fallback"],
            seed_topic="Project: MoleCheck",
            similarity_score=0.0,
            target_topic="Project: MoleCheck",
            target_difficulty=2,
            directive="Stay on MoleCheck.",
            probes=[],
            spoken_question=last["spoken_fallback"],
            followup_spec=last,
        ),
    )
    sys_msg = prompt["messages"][0]["content"]
    assert "WRITE THE NEXT INTERVIEW QUESTION" in sys_msg
    assert "last answer" in sys_msg.lower()
    assert "follow this" in sys_msg.lower() or "follow that answer" in sys_msg.lower()
    assert "dictionary definition" in sys_msg.lower()
    print("[SUCCESS] Project follow-ups climb one ladder and reference the last turn.")


def test_followup_uses_spoken_technical_terms():
    from backend.followup_engine import plan_followup

    spec = plan_followup(
        "I fine-tuned MobileNetV2 in Keras with data augmentation because the malignant class was small.",
        eval_res={"depth": "med", "missing_concepts": [], "score": 0.7},
        candidate_dict={"skills": ["TensorFlow", "Keras"], "projects": ["MoleCheck"]},
        topic="Project: MoleCheck",
        focus="implementation",
        project_name="MoleCheck",
    )
    blob = " ".join([spec["anchor_term"], *(spec["mentioned_terms"] or []), spec["spoken_fallback"]]).lower()
    assert "keras" in blob or "mobilenet" in blob or "augmentation" in blob, spec
    assert "molecheck" in spec["spoken_fallback"].lower()
    assert spec["probe_type"] == "how_built"
    blob_q = spec["spoken_fallback"].lower()
    assert "implement" in blob_q or "built" in blob_q or "train" in blob_q or "wrote" in blob_q, spec["spoken_fallback"]
    from backend.followup_engine import finalize_followup, is_quality_followup
    deep = "On MoleCheck you used Keras — when the classifier drifted, what broke first and what would make you roll it back?"
    assert finalize_followup(deep, spec) == spec["spoken_fallback"]
    overfit = (
        "On MoleCheck you used Keras — the malignant class was small. "
        "What did you do about overfitting, and did validation actually improve?"
    )
    assert is_quality_followup(overfit, spec) is True
    off_topic = "On MoleCheck, how would you design an LRU cache so get and put are both O(1)?"
    assert is_quality_followup(off_topic, spec) is False
    print("[SUCCESS] Follow-ups latch onto spoken technical terms.")


def test_probe_follows_what_they_said():
    from backend.followup_engine import probe_from_answer

    assert probe_from_answer(
        "We used ROS and Kalman filters to navigate without GPS.",
        [],
    ) == "how_built"
    assert probe_from_answer(
        "I fine-tuned MobileNetV2 with data augmentation because the malignant class was small.",
        ["how_built"],
    ) == "regularization"
    assert probe_from_answer(
        "I tracked sensitivity and specificity instead of raw accuracy.",
        ["how_built", "regularization"],
    ) == "metric"


def test_parse_qwen_skill_questions_from_upload():
    from backend.interview_plan import parse_generated_skill_questions

    raw = """```json
    {"slots":[
      {"skill":"Python","questions":[
        "What is the GIL and when do you use multiprocessing instead of threads?",
        "How do generators keep a large Pandas pipeline from loading everything into RAM?",
        "How would you isolate a silent NaN in a NumPy transform before it hits training?"
      ]}
    ]}
    ```"""
    parsed = parse_generated_skill_questions(raw, ["Python", "TensorFlow"], 3)
    assert len(parsed["Python"]) == 3
    assert "TensorFlow" not in parsed
    assert parsed["Python"][0].endswith("?")
    print("[SUCCESS] Upload-time Qwen skill JSON parses.")


def test_plan_locks_skill_bases_and_three_questions():
    from backend.interview_plan import build_interview_plan
    from rag_engine import init_rag, question_bank_rag

    init_rag()
    plan = build_interview_plan(
        {
            "name": "Aditya",
            "skills": ["TensorFlow", "Python"],
            "projects": ["MoleCheck"],
            "bank_track": "aiml",
            "target_track": "aiml",
            "resume_text": (
                "Skills: Python, TensorFlow, Keras, MobileNetV2\n"
                "Projects: MoleCheck - CNN skin lesion classifier with transfer learning and class weights."
            ),
        },
        question_bank_rag=question_bank_rag,
    )
    names = [s.lower() for s in plan["skill_names"]]
    assert any("object-oriented" in n or "oop" in n for n in names), plan["skill_names"]
    assert any("data structure" in n or "dsa" in n for n in names), plan["skill_names"]
    assert plan["questions_per_skill"] >= 3
    for slot in plan["skill_slots"]:
        questions = slot.get("questions") or []
        assert len(questions) >= 3, slot
        assert slot.get("base_question") == questions[0]
        assert len({q.lower() for q in questions}) == len(questions), questions
        for question in questions:
            assert len(str(question).split()) >= 6, question
    tf_slot = next(s for s in plan["skill_slots"] if str(s["skill"]).lower() == "tensorflow")
    blob = " ".join(tf_slot["questions"]).lower()
    assert "mobilenet" in blob or "keras" in blob or "transfer" in blob or "class" in blob, tf_slot
    session = create_session({
        "name": "Aditya",
        "skills": ["TensorFlow", "Python"],
        "projects": ["MoleCheck"],
        "raw_text": "SKILLS Python, TensorFlow, Keras, MobileNetV2\nPROJECTS MoleCheck - CNN with transfer learning.",
        "role": "AI / ML Engineer",
    })
    candidate = session["candidate"]
    state = {
        "stage": "skills_assessment",
        "difficulty_level": 2,
        "current_skill_index": 0,
        "skill_question_count": 0,
        "questions_already_asked": [],
        "action": "CONTINUE",
    }
    first = question_engine.decide(
        candidate_answer="We finished MoleCheck.",
        intent="TECHNICAL_ANSWER",
        fsm_state=state,
        eval_res={"depth": "med", "score": 0.6, "is_skip": False},
        candidate_dict=candidate,
        interview_plan=session["interview_plan"],
    )
    assert first.mode == "TEMPLATE"
    assert first.wants_qwen_phrasing("TECHNICAL_ANSWER") is False
    spoken = (first.spoken_question or "").lower()
    assert "molecheck" not in spoken
    assert any(token in spoken for token in ("class", "inherit", "polymorph", "encapsul", "object", "liskov", "composition"))
    follow = question_engine.decide(
        candidate_answer="I used inheritance so the trainer and the serving adapter shared one interface.",
        intent="TECHNICAL_ANSWER",
        fsm_state={**state, "skill_question_count": 1},
        eval_res={"depth": "med", "score": 0.7, "is_skip": False},
        candidate_dict=candidate,
        interview_plan=session["interview_plan"],
    )
    assert follow.mode == "TEMPLATE"
    assert follow.wants_qwen_phrasing("TECHNICAL_ANSWER") is False
    assert follow.spoken_question
    assert first.spoken_question != follow.spoken_question
    assert "molecheck" not in (follow.spoken_question or "").lower()
    from backend.prompt_builder import build_interviewer_prompt
    prompt = build_interviewer_prompt(
        fsm_state=state,
        candidate_dict=candidate,
        candidate_answer="I used Random Forest on MoleCheck.",
        question_decision=follow,
    )
    sys_msg = prompt["messages"][0]["content"]
    assert "PLANNED ASK" in sys_msg
    assert "never mention a project" in sys_msg.lower() or "do not mention" in sys_msg.lower()
    print("[SUCCESS] OOPs/DSA and skill follow-ups are locked at upload.")


def test_dsa_question_is_not_a_project_followup():
    from backend.followup_engine import finalize_spoken_question
    from backend.question_engine import QuestionDecision, question_engine
    from backend.session_manager import create_session

    session = create_session({
        "name": "Aditya",
        "skills": ["TensorFlow", "Python"],
        "projects": ["MoleCheck", "Mental Health Predictor"],
        "raw_text": "SKILLS Python, TensorFlow\nPROJECTS MoleCheck, Mental Health Predictor",
        "role": "AI / ML Engineer",
    })
    candidate = session["candidate"]
    names = [str(s.get("skill") or "") for s in (session["interview_plan"] or {}).get("skill_slots") or []]
    dsa_idx = next(
        i for i, name in enumerate(names)
        if "data structure" in name.lower() or name.lower() == "dsa"
    )
    dsa = question_engine.decide(
        candidate_answer="On MoleCheck we used MobileNetV2 and a correlation matrix.",
        intent="TECHNICAL_ANSWER",
        fsm_state={
            "stage": "skills_assessment",
            "difficulty_level": 2,
            "current_skill_index": dsa_idx,
            "skill_question_count": 0,
            "questions_already_asked": [],
            "action": "CONTINUE",
        },
        eval_res={"depth": "med", "score": 0.6, "is_skip": False},
        candidate_dict=candidate,
        interview_plan=session["interview_plan"],
    )
    spoken = (dsa.spoken_question or "").lower()
    assert dsa.skill_kind == "foundation"
    assert dsa.wants_qwen_phrasing("TECHNICAL_ANSWER") is False
    assert "molecheck" not in spoken
    assert "mental health" not in spoken
    assert any(token in spoken for token in ("hash", "cache", "linked", "stack", "queue", "array", "complex", "pointer", "tree", "sort", "search", "list"))
    leaked = QuestionDecision(
        mode="TEMPLATE",
        seed_question=dsa.seed_question,
        seed_topic=dsa.seed_topic,
        similarity_score=1.0,
        target_topic=dsa.target_topic,
        target_difficulty=1,
        directive=dsa.directive,
        probes=[],
        spoken_question=dsa.spoken_question,
        skill_kind="foundation",
        block_projects=["MoleCheck", "Mental Health Predictor"],
    )
    hijack = "On MoleCheck you used MobileNetV2 — how would you store that in a stack or queue?"
    assert finalize_spoken_question(hijack, leaked) == dsa.spoken_question
    print("[SUCCESS] DSA questions stay DSA, not project follow-ups.")


def test_topic_changes_use_scripted_bridges():
    from backend.question_engine import question_engine
    from backend.session_manager import create_session
    from backend.prompt_builder import build_interviewer_prompt

    session = create_session({
        "name": "Aditya",
        "skills": ["TensorFlow", "Python"],
        "projects": ["MoleCheck", "Mental Health Predictor"],
        "role": "AI / ML Engineer",
    })
    candidate = session["candidate"]
    plan = session["interview_plan"]
    assert plan["questions_per_project"] == 4

    second = question_engine.decide(
        candidate_answer="On MoleCheck I used Keras and MobileNetV2.",
        intent="TECHNICAL_ANSWER",
        fsm_state={
            "stage": "project_deep_dive",
            "difficulty_level": 1,
            "current_project_index": 1,
            "project_question_count": 0,
            "questions_already_asked": [],
            "action": "CONTINUE",
        },
        eval_res={"depth": "med", "score": 0.6, "is_skip": False},
        candidate_dict=candidate,
        interview_plan=plan,
    )
    spoken = (second.spoken_question or "").lower()
    assert second.scripted is True
    assert second.mode == "TEMPLATE"
    assert "molecheck" in spoken
    assert "mental health" in spoken
    assert "move" in spoken or "let's" in spoken

    skills = question_engine.decide(
        candidate_answer="Random Forest for the second project.",
        intent="TECHNICAL_ANSWER",
        fsm_state={
            "stage": "skills_assessment",
            "difficulty_level": 1,
            "current_skill_index": 0,
            "skill_question_count": 0,
            "questions_already_asked": [],
            "action": "CONTINUE",
        },
        eval_res={"depth": "med", "score": 0.6, "is_skip": False},
        candidate_dict=candidate,
        interview_plan=plan,
    )
    skill_spoken = (skills.spoken_question or "").lower()
    assert "projects" in skill_spoken
    assert "fund" in skill_spoken or "object-oriented" in skill_spoken

    prompt = build_interviewer_prompt(
        fsm_state={"stage": "project_deep_dive", "questions_already_asked": []},
        candidate_dict=candidate,
        candidate_answer="We used Keras.",
        question_decision=second,
    )
    sys_msg = prompt["messages"][0]["content"]
    assert "4 questions" in sys_msg.lower() or "exactly 4" in sys_msg.lower()
    assert "TOPIC CHANGES" in sys_msg
    print("[SUCCESS] Topic changes close the last topic, then open the next; 4 questions per project.")


if __name__ == "__main__":
    test_retrieve_first_modes()
    test_grounded_followup_quality()
    test_skip_does_not_repeat_same_question()
    test_java_is_not_javascript()
    test_qwen_phrasing_keeps_policy()
    test_project_thread_climbs_one_ladder()
    test_followup_uses_spoken_technical_terms()
    test_probe_follows_what_they_said()
    test_parse_qwen_skill_questions_from_upload()
    test_plan_locks_skill_bases_and_three_questions()
    test_dsa_question_is_not_a_project_followup()
    test_topic_changes_use_scripted_bridges()

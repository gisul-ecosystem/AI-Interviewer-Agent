"""
Qwen writes the spoken question. Python already chose the stage, project, skill, and probe.
"""

import json
from typing import Any, Dict, List


def _style_block(interview_style: str, target_role: str, *, skill_phase: bool = False) -> str:
    if interview_style == "behavioral":
        return (
            f"This is an HR / people-operations interview for {target_role}.\n"
            "Ask STAR questions: situation, the action THEY took, and the result.\n"
            "Probe judgment, communication, conflict, hiring decisions, and stakeholder handling.\n"
            "Do not ask coding, system-design, or algorithm questions.\n"
            "GOOD: 'When the hiring manager wanted to skip the take-home, what did you say, and what happened next?'\n"
            "BAD: 'Can you tell me more about that?'"
        )
    if skill_phase:
        return (
            f"This is the SKILLS phase of a technical interview for {target_role}.\n"
            "Projects are over. Ask the planned OOP, DSA, or skill question itself.\n"
            "Do not mention resume projects. Do not ask how they used this in MoleCheck or any other project.\n"
            "OOPs = classes, inheritance, polymorphism, encapsulation. "
            "DSA = arrays, hash maps, stacks, queues, linked lists, time complexity.\n"
            "CV skills = what the language or library is and how it works, in simple terms.\n"
            "GOOD: 'What is the difference between compile-time and runtime polymorphism?'\n"
            "BAD: 'On MoleCheck, how would you put Random Forest into a class hierarchy?'"
        )
    return (
        f"This is a technical interview for {target_role}.\n"
        "Your next question MUST follow a specific claim in THEIR last answer.\n"
        "Name the tool, step, metric, or problem they just said, then go one level deeper on that same claim.\n"
        "Do not invent a topic they did not mention. Do not ask a dictionary definition.\n"
        "Do not ask about production drift, rollback, SLAs, or system design.\n"
        "GOOD: they said they fine-tuned MobileNet because the malignant class was small → "
        "'On MoleCheck you said the malignant class was small — how did you train MobileNet so it did not just memorize the majority class?'\n"
        "BAD: they said they used ROS and Kalman filters → 'What did you do about overfitting?'\n"
        "BAD: 'In simple terms, what is Keras?'"
    )


def build_interviewer_prompt(
    fsm_state: Dict[str, Any],
    candidate_dict: Dict[str, Any],
    candidate_answer: str,
    question_bank_context: List[str] = None,
    last_question: str = "",
    intent: str = "TECHNICAL_ANSWER",
    resume_chunks: List[str] = None,
    question_decision: Any = None
) -> Dict[str, Any]:
    """Build the live phrasing prompt for Qwen3-4B."""
    if question_bank_context is None:
        question_bank_context = []
    if resume_chunks is None:
        resume_chunks = []

    c_name = candidate_dict.get("name") or "Candidate"
    skills = candidate_dict.get("skills", [])
    projects = candidate_dict.get("projects", [])
    sift = candidate_dict.get("skill_sift") or {}
    primary_project = projects[0] if projects else "your recent project"
    primary_skills = ", ".join((sift.get("intersection") or skills)[:3]) if (sift.get("intersection") or skills) else "the target role"
    target_role = candidate_dict.get("target_role") or "the open role"
    interview_style = candidate_dict.get("interview_style") or "technical"

    q_mode = getattr(question_decision, "mode", "GENERATE") if question_decision else "GENERATE"
    seed_q = getattr(question_decision, "seed_question", None) if question_decision else None
    spoken_seed = getattr(question_decision, "spoken_question", None) if question_decision else None
    target_topic = getattr(question_decision, "target_topic", fsm_state.get("current_topic", "Technical Architecture")) if question_decision else fsm_state.get("current_topic", "Technical Architecture")
    raw_diff = getattr(question_decision, "target_difficulty", fsm_state.get("difficulty_level", 2)) if question_decision else fsm_state.get("difficulty_level", 2)
    target_diff = max(1, min(3, int(raw_diff)))
    decision_directive = getattr(question_decision, "directive", "Generate ONE concise follow-up question.") if question_decision else "Generate ONE concise follow-up question."
    selected_question = getattr(question_decision, "selected_question", None) if question_decision else None
    spec = getattr(question_decision, "followup_spec", None) or {} if question_decision else {}
    plan = candidate_dict.get("interview_plan") or {}
    per_proj = max(1, int(plan.get("questions_per_project") or 4))
    project_names = [str(p) for p in (plan.get("project_names") or projects)[:2] if p]
    proj_line = " then ".join(project_names) if project_names else "the resume projects"
    project_contexts = candidate_dict.get("project_contexts") or {}

    stage = fsm_state.get("stage", "project_deep_dive")
    skill_phase = (
        stage == "skills_assessment"
        or str(target_topic).lower().startswith("skill:")
        or bool(getattr(question_decision, "skill_kind", None))
    )
    style_rule = _style_block(interview_style, target_role, skill_phase=skill_phase)
    bank_q = (selected_question or {}).get("question") if selected_question else (spoken_seed or seed_q)

    if q_mode == "RETRIEVE":
        mode_instruction = (
            "MODE: RETRIEVE — PHRASE THE SELECTED BANK QUESTION\n"
            f"You are live with {c_name}. Policy already picked the question. You do not choose a new topic.\n"
            f"Speak this ask in natural spoken English, bound to their resume:\n"
            f"ASK: {bank_q}\n"
            f"You may add a short acknowledgment (max 8 words) that reacts to what they just said.\n"
            "Keep the same technical / behavioral ask. Do not swap in a different question.\n"
            f"{style_rule}\n"
            f"Stay on {target_topic} at difficulty {target_diff}/3. One question. 12-40 spoken words."
        )
        temperature = 0.35
    elif q_mode == "TEMPLATE":
        if skill_phase:
            mode_instruction = (
                "MODE: TEMPLATE — SPEAK THE PLANNED SKILL QUESTION\n"
                f"You are live with {c_name}. Projects are finished. This turn is {target_topic} only.\n"
                f"PLANNED ASK (keep this meaning): {seed_q or spoken_seed}\n"
                f"FALLBACK IF YOU DRIFT: {spoken_seed or seed_q}\n"
                "Do this:\n"
                "1. You may say a short bridge like 'Let's switch to OOPs' or 'Next, DSA' (max 8 words).\n"
                "2. Then ask the planned skill question. Same concept: polymorphism stays polymorphism, "
                "hash maps stay hash maps.\n"
                "3. Do NOT mention any project name. Do NOT reuse tools from their last project answer.\n"
                "4. Do not add a second question.\n"
                f"{style_rule}\n"
                f"One question. 12-40 spoken words. Stay on {target_topic}."
            )
        else:
            mode_instruction = (
                "MODE: TEMPLATE — REPHRASE THE PLANNED ASK, DO NOT REPLACE IT\n"
                f"You are live with {c_name}. Policy already chose the technical ask. You only make it sound spoken.\n"
                f"PLANNED ASK (keep this meaning): {seed_q or spoken_seed}\n"
                f"FALLBACK IF YOU DRIFT: {spoken_seed or seed_q}\n"
                "Do this:\n"
                "1. React in 3-8 words to something concrete they just said.\n"
                "2. Ask the planned question in natural interviewer English — not a quiz card, not a definition dump.\n"
                "3. Keep the same technical target (same concept, same difficulty). You may tighten wording.\n"
                "4. Do not switch topic, skill, or project. Do not add a second question.\n"
                f"{style_rule}\n"
                f"Stay on {target_topic} at difficulty {target_diff}/3. One question. 12-40 spoken words."
            )
        temperature = 0.35
    else:
        anchor = spec.get("anchor_term") or primary_project
        must_probe = spec.get("must_probe") or "a concrete implementation detail"
        mentioned = ", ".join(spec.get("mentioned_terms") or []) or "none extracted"
        fallback = spec.get("spoken_fallback") or spoken_seed or seed_q or ""
        project = spec.get("project_name") or ""
        project_resume_context = str(project_contexts.get(project) or "")
        if not project_resume_context:
            project_resume_context = "\n".join(str(chunk) for chunk in resume_chunks[:2])
        stay_on = f"Stay on '{project}' and say that name.\n" if project else f"Stay on {target_topic}.\n"
        prev_probe = spec.get("previous_probe") or ""
        prev_anchor = spec.get("previous_anchor") or ""
        last_excerpt = spec.get("previous_answer") or spec.get("previous_answer_excerpt") or ""
        q_num = spec.get("ladder_step") or 1
        probe = str(spec.get("probe_type") or "how_built")
        hooks = ", ".join(spec.get("answer_hooks") or []) or mentioned
        last_full = str(candidate_answer or spec.get("last_answer") or "").strip()
        prior_full = str(last_excerpt or "").strip()
        if prior_full and prior_full == last_full:
            prior_full = ""
        probe_hint = {
            "what_used": "Backup only if the answer was thin: ask what they implemented with the named tool.",
            "how_built": "Backup only if the answer was thin: ask how they built the thing they just named.",
            "regularization": "Backup only if they already talked about a model/training: ask what they changed when it overfit or failed on held-out data.",
            "metric": "Backup only if they already talked about results: ask which metric they trusted.",
            "why_simple": "Backup only if the answer was thin: ask why they picked that option.",
        }.get(probe, "Follow a concrete claim from their last answer.")
        thread_line = (
            f"This is question {q_num} of {per_proj} on this project. "
            f"You already asked about '{prev_probe}' on '{prev_anchor}'. "
            f"Follow THIS answer, not a canned script. Stay on this project until {per_proj} questions are done.\n"
            if prev_probe
            else (
                f"This is question 1 of {per_proj} on this project. "
                "Name the project, then ask about something they just said.\n"
            )
        )
        prior_block = (
            f"PREVIOUS ANSWER ON THIS PROJECT (context only; follow the latest answer):\n\"\"\"{prior_full}\"\"\"\n"
            if prior_full
            else ""
        )
        mode_instruction = (
            "MODE: GENERATE — WRITE THE NEXT INTERVIEW QUESTION FROM THEIR LAST ANSWER\n"
            "You are live. The candidate just answered. Your only job is to follow that answer.\n"
            f"CV CONTEXT FOR THIS PROJECT (facts only; never turn another title into this project):\n"
            f"\"\"\"{project_resume_context[:1200] or '(no additional CV context)'}\"\"\"\n"
            f"{prior_block}"
            f"THEIR LAST ANSWER (follow this — the full thing they just said):\n\"\"\"{last_full or '(thin or empty — use the fallback)'}\"\"\"\n"
            "Do this:\n"
            "1. Read the full last answer. Pick ONE concrete claim they made (a tool, model, step, data problem, or number).\n"
            "2. Repeat that claim in a few words so they hear you were listening.\n"
            "3. Ask ONE deeper question about THAT claim — what they implemented, why, what failed, or how they checked it.\n"
            "4. Do not ask about a technique they never mentioned. No overfitting unless they talked about a model or training. "
            "No hash maps unless they said that. No other project.\n"
            "5. Do not ask a dictionary definition. Do not restart with 'walk me through the project'.\n"
            "Do NOT jump to OOPs or DSA on this turn.\n"
            "If they skipped or said they don't know, do not scold them; ask a new question on the same project using the fallback.\n"
            f"{stay_on}{thread_line}"
            f"Backup terms if the answer is thin: {hooks}\n"
            f"Backup name if it appeared: {anchor}\n"
            f"{probe_hint} ({probe})\n"
            f"Fallback only if you cannot follow their words: {fallback}\n"
            f"{style_rule}\n"
            f"One question. 16-48 words. End with ?. Difficulty {target_diff}/3 on {target_topic}."
        )
        temperature = 0.4

    voice = (
        "- One short bridge, then the planned skill question. No lecture.\n"
        "- Do not name a project or reuse a library from the last project answer unless it is this skill.\n"
        "- Sound curious, not robotic. Keep the planned OOP/DSA/skill concept.\n"
        "- Use their name at most once, and only if it is natural."
        if skill_phase
        else
        "- One short reaction to what they just said, then one question. No lecture.\n"
        "- Use a noun they actually used (a library, model, metric, or project). Avoid \"that\", \"your approach\", \"tell me more\".\n"
        "- Sound like a technical interviewer who has read their answer, not a quiz card.\n"
        "- Use their name at most once, and only if it is natural.\n"
        "- This is one thread. The next question should follow a claim in their last answer, not restart."
    )

    agenda = (
        f"INTERVIEW AGENDA — follow this order. Python owns the stage; you do not skip ahead.\n"
        f"1. Short intro.\n"
        f"2. {per_proj} questions on project 1 ({proj_line.split(' then ')[0] if project_names else 'first project'}).\n"
        f"3. {per_proj} questions on project 2"
        f"{(' (' + project_names[1] + ')') if len(project_names) > 1 else ''}, then stop that project.\n"
        f"4. OOPs, then DSA, then CV skills. Skills are not project follow-ups.\n"
        f"5. Close.\n"
        "TOPIC CHANGES: When the topic changes, first thank them for the previous topic in one short sentence, "
        "name the new topic, then ask ONE question. Never mix two topics in one turn. "
        f"Never leave a project before {per_proj} questions unless Python already changed the stage."
    )

    sys_prompt = f"""You are a sharp technical interviewer in a live {target_role} voice interview with {c_name}.
Speak the way a strong human interviewer speaks on a call: calm, specific, one beat ahead of the candidate.
On projects you follow what they just said: name their claim, then go one level deeper on that same claim.

You do not run the interview. Python already chose the stage, topic, and planned ask.
Your only job is to say the next question out loud so it sounds like a conversation, not a form.

{agenda}

VOICE
{voice}

HARD RULES
1. Output ONLY the spoken line. No JSON, markdown, labels, or prefaces like "Question:".
2. Ask exactly ONE question. End with ?.
3. Never invent employers, projects, tools, or numbers that are not in the profile or their answer.
4. Never repeat a question from PREVIOUSLY_ASKED_QUESTIONS.
5. Stay inside {target_role}. Prefer overlapping skills: {primary_skills}.
6. 16 to 48 spoken words.
7. Yeah / yes / yep / sorry / well / okay are acknowledgements, not tools. Never treat them as technologies.
8. In TEMPLATE mode, keep the planned technical ask. Rephrase it; do not replace it with a different concept.
9. On project questions, follow a specific claim from their last answer. Do not ask about a technique they never mentioned. Never ask about drift, rollback, SLAs, or production failure.
10. On skill questions (OOPs, DSA, Python, …), never mention a project. Skills and projects are separate.
11. Do not jump from project 1 to project 2, or from projects to OOPs/DSA, unless POLICY says TRANSITION. Python will ask DSA after OOPs; you do not skip it.

STAGE: {stage}
POLICY: {decision_directive}

{mode_instruction}"""

    user_payload = {
        "INTERVIEW_STAGE": stage,
        "JOB_ROLE": target_role,
        "INTERVIEW_STYLE": interview_style,
        "QUESTION_MODE": q_mode,
        "POLICY_DIRECTIVE": decision_directive,
        "TARGET_TOPIC": target_topic,
        "TARGET_DIFFICULTY": f"{target_diff}/3",
        "SEED_QUESTION": spoken_seed or seed_q,
        "SELECTED_BANK_QUESTION": bank_q,
        "SELECTED_QUESTION_RUBRIC": {
            "id": selected_question.get("id") if selected_question else None,
            "competency": selected_question.get("competency") if selected_question else None,
            "expected_concepts": selected_question.get("expected_concepts", []) if selected_question else [],
            "strong_signals": selected_question.get("strong_signals", []) if selected_question else [],
        },
        "PREVIOUS_QUESTION_ASKED": last_question or "Initial greeting / introduction",
        "CANDIDATE_LAST_ANSWER": candidate_answer,
        "THREAD_SO_FAR": {
            "project": spec.get("project_name"),
            "probes_already_asked": spec.get("asked_probes") or [],
            "previous_anchor": spec.get("previous_anchor"),
            "previous_answer": spec.get("previous_answer") or spec.get("previous_answer_excerpt"),
        },
        "DETECTED_INTENT": intent,
        "CANDIDATE_PROFILE": {
            "name": c_name,
            "college": candidate_dict.get("college"),
            "degree": candidate_dict.get("degree"),
            "projects": projects,
            "project_context": (
                str(project_contexts.get(spec.get("project_name") or "") or "")
                or "\n".join(str(chunk) for chunk in resume_chunks[:2])
            )[:1200],
            "interview_skills": skills[:10],
            "cv_skills": candidate_dict.get("cv_skills", [])[:10],
            "skill_intersection": sift.get("intersection", []),
            "role_gaps": sift.get("role_gaps", []),
            "cv_only": sift.get("cv_only", []),
            "job_description": (candidate_dict.get("job_description") or "")[:1200],
            "domains": candidate_dict.get("domains", [])[:5],
            "relevant_resume_snippet": resume_chunks[:2]
        },
        "TOPIC_PROBES": question_bank_context[:2],
        "PREVIOUSLY_ASKED_QUESTIONS": fsm_state.get("questions_already_asked", [])[-4:],
        "FOLLOWUP_SPEC": spec,
    }

    return {
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)}
        ],
        "temperature": temperature,
        "max_tokens": 200,
    }

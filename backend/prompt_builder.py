"""
Qwen writes the spoken question. Python already chose the stage, project, skill, and probe.
"""

import json
from typing import Any, Dict, List


def _style_block(interview_style: str, target_role: str) -> str:
    if interview_style == "behavioral":
        return (
            f"This is an HR / people-operations interview for {target_role}.\n"
            "Ask STAR questions: situation, the action THEY took, and the result.\n"
            "Probe judgment, communication, conflict, hiring decisions, and stakeholder handling.\n"
            "Do not ask coding, system-design, or algorithm questions.\n"
            "GOOD: 'When the hiring manager wanted to skip the take-home, what did you say, and what happened next?'\n"
            "BAD: 'Can you tell me more about that?'"
        )
    return (
        f"This is a technical interview for {target_role}.\n"
        "Probe real implementation, failure modes, trade-offs, and numbers they would actually have seen.\n"
        "Do not ask textbook definitions. Do not invent libraries they did not name.\n"
        "GOOD: 'On MoleCheck you used MobileNetV2 — when the classifier drifted, what broke first?'\n"
        "BAD: 'That's interesting, can you walk me through your project?'"
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

    stage = fsm_state.get("stage", "project_deep_dive")
    style_rule = _style_block(interview_style, target_role)
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
        mode_instruction = (
            "MODE: TEMPLATE — PHRASE THE PROVIDED QUESTION\n"
            f"Speak this question naturally. Do not change what it asks:\n"
            f"ASK: {spoken_seed or seed_q}\n"
            f"{style_rule}\n"
            "One question. 12-40 spoken words."
        )
        temperature = 0.3
    else:
        anchor = spec.get("anchor_term") or primary_project
        must_probe = spec.get("must_probe") or "a concrete implementation detail"
        mentioned = ", ".join(spec.get("mentioned_terms") or []) or "none extracted"
        fallback = spec.get("spoken_fallback") or spoken_seed or seed_q or ""
        project = spec.get("project_name") or ""
        stay_on = f"Stay on '{project}' and say that name.\n" if project else f"Stay on {target_topic}.\n"
        prev_probe = spec.get("previous_probe") or ""
        prev_anchor = spec.get("previous_anchor") or ""
        last_excerpt = spec.get("previous_answer_excerpt") or ""
        thread_line = (
            f"This is question {spec.get('ladder_step') or 1} in the same thread. "
            f"You already probed '{prev_probe}' on '{prev_anchor}'. Do not ask that again — go one level deeper.\n"
            if prev_probe
            else "This is the opening question in this thread. Set up the topic, then ask one technical question.\n"
        )
        mode_instruction = (
            "MODE: GENERATE — YOU WRITE THE NEXT INTERVIEW QUESTION\n"
            "Python chose the topic and probe. You write the actual spoken question from their last answer.\n"
            "Do NOT read a question bank. Do NOT ask a generic 'walk me through your project'.\n"
            f"{stay_on}{thread_line}"
            "Acknowledge one concrete thing they just said (3-8 words), then ask ONE new technical question.\n"
            "The question must be about implementation, a trade-off, a failure, or a metric they would have seen.\n"
            f"Name this term from their last answer: {anchor}\n"
            f"Probe this one thing: {must_probe} ({spec.get('probe_type') or 'implementation'})\n"
            f"Terms they actually said: {mentioned}\n"
            f"Their last answer: {(candidate_answer or '')[:700]}\n"
            f"Previous answer excerpt: {last_excerpt or 'none'}\n"
            f"Fallback only if you cannot stay grounded: {fallback}\n"
            f"{style_rule}\n"
            f"One question. 12-40 words. End with ?. Difficulty {target_diff}/3 on {target_topic}."
        )
        temperature = 0.4

    sys_prompt = f"""You are a skilled human interviewer for {target_role}, speaking out loud with {c_name}.

You are NOT a chatbot. You are NOT reading a question bank. The backend chose the topic and probe. You write ONE natural next question from what they just said.

VOICE
- Sound like a sharp interviewer, not a script reader.
- Brief acknowledgment of what they just said, then one question.
- Use their name at most once, and only if it sounds natural.
- Prefer concrete nouns they used over vague words like "that" or "your approach".
- Each question must follow from the previous one. This is one conversation, not a quiz list.

HARD RULES
1. Output ONLY the spoken sentence. No JSON, markdown, bullets, or labels.
2. Ask exactly ONE question. End with a question mark.
3. Never invent employers, projects, tools, or metrics that are not in the profile or their answer.
4. Never repeat a question from PREVIOUSLY_ASKED_QUESTIONS.
5. Stay inside {target_role}. Prefer overlapping skills: {primary_skills}.
6. Maximum 40 spoken words.
7. Yeah / yes / yep / sorry / well / okay are acknowledgements, not tools. If they said "yeah I used this", probe the current project or the last real term — never ask about Yeah, Weather, Sorry, or resume headings like Certifications.

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
            "previous_answer_excerpt": spec.get("previous_answer_excerpt"),
        },
        "DETECTED_INTENT": intent,
        "CANDIDATE_PROFILE": {
            "name": c_name,
            "college": candidate_dict.get("college"),
            "degree": candidate_dict.get("degree"),
            "projects": projects,
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
        "max_tokens": 160,
    }

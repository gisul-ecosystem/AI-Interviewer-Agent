"""
Question Engine: Decision & Routing Layer for AI Technical Interviewer.

Python owns the interview plan (stage, project, skill, probe ladder).
Qwen writes live project follow-ups at BASIC technical depth. Skill questions are locked at CV upload.

  TEMPLATE  — policy seed (skills, skip, close). Skills and close stay scripted; skip may be phrased.
  GENERATE  — project follow-ups (Qwen writes the question from the last answer).

Structured stages:
  1. PROJECT_DEEP_DIVE: 2 questions per project
  2. SKILLS_ASSESSMENT: 3 planned questions per skill
  3. CLOSING
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from interviewer.services.resume import is_resume_metadata_title
from backend.role_sift import normalize_track, sift_skills_for_role
from backend.intent_engine import detect_requested_topic
from backend.followup_engine import plan_followup


STRONG_MATCH_THRESHOLD = 0.75
# TF-IDF on a spoken answer is noisy. Below this, treat it as a bank miss
# and ask a context template instead of a weakly related keyword hit.
MIN_RETRIEVE_SCORE = 0.30


def _infer_track(candidate_dict: Dict[str, Any]) -> str:
    raw_role = candidate_dict.get("target_track") or candidate_dict.get("target_role") or candidate_dict.get("role")
    track = normalize_track(str(raw_role or ""))
    if track != "auto":
        return track
    sift = sift_skills_for_role(candidate_dict, role_override="auto")
    return sift["track"]


def _bind_project_question(question: str, project: str) -> str:
    """Force the CV project name into a bank question that talks about 'a project'."""
    if not project or not question:
        return question
    if project.lower() in question.lower():
        return question
    p_clean = project.strip()
    if re.search(r"(?i)\bchoose\s+one\s+project\b", question):
        rest = re.sub(r"(?i)^choose\s+one\s+project\s+(?:from|on)?\s*(?:your\s+resume)?[\s:.,–—-]+", "", question)
        return f"Let's talk about {p_clean}. {rest}"
    if re.search(r"(?i)\bwalk\s+me\s+through\s+one(?:\s+\w+)?\s+project\b", question):
        rest = re.sub(r"(?i)^walk\s+me\s+through\s+one(?:\s+\w+)?\s+project\s+(?:from|on)?\s*(?:your\s+resume)?[\s:.,–—-]+", "", question)
        return f"Walk me through your project {p_clean}. {rest}"
    rewritten = re.sub(r"(?i)\b(?:one(?:\s+\w+)?\s+project|a\s+project)(?:\s+(?:from|on)\s+(?:your\s+)?resume)?\b", p_clean, question)
    rewritten = re.sub(r"(?i)\b(?:one\s+of\s+)?your(?:\s+resume)?\s+projects?\b", p_clean, rewritten)
    if p_clean.lower() in rewritten.lower() and rewritten != question:
        return re.sub(r"\s+", " ", rewritten).strip()
    if question.endswith("?"):
        return f"On {p_clean} — {question}"
    return f"On {p_clean}: {question}"


_STAGE_PROJECT_NAMES = frozenset({
    "project deep dive", "project_deep_dive", "a project on your resume",
    "resume projects", "technical interview", "main project",
    "skills assessment", "warmup", "closing",
})


def _cv_project_name(raw: Any, projects: List[str] | None = None) -> str:
    name = str(raw or "").strip()
    name = re.sub(r"^(project|skill)\s*:\s*", "", name, flags=re.I).strip()
    folded = name.lower().replace("_", " ")
    pool = [p for p in (projects or []) if p and not is_resume_metadata_title(str(p))]
    if not name or folded in _STAGE_PROJECT_NAMES or is_resume_metadata_title(name) or "deep dive" in folded:
        return pool[0] if pool else "a project on your resume"
    if len(name.split()) > 8:
        return pool[0] if pool else "a project on your resume"
    return name


def _format_probes(records: List[Dict[str, Any]]) -> List[str]:
    probes = []
    for record in records:
        topic = record.get("topic", "")
        question = record.get("question", "")
        score = record.get("similarity_score")
        if score is not None:
            probes.append(f"[{topic}] {question} (score: {float(score):.2f})")
        else:
            probes.append(f"[{topic}] {question}")
    return probes


@dataclass
class QuestionDecision:
    mode: str  # "TEMPLATE" | "RETRIEVE" | "GENERATE"
    seed_question: Optional[str]
    seed_topic: Optional[str]
    similarity_score: float
    target_topic: str
    target_difficulty: int  # 1 to 3
    directive: str
    probes: List[str]
    selected_question: Optional[Dict[str, Any]] = None
    spoken_question: Optional[str] = None
    followup_spec: Optional[Dict[str, Any]] = None
    skill_kind: Optional[str] = None  # "foundation" (OOP/DSA) | "cv"
    block_projects: Optional[List[str]] = None

    @property
    def needs_llm(self) -> bool:
        """Policy GENERATE: Qwen must invent the wording of a grounded follow-up."""
        return self.mode == "GENERATE"

    def wants_qwen_phrasing(self, intent: str = "") -> bool:
        """Qwen phrases project follow-ups. Skill / repeat / wrap-up stay scripted."""
        if intent == "REPEAT_REQUEST":
            return False
        if self.mode == "GENERATE":
            return True
        if self.mode != "TEMPLATE":
            return False
        topic = f"{self.seed_topic or ''} {self.target_topic or ''}".lower()
        if any(token in topic for token in ("closing", "completed", "wrap-up", "wrap up")):
            return False
        if self.skill_kind or "skill:" in topic:
            return False
        return True


class QuestionEngine:
    def __init__(self, strong_threshold: float = STRONG_MATCH_THRESHOLD):
        self.strong_threshold = strong_threshold

    def _bank_record(
        self,
        question_bank_rag: Any,
        question_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not question_id or not question_bank_rag or not hasattr(question_bank_rag, "get_record"):
            return None
        return question_bank_rag.get_record(question_id)

    def _retrieve(
        self,
        question_bank_rag: Any,
        query: str,
        exclude_questions: List[str],
        track: Optional[str],
        difficulty: int,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not question_bank_rag or not hasattr(question_bank_rag, "retrieve_scored_records"):
            return []
        hits = question_bank_rag.retrieve_scored_records(
            query=query,
            top_k=max(top_k, 8),
            exclude_questions=exclude_questions,
            track=track,
            target_difficulty=difficulty,
        )
        scored = [
            hit for hit in (hits or [])
            if float(hit.get("similarity_score") or 0.0) >= MIN_RETRIEVE_SCORE
        ]
        if not scored:
            return []
        asked = {(q or "").strip().lower() for q in (exclude_questions or []) if q}
        unused = [h for h in scored if str(h.get("question") or "").strip().lower() not in asked]
        pool = unused or scored
        rotate = len(exclude_questions or []) % len(pool)
        pick = pool[rotate]
        rest = [h for h in pool if h is not pick]
        return [pick] + rest[: max(0, top_k - 1)]

    def _fresh_spoken(self, spoken: str, exclude_questions: List[str]) -> bool:
        needle = (spoken or "").strip().lower()
        if not needle:
            return False
        for prev in exclude_questions or []:
            prev_l = (prev or "").strip().lower()
            if not prev_l:
                continue
            if needle == prev_l or needle in prev_l or prev_l in needle:
                return False
        return True

    def _qwen_question(
        self,
        *,
        candidate_answer: str,
        eval_res: Dict[str, Any],
        candidate_dict: Dict[str, Any],
        topic: str,
        difficulty: int,
        focus: str,
        project_thread: Optional[Dict[str, Any]],
        project_name: str,
        interview_style: str,
        llm_anchor: str,
        exclude_questions: List[str],
        seed_topic: str,
        target_topic: str,
        directive: str,
        spoken_override: Optional[str] = None,
        selected_question: Optional[Dict[str, Any]] = None,
        probes: Optional[List[str]] = None,
    ) -> QuestionDecision:
        """Policy picks topic + probe; Qwen will write the spoken question."""
        spec = plan_followup(
            candidate_answer,
            eval_res=eval_res,
            candidate_dict=candidate_dict,
            topic=topic,
            difficulty=difficulty,
            focus=focus,
            selected_question=selected_question,
            project_thread=project_thread,
            project_name=project_name,
            interview_style=interview_style,
            llm_anchor=llm_anchor,
            asked_questions=exclude_questions,
        )
        spoken = spoken_override or spec["spoken_fallback"]
        if not self._fresh_spoken(spoken, exclude_questions):
            alt = {
                "what_used": "how_built",
                "how_built": "why_simple",
                "why_simple": "what_used",
                "implementation": "how_built",
                "tradeoff": "how_built",
                "metric": "what_used",
            }.get(focus, "how_built")
            spec = plan_followup(
                candidate_answer,
                eval_res=eval_res,
                candidate_dict=candidate_dict,
                topic=topic,
                difficulty=difficulty,
                focus=alt,
                selected_question=selected_question,
                project_thread=project_thread,
                project_name=project_name,
                interview_style=interview_style,
                llm_anchor=llm_anchor,
                asked_questions=exclude_questions,
            )
            spoken = spec["spoken_fallback"]
        spec = {**spec, "spoken_fallback": spoken}
        return QuestionDecision(
            mode="GENERATE",
            seed_question=spoken,
            seed_topic=seed_topic,
            similarity_score=0.0,
            target_topic=target_topic,
            target_difficulty=difficulty,
            directive=directive,
            probes=probes or [],
            selected_question=selected_question,
            spoken_question=spoken,
            followup_spec=spec,
        )

    def decide(
        self,
        candidate_answer: str,
        intent: str,
        fsm_state: Dict[str, Any],
        eval_res: Dict[str, Any],
        candidate_dict: Dict[str, Any],
        question_bank_rag: Any = None,
        exclude_questions: List[str] = None,
        interview_plan: Optional[Dict[str, Any]] = None,
        llm_anchor: str = "",
    ) -> QuestionDecision:
        """
        Evaluates turn context, FSM state, and RAG similarity to produce a QuestionDecision.
        Enforces structured project-then-skills process with difficulty strictly in [1, 3].
        """
        exclude_questions = exclude_questions or fsm_state.get("questions_already_asked", [])
        stage = fsm_state.get("stage", "project_deep_dive")
        raw_diff = fsm_state.get("difficulty_level", 2)
        difficulty = max(1, min(3, int(raw_diff)))
        plan = interview_plan or candidate_dict.get("interview_plan") or {}

        skills = candidate_dict.get("skills") or candidate_dict.get("skill_sift", {}).get("intersection") or []
        projects = [
            p for p in (candidate_dict.get("projects") or [])
            if p and not is_resume_metadata_title(str(p))
        ]
        interview_style = candidate_dict.get("interview_style") or "technical"
        candidate_track = _infer_track(candidate_dict)
        bank_track = candidate_dict.get("bank_track") or plan.get("track") or candidate_track
        c_name = candidate_dict.get("name") or "there"

        # ── 1. Repeat Request ─────────────────────────────────────────────────
        if intent == "REPEAT_REQUEST":
            last_canonical = fsm_state.get("last_canonical_question") or (
                exclude_questions[-1] if exclude_questions else (
                    f"Could you walk me through your project {projects[0]}?" if projects else "Could you walk me through a project on your resume?"
                )
            )
            spoken = f"Sure, no problem. {last_canonical}"
            return QuestionDecision(
                mode="TEMPLATE",
                seed_question=last_canonical,
                seed_topic=fsm_state.get("current_topic", "Technical Interview"),
                similarity_score=1.0,
                target_topic=fsm_state.get("current_topic", "Technical Interview"),
                target_difficulty=difficulty,
                directive="Repeat the previous question verbatim. Do not ask a new question.",
                probes=[],
                spoken_question=spoken,
            )

        requested_topic = detect_requested_topic(candidate_answer, candidate_skills=skills)

        # ── 2. Warmup / Self Introduction ──────────────────────────────────────
        if stage == "warmup" or intent == "SELF_INTRO":
            if not projects:
                first_project = "a project on your resume"
            else:
                first_project = _cv_project_name(projects[0], projects)
            if interview_style == "behavioral":
                opener = (
                    f"Thanks {c_name}. Tell me about a real situation involving hiring, conflict, "
                    f"or stakeholder communication from your background."
                )
                directive = (
                    f"Open with a behavioral STAR question about their background. "
                    f"Use what they just said. Do not ask a coding question."
                )
            else:
                opener = (
                    f"Thanks {c_name}. Let's start with {first_project} — what problem did it solve, "
                    f"and which main tools or models did you use?"
                )
                directive = (
                    f"Open on '{first_project}'. React to their intro. Ask ONE basic technical "
                    f"question that names the project and a tool they used. Stay at simple terms."
                )
            return self._qwen_question(
                candidate_answer=candidate_answer,
                eval_res=eval_res,
                candidate_dict=candidate_dict,
                topic=first_project,
                difficulty=1 if interview_style != "behavioral" else difficulty,
                focus="what_used",
                project_thread={},
                project_name=first_project if interview_style != "behavioral" else "",
                interview_style=interview_style,
                llm_anchor=llm_anchor,
                exclude_questions=exclude_questions,
                seed_topic=f"Project: {first_project}",
                target_topic=f"Project: {first_project}",
                directive=directive,
                spoken_override=opener,
            )

        # ── 3. Explicit topic request ─────────────────────────────────────────
        if requested_topic and intent in ("UNKNOWN_OR_SKIP", "TOPIC_CHANGE"):
            spoken = (
                f"Sure, let's switch to {requested_topic}. What's a real problem you solved with it, "
                f"and how did you know it was working?"
            )
            return self._qwen_question(
                candidate_answer=candidate_answer,
                eval_res=eval_res,
                candidate_dict=candidate_dict,
                topic=requested_topic,
                difficulty=difficulty,
                focus="implementation",
                project_thread={},
                project_name="",
                interview_style=interview_style,
                llm_anchor=llm_anchor,
                exclude_questions=exclude_questions,
                seed_topic=requested_topic,
                target_topic=requested_topic,
                directive=(
                    f"Candidate asked for {requested_topic}. Write ONE practical question on that topic "
                    f"tied to something they just said. Do not pull a generic bank prompt."
                ),
                spoken_override=spoken,
            )

        # ── 4. Skip / Don't Know — move on, never repeat the skipped question ─
        if intent in ("UNKNOWN_OR_SKIP", "TOPIC_CHANGE") or eval_res.get("is_skip"):
            target_diff = difficulty
            ack = "No worries at all, let's pivot."
            if stage in ("project_deep_dive", "technical"):
                p_idx = fsm_state.get("current_project_index", 0)
                current_p = _cv_project_name(
                    projects[p_idx] if p_idx < len(projects) else (projects[0] if projects else ""),
                    projects,
                )
                alt_skill = skills[0] if skills else ""
                spoken = (
                    f"{ack} On {current_p}, what problem did you personally own, "
                    f"and how did you know it was working?"
                )
                if alt_skill and p_idx > 0:
                    spoken = (
                        f"{ack} Have you worked with {alt_skill} in a real system — "
                        f"what did you use it for, and what broke when it didn't go as planned?"
                    )
                if not self._fresh_spoken(spoken, exclude_questions):
                    spoken = (
                        f"{ack} Different angle on {current_p}: what would you change if you rebuilt it?"
                    )
                return QuestionDecision(
                    mode="TEMPLATE",
                    seed_question=spoken,
                    seed_topic=current_p,
                    similarity_score=0.0,
                    target_topic=f"Project: {current_p}",
                    target_difficulty=target_diff,
                    directive="Acknowledge the skip without penalty, then phrase a new question on the next topic. Do not scold them.",
                    probes=[],
                    spoken_question=spoken,
                )

            s_idx = fsm_state.get("current_skill_index", 0)
            current_s = skills[s_idx] if s_idx < len(skills) else (skills[0] if skills else "another skill on your resume")
            next_s = skills[s_idx + 1] if s_idx + 1 < len(skills) else current_s
            spoken = (
                f"{ack} Have you worked with {next_s}? How did you use it in a real system, "
                f"and what broke when it didn't go as planned?"
            )
            if not self._fresh_spoken(spoken, exclude_questions):
                spoken = (
                    f"{ack} Different angle on {current_s}: when would you choose not to use it?"
                )
            return QuestionDecision(
                mode="TEMPLATE",
                seed_question=spoken,
                seed_topic=current_s,
                similarity_score=0.0,
                target_topic=f"Skill: {current_s}",
                target_difficulty=target_diff,
                directive="Acknowledge the skip without penalty and ask a new practical question.",
                probes=[],
                spoken_question=spoken,
            )

        # ── 5. Phase 1: Project Deep-Dive ─────────────────────────────────────
        if stage in ("project_deep_dive", "technical"):
            p_idx = fsm_state.get("current_project_index", 0)
            if not projects:
                current_project = "a project on your resume"
            else:
                if p_idx >= len(projects):
                    p_idx = len(projects) - 1
                current_project = _cv_project_name(projects[p_idx], projects)
            p_q_count = fsm_state.get("project_question_count", 0)
            thread = fsm_state.get("project_thread") or {}
            if interview_style != "behavioral":
                difficulty = 1

            if p_q_count == 0:
                if p_idx > 0:
                    prev_project = _cv_project_name(projects[p_idx - 1], projects)
                    opener = (
                        f"Thanks for walking through {prev_project}. Let's look at {current_project} — "
                        f"what problem did it solve, and which main tools or models did you use?"
                    )
                else:
                    opener = (
                        f"Let's start with {current_project}. What problem did it solve, "
                        f"and which main tools or models did you use?"
                    )
                return self._qwen_question(
                    candidate_answer=candidate_answer,
                    eval_res=eval_res,
                    candidate_dict=candidate_dict,
                    topic=current_project,
                    difficulty=difficulty,
                    focus="what_used",
                    project_thread={},
                    project_name=current_project,
                    interview_style=interview_style,
                    llm_anchor=llm_anchor,
                    exclude_questions=exclude_questions,
                    seed_topic=f"Project: {current_project}",
                    target_topic=f"Project: {current_project}",
                    directive=(
                        f"Open '{current_project}'. Use their last answer. Ask ONE basic technical "
                        f"question that names the project. Stay at simple terms. Do not reuse a bank question."
                    ),
                    spoken_override=opener,
                )

            if p_q_count >= 2:
                focus = "why_simple"
            elif p_q_count >= 1:
                focus = "how_built"
            else:
                focus = "what_used"
            prior = thread.get("last_probe") or "the opener"
            prior_anchor = thread.get("last_anchor") or current_project
            return self._qwen_question(
                candidate_answer=candidate_answer,
                eval_res=eval_res,
                candidate_dict=candidate_dict,
                topic=current_project,
                difficulty=difficulty,
                focus=focus,
                project_thread=thread,
                project_name=current_project,
                interview_style=interview_style,
                llm_anchor=llm_anchor,
                exclude_questions=exclude_questions,
                seed_topic=f"Project: {current_project}",
                target_topic=f"Project: {current_project}",
                directive=(
                    f"FOLLOW-UP: Stay on '{current_project}'. Last probe was '{prior}' on '{prior_anchor}'. "
                    f"Write ONE basic technical question from what they just said. "
                    f"Ask what a tool is, how they used it, or why they picked it. "
                    f"Do not go deeper. Do not invent libraries."
                ),
            )

        # ── 6. Phase 2: Skills Assessment (all 3 questions locked at upload) ──
        if stage in ("skills_assessment", "behavioral"):
            if not skills:
                spoken = (
                    f"That's all I needed on the technical side. Thanks {c_name} — "
                    f"do you have any questions about the role or the team?"
                )
                return QuestionDecision(
                    mode="TEMPLATE",
                    seed_question=spoken,
                    seed_topic="Closing & Candidate Questions",
                    similarity_score=0.0,
                    target_topic="Closing & Candidate Questions",
                    target_difficulty=1,
                    directive="No intersecting skills; close after projects.",
                    probes=[],
                    spoken_question=spoken,
                )
            s_idx = fsm_state.get("current_skill_index", 0)
            if s_idx >= len(skills):
                s_idx = len(skills) - 1
            current_skill = skills[s_idx]
            s_q_count = fsm_state.get("skill_question_count", 0)
            slots = plan.get("skill_slots") or []
            slot = slots[s_idx] if s_idx < len(slots) else {}
            planned = [
                str(q).strip()
                for q in (slot.get("questions") or [slot.get("base_question")])
                if str(q).strip()
            ]
            slot_kind = str(slot.get("kind") or "cv")
            q_idx = min(max(0, s_q_count), max(0, len(planned) - 1))
            planned_q = planned[q_idx] if planned else (
                f"Explain a concrete {current_skill} failure you have seen, "
                f"including the data structure or API involved."
            )

            if s_q_count == 0:
                if s_idx == 0:
                    prefix = "Let's shift from projects to fundamentals. "
                    if slot_kind != "foundation":
                        prefix = f"Let's shift from projects to {current_skill}. "
                else:
                    prefix = f"Next, {current_skill}. "
            else:
                prefix = ""
            spoken = f"{prefix}{planned_q}".strip()
            return QuestionDecision(
                mode="TEMPLATE",
                seed_question=planned_q,
                seed_topic=f"Skill: {current_skill}",
                similarity_score=1.0,
                target_topic=f"Skill: {current_skill}",
                target_difficulty=difficulty,
                directive=(
                    f"SPEAK planned {current_skill} question {q_idx + 1}/{max(1, len(planned))}. "
                    f"This is a skill question, not a project follow-up. "
                    f"Do not mention resume projects or reuse the last project answer. "
                    f"Keep the planned {current_skill} ask."
                ),
                probes=[],
                selected_question=None,
                spoken_question=spoken,
                skill_kind=slot_kind if slot_kind in ("foundation", "cv") else (
                    "foundation" if any(
                        token in str(current_skill).lower()
                        for token in ("object-oriented", "oop", "data structure", "algorithm", "dsa")
                    ) else "cv"
                ),
                block_projects=projects[:6],
            )

        # ── 7. Closing ────────────────────────────────────────────────────────
        if stage == "completed":
            spoken = f"Thank you {c_name}. This interview is complete — we'll share feedback shortly."
            return QuestionDecision(
                mode="TEMPLATE",
                seed_question=spoken,
                seed_topic="Completed",
                similarity_score=0.0,
                target_topic="Completed",
                target_difficulty=1,
                directive="Interview finished. Do not ask another technical question.",
                probes=[],
                spoken_question=spoken,
            )

        if stage == "closing":
            spoken = (
                f"That's all I needed on the technical side. Thanks {c_name} — "
                f"do you have any questions about the role or the team?"
            )
            return QuestionDecision(
                mode="TEMPLATE",
                seed_question=spoken,
                seed_topic="Closing",
                similarity_score=0.0,
                target_topic="Closing & Wrap-up",
                target_difficulty=1,
                directive="Conclude warmly and invite candidate questions.",
                probes=[],
                spoken_question=spoken,
            )

        spoken = "Could you go one level deeper on the approach you just described?"
        return QuestionDecision(
            mode="TEMPLATE",
            seed_question=spoken,
            seed_topic=fsm_state.get("current_topic", "Technical Architecture"),
            similarity_score=0.0,
            target_topic=fsm_state.get("current_topic", "Technical Architecture"),
            target_difficulty=difficulty,
            directive="Fallback single follow-up.",
            probes=[],
            spoken_question=spoken,
        )


# Singleton instance
question_engine = QuestionEngine()

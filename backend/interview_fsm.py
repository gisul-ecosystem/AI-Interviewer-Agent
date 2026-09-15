"""
Interview Finite State Machine (FSM).
Manages structured sequential stages for a 15-minute interview:
  1. WARMUP (intro, then a spoken bridge into project 1)
  2. PROJECT_DEEP_DIVE (4 questions per CV project)
  3. SKILLS_ASSESSMENT (OOPs + DSA, then 3 questions per CV skill)
  4. CLOSING (wrap-up)
  5. COMPLETED (transcript saved, grading queued)

The 15-minute clock may skip leftover projects or extra CV skills, but it must
not close before Object-Oriented Programming and DSA have been asked.

Difficulty level is strictly bounded between 1 and 3 (1=Foundational, 2=Intermediate, 3=Advanced).
"""

from typing import Any, Dict, Optional, Sequence

from interviewer.services.interview_structure import quotas_from_plan

STAGES = ["warmup", "project_deep_dive", "skills_assessment", "closing", "completed"]

_FOUNDATION_TOKENS = (
    "object-oriented",
    "oop",
    "data structure",
    "algorithm",
    "dsa",
)


def _is_foundation_skill(name: str) -> bool:
    n = str(name or "").lower()
    return any(token in n for token in _FOUNDATION_TOKENS)


def _foundation_incomplete(stage: str, state: Dict[str, Any], skills: Sequence[str]) -> bool:
    """True while OOPs or DSA still have not been asked."""
    if not any(_is_foundation_skill(s) for s in skills):
        return False
    if stage in ("warmup", "project_deep_dive", "technical"):
        return True
    if stage in ("skills_assessment", "behavioral"):
        idx = int(state.get("current_skill_index") or 0)
        return any(_is_foundation_skill(s) for s in list(skills)[idx:])
    return False


class InterviewFSM:
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        if initial_state:
            self.state = dict(initial_state)
            self.state.setdefault("stage", "warmup")
            raw_diff = self.state.get("difficulty_level", self.state.get("difficulty", 2))
            self.state["difficulty_level"] = max(1, min(3, int(raw_diff)))
            self.state.setdefault("current_topic", "Introduction")
            self.state.setdefault("questions_asked", 0)
            self.state.setdefault("questions_remaining", 12)
            self.state.setdefault("time_remaining_seconds", 900)
            self.state.setdefault("topics_covered", [])
            self.state.setdefault("questions_already_asked", [])
            self.state.setdefault("candidate_weaknesses", [])
            self.state.setdefault("topic_scores", {})
            self.state.setdefault("action", "CONTINUE")

            # Tracking for structured project & skill phases
            self.state.setdefault("current_project_index", 0)
            self.state.setdefault("project_question_count", 0)
            self.state.setdefault("current_skill_index", 0)
            self.state.setdefault("skill_question_count", 0)
            self.state.setdefault("project_thread", {})
        else:
            self.state = {
                "stage": "warmup",
                "difficulty_level": 2,  # Strictly 1 (foundational), 2 (intermediate), 3 (advanced)
                "current_topic": "Introduction",
                "questions_asked": 0,
                "questions_remaining": 12,
                "time_remaining_seconds": 900,
                "topics_covered": [],
                "questions_already_asked": [],
                "candidate_weaknesses": [],
                "topic_scores": {},
                "action": "CONTINUE",
                "current_project_index": 0,
                "project_question_count": 0,
                "current_skill_index": 0,
                "skill_question_count": 0,
                "project_thread": {},
            }

    def get_dict(self) -> Dict[str, Any]:
        """Returns internal state dictionary."""
        return self.state

    def update_from_evaluation(
        self,
        eval_result: Dict[str, Any],
        intent: str,
        candidate_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Updates FSM state based on turn evaluation metrics, candidate intent,
        and current phase progress (Projects: 3-4 Qs, Skills: 1-2 Qs, Difficulty: 1-3).
        """
        if intent == "REPEAT_REQUEST":
            self.state["action"] = "REPEAT"
            return self.state

        self.state["questions_asked"] = self.state.get("questions_asked", 0) + 1
        self.state["questions_remaining"] = max(0, self.state.get("questions_remaining", 12) - 1)

        current_topic = self.state.get("current_topic", "General Technical")
        topics_cov = self.state.setdefault("topics_covered", [])
        if current_topic not in topics_cov:
            topics_cov.append(current_topic)

        weaknesses = self.state.setdefault("candidate_weaknesses", [])

        # 1. Handle Skip / Don't Know / Topic Pivot — spend one quota slot, don't punish.
        if eval_result.get("is_skip") or intent in ("UNKNOWN_OR_SKIP", "TOPIC_CHANGE"):
            self.state.setdefault("topic_scores", {})
        # 2. Handle Technical Answer Evaluation & Adaptive Difficulty within [1, 3]
        else:
            score = eval_result.get("score", 0.6)
            self.state.setdefault("topic_scores", {})[current_topic] = score

            missing = eval_result.get("missing_concepts", [])
            for concept in missing:
                if concept not in weaknesses:
                    weaknesses.append(concept)

            current_diff = self.state.get("difficulty_level", 2)
            if score >= 0.75:
                self.state["difficulty_level"] = min(3, current_diff + 1)
            elif score <= 0.40:
                self.state["difficulty_level"] = max(1, current_diff - 1)

        # Ensure difficulty stays strictly between 1 and 3
        self.state["difficulty_level"] = max(1, min(3, self.state.get("difficulty_level", 2)))

        # 3. Handle Structured Stage Transitions
        q_asked = self.state["questions_asked"]
        current_stage = self.state["stage"]

        if candidate_dict is None:
            projects = []
            skills = []
            plan = {}
        else:
            from interviewer.services.resume import is_resume_metadata_title
            projects = [
                p for p in (candidate_dict.get("projects") or [])
                if p and not is_resume_metadata_title(str(p))
            ]
            plan = candidate_dict.get("interview_plan") or {}
            skills = list(plan.get("skill_names") or candidate_dict.get("skills") or [])
        quotas = quotas_from_plan(plan)
        per_project = max(1, quotas["questions_per_project"])
        per_skill = max(1, quotas["questions_per_skill"])
        wrap_after = max(30, quotas["wrap_up_seconds"])
        time_left = int(self.state.get("time_remaining_seconds") or 0)
        foundation_open = _foundation_incomplete(current_stage, self.state, skills)
        time_pressed = time_left <= 0 or (
            time_left <= wrap_after
            and current_stage not in ("warmup", "closing", "completed")
            and q_asked >= 2
        )
        # 15-minute contract: wrap leftover projects/CV skills when time is gone,
        # but never close before OOP + DSA have been asked.
        if foundation_open:
            force_close = False
            cut_to_foundation = time_pressed
        else:
            force_close = time_pressed or self.state["questions_remaining"] <= 0
            cut_to_foundation = False

        if force_close:
            action = "WRAP_UP"
        elif eval_result.get("is_skip") or intent in ("UNKNOWN_OR_SKIP", "TOPIC_CHANGE"):
            action = "PIVOT_TOPIC"
        elif eval_result.get("depth") == "low" or eval_result.get("score", 0.6) < 0.5:
            action = "PROBE_DEEPER"
        else:
            action = "CONTINUE"
        self.state["action"] = action

        def _enter_skills() -> None:
            self.state["stage"] = "skills_assessment"
            self.state["current_skill_index"] = 0
            self.state["skill_question_count"] = 0
            self.state["current_topic"] = f"Skill: {skills[0]}"

        def _enter_closing() -> None:
            self.state["stage"] = "closing"
            self.state["current_topic"] = "Closing & Candidate Questions"

        # Phase 1: Warmup -> Project Deep-Dive (or Skills if no projects)
        if current_stage == "warmup" and q_asked >= 1:
            if not projects:
                if skills and not force_close:
                    _enter_skills()
                else:
                    _enter_closing()
            else:
                self.state["stage"] = "project_deep_dive"
                self.state["current_project_index"] = 0
                self.state["project_question_count"] = 0
                self.state["current_topic"] = f"Project: {projects[0]}"
                if cut_to_foundation and skills:
                    _enter_skills()
                elif force_close:
                    if skills:
                        _enter_skills()
                    else:
                        _enter_closing()

        # Phase 2: Projects (mandatory) then skills
        elif current_stage in ("project_deep_dive", "technical"):
            self.state["stage"] = "project_deep_dive"
            p_idx = self.state.get("current_project_index", 0)
            p_count = self.state.get("project_question_count", 0) + 1
            self.state["project_question_count"] = p_count

            # Quota is a hard cap. PROBE_DEEPER only changes what we ask, not how many.
            should_advance_project = force_close or cut_to_foundation or p_count >= per_project

            if should_advance_project:
                self.state["project_thread"] = {}
                if cut_to_foundation and skills:
                    _enter_skills()
                elif (not force_close) and p_idx + 1 < len(projects):
                    self.state["current_project_index"] = p_idx + 1
                    self.state["project_question_count"] = 0
                    self.state["current_topic"] = f"Project: {projects[p_idx + 1]}"
                elif skills and not force_close:
                    _enter_skills()
                else:
                    _enter_closing()

        # Phase 3: Skills (intersection only), then close
        elif current_stage in ("skills_assessment", "behavioral"):
            if not skills or force_close:
                _enter_closing()
            else:
                self.state["stage"] = "skills_assessment"
                s_idx = self.state.get("current_skill_index", 0)
                s_count = self.state.get("skill_question_count", 0) + 1
                self.state["skill_question_count"] = s_count
                # A skip spends one question of this skill. Stay until the quota is done.
                should_advance_skill = s_count >= per_skill

                if should_advance_skill:
                    if s_idx + 1 < len(skills):
                        self.state["current_skill_index"] = s_idx + 1
                        self.state["skill_question_count"] = 0
                        self.state["current_topic"] = f"Skill: {skills[s_idx + 1]}"
                    else:
                        _enter_closing()

        # Phase 4: candidate answered the wrap-up -> interview is done
        elif current_stage == "closing":
            self.state["stage"] = "completed"
            self.state["action"] = "END_INTERVIEW"

        if self.state["stage"] == "completed":
            self.state["action"] = "END_INTERVIEW"

        return self.state

    def set_project_thread(self, thread: Optional[Dict[str, Any]]):
        """Persist the connected project follow-up thread, or clear it."""
        if thread:
            self.state["project_thread"] = dict(thread)
        else:
            self.state["project_thread"] = {}

    def set_current_topic(self, topic: str):
        """Updates the active interview topic and registers it in topics_covered."""
        if topic and topic.strip():
            clean_t = topic.strip()
            self.state["current_topic"] = clean_t
            if clean_t not in self.state.get("topics_covered", []):
                self.state.setdefault("topics_covered", []).append(clean_t)

    def add_asked_question(self, question_text: str):
        """Logs asked question into deduplication cache. Repeat wrappers are not stored."""
        if not question_text or not question_text.strip():
            return
        clean_q = question_text.strip()
        if clean_q.lower().startswith("sure, no problem."):
            return
        self.state["last_canonical_question"] = clean_q
        if "questions_already_asked" not in self.state:
            self.state["questions_already_asked"] = []
        if clean_q not in self.state["questions_already_asked"]:
            self.state["questions_already_asked"].append(clean_q)
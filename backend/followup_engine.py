"""
Grounded follow-up planner.

Quality does not come from asking the LLM to "be a good interviewer".
It comes from:
  1. Anchors from THIS resume + this utterance (Qwen maps "yeah I used this")
  2. Staying on the SAME project and chaining the next probe
  3. Asking a question that names the project and that term
  4. Rejecting generic LLM output and speaking the filled template instead

No global product list. A Rust/gRPC CV works the same as an AIML CV.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "were", "was", "using", "used",
    "into", "then", "have", "been", "they", "them", "for", "our", "you", "your",
    "about", "just", "like", "also", "very", "some", "when", "what", "which",
}

PROJECT_LADDER = ("implementation", "tradeoff", "metric")
FOCUS_TO_PROBE = {
    "implementation": "implementation",
    "failure": "failure_mode",
    "failure_mode": "failure_mode",
    "metric": "metric",
    "tradeoff": "tradeoff",
    "missing_concept": "missing_concept",
}

PROBE_TEMPLATES = {
    "missing_concept": (
        "How did {missing} actually show up in that design — "
        "where did you put it, and what happened if it was wrong?"
    ),
    "failure_mode": (
        "When that path misbehaved — timeout, bad input, or drift — what broke first, "
        "and how did you detect it?"
    ),
    "tradeoff": (
        "Why {anchor} instead of the obvious alternative, and where did it bottleneck?"
    ),
    "metric": (
        "How did you measure whether {anchor} was working — accuracy, latency, error rate — "
        "and what would have made you roll it back?"
    ),
    "implementation": (
        "Walk me through the pipeline you personally built inside {anchor}: "
        "which component did you own, and how did data move to the next step?"
    ),
}

CONNECTED_BEHAVIORAL_TEMPLATES = {
    "implementation": (
        "On {project}, you mentioned {anchor}. What was the situation, what did you personally do, "
        "and what was the result?"
    ),
    "failure_mode": (
        "Staying with {project}: when {anchor} went badly, how did you handle the people involved, "
        "and what would you do differently now?"
    ),
    "metric": (
        "Still on {project} — how did you know {anchor} actually worked? "
        "What changed for the candidate, hiring manager, or team?"
    ),
    "missing_concept": (
        "On {project} you mentioned {anchor}. Where did {missing} show up, "
        "and how did you handle that conversation?"
    ),
    "tradeoff": (
        "On {project}, why {anchor} instead of the easier option, and who did that decision affect?"
    ),
}

CONNECTED_PROJECT_TEMPLATES = {
    "implementation": (
        "On {project} you used {anchor}. Walk me through how you wired it in — "
        "what inputs it took, what it output, and which piece you personally owned?"
    ),
    "failure_mode": (
        "Staying on {project}: when {anchor} hit an unexpected failure or edge case — "
        "what broke first, and how did you detect and handle it?"
    ),
    "metric": (
        "Still on {project} — which metric told you {anchor} was working, "
        "and what benchmark or validation threshold did you aim for?"
    ),
    "missing_concept": (
        "On {project} you mentioned {anchor}. Where did {missing} actually sit in that pipeline, "
        "and what happened if it was wrong?"
    ),
    "tradeoff": (
        "On {project}, why {anchor} instead of alternative approaches — "
        "what trade-offs did you evaluate, and where did it bottleneck?"
    ),
}

SAME_ANCHOR_PROJECT_TEMPLATES = {
    "implementation": (
        "On {project}, walk me through the core pipeline or architecture you personally built — "
        "which component did you own, and how did data flow through to the end result?"
    ),
    "failure_mode": (
        "On {project}, when an unexpected error or edge case occurred — "
        "what broke first, and how did you catch and handle it safely?"
    ),
    "metric": (
        "On {project}, which validation metric or benchmark did you track — "
        "and what target told you it was ready for use?"
    ),
    "missing_concept": (
        "On {project}, how did {missing} actually show up in that design, "
        "and what happened if it was wrong?"
    ),
    "tradeoff": (
        "On {project}, why that architecture over the obvious alternative, and where did it bottleneck?"
    ),
}

SAME_ANCHOR_BEHAVIORAL_TEMPLATES = {
    "implementation": (
        "On {project}, what was the situation, what did you personally do, and what was the result?"
    ),
    "failure_mode": (
        "Staying with {project}: when it went badly, how did you handle the people involved, "
        "and what would you do differently now?"
    ),
    "metric": (
        "Still on {project} — how did you know it actually worked for the people involved?"
    ),
    "missing_concept": (
        "On {project}, where did {missing} show up, and how did you handle that conversation?"
    ),
    "tradeoff": (
        "On {project}, why that approach instead of the easier option, and who did that decision affect?"
    ),
}

GENERIC_FOLLOWUP = re.compile(
    r"\b("
    r"tell me more|can you elaborate|could you elaborate|walk me through that|"
    r"can you explain|could you explain|what do you think|interesting|"
    r"that's great|that is great|thanks for sharing"
    r")\b",
    re.IGNORECASE,
)

GENERIC_PROJECTS = frozenset({
    "", "that part", "a project on your resume", "your recent project", "the project",
    "project deep dive", "project_deep_dive", "resume projects", "main project",
})


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _clean_project_name(topic: str, project_name: str = "") -> str:
    name = (project_name or "").strip() or topic.replace("Project: ", "").replace("Skill: ", "").strip()
    folded = name.lower().replace("_", " ")
    if folded in GENERIC_PROJECTS or "deep dive" in folded:
        return "that part"
    return name or "that part"


def is_valid_anchor(term: str, allowed: Optional[set[str]] = None) -> bool:
    """True for a real tool/project name — never spoken English or a CV section title."""
    from interviewer.services.anchor import CAMELISH, ENGLISH_FUNCTION

    t = str(term or "").strip()
    if len(t) < 3 or t.lower() in ENGLISH_FUNCTION:
        return False
    if allowed is not None:
        allowed_fold = {a.lower() for a in allowed}
        return t.lower() in allowed_fold
    # No CV in hand: camelCase / versioned names only. "Yeah" and "Certifications" fail.
    if CAMELISH.search(t):
        return True
    return bool(t.isupper() and t.isalpha() and 2 <= len(t) <= 8)


def _next_probe(asked: List[str], focus: str, missing: List[str], depth: str) -> str:
    used = {str(p) for p in asked if p}
    requested = FOCUS_TO_PROBE.get(focus or "", "")
    if requested and requested not in used:
        return requested
    if missing and depth != "high" and "missing_concept" not in used:
        return "missing_concept"
    for step in PROJECT_LADDER:
        if step not in used:
            return step
    return requested or "metric"


_ANSWER_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+#./-]{2,}\b")


SPOKEN_TECH_PHRASES = (
    "data augmentation", "transfer learning", "fine tuning", "fine-tuned",
    "cross entropy", "focal loss", "learning rate", "batch size",
    "overfitting", "class imbalance", "confusion matrix", "precision recall",
    "sensitivity", "specificity", "quantization", "dropout", "batch norm",
    "early stopping", "feature engineering", "hyperparameter",
    "inference", "training loop", "data loader", "dataloader",
    "binary classification", "skin lesion", "correlation",
)


def salient_answer_term(answer: str, exclude: Optional[List[str]] = None) -> str:
    """
    A concrete thing the candidate just said, when no CV term matched.

    Keeps the next question tied to their own words instead of falling back to
    the project title and sounding like a fresh, unrelated question.
    """
    from interviewer.services.anchor import CAMELISH, ENGLISH_FUNCTION
    from interviewer.services.resume import TECH_VOCAB

    blocked = {str(x).lower() for x in (exclude or []) if x} | ENGLISH_FUNCTION | STOPWORDS
    text = answer or ""
    lowered = text.lower()
    for phrase in sorted(SPOKEN_TECH_PHRASES, key=len, reverse=True):
        if phrase in lowered and phrase not in blocked:
            return phrase
    for tech in TECH_VOCAB:
        if tech.lower() in blocked:
            continue
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(tech)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE):
            return tech
    words = _ANSWER_TERM_RE.findall(text)
    camel = [
        w for w in words
        if "-" not in w and CAMELISH.search(w) and w.lower() not in blocked and not w.isupper()
    ]
    if camel:
        return camel[0]
    proper = [
        w for w in words[1:]
        if w[:1].isupper() and "-" not in w and w.lower() not in blocked and not w.isupper()
    ]
    if proper:
        return proper[0]
    return ""


def _pick_anchor(
    mentioned: List[str],
    project: str,
    last_anchor: str,
    topic_fallback: str,
    answer: str = "",
) -> str:
    project_fold = _fold(project)
    tech = []
    for term in mentioned:
        folded = _fold(term)
        if not folded:
            continue
        if project_fold and (folded == project_fold or folded in project_fold):
            continue
        tech.append(term)
    if tech:
        return tech[0]
    spoken_term = salient_answer_term(answer, exclude=[project, last_anchor])
    if spoken_term:
        return spoken_term
    if last_anchor and _fold(last_anchor) != project_fold:
        return last_anchor
    if mentioned:
        return mentioned[0]
    return last_anchor or topic_fallback or project


def extract_mentioned_terms(
    candidate_answer: str,
    candidate_dict: Optional[Dict[str, Any]] = None,
    extra_terms: Optional[List[str]] = None,
    limit: int = 5,
) -> List[str]:
    """CV terms plus technical words the candidate actually said."""
    from interviewer.services.anchor import resume_mentions
    from interviewer.services.resume import TECH_VOCAB

    found = list(resume_mentions(
        candidate_answer or "",
        candidate_dict,
        extra=extra_terms,
        limit=limit,
    ))
    seen = {_fold(t) for t in found}
    spoken = salient_answer_term(candidate_answer, exclude=found)
    if spoken and _fold(spoken) not in seen:
        from interviewer.services.anchor import CAMELISH
        known = spoken.lower() in {p.lower() for p in SPOKEN_TECH_PHRASES} or any(
            _fold(spoken) == _fold(tech) for tech in TECH_VOCAB
        ) or bool(CAMELISH.search(spoken))
        if known or not found:
            found.insert(0, spoken)
            seen.add(_fold(spoken))
    answer = candidate_answer or ""
    for tech in TECH_VOCAB:
        if _fold(tech) in seen:
            continue
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(tech)}(?![A-Za-z0-9])", answer, flags=re.IGNORECASE):
            found.append(tech)
            seen.add(_fold(tech))
        if len(found) >= limit:
            break
    return found[:limit]


def plan_followup(
    candidate_answer: str,
    eval_res: Optional[Dict[str, Any]] = None,
    candidate_dict: Optional[Dict[str, Any]] = None,
    topic: str = "",
    difficulty: int = 2,
    focus: str = "implementation",
    selected_question: Optional[Dict[str, Any]] = None,
    project_thread: Optional[Dict[str, Any]] = None,
    project_name: str = "",
    interview_style: str = "technical",
    llm_anchor: str = "",
    asked_questions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Decide the single probe for this turn.

    Project follow-ups stay on the same resume project and climb
    implementation → failure → metric, using the last answer as the anchor.
    """
    eval_res = eval_res or {}
    missing = [str(m) for m in (eval_res.get("missing_concepts") or []) if m]
    depth = eval_res.get("depth") or "med"
    thread = dict(project_thread or {})
    project = _clean_project_name(topic, project_name or thread.get("project") or "")
    same_project = bool(thread.get("project")) and _fold(thread.get("project")) == _fold(project)
    asked = list(thread.get("asked_probes") or []) if same_project else []
    last_anchor = str(thread.get("last_anchor") or "") if same_project else ""

    mentioned = extract_mentioned_terms(
        candidate_answer,
        candidate_dict=candidate_dict,
        extra_terms=missing + ([project] if project else []) + ([last_anchor] if last_anchor else []),
    )
    topic_name = project or "that part"
    resolved = str(llm_anchor or "").strip()
    if resolved:
        mentioned = [resolved] + [m for m in mentioned if _fold(m) != _fold(resolved)]
        anchor = resolved
    else:
        anchor = _pick_anchor(mentioned, project, last_anchor, topic_name, answer=candidate_answer)

    if project and project.lower() not in GENERIC_PROJECTS:
        probe_type = _next_probe(asked, focus, missing, depth)
    elif missing and depth != "high":
        probe_type = "missing_concept"
    elif focus == "tradeoff" or (mentioned and depth == "high"):
        probe_type = "tradeoff" if difficulty >= 2 else "failure_mode"
    elif focus == "failure" or depth == "low":
        probe_type = "failure_mode"
    elif focus == "metric":
        probe_type = "metric"
    else:
        probe_type = "implementation"

    must_probe = {
        "missing_concept": missing[0] if missing else "a missing concept",
        "failure_mode": "failure handling",
        "tradeoff": "trade-off",
        "metric": "validation metric",
        "implementation": "implementation detail",
    }.get(probe_type, probe_type.replace("_", " "))

    if project and project.lower() not in GENERIC_PROJECTS:
        same_anchor = _fold(anchor) == _fold(project) or not anchor
        if interview_style == "behavioral":
            bank = SAME_ANCHOR_BEHAVIORAL_TEMPLATES if same_anchor else CONNECTED_BEHAVIORAL_TEMPLATES
        else:
            bank = SAME_ANCHOR_PROJECT_TEMPLATES if same_anchor else CONNECTED_PROJECT_TEMPLATES
        template = bank.get(probe_type) or PROBE_TEMPLATES[probe_type]
        spoken = template.format(
            project=project,
            anchor=anchor or project,
            missing=missing[0] if missing else must_probe,
        )
        if project.lower() not in spoken.lower():
            spoken = f"On {project} — {spoken}"
    else:
        template = PROBE_TEMPLATES[probe_type]
        spoken = template.format(anchor=anchor, missing=missing[0] if missing else must_probe)

    weak_signals = (selected_question or {}).get("weak_signals") or []
    strong_signals = (selected_question or {}).get("strong_signals") or []
    new_thread = {
        "project": project,
        "last_anchor": anchor,
        "last_probe": probe_type,
        "asked_probes": asked + [probe_type],
        "answer_excerpt": " ".join((candidate_answer or "").split()[:40]),
    }
    previous_probe = str(thread.get("last_probe") or "") if same_project else ""
    ladder_step = len(asked) + 1
    if project.lower() not in GENERIC_PROJECTS:
        grounding = [project, anchor] + mentioned[:2]
    else:
        grounding = [anchor] + mentioned[:2]

    return {
        "probe_type": probe_type,
        "must_probe": must_probe,
        "anchor_term": anchor,
        "mentioned_terms": mentioned,
        "missing_concepts": missing[:3],
        "depth": depth,
        "spoken_fallback": spoken,
        "weak_signals": weak_signals[:3],
        "strong_signals": strong_signals[:3],
        "required_grounding": [g for g in grounding if g],
        "project_name": project,
        "project_thread": new_thread,
        "interview_style": interview_style,
        "previous_anchor": last_anchor,
        "previous_probe": previous_probe,
        "previous_answer_excerpt": str(thread.get("answer_excerpt") or "") if same_project else "",
        "ladder_step": ladder_step,
        "asked_probes": asked,
        "asked_questions": [str(q) for q in (asked_questions or []) if q][-8:],
    }


def mentions_project(text: str, project: str) -> bool:
    """Accept the full project title or its distinctive first token (MoleCheck)."""
    if not project or project.lower() in GENERIC_PROJECTS:
        return True
    lowered = (text or "").lower()
    if project.lower() in lowered:
        return True
    head = re.split(r"[\s\-–—|:]+", project.strip())[0]
    return len(head) >= 4 and head.lower() in lowered


def is_quality_followup(text: str, spec: Optional[Dict[str, Any]] = None) -> bool:
    """Reject generic or ungrounded LLM follow-ups."""
    if not text:
        return False
    clean = " ".join(text.split()).strip()
    clean = re.sub(
        r"^(thanks|thank you|got it|okay|ok|sure|right)[^?]{0,50}[.!]\s*",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    words = clean.split()
    if len(words) < 8 or len(words) > 55:
        return False
    if GENERIC_FOLLOWUP.search(clean):
        return False
    if "?" not in clean:
        return False
    spec = spec or {}
    project = str(spec.get("project_name") or (spec.get("project_thread") or {}).get("project") or "").strip()
    anchor = str(spec.get("anchor_term") or "").strip()
    mentioned = [str(t) for t in (spec.get("mentioned_terms") or []) if t]
    lowered = clean.lower()
    grounded = mentions_project(clean, project)
    if anchor and len(anchor) >= 3 and anchor.lower() in lowered:
        grounded = True
    if any(term.lower() in lowered for term in mentioned if len(term) >= 3):
        grounded = True
    if not grounded:
        return False
    asked = spec.get("asked_questions") or []
    needle = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    for prev in asked:
        prev_n = re.sub(r"[^a-z0-9 ]+", " ", str(prev or "").lower())
        if prev_n and (needle == prev_n or needle in prev_n or prev_n in needle):
            return False
    return True


def _content_tokens(text: str) -> list[str]:
    return [
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", text or "")
        if t.lower() not in STOPWORDS
    ]


def is_quality_phrasing(llm_text: str, seed: str) -> bool:
    """RETRIEVE/TEMPLATE: keep Qwen only if it still asks the same thing."""
    clean = " ".join((llm_text or "").split()).strip()
    words = clean.split()
    if len(words) < 8 or len(words) > 55:
        return False
    if "?" not in clean:
        return False
    if GENERIC_FOLLOWUP.search(clean):
        return False
    seed_tokens = _content_tokens(seed)[:10]
    if not seed_tokens:
        return True
    lowered = clean.lower()
    overlap = sum(1 for t in seed_tokens if t in lowered)
    return overlap >= min(2, len(seed_tokens))


def finalize_followup(llm_text: str, spec: Optional[Dict[str, Any]] = None) -> str:
    """Use the model only when it named the planned probe; otherwise speak the template."""
    spec = spec or {}
    fallback = spec.get("spoken_fallback") or ""
    if is_quality_followup(llm_text, spec):
        return llm_text.strip()
    return fallback


def finalize_spoken_question(llm_text: str, question_decision: Any, fallback: str = "") -> str:
    """Accept Qwen phrasing when it stays on-policy; otherwise speak the seed."""
    spoken = fallback or getattr(question_decision, "spoken_question", None) or getattr(question_decision, "seed_question", None) or ""
    if not llm_text or not str(llm_text).strip():
        return spoken
    mode = getattr(question_decision, "mode", "") or ""
    spec = getattr(question_decision, "followup_spec", None) or {}
    if mode == "GENERATE":
        if not spec.get("spoken_fallback"):
            spec = {**spec, "spoken_fallback": spoken}
        return finalize_followup(llm_text, spec) or spoken
    seed = spoken or getattr(question_decision, "seed_question", None) or ""
    if is_quality_phrasing(llm_text, seed):
        return str(llm_text).strip()
    return spoken

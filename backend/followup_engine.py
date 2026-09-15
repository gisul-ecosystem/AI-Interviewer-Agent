"""
Grounded follow-up planner.

Quality does not come from asking the LLM to "be a good interviewer".
It comes from:
  1. Anchors from THIS resume + this utterance (Qwen maps "yeah I used this")
  2. Staying on the SAME project and asking the next implementation question
  3. Asking a question that names the project and that term
  4. Rejecting generic or production-senior LLM output and speaking the filled template instead

Project probes go deeper than dictionary definitions: what they used, how they built it,
what they did about overfitting or bad data, and how they measured success.
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

PROJECT_LADDER = ("what_used", "how_built", "regularization", "metric")
DEEP_PROJECT_LADDER = ("implementation", "tradeoff", "metric")
FOCUS_TO_PROBE = {
    "implementation": "how_built",
    "what_used": "what_used",
    "how_built": "how_built",
    "regularization": "regularization",
    "overfitting": "regularization",
    "why_simple": "metric",
    "why": "metric",
    "failure": "regularization",
    "failure_mode": "regularization",
    "metric": "metric",
    "tradeoff": "metric",
    "missing_concept": "how_built",
}
DEEP_FOCUS_TO_PROBE = {
    "implementation": "implementation",
    "failure": "failure_mode",
    "failure_mode": "failure_mode",
    "metric": "metric",
    "tradeoff": "tradeoff",
    "missing_concept": "missing_concept",
}

PROBE_TEMPLATES = {
    "what_used": (
        "What did you actually implement with {anchor}, and which part was yours versus pretrained or off the shelf?"
    ),
    "how_built": (
        "Walk through how you built with {anchor}: data in, training or pipeline steps, and the code you wrote."
    ),
    "regularization": (
        "With {anchor}, did the model overfit or struggle on held-out data? What did you change, and did it help?"
    ),
    "why_simple": (
        "Why did you pick {anchor} instead of a more common option, and what broke if you chose wrong?"
    ),
    "missing_concept": (
        "How did {missing} show up in that work, and what did you personally do about it?"
    ),
    "failure_mode": (
        "When {anchor} failed on unseen data, what went wrong, and what did you change?"
    ),
    "tradeoff": (
        "Why {anchor} instead of a more common option, and what did you give up?"
    ),
    "metric": (
        "How did you evaluate {anchor}? Why that metric, and what did a miss look like?"
    ),
    "implementation": (
        "Walk me through how you used {anchor}: which part did you write, and how did data move?"
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
    "what_used": (
        "On {project} you mentioned {anchor}. What did you actually implement with it — "
        "architecture, layers, or training — and what was yours versus pretrained or a library default?"
    ),
    "how_built": (
        "On {project}, walk me through how you built with {anchor}: the data split, "
        "what went into the model or pipeline, and which part you wrote yourself."
    ),
    "regularization": (
        "On {project} you used {anchor}. Did it overfit or fail on held-out data? "
        "What did you change — dropout, augmentation, early stopping, class weights — and did validation improve?"
    ),
    "why_simple": (
        "On {project}, why {anchor} instead of a more common option, and what would have broken with the easier pick?"
    ),
    "implementation": (
        "On {project} you used {anchor}. What did you implement yourself, and how did data move through it?"
    ),
    "failure_mode": (
        "On {project}, when {anchor} failed on unseen data, what went wrong and what did you change?"
    ),
    "metric": (
        "On {project}, how did you evaluate {anchor}? Why that metric instead of raw accuracy, "
        "and what did a wrong prediction look like?"
    ),
    "missing_concept": (
        "On {project} you mentioned {anchor}. How did {missing} show up there, and what did you do about it?"
    ),
    "tradeoff": (
        "On {project}, why {anchor} instead of a more common option, and what did that choice cost you?"
    ),
}

SAME_ANCHOR_PROJECT_TEMPLATES = {
    "what_used": (
        "On {project}, which model or library did you use, and which part of it did you implement yourself?"
    ),
    "how_built": (
        "On {project}, walk me through how you actually built it: data in, training or pipeline, and the code you wrote."
    ),
    "regularization": (
        "On {project}, models often memorize the training set. What did you do about overfitting or class imbalance, "
        "and how did you know it helped on validation?"
    ),
    "why_simple": (
        "On {project}, why that model or library instead of a simpler baseline, and what failed if you chose wrong?"
    ),
    "implementation": (
        "On {project}, which model or tool did you use, and what did you personally implement?"
    ),
    "failure_mode": (
        "On {project}, when the model failed on unseen data, what went wrong and what did you change?"
    ),
    "metric": (
        "On {project}, which metric did you trust, why not accuracy alone, and what did a miss look like?"
    ),
    "missing_concept": (
        "On {project}, how did {missing} show up in what you built, and what did you do about it?"
    ),
    "tradeoff": (
        "On {project}, why that approach instead of a more common option, and what did you give up?"
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


DEEP_PROJECT_ASK = re.compile(
    r"\b("
    r"bottleneck|drift(?:ed|ing)?|roll\s*back|production|latency|"
    r"broke first|failed first|edge case|system design|distributed|"
    r"throughput|observability|quantization|sla\b"
    r")\b",
    re.IGNORECASE,
)

_THIN_ANSWER = re.compile(
    r"^(yeah|yes|yep|ok|okay|sure|skip|pass|idk|next|no|nah|"
    r"i don't know|i do not know|not sure|nothing)[\s.!?]*$",
    re.IGNORECASE,
)
_ANSWER_METRIC = re.compile(
    r"\b(accuracy|precision|recall|f1|auc|sensitivity|specificity|"
    r"confusion\s+matrix|metric|false\s+negatives?|false\s+positives?)\b",
    re.IGNORECASE,
)
_ANSWER_REG = re.compile(
    r"\b(overfit(?:ting)?|dropout|augment(?:ation)?|class[- ]weights?|"
    r"imbalance|early\s+stopp?(?:ing)?|regulariz(?:e|ed|ation)?|"
    r"validation\s+loss|held[- ]out)\b",
    re.IGNORECASE,
)
_ANSWER_ML = re.compile(
    r"\b(keras|tensorflow|pytorch|sklearn|scikit[- ]learn|cnn|lstm|"
    r"transformer|neural|deep\s+learning|mobilenet\w*|resnet\w*|bert|"
    r"xgboost|random\s+forest|classifiers?|train(?:ed|ing)|epochs?|"
    r"fine[- ]tun(?:e|ed|ing)?)\b",
    re.IGNORECASE,
)


def thin_answer(answer: str) -> bool:
    text = " ".join(str(answer or "").split()).strip()
    if len(text.split()) < 6:
        return True
    return bool(_THIN_ANSWER.match(text))


def probe_from_answer(answer: str, asked: Optional[List[str]] = None) -> str:
    """Pick the next probe from what they just said, not a fixed Q3=overfitting slot."""
    used = {str(p) for p in (asked or []) if p}
    text = answer or ""
    if _ANSWER_METRIC.search(text) and "metric" not in used:
        return "metric"
    if _ANSWER_REG.search(text) and "regularization" not in used:
        return "regularization"
    if _ANSWER_ML.search(text):
        if "how_built" not in used:
            return "how_built"
        if "regularization" not in used:
            return "regularization"
        if "metric" not in used:
            return "metric"
    if "how_built" not in used:
        return "how_built"
    if "metric" not in used:
        return "metric"
    return "what_used"


def _next_probe(
    asked: List[str],
    focus: str,
    missing: List[str],
    depth: str,
    *,
    basic: bool = True,
) -> str:
    used = {str(p) for p in asked if p}
    mapping = FOCUS_TO_PROBE if basic else DEEP_FOCUS_TO_PROBE
    ladder = PROJECT_LADDER if basic else DEEP_PROJECT_LADDER
    requested = mapping.get(focus or "", "")
    if requested and requested not in used:
        return requested
    if not basic and missing and depth != "high" and "missing_concept" not in used:
        return "missing_concept"
    for step in ladder:
        if step not in used:
            return step
    return requested or ("how_built" if basic else "regularization")


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

    Project follow-ups stay on the same resume project and follow the last
    answer. The probe is a backup angle when the utterance is thin.
    """
    eval_res = eval_res or {}
    missing = [str(m) for m in (eval_res.get("missing_concepts") or []) if m]
    depth = eval_res.get("depth") or "med"
    thread = dict(project_thread or {})
    project = _clean_project_name(topic, project_name or thread.get("project") or "")
    same_project = bool(thread.get("project")) and _fold(thread.get("project")) == _fold(project)
    asked = list(thread.get("asked_probes") or []) if same_project else []
    last_anchor = str(thread.get("last_anchor") or "") if same_project else ""
    prior_answers = list(thread.get("recent_answers") or []) if same_project else []
    full_answer = " ".join(str(candidate_answer or "").split())[:4000]
    previous_full = str(prior_answers[-1] or "").strip() if prior_answers else ""

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

    basic_project = interview_style != "behavioral"
    if project and project.lower() not in GENERIC_PROJECTS:
        probe_type = _next_probe(asked, focus, missing, depth, basic=basic_project)
    elif basic_project:
        probe_type = _next_probe(asked, focus, missing, depth, basic=True)
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
        "what_used": "what they implemented with that tool, not a definition",
        "how_built": "how they trained or built it, and which code they wrote",
        "regularization": "what they did about overfitting, class imbalance, or bad validation",
        "why_simple": "why they picked it and what the alternative would have broken",
        "missing_concept": missing[0] if missing else "a missing concept",
        "failure_mode": "what failed on unseen data and what they changed",
        "tradeoff": "why they picked that option and what it cost",
        "metric": "which metric they trusted and what a miss looked like",
        "implementation": "what they built and which part they wrote",
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
        "last_answer": full_answer,
        "recent_answers": ([a for a in prior_answers if a] + ([full_answer] if full_answer else []))[-2:],
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
        "previous_answer_excerpt": previous_full,
        "previous_answer": previous_full,
        "ladder_step": ladder_step,
        "asked_probes": asked,
        "asked_questions": [str(q) for q in (asked_questions or []) if q][-8:],
        "last_answer": full_answer,
        "answer_hooks": [t for t in mentioned[:4] if t and _fold(t) != _fold(project)],
        "thin_answer": thin_answer(candidate_answer),
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
    if spec.get("interview_style") != "behavioral" and spec.get("project_name"):
        if DEEP_PROJECT_ASK.search(clean):
            return False
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
    last_answer = str(spec.get("last_answer") or spec.get("previous_answer_excerpt") or "").strip()
    if last_answer and not spec.get("thin_answer") and not thin_answer(last_answer):
        if not _follows_last_answer(clean, spec, last_answer):
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


def _follows_last_answer(question: str, spec: Dict[str, Any], last_answer: str) -> bool:
    """True when the question names something they actually just said."""
    lowered = (question or "").lower()
    hooks = [str(t) for t in (spec.get("answer_hooks") or spec.get("mentioned_terms") or []) if t]
    anchor = str(spec.get("anchor_term") or "").strip()
    if anchor and len(anchor) >= 3 and anchor.lower() in lowered:
        return True
    if any(term.lower() in lowered for term in hooks if len(term) >= 3):
        return True
    project = str(spec.get("project_name") or "").lower()
    answer_tokens = [
        t for t in _content_tokens(last_answer)
        if len(t) >= 4 and t not in project and t not in {"project", "used", "using", "work", "working"}
    ]
    q_tokens = set(_content_tokens(question))
    return any(t in q_tokens for t in answer_tokens)


SKILL_PROJECT_LEAK = re.compile(
    r"\b("
    r"your project|this project|that project|in one of your projects|"
    r"on your resume|walk me through"
    r")\b",
    re.IGNORECASE,
)


def is_quality_phrasing(
    llm_text: str,
    seed: str,
    *,
    block_projects: Optional[List[str]] = None,
) -> bool:
    """RETRIEVE/TEMPLATE: keep Qwen only if it still asks the same thing."""
    clean = " ".join((llm_text or "").split()).strip()
    words = clean.split()
    if len(words) < 8 or len(words) > 55:
        return False
    if "?" not in clean:
        return False
    if GENERIC_FOLLOWUP.search(clean):
        return False
    if block_projects:
        if SKILL_PROJECT_LEAK.search(clean):
            return False
        for project in block_projects:
            if project and mentions_project(clean, project):
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
    seed = getattr(question_decision, "seed_question", None) or spoken
    topic = f"{getattr(question_decision, 'target_topic', '')} {getattr(question_decision, 'seed_topic', '')}".lower()
    is_skill = bool(getattr(question_decision, "skill_kind", None)) or "skill:" in topic
    block_projects = list(getattr(question_decision, "block_projects", None) or [])
    if is_skill and SKILL_PROJECT_LEAK.search(str(llm_text) or ""):
        return spoken
    if is_quality_phrasing(llm_text, seed, block_projects=block_projects or None):
        return str(llm_text).strip()
    return spoken

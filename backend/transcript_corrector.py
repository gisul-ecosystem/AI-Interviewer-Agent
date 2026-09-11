"""
Dynamic Phonetic STT Name & Term Corrector.
Corrects Speech-to-Text mishearings using candidate resume context and fast LLM phonetic context.
"""

import asyncio
import re

_DYNAMIC_CV_RULES = []
_CACHED_CV_CONTEXT = {}

# Common Indian ASR phonetic mishearings in tech stack & engineering
STATIC_TECH_PHONETICS = [
    (r"\bpie\s*torch\b", "PyTorch"),
    (r"\bpy\s*torch\b", "PyTorch"),
    (r"\bsick\s*it\s*learn\b", "scikit-learn"),
    (r"\bsigh\s*kit\s*learn\b", "scikit-learn"),
    (r"\bsklearn\b", "scikit-learn"),
    (r"\btensor\s*flo\b", "TensorFlow"),
    (r"\btensor\s*flow\b", "TensorFlow"),
    (r"\bras\s*net\b", "ResNet"),
    (r"\brest\s*net\b", "ResNet"),
    (r"\bmovile(?:\s*net)?(?:\s*v\s*2)?\b", "MobileNetV2"),
    (r"\bmobile\s*net(?:\s*v\s*2)?\b", "MobileNetV2"),
    (r"\bmobilenet(?:\s*v\s*2)?\b", "MobileNetV2"),
    (r"\bbittech\b|\bb\s*tech\b|\bb\.\s*tech\b", "B.Tech"),
    (r"\bjbm\b", "JVM"),
    (r"\bsequel\b", "SQL"),
    (r"\bhash\s*map\b", "HashMap"),
    (r"\btree\s*ify\b", "treeify"),
    (r"\blru\s*cache\b", "LRU cache"),
]


def build_dynamic_phonetic_rules_from_cv(cv_features: dict) -> list:
    """Generates phonetic regex replacement rules from extracted CV candidate metadata."""
    rules = []

    # Include static tech phonetic rules
    for pat, rep in STATIC_TECH_PHONETICS:
        rules.append((re.compile(pat, re.IGNORECASE), rep))

    if not cv_features:
        return rules

    full_name_str = cv_features.get("name", "")
    if full_name_str and len(full_name_str.strip()) >= 3:
        clean_name = re.sub(r"^(candidate name:|name:)\s*", "", full_name_str, flags=re.IGNORECASE).strip()
        parts = clean_name.split()
        first = parts[0] if parts else ""
        last = parts[-1] if len(parts) > 1 else ""
        if first:
            rules.append((re.compile(rf"\b{re.escape(first.lower())}\b", re.IGNORECASE), first))
            if last and last.lower() != first.lower():
                rules.append((re.compile(rf"\b{re.escape(last.lower())}\b", re.IGNORECASE), last))
                rules.append((re.compile(rf"\b{re.escape(first.lower())}\s+{re.escape(last.lower())}\b", re.IGNORECASE), clean_name))

    college = cv_features.get("college", "")
    if college and len(str(college).strip()) >= 4:
        token = str(college).strip().split(",")[0].strip()
        rules.append((re.compile(rf"\b{re.escape(token.lower())}\b", re.IGNORECASE), token))

    return rules


def register_cv_features_for_phonetics(cv_features: dict):
    """No-op. Phonetics are per-session; do not mutate process-wide rules."""
    return


def fuzzy_correct_against_hotwords(text: str, hotwords: list, cutoff: float = 0.78) -> str:
    """Word-level fuzzy correction against a session hotword list. No LLM."""
    import difflib

    if not text or not hotwords:
        return text

    hw_lower = {w.lower(): w for w in hotwords}
    hw_keys = list(hw_lower.keys())
    words = text.split()
    corrected = []
    for word in words:
        clean_word = word.strip(".,!?;:'\"").lower()
        if len(clean_word) < 4 or clean_word in hw_lower:
            corrected.append(word)
            continue
        matches = difflib.get_close_matches(clean_word, hw_keys, n=1, cutoff=cutoff)
        if matches:
            prefix = word[: len(word) - len(word.lstrip(".,!?;:'\""))]
            suffix = word[len(word.rstrip(".,!?;:'\"")) :]
            corrected.append(prefix + hw_lower[matches[0]] + suffix)
        else:
            corrected.append(word)
    return " ".join(corrected)


def apply_pattern_list(text: str, patterns: list) -> str:
    """Apply serializable [[pattern, replacement], ...] phonetic rules."""
    if not text or not patterns:
        return text
    result = text
    for item in patterns:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                result = re.sub(item[0], item[1], result, flags=re.IGNORECASE)
        except Exception:
            continue
    return result


def correct_transcript_fast(
    raw_transcript: str,
    candidate_dict: dict = None,
    hotwords: list = None,
    phonetic_patterns: list = None,
    active_topic: str = "",
) -> str:
    """Hot-path STT correction: CV-term restore, regex, then conservative fuzzy. No LLM."""
    if not raw_transcript:
        return raw_transcript
    from interviewer.services.stt_lexicon import protect_transcript

    text = protect_transcript(
        raw_transcript,
        candidate_dict,
        extra_terms=hotwords,
        active_topic=active_topic or "",
    )
    if phonetic_patterns:
        text = apply_pattern_list(text, phonetic_patterns)
    else:
        text = apply_phonetic_rules(text)
    return text


def apply_phonetic_rules(text: str) -> str:
    """Applies static tech phonetics plus optional session-registered CV rules."""
    if not text:
        return text
    res = text
    for pat, rep in STATIC_TECH_PHONETICS:
        try:
            res = re.sub(pat, rep, res, flags=re.IGNORECASE)
        except Exception:
            pass
    for rx, replacement in _DYNAMIC_CV_RULES:
        try:
            if rx.search(res):
                res = rx.sub(replacement, res)
        except Exception:
            pass
    return res


async def correct_transcript_with_llm(raw_transcript: str, candidate_dict: dict = None) -> str:
    """
    Corrects candidate STT mishearings using dynamic phonetic rules and high-precision LLM context.
    Operates on both introductory statements and technical explanations.
    """
    if not raw_transcript or len(raw_transcript.strip()) < 3:
        return raw_transcript

    # 1. Apply fast regex phonetic replacements first
    fast_corrected = apply_phonetic_rules(raw_transcript)

    # 2. Skip LLM for single-word skips or empty queries
    from backend.intent_engine import detect_candidate_intent
    intent = detect_candidate_intent(raw_transcript)
    if raw_transcript.strip().lower() in ["skip", "pass", "idk", "next", "no", "yes"]:
        return raw_transcript.strip()

    # 3. Build candidate context
    cand = candidate_dict or {}

    cand_name = cand.get("name") or "Candidate"
    skills_str = ", ".join(cand.get("skills", [])[:8]) if cand.get("skills") else "Computer Science"
    projects_str = ", ".join(cand.get("projects", [])[:3]) if cand.get("projects") else "Software Engineering"

    sys_msg = (
        "You are an expert Speech-to-Text (STT) phonetic corrector for live technical interviews.\n"
        "Your task: Correct acoustic and phonetic mishearings in candidate speech "
        "(e.g., pie torch -> PyTorch, sick it learn -> scikit-learn, ras net -> ResNet, jbm -> JVM, "
        "sequel -> SQL, mole check -> MoleCheck, aditia -> Aditya, b tech -> B.Tech) "
        "using the candidate resume context.\n"
        "RULES:\n"
        "1. Fix misheard technical tools, libraries, colleges, names, and frameworks.\n"
        "2. Keep the candidate's exact meaning, sentence structure, and conversational flow intact.\n"
        "3. Output ONLY the plain corrected transcript text without quotes, commentary, or markdown."
    )

    user_msg = (
        f"CANDIDATE CONTEXT:\n"
        f"- Name: {cand_name}\n"
        f"- Stated Skills: {skills_str}\n"
        f"- Projects: {projects_str}\n\n"
        f"RAW STT TRANSCRIPT:\n\"{fast_corrected}\""
    )

    try:
        from interviewer.adapters.registry import get_llm
        from interviewer.ports.llm import LLMRequest, LLMTask, Message

        result = await get_llm().complete(
            LLMRequest(
                task=LLMTask.PHRASING,
                messages=[
                    Message(role="system", content=sys_msg),
                    Message(role="user", content=user_msg),
                ],
                temperature=0.1,
                max_tokens=200,
                deadline_ms=4500,
            )
        )
        if result.ok:
            out = result.text.strip()
            out = re.sub(r'^["\']|["\']$', '', out).strip()
            if out and len(out) >= 2:
                return out
    except Exception as exc:
        print(f"[Corrector LLM Exception]: {exc}")

    return fast_corrected

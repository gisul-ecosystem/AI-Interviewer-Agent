"""
Protect resume proper nouns in STT transcripts.

Nemotron often turns MoleCheck into "model" and MobileNetV2 into "movile".
Word-level fuzzy matching against short fragments ("Mole", "Net") makes that
worse, so we correct against full CV terms and their spoken variants only.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable

CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)")
CAMEL_TERM = re.compile(r"\b[A-Z][a-zA-Z]*(?:[A-Z][a-zA-Z0-9]*)+\b")
VERSIONED = re.compile(r"\b[A-Za-z][A-Za-z0-9]+(?:v|V)\d+\b")

# Words we must never fuzzy-replace — they are real English / ML vocabulary.
PROTECTED_WORDS = frozenset({
    "model", "models", "mobile", "check", "net", "network", "system", "image",
    "classifier", "project", "python", "java", "data", "deep", "learning",
    "machine", "neural", "object", "structure", "structures", "algorithm",
    "identification", "predictor", "health", "mental",
})

# Hand-tuned ASR confusions for common resume terms. Keys are canonical lower.
KNOWN_VARIANTS: dict[str, tuple[str, ...]] = {
    "molecheck": (
        "mole check", "molecheck", "mool check", "mall check", "mol check",
        "more check", "mould check", "mole cheque", "model check", "mole-check",
    ),
    "mobilenetv2": (
        "mobile net v2", "mobile net v 2", "mobile net", "mobilenet v2",
        "mobilenet", "movile", "movile net", "mobile network v2", "mobil net",
        "mobile net v two", "mobile v2",
    ),
    "mobilenet": ("mobile net", "movile", "mobil net", "mobile network"),
    "resnet": ("rest net", "ras net", "res net", "resnett"),
    "resnet50": ("res net 50", "rest net 50", "resnet 50", "ras net 50"),
    "efficientnet": ("efficient net", "efficientnet"),
    "pytorch": ("pie torch", "py torch", "pytouch", "pi torch"),
    "tensorflow": ("tensor flo", "tensor flow", "tenser flow"),
    "algosync": ("algo sync", "algo-sync"),
}

# Single-token STT errors. Protected English words are only restored when the
# CV has that term AND the utterance looks like a project/tool name, not "a model".
NAME_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "molecheck": ("model", "models", "mould", "mole"),
    "mobilenetv2": ("movile", "movil"),
    "mobilenet": ("movile", "movil"),
}

ML_MODEL_CONTEXT = re.compile(
    r"\b("
    r"train(?:ed|ing)?|cnn|neural|classif\w*|accurac\w*|inference|pretrained|"
    r"fine[- ]?tun\w*|deep learning|machine learning|\bml\b|language model|"
    r"\bllms?\b|as (?:the|a) model|(?:a|the) models? (?:was|is|were|to|for)"
    r")\b",
    re.IGNORECASE,
)

PROJECT_NAME_USAGE = re.compile(
    r"\b("
    r"built|made|developed|worked on|called|named|my project|"
    r"project called"
    r")\b",
    re.IGNORECASE,
)


def fold_term(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def split_camel(term: str) -> str:
    text = CAMEL_SPLIT.sub(" ", str(term or "").strip())
    return re.sub(r"[\s_\-]+", " ", text).strip()


def collect_cv_terms(features: dict[str, Any] | None, extra_text: str = "") -> list[str]:
    """Full proper nouns from the resume — never 3-letter fragments."""
    features = features or {}
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        text = re.sub(r"\s+", " ", str(raw or "")).strip(" -–—,")
        if not text or text.lower() in {"null", "none"}:
            return
        key = text.lower()
        if key in seen or len(text) < 4:
            return
        seen.add(key)
        found.append(text)
        camel = split_camel(text)
        if camel.lower() != key and len(camel) >= 4 and camel.lower() not in seen:
            seen.add(camel.lower())
            found.append(camel)

    for key in ("name", "college", "degree", "company"):
        add(features.get(key))
    for key in ("skills", "projects", "certifications", "domains", "cv_skills"):
        val = features.get(key) or []
        if isinstance(val, str):
            val = [val]
        for item in val:
            add(item)
            for piece in CAMEL_TERM.findall(str(item)):
                add(piece)
            for piece in VERSIONED.findall(str(item)):
                add(piece)

    blob = extra_text or str(features.get("raw_text") or features.get("resume_context") or "")
    for piece in CAMEL_TERM.findall(blob):
        add(piece)
    for piece in VERSIONED.findall(blob):
        add(piece)
    return found


def _variant_phrases(canonical: str) -> list[str]:
    spaced = split_camel(canonical)
    phrases = {canonical, spaced, canonical.lower(), spaced.lower()}
    folded = re.sub(r"[^a-z0-9]+", "", canonical.lower())
    known = KNOWN_VARIANTS.get(folded, ())
    phrases.update(known)
    if re.search(r"v\d+$", folded):
        base = re.sub(r"v\d+$", "", folded)
        phrases.update(KNOWN_VARIANTS.get(base, ()))
    return [p for p in phrases if p and len(p) >= 3]


def build_term_patterns(terms: Iterable[str]) -> list[tuple[str, str]]:
    """Longest-first regex replacements from spoken variants -> canonical term."""
    rules: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for term in sorted(terms, key=lambda t: len(t), reverse=True):
        for variant in _variant_phrases(term):
            if variant.lower() == term.lower():
                continue
            if variant.lower() in PROTECTED_WORDS:
                continue
            key = (variant.lower(), term)
            if key in seen:
                continue
            seen.add(key)
            pattern = r"\b" + re.escape(variant).replace(r"\ ", r"[\s\-]+") + r"\b"
            rules.append((pattern, term))
    return rules


def _ngrams(words: list[str], n: int) -> list[tuple[int, str]]:
    out = []
    for i in range(len(words) - n + 1):
        out.append((i, " ".join(words[i : i + n])))
    return out


def fuzzy_restore_terms(text: str, terms: Iterable[str], cutoff: float = 0.82) -> str:
    """
    Restore a 1–3 word window to a CV term when it is clearly a mishearing.

    Common words like 'model' are only replaced when the window is closer to a
    CV term than to ordinary English — and only against the full term, never
    against 'Mole' / 'Net' fragments.
    """
    if not text:
        return text
    targets = []
    for term in terms:
        full = str(term).strip()
        if len(full) < 5:
            continue
        targets.append(full)
        spaced = split_camel(full)
        if spaced != full:
            targets.append(spaced)
    if not targets:
        return text

    words = text.split()
    used = set()
    replaced = list(words)
    for n in (3, 2, 1):
        for idx, window in _ngrams(words, n):
            if any(i in used for i in range(idx, idx + n)):
                continue
            raw = window.strip(".,!?;:'\"").lower()
            if n == 1 and raw in PROTECTED_WORDS:
                continue
            if len(raw) < 4:
                continue
            best_term = None
            best_score = cutoff
            for term in targets:
                score = SequenceMatcher(None, raw, term.lower()).ratio()
                spaced = split_camel(term).lower()
                if spaced != term.lower():
                    score = max(score, SequenceMatcher(None, raw, spaced).ratio())
                folded = re.sub(r"[^a-z0-9]+", "", term.lower())
                raw_fold = re.sub(r"[^a-z0-9]+", "", raw)
                if raw_fold and folded:
                    score = max(score, SequenceMatcher(None, raw_fold, folded).ratio())
                if score > best_score:
                    best_score = score
                    best_term = term
            if best_term:
                replaced[idx] = best_term
                for j in range(idx + 1, idx + n):
                    replaced[j] = ""
                used.update(range(idx, idx + n))
    return " ".join(w for w in replaced if w)


def _canonical_for_confusion(folded: str, term_map: dict[str, str]) -> str | None:
    if folded in term_map:
        return term_map[folded]
    for key, term in term_map.items():
        if folded in key or key in folded:
            return term
    return None


def restore_confusions(text: str, terms: Iterable[str], active_topic: str = "") -> str:
    """Restore known STT name errors (model→MoleCheck, movile→MobileNetV2)."""
    if not text:
        return text
    term_map = {fold_term(t): str(t) for t in terms if fold_term(str(t))}
    if not term_map:
        return text
    topic_fold = fold_term(active_topic)
    result = text

    protected_jobs: list[tuple[str, str, bool]] = []
    for canonical_key, confusions in NAME_CONFUSIONS.items():
        canonical = _canonical_for_confusion(canonical_key, term_map)
        if not canonical:
            continue
        on_topic = bool(topic_fold) and (canonical_key in topic_fold or fold_term(canonical) in topic_fold)
        for confused in confusions:
            if confused.lower() in PROTECTED_WORDS:
                protected_jobs.append((confused, canonical, on_topic))
            else:
                result = re.sub(rf"\b{re.escape(confused)}\b", canonical, result, flags=re.IGNORECASE)

    if not protected_jobs:
        return result

    words = result.split()
    for i, word in enumerate(words):
        bare = word.strip(".,!?;:'\"").lower()
        window = " ".join(words[max(0, i - 3) : i + 4])
        for confused, canonical, on_topic in protected_jobs:
            if bare != confused.lower():
                continue
            if _should_restore_protected(window, confused, on_topic):
                prefix = word[: len(word) - len(word.lstrip(".,!?;:'\""))]
                suffix = word[len(word.rstrip(".,!?;:'\"")) :]
                words[i] = prefix + canonical + suffix
                break
    return " ".join(words)


def _should_restore_protected(window: str, confused: str, on_topic: bool) -> bool:
    """Do not turn ordinary English 'model' into a project name."""
    if ML_MODEL_CONTEXT.search(window):
        return False
    if on_topic:
        return True
    if PROJECT_NAME_USAGE.search(window):
        return True
    if re.search(rf"\b(my|the|on|called|named)\s+{re.escape(confused)}\b", window, re.IGNORECASE):
        return True
    if re.search(rf"\b{re.escape(confused)}\s+(project|app|system)\b", window, re.IGNORECASE):
        return True
    return False


def protect_transcript(
    text: str,
    features: dict[str, Any] | None,
    extra_terms: Iterable[str] | None = None,
    active_topic: str = "",
) -> str:
    """Apply phrase rules, known name confusions, then conservative fuzzy restore."""
    if not text:
        return text
    terms = collect_cv_terms(features)
    for extra in extra_terms or ():
        raw = str(extra or "").strip()
        if not raw or len(raw) < 5:
            continue
        if raw.lower() in PROTECTED_WORDS:
            continue
        if raw not in terms:
            terms.append(raw)
    result = text
    for pattern, replacement in build_term_patterns(terms):
        try:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        except re.error:
            continue
    result = restore_confusions(result, terms, active_topic=active_topic)
    return fuzzy_restore_terms(result, terms)

"""
Resume ingestion.

Resumes are short. The whole cleaned text goes into prompt context; this module
only extracts the fields policy needs: project titles (mandatory for the
interview plan) and skills (later intersected with the job role).

Heuristic parsing is authoritative for control flow. An LLM EXTRACT call may
enrich fields; if it fails, ingestion still succeeds.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

# PDF fonts often emit compatibility ligatures that break Windows cp1252 logs
# and skill matching ("classiﬁcation" vs "classification").
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def normalize_resume_text(text: str) -> str:
    """Make PDF/Unicode resume text parseable and printable."""
    if not text:
        return ""
    cleaned = str(text).replace("\ufeff", "")
    cleaned = unicodedata.normalize("NFKC", cleaned)
    for src, dst in _LIGATURES.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = (
        cleaned.replace("\u00a0", " ")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\x00", "")
        .replace("\ufeff", "")
    )
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _repair_resume_layout(cleaned)
    return cleaned.strip()

# Tokens that look like a name/degree field leaked into skills or projects.
INVALID_FIELD_WORDS = frozenset({
    "skill", "skills", "leetcode", "demonstrated", "b.tech", "education",
    "experience", "projects", "summary", "curriculum", "vitae", "resume",
    "page", "http", "github", "email", "phone", "profile", "overview",
    "problems", "hands-on", "certifications", "certification", "achievements",
    "internship", "internships", "domains", "objective", "hobbies",
    "references", "declaration", "awards", "publications", "languages",
    "interests", "academics",
})

GENERIC_PROJECT_TITLES = frozenset({
    "technical project", "professional experience", "technical projects",
    "main project", "academic project", "personal project", "key project",
    "selected project", "work experience", "project experience",
    "project deep dive", "deep dive", "a project on your resume",
    "resume projects", "technical interview", "skills assessment",
    "warmup", "closing",
})


def is_resume_metadata_title(text: str) -> bool:
    """True when a 'project' is actually a section heading or education line."""
    low = re.sub(r"[^a-z0-9.+]+", " ", str(text or "").lower()).strip()
    if not low or len(low) < 2:
        return True
    if low in INVALID_FIELD_WORDS or low in GENERIC_PROJECT_TITLES:
        return True
    tokens = [t for t in low.split() if t]
    filler = {"and", "or", "of", "the", "my", "&"}
    content = [t for t in tokens if t not in filler]
    if content and all(t in INVALID_FIELD_WORDS for t in content):
        return True
    if any(t in {"gpa", "cgpa", "btech", "b.tech", "bachelor", "masters"} for t in tokens):
        return True
    if "deep" in tokens and "dive" in tokens:
        return True
    return False

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "skills": (
        "technical skills", "core skills", "tech stack", "technologies", "skills",
    ),
    "projects": (
        "technical projects", "academic projects", "personal projects",
        "key projects", "selected projects", "major projects", "project work",
        "projects",
    ),
    "experience": (
        "professional experience", "work experience", "work history",
        "employment history",         "industrial training", "internships",
        "employment", "experience",
    ),
    "education": (
        "educational qualifications", "academic qualifications",
        "qualification", "education", "academics",
    ),
}


def _header_aliases_longest_first() -> list[str]:
    aliases = [alias for group in SECTION_ALIASES.values() for alias in group]
    aliases.extend(
        [
            "name", "candidate name", "email", "phone", "certifications",
            "domains", "summary", "objective", "achievements",
        ]
    )
    # Longer headers first so "work experience" wins over "experience".
    return sorted(set(aliases), key=len, reverse=True)


_HEADER_ALT = "|".join(re.escape(alias) for alias in _header_aliases_longest_first())

SECTION_HEADER_RE = re.compile(
    rf"^(?:[\-\*\u2022]\s*)?(?P<header>{_HEADER_ALT})\s*:?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

NEXT_HEADER_RE = re.compile(
    rf"^(?:[\-\*\u2022]\s*)?(?:{_HEADER_ALT})\s*:?\s*$",
    re.IGNORECASE,
)

# PDFs often glue the next heading onto the previous line: "PythonPROJECTS".
_GLUED_SECTION_RE = re.compile(
    r"(?<=[a-z0-9.\)])(?P<header>EDUCATION|ACADEMICS|EXPERIENCE|INTERNSHIPS|"
    r"PROJECTS|SKILLS|CERTIFICATIONS|ACHIEVEMENTS)\b",
    re.IGNORECASE,
)
_MIDLINE_SECTION_RE = re.compile(
    r"(?<=\S)[ \t]*(?=(?:EDUCATION|ACADEMICS|EXPERIENCE|INTERNSHIPS|"
    r"PROJECTS|SKILLS|CERTIFICATIONS|ACHIEVEMENTS)\b)",
    re.IGNORECASE,
)

DEGREE_RE = re.compile(
    r"(?i)\b("
    r"b\.?\s*tech(?:nology)?"
    r"|m\.?\s*tech(?:nology)?"
    r"|bachelor(?:'s)?(?:\s+of\s+(?:technology|engineering|science|computer applications|arts))?"
    r"|master(?:'s)?(?:\s+of\s+(?:technology|engineering|science|computer applications|arts|business administration))?"
    r"|b\.e\."
    r"|m\.e\."
    r"|m\.?c\.?a\.?"
    r"|b\.?c\.?a\.?"
    r"|m\.?b\.?a\.?"
    r"|b\.?\s*sc\.?"
    r"|m\.?\s*sc\.?"
    r"|ph\.?\s*d\.?"
    r"|diploma"
    r")\b"
    r"(?:\s+in\s+[A-Za-z][A-Za-z0-9 &./+-]{1,50})?"
)

COLLEGE_HINT_RE = re.compile(
    r"(?i)\b(university|institute|college|iit|nit|iiit|polytechnic|vidyapeeth)\b",
)

ROLE_HINT_RE = re.compile(
    r"(?i)\b("
    r"intern|internship|trainee|engineer|developer|programmer|analyst|"
    r"scientist|associate|consultant|architect|researcher|designer|"
    r"sde|swe|fullstack|full-stack|backend|front-end|frontend|"
    r"data scientist|machine learning|software"
    r")\b",
)

PRESENT_RE = re.compile(r"(?i)\b(present|current|ongoing|now)\b")

PROJECT_DESC_RE = re.compile(
    r"(?i)^(a |an |the |this |built |developed |implemented |created |"
    r"designed |worked |responsible |using |technologies?:|"
    r"engineered |evaluated |performed |trained |fine[- ]tuned |"
    r"achieved |improved |reduced |increased |conducted |analyzed )"
)

_RESUME_VERBS = frozenset({
    "engineered", "implemented", "evaluated", "performed", "developed", "built",
    "created", "designed", "trained", "worked", "used", "using", "responsible",
    "achieved", "improved", "reduced", "increased", "conducted", "analyzed",
    "deployed", "integrated", "optimized", "tested",
})

_PROJECT_FRAGMENTS = frozenset({
    "dataset", "specificity", "sensitivity", "precision", "recall", "accuracy",
    "negatives", "correlation", "curves", "pipeline", "augmentation",
    "malignant", "benign", "distribution", "classification", "model",
    "matrix", "confusion", "analysis", "evaluation", "results", "metrics",
    "roc", "auc", "f1", "score", "threshold", "layer", "features", "parameters",
})

_PROJECT_SUFFIX_RE = re.compile(
    r"(?i)\s+(live\s+demo|demo|github|website|app)$"
)

_VERB_SPLIT_RE = re.compile(
    r"(?<=[A-Za-z0-9)])[,; ]+(?=(?:Engineered|Implemented|Evaluated|Performed|"
    r"Developed|Built|Created|Designed|Trained|Achieved|Improved|Analyzed|"
    r"Deployed|Integrated|Optimized)\b)",
    re.IGNORECASE,
)

_TITLE_AFTER_PERIOD_RE = re.compile(
    r"(?<=[A-Za-z]{4}[a-z0-9)])\.\s+(?=[A-Z][A-Za-z0-9+#]{2,})"
)

PROJECT_TITLE_RE = re.compile(
    r"^(?:[\-\*\u2022]\s*)?(?P<title>[A-Za-z][A-Za-z0-9+#./ ]{1,60}?)"
    r"\s*(?:[-–—|:]\s+|\s+[–—]\s+)",
)

TECH_VOCAB = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "HTML", "CSS", "SQL",
    "Rust", "Go", "Golang", "Kotlin", "Swift",
    "React.js", "React", "Node.js", "Express.js", "Next.js", "Vue.js",
    "MongoDB", "MySQL", "PostgreSQL", "SQLite", "Redis", "DuckDB", "Firebase",
    "Machine Learning", "Deep Learning", "Neural Networks", "LLMs",
    "TensorFlow", "PyTorch", "Scikit-Learn", "OpenCV", "NumPy", "Pandas", "Keras",
    "LangGraph", "LangChain", "vLLM", "RAG", "gRPC",
    "MobileNet", "MobileNetV2", "ResNet", "ResNet50", "EfficientNet", "YOLO",
    "FastAPI", "Flask", "Django", "REST API",
    "Git", "GitHub", "Docker", "Kubernetes", "Terraform", "AWS", "Linux",
    "Data Structures", "Algorithms", "DSA",
]


def _unglue_section_headers(text: str) -> str:
    return _GLUED_SECTION_RE.sub(lambda m: "\n" + m.group("header"), text or "")


def _repair_resume_layout(text: str) -> str:
    """Undo PDF column glue so SKILLS and PROJECTS stay separate sections."""
    if not text:
        return ""
    repaired = _unglue_section_headers(text)
    repaired = _MIDLINE_SECTION_RE.sub("\n", repaired)
    repaired = re.sub(r"[ \t]*[•\u2022]\s*", "\n• ", repaired)
    return repaired


def _lines(text: str) -> list[str]:
    text = _unglue_section_headers(text or "")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _section_kind(header: str) -> str | None:
    key = header.strip().lower()
    for kind, aliases in SECTION_ALIASES.items():
        if key in aliases:
            return kind
    return None


def _collect_sections(text: str) -> dict[str, list[str]]:
    """Split a resume into named sections. Unlabelled leading lines go to 'preamble'."""
    sections: dict[str, list[str]] = {kind: [] for kind in SECTION_ALIASES}
    sections["preamble"] = []
    current = "preamble"
    for line in _lines(text):
        match = SECTION_HEADER_RE.match(line)
        if match:
            kind = _section_kind(match.group("header"))
            rest = (match.group("rest") or "").strip()
            if kind:
                current = kind
                if rest:
                    sections[kind].append(rest)
                continue
            # Certifications / summary / contact headers must not leak into skills.
            current = "preamble"
            continue
        if NEXT_HEADER_RE.match(line) and not SECTION_HEADER_RE.match(line):
            current = "preamble"
            continue
        sections.setdefault(current, []).append(line)
    return sections


JOB_LINE_RE = re.compile(
    r"(?:"
    r"\b(intern|internship|employee|associate)\b"
    r"|\b(19|20)\d{2}\s*[–—-]\s*(present|\d{4})\b"
    r")",
    re.IGNORECASE,
)


def looks_like_job_line(text: str) -> bool:
    """True for employment bullets, not software project titles."""
    low = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not low:
        return False
    if JOB_LINE_RE.search(low) and any(sep in low for sep in (",", " at ", " | ", " - ", " – ", " — ")):
        return True
    if re.search(r"\b(intern|internship)\b", low) and re.search(r"\b(19|20)\d{2}\b", low):
        return True
    return False


def _looks_like_skill_csv(text: str) -> bool:
    if ROLE_HINT_RE.search(text or "") or re.search(r"\b(19|20)\d{2}\b", text or ""):
        return False
    parts = [p.strip() for p in re.split(r"[,/]", text or "") if p.strip()]
    if len(parts) < 3:
        return False
    if any(PROJECT_DESC_RE.match(p) or _looks_like_project_fragment(p) for p in parts):
        return False
    return all(len(p) < 32 and len(p.split()) <= 4 for p in parts)


def _is_skillish_title(text: str) -> bool:
    """True when a token belongs in SKILLS, not PROJECTS."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" -–—|:•*")
    if not cleaned:
        return True
    low = cleaned.lower()
    if low in _tech_vocab_lower():
        return True
    if _looks_like_skill_csv(cleaned):
        return True
    if re.match(r"(?i)^(programming|libraries|frameworks|tools|core|languages|ai/?ml)\b", cleaned):
        return True
    if any(
        phrase in low
        for phrase in (
            "data structure",
            "object-oriented",
            "object oriented",
            "machine learning",
            "deep learning",
            "neural network",
        )
    ):
        return True
    words = [w for w in re.findall(r"[a-z0-9+#]+", low) if w not in {"and", "or", "of", "the"}]
    if words and all(w in _tech_vocab_lower() or w in {"dsa", "oop", "oops"} for w in words):
        return True
    return False


def _strip_date_tail(text: str) -> str:
    return re.sub(
        r"(?i)[,(]?\s*(?:"
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?|(?:19|20)\d{2}"
        r")\b.*$",
        "",
        text or "",
    ).strip(" ,;-–—|")


def _parse_job_line(line: str) -> tuple[str | None, str | None]:
    """Best-effort (role, company) from an experience bullet."""
    text = re.sub(r"\s+", " ", str(line or "")).strip(" -–—|:•*")
    if not text or len(text) > 180:
        return None, None
    parts = [p.strip() for p in re.split(r"\s*[|•–—]\s*", text) if p.strip()]
    if len(parts) >= 2:
        left, right = parts[0], parts[1]
        left_core = _strip_date_tail(left) or left
        right_core = _strip_date_tail(right) or right
        if "," in right_core:
            right_core = right_core.split(",")[0].strip()
        if ROLE_HINT_RE.search(left_core) and not COLLEGE_HINT_RE.search(left_core):
            return left_core[:80], right_core[:80]
        if ROLE_HINT_RE.search(right_core) and not COLLEGE_HINT_RE.search(right_core):
            return right_core[:80], left_core[:80]
        return left_core[:80], right_core[:80]

    at_match = re.search(r"^(?P<role>.+?)\s+(?:at|@)\s+(?P<company>.+)$", text, re.I)
    if at_match:
        role = _strip_date_tail(at_match.group("role"))
        company = _strip_date_tail(at_match.group("company"))
        if role and company:
            return role[:80], company[:80]

    comma_parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(comma_parts) >= 2 and ROLE_HINT_RE.search(comma_parts[0]):
        role = comma_parts[0]
        company = _strip_date_tail(comma_parts[1])
        if company:
            return role[:80], company[:80]
    return None, None


def extract_education(text: str) -> tuple[str | None, str | None]:
    sections = _collect_sections(text)
    pool = list(sections.get("education") or ())
    if not pool:
        pool = _lines(text)

    degree = college = None
    for line in pool:
        clean_line = re.sub(r"(?i)\s*(?:cgpa|gpa|percentage|grade)\s*:?\s*[\d.]+(?:\s*/\s*[\d.]+)?", "", line).strip()
        if degree is None:
            match = DEGREE_RE.search(clean_line)
            if match:
                snippet = match.group(0).strip(" ,;")
                extra = clean_line[match.end():].strip(" ,;-–—|")
                extra = _strip_date_tail(extra)
                if extra.lower().startswith("in ") and len(extra) < 60:
                    snippet = f"{snippet} {extra}".strip()
                snippet = _strip_date_tail(snippet)
                degree = re.sub(r"\s+", " ", snippet)[:120]
        if college is None and COLLEGE_HINT_RE.search(clean_line) and not DEGREE_RE.search(clean_line):
            college_candidate = _strip_date_tail(clean_line) or clean_line
            college_candidate = re.sub(r"(?i)\b(?:cgpa|gpa)[\s:]*[\d.]+\b", "", college_candidate)
            college = re.sub(r"\s+", " ", college_candidate).strip(" ,;")[:120]
        if degree and college:
            break
    return degree, college


def extract_current_job(text: str) -> tuple[str | None, str | None, str | None]:
    """Current / most recent role and company from EXPERIENCE, not PROJECTS."""
    lines = list(_collect_sections(text).get("experience") or ())
    if not lines:
        return None, None, None
    ranked = sorted(
        lines,
        key=lambda line: (
            0 if PRESENT_RE.search(line) else 1,
            0 if ROLE_HINT_RE.search(line) else 1,
        ),
    )
    for line in ranked:
        if PROJECT_DESC_RE.match(line) or _looks_like_skill_csv(line):
            continue
        role, company = _parse_job_line(line)
        if role or company:
            return role, company, line[:160]
    return None, None, None


def _tech_vocab_lower() -> set[str]:
    return {t.lower() for t in TECH_VOCAB}


def _looks_like_project_fragment(text: str) -> bool:
    tokens = re.findall(r"[a-z]+", str(text or "").lower())
    if not tokens:
        return True
    if tokens[0] in _RESUME_VERBS:
        return True
    if len(tokens) == 1 and tokens[0] in _PROJECT_FRAGMENTS:
        return True
    if tokens and all(t in _PROJECT_FRAGMENTS for t in tokens):
        return True
    return False


def _is_project_name(text: str) -> bool:
    """True only for a short product/project title, never a bullet description."""
    cleaned = _PROJECT_SUFFIX_RE.sub("", re.sub(r"\s+", " ", str(text or "")).strip(" -–—|:•*.,"))
    if not cleaned or PROJECT_DESC_RE.match(cleaned) or _looks_like_project_fragment(cleaned):
        return False
    if _is_skillish_title(cleaned):
        return False
    if cleaned.lower() in _tech_vocab_lower():
        return False
    words = cleaned.split()
    if not 1 <= len(words) <= 6:
        return False
    filler = {"of", "and", "for", "the", "in", "or", "with", "using", "by", "from"}
    if words[0].lower() in filler:
        return False
    named = 0
    for word in words:
        token = re.sub(r"[^A-Za-z0-9+#]", "", word)
        if not token:
            return False
        if token.lower() in filler:
            continue
        if token.lower() in _RESUME_VERBS or token.lower() in _PROJECT_FRAGMENTS:
            return False
        if not re.match(r"^[A-Z][A-Za-z0-9+#]*$", token):
            return False
        named += 1
    return named >= 1


def project_title(raw: str) -> str | None:
    """
    Turn a project bullet into a speakable title.

    'Ferrite Mesh - A service mesh sidecar...' -> 'Ferrite Mesh'
    Description-only bullets are dropped, not truncated into fake titles.
    """
    text = re.sub(r"\s+", " ", str(raw or "")).strip(" -–—|:•*")
    if not text:
        return None
    lowered = text.lower()
    if any(word in lowered for word in INVALID_FIELD_WORDS if word in ("resume", "curriculum", "vitae")):
        return None
    if PROJECT_DESC_RE.match(text):
        return None

    match = PROJECT_TITLE_RE.match(text)
    if match:
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        title = _PROJECT_SUFFIX_RE.sub("", title).strip()
        if _is_project_name(title):
            return title

    candidate = _PROJECT_SUFFIX_RE.sub("", text).strip()
    if _is_project_name(candidate):
        return candidate
    return None


def _project_chunks(line: str) -> list[str]:
    """Split a PROJECTS line that glued titles and description bullets together."""
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    if not text:
        return []
    # If the line starts with a bullet point or action verb, it is a project description bullet, NOT glued project titles!
    if re.match(r"^[\s\-*•\u2022]\s*", line) or PROJECT_DESC_RE.match(text) or _looks_like_project_fragment(text):
        # A line starting with a bullet could still be: • Ferrite Mesh - A service mesh...
        # which has a title before the delimiter. But if it does NOT have a delimiter, it's purely a description!
        if not re.search(r"\s[-–—|:]\s+", text):
            return []
    pieces = [part.strip() for part in _TITLE_AFTER_PERIOD_RE.split(text) if part.strip()]
    verb_parts: list[str] = []
    for piece in pieces or [text]:
        parts = [part.strip(" ,;") for part in _VERB_SPLIT_RE.split(piece) if part.strip(" ,;")]
        verb_parts.extend(parts or [piece])
    if len(verb_parts) > 1:
        expanded: list[str] = []
        for part in verb_parts:
            if "," in part and len(part) > 40:
                expanded.extend(p.strip() for p in part.split(",") if p.strip())
            else:
                expanded.append(part)
        return expanded
    if re.search(r"\s[-–—|:]\s+", text):
        return [text]
    if "," in text and len(text) > 40:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        kept: list[str] = []
        for part in parts:
            if PROJECT_DESC_RE.match(part) or _looks_like_project_fragment(part) or len(part.split()) > 8:
                continue
            kept.append(part)
        return kept
    return [text]


def extract_projects(text: str, stated: Iterable[Any] | None = None) -> list[str]:
    """Project titles from structured fields plus the PROJECTS / EXPERIENCE sections."""
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        title = project_title(str(raw)) if raw else None
        if not title or is_resume_metadata_title(title) or looks_like_job_line(title) or looks_like_job_line(str(raw)):
            return
        if _is_skillish_title(title) or PROJECT_DESC_RE.match(title) or not _is_project_name(title):
            return
        key = title.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(title)

    if isinstance(stated, str):
        stated = [part.strip() for part in stated.split(",") if part.strip()]
    for item in stated or ():
        add(item)

    sections = _collect_sections(text)
    for line in sections.get("projects") or ():
        if _looks_like_skill_csv(line) or looks_like_job_line(line):
            continue
        for chunk in _project_chunks(line):
            if PROJECT_DESC_RE.match(chunk) or _looks_like_skill_csv(chunk):
                continue
            add(chunk)
    if not found:
        for line in sections.get("experience") or ():
            if PROJECT_DESC_RE.match(line) or _looks_like_skill_csv(line):
                continue
            add(line)

    return found[:8]


def extract_skills(text: str, stated: Iterable[Any] | None = None) -> list[str]:
    """Skills from structured fields, the SKILLS section, and a known-tech scan."""
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        skill = re.sub(r"^\s*[\-\*\u2022•]+\s*", "", str(raw or ""))
        if ":" in skill:
            before, after = skill.split(":", 1)
            if any(h in before.lower() for h in ("programming", "languages", "libraries", "frameworks", "tools", "core", "skills", "technologies", "databases", "ai", "ml")):
                skill = after
        skill = re.sub(r"\s+", " ", skill).strip(" ,;")
        if not skill or len(skill) > 45:
            return
        lowered = skill.lower()
        if lowered in INVALID_FIELD_WORDS:
            return
        if any(word in lowered for word in INVALID_FIELD_WORDS if word not in ("dsa",)):
            return
        if lowered in seen:
            return
        seen.add(lowered)
        found.append(skill)

    if isinstance(stated, str):
        stated = [part.strip() for part in stated.split(",") if part.strip()]
    for item in stated or ():
        add(item)

    for line in _collect_sections(text).get("skills") or ():
        clean_line = re.sub(r"^\s*[\-\*\u2022•]+\s*", "", line)
        if ":" in clean_line:
            before, after = clean_line.split(":", 1)
            if any(h in before.lower() for h in ("programming", "languages", "libraries", "frameworks", "tools", "core", "skills", "technologies", "databases", "ai", "ml")):
                clean_line = after
        if "," in clean_line or "|" in clean_line or "/" in clean_line:
            parts = re.split(r"[,|/]", clean_line)
        else:
            parts = [clean_line]
        for part in parts:
            add(part)

    for tech in TECH_VOCAB:
        if re.search(rf"\b{re.escape(tech)}\b", text or "", flags=re.IGNORECASE):
            add(tech)

    return found[:24]


def parse_resume(text: str) -> dict[str, Any]:
    """Heuristic resume parse. Always returns projects and skills lists (possibly empty)."""
    text = normalize_resume_text(text)
    lines = _lines(text)
    name = degree = college = role = experience = company = None
    certifications: list[str] = []
    domains: list[str] = []

    for line in lines:
        lower = line.lower()
        if lower.startswith(("candidate name:", "name:")):
            name = line.split(":", 1)[1].strip()
        elif lower.startswith(("college:", "university:", "institute:")):
            college = line.split(":", 1)[1].strip()
        elif lower.startswith("degree:"):
            degree = line.split(":", 1)[1].strip()
        elif lower.startswith(("current role:", "role:")):
            role = line.split(":", 1)[1].strip()
        elif lower.startswith("experience:") and not lower.startswith("work experience"):
            experience = line.split(":", 1)[1].strip()
        elif lower.startswith(("company:", "employer:")):
            company = line.split(":", 1)[1].strip()
        elif lower.startswith("certifications:"):
            certifications = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        elif lower.startswith("domains:"):
            domains = [d.strip() for d in line.split(":", 1)[1].split(",") if d.strip()]

    if not name and lines:
        first = lines[0]
        if "@" not in first and "http" not in first.lower() and len(first.split()) <= 5:
            name = first

    inferred_degree, inferred_college = extract_education(text)
    inferred_role, inferred_company, inferred_experience = extract_current_job(text)
    degree = degree or inferred_degree
    college = college or inferred_college
    role = role or inferred_role
    company = company or inferred_company
    experience = experience or inferred_experience

    skills = extract_skills(text)
    projects = [
        p for p in extract_projects(text)
        if p.lower() not in {s.lower() for s in skills} and not _is_skillish_title(p)
    ]

    return {
        "name": name,
        "college": college,
        "degree": degree,
        "role": role,
        "experience": experience,
        "company": company,
        "skills": skills,
        "projects": projects,
        "certifications": certifications,
        "domains": domains,
        "raw_text": (text or "")[:6000],
    }


def ingest_resume(text: str, llm_fields: dict[str, Any] | None = None) -> dict[str, Any]:
    """Single ingestion entry: normalize, heuristic parse, optional LLM field merge."""
    cleaned = normalize_resume_text(text)
    return sanitize_features(llm_fields or {}, cleaned)


def sanitize_features(features: dict[str, Any], raw_text: str) -> dict[str, Any]:
    """Keep LLM-extracted fields, then re-derive projects and skills from the source text."""
    if not isinstance(features, dict):
        features = {}
    raw_text = normalize_resume_text(raw_text)
    parsed = parse_resume(raw_text)
    merged = dict(parsed)
    for key in ("name", "college", "degree", "role", "experience", "company", "certifications", "domains"):
        value = features.get(key)
        if value not in (None, "", [], "null"):
            merged[key] = value
    merged["skills"] = extract_skills(raw_text, features.get("skills") or parsed.get("skills"))
    skill_keys = {str(s).lower() for s in merged["skills"]}
    merged["projects"] = [
        p for p in extract_projects(raw_text, features.get("projects") or parsed.get("projects"))
        if p.lower() not in skill_keys and not _is_skillish_title(p)
    ]
    merged["raw_text"] = (raw_text or merged.get("raw_text") or "")[:6000]
    return merged

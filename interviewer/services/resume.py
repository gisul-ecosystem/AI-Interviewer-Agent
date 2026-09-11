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
        "relevant projects", "course projects", "academic project",
        "personal project", "major project", "projects",
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
# Only split when the section keyword is in ALL-CAPS (a glued PDF header),
# not when it appears in lowercase prose (e.g. "practical skills in...").
_MIDLINE_SECTION_RE = re.compile(
    r"(?<=[A-Z0-9.!?|/])[ \t/|]*(?=(?:EDUCATION|ACADEMICS|EXPERIENCE|INTERNSHIPS|"
    r"PROJECTS|SKILLS|CERTIFICATIONS|ACHIEVEMENTS)\b)",
    # No re.IGNORECASE — we intentionally only match uppercase keywords here.
)
_MIDLINE_PHRASE_RE = re.compile(
    r"(?<=\S)[ \t/|]*(?=(?:Technical\s+Skills|Core\s+Skills|Academic\s+Projects|"
    r"Personal\s+Projects|Key\s+Projects|Selected\s+Projects|Relevant\s+Projects|"
    r"Work\s+Experience|Professional\s+Experience|Project\s+Work)\b)",
    re.IGNORECASE,
)
_TWO_HEADER_RE = re.compile(
    r"\b(?P<a>EDUCATION|ACADEMICS|EXPERIENCE|INTERNSHIPS|PROJECTS|SKILLS|"
    r"CERTIFICATIONS|ACHIEVEMENTS)\b[ \t]*[/|&]+[ \t]*"
    r"\b(?P<b>EDUCATION|ACADEMICS|EXPERIENCE|INTERNSHIPS|PROJECTS|SKILLS|"
    r"CERTIFICATIONS|ACHIEVEMENTS)\b",
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

_TITLE_SMALLWORDS = frozenset({
    "a", "an", "the", "of", "and", "for", "in", "or", "with", "on", "to", "by",
    "from", "using", "vs", "vs.",
})
_TITLE_NOUNS = frozenset({
    "system", "app", "application", "website", "portal", "platform", "dashboard",
    "detector", "classifier", "predictor", "tracker", "manager", "management",
    "chatbot", "chat", "assistant", "tool", "service", "network", "identification",
    "monitoring", "detection", "recognition", "automation", "store", "shop",
    "commerce", "ecommerce", "voting", "attendance", "inventory", "booking",
    "scheduler", "recommendation", "engine", "hackathon", "analyzer", "finder",
    "generator", "monitor", "planner", "reminder", "blog", "forum", "game",
    "clone", "board", "tracker", "extension", "plugin", "api", "bot",
})
_BARE_DEMO_TITLES = frozenset({
    "live demo", "demo", "github", "source code", "website", "link", "code",
})
_PROJECT_NUM_RE = re.compile(
    r"^(?:[\-\*\u2022]\s*)?(?:(?:project|proj)\s*)?(?:\(?\d{1,2}\)?[.):\-]|\(?[A-Z]\)[.):\-]?)\s+",
    re.IGNORECASE,
)
_PROJECT_LABEL_RE = re.compile(
    r"^(?:title|name|project(?:\s*(?:name|title))?)\s*:\s*",
    re.IGNORECASE,
)

_PROJECT_FRAGMENTS = frozenset({
    "dataset", "specificity", "sensitivity", "precision", "recall", "accuracy",
    "negatives", "correlation", "curves", "pipeline", "augmentation",
    "malignant", "benign", "distribution", "classification", "model",
    "matrix", "confusion", "analysis", "evaluation", "results", "metrics",
    "roc", "auc", "f1", "f2", "f1-score", "f2-score", "score", "threshold",
    "layer", "features", "parameters", "bleu", "rouge", "perplexity", "map",
    "top-1", "top-5", "mse", "mae", "rmse", "loss",
})

# Titles starting with a descriptor adjective are descriptions, not project names.
# e.g. "AI-generated Text", "Human-generated Text Classification", "Fine-tuned BERT"
_DESCRIPTOR_ADJ_RE = re.compile(
    r"^(?:AI|Human|Machine|Auto|Pre|Fine|Custom|Auto)[- ]"
    r"(?:generated|tuned|trained|built|made|designed|labelled|labeled)\b",
    re.IGNORECASE,
)

# Metric identifiers like "F1-score", "F1", "Top-1" that start with F/T + digit
_METRIC_ID_RE = re.compile(r"^[A-Za-z]{1,3}\d[-_]", re.IGNORECASE)

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
    repaired = _TWO_HEADER_RE.sub(lambda m: m.group("a") + "\n" + m.group("b"), text)
    repaired = _unglue_section_headers(repaired)
    repaired = _MIDLINE_PHRASE_RE.sub("\n", repaired)
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



# Words that indicate the matched "section header" is actually a prose continuation,
# e.g. "experience in Machine Learning" or "projects including Deepfake...".
_PROSE_CONTINUATION_RE = re.compile(
    r"^(?:in\b|including\b|of\b|with\b|for\b|on\b|at\b|by\b|to\b|from\b|and\b|"
    r"as\b|are\b|is\b|was\b|were\b|that\b|which\b|such\b|like\b)",
    re.IGNORECASE,
)


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
                # Guard: if rest looks like prose continuation ("experience in ML",
                # "projects including Foo"), this is not a real section header —
                # it's a prose line where the section keyword appears at the start.
                if rest and _PROSE_CONTINUATION_RE.match(rest):
                    sections.setdefault(current, []).append(line)
                    continue
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
    # ML/AI class names and sklearn pipeline components are skills, not projects.
    if re.match(
        r"(?i)^(ColumnTransformer|StandardScaler|MinMaxScaler|LabelEncoder|OneHotEncoder|"
        r"Pipeline|GridSearchCV|RandomizedSearchCV|LogisticRegression|DecisionTree|"
        r"RandomForest|GradientBoosting|XGBoost|LightGBM|CatBoost|SVM|KNN|KMeans)$",
        cleaned,
    ):
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


def _strip_project_prefix(text: str) -> str:
    """Drop '1.', 'Project 2:', 'Title:' so the actual name can be parsed."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" -–—|:•*")
    cleaned = _PROJECT_NUM_RE.sub("", cleaned).strip(" -–—|:•*")
    cleaned = _PROJECT_LABEL_RE.sub("", cleaned).strip(" -–—|:•*")
    return cleaned


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


def _is_name_like_token(token: str, *, first: bool, loose: bool) -> bool:
    raw = str(token or "")
    stripped = re.sub(r"[^A-Za-z0-9+#]", "", raw)
    if not stripped:
        return False
    low = stripped.lower()
    if low in _TITLE_SMALLWORDS:
        return not first
    if low in _RESUME_VERBS:
        return False
    if first and low in _PROJECT_FRAGMENTS:
        return False
    if stripped[0].isdigit():
        return True
    if stripped[0].isupper():
        return True
    if re.search(r"[a-z][A-Z]", stripped):
        return True
    if low in {"iot", "ai", "ml", "nlp", "cv", "ios", "ar", "vr"}:
        return True
    if "commerc" in low or low in {"ecommerce", "nextjs", "nodejs"}:
        return True
    if not first and (low in _TITLE_NOUNS or (loose and stripped.isalpha() and 2 <= len(stripped) <= 18)):
        return True
    if first and loose and stripped.isalpha() and 3 <= len(stripped) <= 24 and low not in _PROJECT_FRAGMENTS:
        return True
    return False


def _is_project_name(text: str, *, loose: bool = False) -> bool:
    """True for a short product/project title, never a bullet description."""
    cleaned = _PROJECT_SUFFIX_RE.sub("", _strip_project_prefix(text)).strip(" -–—|:•*.,")
    if not cleaned or PROJECT_DESC_RE.match(cleaned) or _looks_like_project_fragment(cleaned):
        return False
    if cleaned.lower() in _BARE_DEMO_TITLES or cleaned.lower() in GENERIC_PROJECT_TITLES:
        return False
    if _is_skillish_title(cleaned):
        return False
    if cleaned.lower() in _tech_vocab_lower():
        return False
    # Fix 5: Reject adjective-phrase titles like "AI-generated Text Classification"
    if _DESCRIPTOR_ADJ_RE.match(cleaned):
        return False
    # Fix 6: Reject metric identifiers like "F1-score", "F2-score"
    if _METRIC_ID_RE.match(cleaned):
        return False
    # Reject pure all-caps abbreviations that are skills, not project names
    # e.g. "ML", "AI", "NLP", "DSA" are skills; real project names have more context.
    if re.match(r'^[A-Z]{1,4}$', cleaned):
        return False
    # Reject names that are hyphenated compound terms ending in a word class
    # like "Question-Answering" (a task type, not a product name) unless followed by
    # a product noun like "system", "app" etc.
    if cleaned.endswith(".") or cleaned.endswith(","):
        cleaned = cleaned.rstrip(".,")
        if not cleaned:
            return False
    words = cleaned.split()
    if not 1 <= len(words) <= 8:
        return False
    named = 0
    for i, word in enumerate(words):
        if not _is_name_like_token(word, first=(i == 0), loose=loose):
            return False
        low = re.sub(r"[^a-z0-9+#]+", "", word.lower())
        if low not in _TITLE_SMALLWORDS:
            named += 1
    if named < 1:
        return False
    # Long prose-like lines are descriptions, even when every word is capitalised
    # by a PDF template. Explicit numbered/labelled titles use loose=True.
    if not loose and named >= 5 and not any(
        word.lower() in _TITLE_NOUNS for word in words
    ):
        return False
    return True


def project_title(raw: str, *, loose: bool = True) -> str | None:
    """
    Turn a project bullet into a speakable title.

    'Ferrite Mesh - A service mesh sidecar...' -> 'Ferrite Mesh'
    Description-only bullets are dropped, not truncated into fake titles.
    """
    text = _strip_project_prefix(raw)
    if not text:
        return None
    lowered = text.lower()
    if any(word in lowered for word in INVALID_FIELD_WORDS if word in ("resume", "curriculum", "vitae")):
        return None
    if PROJECT_DESC_RE.match(text):
        return None
    if lowered in _BARE_DEMO_TITLES:
        return None

    match = PROJECT_TITLE_RE.match(text)
    if match:
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        title = _PROJECT_SUFFIX_RE.sub("", title).strip().rstrip(".,:;")
        title = _strip_project_prefix(title)
        if _is_project_name(title, loose=loose):
            return title

    candidate = _PROJECT_SUFFIX_RE.sub("", text).strip().rstrip(".,:;")
    candidate = _strip_project_prefix(candidate)
    if _is_project_name(candidate, loose=loose):
        return candidate
    return None


def _project_chunks(line: str) -> list[str]:
    """Split a PROJECTS line that glued titles and description bullets together."""
    raw = str(line or "").strip()
    if not raw:
        return []
    spaced = [part.strip() for part in re.split(r"[ \t]{2,}", raw) if part.strip()]
    if len(spaced) >= 2:
        return spaced
    text = re.sub(r"\s+", " ", raw).strip()
    numbered = [
        part.strip(" -–—|:•*")
        for part in re.split(r"(?:(?<=\S)\s+)?(?=\d{1,2}[.)]\s+)", text)
        if part.strip(" -–—|:•*")
    ]
    if len(numbered) >= 2:
        return numbered
    spaced = [part.strip() for part in re.split(r"\s{2,}", text) if part.strip()]
    if len(spaced) >= 2:
        return spaced
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
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        kept: list[str] = []
        for part in parts:
            if PROJECT_DESC_RE.match(part) or _looks_like_project_fragment(part) or len(part.split()) > 8:
                continue
            kept.append(part)
        if len(kept) >= 2 or (kept and len(text) > 40):
            return kept
    return [text]


def extract_projects(text: str, stated: Iterable[Any] | None = None) -> list[str]:
    """Project titles from LLM candidates and the PROJECTS section.

    LLM candidates are accepted only when their words occur in the source CV.
    Descriptions are never promoted merely because they are inside PROJECTS.
    """
    found: list[str] = []
    seen: set[str] = set()
    source_fold = re.sub(r"[^a-z0-9+#]+", " ", text.lower())
    source_tokens = set(source_fold.split())

    def has_source_evidence(title: str) -> bool:
        title_fold = re.sub(r"[^a-z0-9+#]+", " ", title.lower()).strip()
        if title_fold and title_fold in source_fold:
            return True
        tokens = [
            token for token in title_fold.split()
            if token not in _TITLE_SMALLWORDS and len(token) >= 2
        ]
        return bool(tokens) and all(token in source_tokens for token in tokens)

    def add(raw: Any, *, loose: bool = False, require_evidence: bool = False) -> None:
        title = project_title(str(raw), loose=loose) if raw else None
        if not title or is_resume_metadata_title(title) or looks_like_job_line(title) or looks_like_job_line(str(raw)):
            return
        if require_evidence and not has_source_evidence(title):
            return
        if _is_skillish_title(title) or PROJECT_DESC_RE.match(title) or not _is_project_name(title, loose=loose):
            return
        key = title.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(title)

    if isinstance(stated, str):
        stated = [part.strip() for part in stated.split(",") if part.strip()]
    for item in stated or ():
        # LLM-stated projects get trusted if:
        # 1. The words appear in the source CV (evidence check)
        # 2. It's not a skill/tool name
        # 3. It doesn't look like a section heading or job line
        # We skip the strict _is_project_name() heuristic — the LLM already parsed the structure.
        raw = str(item or "").strip()
        if not raw:
            continue
        raw_clean = _strip_project_prefix(raw).strip(" -\u2013\u2014|:\u2022*.,")
        raw_clean = _PROJECT_SUFFIX_RE.sub("", raw_clean).strip().rstrip(".,:;")
        if not raw_clean or len(raw_clean) < 2:
            continue
        if is_resume_metadata_title(raw_clean) or looks_like_job_line(raw_clean):
            continue
        if _is_skillish_title(raw_clean) or raw_clean.lower() in _tech_vocab_lower():
            continue
        if PROJECT_DESC_RE.match(raw_clean):
            continue
        if _DESCRIPTOR_ADJ_RE.match(raw_clean) or _METRIC_ID_RE.match(raw_clean):
            continue
        if re.match(r'^[A-Z]{1,4}$', raw_clean):  # pure abbreviation (ML, AI)
            continue
        # Require that the title's key words exist in the resume text.
        if not has_source_evidence(raw_clean):
            continue
        key = raw_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(raw_clean)

    sections = _collect_sections(text)
    for line in sections.get("projects") or ():
        if _looks_like_skill_csv(line) or looks_like_job_line(line):
            continue
        for chunk in _project_chunks(line):
            if PROJECT_DESC_RE.match(chunk) or _looks_like_skill_csv(chunk):
                continue
            explicit = bool(_PROJECT_NUM_RE.match(chunk) or _PROJECT_LABEL_RE.match(chunk))
            add(chunk, loose=explicit)
    # Fix 4: Fallback only within PROJECTS section (numbered/labelled lines that
    # _project_chunks may have missed), NOT the whole resume — prevents summary
    # and experience lines from being promoted into project titles.
    if len(found) < 2:
        for line in sections.get("projects") or []:
            if looks_like_job_line(line) or _looks_like_skill_csv(line):
                continue
            if _PROJECT_NUM_RE.match(line) or _PROJECT_LABEL_RE.match(line):
                add(line, loose=True)
    if not found:
        for line in sections.get("experience") or ():
            if PROJECT_DESC_RE.match(line) or _looks_like_skill_csv(line):
                continue
            add(line, loose=False)

    return found[:8]


def extract_project_contexts(text: str, projects: Iterable[str]) -> dict[str, str]:
    """Return source-CV context around every accepted project title."""
    normalized = normalize_resume_text(text)
    flat = " ".join(_lines(normalized))
    contexts: dict[str, str] = {}
    for project in projects:
        title = str(project or "").strip()
        if not title:
            continue
        match = re.search(re.escape(title), flat, flags=re.IGNORECASE)
        if not match:
            continue
        start = max(0, match.start() - 120)
        end = min(len(flat), match.end() + 700)
        snippet = re.sub(r"\s+", " ", flat[start:end]).strip()
        contexts[title] = snippet
    return contexts


# Fix 2 & Fix 3 helpers at module level so they are compiled once.
# Rejects tokens that are clearly sentence fragments, not skill names.
_SKILL_SENTENCE_START_RE = re.compile(
    r"^(?:in|and|the|with|of|for|on|at|by|to|from|including|focusing|focusing\s+on|as\s+well)\b",
    re.IGNORECASE,
)
# Matches "Word:Value" (no space) category-prefix patterns like "Frontend:React.js"
_SKILL_CATEGORY_PREFIX_RE = re.compile(
    r"^[A-Za-z][A-Za-z &/]{0,30}:[^ ]",
)


def extract_skills(text: str, stated: Iterable[Any] | None = None) -> list[str]:
    """Skills from structured fields, the SKILLS section, and a known-tech scan."""
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        skill = re.sub(r"^\s*[\-\*\u2022•]+\s*", "", str(raw or ""))
        # Strip known category-header prefixes (with or without space after colon).
        # Handles both "Languages: Python" and "Frontend:React.js" (Fix 3).
        if ":" in skill:
            before, after = skill.split(":", 1)
            before_low = before.lower()
            is_category = any(
                h in before_low
                for h in (
                    "programming", "languages", "libraries", "frameworks",
                    "tools", "core", "skills", "technologies", "databases",
                    "ai", "ml", "frontend", "backend", "deployment", "cloud",
                    "devops", "web", "mobile", "data", "concepts",
                )
            )
            if is_category:
                skill = after
        skill = re.sub(r"\s+", " ", skill).strip(" ,;")
        if not skill or len(skill) > 45:
            return
        # Fix 2: Reject sentence-fragment tokens that are not real skill names.
        if _SKILL_SENTENCE_START_RE.match(skill):
            return
        # Reject tokens ending with sentence punctuation (continuation fragments).
        if skill.endswith(".") or skill.endswith(",") or skill.endswith(":"):
            return
        # Fix 3: Reject remaining "Category:Value" tokens that slipped through
        # (e.g. the token itself is "Frontend:React.js" after being split by pipe).
        if _SKILL_CATEGORY_PREFIX_RE.match(skill):
            colon_idx = skill.index(":")
            skill = skill[colon_idx + 1:].strip()
            if not skill:
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
        # Fix 2: Skip lines that are clearly prose sentences, not skill lists.
        # A skills line that is a sentence fragment (starts with verb/preposition)
        # and has no comma/pipe/colon separators is almost certainly an experience bullet.
        is_prose = (
            _SKILL_SENTENCE_START_RE.match(clean_line)
            or (PROJECT_DESC_RE.match(clean_line) and "," not in clean_line)
        )
        if is_prose:
            continue
        if ":" in clean_line:
            before, after = clean_line.split(":", 1)
            before_low = before.lower()
            is_category = any(
                h in before_low
                for h in (
                    "programming", "languages", "libraries", "frameworks",
                    "tools", "core", "skills", "technologies", "databases",
                    "ai", "ml", "frontend", "backend", "deployment", "cloud",
                    "devops", "web", "mobile", "data", "concepts",
                )
            )
            if is_category:
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
    """
    Merge LLM-extracted fields with heuristic parse results.

    Priority: LLM wins for fields it returned, heuristic fills gaps.
    For projects and skills, LLM candidates are used as primary source
    (with source-evidence check), then heuristic results fill any gaps.
    """
    if not isinstance(features, dict):
        features = {}
    raw_text = normalize_resume_text(raw_text)
    parsed = parse_resume(raw_text)
    merged = dict(parsed)

    # Scalar fields: LLM wins if non-null.
    for key in ("name", "college", "degree", "role", "experience", "company", "certifications", "domains"):
        value = features.get(key)
        if value not in (None, "", [], "null"):
            merged[key] = value

    # --- Skills ---
    # LLM skills are trusted first (they understand context), then heuristic fills gaps.
    llm_skills = features.get("skills") or []
    heuristic_skills = parsed.get("skills") or []
    merged["skills"] = extract_skills(raw_text, llm_skills or heuristic_skills)

    skill_keys = {str(s).lower() for s in merged["skills"]}

    # --- Projects ---
    # Use LLM candidates as primary; heuristic candidates as secondary (fallback).
    llm_projects = features.get("projects") or []
    heuristic_projects = parsed.get("projects") or []

    if llm_projects:
        # LLM provided projects: trust them (with evidence + minimal checks);
        # then add any heuristic projects that LLM missed.
        primary = extract_projects(raw_text, llm_projects)
        # Fill gaps: heuristic projects not already found by LLM
        primary_keys = {p.lower() for p in primary}
        secondary = [
            p for p in extract_projects(raw_text, heuristic_projects)
            if p.lower() not in primary_keys
        ]
        all_projects = primary + secondary
    else:
        # No LLM projects: fall back entirely to heuristic.
        all_projects = extract_projects(raw_text, heuristic_projects)

    merged["projects"] = [
        p for p in all_projects
        if p.lower() not in skill_keys and not _is_skillish_title(p)
    ]
    merged["project_contexts"] = extract_project_contexts(raw_text, merged["projects"])
    merged["raw_text"] = (raw_text or merged.get("raw_text") or "")[:6000]
    return merged

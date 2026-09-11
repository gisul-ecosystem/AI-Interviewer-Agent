"""
Semantic Candidate Intent Detection Engine.
Classifies candidate speech into UNKNOWN_OR_SKIP, SELF_INTRO, or TECHNICAL_ANSWER.
"""

import re

UNKNOWN_PATTERNS = [
    r"\b(?:i\s*)?(?:don'?t|do\s*not|dont)\s*know\b",
    r"\b(?:i\s*)?have\s*no\s*idea\b",
    r"\bno\s*(?:idea|clue)\b",
    r"\b(?:i\s*)?(?:am\s*not|'m\s*not|not)\s*(?:sure|familiar|aware)\b",
    r"\bhavens?t\s*(?:read|studied|looked|worked|heard)\b",
    r"\bhaven'?t\s*(?:read|studied|looked|worked|heard)\b",
    r"\bnot\s*(?:read|studied|familiar|aware)\b",
    r"\bnever\s*(?:worked|studied|used|read|heard)\b",
    r"\bcan\s*(?:we|you|u)\s*skip\b",
    r"\b(?:please\s+)?skip(?:\s+(?:this|that|it|the(?:\s+\w+)?\s+question))?\b",
    r"\bi\s+(?:want\s+to|would\s+like\s+to)\s+skip\b",
    r"\blet'?s\s+skip\b",
    r"\bskip\s+(?:this|that|it)\b",
    r"\bpass\s+on\s+(?:this|that|it)\b",
    r"\bnext\s*(?:question)\b",
    r"\b(?:let'?s|please)\s+move\s+on\b",
    r"\bmove\s+on\s+(?:please|to\s+(?:the\s+)?(?:next|another))\b",
    r"\blet'?s\s*(?:talk\s*about\s+something\s+else)\b",
    r"\bi\s*forgot\b",
    r"\bi\s*(?:don'?t|dont)\s*remember\b",
    r"\bidk\b",
    r"\bdunno\b",
    r"\bnot\s*my\s*area\b",
    r"\bhaven'?t\s*had\s*a\s*chance\b",
    r"\bsorry\s*,?\s*(?:i\s*)?(?:don'?t|dont|haven'?t|no)\b"
]

TOPIC_KEYWORDS_MAP = {
    "oop": "Object-Oriented Programming",
    "oops": "Object-Oriented Programming",
    "object oriented": "Object-Oriented Programming",
    "object-oriented": "Object-Oriented Programming",
    "java": "Java",
    "python": "Python",
    "sql": "MySQL & Database Management",
    "mysql": "MySQL & Database Management",
    "dbms": "MySQL & Database Management",
    "database": "MySQL & Database Management",
    "databases": "MySQL & Database Management",
    "dsa": "Data Structures and Algorithms",
    "data structure": "Data Structures and Algorithms",
    "data structures": "Data Structures and Algorithms",
    "algorithm": "Data Structures and Algorithms",
    "algorithms": "Data Structures and Algorithms",
    "c++": "C++",
    "cpp": "C++",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "deep learning": "Deep Learning",
    "neural network": "Neural Networks",
    "neural networks": "Neural Networks",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
}


def detect_requested_topic(text: str, candidate_skills: list = None) -> str | None:
    """
    Extracts an explicit technical topic requested by candidate in speech
    (e.g., 'Can we switch to Java?', 'Ask me OOP questions', 'Can you change to SQL?').
    Cross-references with candidate resume skills when available.
    """
    if not text:
        return None
    clean = text.lower()

    # 1. Match specific CS domain keywords (word boundary, never "java" ⊂ "javascript")
    for kw, canonical in TOPIC_KEYWORDS_MAP.items():
        if re.search(rf"\b{re.escape(kw)}\b", clean):
            if candidate_skills:
                for skill in candidate_skills:
                    if re.search(rf"\b{re.escape(str(skill).lower())}\b", clean):
                        return skill
            return canonical

    # 2. Check candidate resume skills directly
    if candidate_skills:
        for skill in candidate_skills:
            if len(skill) > 2 and re.search(rf"\b{re.escape(skill.lower())}\b", clean):
                return skill

    return None


INTRO_PATTERNS = [
    r"\bmy\s*name\s*is\b",
    r"\bmyself\s+[a-z]+\b",
    r"\bi\s*(?:am|'m)\s+(?:a\s+)?(?:fresh\s+)?(?:student|graduate|fresher|developer|engineer|coder)\b",
    r"\b(?:let\s*me\s*)?introduce\s*myself\b",
    r"\babout\s*myself\b",
    r"\bi\s*(?:am|'m)\s+pursuing\s+(?:my\s+)?(?:b\.?tech|degree|masters?|b\.?e|bachelor)\b",
    r"\bi\s*study\s+(?:at|computer|cs|engineering|science)\b",
    r"\bgraduated\s+from\b",
    r"\bhello\s*,?\s*(?:my\s*name\s*is|i\s*am\s+[a-z]+)\b"
]

TECHNICAL_ACTION_TERMS = {
    "using", "built", "implemented", "developed", "trained", "training", "dataset",
    "model", "algorithm", "pytorch", "tensorflow", "keras", "resnet", "flask", "fastapi",
    "react", "node", "database", "sql", "docker", "pipeline", "classification",
    "accuracy", "loss", "metrics", "optimization", "deployed", "api", "backend", "frontend"
}

REPEAT_PATTERNS = [
    r"\b(?:can|could|would)\s*you\s*(?:please\s*)?repeat\b",
    r"\brepeat\s*(?:the\s*)?(?:question|that|please)?\b",
    r"\b(?:didn'?t|did\s*not)\s*(?:hear|catch|get)\s*(?:that|you|clearly)?\b",
    r"\b(?:couldn'?t|could\s*not)\s*hear\s*(?:you|that)?\b",
    r"\b(?:say|speak)\s*(?:that\s*)?again\b",
    r"\bpardon\b",
    r"\bwhat\s*was\s*(?:the\s*)?question\b",
    r"\bcome\s*again\b",
    r"\bnot\s*(?:clear|audible)\b",
    r"\byou\s*are\s*not\s*audible\b",
    r"\byour\s*voice\s*(?:is\s*)?(?:breaking|low|unclear)\b"
]


PIVOT_PATTERNS = [
    r"\b(?:can|could)\s*(?:we|you)\s*(?:switch|change|talk|pivot)\s*(?:to|about)\b",
    r"\blet'?s\s*(?:switch\s*to|talk\s*about|discuss)\b",
    r"\bi\s*(?:want|would\s+like)\s+to\s+(?:talk|discuss|switch)\b",
    r"\bchange\s*(?:the\s*)?(?:topic|question)\s+to\b",
    r"\bask\s*me\s*(?:about|on)\b",
]


def detect_candidate_intent(
    text: str,
    stage: str = None,
    questions_asked: int = None
) -> str:
    """
    Classifies transcript text into:
    - REPEAT_REQUEST: Candidate could not hear or asked to repeat the question
    - TOPIC_CHANGE: Explicit request to switch topic (not a skip)
    - UNKNOWN_OR_SKIP: Negative intent, candidate doesn't know or wants to skip
    - SELF_INTRO: Introduction, greeting, or background declaration (only allowed in warmup/turn 0)
    - TECHNICAL_ANSWER: Default technical response
    """
    if not text or not text.strip():
        return "UNKNOWN_OR_SKIP"

    clean_txt = text.strip().lower()

    # 1. Check for repeat request first
    for pat in REPEAT_PATTERNS:
        if re.search(pat, clean_txt):
            return "REPEAT_REQUEST"

    # 2. Direct short negative replies
    if clean_txt in ["no", "nope", "nah", "skip", "idk", "pass", "next", "skip it", "skip this", "next question"]:
        return "UNKNOWN_OR_SKIP"

    # Short utterances that mention skip/pass should not be treated as technical answers
    # (but "skip connections" in a real ResNet answer is longer and stays TECHNICAL_ANSWER).
    word_count = len(re.findall(r"\b[\w']+\b", clean_txt))
    if word_count <= 12 and re.search(r"\b(skip|idk|dunno|next question)\b", clean_txt):
        if not re.search(r"\bskip connections?\b", clean_txt):
            return "UNKNOWN_OR_SKIP"
    if word_count <= 8 and re.search(r"\bpass\s+on\s+(this|that|it)\b", clean_txt):
        return "UNKNOWN_OR_SKIP"

    requested = detect_requested_topic(text)
    if requested and any(re.search(pat, clean_txt) for pat in PIVOT_PATTERNS):
        return "TOPIC_CHANGE"

    for pat in UNKNOWN_PATTERNS:
        if re.search(pat, clean_txt):
            return "UNKNOWN_OR_SKIP"

    # 4. Context check: Self-introduction is ONLY valid during the warmup stage / turn 0.
    # Once the candidate has moved past warmup, or if they mention a project or technical work,
    # it is ALWAYS a TECHNICAL_ANSWER.
    in_warmup = (stage is None or stage == "warmup") and (questions_asked is None or questions_asked == 0)

    if in_warmup:
        # If candidate already mentions a project or technical work, treat as technical answer immediately
        project_work_indicators = {
            "project", "built", "developed", "implemented", "trained", "training",
            "dataset", "model", "system", "app", "application", "created", "working",
            "classifier", "pipeline", "classification", "backend", "frontend", "api"
        }
        words = set(re.findall(r"\b[a-z]+\b", clean_txt))
        if words.intersection(TECHNICAL_ACTION_TERMS) or words.intersection(project_work_indicators):
            return "TECHNICAL_ANSWER"

        # Pure greeting or name declaration with no technical claims
        for pat in INTRO_PATTERNS:
            if re.search(pat, clean_txt):
                return "SELF_INTRO"

    return "TECHNICAL_ANSWER"

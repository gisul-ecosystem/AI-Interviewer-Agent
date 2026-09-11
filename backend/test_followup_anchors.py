"""Follow-up anchors come from THIS resume + Qwen, not a global tech list."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.followup_engine import (
    extract_mentioned_terms,
    is_quality_followup,
    is_valid_anchor,
    plan_followup,
)
from interviewer.ports.llm import LLMResult
from interviewer.services.anchor import resolve_followup_anchor
from interviewer.services.resume import parse_resume


def test_yeah_is_not_a_tool():
    cv = {"skills": ["PyTorch", "Python"], "projects": ["MoleCheck"]}
    mentioned = extract_mentioned_terms(
        "Yeah I used PyTorch for the classifier",
        candidate_dict=cv,
    )
    assert "Yeah" not in mentioned
    assert any(t.lower() == "pytorch" for t in mentioned), mentioned

    spec = plan_followup(
        "Yeah I have used this",
        eval_res={"depth": "med", "missing_concepts": [], "score": 0.6},
        candidate_dict=cv,
        topic="MoleCheck",
        focus="failure",
        project_name="MoleCheck",
        project_thread={"project": "MoleCheck", "last_anchor": "PyTorch", "asked_probes": ["implementation"]},
    )
    assert spec["anchor_term"].lower() != "yeah", spec
    assert spec["anchor_term"].lower() in {"pytorch", "molecheck"}
    assert "yeah" not in spec["spoken_fallback"].lower()
    print("[ok] Yeah is discourse, not a tool")


def test_certifications_header_is_not_a_skill():
    parsed = parse_resume(
        """
Priya Sharma

SKILLS
Python, PyTorch

CERTIFICATIONS
AWS Cloud Practitioner

PROJECTS
MoleCheck - skin lesion classifier
"""
    )
    lowered_skills = {s.lower() for s in parsed["skills"]}
    lowered_projects = {p.lower() for p in parsed["projects"]}
    assert "certifications" not in lowered_skills
    assert "certifications" not in lowered_projects
    assert "molecheck" in lowered_projects

    mentioned = extract_mentioned_terms(
        "Certifications is on my resume and Sorry I also used Weather",
        candidate_dict={"skills": parsed["skills"] + ["Certifications"], "projects": parsed["projects"]},
    )
    assert not any(t.lower() in {"certifications", "sorry", "weather"} for t in mentioned), mentioned
    print("[ok] Certifications header filtered")


def test_rejects_yeah_misbehaved_question():
    spec = {
        "project_name": "MoleCheck",
        "required_grounding": ["MoleCheck", "PyTorch"],
        "spoken_fallback": "On MoleCheck, you mentioned PyTorch. What broke first?",
    }
    assert is_quality_followup("When Yeah misbehaved — timeout, bad input, or drift — what broke first?", spec) is False
    assert is_valid_anchor("Yeah") is False
    assert is_valid_anchor("Certifications") is False
    print("[ok] bizarre filler questions rejected")


def test_any_resume_without_global_tech_list():
    """A backend CV must work even though PyTorch is not on a hardcoded allowlist."""
    cv = {
        "skills": ["Rust", "tonic", "gRPC"],
        "projects": ["Ferrite Mesh"],
        "raw_text": "Ferrite Mesh\nSkills: Rust, tonic, gRPC\n",
    }
    mentioned = extract_mentioned_terms(
        "On Ferrite Mesh I wired gRPC with tonic for backpressure",
        candidate_dict=cv,
    )
    lowered = {t.lower() for t in mentioned}
    assert "pytorch" not in lowered
    assert "grpc" in lowered or "tonic" in lowered, mentioned

    spec = plan_followup(
        "Yeah I have used this",
        eval_res={"depth": "med", "missing_concepts": [], "score": 0.6},
        candidate_dict=cv,
        topic="Ferrite Mesh",
        focus="failure",
        project_name="Ferrite Mesh",
        project_thread={
            "project": "Ferrite Mesh",
            "last_anchor": "tonic",
            "asked_probes": ["implementation"],
        },
    )
    assert spec["anchor_term"].lower() in {"tonic", "grpc", "ferrite mesh"}
    assert "yeah" not in spec["spoken_fallback"].lower()
    print("[ok] Ferrite Mesh / gRPC CV needs no PyTorch dictionary")


def test_named_cv_term_skips_llm():
    cv = {"skills": ["Rust", "gRPC"], "projects": ["Ferrite Mesh"]}
    calls = {"n": 0}

    class _Boom:
        async def complete(self, request):
            calls["n"] += 1
            raise AssertionError("LLM must not run when the answer already names a CV term")

    with patch("interviewer.adapters.registry.get_llm", return_value=_Boom()):
        got = asyncio.run(
            resolve_followup_anchor(
                "I used gRPC for streaming",
                cv,
                last_anchor="Rust",
                project="Ferrite Mesh",
            )
        )
    assert got.lower() == "grpc"
    assert calls["n"] == 0
    print("[ok] named CV term is a fast path")


def test_vague_answer_llm_must_stay_on_this_cv():
    cv = {"skills": ["Rust", "gRPC"], "projects": ["Ferrite Mesh"]}
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=LLMResult(ok=True, text='{"anchor": "gRPC"}'))
    with patch("interviewer.adapters.registry.get_llm", return_value=mock_llm):
        got = asyncio.run(
            resolve_followup_anchor(
                "Yeah I have used this",
                cv,
                last_question="Walk me through gRPC on Ferrite Mesh",
                last_anchor="gRPC",
                project="Ferrite Mesh",
            )
        )
    assert got.lower() in {"grpc", "rust", "ferrite mesh"}
    mock_llm.complete.assert_awaited()

    mock_llm.complete = AsyncMock(return_value=LLMResult(ok=True, text='{"anchor": "PyTorch"}'))
    with patch("interviewer.adapters.registry.get_llm", return_value=mock_llm):
        rejected = asyncio.run(
            resolve_followup_anchor(
                "Yeah I have used this",
                cv,
                last_anchor="gRPC",
                project="Ferrite Mesh",
            )
        )
    assert rejected.lower() != "pytorch"
    assert rejected.lower() in {"grpc", "ferrite mesh", "rust"}
    print("[ok] Qwen cannot invent a tool that is not on this resume")


def test_same_anchor_does_not_repeat_project_name():
    spec = plan_followup(
        "I owned the whole pipeline end to end.",
        eval_res={"depth": "med", "missing_concepts": [], "score": 0.6},
        candidate_dict={"skills": ["PyTorch"], "projects": ["MoleCheck"]},
        topic="MoleCheck",
        focus="implementation",
        project_name="MoleCheck",
        project_thread={"project": "MoleCheck", "last_anchor": "MoleCheck", "asked_probes": []},
    )
    spoken = spec["spoken_fallback"].lower()
    assert spoken.count("molecheck") == 1, spec["spoken_fallback"]
    assert "you mentioned molecheck" not in spoken
    print("[ok] collapsed On MoleCheck, you mentioned MoleCheck")


if __name__ == "__main__":
    test_yeah_is_not_a_tool()
    test_certifications_header_is_not_a_skill()
    test_rejects_yeah_misbehaved_question()
    test_any_resume_without_global_tech_list()
    test_named_cv_term_skips_llm()
    test_vague_answer_llm_must_stay_on_this_cv()
    test_same_anchor_does_not_repeat_project_name()
    print("[SUCCESS] Follow-up anchors are resume-grounded, not hardcoded per product.")

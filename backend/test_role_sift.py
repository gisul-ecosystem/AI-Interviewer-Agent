"""Role plan: projects from the CV, skills = CV ∩ role."""

from backend.role_sift import sift_skills_for_role
from interviewer.services.resume import parse_resume


ZERO_COVERAGE = """
Arjun Nair
Bengaluru

SKILLS
Rust, Go, Kubernetes, Terraform, gRPC, LangGraph, vLLM, DuckDB

PROJECTS
Ferrite Mesh - A service mesh sidecar written in Rust handling mTLS and retries for 40 microservices.
Orchestra - LangGraph agent orchestrator serving vLLM behind gRPC with DuckDB checkpointing.
"""


def test_aiml_intersection():
    sift = sift_skills_for_role(
        {
            "skills": ["Python", "PyTorch", "React", "MongoDB"],
            "projects": ["MoleCheck"],
        },
        role_override="aiml",
    )
    assert sift["track"] == "aiml"
    assert any("pytorch" in s.lower() or "python" in s.lower() for s in sift["intersection"])
    assert any("react" in s.lower() for s in sift["cv_only"])
    assert sift["interview_skills"] == sift["intersection"]
    assert "React" not in sift["interview_skills"]
    print("[ok] AIML intersection", sift["intersection"], "gaps", sift["role_gaps"][:3])


def test_webd_and_hr():
    web = sift_skills_for_role(
        {"skills": ["React.js", "Node.js", "HTML"], "projects": ["Campus portal"]},
        role_override="webd",
    )
    assert web["track"] == "webd"
    assert web["style"] == "technical"
    assert web["interview_projects"] == ["Campus portal"]
    hr = sift_skills_for_role({"skills": ["Communication", "Hiring"], "projects": []}, role_override="hr")
    assert hr["track"] == "hr"
    assert hr["style"] == "behavioral"
    print("[ok] WebD", web["interview_skills"][:4], "HR", hr["interview_skills"][:4])


def test_backend_does_not_invent_python():
    parsed = parse_resume(ZERO_COVERAGE)
    assert "Ferrite Mesh" in parsed["projects"]
    assert "Orchestra" in parsed["projects"]
    sift = sift_skills_for_role(parsed, role_override="backend")
    interview = [s.lower() for s in sift["interview_skills"]]
    assert "python" not in interview
    assert "node.js" not in interview
    assert any(s in interview for s in ("rust", "go", "kubernetes", "grpc"))
    assert sift["interview_projects"] == parsed["projects"][:3]
    assert "Technical project" not in sift["interview_projects"]
    print("[ok] backend intersection", sift["interview_skills"], "projects", sift["interview_projects"])


def test_empty_intersection_is_project_only():
    sift = sift_skills_for_role(
        {"skills": ["Blender", "Figma"], "projects": ["Campus portal"]},
        role_override="aiml",
    )
    assert sift["intersection"] == []
    assert sift["interview_skills"] == []
    assert sift["interview_projects"] == ["Campus portal"]
    print("[ok] empty intersection stays empty; projects still drive the interview")


def test_metadata_never_becomes_a_project():
    sift = sift_skills_for_role(
        {
            "skills": ["Python"],
            "projects": ["Certifications", "Education", "MoleCheck", "Domains"],
        },
        role_override="aiml",
    )
    lowered = [p.lower() for p in sift["interview_projects"]]
    assert "certifications" not in lowered
    assert "education" not in lowered
    assert "domains" not in lowered
    assert "molecheck" in lowered
    print("[ok] metadata titles stripped from projects", sift["interview_projects"])


def test_no_jd_does_not_guess_aiml():
    rust = sift_skills_for_role(
        {"skills": ["Rust", "gRPC"], "projects": ["Ferrite Mesh"]},
        role_override="auto",
    )
    assert rust["track"] != "aiml"
    assert rust["track"] in ("backend", "resume")
    assert "pytorch" not in [s.lower() for s in rust["interview_skills"]]

    unknown = sift_skills_for_role(
        {"skills": ["Blender", "Figma"], "projects": ["Campus portal"]},
        role_override="auto",
    )
    assert unknown["track"] == "resume"
    assert unknown["interview_skills"]
    assert not unknown["role_gaps"]
    print("[ok] no-JD track", rust["track"], unknown["track"], unknown["interview_skills"])


if __name__ == "__main__":
    test_aiml_intersection()
    test_webd_and_hr()
    test_backend_does_not_invent_python()
    test_empty_intersection_is_project_only()
    test_metadata_never_becomes_a_project()
    test_no_jd_does_not_guess_aiml()
    print("[SUCCESS] Role sift checks passed.")

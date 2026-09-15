"""Resume extraction: projects are required, skills come from the CV text."""

from pathlib import Path

from interviewer.services.resume import ingest_resume, normalize_resume_text, parse_resume, project_title

ROOT = Path(__file__).resolve().parents[1]


def test_multiline_projects_and_skills():
    parsed = parse_resume(
        """
Arjun Nair

SKILLS
Rust, Go, Kubernetes, Terraform, gRPC, LangGraph, vLLM, DuckDB

PROJECTS
Ferrite Mesh - A service mesh sidecar written in Rust handling mTLS and retries for 40 microservices.
Orchestra - LangGraph agent orchestrator serving vLLM behind gRPC with DuckDB checkpointing.
"""
    )
    assert parsed["projects"] == ["Ferrite Mesh", "Orchestra"]
    lowered = {s.lower() for s in parsed["skills"]}
    for must in ("rust", "go", "kubernetes", "terraform", "grpc", "langgraph", "vllm", "duckdb"):
        assert must in lowered, must
    print("[ok]", parsed["projects"], parsed["skills"][:8])


def test_long_description_becomes_title():
    assert project_title("MoleCheck - A CNN image classifier for skin lesions using PyTorch") == "MoleCheck"
    parsed = parse_resume(
        "TECHNICAL SKILLS\nPython, PyTorch, Docker, SQL\n\nPROJECTS\n"
        "MoleCheck - A CNN image classifier for skin lesions using PyTorch and transfer learning."
    )
    assert "MoleCheck" in parsed["projects"]
    print("[ok] MoleCheck kept, long description stripped")


def test_pdf_ligatures_and_classification():
    text = (
        "Aditya Bargujar\n\nSKILLS\nPython, TensorFlow, classiﬁcation, Speciﬁcity\n\n"
        "PROJECTS\nMoleCheck - A CNN image classiﬁer for skin lesions.\n"
    )
    cleaned = normalize_resume_text(text)
    assert "\ufb01" not in cleaned
    assert "classification" in cleaned
    parsed = ingest_resume(text)
    assert parsed["name"] == "Aditya Bargujar"
    assert "MoleCheck" in parsed["projects"]
    lowered = {s.lower() for s in parsed["skills"]}
    assert "python" in lowered
    assert "tensorflow" in lowered
    print("[ok] ligatures normalized", parsed["projects"], parsed["skills"][:6])


def test_job_line_is_not_a_project():
    parsed = ingest_resume(
        """
Aditya Bargujar

SKILLS
Python, PyTorch

EXPERIENCE
Technology Intern, GISUL, 2026

PROJECTS
MoleCheck - A CNN image classifier for skin lesions using PyTorch.
Mental Health Predictor - End-to-end mental health screening model.
"""
    )
    assert parsed["projects"] == ["MoleCheck", "Mental Health Predictor"]
    print("[ok] internship not treated as a project", parsed["projects"])


def test_unlabeled_education_experience_and_project_titles():
    parsed = ingest_resume(
        """
Aditya Bargujar
aditya@example.com

EDUCATION
B.Tech in Computer Science
Jaypee University of Information Technology
CGPA: 8.21

EXPERIENCE
Software Developer Intern | GISUL | May 2025 – Present
Built internal dashboards for field teams.

PROJECTS
Ferrite Mesh
A service mesh sidecar written in Rust handling mTLS and retries.
MoleCheck
A CNN image classifier for skin lesions using PyTorch.
"""
    )
    assert parsed["degree"] and "b.tech" in parsed["degree"].lower()
    assert "jaypee" in (parsed["college"] or "").lower()
    assert parsed["role"] and "intern" in parsed["role"].lower()
    assert parsed["company"] and "gisul" in parsed["company"].lower()
    assert "Ferrite Mesh" in parsed["projects"]
    assert "MoleCheck" in parsed["projects"]
    assert not any("service mesh sidecar" in p.lower() for p in parsed["projects"])
    print("[ok] unlabeled CV fields", parsed["degree"], parsed["role"], parsed["company"], parsed["projects"])


def test_glued_project_line_keeps_titles_not_descriptions():
    parsed = ingest_resume(
        """
Aditya Bargujar

PROJECTS
MoleCheck, Engineered a binary skin lesion classification model, Implemented an end-to-end TensorFlow pipeline.
Mental Health Predictor Live Demo
Performed exploratory data analysis including target distribution.
Project Deep Dive
"""
    )
    titles = [p.lower() for p in parsed["projects"]]
    assert "molecheck" in titles
    assert any("mental health" in p for p in titles)
    assert not any("engineered" in p for p in titles)
    assert not any("deep dive" in p for p in titles)
    print("[ok] glued project line", parsed["projects"])


def test_verb_glued_project_line_without_commas():
    parsed = ingest_resume(
        "PROJECTS\n"
        "MoleCheck Engineered a binary skin lesion classification model (Malignant vs. Benign) dataset. "
        "Mental Health Predictor Live Demo Performed exploratory data analysis including target distribution."
    )
    assert parsed["projects"] == ["MoleCheck", "Mental Health Predictor"]
    print("[ok] verb-glued project line", parsed["projects"])


def test_full_candidate_resume_extraction():
    sample = (
        "Aditya Bargujar\n"
        "Education\n"
        "Bachelor of Technology in Computer Science Engineering 2023 – 2027\n"
        "Jaypee University of Engineering and Technology, Guna CGPA: 7.2\n"
        "Skills\n"
        "• AI/ML: Machine Learning, Deep Learning, Neural Networks, LLMs\n"
        "• Programming: Python, Java, C++, JavaScript\n"
        "• Libraries & Frameworks: TensorFlow, Scikit-learn, NumPy, Pandas\n"
        "• Tools: Git, GitHub, MySQL\n"
        "• Core: Data Structures and Algorithms, Object-Oriented Programming\n"
        "Experience\n"
        "Technology Intern – GISUL, Bangalore August 2026 – Present\n"
        "Projects\n"
        "MoleCheck – Mole Identification System\n"
        "• Engineered a binary skin lesion classification model (Malignant vs. Benign) using MobileNetV2 with transfer learning on the ISIC 2019 dataset.\n"
        "• Implemented an end-to-end TensorFlow/Keras pipeline with data augmentation and class-weight balancing to address class imbalance.\n"
        "• Evaluated performance using Sensitivity, Specificity, Precision-Recall curves, and Confusion Matrix, emphasizing reduction of false negatives.\n"
        "Mental Health Predictor Live Demo\n"
        "• Performed exploratory data analysis including target distribution, correlation analysis, and relationships between stress, usage, sleep, and outcome score.\n"
    )
    parsed = ingest_resume(sample)
    assert parsed["projects"] == ["MoleCheck", "Mental Health Predictor"]
    assert "and Confusion Matrix" not in parsed["projects"]
    assert "Confusion Matrix" not in parsed["projects"]
    assert parsed["college"] == "Jaypee University of Engineering and Technology, Guna"
    assert parsed["degree"] == "Bachelor of Technology in Computer Science Engineering"
    assert parsed["company"] == "GISUL"
    assert "• AI" not in parsed["skills"]
    assert "ML: Machine Learning" not in parsed["skills"]
    assert "Python" in parsed["skills"]
    assert "Machine Learning" in parsed["skills"]
    print("[ok] full candidate resume parsed perfectly:", parsed["projects"], parsed["skills"][:5])


def test_resume_txt_does_not_treat_bullets_as_titles():
    parsed = ingest_resume(Path(ROOT / "resume.txt").read_text(encoding="utf-8") if (ROOT / "resume.txt").is_file() else (
        "Candidate Name: Aditya Bargujar\n"
        "Core Skills: Python, TensorFlow, Keras, MobileNetV2, Data Structures and Algorithms\n"
        "Projects: MoleCheck, Engineered a binary skin lesion classification model, dataset., "
        "Implemented an end-to-end TensorFlow/Keras pipeline, Mental Health Predictor Live Demo, "
        "Performed exploratory data analysis including target distribution\n"
    ))
    assert parsed["projects"] == ["MoleCheck", "Mental Health Predictor"], parsed["projects"]
    assert "Python" not in parsed["projects"]
    assert "TensorFlow" not in parsed["projects"]
    assert "Engineered" not in " ".join(parsed["projects"])
    print("[ok] resume dump keeps titles only", parsed["projects"])


def test_skills_are_not_copied_into_projects():
    parsed = ingest_resume(
        """
Aditya Bargujar
SKILLS
Python, TensorFlow, Keras, Object-Oriented Programming, Data Structures and Algorithms
PROJECTS
Python, TensorFlow
MoleCheck - CNN classifier using MobileNetV2.
Data Structures and Algorithms
"""
    )
    assert parsed["projects"] == ["MoleCheck"], parsed["projects"]
    lowered = {s.lower() for s in parsed["skills"]}
    assert "python" in lowered
    assert "tensorflow" in lowered
    print("[ok] skills stayed out of projects", parsed["projects"], parsed["skills"][:6])


def test_numbered_and_sentence_case_web_projects():
    parsed = ingest_resume(
        """
Riya Sharma
SKILLS
HTML, CSS, JavaScript, SQL

PROJECTS
1. Campus portal
2. E-commerce website using React
3. Chat Application | Node.js, Socket.io
"""
    )
    lowered = [p.lower() for p in parsed["projects"]]
    assert any("campus" in p and "portal" in p for p in lowered), parsed["projects"]
    assert any("commerce" in p or "e-commerce" in p for p in lowered), parsed["projects"]
    assert any("chat" in p for p in lowered), parsed["projects"]
    print("[ok] numbered web projects", parsed["projects"])


def test_title_label_and_academic_project_header():
    parsed = ingest_resume(
        """
Karan Mehta
ACADEMIC PROJECT
Title: Online Voting System
Developed a secure voting portal with JWT auth.
Title: IoT Based Smart Irrigation
"""
    )
    blob = " ".join(parsed["projects"]).lower()
    assert "voting" in blob, parsed["projects"]
    assert "irrigation" in blob or "iot" in blob, parsed["projects"]
    assert "title" not in blob
    print("[ok] labeled academic projects", parsed["projects"])


def test_two_column_skills_projects_header():
    parsed = ingest_resume(
        """
Aditya Bargujar
SKILLS          PROJECTS
Python          MoleCheck – Mole Identification System
TensorFlow      Mental Health Predictor Live Demo
"""
    )
    titles = [p.lower() for p in parsed["projects"]]
    assert "molecheck" in titles, parsed["projects"]
    assert any("mental health" in p for p in titles), parsed["projects"]
    print("[ok] two-column header", parsed["projects"])


def test_live_demo_is_not_a_project():
    parsed = ingest_resume(
        """
PROJECTS
Mental Health Predictor Live Demo
Live Demo
MoleCheck - CNN classifier
"""
    )
    lowered = [p.lower() for p in parsed["projects"]]
    assert "live demo" not in lowered
    assert "molecheck" in lowered
    assert any("mental health" in p for p in lowered)
    print("[ok] live demo stripped", parsed["projects"])


def test_llm_hallucinated_titles_are_dropped():
    parsed = ingest_resume(
        """
Aditya Bargujar
SKILLS
Python, TensorFlow
PROJECTS
MoleCheck - A CNN image classifier.
""",
        llm_fields={"projects": ["MadeUpApp", "TensorFlow", "MoleCheck"]},
    )
    assert parsed["projects"] == ["MoleCheck"], parsed["projects"]
    assert parsed["project_contexts"]["MoleCheck"]
    print("[ok] hallucinated titles dropped", parsed["projects"])


def test_llm_bullet_dump_is_not_kept_as_titles():
    parsed = ingest_resume(
        Path(ROOT / "resume.txt").read_text(encoding="utf-8"),
        llm_fields={
            "projects": [
                "MoleCheck",
                "Engineered a binary skin lesion classification model (Malignant vs. Benign)",
                "dataset.",
                "Implemented an end-to-end TensorFlow/Keras pipeline with data augmentation and",
                "Evaluated performance using Sensitivity, Specificity, Precision-Recall curves",
                "negatives.",
                "Mental Health Predictor Live Demo",
                "Performed exploratory data analysis including target distribution, correlation",
            ]
        },
    )
    assert parsed["projects"] == ["MoleCheck", "Mental Health Predictor"], parsed["projects"]
    print("[ok] LLM bullet dump filtered", parsed["projects"])


if __name__ == "__main__":
    test_multiline_projects_and_skills()
    test_long_description_becomes_title()
    test_pdf_ligatures_and_classification()
    test_job_line_is_not_a_project()
    test_unlabeled_education_experience_and_project_titles()
    test_glued_project_line_keeps_titles_not_descriptions()
    test_verb_glued_project_line_without_commas()
    test_resume_txt_does_not_treat_bullets_as_titles()
    test_skills_are_not_copied_into_projects()
    test_full_candidate_resume_extraction()
    test_numbered_and_sentence_case_web_projects()
    test_title_label_and_academic_project_header()
    test_two_column_skills_projects_header()
    test_live_demo_is_not_a_project()
    test_llm_hallucinated_titles_are_dropped()
    test_llm_bullet_dump_is_not_kept_as_titles()
    print("[SUCCESS] Resume extraction checks passed.")

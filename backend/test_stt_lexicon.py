"""STT restore: MoleCheck/model and MobileNetV2/movile must not clobber English."""

from backend.transcript_corrector import correct_transcript_fast
from interviewer.services.stt_lexicon import protect_transcript


CV = {
    "name": "Priya Sharma",
    "skills": ["PyTorch", "MobileNetV2", "Python"],
    "projects": ["MoleCheck"],
    "raw_text": "MoleCheck uses MobileNetV2 for skin-lesion classification.",
}


def test_molecheck_and_mobilenet_from_stt():
    features = CV
    built = protect_transcript(
        "I built model using movile for the classifier",
        features,
        active_topic="Project: MoleCheck",
    )
    assert "MoleCheck" in built, built
    assert "MobileNetV2" in built, built
    assert "model" not in built.lower() or "MoleCheck" in built

    fast = correct_transcript_fast(
        "I worked on model with movile",
        candidate_dict=features,
        hotwords=["MoleCheck", "MobileNetV2"],
        active_topic="Project: MoleCheck",
    )
    assert "MoleCheck" in fast, fast
    assert "MobileNetV2" in fast, fast

    print("[SUCCESS] MoleCheck/model and MobileNetV2/movile restored.")


def test_english_model_is_kept():
    kept = protect_transcript(
        "I trained a model for classification accuracy",
        CV,
        active_topic="Skill: PyTorch",
    )
    assert "trained a model" in kept.lower(), kept
    assert "MoleCheck" not in kept

    print("[SUCCESS] Ordinary English 'model' is not rewritten.")


def test_short_fragments_do_not_eat_model():
    from interviewer.services.stt_lexicon import collect_cv_terms

    terms = collect_cv_terms(CV)
    lowered = {t.lower() for t in terms}
    assert "molecheck" in lowered
    assert "mole" not in lowered
    assert "check" not in lowered
    print("[SUCCESS] CV terms keep full names, not Mole/Check fragments.")


if __name__ == "__main__":
    test_molecheck_and_mobilenet_from_stt()
    test_english_model_is_kept()
    test_short_fragments_do_not_eat_model()

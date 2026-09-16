import io
import wave
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from interviewer.services.whisper_stt import (
    _clean_caption,
    _dedupe_transcript,
    _pcm16_to_wav,
    _pcm_is_silent,
    _api_model_name,
    WhisperStreamSession,
    merge_captions,
    SAMPLE_RATE,
    BYTES_PER_SAMPLE,
)


def test_clean_caption():
    assert _clean_caption("Thank you for watching.") == ""
    assert _clean_caption("Thanks.") == ""
    assert _clean_caption("bye!") == ""
    assert _clean_caption("I worked on Python and FastAPI.") == "I worked on Python and FastAPI."
    assert _clean_caption("   ") == ""
    assert _clean_caption("okay") == ""
    assert _clean_caption("Thank you for watching. I built MoleCheck in PyTorch.") == "I built MoleCheck in PyTorch."
    assert _clean_caption("Please subscribe") == ""


def test_api_model_name():
    assert _api_model_name("large-v3-turbo") == "whisper-large-v3-turbo"
    assert _api_model_name("turbo") == "whisper-large-v3-turbo"
    assert _api_model_name("whisper-large-v3-turbo") == "whisper-large-v3-turbo"
    assert _api_model_name("large-v3") == "whisper-large-v3"
    assert _api_model_name("") == "whisper-large-v3-turbo"


def test_pcm16_to_wav():
    # 1 second of silence
    pcm = b"\x00" * (SAMPLE_RATE * BYTES_PER_SAMPLE)
    wav_bytes = _pcm16_to_wav(pcm)
    assert len(wav_bytes) > len(pcm)
    assert wav_bytes.startswith(b"RIFF")
    
    # Read back with wave
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == BYTES_PER_SAMPLE
        assert wf.getframerate() == SAMPLE_RATE
        frames = wf.readframes(SAMPLE_RATE)
        assert len(frames) == len(pcm)


def test_pcm_is_silent():
    # Pure silence
    pcm_silence = b"\x00" * (SAMPLE_RATE * BYTES_PER_SAMPLE)
    assert _pcm_is_silent(pcm_silence) is True

    # Too short audio (< 0.2s)
    short_pcm = b"\x00" * (SAMPLE_RATE // 10)
    assert _pcm_is_silent(short_pcm) is True

    # Loud speech simulation (sine wave with RMS ~ 3500)
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    loud_tone = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16).tobytes()
    assert _pcm_is_silent(loud_tone, rms_floor=220.0) is False


def test_whisper_stream_session_buffering():
    session = WhisperStreamSession(prompt="MoleCheck Python")
    assert session.has_enough() is False
    assert session.last_text == ""

    # Add 1 second of PCM (not enough for 2.0s threshold)
    pcm_1s = b"\x01\x00" * SAMPLE_RATE
    session.add_pcm(pcm_1s)
    assert session.has_enough() is False

    # Add another 1.5 seconds of PCM (total 2.5s >= 2.0s)
    pcm_1_5s = b"\x01\x00" * int(SAMPLE_RATE * 1.5)
    session.add_pcm(pcm_1_5s)
    assert session.has_enough() is True

    # Test reset
    session.reset()
    assert session.has_enough() is False
    assert session.last_text == ""


def test_merge_captions_keeps_start_of_answer():
    start = "I built MoleCheck as a skin lesion classifier using TensorFlow and Keras"
    window = "using TensorFlow and Keras with data augmentation"
    merged = merge_captions(start, window)
    assert merged.lower().startswith("i built molecheck")
    assert "data augmentation" in merged.lower()
    assert merge_captions(start, "using TensorFlow and Keras") == start
    print("[ok] caption merge keeps the start of the answer")


def test_dedupe_transcript_drops_repeated_sentence():
    once = "I built MoleCheck in PyTorch."
    assert _dedupe_transcript(f"{once} {once}") == once
    doubled = "this is a longer answer about my project this is a longer answer about my project"
    out = _dedupe_transcript(doubled)
    assert out.lower().count("this is a longer") == 1
    assert _dedupe_transcript("I used FastAPI for routing.") == "I used FastAPI for routing."

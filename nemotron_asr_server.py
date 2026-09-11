"""
nemotron_asr_server.py  —  v2 (Max Accuracy Edition)

WebSocket ASR server for nvidia/nemotron-speech-streaming-en-0.6b
using NeMo's cache-aware streaming inference.

Accuracy improvements over v1:
  - RIGHT_CONTEXT bumped from 6 → 13  (model sees ~1040ms of future audio
    before committing a word — significantly fewer substitutions)
  - AUDIO_BUFFER aligned to model's real shift_size stride (560ms) instead
    of the mismatched 400ms that caused partial frame boundaries
  - Flush-on-disconnect: leftover audio in the buffer is silence-padded and
    processed so the final words are never silently dropped
  - Silence flush detection: if a client sends explicit silence flush frames
    the server drains its buffer immediately without waiting for a full chunk
  - GPU inference offloaded to a thread-pool via asyncio.to_thread so the
    FastAPI event loop is never blocked
  - Enriched /health endpoint exposes the full streaming config so clients
    can auto-detect chunk sizing
  - Per-connection metrics (RTF, latency) logged on disconnect

Usage (on the inference machine):
    uvicorn nemotron_asr_server:app --host 0.0.0.0 --port 1111 --workers 1
"""

import asyncio
import json
import logging
import time
from typing import Optional

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import nemo.collections.asr as nemo_asr

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("nemotron_asr")

# ─── Model & Streaming Configuration ─────────────────────────────────────────
MODEL_NAME   = "nvidia/nemotron-speech-streaming-en-0.6b"
SAMPLE_RATE  = 16000

# Context window — the biggest lever for accuracy:
#   [70, 0]   → fastest, highest WER (no right context)
#   [70, 1]   → ~80 ms latency
#   [70, 6]   → ~480 ms latency
#   [70, 13]  → ~1040 ms latency
#   [140, 13] → ~1040 ms latency with 11.2s historic memory  ← v3: max accuracy
LEFT_CONTEXT  = 140
RIGHT_CONTEXT = 13

# Audio processing buffer — MUST match the model's chunk stride.
# shift_size[1] for RIGHT_CONTEXT=13 is 56 frames × 10 ms = 560 ms.
# We use exactly 560 ms to avoid partial-frame boundary artefacts.
AUDIO_BUFFER_MS      = 560
AUDIO_BUFFER_SAMPLES = int(SAMPLE_RATE * AUDIO_BUFFER_MS / 1000)  # 8 960 samples

# Silence-flush: if ≥ this many consecutive silent samples arrive, drain buffer
SILENCE_RMS_THRESHOLD = 1e-4          # float32 RMS below this = silence
SILENCE_FLUSH_MS      = 1600          # flush after 1.6 s of silence (prevents mid-clause resets)
SILENCE_FLUSH_SAMPLES = int(SAMPLE_RATE * SILENCE_FLUSH_MS / 1000)

# ─── Load Model (once at startup) ─────────────────────────    ────────────────────
logger.info("Loading model: %s …", MODEL_NAME)
_t0 = time.monotonic()
model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
model = model.to("cuda")
model.eval()
model.encoder.set_default_att_context_size([LEFT_CONTEXT, RIGHT_CONTEXT])
model.encoder.setup_streaming_params()
logger.info(
    "Model ready on %s  (%.1f s load time)",
    torch.cuda.get_device_name(0),
    time.monotonic() - _t0,
)
logger.info("Streaming cfg: %s", model.encoder.streaming_cfg.__dict__)

# Pre-compute constants from the model's streaming config
_PRE_ENCODE_CACHE_SIZE = model.encoder.streaming_cfg.pre_encode_cache_size[1]
_NUM_CHANNELS          = model.cfg.preprocessor.features
_DROP_EXTRA            = model.encoder.streaming_cfg.drop_extra_pre_encoded

# A thread-level lock so only one inference step runs at a time on the GPU
# (multi-client safety — GPU is not re-entrant for NeMo streaming state)
_gpu_lock = asyncio.Lock()

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Nemotron ASR Server", version="2.0")


# ─── Per-Connection State ─────────────────────────────────────────────────────
class StreamState:
    """
    Holds all per-connection cache tensors and audio buffering state.
    One instance per WebSocket client — never shared between connections.
    """

    def __init__(self):
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1, device=model.device)

        self.cache_pre_encode = torch.zeros(
            (1, _NUM_CHANNELS, _PRE_ENCODE_CACHE_SIZE), device=model.device
        )
        self.previous_hypotheses: Optional[object] = None
        self.pred_out_stream:     Optional[object] = None

        self.step_num     = 0
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.silence_acc  = 0          # running count of silent samples

        # Metrics
        self.connect_time    = time.monotonic()
        self.total_audio_s   = 0.0
        self.total_infer_s   = 0.0
        self.chunks_inferred = 0


# ─── Core Inference (sync — runs in thread via asyncio.to_thread) ─────────────
def _transcribe_chunk_sync(state: StreamState, audio_chunk: np.ndarray) -> str:
    """
    One cache-aware streaming inference step.
    Runs synchronously; must be called via asyncio.to_thread().
    """
    t0 = time.monotonic()

    audio_tensor = torch.tensor(audio_chunk, device=model.device).unsqueeze(0)
    audio_len    = torch.tensor([audio_tensor.shape[1]], device=model.device)

    # Preprocessor (mel filterbank)
    processed_signal, processed_signal_length = model.preprocessor(
        input_signal=audio_tensor, length=audio_len
    )

    # Prepend pre-encode cache
    processed_signal        = torch.cat([state.cache_pre_encode, processed_signal], dim=-1)
    processed_signal_length = processed_signal_length + state.cache_pre_encode.shape[-1]
    state.cache_pre_encode  = processed_signal[:, :, -_PRE_ENCODE_CACHE_SIZE:]

    drop_extra = 0 if state.step_num == 0 else _DROP_EXTRA

    with torch.no_grad():
        (
            state.pred_out_stream,
            transcribed_texts,
            state.cache_last_channel,
            state.cache_last_time,
            state.cache_last_channel_len,
            state.previous_hypotheses,
        ) = model.conformer_stream_step(
            processed_signal         = processed_signal,
            processed_signal_length  = processed_signal_length,
            cache_last_channel       = state.cache_last_channel,
            cache_last_time          = state.cache_last_time,
            cache_last_channel_len   = state.cache_last_channel_len,
            keep_all_outputs         = (state.step_num == 0),
            previous_hypotheses      = state.previous_hypotheses,
            previous_pred_out        = state.pred_out_stream,
            drop_extra_pre_encoded   = drop_extra,
            return_transcription     = True,
        )

    hyp  = transcribed_texts[0] if transcribed_texts else None
    text = hyp.text if hyp is not None else ""

    elapsed = time.monotonic() - t0
    state.total_infer_s   += elapsed
    state.chunks_inferred += 1
    state.step_num        += 1

    logger.debug(
        "step=%d  drop=%d  infer=%.0f ms  text=%r",
        state.step_num - 1, drop_extra, elapsed * 1000, text,
    )
    return text


async def transcribe_chunk(state: StreamState, audio_chunk: np.ndarray) -> str:
    """Async wrapper — acquires GPU lock then runs inference in a thread."""
    async with _gpu_lock:
        return await asyncio.to_thread(_transcribe_chunk_sync, state, audio_chunk)


# ─── Buffer Drain Helper ──────────────────────────────────────────────────────
async def drain_buffer(state: StreamState, ws: WebSocket, pad_silence: bool = False) -> None:
    """
    Process all complete chunks currently in audio_buffer.
    If pad_silence=True, the final partial chunk (if any) is zero-padded
    and also processed — use this on disconnect to avoid losing final words.
    """
    while state.audio_buffer.size >= AUDIO_BUFFER_SAMPLES:
        chunk = state.audio_buffer[:AUDIO_BUFFER_SAMPLES]
        state.audio_buffer = state.audio_buffer[AUDIO_BUFFER_SAMPLES:]
        text = await transcribe_chunk(state, chunk)
        await ws.send_text(json.dumps({"text": text}))

    if pad_silence and state.audio_buffer.size > 0:
        # Pad the remaining partial chunk with silence and process it
        padded = np.zeros(AUDIO_BUFFER_SAMPLES, dtype=np.float32)
        padded[: state.audio_buffer.size] = state.audio_buffer
        state.audio_buffer = np.empty(0, dtype=np.float32)
        text = await transcribe_chunk(state, padded)
        if text:
            await ws.send_text(json.dumps({"text": text}))


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Returns model status, GPU info, and the full streaming configuration
    so clients can auto-detect the correct chunk sizing.
    """
    cfg = model.encoder.streaming_cfg
    shift_size = list(getattr(cfg, "chunk_size", [0, AUDIO_BUFFER_SAMPLES // 160]))
    return {
        "status":  "ok",
        "model":   MODEL_NAME,
        "device":  torch.cuda.get_device_name(0),
        "streaming_config": {
            "left_context":          LEFT_CONTEXT,
            "right_context":         RIGHT_CONTEXT,
            "audio_buffer_ms":       AUDIO_BUFFER_MS,
            "audio_buffer_samples":  AUDIO_BUFFER_SAMPLES,
            "shift_size":            shift_size,
            "pre_encode_cache_size": _PRE_ENCODE_CACHE_SIZE,
        },
    }


@app.websocket("/ws/transcribe")
async def transcribe_ws(
    ws:          WebSocket,
    sample_rate: int = SAMPLE_RATE,
    channels:    int = 1,
):
    """
    WebSocket endpoint.

    Client protocol:
      - Send raw PCM16-LE audio bytes continuously (any sample rate via ?sample_rate=N)
      - Optionally send JSON {"action": "flush"} to force a mid-session drain
      - Server streams back JSON {"text": "<current transcript>"} after each chunk
      - On disconnect the server processes any remaining buffered audio first
    """
    await ws.accept()
    state = StreamState()

    logger.info(
        "Client connected — input %d Hz / %d ch → resample to %d Hz mono",
        sample_rate, channels, SAMPLE_RATE,
    )

    # Build resampler if needed (torchaudio polyphase — same quality as scipy.resample_poly)
    resampler: Optional[torchaudio.transforms.Resample] = None
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=SAMPLE_RATE
        ).to(model.device)

    last_text = ""

    try:
        while True:
            message = await ws.receive()

            # ── JSON control messages ──────────────────────────────────────
            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                    action  = payload.get("action", "")
                    if action == "flush":
                        logger.info("Flush requested by client — draining buffer with silence pad")
                        await drain_buffer(state, ws, pad_silence=True)
                    elif action == "stop":
                        break
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            # ── Binary audio frames ────────────────────────────────────────
            audio_bytes = message.get("bytes", b"")
            if not audio_bytes:
                continue

            raw = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if raw.size == 0:
                continue

            # Downmix multi-channel to mono
            if channels > 1:
                raw = raw.reshape(-1, channels).mean(axis=1)

            # Polyphase resample on GPU if needed
            if resampler is not None:
                raw = resampler(
                    torch.tensor(raw, device=model.device)
                ).cpu().numpy()

            raw = raw.astype(np.float32, copy=False)
            state.total_audio_s += raw.size / SAMPLE_RATE

            # ── Silence detection for auto-flush ──────────────────────────
            rms = float(np.sqrt(np.mean(raw ** 2)))
            if rms < SILENCE_RMS_THRESHOLD:
                state.silence_acc += raw.size
            else:
                state.silence_acc = 0   # reset on any speech

            state.audio_buffer = np.concatenate((state.audio_buffer, raw))

            # Auto-flush if prolonged silence detected (end of speech)
            if state.silence_acc >= SILENCE_FLUSH_SAMPLES and state.audio_buffer.size > 0:
                logger.info("Auto-flush: %.0f ms silence detected", SILENCE_FLUSH_SAMPLES / SAMPLE_RATE * 1000)
                state.silence_acc = 0
                await drain_buffer(state, ws, pad_silence=True)
                continue

            # ── Normal chunk processing ────────────────────────────────────
            await drain_buffer(state, ws, pad_silence=False)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Unexpected error in WebSocket handler: %s", exc)
    finally:
        # Process any leftover audio so final words are never lost
        try:
            if state.audio_buffer.size > 0:
                logger.info(
                    "Flushing %d remaining samples on disconnect …",
                    state.audio_buffer.size,
                )
                await drain_buffer(state, ws, pad_silence=True)
        except Exception:
            pass

        # Log per-session performance metrics
        session_s = time.monotonic() - state.connect_time
        rtf = state.total_infer_s / max(state.total_audio_s, 1e-6)
        logger.info(
            "Client disconnected — session=%.1f s  audio=%.1f s  "
            "infer=%.1f s  chunks=%d  RTF=%.3f",
            session_s,
            state.total_audio_s,
            state.total_infer_s,
            state.chunks_inferred,
            rtf,
        )

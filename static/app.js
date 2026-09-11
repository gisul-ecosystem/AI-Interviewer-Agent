/**
 * Gisul AI — Live Voice Interview & Streaming STT Client.
 *
 * Architecture:
 * 1. Web Audio API (AudioWorklet / ScriptProcessor) captures mic audio at native sample rate.
 * 2. Continuous phase-preserved resampler converts native SR → 16000 Hz PCM16 LE seamlessly
 *    without boundary clicks, clicks, pops, or 11Hz amplitude modulation glitches.
 * 3. Sends complete 160ms chunks (2560 samples = 5120 bytes) to FastAPI WebSocket relay.
 * 4. Displays real-time Whisper STT captions.
 * 5. Integrates Qwen3-4B phrasing and Kokoro TTS (no Chrome voice).
 */

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_MS = 160;
const CHUNK_SAMPLES = Math.floor(TARGET_SAMPLE_RATE * CHUNK_MS / 1000); // 2560 samples
const CHUNK_BYTES = CHUNK_SAMPLES * 2; // 5120 bytes (PCM16)

// State Variables
let isRecording = false;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let audioWorkletNode = null;
let ws = null;
let currentTranscript = '';
let startTime = 0;
let timerInterval = null;
let analyser = null;
let animationFrameId = null;
let currentSystemPrompt = '';
let currentMode = 'interview';
// Created once per interview and included with every answer so the backend can
// preserve question history, difficulty, and topic coverage.
let activeInterviewSessionId = null;
let activeResumeToken = null;
try {
  activeResumeToken = sessionStorage.getItem('gisul_resume_token') || null;
} catch (_) { }
let sessionInitialization = null;
let interviewDurationSeconds = 900;
let interviewEnded = false;
let agendaConfirmed = false;
let openingGreeting = '';
let speakQueue = [];
let speakChain = Promise.resolve();
let ttsBusy = false;
let ttsGeneration = 0;

// Silence-based auto follow-up
let silenceTimer = null;
let lastTranscriptSnapshot = '';
const SILENCE_THRESHOLD_MS = 8000;  // thinking pauses are normal; don't cut a live answer
const MIC_QUIET_MS = 6500;
const MIN_AUTO_SUBMIT_WORDS = 12;
const VOICE_RMS_THRESHOLD = 0.012;
const STT_JUNK_RE = /^(thank you\.?|thanks\.?|thank you for watching\.?|thanks for watching\.?|bye\.?|you\.?|the\.?|a\.?)$/i;
let questionStreamOpen = false;
let lastMicVoiceAt = 0;
let didSpeakChunk = false;
const VOICE_CMD_RE = /\b(pass|skip|idk|dunno|repeat|pardon|don'?t know|dont know|not sure|next question|didn'?t hear|can you repeat|could you repeat)\b/i;
function isVoiceCommand(text) {
  const t = (text || '').trim();
  if (!t) return false;
  const words = t.split(/\s+/).filter(Boolean);
  if (words.length > 12) return false;
  if (/\bskip connections?\b/i.test(t)) return false;
  return VOICE_CMD_RE.test(t);
}
let followUpInProgress = false;
let isFinalizingUtterance = false;
let greetingPlayed = false;
let isAiSpeaking = false;
let candidateHasSpokenInTurn = false;
let isEchoCooldown = false;
let shouldClearTranscriptOnNextSpeech = false;
let turnPhase = 'idle'; // idle | ai_speaking | cooldown | listening | processing
let sttReconnectAttempts = 0;
let currentVoice = 'af_heart'; // Kokoro TTS voice model (e.g. af_heart, af_bella, am_adam, bf_emma)

// Rolling PCM16 / Float32 resample buffer & state
let resampleBuffer = new Float32Array(0);
let resamplePhase = 0;

// ── Question window vs listening window ──
// While the AI is speaking, the mic stays closed. Energy-based barge-in (VAD)
// is off: speaker bleed, a cough, or "uh" was cutting the question mid-sentence.
// After TTS ends we wait out echo, drop that preroll (it is the question, not
// an answer), then open a clean listening window.
const PREROLL_MAX_SAMPLES = TARGET_SAMPLE_RATE * 2;
const BARGE_IN_FLUSH_SAMPLES = Math.floor(TARGET_SAMPLE_RATE * 0.6);
let prerollBuffer = new Float32Array(0);
let prerollFlushSamples = PREROLL_MAX_SAMPLES;

const BARGE_IN_ENABLED = false;
const BARGE_IN_MIN_FRAMES = 12;
const BARGE_IN_ABS_FLOOR = 0.08;
const BARGE_IN_NOISE_MULT = 6.0;
const BARGE_IN_GRACE_MS = 2500;
const LISTENING_WINDOW_MS = 2000;
let noiseFloor = 0.005;
let bargeInFrames = 0;
let currentAudioUrl = null;

// The server drops audio while it reconnects the upstream STT socket, so hold
// the gate shut until it acknowledges. Buffered audio is flushed after.
let sttReady = true;
let sttResetSafetyTimer = null;

// DOM Elements
const toggleMicBtn = document.getElementById('toggleMicBtn');
const micBtnText = document.getElementById('micBtnText');
const generateQuestionBtn = document.getElementById('generateQuestionBtn'); // hidden, kept for compat
const clearBtn = document.getElementById('clearBtn');
const liveText = document.getElementById('liveText');
const emptyTranscript = document.getElementById('emptyTranscript');
const transcriptBox = document.getElementById('transcriptBox');
const speechTimer = document.getElementById('speechTimer');
const wordCount = document.getElementById('wordCount');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const modeInterviewBtn = document.getElementById('modeInterviewBtn');
const modeGeneralBtn = document.getElementById('modeGeneralBtn');
const roleSelect = document.getElementById('roleSelect');
const jobDescriptionInput = document.getElementById('jobDescriptionInput');
if (jobDescriptionInput) {
  try {
    const savedJd = localStorage.getItem('gisul_job_description');
    if (savedJd) jobDescriptionInput.value = savedJd;
  } catch (_) { }
  jobDescriptionInput.addEventListener('input', () => {
    try { localStorage.setItem('gisul_job_description', jobDescriptionInput.value); } catch (_) { }
    agendaConfirmed = false;
    activeInterviewSessionId = null;
  });
}
const visualizerOverlay = document.getElementById('visualizerOverlay');
const waveformCanvas = document.getElementById('waveformCanvas');
const canvasCtx = waveformCanvas.getContext('2d');

const aiBubble = document.getElementById('aiBubble');
const emptyAiResponse = document.getElementById('emptyAiResponse');
const aiText = document.getElementById('aiText');
const typingCursor = document.getElementById('typingCursor');
const replayVoiceBtn = document.getElementById('replayVoiceBtn');
const voiceSynthesisToggle = document.getElementById('voiceSynthesisToggle');
const historyList = document.getElementById('historyList');
const noHistoryText = document.getElementById('noHistoryText');

// Final Transcript Elements
const finalTranscriptBox = document.getElementById('finalTranscriptBox');
const finalTranscriptText = document.getElementById('finalTranscriptText');
const finalMeta = document.getElementById('finalMeta');
const copyTranscriptBtn = document.getElementById('copyTranscriptBtn');

// Modal Elements
const openSettingsBtn = document.getElementById('openSettingsBtn');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');
const cancelSettingsBtn = document.getElementById('cancelSettingsBtn');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const settingsModal = document.getElementById('settingsModal');
const promptTextarea = document.getElementById('promptTextarea');


// ─── System Prompt ───────────────────────────────────────────────────────────

async function loadSystemPrompt() {
  try {
    const res = await fetch('/api/system-prompt');
    const data = await res.json();
    currentSystemPrompt = data.prompt || '';
    promptTextarea.value = currentSystemPrompt;
  } catch (err) {
    console.error('Failed to load system prompt:', err);
  }
}
loadSystemPrompt();


// ─── Interview Session ──────────────────────────────────────────────────────

function getJobDescription() {
  return (jobDescriptionInput && jobDescriptionInput.value ? jobDescriptionInput.value : '').trim();
}

function setTurnPhase(next) {
  turnPhase = next;
  isAiSpeaking = next === 'ai_speaking';
  isEchoCooldown = next === 'cooldown';
  followUpInProgress = next === 'ai_speaking' || next === 'processing' || next === 'cooldown';
}

function micGateOpen() {
  return turnPhase === 'listening' && !isEchoCooldown && sttReady;
}

function liveInterviewWsUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const params = new URLSearchParams({ mode: currentMode });
  if (activeInterviewSessionId) params.set('session_id', activeInterviewSessionId);
  return `${protocol}//${window.location.host}/ws/live-interview?${params.toString()}`;
}

function persistResumeToken(token) {
  activeResumeToken = token || null;
  try {
    if (token) sessionStorage.setItem('gisul_resume_token', token);
    else sessionStorage.removeItem('gisul_resume_token');
  } catch (_) { }
}

function clearSpeakQueue() {
  ttsGeneration += 1;
  speakQueue = [];
  speakChain = Promise.resolve();
  ttsBusy = false;
}

function enqueueSpeak(text) {
  const chunk = cleanAiText(text);
  if (!chunk || !voiceSynthesisToggle || !voiceSynthesisToggle.checked) return;
  speakQueue.push(chunk);
  didSpeakChunk = true;
  setTurnPhase('ai_speaking');
  if (!ttsBusy && !currentAudioPlayer) {
    ttsBusy = true;
    playQueuedSpeak();
  }
}

async function playQueuedSpeak() {
  const next = speakQueue.shift();
  if (!next) {
    ttsBusy = false;
    if (!questionStreamOpen && (isAiSpeaking || followUpInProgress || turnPhase === 'ai_speaking')) {
      onAiFinishedSpeaking(true);
    }
    return;
  }
  await speakText(next);
}

const agendaOverlay = document.getElementById('agendaOverlay');
const agendaProjects = document.getElementById('agendaProjects');
const agendaSkills = document.getElementById('agendaSkills');
const agendaLead = document.getElementById('agendaLead');
const agendaMeta = document.getElementById('agendaMeta');
const agendaConfirmBtn = document.getElementById('agendaConfirmBtn');
const agendaEditBtn = document.getElementById('agendaEditBtn');

function renderAgendaTags(node, items, emptyLabel) {
  if (!node) return;
  node.innerHTML = '';
  const list = (items || []).filter(Boolean);
  if (!list.length) {
    const tag = document.createElement('span');
    tag.className = 'skill-tag';
    tag.innerText = emptyLabel;
    node.appendChild(tag);
    return;
  }
  list.forEach((item) => {
    const tag = document.createElement('span');
    tag.className = 'skill-tag';
    tag.innerText = item;
    node.appendChild(tag);
  });
}

async function showAgendaConfirmation() {
  if (!activeResumeToken) {
    setStatus('error', 'Upload a resume first');
    return false;
  }
  const roleVal = roleSelect ? roleSelect.value : 'auto';
  const response = await fetch('/api/preview-agenda', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      resume_token: activeResumeToken,
      role: roleVal,
      job_description: getJobDescription(),
    }),
  });
  if (!response.ok) {
    setStatus('error', 'Could not build the interview agenda');
    return false;
  }
  const data = await response.json();
  const projects = data.projects || [];
  const skills = data.skills || [];
  if (agendaLead) {
    agendaLead.innerText = `We will be interviewing you on: ${projects.length ? projects.map((p, i) => `Project ${String.fromCharCode(65 + i)} (${p})`).join(', ') : 'your resume projects'
      }${skills.length ? `, then ${skills.join(', ')}` : ''}. Is this correct?`;
  }
  renderAgendaTags(agendaProjects, projects, 'No software projects found — re-upload the resume');
  renderAgendaTags(agendaSkills, skills, 'Skills will follow the projects');
  if (agendaMeta) {
    agendaMeta.innerText = `${data.duration_minutes || 15}-minute ${data.role_label || 'technical'} interview`;
  }
  if (agendaOverlay) agendaOverlay.hidden = false;
  return true;
}

if (agendaConfirmBtn) {
  agendaConfirmBtn.addEventListener('click', async () => {
    agendaConfirmed = true;
    if (agendaOverlay) agendaOverlay.hidden = true;
    try {
      await initInterviewSession();
      setStatus('', 'Agenda confirmed — click Start Interview');
    } catch (err) {
      agendaConfirmed = false;
      console.error(err);
      setStatus('error', 'Could not start the interview session');
    }
  });
}

if (agendaEditBtn) {
  agendaEditBtn.addEventListener('click', () => {
    if (agendaOverlay) agendaOverlay.hidden = true;
    if (settingsModal) settingsModal.classList.add('active');
  });
}

async function initInterviewSession() {
  if (currentMode !== 'interview') return null;
  if (sessionInitialization) return sessionInitialization;

  sessionInitialization = (async () => {
    const roleVal = roleSelect ? roleSelect.value : 'auto';
    const response = await fetch('/api/start-interview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        role: roleVal,
        job_description: getJobDescription(),
        resume_token: activeResumeToken || undefined,
      }),
    });

    if (!response.ok) throw new Error(`Could not start interview (${response.status})`);

    const data = await response.json();
    if (data.status !== 'ok' || !data.session_id) {
      throw new Error(data.message || 'The server did not return an interview session');
    }

    activeInterviewSessionId = data.session_id;
    greetingPlayed = false;
    interviewEnded = false;
    openingGreeting = data.opening_greeting || data.session?.opening_greeting || '';
    const remaining = Number(data.session?.interview_state?.time_remaining_seconds);
    interviewDurationSeconds = Number.isFinite(remaining) && remaining > 0 ? remaining : 900;
    console.log(`[Interview] Started session ${activeInterviewSessionId} (${interviewDurationSeconds}s)`);
    return activeInterviewSessionId;
  })();

  try {
    return await sessionInitialization;
  } finally {
    sessionInitialization = null;
  }
}

async function ensureInterviewSession() {
  if (currentMode !== 'interview') return null;
  return activeInterviewSessionId || initInterviewSession();
}


// ─── Status Helper ────────────────────────────────────────────────────────────

function setStatus(state, text) {
  statusBadge.className = 'status-badge ' + state;
  statusText.innerText = text;
}


// ─── Start / Stop ─────────────────────────────────────────────────────────────

toggleMicBtn.addEventListener('click', () => {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});


async function startRecording() {
  try {
    setStatus('thinking', 'Requesting Mic...');

    if (currentMode === 'interview') {
      if (!activeResumeToken) {
        setStatus('error', 'Upload a resume before starting');
        return;
      }
      if (!agendaConfirmed) {
        await showAgendaConfirmation();
        return;
      }
      setStatus('thinking', 'Preparing interview...');
      await ensureInterviewSession();
    }

    // Reset resampler state
    resampleBuffer = new Float32Array(0);
    resamplePhase = 0;

    // Hide final transcript from previous session
    finalTranscriptBox.style.display = 'none';

    // 1. Capture microphone
    await initAudioCapture();

    setStatus('listening', 'Connecting to STT...');

    // 2. Open WebSocket
    const wsUrl = liveInterviewWsUrl();
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    let hasError = false;

    ws.onopen = () => {
      console.log(`[WS Connected] Native SR: ${audioContext.sampleRate}Hz → 16000Hz (Smooth Phase-Preserved Resample)`);
      isRecording = true;
      sttReady = true;
      resetPreroll();
      toggleMicBtn.classList.add('recording');
      micBtnText.innerText = 'Stop Interview';
      if (generateQuestionBtn) generateQuestionBtn.disabled = true;
      setStatus('listening', 'Listening (Live ASR)');
      visualizerOverlay.classList.add('hidden');
      startTimer();

      // Auto-greeting: ask candidate for intro when interview starts
      if (currentMode === 'interview' && !greetingPlayed) {
        greetingPlayed = true;
        const greeting = openingGreeting || 'Hi, I am your interviewer. Please introduce yourself and the work you are proudest of.';
        setStatus('speaking', 'AI Greeting...');
        emptyAiResponse.style.display = 'none';
        aiBubble.style.display = 'flex';
        aiText.innerText = greeting;
        typingCursor.style.display = 'none';
        if (voiceSynthesisToggle.checked) {
          setTurnPhase('ai_speaking');
          speakText(greeting);
        } else {
          setTurnPhase('ai_speaking');
          setTimeout(() => onAiFinishedSpeaking(true), 1600);
        }
      } else {
        setTurnPhase('listening');
        startSilencePolling();
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'reset_ack') {
          console.log('[STT Relay] Clean STT session acknowledged for turn.');
          sttReady = true;
          if (sttResetSafetyTimer) { clearTimeout(sttResetSafetyTimer); sttResetSafetyTimer = null; }
          return;
        }
        if (data.type === 'transcript') {
          // Failsafe: Only force-unlock if stuck without audio playing for over 35s
          if (isAiSpeaking && !questionStreamOpen && aiSpeakingStartTime > 0 && (Date.now() - aiSpeakingStartTime > 45000)) {
            if (!currentAudioPlayer || currentAudioPlayer.paused || currentAudioPlayer.ended) {
              console.warn('[STT Listener Failsafe] Force unlocking stuck AI speaking state.');
              onAiFinishedSpeaking(true);
            }
          }
          if (followUpInProgress || isAiSpeaking || isEchoCooldown || questionStreamOpen || !micGateOpen()) return;
          updateTranscript(data.text);
        } else if (data.type === 'error') {
          console.error('[STT Error]', data.message);
          setStatus('error', 'Captions glitched — keep speaking, reconnecting');
        }
      } catch (err) {
        console.error('[WS] Message parse error:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
      setStatus('error', 'Caption link dropped — reconnecting');
    };

    ws.onclose = () => {
      if (!isRecording || interviewEnded) return;
      scheduleSttReconnect();
    };

  } catch (err) {
    console.error('[Recording] Start failed:', err);
    setStatus('error', 'Mic Error: ' + err.message);
    stopRecording(true);
  }
}


// ─── Pre-roll Buffer & Barge-in Detection ────────────────────────────────────

/** Recycle the upstream STT stream and keep the mic gate shut until it is back. */
function sendSttReset() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  try {
    ws.send(JSON.stringify({ action: 'reset' }));
    sttReady = false;
    if (sttResetSafetyTimer) clearTimeout(sttResetSafetyTimer);
    // Never deadlock the mic if the ack is lost.
    sttResetSafetyTimer = setTimeout(() => {
      if (!sttReady) {
        console.warn('[STT Relay] reset_ack missing after 2s; reopening mic gate.');
        sttReady = true;
      }
    }, 2000);
  } catch (_) {
    sttReady = true;
  }
}

function scheduleSttReconnect() {
  if (!isRecording || interviewEnded) return;
  const wait = Math.min(8000, 700 * Math.pow(2, sttReconnectAttempts++));
  setTimeout(() => {
    if (!isRecording || interviewEnded) return;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    const wsUrl = liveInterviewWsUrl();
    try {
      ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => {
        sttReconnectAttempts = 0;
        sttReady = true;
        setStatus('listening', turnPhase === 'listening' ? 'Captions restored' : statusText.innerText);
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'reset_ack') { sttReady = true; return; }
          if (data.type === 'transcript' && micGateOpen()) updateTranscript(data.text);
        } catch (_) { }
      };
      ws.onerror = () => setStatus('error', 'Caption reconnect failed');
      ws.onclose = () => { if (isRecording && !interviewEnded) scheduleSttReconnect(); };
    } catch (_) {
      scheduleSttReconnect();
    }
  }, wait);
}

function resetPreroll() {
  prerollBuffer = new Float32Array(0);
  prerollFlushSamples = PREROLL_MAX_SAMPLES;
  bargeInFrames = 0;
}

function pushPreroll(samples) {
  if (!samples || !samples.length) return;
  const merged = new Float32Array(prerollBuffer.length + samples.length);
  merged.set(prerollBuffer);
  merged.set(samples, prerollBuffer.length);
  prerollBuffer = merged.length > PREROLL_MAX_SAMPLES
    ? merged.slice(merged.length - PREROLL_MAX_SAMPLES)
    : merged;
}

function drainPreroll() {
  if (!prerollBuffer.length) return new Float32Array(0);
  const limit = prerollFlushSamples;
  const out = prerollBuffer.length > limit ? prerollBuffer.slice(prerollBuffer.length - limit) : prerollBuffer;
  prerollBuffer = new Float32Array(0);
  prerollFlushSamples = PREROLL_MAX_SAMPLES;
  return out;
}

function frameRMS(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

/** True once the candidate has spoken over the AI for long enough to be intentional. */
function detectBargeIn(samples) {
  const rms = frameRMS(samples);
  const threshold = Math.max(BARGE_IN_ABS_FLOOR, noiseFloor * BARGE_IN_NOISE_MULT);

  if (rms < threshold) {
    noiseFloor = noiseFloor * 0.95 + rms * 0.05;  // track the room, not the speaker
    bargeInFrames = 0;
    return false;
  }

  bargeInFrames++;
  return bargeInFrames >= BARGE_IN_MIN_FRAMES;
}

function stopAiPlayback() {
  if (currentAudioPlayer) {
    try {
      // Detach handlers first: pausing would otherwise fire onerror and
      // restart the question through the WebSpeech fallback.
      currentAudioPlayer.onended = null;
      currentAudioPlayer.onerror = null;
      currentAudioPlayer.pause();
    } catch (_) { }
    currentAudioPlayer = null;
  }
  if (currentAudioUrl) {
    try { URL.revokeObjectURL(currentAudioUrl); } catch (_) { }
    currentAudioUrl = null;
  }
  if ('speechSynthesis' in window) {
    try { window.speechSynthesis.cancel(); } catch (_) { }
  }
}

/**
 * Candidate started talking over the AI. Cut the question short and open the
 * mic immediately — no echo cooldown, no STT reset, because both would swallow
 * the words they are saying right now.
 */
function handleBargeIn() {
  console.log('[Barge-in] Candidate speech detected during playback; stopping AI.');
  clearSpeakQueue();
  stopAiPlayback();
  resampleBuffer = new Float32Array(0);

  isAiSpeaking = false;
  followUpInProgress = false;
  isEchoCooldown = false;
  candidateHasSpokenInTurn = true;
  shouldClearTranscriptOnNextSpeech = true;
  currentTranscript = '';
  liveText.innerText = '';
  lastTranscriptChangeTime = Date.now();
  bargeInFrames = 0;

  // Flush only the speech onset, not 2s of AI echo tail.
  prerollFlushSamples = BARGE_IN_FLUSH_SAMPLES;

  setStatus('listening', 'Listening...');
  if (isRecording) startSilencePolling();
}


// ─── Audio Capture & Continuous Seamless Resampling ──────────────────────────

async function initAudioCapture() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    }
  });

  audioContext = new (window.AudioContext || window.webkitAudioContext)();

  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  console.log(`[AudioContext] Native Rate: ${audioContext.sampleRate} Hz`);

  const source = audioContext.createMediaStreamSource(mediaStream);

  // Setup Analyser for visualizer
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);
  drawVisualizer();

  const nativeSR = audioContext.sampleRate;

  // Use ScriptProcessor with 4096 buffer size
  scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
  source.connect(scriptProcessor);
  const mute = audioContext.createGain();
  mute.gain.value = 0;
  scriptProcessor.connect(mute);
  mute.connect(audioContext.destination);

  window._audioCtx = audioContext;
  window._sp = scriptProcessor;
  window._src = source;

  scriptProcessor.onaudioprocess = (e) => {
    if (!isRecording) {
      resampleBuffer = new Float32Array(0);
      resetPreroll();
      return;
    }

    const inputSamples = e.inputBuffer.getChannelData(0);

    // ── Phase-Preserved Continuous Resampling ──
    // Always resample, even when muted, so resampler phase stays continuous.
    const resampled = resamplePhasePreserved(inputSamples, nativeSR, TARGET_SAMPLE_RATE);

    const socketReady = ws && ws.readyState === WebSocket.OPEN && sttReady;
    const gateOpen = socketReady && micGateOpen();

    if (!gateOpen) {
      pushPreroll(resampled);
      return;
    }

    let energy = 0;
    for (let i = 0; i < resampled.length; i += 8) energy += resampled[i] * resampled[i];
    const rms = Math.sqrt(energy / Math.max(1, resampled.length / 8));
    if (rms >= VOICE_RMS_THRESHOLD) lastMicVoiceAt = Date.now();

    // Gate is open: prepend anything captured while it was shut.
    const pending = drainPreroll();
    const merged = new Float32Array(resampleBuffer.length + pending.length + resampled.length);
    merged.set(resampleBuffer);
    merged.set(pending, resampleBuffer.length);
    merged.set(resampled, resampleBuffer.length + pending.length);
    resampleBuffer = merged;

    // Send complete 560ms chunks (8960 samples = 17920 bytes)
    while (resampleBuffer.length >= CHUNK_SAMPLES) {
      const chunkFloat = resampleBuffer.subarray(0, CHUNK_SAMPLES);
      resampleBuffer = resampleBuffer.slice(CHUNK_SAMPLES);
      const pcm16 = float32ToPCM16(chunkFloat);
      try {
        ws.send(pcm16.buffer);
      } catch (_) { }
    }
  };
}


/**
 * Continuous phase-preserved linear resampler.
 * Eliminates boundary pops, clicks, and 11Hz amplitude modulation clicks
 * by carrying fractional phase state across audio callbacks.
 */
function resamplePhasePreserved(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const output = [];

  let pos = resamplePhase;
  while (pos < input.length) {
    const idx = Math.floor(pos);
    const frac = pos - idx;

    const a = input[idx];
    const b = (idx + 1 < input.length) ? input[idx + 1] : a;

    output.push(a + frac * (b - a));
    pos += ratio;
  }

  // Preserve fractional phase remainder for the next buffer callback
  resamplePhase = pos - input.length;

  return new Float32Array(output);
}


/**
 * Convert Float32 [-1, 1] to PCM16 Int16Array with clipping protection.
 */
function float32ToPCM16(float32) {
  const pcm = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return pcm;
}


// ─── Stop Recording ───────────────────────────────────────────────────────────

function stopRecording(preserveStatus = false) {
  isRecording = false;
  clearSilenceTimer();
  toggleMicBtn.classList.remove('recording');
  micBtnText.innerText = 'Start Interview';
  if (generateQuestionBtn) generateQuestionBtn.disabled = true;
  if (!preserveStatus) setStatus('', 'Ready');
  visualizerOverlay.classList.remove('hidden');
  stopTimer();

  // Send remaining buffer padded with silence if non-empty
  if (ws && ws.readyState === WebSocket.OPEN && resampleBuffer.length > 0) {
    const padded = new Float32Array(CHUNK_SAMPLES);
    padded.set(resampleBuffer.subarray(0, Math.min(resampleBuffer.length, CHUNK_SAMPLES)));
    try {
      ws.send(float32ToPCM16(padded).buffer);
    } catch (_) { }
  }
  resampleBuffer = new Float32Array(0);
  resamplePhase = 0;
  resetPreroll();
  stopAiPlayback();

  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (audioContext) {
    try { audioContext.close(); } catch (_) { }
    audioContext = null;
  }
  if (ws) {
    if (ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ action: 'flush' })); } catch (_) { }
      setTimeout(() => { try { ws.close(); } catch (_) { } ws = null; }, 400);
    } else {
      try { ws.close(); } catch (_) { }
      ws = null;
    }
  }
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  clearCanvas();

  // ── Show Final Transcript Result ──
  showFinalTranscript();
}


// ─── Transcript ───────────────────────────────────────────────────────────────

function mergeStreamingTranscript(previous, incoming) {
  const prev = (previous || '').trim();
  const next = (incoming || '').trim();
  if (!next) return prev;
  if (!prev) return next;
  if (next === prev) return next;
  if (next.startsWith(prev)) return next;
  if (prev.startsWith(next)) return prev;
  const prevWords = prev.split(/\s+/);
  const nextWords = next.split(/\s+/);
  // Sliding Whisper windows are shorter than the accumulated answer — never drop the start.
  if (nextWords.length + 2 < prevWords.length && prev.toLowerCase().includes(next.toLowerCase())) {
    return prev;
  }
  const max = Math.min(prevWords.length, nextWords.length);
  for (let k = max; k >= 2; k--) {
    const tail = prevWords.slice(-k).join(' ').toLowerCase();
    const head = nextWords.slice(0, k).join(' ').toLowerCase();
    if (tail === head) {
      return [...prevWords, ...nextWords.slice(k)].join(' ');
    }
  }
  if (nextWords.length + 3 < prevWords.length) return prev;
  return `${prev} ${next}`;
}

function updateTranscript(text) {
  if (!text || isAiSpeaking || followUpInProgress || isEchoCooldown || questionStreamOpen) return;
  if (STT_JUNK_RE.test(text.trim())) return;

  // Clear previous candidate answer display only when candidate starts speaking new turn
  if (shouldClearTranscriptOnNextSpeech) {
    shouldClearTranscriptOnNextSpeech = false;
    currentTranscript = '';
    liveText.innerText = '';
    emptyTranscript.style.display = 'none';
  }

  const trimmed = text.trim();
  const words = trimmed.split(/\s+/).filter(Boolean);
  const fillerSet = new Set([
    'yeah', 'yes', 'yep', 'yup', 'ok', 'okay', 'sure', 'right',
    'uh', 'um', 'uhhuh', 'hmm', 'alright', 'fine', 'so', 'well',
    'like', 'actually', 'basically'
  ]);
  const nonFillerWords = words.filter(w => !fillerSet.has(w.toLowerCase().replace(/[^a-z]/g, '')));
  const isExplicitIntent = isVoiceCommand(trimmed);

  // Mark that candidate has begun answering (including one-word skip/pass/repeat)
  if (isExplicitIntent || nonFillerWords.length >= 2 || (words.length >= 3 && nonFillerWords.length >= 1)) {
    candidateHasSpokenInTurn = true;
  }

  if (trimmed && trimmed !== currentTranscript.trim()) {
    lastTranscriptChangeTime = Date.now();
  }
  currentTranscript = mergeStreamingTranscript(currentTranscript, text);
  liveText.innerText = currentTranscript;
  emptyTranscript.style.display = 'none';
  const keptWords = currentTranscript.split(/\s+/).filter(Boolean);
  wordCount.innerText = `${keptWords.length} words`;
  if (generateQuestionBtn && currentMode === 'interview') generateQuestionBtn.disabled = keptWords.length === 0;
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
}


// ─── Final Transcript Display ─────────────────────────────────────────────────

function showFinalTranscript() {
  const text = currentTranscript.trim();
  if (!text) return;

  const words = text.split(/\s+/).filter(Boolean);
  const elapsed = speechTimer.innerText || '00:00';

  finalTranscriptText.innerText = text;
  finalMeta.innerText = `${words.length} words · ${elapsed}`;
  finalTranscriptBox.style.display = 'block';
}

copyTranscriptBtn.addEventListener('click', () => {
  const text = finalTranscriptText.innerText;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const orig = copyTranscriptBtn.title;
    copyTranscriptBtn.title = 'Copied!';
    copyTranscriptBtn.style.color = 'var(--accent-emerald)';
    setTimeout(() => {
      copyTranscriptBtn.title = orig;
      copyTranscriptBtn.style.color = '';
    }, 1500);
  }).catch(() => { });
});


// ─── Waveform Visualizer ──────────────────────────────────────────────────────

function drawVisualizer() {
  if (!analyser) return;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  const draw = () => {
    animationFrameId = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = 'rgba(7, 9, 14, 0.4)';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

    canvasCtx.lineWidth = 2.5;
    const gradient = canvasCtx.createLinearGradient(0, 0, waveformCanvas.width, 0);
    gradient.addColorStop(0, '#00f0ff');
    gradient.addColorStop(0.5, '#6366f1');
    gradient.addColorStop(1, '#a855f7');
    canvasCtx.strokeStyle = gradient;

    canvasCtx.beginPath();
    const sliceWidth = waveformCanvas.width / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * waveformCanvas.height) / 2;
      i === 0 ? canvasCtx.moveTo(x, y) : canvasCtx.lineTo(x, y);
      x += sliceWidth;
    }
    canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
    canvasCtx.stroke();
  };
  draw();
}

function clearCanvas() {
  canvasCtx.fillStyle = 'rgba(7, 9, 14, 1)';
  canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
}


// ─── Timer ────────────────────────────────────────────────────────────────────

function formatClock(totalSeconds) {
  const clamped = Math.max(0, Math.floor(totalSeconds));
  const mins = String(Math.floor(clamped / 60)).padStart(2, '0');
  const secs = String(clamped % 60).padStart(2, '0');
  return `${mins}:${secs}`;
}

function startTimer() {
  startTime = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const remaining = interviewDurationSeconds - elapsed;
    speechTimer.innerText = formatClock(remaining);
    if (remaining <= 0 && !interviewEnded && currentMode === 'interview') {
      finishInterviewOnClient('time');
    }
  }, 1000);
}

async function finishInterviewOnClient(reason) {
  if (interviewEnded) return;
  interviewEnded = true;
  stopSilencePolling();
  if (isRecording) stopRecording();
  stopTimer();
  if (generateQuestionBtn) generateQuestionBtn.disabled = true;
  setStatus('ready', reason === 'time' ? 'Time is up — wrapping up' : 'Interview complete');
  if (!activeInterviewSessionId) return;
  try {
    await fetch('/api/end-interview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: activeInterviewSessionId }),
    });
  } catch (err) {
    console.warn('[Interview] end-interview failed', err);
  }
  openScorecardModal(activeInterviewSessionId);
}

function stopTimer() {
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}


// ─── Silence-Based Auto Follow-Up ────────────────────────────────────────────

let silencePollInterval = null;
let lastTranscriptChangeTime = 0;

function startSilencePolling() {
  stopSilencePolling();
  lastTranscriptChangeTime = Date.now();

  silencePollInterval = setInterval(() => {
    if (!isRecording || followUpInProgress || isFinalizingUtterance || isAiSpeaking || isEchoCooldown || questionStreamOpen || currentMode !== 'interview' || interviewEnded) return;

    // CRITICAL: NEVER auto-trigger follow-up if candidate has not actively spoken in this turn!
    if (!candidateHasSpokenInTurn) {
      return;
    }

    const now = Date.now();
    const currentText = currentTranscript.trim();
    if (!currentText || STT_JUNK_RE.test(currentText)) return;

    const words = currentText.split(/\s+/).filter(Boolean);
    const isShortOrRepeat = isVoiceCommand(currentText);
    if (words.length < 2 && !isShortOrRepeat) return;

    // Do NOT auto-trigger follow-up if candidate only said thinking/acknowledgment filler words
    const fillerWords = new Set([
      'yeah', 'yes', 'yep', 'yup', 'ok', 'okay', 'sure', 'right',
      'uh', 'um', 'uhhuh', 'hmm', 'alright', 'fine', 'so', 'well',
      'like', 'actually', 'basically'
    ]);
    const nonFillerWords = words.filter(w => !fillerWords.has(w.toLowerCase().replace(/[^a-z]/g, '')));
    if (!isShortOrRepeat && (nonFillerWords.length < 2 || words.every(w => fillerWords.has(w.toLowerCase().replace(/[^a-z]/g, ''))))) {
      return;
    }

    // Mic energy is the source of truth. Caption text can freeze while you are
    // still talking (Whisper re-sends the same rolling window).
    const micQuietMs = now - (lastMicVoiceAt || now);
    const captionIdleMs = now - lastTranscriptChangeTime;
    if (!isShortOrRepeat && micQuietMs < MIC_QUIET_MS) return;

    let requiredSilence = SILENCE_THRESHOLD_MS;
    if (isShortOrRepeat) {
      requiredSilence = 1400;
    } else if (words.length < MIN_AUTO_SUBMIT_WORDS) {
      requiredSilence = 9000;
    }

    if (captionIdleMs >= requiredSilence && (isShortOrRepeat || micQuietMs >= MIC_QUIET_MS)) {
      console.log(`[Silence Trigger] Candidate finished speaking (mic quiet ${micQuietMs}ms, captions idle ${captionIdleMs}ms, ${words.length} words)`);
      triggerAutoFollowUp();
    }
  }, 350);
}

function stopSilencePolling() {
  if (silencePollInterval) { clearInterval(silencePollInterval); silencePollInterval = null; }
}

function clearSilenceTimer() {
  stopSilencePolling();
}

let currentAudioPlayer = null;

function cleanAiText(text) {
  if (!text) return '';
  let str = text.trim();
  if (str.startsWith('{') && str.includes('"question"')) {
    try {
      const parsed = JSON.parse(str);
      if (parsed && parsed.question) return String(parsed.question).trim();
    } catch (_) {
      const match = str.match(/"question"\s*:\s*"([^"]+)"/);
      if (match) return match[1].trim();
    }
  }
  str = str.replace(/^```(?:json)?/i, '').replace(/```$/i, '');
  str = str.replace(/^{\s*"question"\s*:\s*"?/i, '');
  str = str.replace(/"\s*,\s*"action"\s*:.*$/is, '');
  str = str.replace(/"\s*}$/, '');
  str = str.replace(/^["'\s]+|["'\s]+$/g, '');
  return str.trim();
}

async function triggerAutoFollowUp(overrideAnswer = null) {
  let candidateAnswer = overrideAnswer ? overrideAnswer.trim() : currentTranscript.trim();
  if (interviewEnded || followUpInProgress || isFinalizingUtterance || isAiSpeaking || !candidateAnswer) return;

  const fillerSet = new Set([
    'yeah', 'yes', 'yep', 'yup', 'ok', 'okay', 'sure', 'right',
    'uh', 'um', 'uhhuh', 'hmm', 'alright', 'fine', 'so', 'well',
    'like', 'actually', 'basically'
  ]);
  const isExplicitCmd = isVoiceCommand(candidateAnswer);

  if (!overrideAnswer) {
    const wList = candidateAnswer.split(/\s+/).filter(Boolean);
    const nonFillers = wList.filter(w => !fillerSet.has(w.toLowerCase().replace(/[^a-z]/g, '')));
    if (!isExplicitCmd && (nonFillers.length < 2 || wList.every(w => fillerSet.has(w.toLowerCase().replace(/[^a-z]/g, ''))))) {
      return;
    }

    isFinalizingUtterance = true;
    try {
      if (!isExplicitCmd && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'flush' }));
        await new Promise((resolve) => setTimeout(resolve, 1100));
        candidateAnswer = currentTranscript.trim() || candidateAnswer;
      }
    } finally {
      isFinalizingUtterance = false;
    }
  }

  followUpInProgress = true;
  isAiSpeaking = true;
  questionStreamOpen = true;
  didSpeakChunk = false;
  setTurnPhase('processing');
  clearSilenceTimer();
  if (generateQuestionBtn) generateQuestionBtn.disabled = true;

  sendSttReset();

  if (!overrideAnswer) {
    currentTranscript = '';
  }
  lastTranscriptChangeTime = Date.now();

  emptyTranscript.style.display = 'none';
  liveText.innerText = candidateAnswer;

  setStatus('thinking', 'AI Generating Question...');
  emptyAiResponse.style.display = 'none';
  aiBubble.style.display = 'flex';
  aiText.innerText = '';
  typingCursor.style.display = 'inline-block';

  let fullAiReply = '';

  try {
    const roleVal = roleSelect ? roleSelect.value : 'auto';
    const sessionId = await ensureInterviewSession();
    const response = await fetch('/api/interview-turn-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript: candidateAnswer, role: roleVal, session_id: sessionId })
    });

    if (!response.ok) throw new Error(`Streaming endpoint error ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        const eventMatch = line.match(/^event:\s*(.+)$/m);
        const dataMatch = line.match(/^data:\s*(.+)$/m);
        if (!eventMatch || !dataMatch) continue;

        const event = eventMatch[1].trim();
        const data = JSON.parse(dataMatch[1].trim());

        if (event === 'corrected_transcript' && data.text) {
          candidateAnswer = data.text;
          liveText.innerText = candidateAnswer;
        } else if (event === 'sentence_chunk' && data.text) {
          const chunk = cleanAiText(data.text);
          if (chunk && !fullAiReply.includes(chunk)) {
            fullAiReply = fullAiReply ? `${fullAiReply} ${chunk}` : chunk;
            aiText.innerText = fullAiReply;
            typingCursor.style.display = 'inline-block';
            if (voiceSynthesisToggle.checked) enqueueSpeak(chunk);
          }
        } else if (event === 'done') {
          if (data.session_id) activeInterviewSessionId = data.session_id;
          if (data.full_question) {
            fullAiReply = cleanAiText(data.full_question);
            aiText.innerText = fullAiReply;
          }
          if (typeof data.time_remaining_seconds === 'number') {
            interviewDurationSeconds = data.time_remaining_seconds + Math.floor((Date.now() - startTime) / 1000);
          }
          if (data.interview_complete || data.stage === 'completed') {
            finishInterviewOnClient('complete');
          }
        }
      }
    }

    typingCursor.style.display = 'none';
    addHistoryItem(candidateAnswer, fullAiReply);
    questionStreamOpen = false;

    const ttsPending = ttsBusy || currentAudioPlayer || speakQueue.length;
    if (voiceSynthesisToggle.checked && fullAiReply) {
      if (!ttsPending) {
        await speakText(fullAiReply);
      }
    } else if (!ttsPending) {
      onAiFinishedSpeaking(true);
    }

  } catch (err) {
    questionStreamOpen = false;
    console.warn('[Streaming Turn Fallback] Trying standard turn:', err);
    try {
      const roleVal = roleSelect ? roleSelect.value : 'auto';
      const turnRes = await fetch('/api/interview-turn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: candidateAnswer, role: roleVal, session_id: activeInterviewSessionId }),
      });
      if (turnRes.ok) {
        const turnData = await turnRes.json();
        const rawQ = turnData.llm_response?.question || turnData.llm_response || 'Could you walk me through your technical projects?';
        fullAiReply = cleanAiText(typeof rawQ === 'object' ? JSON.stringify(rawQ) : String(rawQ));
        aiText.innerText = fullAiReply;
        if (voiceSynthesisToggle.checked && fullAiReply) await speakText(fullAiReply);
        else onAiFinishedSpeaking(true);
        if (turnData.interview_complete || turnData.interview_state?.stage === 'completed') {
          finishInterviewOnClient('complete');
        }
      } else {
        onAiFinishedSpeaking(true);
      }
    } catch (_) {
      onAiFinishedSpeaking(true);
    }
  }
}

if (generateQuestionBtn) {
  generateQuestionBtn.addEventListener('click', () => triggerAutoFollowUp());
}

// Keyboard Answer Mode Handlers
const textAnswerInput = document.getElementById('textAnswerInput');
const sendTextAnswerBtn = document.getElementById('sendTextAnswerBtn');

function handleTypedAnswerSubmit() {
  if (!textAnswerInput) return;
  const typedVal = textAnswerInput.value.trim();
  if (!typedVal) return;
  textAnswerInput.value = '';
  triggerAutoFollowUp(typedVal);
}

if (sendTextAnswerBtn) {
  sendTextAnswerBtn.addEventListener('click', handleTypedAnswerSubmit);
}

if (textAnswerInput) {
  textAnswerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleTypedAnswerSubmit();
    }
  });
}


function onAiFinishedSpeaking(force = false) {
  if (!force && (questionStreamOpen || speakQueue.length || (currentAudioPlayer && !currentAudioPlayer.ended && !currentAudioPlayer.paused))) {
    return;
  }
  if (force) {
    speakQueue = [];
    ttsBusy = false;
    ttsGeneration += 1;
    questionStreamOpen = false;
    if (currentAudioPlayer) {
      try { currentAudioPlayer.pause(); } catch (_) { }
    }
  }
  setTurnPhase('cooldown');
  currentAudioPlayer = null;
  resampleBuffer = new Float32Array(0);
  resetPreroll();
  sendSttReset();
  currentTranscript = '';
  candidateHasSpokenInTurn = false;
  isEchoCooldown = true;
  followUpInProgress = true;
  isAiSpeaking = false;
  ttsBusy = false;
  setStatus('speaking', 'Question finished — listening starts in a moment');

  setTimeout(() => {
    if (questionStreamOpen || speakQueue.length || (currentAudioPlayer && !currentAudioPlayer.paused)) {
      return;
    }
    resetPreroll();
    resampleBuffer = new Float32Array(0);
    isAiSpeaking = false;
    followUpInProgress = false;
    isEchoCooldown = false;
    candidateHasSpokenInTurn = false;
    shouldClearTranscriptOnNextSpeech = true;
    currentTranscript = '';
    liveText.innerText = '';
    emptyTranscript.style.display = 'flex';
    wordCount.innerText = '0 words';
    lastTranscriptChangeTime = Date.now();
    lastMicVoiceAt = Date.now();
    if (isRecording) {
      if (generateQuestionBtn) generateQuestionBtn.disabled = false;
      setStatus('listening', 'Your turn — take your time, then pause or tap I’m done');
      setTurnPhase('listening');
      startSilencePolling();
    } else {
      setStatus('', 'Ready');
    }
  }, LISTENING_WINDOW_MS);
}

let aiSpeakingStartTime = 0;

async function speakText(rawText) {
  const text = cleanAiText(rawText);
  const generation = ttsGeneration;
  if (!text) {
    if (!speakQueue.length && !questionStreamOpen) onAiFinishedSpeaking(true);
    return;
  }

  stopAiPlayback();

  setTurnPhase('ai_speaking');
  didSpeakChunk = true;
  ttsBusy = true;
  aiSpeakingStartTime = Date.now();
  resetPreroll();
  setStatus('speaking', 'Kokoro synthesizing…');

  let playbackFailsafe = null;
  const clearAllFailsafes = () => {
    if (playbackFailsafe) { clearTimeout(playbackFailsafe); playbackFailsafe = null; }
  };

  try {
    let blob = null;
    let lastErr = '';
    for (let attempt = 0; attempt < 3; attempt++) {
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.trim(),
          voice: currentVoice,
          speed: 1.0
        }),
      });
      if (response.ok) {
        const next = await response.blob();
        if (next && next.size > 64) {
          blob = next;
          break;
        }
        lastErr = 'empty Kokoro audio';
      } else {
        lastErr = `Kokoro HTTP ${response.status}`;
      }
      await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
    }
    if (!blob) throw new Error(lastErr || 'Kokoro TTS unavailable');

    if (generation !== ttsGeneration) {
      return;
    }

    const audioUrl = URL.createObjectURL(blob);
    const audio = new Audio(audioUrl);
    currentAudioPlayer = audio;
    currentAudioUrl = audioUrl;

    audio.onloadedmetadata = () => {
      const expectedMs = Math.max((audio.duration || 20) * 1000 + 6000, 25000);
      playbackFailsafe = setTimeout(() => {
        if (currentAudioPlayer && !currentAudioPlayer.paused && !currentAudioPlayer.ended) {
          return;
        }
        if (isAiSpeaking || followUpInProgress) {
          console.warn('[Kokoro] Playback timeout. Opening listening window.');
          onAiFinishedSpeaking(true);
        }
      }, expectedMs);
    };

    audio.onplay = () => setStatus('speaking', 'Kokoro speaking…');
    audio.onended = () => {
      clearAllFailsafes();
      URL.revokeObjectURL(audioUrl);
      currentAudioUrl = null;
      currentAudioPlayer = null;
      if (speakQueue.length) {
        playQueuedSpeak();
      } else if (questionStreamOpen) {
        ttsBusy = false;
        isAiSpeaking = true;
        followUpInProgress = true;
        setStatus('speaking', 'Finishing the question…');
      } else {
        ttsBusy = false;
        onAiFinishedSpeaking(true);
      }
    };
    audio.onerror = () => {
      clearAllFailsafes();
      currentAudioPlayer = null;
      ttsBusy = false;
      setStatus('error', 'Kokoro playback failed — question is on screen');
      onAiFinishedSpeaking(true);
    };

    const playPromise = audio.play();
    if (playPromise !== undefined) {
      playPromise.catch((err) => {
        clearAllFailsafes();
        console.warn('Kokoro play blocked:', err);
        ttsBusy = false;
        setStatus('error', 'Allow audio playback — Kokoro only, no Chrome voice');
        onAiFinishedSpeaking(true);
      });
    }
  } catch (err) {
    clearAllFailsafes();
    ttsBusy = false;
    setStatus('error', 'Kokoro TTS unavailable — question is on screen');
    onAiFinishedSpeaking(true);
  }
}

replayVoiceBtn.addEventListener('click', () => {
  const text = aiText.innerText;
  if (text) speakText(text);
});


// ─── Clear ──────────────────────────────────────────────────────────────

clearBtn.addEventListener('click', () => {
  if (isRecording) stopRecording();
  clearSilenceTimer();
  followUpInProgress = false;
  currentTranscript = '';
  liveText.innerText = '';
  emptyTranscript.style.display = 'flex';
  wordCount.innerText = '0 words';
  speechTimer.innerText = '00:00';
  finalTranscriptBox.style.display = 'none';
  if (generateQuestionBtn) generateQuestionBtn.disabled = true;
});


// ─── History ──────────────────────────────────────────────────────────────────

function addHistoryItem(candidateText, aiReply, llmResp = null) {
  noHistoryText.style.display = 'none';
  const item = document.createElement('div');
  item.className = 'history-item';

  let evalBadge = '';
  if (llmResp && llmResp.evaluation) {
    const evalData = llmResp.evaluation;
    const action = llmResp.action || 'ASK_FOLLOWUP';
    const diff = llmResp.next_difficulty || 3;
    const topic = llmResp.topic || '';
    evalBadge = `
      <div class="eval-badge-bar" style="margin-top: 8px; padding: 6px 10px; background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 6px; font-size: 0.78rem; display: flex; gap: 10px; flex-wrap: wrap; color: var(--text-muted);">
        <span>🎯 Action: <strong style="color: #38bdf8;">${action}</strong></span>
        <span>⭐ Overall: <strong style="color: #34d399;">${evalData.overall || 0}/10</strong></span>
        <span>📊 Correctness: ${evalData.correctness || 0}/10</span>
        <span>💡 Depth: ${evalData.depth || 0}/10</span>
        <span>⚡ Diff: ${diff}/5</span>
        ${topic ? `<span>📌 Topic: <em>${topic}</em></span>` : ''}
      </div>
    `;
  }

  item.innerHTML = `
    <div class="turn-label">Candidate Answer:</div>
    <div style="color: var(--text-muted); margin-bottom: 6px;">"${candidateText}"</div>
    <div class="turn-label" style="color: var(--accent-purple);">AI Follow-up:</div>
    <div>${aiReply}</div>
    ${evalBadge}
  `;
  historyList.prepend(item);
}


// ─── Settings Modal & Resume Upload ───────────────────────────────────────────

const quickUploadBtn = document.getElementById('quickUploadBtn');
const resumeDropzone = document.getElementById('resumeDropzone');
const resumeFileInput = document.getElementById('resumeFileInput');
const dropzoneContent = document.getElementById('dropzoneContent');
const uploadStatus = document.getElementById('uploadStatus');
const uploadStatusText = document.getElementById('uploadStatusText');
const extractedFeaturesCard = document.getElementById('extractedFeaturesCard');
const featName = document.getElementById('featName');
const featRole = document.getElementById('featRole');
const featCollege = document.getElementById('featCollege');
const featDegree = document.getElementById('featDegree');
const featCompany = document.getElementById('featCompany');
const featProjects = document.getElementById('featProjects');
const featSkills = document.getElementById('featSkills');

openSettingsBtn.addEventListener('click', () => {
  promptTextarea.value = currentSystemPrompt;
  settingsModal.classList.add('active');
});

if (quickUploadBtn) {
  quickUploadBtn.addEventListener('click', () => {
    promptTextarea.value = currentSystemPrompt;
    settingsModal.classList.add('active');
  });
}

closeSettingsBtn.addEventListener('click', () => settingsModal.classList.remove('active'));
cancelSettingsBtn.addEventListener('click', () => settingsModal.classList.remove('active'));

saveSettingsBtn.addEventListener('click', async () => {
  currentSystemPrompt = promptTextarea.value.trim();
  try {
    await fetch('/api/system-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: currentSystemPrompt }),
    });
  } catch (err) {
    console.error('Failed to save prompt:', err);
  }
  settingsModal.classList.remove('active');
});


// ─── Resume Drag & Drop / File Select Handler ─────────────────────────────────

if (resumeDropzone && resumeFileInput) {
  resumeDropzone.addEventListener('click', () => resumeFileInput.click());

  resumeDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    resumeDropzone.classList.add('dragover');
  });

  resumeDropzone.addEventListener('dragleave', () => {
    resumeDropzone.classList.remove('dragover');
  });

  resumeDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    resumeDropzone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      uploadResumeFile(files[0]);
    }
  });

  resumeFileInput.addEventListener('change', () => {
    if (resumeFileInput.files && resumeFileInput.files.length > 0) {
      uploadResumeFile(resumeFileInput.files[0]);
    }
  });
}

async function uploadResumeFile(file) {
  if (!file) return;

  dropzoneContent.style.display = 'none';
  uploadStatus.style.display = 'flex';
  uploadStatusText.innerText = `Extracting features from "${file.name}"...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/api/upload-resume', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (data.status === 'ok') {
      currentSystemPrompt = data.prompt || '';
      promptTextarea.value = currentSystemPrompt;

      // Bind this browser to the uploaded resume so concurrent interviews don't share CVs
      if (data.resume_token) {
        persistResumeToken(data.resume_token);
        agendaConfirmed = false;
        activeInterviewSessionId = null;
      }
      const feat = data.extracted_features || {};
      featName.innerText = feat.name || 'Candidate';
      featRole.innerText = feat.role || feat.domains?.[0] || 'Technical candidate';
      featCollege.innerText = feat.college || '—';
      featDegree.innerText = feat.degree || '—';
      if (featCompany) featCompany.innerText = feat.company || '—';

      const fillTags = (el, values, emptyLabel) => {
        if (!el) return;
        el.innerHTML = '';
        const arr = Array.isArray(values) ? values.filter(Boolean) : [values].filter(Boolean);
        if (!arr.length) {
          const tag = document.createElement('span');
          tag.className = 'skill-tag';
          tag.innerText = emptyLabel;
          el.appendChild(tag);
          return;
        }
        arr.forEach((item) => {
          const tag = document.createElement('span');
          tag.className = 'skill-tag';
          tag.innerText = item;
          el.appendChild(tag);
        });
      };
      fillTags(featProjects, feat.projects, 'No projects parsed — check the PDF text');
      fillTags(featSkills, feat.skills, 'No skills parsed — check the PDF text');

      extractedFeaturesCard.style.display = 'flex';
      setStatus('speaking', 'Resume Features Extracted!');

      await loadCandidateProfile();
      await showAgendaConfirmation();

      setTimeout(() => setStatus('', 'Ready'), 2500);

    } else {
      alert('Failed to process resume: ' + (data.message || 'Unknown error'));
    }
  } catch (err) {
    console.error('Resume upload error:', err);
    alert('Error uploading resume file: ' + err.message);
  } finally {
    uploadStatus.style.display = 'none';
    dropzoneContent.style.display = 'block';
  }
}


// ─── Operating Mode Switcher ───────────────────────────────────────────────

if (modeInterviewBtn && modeGeneralBtn) {
  modeInterviewBtn.addEventListener('click', () => {
    if (currentMode === 'interview') return;
    currentMode = 'interview';
    modeInterviewBtn.classList.add('active');
    modeGeneralBtn.classList.remove('active');
    console.log('[Mode Switched] Active mode: Interview Mode (Resume-Aware Phonetic Cleaning)');
    if (activeResumeToken && !agendaConfirmed) {
      showAgendaConfirmation().catch((err) => console.error('[Interview] Agenda preview failed:', err));
    }
  });

  modeGeneralBtn.addEventListener('click', () => {
    if (currentMode === 'general') return;
    currentMode = 'general';
    modeGeneralBtn.classList.add('active');
    modeInterviewBtn.classList.remove('active');
    console.log('[Mode Switched] Active mode: General STT Mode (Pure Raw Audio)');
  });
}

if (roleSelect) {
  roleSelect.addEventListener('change', () => {
    if (currentMode !== 'interview' || isRecording || followUpInProgress) return;
    activeInterviewSessionId = null;
    agendaConfirmed = false;
    openingGreeting = '';
  });
}

// ─── Dashboard Candidate CV Profile Loader ──────────────────────────────────
const dashName = document.getElementById('dashName');
const dashSubtitle = document.getElementById('dashSubtitle');
const dashCollege = document.getElementById('dashCollege');
const dashDegree = document.getElementById('dashDegree');
const dashRole = document.getElementById('dashRole');
const dashCompany = document.getElementById('dashCompany');
const dashProjects = document.getElementById('dashProjects');
const dashSkills = document.getElementById('dashSkills');
const dashUploadBtn = document.getElementById('dashUploadBtn');

if (dashUploadBtn) {
  dashUploadBtn.addEventListener('click', () => {
    promptTextarea.value = currentSystemPrompt;
    settingsModal.classList.add('active');
  });
}

async function loadCandidateProfile() {
  try {
    const qs = activeResumeToken ? `?resume_token=${encodeURIComponent(activeResumeToken)}` : '';
    const res = await fetch(`/api/candidate-profile${qs}`);
    if (!res.ok) return;
    const data = await res.json();
    const feat = data.features || {};
    const hotwords = data.hotwords || [];

    const name = feat.name || 'General Candidate';
    const college = feat.college || 'Not specified';
    const degree = feat.degree || 'Technical Degree';
    const role = feat.role || 'Software Engineering Track';
    const company = feat.company || (Array.isArray(feat.companies) ? feat.companies.join(', ') : 'Not specified');

    if (dashName) dashName.innerText = `Candidate: ${name}`;
    if (dashSubtitle) dashSubtitle.innerText = `${degree} • ${college}`;
    if (dashCollege) dashCollege.innerText = college;
    if (dashDegree) dashDegree.innerText = degree;
    if (dashRole) dashRole.innerText = role;
    if (dashCompany) dashCompany.innerText = company;

    const projectList = Array.isArray(feat.projects) ? feat.projects.filter(Boolean) : [];
    if (dashProjects) {
      dashProjects.innerHTML = '';
      if (projectList.length) {
        projectList.slice(0, 8).forEach((title) => {
          const tag = document.createElement('span');
          tag.className = 'skill-tag';
          tag.innerText = title;
          dashProjects.appendChild(tag);
        });
      } else {
        dashProjects.innerHTML = '<span class="skill-tag" style="opacity:0.6;">No projects extracted yet — upload CV</span>';
      }
    }

    // Combine skills, domains, and hotwords
    let allSkills = [];
    ['skills', 'domains', 'certifications'].forEach(k => {
      const v = feat[k];
      if (Array.isArray(v)) allSkills.push(...v);
      else if (typeof v === 'string' && v.trim()) allSkills.push(v);
    });

    if (allSkills.length === 0 && hotwords.length > 0) {
      allSkills = hotwords;
    }

    // Deduplicate
    const uniqueSkills = [...new Set(allSkills)].slice(0, 16);

    if (dashSkills) {
      dashSkills.innerHTML = '';
      if (uniqueSkills.length > 0) {
        uniqueSkills.forEach(s => {
          const tag = document.createElement('span');
          tag.className = 'skill-tag';
          tag.innerText = s;
          dashSkills.appendChild(tag);
        });
      } else {
        dashSkills.innerHTML = '<span class="skill-tag" style="opacity:0.6;">No terms extracted yet — upload CV</span>';
      }
    }

    // Also populate settings modal features if present
    if (featName) featName.innerText = name;
    if (featRole) featRole.innerText = role;
    if (featCollege) featCollege.innerText = college;
    if (featDegree) featDegree.innerText = degree;
    if (featCompany) featCompany.innerText = company;
    if (featProjects) {
      featProjects.innerHTML = '';
      if (projectList.length) {
        projectList.forEach((title) => {
          const tag = document.createElement('span');
          tag.className = 'skill-tag';
          tag.innerText = title;
          featProjects.appendChild(tag);
        });
      }
    }
    if (featSkills && uniqueSkills.length > 0) {
      featSkills.innerHTML = '';
      uniqueSkills.forEach(s => {
        const tag = document.createElement('span');
        tag.className = 'skill-tag';
        tag.innerText = s;
        featSkills.appendChild(tag);
      });
      if (extractedFeaturesCard) extractedFeaturesCard.style.display = 'flex';
    }

  } catch (err) {
    console.warn('[Candidate Profile] Failed to load:', err);
  }
}

// Load Candidate Profile on startup
document.addEventListener('DOMContentLoaded', () => {
  loadCandidateProfile();
  wireScorecardAndHistoryUi();
});
loadCandidateProfile();

let lastScorecardReport = null;
let lastScorecardSessionId = null;

const REC_LABELS = {
  strong_hire: 'Strong Hire',
  hire: 'Hire',
  borderline: 'Borderline',
  no_hire: 'No Hire',
  insufficient_data: 'Insufficient data',
};

function scoreToHundred(score, outOf) {
  const max = Number(outOf) || 5;
  if (!max) return 0;
  return Math.round((Number(score) || 0) / max * 100);
}

function collectEvidence(report) {
  const seen = new Set();
  const out = [];
  const push = (item) => {
    const quote = (item && item.quote) ? String(item.quote).trim() : '';
    if (!quote || seen.has(quote)) return;
    seen.add(quote);
    out.push({ quote, why: item.why || '' });
  };
  (report.competencies || []).forEach((c) => (c.evidence || []).forEach(push));
  (report.turn_grades || []).forEach((t) => (t.evidence || []).forEach(push));
  return out.slice(0, 8);
}

function renderScorecardHtml(report, statusNote) {
  if (!report || report.status === 'running') {
    return `<p>${statusNote || 'Grading your answers…'}</p>`;
  }
  const hundred = scoreToHundred(report.overall_score, report.overall_out_of || 5);
  const verdict = REC_LABELS[report.recommendation] || report.recommendation || '—';
  const graded = report.graded_by === 'llm' ? 'LLM grade' : report.graded_by === 'mixed' ? 'Mixed (LLM + heuristic)' : report.graded_by === 'heuristic' ? 'Heuristic grade' : (statusNote || '');
  const comps = (report.competencies || []).map((c) => {
    const pct = Math.min(100, Math.round((Number(c.score) || 0) / 5 * 100));
    return `<div class="comp-row"><header><span>${c.competency}</span><span>${c.score}/5</span></header><div class="comp-bar"><span style="width:${pct}%"></span></div></div>`;
  }).join('');
  const strengths = (report.strengths || []).map((s) => `<li>${s}</li>`).join('') || '<li>None listed</li>';
  const gaps = (report.weaknesses || []).map((s) => `<li>${s}</li>`).join('') || '<li>None listed</li>';
  const quotes = collectEvidence(report).map((e) => `<blockquote>“${e.quote}”${e.why ? `<footer>${e.why}</footer>` : ''}</blockquote>`).join('') || '<p>No verbatim quotes stored.</p>';
  return `
    <div class="scorecard-hero"><div class="scorecard-score">${hundred}<small>/100</small></div><div class="scorecard-verdict">${verdict}</div></div>
    <p class="scorecard-sub">${graded}</p>
    <h3>Competencies</h3>
    ${comps || '<p>No competency scores yet.</p>'}
    <h3>Strengths</h3><ul>${strengths}</ul>
    <h3>Identified gaps</h3><ul>${gaps}</ul>
    <h3>Evidence quotes</h3><div class="scorecard-quotes">${quotes}</div>
  `;
}

function showScorecardModal() {
  const modal = document.getElementById('scorecardModal');
  if (modal) modal.hidden = false;
}

function hideScorecardModal() {
  const modal = document.getElementById('scorecardModal');
  if (modal) modal.hidden = true;
}

async function openScorecardModal(sessionId) {
  lastScorecardSessionId = sessionId;
  lastScorecardReport = null;
  const title = document.getElementById('scorecardTitle');
  const sub = document.getElementById('scorecardSub');
  const body = document.getElementById('scorecardBody');
  if (title) title.innerText = 'Interview complete';
  if (sub) sub.innerText = 'Grading your answers…';
  if (body) body.innerHTML = renderScorecardHtml({ status: 'running' }, 'Grading your answers…');
  showScorecardModal();
  for (let i = 0; i < 17; i++) {
    try {
      const res = await fetch(`/api/interview-report/${encodeURIComponent(sessionId)}`);
      if (res.ok) {
        const data = await res.json();
        const report = data.report || {};
        if (report.status === 'done' || (report.overall_score != null && report.status !== 'running')) {
          lastScorecardReport = report;
          if (sub) sub.innerText = report.candidate_name || report.target_role || '';
          if (body) body.innerHTML = renderScorecardHtml(report);
          return;
        }
        if (body) body.innerHTML = renderScorecardHtml({ status: 'running' }, 'Grading still running…');
      }
    } catch (err) {
      console.warn('[Scorecard] poll failed', err);
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
  if (body) body.innerHTML = '<p>Grading timed out. Open Past interviews in a moment, or download JSON after refresh.</p>';
}

function downloadReportJson() {
  if (!lastScorecardReport) return;
  const blob = new Blob([JSON.stringify(lastScorecardReport, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `interview-report-${lastScorecardSessionId || 'session'}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function openHistoryDrawer() {
  const drawer = document.getElementById('historyDrawer');
  const backdrop = document.getElementById('historyDrawerBackdrop');
  if (drawer) drawer.hidden = false;
  if (backdrop) backdrop.hidden = false;
  loadPastInterviews();
}

function closeHistoryDrawer() {
  const drawer = document.getElementById('historyDrawer');
  const backdrop = document.getElementById('historyDrawerBackdrop');
  if (drawer) drawer.hidden = true;
  if (backdrop) backdrop.hidden = true;
}

function formatUnixLocal(ts) {
  const n = Number(ts);
  if (!n) return '';
  return new Date(n < 1e12 ? n * 1000 : n).toLocaleString();
}

function formatDurationClock(seconds) {
  if (seconds == null || seconds === '') return '';
  const total = Math.max(0, Math.floor(Number(seconds)));
  if (Number.isNaN(total)) return '';
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return secs ? `${minutes}m ${secs}s` : `${minutes}m`;
  return `${secs}s`;
}

function formatInterviewTiming(row, detailed) {
  const start = formatUnixLocal(row && (row.started_at || row.created_at));
  const end = formatUnixLocal(row && row.ended_at);
  const dur = formatDurationClock(row && row.duration_seconds);
  const parts = [];
  if (start) parts.push(detailed ? `Started ${start}` : start);
  if (end) parts.push(detailed ? `Ended ${end}` : `ended ${end}`);
  if (dur) parts.push(dur);
  return parts.join(' · ') || 'Time not recorded';
}

async function loadPastInterviews() {
  const list = document.getElementById('historyInterviewList');
  const role = (document.getElementById('historyRoleFilter') || {}).value || '';
  const rec = (document.getElementById('historyRecFilter') || {}).value || '';
  if (list) list.innerHTML = '<p>Loading…</p>';
  const params = new URLSearchParams({ limit: '50' });
  if (role) params.set('role', role);
  if (rec) params.set('recommendation', rec);
  try {
    const res = await fetch(`/api/interviews?${params}`);
    const data = await res.json();
    const rows = data.interviews || [];
    if (!rows.length) {
      if (list) list.innerHTML = '<p>No completed interviews yet.</p>';
      return;
    }
    if (list) {
      list.innerHTML = rows.map((row) => {
        const score = row.overall_score != null ? scoreToHundred(row.overall_score, 5) : '—';
        const recLabel = REC_LABELS[row.recommendation] || row.recommendation || 'ungraded';
        const clock = formatInterviewTiming(row);
        return `<button type="button" class="history-row" data-session="${row.session_id}">
          <strong>${row.candidate_name || 'Candidate'}</strong>
          <div>${row.target_role || ''} · ${score}/100 · ${recLabel}</div>
          <div>${clock} · ${row.turn_count || 0} turns</div>
        </button>`;
      }).join('');
      list.querySelectorAll('.history-row').forEach((btn) => {
        btn.addEventListener('click', () => showPastInterview(btn.getAttribute('data-session')));
      });
    }
  } catch (err) {
    if (list) list.innerHTML = '<p>Could not load interviews.</p>';
  }
}

async function showPastInterview(sessionId) {
  const detail = document.getElementById('historyInterviewDetail');
  if (detail) detail.innerHTML = '<p>Loading transcript…</p>';
  try {
    const [histRes, repRes] = await Promise.all([
      fetch(`/api/interview-history/${encodeURIComponent(sessionId)}`),
      fetch(`/api/interview-report/${encodeURIComponent(sessionId)}`),
    ]);
    const hist = histRes.ok ? await histRes.json() : {};
    const rep = repRes.ok ? await repRes.json() : {};
    const report = rep.report || {};
    lastScorecardReport = report.status === 'done' ? report : lastScorecardReport;
    lastScorecardSessionId = sessionId;
    const turns = (hist.history || []).map((t) => `
      <div class="turn-block">
        <strong>Q:</strong> ${t.asked_question || t.interviewer_question || ''}<br>
        <strong>A:</strong> ${t.candidate_answer || ''}
      </div>`).join('');
    if (detail) {
      detail.innerHTML = `${renderScorecardHtml(report.status === 'done' ? report : { status: 'running' }, report.status === 'running' ? 'Still grading…' : '')}
        <p class="history-timing">${formatInterviewTiming(hist, true)}</p>
        <h3>Transcript</h3>${turns || '<p>No turns stored.</p>'}`;
    }
  } catch (err) {
    if (detail) detail.innerHTML = '<p>Could not load this interview.</p>';
  }
}

function wireScorecardAndHistoryUi() {
  const closeScore = document.getElementById('closeScorecardBtn');
  const dl = document.getElementById('downloadReportJsonBtn');
  const printBtn = document.getElementById('printReportBtn');
  const openHistFromScore = document.getElementById('scorecardOpenHistoryBtn');
  const pastBtn = document.getElementById('pastInterviewsBtn');
  const closeHist = document.getElementById('closeHistoryDrawerBtn');
  const backdrop = document.getElementById('historyDrawerBackdrop');
  const roleF = document.getElementById('historyRoleFilter');
  const recF = document.getElementById('historyRecFilter');
  if (closeScore) closeScore.addEventListener('click', hideScorecardModal);
  if (dl) dl.addEventListener('click', downloadReportJson);
  if (printBtn) printBtn.addEventListener('click', () => window.print());
  if (openHistFromScore) openHistFromScore.addEventListener('click', () => { hideScorecardModal(); openHistoryDrawer(); });
  if (pastBtn) pastBtn.addEventListener('click', openHistoryDrawer);
  if (closeHist) closeHist.addEventListener('click', closeHistoryDrawer);
  if (backdrop) backdrop.addEventListener('click', closeHistoryDrawer);
  if (roleF) roleF.addEventListener('change', loadPastInterviews);
  if (recF) recF.addEventListener('change', loadPastInterviews);
}

/**
 * Headless check of the pre-roll buffer and barge-in detector.
 *
 * Mirrors the logic in app.js (which cannot be imported directly because it
 * touches the DOM on load). Run: node static/test_audio_gate.mjs
 */

const TARGET_SAMPLE_RATE = 16000;
const PREROLL_MAX_SAMPLES = TARGET_SAMPLE_RATE * 2;
const BARGE_IN_FLUSH_SAMPLES = Math.floor(TARGET_SAMPLE_RATE * 0.6);
const BARGE_IN_MIN_FRAMES = 3;
const BARGE_IN_ABS_FLOOR = 0.02;
const BARGE_IN_NOISE_MULT = 3.0;

let prerollBuffer = new Float32Array(0);
let prerollFlushSamples = PREROLL_MAX_SAMPLES;
let noiseFloor = 0.005;
let bargeInFrames = 0;

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

function detectBargeIn(samples) {
  const rms = frameRMS(samples);
  const threshold = Math.max(BARGE_IN_ABS_FLOOR, noiseFloor * BARGE_IN_NOISE_MULT);
  if (rms < threshold) {
    noiseFloor = noiseFloor * 0.95 + rms * 0.05;
    bargeInFrames = 0;
    return false;
  }
  bargeInFrames++;
  return bargeInFrames >= BARGE_IN_MIN_FRAMES;
}

// ── Signal helpers ────────────────────────────────────────────────────────────
const FRAME = 1365; // ~85ms of 16k audio, matching a 4096-sample callback at 48k
const frame = (amp) => Float32Array.from({ length: FRAME }, () => (Math.random() * 2 - 1) * amp);
const silence = () => frame(0.002);
const speech = () => frame(0.25);

function assert(cond, msg) {
  if (!cond) { console.error('FAIL: ' + msg); process.exit(1); }
}

// 1. Audio captured while muted is retained, not deleted.
resetPreroll();
for (let i = 0; i < 5; i++) pushPreroll(speech());
assert(prerollBuffer.length === FRAME * 5, 'pre-roll should retain muted audio');
const flushed = drainPreroll();
assert(flushed.length === FRAME * 5, 'gate opening should release every buffered sample');
assert(prerollBuffer.length === 0, 'pre-roll should empty after flush');
console.log(`[OK] Muted audio is buffered and released: ${flushed.length} samples (~${Math.round(flushed.length / 16)}ms) recovered.`);

// 2. The buffer is bounded so a long AI turn cannot grow memory forever.
resetPreroll();
for (let i = 0; i < 200; i++) pushPreroll(speech()); // ~17s
assert(prerollBuffer.length === PREROLL_MAX_SAMPLES, 'pre-roll must cap at 2s');
console.log(`[OK] Pre-roll capped at ${prerollBuffer.length} samples (2s), no unbounded growth.`);

// 3. Barge-in flush sends only the speech onset, not the AI echo tail.
resetPreroll();
for (let i = 0; i < 30; i++) pushPreroll(silence()); // echo tail
for (let i = 0; i < 8; i++) pushPreroll(speech());   // candidate starts
prerollFlushSamples = BARGE_IN_FLUSH_SAMPLES;
const onset = drainPreroll();
assert(onset.length === BARGE_IN_FLUSH_SAMPLES, 'barge-in flush should be limited to the onset window');
assert(frameRMS(onset) > 0.1, 'the onset window should contain the candidate, not silence');
console.log(`[OK] Barge-in flushes only ${Math.round(onset.length / 16)}ms of onset (rms ${frameRMS(onset).toFixed(3)}).`);

// 4. Room noise and short clicks must not interrupt the AI.
noiseFloor = 0.005; bargeInFrames = 0;
let tripped = false;
for (let i = 0; i < 50; i++) tripped = detectBargeIn(silence()) || tripped;
assert(!tripped, 'silence must never trigger barge-in');
tripped = detectBargeIn(speech()) || detectBargeIn(silence());
assert(!tripped, 'a single loud frame (cough/click) must not trigger barge-in');
console.log('[OK] Silence and one-off clicks do not interrupt the AI.');

// 5. Sustained speech does interrupt.
noiseFloor = 0.005; bargeInFrames = 0;
let firedAt = -1;
for (let i = 0; i < 10; i++) {
  if (detectBargeIn(speech())) { firedAt = i + 1; break; }
}
assert(firedAt === BARGE_IN_MIN_FRAMES, `barge-in should fire after ${BARGE_IN_MIN_FRAMES} frames, fired at ${firedAt}`);
console.log(`[OK] Sustained speech interrupts after ${firedAt} frames (~${Math.round(firedAt * FRAME / 16)}ms).`);

// 6. In a noisy room the floor adapts so normal background stays below threshold.
noiseFloor = 0.005; bargeInFrames = 0;
for (let i = 0; i < 100; i++) detectBargeIn(frame(0.012)); // steady fan/room hum
assert(bargeInFrames === 0, 'steady room noise must not accumulate barge-in frames');
console.log(`[OK] Noise floor adapted to ${noiseFloor.toFixed(4)}; background hum ignored.`);

console.log('\n[SUCCESS] Pre-roll buffering and barge-in verified.');

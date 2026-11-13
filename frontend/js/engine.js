// Voice Cloner engine: the five models running in the browser with ONNX Runtime Web.
// It follows vc_lib/infer.py step for step (text -> FastSpeech -> shallow diffusion -> vocoder), so the
// browser speaks the way the Python pipeline does. Needs the global `ort` (onnxruntime-web).
//
//   const engine = await createEngine("models/", { onProgress: (loaded, total) => ... });
//   const wav = await engine.speak("Hello there.", engine.voices[0].embedding, { speed: 1 });   // Float32Array, 22,050 Hz
//   const voice = await engine.embedClip(samples22k);                                         // 64 numbers

const MODELS = ["tts_encoder", "tts_decoder", "diffusion", "vocoder", "voice_encoder"];
// Every run gets fresh copies of its inputs: in worker mode ONNX Runtime moves the input memory to the worker.
const f32 = (data, dims) => new ort.Tensor("float32", Float32Array.from(data), dims);

export async function createEngine(base, { onProgress } = {}) {
  const cfg = await (await fetch(base + "pipeline.json")).json();
  const bytes = await fetchAll(MODELS.map((m) => `${base}${m}.onnx`), onProgress);
  const sessions = {};
  for (let i = 0; i < MODELS.length; i++) {
    sessions[MODELS[i]] = await ort.InferenceSession.create(bytes[i], { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }
  return new Engine(cfg, sessions);
}

// Download every model with one combined progress count (for the intro counter).
async function fetchAll(urls, onProgress) {
  const sizes = await Promise.all(urls.map(async (u) => +(await fetch(u, { method: "HEAD" })).headers.get("content-length") || 0));
  const total = sizes.reduce((a, b) => a + b, 0);
  let loaded = 0;
  return Promise.all(urls.map(async (u) => {
    const res = await fetch(u);
    if (!res.ok) throw new Error(`Couldn't download ${u} (${res.status})`);
    const reader = res.body.getReader(), parts = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      loaded += value.length;
      onProgress && onProgress(loaded, total);
    }
    const out = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }));
}

class Engine {
  constructor(cfg, s) {
    this.cfg = cfg;
    this.s = s;
    this.sr = cfg.sample_rate;
    this.voices = cfg.voices;
    this.melBasis = cfg.mel_basis.map((r) => Float32Array.from(r));
  }

  // ---------------------------------------------------------------- speaking
  // Returns { audio: Float32Array at 22,050 Hz, mel: { data, T } (the phrases' spectrograms side by side) },
  // or null when the text has no words to speak.
  async speak(text, embedding, { speed = 1, seed = 0, onProgress } = {}) {
    const spk = Float32Array.from(embedding);
    const pace = 1 / Math.min(Math.max(speed, 0.75), 1.33);
    const rand = gaussian(seed);
    const pieces = [], mels = [], chunks = splitText(text, Math.min(this.cfg.max_chars, 240));
    const hop = this.cfg.hop;
    for (const [i, [chunk, pause]] of chunks.entries()) {
      onProgress && onProgress(i, chunks.length);
      const segs = await this.plan(chunk, spk, pace);
      for (const [k, seg] of segs.entries()) {
        const mel = await this.decodeChunk(seg.h, seg.dur, spk, rand, seg.fs, seg.fe);
        mels.push(mel);
        let wav = await this.vocode(mel);
        // a joined side continues straight into its neighbour: cut exactly at the frame, keep no padding
        if (seg.jl) wav = wav.subarray(2 * hop);
        if (seg.jr) wav = wav.subarray(0, wav.length - 12 * hop);
        const t = trimSilence(wav, this.sr, 35, 60, !seg.jl, !seg.jr);
        pieces.push(t.length > this.sr / 20 ? t : wav);
        if (pause && k === segs.length - 1) pieces.push(new Float32Array(Math.round(this.sr * 0.15)));
      }
    }
    if (!mels.length) return null;
    const T = mels.reduce((a, m) => a + m.T, 0), data = new Float32Array(80 * T);
    let off = 0;
    for (const m of mels) { for (let r = 0; r < 80; r++) data.set(m.data.subarray(r * m.T, (r + 1) * m.T), r * T + off); off += m.T; }
    return { audio: normalize(crossfade(pieces, Math.round(this.sr * 0.025))), mel: { data, T } };
  }

  // The models were trained on 128-frame clips; past that the decoder's output decays into a hum. So a chunk
  // that would need more frames is split at word boundaries into parts (vc_lib.infer.split_by_frames). Each
  // part is spoken together with a neighbouring word on each side, and only the part's own frames are kept
  // (cut in the middle of the word gaps), so a join sounds like the middle of a sentence. Mirrors infer._plan.
  // Returns [{ h, dur, fs, fe, jl, jr }]: the window's encoding, the frames to keep, and which sides are joins.
  async plan(chunk, spk, pace) {
    const sum = (a) => a.reduce((x, y) => x + y, 0);
    const limit = this.cfg.max_frames || 128, ctxWords = 1, reserve = Math.min(Math.floor(56 * Math.max(pace, 1)), Math.floor(limit / 2));
    const enc = await this.encodeChunk(chunk, spk, pace);
    const dur = enc.dur.slice(0, chunk.length);
    const parts = sum(dur) > limit ? splitByFrames(chunk, dur, limit - reserve) : [chunk];
    if (parts.length === 1) return [{ h: enc.h, dur: enc.dur, fs: 0, fe: sum(enc.dur), jl: false, jr: false }];
    if (!chunk.includes(" ")) {                                 // letter pieces of one long word: no context to share
      const out = [];
      for (const p of parts) out.push(...(await this.plan(p, spk, pace)));
      return out;
    }
    const words = chunk.split(" "), segs = [];
    let a = 0;
    for (const [i, part] of parts.entries()) {
      const n = part.split(" ").length;
      let cl = Math.min(ctxWords, a), cr = Math.min(ctxWords, words.length - a - n), ws, w;
      for (;;) {
        ws = words.slice(a - cl, a + n + cr);
        w = await this.encodeChunk(ws.join(" "), spk, pace);
        if (sum(w.dur) <= limit || cl + cr === 0) break;
        if (cl >= cr) cl--; else cr--;                       // too long: give up context, largest side first
      }
      if (sum(w.dur) > limit && part.includes(" ")) {           // too long even alone: split this part again
        segs.push(...(await this.plan(part, spk, pace)));
        a += n;
        continue;
      }
      const wdur = w.dur, cs = cl ? ws.slice(0, cl).join(" ").length + 1 : 0, ce = cs + part.length;
      const fs = cl ? sum(wdur.slice(0, cs - 1)) + Math.floor(wdur[cs - 1] / 2) : 0;        // middle of the gap before
      const fe = cr ? sum(wdur.slice(0, ce)) + Math.floor(wdur[ce] / 2) : sum(wdur);          // middle of the gap after
      segs.push({ h: w.h, dur: wdur, fs, fe, jl: cl > 0, jr: cr > 0 });  // a side that lost its context is trimmed like a chunk end
      a += n;
    }
    return segs;
  }

  async encodeChunk(chunk, spk, pace) {
    const ids = [...chunk].map((c) => BigInt(this.cfg.char2id[c] ?? this.cfg.unk_id));
    if (ids.length < 2) ids.push(BigInt(this.cfg.char2id[" "]));
    const enc = await this.s.tts_encoder.run({ ids: new ort.Tensor("int64", BigInt64Array.from(ids), [1, ids.length]), spk: f32(spk, [1, 64]) });
    // durations: frames per character = round(max(e^logd - 1, 0) * pace), at least 1 (FastSpeechTTS.predict_durations)
    const dur = Array.from(enc.logd.data, (v) => Math.max(1, Math.round(Math.max(Math.exp(v) - 1, 0) * pace)));
    return { h: enc.h.data, dur };
  }

  // Decodes a window and keeps frames fs..fe of the result.
  async decodeChunk(h, dur, spk, rand, fs = 0, fe = Infinity) {
    const L = dur.length, D = 256;
    let T = Math.min(dur.reduce((a, b) => a + b, 0), 1024);
    T = Math.max(T, 2);
    const x = new Float32Array(T * D);
    for (let i = 0, f = 0; i < L && f < T; i++) for (let k = 0; k < dur[i] && f < T; k++, f++) x.set(h.subarray(i * D, i * D + D), f * D);
    const coarse = (await this.s.tts_decoder.run({ x: new ort.Tensor("float32", x, [1, T, D]), spk: f32(spk, [1, 64]) })).mel.data;   // (80, T)
    const mel = await this.refine(coarse, T, spk, rand);
    fe = Math.min(fe, T);
    if (fs === 0 && fe === T) return mel;
    const n = fe - fs, data = new Float32Array(80 * n);
    for (let r = 0; r < 80; r++) data.set(mel.data.subarray(r * T + fs, r * T + fe), r * n);
    return { data, T: n };
  }

  // Shallow diffusion (DiffSinger): noise the coarse mel to step strength*T, then a few DDIM steps back.
  async refine(coarse, T, spk, rand) {
    const { alpha_bar: ab, T: steps, strength, steps: ddim } = this.cfg.diffusion;
    const Tp = T + ((4 - (T % 4)) % 4), n = 80 * Tp;
    const cond = new Float32Array(n).fill(-1);                    // padded frames are silence (0 -> -1)
    for (let m = 0; m < 80; m++) for (let t = 0; t < T; t++) cond[m * Tp + t] = coarse[m * T + t] * 2 - 1;
    const tStart = Math.max(1, Math.min(steps - 1, Math.round(strength * (steps - 1))));
    const nSteps = Math.max(2, Math.round(ddim * tStart / (steps - 1)));
    const ts = Array.from({ length: nSteps }, (_, i) => Math.trunc(tStart - (tStart * i) / (nSteps - 1)));
    let x = new Float32Array(n);
    const a0 = ab[tStart];
    for (let i = 0; i < n; i++) x[i] = Math.sqrt(a0) * cond[i] + Math.sqrt(1 - a0) * rand();
    for (let i = 0; i < ts.length; i++) {
      const a = ab[ts[i]], ap = i + 1 < ts.length ? ab[ts[i + 1]] : 1;
      const eps = (await this.s.diffusion.run({ x: f32(x, [1, 1, 80, Tp]), t: new ort.Tensor("int64", BigInt64Array.of(BigInt(ts[i])), [1]), cond: f32(cond, [1, 1, 80, Tp]), spk: f32(spk, [1, 64]) })).eps.data;
      const next = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        const x0 = Math.min(1, Math.max(-1, (x[j] - Math.sqrt(1 - a) * eps[j]) / Math.sqrt(a)));
        const e = (x[j] - Math.sqrt(a) * x0) / Math.sqrt(1 - a);
        next[j] = Math.sqrt(ap) * x0 + Math.sqrt(1 - ap) * e;
      }
      x = next;
    }
    const mel = new Float32Array(80 * T);
    for (let m = 0; m < 80; m++) for (let t = 0; t < T; t++) mel[m * T + t] = Math.min(1, Math.max(0, (x[m * Tp + t] + 1) / 2));
    return { data: mel, T };
  }

  async vocode({ data, T }) {
    const Tw = T + 2 + 12;                                        // 2 silent frames before, 12 after (infer.py)
    const m = new Float32Array(80 * Tw);
    for (let r = 0; r < 80; r++) m.set(data.subarray(r * T, r * T + T), r * Tw + 2);
    return (await this.s.vocoder.run({ mel: new ort.Tensor("float32", m, [1, 1, 80, Tw]) })).wav.data;
  }

  // ---------------------------------------------------------------- a voice from a recording
  // samples: mono Float32Array at 22,050 Hz. Returns { embedding, speechSeconds }.
  async embedClip(samples) {
    let y = trimSilence(samples, this.sr, 30, 0);
    const speechSeconds = speechLength(samples, this.sr, 30);
    y = normalize(y);
    const mel = this.mel(y), T = mel.T, W = this.cfg.seg, wins = [];
    for (let s = 0; s < Math.max(1, T - W + 1); s += W / 2) wins.push(s);
    const data = new Float32Array(wins.length * 80 * W);
    wins.forEach((s, k) => { for (let r = 0; r < 80; r++) for (let t = 0; t < W && s + t < T; t++) data[k * 80 * W + r * W + t] = mel.data[r * T + s + t]; });
    const mu = (await this.s.voice_encoder.run({ windows: new ort.Tensor("float32", data, [wins.length, 1, 80, W]) })).mu.data;
    const emb = new Array(64).fill(0);
    for (let k = 0; k < wins.length; k++) for (let i = 0; i < 64; i++) emb[i] += mu[k * 64 + i] / wins.length;
    return { embedding: emb, speechSeconds };
  }

  // librosa.feature.melspectrogram (centred, zero padding, Hann window, power 2) -> dB -> 0..1 (vc_lib.common.wav_to_mel)
  mel(y) {
    const N = this.cfg.n_fft, H = this.cfg.hop, pad = N / 2;
    const yp = new Float32Array(y.length + 2 * pad); yp.set(y, pad);
    const T = 1 + Math.floor((yp.length - N) / H), bins = N / 2 + 1;
    const win = Float32Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N));
    const [lo, hi] = this.cfg.db_range, out = new Float32Array(80 * T);
    const re = new Float64Array(N), im = new Float64Array(N), power = new Float64Array(bins);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < N; i++) { re[i] = yp[t * H + i] * win[i]; im[i] = 0; }
      fft(re, im);
      for (let k = 0; k < bins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
      for (let m = 0; m < 80; m++) {
        let s = 0; const b = this.melBasis[m];
        for (let k = 0; k < bins; k++) s += b[k] * power[k];
        const db = 10 * Math.log10(Math.max(s, 1e-10));
        out[m * T + t] = Math.min(1, Math.max(0, (db - lo) / (hi - lo)));
      }
    }
    return { data: out, T };
  }
}

// ---------------------------------------------------------------- text (vc_lib.common.clean_text, infer.split_text)
const ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(" ");
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
function under1000(n) {
  const h = Math.floor(n / 100), r = n % 100, parts = [];
  if (h) parts.push(ONES[h] + " hundred");
  if (r) parts.push((h ? "and " : "") + (r < 20 ? ONES[r] : TENS[Math.floor(r / 10)] + (r % 10 ? " " + ONES[r % 10] : "")));
  return parts.join(" ");
}
export function numberToWords(n) {                             // like num2words in English: 1005 -> "one thousand and five"
  if (n === 0) return "zero";
  if (n >= 1e12) return String(n).split("").map((d) => ONES[+d]).join(" ");
  const scales = [[1e9, "billion"], [1e6, "million"], [1e3, "thousand"]], parts = [];
  for (const [v, name] of scales) if (n >= v) { parts.push(under1000(Math.floor(n / v)) + " " + name); n %= v; }
  if (n) parts.push((parts.length && n < 100 ? "and " : "") + under1000(n));
  return parts.join(" ");
}
export function cleanText(s) {
  s = String(s).toLowerCase().replace(/\d+/g, (d) => " " + numberToWords(Number(d)) + " ");
  return s.replace(/[’‘]/g, "'").replace(/[^a-z' ]/g, " ").replace(/\s+/g, " ").trim();
}
// Split a chunk at word boundaries into even parts, each under `limit` predicted frames (vc_lib.infer.split_by_frames).
export function splitByFrames(chunk, durations, limit) {
  const total = durations.reduce((a, b) => a + b, 0);
  if (total <= limit) return [chunk];
  const target = total / Math.ceil(total / limit);
  const parts = [];
  if (!chunk.includes(" ")) {                                 // one very long "word": cut between letters
    let cur = "", curFrames = 0;
    [...chunk].forEach((ch, i) => {
      if (cur && curFrames + durations[i] / 2 > target) { parts.push(cur); cur = ""; curFrames = 0; }
      cur += ch; curFrames += durations[i];
    });
    return [...parts, cur];
  }
  let cur = [], curFrames = 0, pos = 0;
  for (const word of chunk.split(" ")) {
    const frames = durations.slice(pos, pos + word.length + 1).reduce((a, b) => a + b, 0);   // the word plus the space after it
    pos += word.length + 1;
    if (cur.length && curFrames + frames / 2 > target) { parts.push(cur.join(" ")); cur = []; curFrames = 0; }
    cur.push(word); curFrames += frames;
  }
  parts.push(cur.join(" "));
  return parts;
}

export function splitText(text, maxChars) {
  const out = [];
  for (const phrase of text.trim().split(/(?<=[.,!?;:])\s+|\n+/)) {
    // a word longer than maxChars is cut into pieces that fit (the text encoder has 256 positions)
    const words = cleanText(phrase).split(" ").filter(Boolean).flatMap((w) => w.match(new RegExp(`.{1,${maxChars}}`, "g")));
    if (!words.length) continue;
    const chunks = [];
    let cur = "";
    for (const w of words) {
      if (cur && cur.length + 1 + w.length > maxChars) { chunks.push(cur); cur = w; } else cur = cur ? cur + " " + w : w;
    }
    chunks.push(cur);
    const pause = /[.,!?;:]$/.test(phrase.trim());
    chunks.forEach((c, k) => out.push([c, pause && k === chunks.length - 1]));
  }
  return out;
}

// ---------------------------------------------------------------- audio helpers
function gaussian(seed) {                                      // seeded normal random numbers (Box-Muller on mulberry32)
  let a = seed >>> 0, spare = null;
  const u = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    const r = Math.sqrt(-2 * Math.log(u() || 1e-12)), th = 2 * Math.PI * u();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
}

function frameDb(y, frame = 2048, hop = 512) {                 // librosa.feature.rms (centred) in dB relative to the loudest frame
  const pad = frame / 2, n = 1 + Math.floor(y.length / hop), db = new Float32Array(n);
  let mx = 1e-20;
  const ms = new Float64Array(n);
  for (let f = 0; f < n; f++) {
    let s = 0;
    for (let i = 0; i < frame; i++) { const j = f * hop + i - pad; if (j >= 0 && j < y.length) s += y[j] * y[j]; }
    ms[f] = s / frame; mx = Math.max(mx, ms[f]);
  }
  for (let f = 0; f < n; f++) db[f] = 10 * Math.log10(Math.max(ms[f], 1e-10) / mx);
  return db;
}
// librosa.effects.trim, keeping keepMs of margin like vc_lib.infer.trim_silence (0 = plain trim)
export function trimSilence(y, sr, topDb, keepMs = 60, left = true, right = true) {
  const db = frameDb(y);
  let first = -1, last = -1;
  db.forEach((v, i) => { if (v > -topDb) { if (first < 0) first = i; last = i; } });
  if (first < 0) return new Float32Array(0);
  const a = first * 512, b = Math.min(y.length, (last + 1) * 512), keep = Math.round((sr * keepMs) / 1000);
  return y.slice(left ? Math.max(0, a - Math.floor(keep / 2)) : 0, right ? Math.min(y.length, b + keep) : y.length);
}
function speechLength(y, sr, topDb) {                          // seconds of non-silent audio (librosa.effects.split)
  const db = frameDb(y);
  return (db.filter((v) => v > -topDb).length * 512) / sr;
}
// Simple checks on a recording: how much speech, whether it clips, how loud the quiet parts are.
export function clipChecks(y, sr) {
  let peak = 0, clipped = 0;
  for (const v of y) { const a = Math.abs(v); peak = Math.max(peak, a); if (a > 0.99) clipped++; }
  const db = Array.from(frameDb(y)).sort((a, b) => a - b);
  const floor = db.length ? db[Math.floor(db.length * 0.1)] : 0;        // the quietest tenth, relative to the loudest frame
  return { speechSeconds: speechLength(y, sr, 30), clipped: clipped > sr * 0.01, noisy: floor > -30, silent: peak < 0.01 };
}
function crossfade(pieces, xf) {
  let out = new Float32Array(0);
  for (const p of pieces) {
    if (out.length > xf && p.length > xf && xf > 0) {
      const next = new Float32Array(out.length + p.length - xf);
      next.set(out.subarray(0, out.length - xf));
      for (let i = 0; i < xf; i++) { const f = i / (xf - 1); next[out.length - xf + i] = out[out.length - xf + i] * (1 - f) + p[i] * f; }
      next.set(p.subarray(xf), out.length);
      out = next;
    } else {
      const next = new Float32Array(out.length + p.length); next.set(out); next.set(p, out.length); out = next;
    }
  }
  return out;
}
function normalize(y, peak = 0.95) {
  let m = 0;
  for (const v of y) m = Math.max(m, Math.abs(v));
  return m < 1e-8 ? y : y.map((v) => (v / m) * peak);
}
function fft(re, im) {                                         // in-place radix-2 FFT
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci, vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi; re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

// 16-bit WAV file from samples, for playback and download
export function toWav(y, sr) {
  const buf = new ArrayBuffer(44 + y.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + y.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, y.length * 2, true);
  for (let i = 0; i < y.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, y[i])) * 32767, true);
  return new Blob([buf], { type: "audio/wav" });
}

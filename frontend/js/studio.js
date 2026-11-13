// The Studio: type text, choose or add a voice, hear it. Everything runs on this device.
import { requireUser, signOut } from "./auth.js";
import { createEngine, toWav, clipChecks } from "./engine.js";
import { SAMPLE_GRID, gridFromMel, fitCanvas, drawLandscape, drawMic, line } from "./landscape.js";

const $ = (id) => document.getElementById(id);
const user = await requireUser();
if (!user) throw new Error("not signed in");
$("who").textContent = user.email || "";
$("signout").onclick = signOut;

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const S = {                                                    // everything the page shows comes from here
  engine: null, loadFraction: 0, loadError: false,
  phase: "loading",                                            // loading | idle | speaking | playing | recording
  voice: "p225", filter: "all", query: "",
  mine: load("vc-my-voice"),                                   // { embedding, secs } for this visit only
  grid: SAMPLE_GRID, riseStart: 0, hasAudio: false, audioUrl: null, duration: 0,
  micHistory: [], recStart: 0,
};
if (S.mine) S.voice = "mine";
function load(k) { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } }
function save(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch {} }

// ---------------------------------------------------------------- models
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
ort.env.wasm.proxy = true;                                     // run the models off the main thread so the page stays smooth
createEngine("models/", { onProgress: (l, t) => { S.loadFraction = t ? l / t : 0; render(); } })
  .then((e) => { S.engine = e; toIdle(); renderVoices(); render(); })
  .catch(() => { S.loadError = true; render(); });

// ---------------------------------------------------------------- stage drawing
const audio = $("audio");
(function loop(now) {
  const { ctx, W, H } = fitCanvas($("stage-canvas"));
  const bar = document.querySelector(".stage-bar").offsetHeight, base = H - bar - 40, height = Math.min((base - 40) * 0.42, 120);
  if (S.phase === "recording") drawMic(ctx, W, H, S.micHistory);
  else if (S.phase === "loading") {
    const w = Math.min(W * 0.5, 420) * Math.max(0.05, S.loadFraction);
    ctx.strokeStyle = "#4CB572";
    line(ctx, [[W / 2 - w / 2, H * 0.45], [W / 2 + w / 2, H * 0.45]], 0.5 + 0.3 * Math.sin(now / 300), 1.6);
  } else {
    const rise = Math.min(1, (now - S.riseStart) / 600);
    const playT = S.phase === "playing" && S.duration ? Math.min(1, audio.currentTime / S.duration) : null;
    const amp = S.phase === "idle" && !reduced ? 1 + 0.035 * Math.sin(now / 900) : 1;
    drawLandscape(ctx, W, H, S.grid, { base, height, amp, rows: S.phase === "speaking" ? 30 : 1 + 29 * (reduced ? 1 : rise), playT, dim: S.phase === "speaking" ? 0.35 : S.phase === "idle" ? 0.85 : 1 });
  }
  if (S.phase === "playing") $("time").textContent = `${fmt(audio.currentTime)} / ${fmt(S.duration)}`;
  if (S.phase === "recording") tickRecording(now);
  requestAnimationFrame(loop);
})(performance.now());

// ---------------------------------------------------------------- voices
const voiceInfo = (v) => [v.gender && v.gender[0].toUpperCase() + v.gender.slice(1), v.accent !== "Other" && v.accent, v.region].filter(Boolean).join(", ");
const selectedVoice = () => (S.voice === "mine" && S.mine ? { id: "mine", label: "My voice", embedding: S.mine.embedding } : (() => { const v = S.engine.voices.find((x) => x.id === S.voice) || S.engine.voices[0]; return { id: v.id, label: v.id.toUpperCase(), embedding: v.embedding, info: voiceInfo(v) }; })());

const ICON_PLAY = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5v9l7-4.5z" fill="currentColor"/></svg>';
const ICON_TICK = '<svg class="tick" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.5 8.5l3.5 3.5 7.5-8"/></svg>';
function voiceRow(id, name, detail, on) {
  const row = document.createElement("div");
  row.className = "voice" + (on ? " on" : "");
  row.innerHTML = `<button type="button" class="pv" aria-label="Preview ${name}">${ICON_PLAY}</button>
    <button type="button" class="pick" aria-pressed="${on}"><span class="name"></span><span class="detail"></span></button>${on ? ICON_TICK : ""}`;
  row.querySelector(".name").textContent = name;
  row.querySelector(".detail").textContent = detail;
  row.querySelector(".pick").onclick = () => { S.voice = id; renderVoices(); render(); };
  row.querySelector(".pv").onclick = () => { S.voice = id; renderVoices(); speak("Hello, this is how I sound."); };
  return row;
}
function renderVoices() {
  const mine = $("mine");
  mine.replaceChildren();
  if (S.mine) {
    const row = voiceRow("mine", "My voice", `Added today, ${S.mine.secs} seconds`, S.voice === "mine");
    const replace = document.createElement("button");
    replace.className = "btn-text small"; replace.textContent = "Replace"; replace.onclick = openPanel;
    row.append(replace);
    mine.append(row);
  } else {
    mine.innerHTML = `<div class="add-box"><span class="small muted">Record or upload 5 to 20 seconds.</span><button type="button" class="btn-outline" id="add">Add your voice</button></div>`;
    $("add").onclick = openPanel;
  }
  if (!S.engine) return;
  const q = S.query.trim().toLowerCase();
  const list = S.engine.voices.filter((v) => (S.filter === "all" || v.gender === S.filter) && (!q || `${v.id} ${voiceInfo(v)}`.toLowerCase().includes(q)));
  $("list").replaceChildren(...list.map((v) => voiceRow(v.id, v.id.toUpperCase(), voiceInfo(v) + (v.seen_in_training ? "" : ". Not heard in training"), v.id === S.voice)));
  $("no-results").hidden = list.length > 0;
}
$("query").oninput = (e) => { S.query = e.target.value; renderVoices(); };
document.querySelectorAll(".chip").forEach((c) => (c.onclick = () => {
  S.filter = c.dataset.g;
  document.querySelectorAll(".chip").forEach((x) => x.setAttribute("aria-pressed", x === c));
  renderVoices();
}));

// ---------------------------------------------------------------- speaking
const text = $("text");
text.oninput = render;
text.onkeydown = (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); speak(); } };
$("speed").oninput = (e) => { $("speed-out").textContent = (+e.target.value).toFixed(2) + "×"; };
$("speak").onclick = () => speak();

async function speak(override) {
  const words = override || text.value.trim();
  if (!S.engine || !words || S.phase === "speaking" || S.phase === "recording") return;
  audio.pause();
  $("speak-error").hidden = true;
  S.phase = "speaking"; render();
  const bar = $("speak-bar");
  try {
    const r = await S.engine.speak(words, selectedVoice().embedding, {
      speed: +$("speed").value, seed: Math.floor(Math.random() * 2 ** 31),
      onProgress: (i, n) => { bar.style.width = `${Math.round((i / n) * 100)}%`; },
    });
    if (!r) { S.phase = "idle"; showError("There are no words to speak. Type English words; digits are read as numbers."); return; }
    if (S.audioUrl) URL.revokeObjectURL(S.audioUrl);
    S.audioUrl = URL.createObjectURL(toWav(r.audio, S.engine.sr));
    S.duration = r.audio.length / S.engine.sr;
    S.grid = gridFromMel(r.mel); S.riseStart = performance.now(); S.hasAudio = true;
    audio.src = S.audioUrl;
    const dl = $("download");
    dl.href = S.audioUrl; dl.download = `voice-cloner-${selectedVoice().id}.wav`; dl.removeAttribute("aria-disabled");
    play();
  } catch (err) {
    console.error(err);
    S.phase = "idle";
    showError("Something went wrong while speaking. Try a shorter text, or reload the page.");
  } finally {
    bar.style.width = "0";
  }
}
function showError(m) { $("speak-error").textContent = m; $("speak-error").hidden = false; render(); }
function toIdle() { if (S.phase !== "recording") S.phase = "idle"; }     // never ends a recording in progress
function play() { audio.currentTime = 0; audio.play().then(() => { if (S.phase !== "recording") S.phase = "playing"; render(); }).catch(() => { toIdle(); render(); }); }
audio.onended = () => { toIdle(); $("time").textContent = `${fmt(S.duration)} / ${fmt(S.duration)}`; render(); };
$("play").onclick = () => { if (S.phase === "recording") return; if (S.phase === "playing") { audio.pause(); S.phase = "idle"; render(); } else if (S.hasAudio) { audio.play(); S.phase = "playing"; render(); } };

// ---------------------------------------------------------------- the page's words and buttons
function loadingText() {
  return S.loadFraction >= 1 ? "Getting the voice models ready…" : `Loading the voice models… ${Math.round(S.loadFraction * 100)}%`;
}
function render() {
  const empty = !text.value.trim(), v = S.engine ? selectedVoice() : null;
  $("count").textContent = `${text.value.length} of 1000 characters`;
  $("selected").textContent = v ? `${v.label} selected` : "";
  const speaking = S.phase === "speaking", playing = S.phase === "playing";
  const btn = $("speak");
  btn.disabled = !S.engine || empty || speaking || S.phase === "recording";
  btn.classList.toggle("busy", speaking);
  btn.firstChild.textContent = speaking ? "Speaking…" : "Speak";
  $("speak-hint").textContent = S.loadError ? "Couldn’t download the voice models. Check your connection and reload the page."
    : !S.engine ? loadingText()
    : empty ? "Type something to speak" : speaking ? "Making the audio on this device" : `Voice: ${v.label}${v.info ? " (" + v.info + ")" : ""}. Ctrl + Enter also speaks.`;
  $("status").textContent = S.loadError ? "Couldn’t download the voice models."
    : !S.engine ? loadingText()
    : S.phase === "recording" ? `Listening… ${Math.floor((performance.now() - S.recStart) / 1000)} s`
    : $("panel").hidden === false ? "Your recording will show here."
    : speaking ? `Speaking in ${v.label}…` : playing ? `Playing ${v.label}`
    : S.hasAudio ? "Done. Play it again or download it."
    : S.voice === "mine" ? "Your voice is ready. Press Speak." : "Ready. Type something and press Speak.";
  $("play").disabled = !S.hasAudio || S.phase === "recording";
  $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
  $("icon-play").hidden = playing; $("icon-pause").hidden = !playing;
  if (!S.hasAudio) $("time").textContent = "0:00";
}

// ---------------------------------------------------------------- add your voice
const P = { op: 0, acquiring: false, tab: "record", result: null, stream: null, recorder: null, analyser: null, ctx: null, opener: null };
function openPanel(e) {
  P.opener = e && e.currentTarget;
  $("panel").hidden = false; $("scrim").hidden = false;
  resetClip(); $("consent").checked = false; setTab("record"); updateUse(); render();
  $("close").focus();
}
function closePanel() {
  if (S.phase === "recording") stopRecording(true);
  P.op++;                                                      // anything still decoding or asking for the mic is now stale
  $("panel").hidden = true; $("scrim").hidden = true; render();
  (P.opener && document.contains(P.opener) ? P.opener : $("query")).focus();
}
$("close").onclick = closePanel;
$("scrim").onclick = closePanel;
$("panel").onkeydown = (e) => { if (e.key === "Escape") closePanel(); };
function setTab(t) {
  P.tab = t;
  $("tab-record").setAttribute("aria-pressed", t === "record"); $("tab-upload").setAttribute("aria-pressed", t === "upload");
  $("pane-record").hidden = t !== "record"; $("pane-upload").hidden = t !== "upload";
  resetClip(); updateUse();
}
$("tab-record").onclick = () => { if (S.phase !== "recording") setTab("record"); };
$("tab-upload").onclick = () => { if (S.phase !== "recording") setTab("upload"); };
function resetClip() {
  P.op++; P.result = null;
  $("checks").hidden = true; $("clip-error").hidden = true; $("file-name").hidden = true;
  $("rec-bar").style.width = "0"; $("rec-time").textContent = "0:00 / 0:20"; $("rec-label").textContent = "Start recording";
  updateUse();
}
function clipError(m, op = P.op) { if (op !== P.op) return; $("clip-error").textContent = m; $("clip-error").hidden = false; }

$("rec").onclick = () => (S.phase === "recording" ? stopRecording(false) : startRecording());
async function startRecording() {
  if (S.phase === "recording" || P.acquiring || (P.recorder && P.recorder.state === "recording")) return;
  if (S.phase === "speaking") return clipError("Wait for the current sentence to finish, then record.");
  audio.pause();
  resetClip();
  const op = P.op;
  P.acquiring = true; $("rec").disabled = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (err) {
    clipError(err && err.name === "NotAllowedError"
      ? "Microphone access is blocked. Allow it from the icon in your browser’s address bar, or upload a file instead."
      : "Couldn’t find a microphone. Plug one in, or upload a file instead.", op);
    return;
  } finally {
    P.acquiring = false; $("rec").disabled = false;
  }
  if (op !== P.op || $("panel").hidden || S.phase === "speaking") { stream.getTracks().forEach((t) => t.stop()); return; }
  P.stream = stream;
  P.ctx = new AudioContext();
  P.analyser = P.ctx.createAnalyser(); P.analyser.fftSize = 2048;
  P.ctx.createMediaStreamSource(P.stream).connect(P.analyser);
  const chunks = [];
  P.recorder = new MediaRecorder(P.stream);
  P.recorder.ondataavailable = (e) => chunks.push(e.data);
  P.recorder.onstop = async () => {
    if (P.discard) return;
    let samples;
    try { samples = await decode(await new Blob(chunks).arrayBuffer()); }
    catch { return clipError("Couldn’t read the recording. Try again, or upload a file instead.", op); }
    await useClip(samples, op);
  };
  P.discard = false;
  P.recorder.start();
  S.recStart = performance.now(); S.micHistory = []; S.phase = "recording";
  $("rec").setAttribute("aria-pressed", "true"); $("rec-label").textContent = "Stop";
  updateUse(); render();
}
function tickRecording(now) {
  const t = (now - S.recStart) / 1000;
  const buf = new Float32Array(P.analyser.fftSize);
  P.analyser.getFloatTimeDomainData(buf);
  S.micHistory.push(buf); if (S.micHistory.length > 6) S.micHistory.shift();
  $("rec-bar").style.width = `${Math.min(100, (t / 20) * 100)}%`;
  $("rec-time").textContent = `${fmt(t)} / 0:20`;
  if (Math.floor(t) !== Math.floor(t - 0.02)) render();
  if (t >= 20) stopRecording(false);
}
function stopRecording(discard) {
  const t = (performance.now() - S.recStart) / 1000;
  P.discard = discard || t < 5;
  if (P.recorder && P.recorder.state !== "inactive") P.recorder.stop();
  P.stream && P.stream.getTracks().forEach((tr) => tr.stop());
  P.ctx && P.ctx.close();
  S.phase = S.engine ? "idle" : "loading";
  $("rec").setAttribute("aria-pressed", "false"); $("rec-label").textContent = "Record again";
  if (!discard && t < 5) clipError("That was under 5 seconds. Read a little more of the passage and try again.");
  updateUse(); render();
}

$("file").onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  resetClip();
  const op = P.op;
  $("file-name").textContent = f.name; $("file-name").hidden = false;
  if (f.size > 20 * 1024 * 1024) return clipError("That file is over 20 MB. Use a clip under one minute.");
  let samples;
  try { samples = await decode(await f.arrayBuffer()); }
  catch { return clipError("Couldn’t read that file. Use a WAV, MP3, M4A or OGG recording.", op); }
  await useClip(samples, op);
};

async function decode(bytes) {                                 // any format the browser can play -> mono, 22,050 Hz, max 60 s
  const ctx = new OfflineAudioContext(1, 1, 22050);
  const buf = await ctx.decodeAudioData(bytes);
  const ch = buf.getChannelData(0);
  return ch.slice(0, Math.min(ch.length, 22050 * 60));
}
async function useClip(samples, op) {
  if (op !== P.op) return;
  if (!S.engine) return clipError("The voice models are still loading. Try again in a moment.", op);
  const c = clipChecks(samples, 22050);
  if (c.silent) return clipError("The clip is silent. Check the microphone, then record again.", op);
  if (c.speechSeconds < 3) return clipError(`The clip has only ${c.speechSeconds.toFixed(1)} seconds of speech. Record at least 5 seconds in a quiet room.`, op);
  let r;
  try { r = await S.engine.embedClip(samples); }
  catch (err) { console.error(err); return clipError("Couldn’t make a voice from this clip. Try again, or use a different clip.", op); }
  if (op !== P.op) return;                                     // a newer clip, a reset or a closed panel replaced this one
  P.result = { embedding: r.embedding, secs: Math.round(r.speechSeconds) };
  const items = [
    [true, `${Math.round(c.speechSeconds)} seconds of speech`],
    [!c.noisy, c.noisy ? "Background is noisy. A quieter room gives a closer voice" : "Background is quiet"],
    [!c.clipped, c.clipped ? "Some parts are too loud and distort. Move back from the mic" : "No distortion"],
  ];
  $("checks").innerHTML = items.map(([ok, t]) => `<li class="${ok ? "" : "warn"}">${ok ? ICON_TICK.replace('class="tick"', 'style="color:var(--link);flex:none"') : '<span aria-hidden="true" style="flex:none;width:16px;text-align:center;font-weight:700">!</span>'}<span></span></li>`).join("");
  $("checks").querySelectorAll("li span:last-child").forEach((s, i) => (s.textContent = items[i][1]));
  $("checks").hidden = false;
  updateUse();
}
$("consent").onchange = updateUse;
function updateUse() {
  const ready = !!P.result, ok = ready && $("consent").checked;
  $("use").disabled = !ok;
  $("use-hint").textContent = S.phase === "recording" ? "Recording…" : !ready ? (P.tab === "record" ? "Record at least 5 seconds first" : "Choose a file first") : !$("consent").checked ? "Tick the permission box first" : "Ready";
}
$("use").onclick = () => {
  if ($("use").disabled || !P.result) return;
  S.mine = P.result; save("vc-my-voice", S.mine);
  S.voice = "mine"; S.hasAudio = false;
  closePanel(); renderVoices(); render();
  $("speak").focus();
};

renderVoices();
render();

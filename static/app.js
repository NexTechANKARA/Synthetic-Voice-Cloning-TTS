// Global State & Cache
const state = {
  sources: {
    ae: { type: 'preset', file: null, blob: null },
    vae: { type: 'preset', file: null, blob: null }
  },
  recordings: {
    ae: { mediaRecorder: null, chunks: [], timer: null, seconds: 0 },
    vae: { mediaRecorder: null, chunks: [], timer: null, seconds: 0 }
  },
  spectrogramCache: {
    ae: { clean: null, noisy: null, denoised: null },
    vae: { input: null, reconstructed: null },
    tts: { generated: null },
    pipe: { generated: null, reference: null }
  }
};

// Initialize app on page load
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSystemStatus();
});

// Navigation Pill Tabs
function initNavigation() {
  const pills = document.querySelectorAll('.nav-pill');
  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      pills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');

      const targetTab = pill.getAttribute('data-tab');
      document.querySelectorAll('.tab-content').forEach(tc => {
        tc.classList.remove('active');
      });
      document.getElementById(targetTab).classList.add('active');
    });
  });
}

// System Status Check
async function initSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'ready') {
      document.getElementById('systemStatusText').innerText = 'All 3 Checkpoints Loaded';
    }
  } catch (err) {
    document.getElementById('systemStatusText').innerText = 'Backend Offline';
    document.getElementById('systemStatusBadge').style.borderColor = 'rgba(248, 113, 113, 0.4)';
  }
}

// Audio Source Segment Switcher
function setAudioSource(model, type) {
  state.sources[model].type = type;
  const parent = document.getElementById(`tab-${model}`);
  const btns = parent.querySelectorAll('.segmented-control .segment-btn');
  btns.forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-source') === type);
  });

  const panels = parent.querySelectorAll('.source-panel');
  panels.forEach(p => p.classList.remove('active'));
  document.getElementById(`${model}-source-${type}`).classList.add('active');
}

// File Selection Handler
function handleFileSelect(model, input) {
  if (input.files && input.files[0]) {
    const file = input.files[0];
    state.sources[model].file = file;
    document.getElementById(`${model}-file-label`).innerText = file.name;
  }
}

// Converts an AudioBuffer from the browser AudioContext into a true 16-bit PCM RIFF WAV Blob
function audioBufferToWav(buffer, targetSampleRate = 22050) {
  const channelData = buffer.getChannelData(0);
  let samples;
  if (buffer.sampleRate !== targetSampleRate) {
    const ratio = buffer.sampleRate / targetSampleRate;
    const newLen = Math.round(channelData.length / ratio);
    samples = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      samples[i] = channelData[Math.floor(i * ratio)] || 0;
    }
  } else {
    samples = channelData;
  }

  const numSamples = samples.length;
  const wavBuffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(wavBuffer);

  const writeString = (v, offset, str) => {
    for (let i = 0; i < str.length; i++) v.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, targetSampleRate, true);
  view.setUint32(28, targetSampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}

// Microphone Recording Handler
async function toggleRecord(model) {
  const recState = state.recordings[model];
  const btn = document.getElementById(`${model}-rec-btn`);
  const text = document.getElementById(`${model}-rec-text`);
  const timer = document.getElementById(`${model}-rec-timer`);

  if (!recState.mediaRecorder || recState.mediaRecorder.state === 'inactive') {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recState.chunks = [];
      recState.mediaRecorder = new MediaRecorder(stream);

      recState.mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) recState.chunks.push(e.data);
      };

      recState.mediaRecorder.onstop = async () => {
        try {
          const rawBlob = new Blob(recState.chunks, { type: recState.mediaRecorder.mimeType || 'audio/webm' });
          const arrayBuffer = await rawBlob.arrayBuffer();
          const AudioContextClass = window.AudioContext || window.webkitAudioContext;
          const audioCtx = new AudioContextClass();
          const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
          const wavBlob = audioBufferToWav(audioBuffer, 22050);
          state.sources[model].blob = wavBlob;
          text.innerText = 'Recorded (PCM WAV ready)';
        } catch (err) {
          console.warn('Audio decoding fallback:', err);
          const fallbackBlob = new Blob(recState.chunks, { type: 'audio/wav' });
          state.sources[model].blob = fallbackBlob;
          text.innerText = 'Recorded (ready)';
        }
        stream.getTracks().forEach(track => track.stop());
        btn.classList.remove('recording');
        clearInterval(recState.timer);
      };

      recState.mediaRecorder.start();
      btn.classList.add('recording');
      text.innerText = 'Recording...';
      recState.seconds = 0;
      timer.innerText = '00:00';

      recState.timer = setInterval(() => {
        recState.seconds++;
        const s = recState.seconds < 10 ? `0${recState.seconds}` : recState.seconds;
        timer.innerText = `00:${s}`;
        if (recState.seconds >= 3) {
          recState.mediaRecorder.stop();
        }
      }, 1000);

    } catch (err) {
      alert('Microphone access denied or unavailable: ' + err.message);
    }
  } else {
    recState.mediaRecorder.stop();
  }
}

// ==================== Model 1: Autoencoder ====================
async function runAutoencoder() {
  const btn = document.getElementById('ae-run-btn');
  btn.classList.add('loading');
  const noiseLevel = parseFloat(document.getElementById('ae-noise-slider').value);

  const formData = new FormData();
  formData.append('noise_level', noiseLevel);

  const src = state.sources.ae;
  if (src.type === 'preset') {
    formData.append('sample_id', document.getElementById('ae-preset-select').value);
  } else if (src.type === 'upload' && src.file) {
    formData.append('audio_file', src.file);
  } else if (src.type === 'record' && src.blob) {
    formData.append('audio_file', src.blob, 'mic_clip.wav');
  } else {
    formData.append('sample_id', 'male_voice.wav');
  }

  try {
    const res = await fetch('/api/models/autoencoder/infer', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || 'Inference failed');

    // Update metrics
    document.getElementById('ae-metric-noisy-mse').innerText = data.metrics.noisy_mse;
    document.getElementById('ae-metric-denoised-mse').innerText = data.metrics.denoised_mse;
    document.getElementById('ae-metric-improvement').innerText = `+${data.metrics.noise_reduction_pct}%`;

    // Cache spectrograms
    state.spectrogramCache.ae = data.spectrograms;
    document.getElementById('ae-placeholder').classList.add('hidden');
    drawSpectrogram('ae-spectrogram-canvas', data.spectrograms.denoised);

    // Update audio players
    document.getElementById('ae-audio-denoised').src = data.audio.denoised;
    document.getElementById('ae-audio-noisy').src = data.audio.noisy;
    document.getElementById('ae-audio-clean').src = data.audio.clean;

  } catch (err) {
    alert('Error running Autoencoder: ' + err.message);
  } finally {
    btn.classList.remove('loading');
  }
}

// ==================== Model 2: Speaker VAE ====================
async function runSpeakerVAE() {
  const btn = document.getElementById('vae-run-btn');
  btn.classList.add('loading');
  const temp = parseFloat(document.getElementById('vae-temp-slider').value);

  const formData = new FormData();
  formData.append('temperature', temp);

  const src = state.sources.vae;
  if (src.type === 'preset') {
    formData.append('sample_id', document.getElementById('vae-preset-select').value);
  } else if (src.type === 'upload' && src.file) {
    formData.append('audio_file', src.file);
  } else if (src.type === 'record' && src.blob) {
    formData.append('audio_file', src.blob, 'mic_clip.wav');
  } else {
    formData.append('sample_id', 'female_voice.wav');
  }

  try {
    const res = await fetch('/api/models/speaker_vae/infer', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || 'Inference failed');

    // Metrics
    document.getElementById('vae-metric-speaker').innerText = data.metrics.predicted_speaker;
    document.getElementById('vae-metric-conf').innerText = `${data.metrics.confidence_pct}%`;
    document.getElementById('vae-metric-kl').innerText = data.metrics.kl_divergence;

    // Speaker probability distribution bars
    const barsContainer = document.getElementById('vae-speaker-bars');
    barsContainer.innerHTML = '';
    data.speaker_distribution.forEach(item => {
      const isTop = item.id === data.metrics.predicted_speaker;
      const bar = document.createElement('div');
      bar.className = `spk-bar-item ${isTop ? 'winner' : ''}`;
      bar.innerHTML = `
        <div class="spk-bar-top">
          <span>${item.id}</span>
          <span>${item.prob}%</span>
        </div>
        <div class="spk-bar-bg">
          <div class="spk-bar-fill" style="width: ${Math.max(4, item.prob)}%"></div>
        </div>
      `;
      barsContainer.appendChild(bar);
    });

    // 64D Latent Vector Grid
    const latentGrid = document.getElementById('vae-latent-grid');
    latentGrid.innerHTML = '';
    const zVals = data.latent.z;
    const maxZ = Math.max(...zVals.map(Math.abs), 1.0);

    zVals.forEach((val, idx) => {
      const cell = document.createElement('div');
      cell.className = 'latent-cell';
      cell.title = `Dim ${idx}: ${val}`;
      // Map [-maxZ, maxZ] to purple-lavender-cyan hue
      const norm = (val / maxZ + 1) / 2; // [0, 1]
      const r = Math.round(120 + 135 * norm);
      const g = Math.round(80 + 100 * norm);
      const b = Math.round(220 + 35 * norm);
      cell.style.background = `rgb(${r}, ${g}, ${b})`;
      latentGrid.appendChild(cell);
    });

    // Spectrogram
    state.spectrogramCache.vae = data.spectrograms;
    document.getElementById('vae-placeholder').classList.add('hidden');
    drawSpectrogram('vae-spectrogram-canvas', data.spectrograms.reconstructed);

    // Audio
    document.getElementById('vae-audio-recon').src = data.audio.reconstructed;
    document.getElementById('vae-audio-input').src = data.audio.input;

  } catch (err) {
    alert('Error running Speaker VAE: ' + err.message);
  } finally {
    btn.classList.remove('loading');
  }
}

// ==================== Model 3: Transformer TTS ====================
function setTTSText(txt) {
  document.getElementById('tts-text-input').value = txt;
}

async function runTransformerTTS() {
  const btn = document.getElementById('tts-run-btn');
  btn.classList.add('loading');
  const text = document.getElementById('tts-text-input').value.trim();
  const speakerId = document.getElementById('tts-speaker-select').value;

  const formData = new FormData();
  formData.append('text', text || 'Voice cloning with neural transformers');
  formData.append('speaker_id', speakerId);

  try {
    const res = await fetch('/api/models/transformer_tts/infer', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || 'Inference failed');

    // Token Chips
    const tokensContainer = document.getElementById('tts-tokens-container');
    tokensContainer.innerHTML = '';
    document.getElementById('tts-token-count').innerText = `${data.tokens.length} characters`;

    data.tokens.forEach(tok => {
      const chip = document.createElement('div');
      chip.className = 'token-chip';
      const displayChar = tok.char === ' ' ? '␣' : tok.char;
      chip.innerHTML = `<span>${displayChar}</span><span class="token-id">${tok.id}</span>`;
      tokensContainer.appendChild(chip);
    });

    // Spectrogram
    document.getElementById('tts-placeholder').classList.add('hidden');
    drawSpectrogram('tts-spectrogram-canvas', data.spectrogram);

    // Audio
    document.getElementById('tts-audio').src = data.audio;

  } catch (err) {
    alert('Error running Transformer TTS: ' + err.message);
  } finally {
    btn.classList.remove('loading');
  }
}

// ==================== Model 4: Pipeline ====================
async function runPipeline() {
  const btn = document.getElementById('pipe-run-btn');
  btn.classList.add('loading');
  const text = document.getElementById('pipe-text-input').value.trim();
  const sampleId = document.getElementById('pipe-preset-select').value;
  const noise = parseFloat(document.getElementById('pipe-noise-slider').value);

  const formData = new FormData();
  formData.append('text', text);
  formData.append('sample_id', sampleId);
  formData.append('noise_level', noise);

  try {
    const res = await fetch('/api/models/pipeline/infer', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || 'Inference failed');

    document.getElementById('pipe-speaker-tag').innerText = `Cloned Voice (${data.cloned_speaker})`;

    state.spectrogramCache.pipe = {
      generated: data.generated_spectrogram,
      reference: data.reference_spectrogram
    };

    document.getElementById('pipe-placeholder').classList.add('hidden');
    drawSpectrogram('pipe-spectrogram-canvas', data.generated_spectrogram);

    document.getElementById('pipe-audio-generated').src = data.generated_audio;
    document.getElementById('pipe-audio-reference').src = data.reference_audio;

  } catch (err) {
    alert('Error running Pipeline: ' + err.message);
  } finally {
    btn.classList.remove('loading');
  }
}

// ==================== Spectrogram Canvas Colormap Renderer ====================
function switchSpecTab(model, type) {
  const parent = document.getElementById(`tab-${model}`);
  const tabs = parent.querySelectorAll('.spec-tab');
  tabs.forEach(t => {
    t.classList.toggle('active', t.getAttribute('onclick').includes(`'${type}'`));
  });

  const matrix = state.spectrogramCache[model][type];
  if (matrix) {
    drawSpectrogram(`${model}-spectrogram-canvas`, matrix);
  }
}

// Renders an 80 x 128 mel matrix onto Canvas with cosmic violet/lavender colormap
function drawSpectrogram(canvasId, matrix) {
  if (!matrix || !matrix.length) return;
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');

  const rows = matrix.length;      // 80 mel frequency bands
  const cols = matrix[0].length;   // 128 time frames
  const w = canvas.width;
  const h = canvas.height;

  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;

  // Render vertically flipped so low frequencies are at bottom
  for (let y = 0; y < h; y++) {
    const melRow = Math.floor(((h - 1 - y) / h) * rows);
    for (let x = 0; x < w; x++) {
      const melCol = Math.floor((x / w) * cols);
      const val = matrix[melRow][melCol]; // [0, 1]

      // Colormap: Deep Indigo -> Lavender -> Neon Pink -> Gold
      let r, g, b;
      if (val < 0.3) {
        const t = val / 0.3;
        r = Math.round(20 + 50 * t);
        g = Math.round(15 + 30 * t);
        b = Math.round(50 + 120 * t);
      } else if (val < 0.7) {
        const t = (val - 0.3) / 0.4;
        r = Math.round(70 + 130 * t);
        g = Math.round(45 + 50 * t);
        b = Math.round(170 + 60 * t);
      } else {
        const t = (val - 0.7) / 0.3;
        r = Math.round(200 + 55 * t);
        g = Math.round(95 + 130 * t);
        b = Math.round(230 - 30 * t);
      }

      const pixelIdx = (y * w + x) * 4;
      data[pixelIdx] = r;
      data[pixelIdx + 1] = g;
      data[pixelIdx + 2] = b;
      data[pixelIdx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

// ==================== Ethical Governance & AAC Handlers ====================

let aacCategories = {};
let verifierFile = null;

// Modal handlers
function openEthicsModal() {
  document.getElementById('ethicsModal').classList.add('active');
}

function closeEthicsModal() {
  document.getElementById('ethicsModal').classList.remove('active');
}

// Load AAC Phrases from Backend
async function fetchAAC() {
  try {
    const res = await fetch('/api/assistive/phrases');
    const data = await res.json();
    if (data.success) {
      aacCategories = data.categories;
      loadAACCategory('Urgent & Health');
    }
  } catch (err) {
    console.error("Failed to load AAC phrases:", err);
  }
}

// Render AAC Category
function loadAACCategory(catName) {
  const container = document.getElementById('aac-phrase-grid');
  if (!container) return;

  // Update active button
  const btns = document.querySelectorAll('#aac-category-selector .segment-btn');
  btns.forEach(b => {
    b.classList.toggle('active', b.innerText.includes(catName.split(' ')[0]));
  });

  const phrases = aacCategories[catName] || [];
  container.innerHTML = phrases.map(p => `
    <button class="aac-phrase-btn" onclick="speakAACPhrase('${p.replace(/'/g, "\\'")}')">
      <span>${p}</span>
      <span class="phrase-play-icon">▶</span>
    </button>
  `).join('');
}

// Speak AAC Phrase
async function speakAACPhrase(phrase) {
  const speaker = document.getElementById('aac-speaker-select').value;
  const activeLabel = document.getElementById('aac-active-phrase');
  const player = document.getElementById('aac-audio-player');

  activeLabel.innerText = `Speaking: "${phrase}"...`;

  try {
    const formData = new FormData();
    formData.append('phrase', phrase);
    formData.append('speaker_id', speaker);

    const res = await fetch('/api/assistive/speak', {
      method: 'POST',
      body: formData
    });

    if (res.status === 429) {
      const err = await res.json();
      activeLabel.innerText = `⚠️ ${err.detail}`;
      return;
    }

    const data = await res.json();
    if (data.success) {
      player.src = data.audio;
      player.play();
      activeLabel.innerText = `Spoken in [${speaker}]: "${phrase}"`;
    } else {
      activeLabel.innerText = "Error synthesizing speech";
    }
  } catch (err) {
    activeLabel.innerText = `Synthesis failed: ${err.message}`;
  }
}

// Speak Custom AAC Message
function speakCustomAAC() {
  const input = document.getElementById('aac-custom-input');
  const phrase = input.value.trim();
  if (phrase) {
    speakAACPhrase(phrase);
    input.value = '';
  }
}

// Handle Verifier File Selection
function handleVerifierFile(input) {
  if (input.files && input.files[0]) {
    verifierFile = input.files[0];
    document.getElementById('verifier-file-label').innerText = verifierFile.name;
  }
}

// Run Watermark & Provenance Inspection
async function runWatermarkVerification() {
  if (!verifierFile) {
    alert("Please select a WAV file to verify.");
    return;
  }

  const placeholder = document.getElementById('verifier-placeholder');
  const report = document.getElementById('verifier-report-content');
  const badge = document.getElementById('verifier-badge');
  const conf = document.getElementById('verifier-confidence');

  placeholder.style.display = 'block';
  placeholder.innerHTML = '<span>Scanning audio spectrum & metadata...</span>';
  report.style.display = 'none';

  try {
    const formData = new FormData();
    formData.append('audio_file', verifierFile);

    const res = await fetch('/api/ethics/verify-watermark', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();

    if (data.success) {
      const r = data.result;
      placeholder.style.display = 'none';
      report.style.display = 'block';

      if (r.is_synthetic) {
        badge.innerText = 'VERIFIED SYNTHETIC';
        badge.className = 'verifier-status-badge';
        conf.innerText = `${r.confidence_pct}% Confidence`;
      } else {
        badge.innerText = 'UNMARKED AUDIO';
        badge.className = 'verifier-status-badge unmarked';
        conf.innerText = 'No Watermark Found';
      }

      document.getElementById('v-method').innerText = r.verification_method;
      document.getElementById('v-compliance').innerText = r.ethical_compliance;
      document.getElementById('v-speaker').innerText = (r.provenance && r.provenance.SPK) ? r.provenance.SPK : 'Unknown';
      document.getElementById('v-signature').innerText = (r.provenance && r.provenance.SYNTH) ? r.provenance.SYNTH : 'None';
      
      const ts = (r.provenance && r.provenance.TS) ? new Date(parseInt(r.provenance.TS) * 1000).toLocaleString() : 'N/A';
      document.getElementById('v-ts').innerText = ts;
    }
  } catch (err) {
    placeholder.innerHTML = `<span>Verification error: ${err.message}</span>`;
  }
}

// Fetch AAC on page load
document.addEventListener('DOMContentLoaded', () => {
  fetchAAC();
});


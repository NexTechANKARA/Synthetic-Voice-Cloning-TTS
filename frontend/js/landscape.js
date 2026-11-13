// The spectrogram landscape: a mel spectrogram drawn as a 3D wire mesh (time left to right, pitch front to back).

const ROWS = 30, COLS = 44;
const LINE = "#4CB572";

// A sentence spoken by the app (voice P225), used before the visitor has made one of their own.
const SAMPLE = "01357988753222233113564334433333565321223210|01358977753322233113664333322333455321222210|00247966653322223113554323222332455321222210|01347966653332123113554223223433465321222210|01458a77765542124224665323224643688532233310|0167bc87898752246557aa73233357437bb853345520|0158cc768a8752248669dea4222345437bda53345520|0157aa769a7642148669efa4232345337cda53355520|01569866896652259659dd94333345337dea63356620|02678645665763257559dd9423336743aghc63467730|0288753344466324534adc8323347954bhhd74478830|0288643233355213434ab96312347a65affe84468720|026763233334321343599642122379648cee94467620|015652233234321234576542112368537bde94456510|014652233344421234576542122257436acd84455410|014653333455421245677642222257436acd84455420|0156643445654212456776422222463259ab84355520|02566544456642124566654212224532589974345420|02456543455542124455543111113532478863345420|0256764445564212445654321112463259ab74345520|027898555777632356688752122368437dgga5467620|04ccca7679aa8424889bdb7323448b74bknlc669ba40|06hhgc889ccdb537cccgjhb42555cib7gsupe78dhg60|08kllhbacghhe76ahgflpnf53676fog9lyyqe8ainm90|1apstpgdgnooja8emkjoroh647a7gsjbnzxoe9emsrc0|1bpttpidgnppmc9gppouwtl848b8gskcnwuogbgpuue1|1boturkdiqrsofbhopptvtma48c9hqjcltrpjdiqtte1|3erutqkdjqsutkejquvxywrd59fcjsmfnvvungmuxve1|5djihffcfhhkojeglpqrpnle7afchmjejopplhkpqnb1|7abcdceffeceggeefjjhebccbefceeeeeeffggfhfe81";
export const SAMPLE_GRID = SAMPLE.split("|").map((r) => [...r].map((c) => parseInt(c, 36) / 35));

// A 30 x 44 height grid from a mel spectrogram ({ data: 80*T values in 0..1, T }), the way the sample was made:
// the lower 56 mel bands (where speech energy is), averaged into cells, stretched to 0..1 and sharpened.
export function gridFromMel({ data, T }) {
  const grid = [];
  let lo = Infinity, hi = -Infinity;
  for (let r = 0; r < ROWS; r++) {
    const row = [], b0 = Math.floor((r * 56) / ROWS), b1 = Math.max(b0 + 1, Math.floor(((r + 1) * 56) / ROWS));
    for (let c = 0; c < COLS; c++) {
      const t0 = Math.floor((c * T) / COLS), t1 = Math.max(t0 + 1, Math.floor(((c + 1) * T) / COLS));
      let s = 0, n = 0;
      for (let b = b0; b < b1; b++) for (let t = t0; t < t1 && t < T; t++) { s += data[(55 - b) * T + t]; n++; }
      const v = n ? s / n : 0;
      row.push(v); lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    grid.push(row);
  }
  return grid.map((row) => row.map((v) => Math.pow((v - lo) / (hi - lo || 1), 2.6)));
}

export function fitCanvas(cv) {
  const r = cv.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(d, 0, 0, d, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, W: r.width, H: r.height };
}

export function line(ctx, pts, alpha, width) {
  ctx.globalAlpha = alpha; ctx.lineWidth = width; ctx.beginPath();
  pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
}

// rows: how many rows are visible from the back (fractional, for the rising animation);
// playT: playhead position 0..1 (lights up what has been heard) or null.
export function drawLandscape(ctx, W, H, grid, { base = H - 70, height = Math.min(H * 0.26, 120), amp = 1, rows = ROWS, playT = null, dim = 1, spread = Math.min(W * 0.9, 1100) } = {}) {
  const P = (t, f, z) => { const d = 1 + f * 1.1; return [W / 2 + (t - 0.5) * spread / d, base - f * height * 0.95 - z * height * amp / d]; };
  ctx.strokeStyle = LINE; ctx.fillStyle = "#F2F6F1"; ctx.lineJoin = "round"; ctx.lineCap = "round";
  for (let i = 0; i < ROWS; i++) {
    const vis = Math.max(0, Math.min(1, rows - i));
    if (vis <= 0) continue;
    const f = 1 - i / (ROWS - 1), row = grid[ROWS - 1 - i];
    const pts = row.map((z, j) => P(j / (COLS - 1), f, z * vis));
    const a = (0.22 + 0.78 * i / (ROWS - 1)) * vis * dim, w = i === ROWS - 1 ? 1.6 : 1.1;
    if (playT == null) line(ctx, pts, a, w);
    else {
      const k = Math.max(1, Math.round(playT * (COLS - 1)));
      line(ctx, pts.slice(0, k + 1), Math.min(1, a + 0.25), w + 0.2);
      line(ctx, pts.slice(k), a * 0.3, w);
    }
  }
  for (let j = 0; j < COLS; j += 2) {
    const pts = [];
    for (let i = 0; i < Math.min(ROWS, Math.ceil(rows)); i++) pts.push(P(j / (COLS - 1), 1 - i / (ROWS - 1), grid[ROWS - 1 - i][j] * Math.max(0, Math.min(1, rows - i))));
    if (pts.length > 1) line(ctx, pts, 0.16 * dim, 1);
  }
  if (playT != null) { ctx.globalAlpha = 0.75; ctx.fillRect(W / 2 + (playT - 0.5) * spread, base - height * 2.1, 1.5, height * 2.1); }
  ctx.globalAlpha = 1;
}

// The intro's picture at progress q (0..1): a flat line, then trembling, then a waveform shaped like the
// landscape's front row, which finally tips back into the full landscape.
export function drawIntro(ctx, W, H, q, now, grid = SAMPLE_GRID) {
  ctx.strokeStyle = LINE; ctx.lineJoin = "round"; ctx.lineCap = "round";
  const cy = H * 0.45, front = grid[0];
  if (q < 0.82) {
    const grow = Math.max(0, Math.min(1, (q - 0.18) / 0.5)), width = 160 + grow * Math.min(W * 0.55, 620), amp = 2 + grow * 46, n = 220, pts = [];
    for (let k = 0; k < n; k++) {
      const t = k / (n - 1), env = front[Math.min(COLS - 1, Math.floor(t * COLS))] * Math.sin(Math.PI * t);
      const wob = Math.sin(k * 0.9 + now / 70) * (0.25 + 0.75 * grow);
      pts.push([W / 2 - width / 2 + t * width, cy + wob * amp * (grow < 0.05 ? 0.15 : env + 0.04)]);
    }
    line(ctx, pts, 1, 1.6);
  } else {
    const k = Math.min(1, (q - 0.82) / 0.18);
    drawLandscape(ctx, W, H, grid, { base: cy + 60 + k * 40, height: Math.min(H * 0.2, 150), rows: 1 + k * (ROWS - 1) * 1.02 });
  }
  ctx.globalAlpha = 1;
}

// A live microphone trace with fading echoes behind it (samples: the latest Float32Array from an AnalyserNode).
export function drawMic(ctx, W, H, history) {
  ctx.strokeStyle = LINE; ctx.lineJoin = "round";
  const width = Math.min(W * 0.7, 820), cy = H * 0.48;
  history.forEach((samples, h) => {
    const back = history.length - 1 - h, n = 240, pts = [];
    for (let k = 0; k < n; k++) {
      const v = samples[Math.floor((k / n) * samples.length)] || 0;
      pts.push([W / 2 - width / 2 + (k / (n - 1)) * width + back * 6, cy - back * 16 + Math.max(-1, Math.min(1, v * 3)) * 60 * Math.sin((Math.PI * k) / (n - 1))]);
    }
    line(ctx, pts, back === 0 ? 1 : Math.max(0.08, 0.5 - back * 0.08), back === 0 ? 1.6 : 1);
  });
  ctx.globalAlpha = 1;
}

// Model results: charts drawn as SVG from data/results.json and the evaluation numbers below.
const $ = (id) => document.getElementById(id);
const { curves } = await (await fetch("data/results.json")).json();
const ORDER = ["ae", "vae", "fs", "diff", "gan", "base"];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(3) : v.toFixed(4));
const pct = (a, b) => { const c = ((b - a) / a) * 100; return (c < 0 ? "−" : "+") + Math.abs(c).toFixed(Math.abs(c) < 10 ? 1 : 0) + "%"; };
const tick = (v) => (v === 0 ? "0" : String(+v.toFixed(3)));
function niceTicks(lo, hi, n) {
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw)), e = raw / mag;
  const step = (e >= 5 ? 10 : e >= 2 ? 5 : e >= 1 ? 2 : 1) * mag, out = [];
  for (let v = Math.floor(lo / step) * step; v <= Math.ceil(hi / step) * step + step / 2; v += step) out.push(+v.toFixed(10));
  return out;
}

// ---------------------------------------------------------------- training curves
const S = { model: "fs", view: "chart" };
const X0 = 56, X1 = 710, Y0 = 12, Y1 = 268;
$("models").innerHTML = ORDER.map((k) => `<button type="button" class="chip" data-k="${k}">${esc(curves[k].label)}</button>`).join("");
$("models").onclick = (e) => { const b = e.target.closest("button"); if (b) { S.model = b.dataset.k; drawCurves(); } };
$("view-chart").onclick = () => { S.view = "chart"; drawCurves(); };
$("view-table").onclick = () => { S.view = "table"; drawCurves(); };

function drawCurves() {
  const m = curves[S.model], n = m.train.length;
  document.querySelectorAll("#models button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.k === S.model));
  $("view-chart").setAttribute("aria-pressed", S.view === "chart"); $("view-table").setAttribute("aria-pressed", S.view === "table");
  $("curve-chart").hidden = S.view !== "chart"; $("curve-table").hidden = S.view !== "table";
  $("curve-note").textContent = m.note;
  const series = [{ name: m.train_name, color: "var(--c2)", v: m.train }, { name: m.val_name, color: "var(--c1)", v: m.val }];
  $("curve-legend").innerHTML = series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("");
  const best = (a) => Math.min(...a);
  $("curve-table").innerHTML = `<table><thead><tr><th>Loss</th><th class="r">First epoch</th><th class="r">Last epoch</th><th class="r">Best</th><th class="r">Change</th></tr></thead><tbody>${series
    .map((s) => `<tr><td>${esc(s.name)}</td><td class="r">${fmt(s.v[0])}</td><td class="r">${fmt(s.v[n - 1])}</td><td class="r">${fmt(best(s.v))} (epoch ${s.v.indexOf(best(s.v)) + 1})</td><td class="r">${pct(s.v[0], s.v[n - 1])}</td></tr>`).join("")}</tbody></table>`;

  const all = m.train.concat(m.val), lo = Math.min(...all), hi = Math.max(...all);
  const yt = niceTicks(Math.max(0, lo - (hi - lo) * 0.05), hi, 4), ya = yt[0], yb = yt[yt.length - 1];
  const sx = (e) => X0 + ((e - 1) / (n - 1)) * (X1 - X0), sy = (v) => Y1 - ((v - ya) / (yb - ya)) * (Y1 - Y0);
  let ly = series.map((s) => sy(s.v[n - 1]) + 4);
  if (Math.abs(ly[0] - ly[1]) < 14) { const mid = (ly[0] + ly[1]) / 2; ly = ly[0] <= ly[1] ? [mid - 7, mid + 7] : [mid + 7, mid - 7]; }
  const step = n <= 50 ? 10 : 20, xt = [1];
  for (let e = step; e <= n; e += step) xt.push(e);
  const svg = `<svg viewBox="0 0 800 300" role="img" aria-label="${esc(`${m.label}: ${m.train_name} falls from ${fmt(m.train[0])} to ${fmt(m.train[n - 1])}, ${m.val_name} from ${fmt(m.val[0])} to ${fmt(m.val[n - 1])} over ${n} epochs`)}">
    ${yt.map((v) => `<line x1="${X0}" x2="${X1}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/><text x="48" y="${sy(v) + 4}" text-anchor="end" font-size="12" fill="var(--muted)">${tick(v)}</text>`).join("")}
    ${xt.map((e) => `<text x="${sx(e)}" y="292" text-anchor="middle" font-size="12" fill="var(--muted)">${e}</text>`).join("")}
    ${(m.marks || []).map((k) => `<line x1="${sx(k.epoch)}" x2="${sx(k.epoch)}" y1="${Y0}" y2="${Y1}" stroke="var(--muted)" stroke-dasharray="3 4"/><text x="${sx(k.epoch) + 6}" y="24" font-size="12" fill="var(--muted)">${esc(k.label)}</text>`).join("")}
    ${series.map((s, i) => `<path d="M${s.v.map((v, j) => `${sx(j + 1).toFixed(1)} ${sy(v).toFixed(1)}`).join("L")}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/><text x="718" y="${ly[i]}" font-size="12" fill="var(--ink)">${fmt(s.v[n - 1])}</text>`).join("")}
    <g id="hover" hidden><line id="hl" y1="${Y0}" y2="${Y1}" stroke="var(--ink)" stroke-opacity=".35"/>${series.map((s, i) => `<circle id="hd${i}" r="4.5" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`).join("")}</g>
    <rect x="${X0}" y="${Y0}" width="${X1 - X0}" height="${Y1 - Y0}" fill="transparent"/></svg><div class="tip" id="tip" hidden></div>`;
  const box = $("curve");
  box.innerHTML = svg;
  const el = box.querySelector("svg");
  el.onmousemove = (e) => {
    const r = el.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * 800;
    if (x < X0 - 10 || x > X1 + 10) return el.onmouseleave();
    const ep = Math.min(n, Math.max(1, Math.round(((x - X0) / (X1 - X0)) * (n - 1)) + 1)), hx = sx(ep);
    $("hover").hidden = false; $("hl").setAttribute("x1", hx); $("hl").setAttribute("x2", hx);
    series.forEach((s, i) => { $("hd" + i).setAttribute("cx", hx); $("hd" + i).setAttribute("cy", sy(s.v[ep - 1])); });
    const tip = $("tip");
    tip.innerHTML = `<div style="font-weight:600">Epoch ${ep}</div>${series.map((s) => `<div>${esc(s.name)}: ${fmt(s.v[ep - 1])}</div>`).join("")}`;
    tip.hidden = false; tip.style.left = `${(hx / 800) * 100}%`; tip.style.transform = hx > 560 ? "translateX(calc(-100% - 12px))" : "translateX(12px)";
  };
  el.onmouseleave = () => { $("hover").hidden = true; $("tip").hidden = true; };
}
drawCurves();

// ---------------------------------------------------------------- original against final
const pair = (title, note, a, b) => `<div style="display:flex;flex-direction:column;gap:10px"><span style="font-size:14px;font-weight:600">${title}</span>
  <div class="vbars">${[[a, "var(--c2)", "Original"], [b, "var(--c1)", "Final"]].map(([v, c, who]) => `<div title="${who}: ${v.toFixed(3)}"><b>${v.toFixed(3)}</b><span style="height:${Math.max(2, v * 140)}px;background:${c}"></span></div>`).join("")}</div>
  <span class="small muted">${note}</span></div>`;
$("pairs").innerHTML = pair("Word error rate (lower is better)", "Whisper, 7 test sentences", 0.96, 0.083) + pair("Voice similarity (higher is better)", "Output against the target voice", 0.629, 0.744);

$("drops").innerHTML = [["Speaker VAE", 83], ["GAN vocoder", 82], ["FastSpeech", 35], ["Diffusion", 32], ["Autoencoder", 16]]
  .map(([n, v]) => `<div class="hbar"><span>${n}</span><span class="track"><span class="fill" style="width:${v}%"></span></span><span style="text-align:right;font-weight:600" class="num">−${v}%</span></div>`).join("");

// ---------------------------------------------------------------- diffusion strength
const sxs = [0, 0.1, 0.2, 0.3, 0.5, 1.0], SX = (v) => 44 + v * 456, SY = (v) => 230 - (v / 0.8) * 220;
const lines = [{ name: "Retrained", c: "var(--c2)", y: [0.083, 0.101, 0.124, 0.135, 0.303, 0.796] }, { name: "Original", c: "var(--c1)", y: [0.083, 0.083, 0.124, 0.171, 0.153, 0.417] }];
$("strength").innerHTML = `<svg viewBox="0 0 520 260" role="img" aria-label="Word error rate against diffusion strength for two diffusion models">
  ${[0, 0.2, 0.4, 0.6, 0.8].map((v) => `<line x1="44" x2="500" y1="${SY(v)}" y2="${SY(v)}" stroke="var(--grid)"/><text x="36" y="${SY(v) + 4}" text-anchor="end" font-size="12" fill="var(--muted)">${tick(v)}</text>`).join("")}
  ${[0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => `<text x="${SX(v)}" y="252" text-anchor="middle" font-size="12" fill="var(--muted)">${tick(v)}</text>`).join("")}
  <line x1="${SX(0.1)}" x2="${SX(0.1)}" y1="10" y2="230" stroke="var(--muted)" stroke-dasharray="3 4"/><text x="${SX(0.1) + 6}" y="20" font-size="12" fill="var(--muted)">Used: 0.1</text>
  ${lines.map((l) => `<path d="M${l.y.map((y, i) => `${SX(sxs[i])} ${SY(y).toFixed(1)}`).join("L")}" fill="none" stroke="${l.c}" stroke-width="2" stroke-linejoin="round"/>${l.y.map((y, i) => `<circle cx="${SX(sxs[i])}" cy="${SY(y).toFixed(1)}" r="5" fill="${l.c}" stroke="var(--surface)" stroke-width="2"><title>${l.name} diffusion, strength ${sxs[i]}: WER ${y.toFixed(3)}</title></circle>`).join("")}`).join("")}
</svg>`;

// ---------------------------------------------------------------- model sizes
const sizes = [["FastSpeech-style Transformer", 27.26, "var(--c1)", "Text to spectrogram: the biggest job"], ["Speaker VAE", 8.12, "var(--c3)", "Voice to 64 numbers"], ["Other three models", 3.58, "var(--c2)", "Vocoder 2.2 MB, diffusion 1.2 MB, autoencoder 0.2 MB"]];
const total = sizes.reduce((a, s) => a + s[1], 0);
let a0 = 0;
const arc = (r0, r1, a, b) => { const p = (r, t) => [100 + r * Math.sin(t), 100 - r * Math.cos(t)], big = b - a > Math.PI ? 1 : 0, [x0, y0] = p(r1, a), [x1, y1] = p(r1, b), [x2, y2] = p(r0, b), [x3, y3] = p(r0, a);
  return `M${x0} ${y0}A${r1} ${r1} 0 ${big} 1 ${x1} ${y1}L${x2} ${y2}A${r0} ${r0} 0 ${big} 0 ${x3} ${y3}Z`; };
const slices = sizes.map(([name, mb, c, detail]) => { const a1 = a0 + (mb / total) * Math.PI * 2, d = arc(58, 92, a0, a1); a0 = a1; return { name, mb, c, detail, d, share: Math.round((mb / total) * 100) }; });
$("donut").innerHTML = `<svg viewBox="0 0 200 200" role="img" aria-label="Share of the 39 MB of model files: FastSpeech 70%, speaker VAE 21%, the other three models 9%" style="width:190px;height:190px;flex:none">
  ${slices.map((s) => `<path d="${s.d}" fill="${s.c}" stroke="var(--surface)" stroke-width="2"><title>${s.name}: ${s.mb.toFixed(1)} MB, ${s.share}%</title></path>`).join("")}
  <text x="100" y="98" text-anchor="middle" font-family="Unbounded, sans-serif" font-size="22" fill="var(--ink)">39 MB</text><text x="100" y="118" text-anchor="middle" font-size="12" fill="var(--muted)">five models</text></svg>
  <ul style="list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:12px;font-size:14px;flex:1;min-width:200px">
  ${slices.map((s) => `<li style="display:flex;gap:10px"><span aria-hidden="true" style="flex:none;margin-top:4px;width:12px;height:12px;border-radius:3px;background:${s.c}"></span><span><strong style="font-weight:600">${s.name}</strong> ${s.mb.toFixed(1)} MB (${s.share}%)<br><span class="muted">${s.detail}</span></span></li>`).join("")}</ul>`;

// ---------------------------------------------------------------- all losses
const rows = [
  ["Denoising autoencoder", "Train (L1 + MSE)", 0.0534, 0.0297], ["", "Validation", 0.0352, 0.0297],
  ["Speaker VAE", "Total loss", 1.3482, 0.0679], ["", "Reconstruction MSE", 0.028, 0.0099], ["", "KL divergence", 23.4527, 6.196], ["", "Speaker cross-entropy", 1.3191, 0.0549], ["", "Validation loss", 0.3801, 0.0628], ["", "Validation speaker accuracy", "89.42%", "98.61%", "+9.2 points"],
  ["FastSpeech-style Transformer", "Total loss", 4.2272, 2.5431], ["", "Mel L1", 0.121, 0.0605], ["", "Duration loss (log frames)", 0.3908, 0.137], ["", "Alignment (forward-sum) loss", 3.6908, 2.3391], ["", "Validation mel L1", 0.103, 0.0669], ["", "Validation duration loss", 0.3602, 0.1375],
  ["Diffusion refiner", "Noise-prediction MSE (train)", 0.1467, 0.0421], ["", "Validation", 0.0591, 0.0401],
  ["GAN vocoder", "Generator total", 40.8445, 21.9577], ["", "Discriminator", 0.4784, 0.0024], ["", "Log-mel L1 (train)", 0.8406, 0.345], ["", "Validation log-mel L1 (unseen voices)", 1.5002, 0.2744],
];
$("all").innerHTML = `<table><thead><tr><th>Model</th><th>Loss</th><th class="r">First epoch</th><th class="r">Last epoch</th><th class="r">Change</th></tr></thead><tbody>${rows
  .map(([m, l, f, t, c]) => `<tr><td style="font-weight:600;white-space:nowrap">${m}</td><td>${l}</td><td class="r">${typeof f === "number" ? f.toFixed(4) : f}</td><td class="r">${typeof t === "number" ? t.toFixed(4) : t}</td><td class="r">${c || pct(f, t)}</td></tr>`).join("")}</tbody></table>`;

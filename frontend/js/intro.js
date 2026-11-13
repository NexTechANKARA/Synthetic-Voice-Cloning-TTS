// The intro: a flat line becomes a voice while the models download, with a rolling 000-100 counter.
// It never runs ahead of the real download, and at the end the dark screen shrinks into `target`.
import { fitCanvas } from "./landscape.js";

const MIN_MS = 2400;

export function runIntro({ overlay, canvas, counter, status, skip, target, progress, draw, onReveal }) {
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  overlay.hidden = false;
  counter.innerHTML = [0, 1, 2].map(() => `<span class="col"><span class="strip">${[...Array(10).keys()].map((d) => `<span>${d}</span>`).join("")}</span></span>`).join("");
  const strips = counter.querySelectorAll(".strip");
  const t0 = performance.now();
  let shown = -1, leaving = false, scheduled = false;

  const finish = () => {
    if (leaving) return;
    leaving = true;
    setCount(100);
    status.textContent = progress.failed ? "Couldn’t download the voice models yet. You can still sign in." : "Ready";
    const r = target.getBoundingClientRect();
    overlay.style.clipPath = reduced ? "" : `inset(${r.top}px ${innerWidth - r.right}px ${innerHeight - r.bottom}px ${r.left}px round 28px)`;
    overlay.style.opacity = "0";
    onReveal();
    setTimeout(() => { overlay.hidden = true; }, reduced ? 300 : 1400);
  };
  skip.onclick = finish;

  function setCount(n) {
    if (n === shown) return;
    shown = n;
    String(n).padStart(3, "0").split("").forEach((d, i) => (strips[i].style.transform = `translateY(-${d}em)`));
  }

  (function loop(now) {
    if (overlay.hidden) return;
    const q = leaving ? 1 : Math.min((now - t0) / MIN_MS, progress.failed ? 1 : progress.fraction);
    setCount(Math.round(q * 100));
    if (!leaving) status.textContent = q < 0.25 && !progress.mb ? "Loading the page"
      : progress.fraction < 1 ? `Downloading the voice models${progress.mb ? ", " + progress.mb : ""}` : "Getting the models ready";
    const { ctx, W, H } = fitCanvas(canvas);
    if (!reduced) draw(ctx, W, H, q, now);
    if (q >= 1 && !leaving && !scheduled) { scheduled = true; setTimeout(finish, 350); }
    requestAnimationFrame(loop);
  })(t0);
}

// Sign-in page: the intro (while the voice models download), then email or Google sign-in.
import { client, friendlyError } from "./auth.js";
import { createEngine } from "./engine.js";
import { SAMPLE_GRID, fitCanvas, drawLandscape, drawIntro } from "./landscape.js";
import { runIntro } from "./intro.js";

const $ = (id) => document.getElementById(id);
const recovering = /type=recovery/.test(location.hash);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

// Signed-in visitors skip all of this and go straight to the Studio.
const { data } = await client.auth.getSession();
if (data.session && !recovering) location.replace("studio.html");
else start();

function start() {
  // the breathing landscape on the left
  const cv = $("stage-canvas");
  (function loop(now) {
    const { ctx, W, H } = fitCanvas(cv);
    drawLandscape(ctx, W, H, SAMPLE_GRID, { base: H - 40, height: W < 600 ? Math.min(H * 0.13, 60) : Math.min(H * 0.22, 130), amp: reduced ? 1 : 1 + 0.04 * Math.sin(now / 900), spread: Math.min(W * 1.05, 1000) });
    requestAnimationFrame(loop);
  })(performance.now());

  // download the models now so the Studio opens fast; the intro counter shows the real progress
  const progress = { fraction: 0, mb: "", failed: false };
  if (typeof ort === "undefined") {
    progress.failed = true;                                    // the model library didn't load: sign-in still works
  } else {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    createEngine("models/", { onProgress: (l, t) => { progress.fraction = t ? l / t : 0; progress.mb = `${(l / 1e6).toFixed(0)} of ${(t / 1e6).toFixed(0)} MB`; } })
      .then(() => { progress.fraction = 1; })
      .catch(() => { progress.failed = true; });
  }

  let seen = false;
  try { seen = sessionStorage.getItem("vc-intro") === "1"; } catch {}
  const reveal = () => document.body.classList.remove("waiting");
  if (seen || recovering) reveal();
  else {
    try { sessionStorage.setItem("vc-intro", "1"); } catch {}
    runIntro({ overlay: $("intro"), canvas: $("intro-canvas"), counter: $("counter"), status: $("intro-status"), skip: $("skip"),
      target: $("stage"), progress, draw: drawIntro, onReveal: reveal });
  }
  setUpForms();
}

function setUpForms() {
  let mode = "signin";
  const show = (id) => ["view-auth", "view-reset", "view-newpass", "view-message"].forEach((v) => ($(v).hidden = v !== id));
  const message = (title, text, { studio = false, back = false } = {}) => {
    $("msg-title").textContent = title; $("msg-text").textContent = text;
    $("msg-action").hidden = !studio; $("msg-back").hidden = !back; show("view-message");
  };
  const busy = (btn, on, label) => { btn.disabled = on; btn.classList.toggle("busy", on); if (label) btn.textContent = label; };
  const studioUrl = new URL("studio.html", location.href).href;

  if (recovering) show("view-newpass");

  const setMode = (m) => {
    mode = m;
    const signin = m === "signin";
    $("auth-title").textContent = signin ? "Welcome back" : "Create your account";
    $("auth-lead").textContent = signin ? "Sign in to use the Studio." : "Free, and it takes under a minute.";
    $("submit").textContent = signin ? "Sign in" : "Create account";
    $("switch-text").textContent = signin ? "New here?" : "Already have an account?";
    $("switch").textContent = signin ? "Create an account" : "Sign in";
    $("forgot").hidden = !signin;
    $("password").autocomplete = signin ? "current-password" : "new-password";
    $("pass-help").textContent = signin ? "" : "At least 8 characters.";
    $("auth-error").hidden = true;
  };
  $("switch").onclick = () => setMode(mode === "signin" ? "create" : "signin");
  $("forgot").onclick = () => { $("reset-email").value = $("email").value; show("view-reset"); $("reset-email").focus(); };
  document.querySelectorAll(".back").forEach((b) => (b.onclick = () => { show("view-auth"); $("email").focus(); }));
  document.querySelector(".pass .show").onclick = (e) => {
    const on = $("password").type === "password";
    $("password").type = on ? "text" : "password"; e.target.textContent = on ? "Hide" : "Show"; e.target.setAttribute("aria-pressed", on);
  };

  const emailError = (v) => (!v ? "Enter your email address." : /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v) ? "" : "That doesn’t look like an email address. Check for a missing @ or dot.");
  $("auth-form").onsubmit = async (e) => {
    e.preventDefault();
    const email = $("email").value.trim(), password = $("password").value;
    const ee = emailError(email);
    const pe = !password ? "Enter your password." : mode === "create" && password.length < 8 ? "Use at least 8 characters." : "";
    $("email-err").textContent = ee; $("email-err").hidden = !ee; $("email").setAttribute("aria-invalid", !!ee);
    $("pass-help").textContent = pe || (mode === "create" ? "At least 8 characters." : "");
    $("pass-help").className = pe ? "error" : "small muted"; $("password").setAttribute("aria-invalid", !!pe);
    if (ee || pe) return;
    const btn = $("submit");
    busy(btn, true, mode === "signin" ? "Signing in…" : "Creating account…");
    const res = mode === "signin"
      ? await client.auth.signInWithPassword({ email, password })
      : await client.auth.signUp({ email, password, options: { emailRedirectTo: studioUrl } });
    busy(btn, false, mode === "signin" ? "Sign in" : "Create account");
    if (res.error) { $("auth-error").textContent = friendlyError(res.error); $("auth-error").hidden = false; return; }
    if (res.data.session) location.replace("studio.html");
    else message("Check your email", `We sent a link to ${email}. Open it to confirm your account, and the Studio will open.`, { back: true });
  };

  $("google").onclick = async () => {
    const { error } = await client.auth.signInWithOAuth({ provider: "google", options: { redirectTo: studioUrl } });
    if (error) { $("auth-error").textContent = friendlyError(error); $("auth-error").hidden = false; }
  };

  $("reset-form").onsubmit = async (e) => {
    e.preventDefault();
    const email = $("reset-email").value.trim(), err = emailError(email);
    if (err) { $("reset-error").textContent = err; $("reset-error").hidden = false; return; }
    const { error } = await client.auth.resetPasswordForEmail(email, { redirectTo: new URL("index.html", location.href).href });
    if (error) { $("reset-error").textContent = friendlyError(error); $("reset-error").hidden = false; return; }
    message("Check your email", `If there’s an account for ${email}, we’ve sent a link to choose a new password.`, { back: true });
  };

  $("newpass-form").onsubmit = async (e) => {
    e.preventDefault();
    const password = $("newpass").value;
    if (password.length < 8) { $("newpass-error").textContent = "Use at least 8 characters."; $("newpass-error").hidden = false; return; }
    const { error } = await client.auth.updateUser({ password });
    if (error) { $("newpass-error").textContent = friendlyError(error); $("newpass-error").hidden = false; return; }
    history.replaceState(null, "", "index.html");
    message("Password saved", "You’re signed in with your new password.", { studio: true });
  };
}

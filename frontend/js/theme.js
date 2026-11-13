// Light or dark: the visitor's choice (Auto, Light, Dark), remembered in this browser.
// Loaded in <head> before the page draws, so it never flashes the wrong colours.
// Fills every <div class="theme"> on the page with the three-way switch.
(function () {
  var KEY = "vc-theme", root = document.documentElement, mq = matchMedia("(prefers-color-scheme: dark)");
  function choice() { try { return localStorage.getItem(KEY) || "auto"; } catch (e) { return "auto"; } }
  function apply() {
    var c = choice();
    root.dataset.theme = c === "dark" || (c === "auto" && mq.matches) ? "dark" : "light";
    document.querySelectorAll(".theme button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.choice === c); });
  }
  apply();
  mq.addEventListener("change", apply);

  var ICONS = {
    auto: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M8 2a6 6 0 0 1 0 12z" fill="currentColor"/></svg>',
    light: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="3"/><path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1"/></svg>',
    dark: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>',
  };
  var LABELS = { auto: "Match my device", light: "Light", dark: "Dark" };
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".theme").forEach(function (box) {
      box.setAttribute("role", "group");
      box.setAttribute("aria-label", "Colour theme");
      box.innerHTML = ["auto", "light", "dark"].map(function (c) {
        return '<button type="button" data-choice="' + c + '" aria-label="' + LABELS[c] + '" title="' + LABELS[c] + '">' + ICONS[c] + "</button>";
      }).join("");
      box.onclick = function (e) {
        var b = e.target.closest("button");
        if (!b) return;
        try { localStorage.setItem(KEY, b.dataset.choice); } catch (err) {}
        apply();
      };
    });
    apply();
  });
})();

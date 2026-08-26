/* Neural Options Lab - dashboard logic
   Talks to the FastAPI backend, renders Plotly charts, animates numbers. */

"use strict";

// ─────────────────────────────────────────────── state & element handles ──
const $ = (id) => document.getElementById(id);

const state = {
  spot: 100, strike: 100, maturity: 1.0, sigma: 0.25, rate: 0.04,
  optionType: "call", mcPaths: 50000,
};

// Desaturated institutional palette: blue = neural/deep,
// amber = classical benchmarks (MC, BS delta), gray = secondary.
const COLORS = {
  nn: "#5a8cc8", mc: "#c4835c", violet: "#8891a3",
  ink: "#7d8a9e", grid: "rgba(255,255,255,0.06)",
  good: "#5a9e78", warn: "#c45c5c",
};

const PLOT_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, -apple-system, SF Pro Text, sans-serif", color: COLORS.ink,
          size: 11.5 },
  margin: { l: 52, r: 16, t: 12, b: 42 },
  showlegend: true,
  legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11 } },
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

// ───────────────────────────────────────────────────────────── utilities ──
const debounce = (fn, ms) => {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

const fmtMoney = (v) => "$" + v.toFixed(4);
const fmtMs = (ms) => ms >= 1000 ? (ms / 1000).toFixed(2) + " s"
  : ms >= 10 ? ms.toFixed(0) + " ms"
  : ms >= 1 ? ms.toFixed(1) + " ms"
  : (ms * 1000).toFixed(0) + " µs";

// Tween a numeric readout for that premium feel. Falls back to setting the
// value directly when the tab is hidden (rAF is throttled there).
const tweens = new Map();
function animateNumber(el, target, format) {
  const start = tweens.has(el) ? tweens.get(el) : target;
  tweens.set(el, target);
  if (document.hidden) { el.textContent = format(target); return; }
  const t0 = performance.now(), dur = 380;
  let finished = false;
  const step = (now) => {
    if (tweens.get(el) !== target) return;
    const p = Math.min((now - t0) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = format(start + (target - start) * ease);
    if (p < 1) requestAnimationFrame(step); else finished = true;
  };
  requestAnimationFrame(step);
  setTimeout(() => {
    if (!finished && tweens.get(el) === target) el.textContent = format(target);
  }, dur + 150);
}

function clearShimmer(plotId) {
  const shim = $(plotId).querySelector(".shimmer");
  if (shim) shim.remove();
}

function optionBody() {
  return {
    spot: state.spot, strike: state.strike, maturity: state.maturity,
    sigma: state.sigma, rate: state.rate, option_type: state.optionType,
  };
}

// ─────────────────────────────────────────────────────────────── controls ──
function bindSlider(id, onChange) {
  const el = $("in-" + id);
  const paint = () => {
    const pct = (el.value - el.min) / (el.max - el.min) * 100;
    el.style.setProperty("--fill", pct + "%");
  };
  el.addEventListener("input", () => { paint(); onChange(parseFloat(el.value)); });
  paint();
}

// Maturities at or below 12 trading days route to the 0DTE rough-vol
// surrogate, which is trained on a narrower moneyness band.
const ZERO_DTE_CUTOFF = 12 / 252;
const is0dte = () => state.maturity <= ZERO_DTE_CUTOFF + 1e-9;

function refreshReadouts() {
  $("val-spot").textContent = state.spot;
  $("val-strike").textContent = state.strike;
  $("val-maturity").textContent = is0dte()
    ? Math.max(1, Math.round(state.maturity * 252)) + "d"
    : state.maturity.toFixed(2) + "y";
  $("val-sigma").textContent = Math.round(state.sigma * 100) + "%";
  $("val-rate").textContent = (state.rate * 100).toFixed(1) + "%";
  const m = state.spot / state.strike;
  $("moneyness-val").textContent = m.toFixed(2) +
    (is0dte() ? " · 0DTE" : "");
  const [lo, hi] = is0dte() ? [0.85, 1.15] : [0.5, 2.0];
  const outside = m < lo || m > hi;
  $("domain-warning").textContent = "Outside " + (is0dte() ? "0DTE rough-vol" : "trained") + " domain [" + lo + ", " + hi + "] (surrogate is extrapolating)";
    (is0dte() ? "0DTE rough-vol" : "trained") + " domain [" + lo + ", " +
    hi + "] (surrogate is extrapolating)";
  $("domain-warning").classList.toggle("show", outside);
}

function bindSegmented(containerId, onPick) {
  const box = $(containerId);
  box.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      box.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      onPick(btn.dataset.value);
    });
  });
}

// ───────────────────────────────────────────────────────── price + greeks ──
let priceSeq = 0;
let lastNNPrice = null;
async function updatePrice() {
  document.querySelector(".results").classList.add("updating");
  const seq = ++priceSeq;
  try {
    const d = await api("/api/price", { ...optionBody(), mc_paths: state.mcPaths });
    if (seq !== priceSeq) return; // a newer request superseded this one

    lastNNPrice = d.nn.price;
    animateNumber($("nn-price"), d.nn.price, fmtMoney);
    animateNumber($("mc-price"), d.mc.price, fmtMoney);
    $("nn-latency").textContent = fmtMs(d.nn.latency_ms) +
      (is0dte() ? " · 0DTE net" : "");
    $("mc-latency").textContent = fmtMs(d.mc.latency_ms) +
      " · " + d.mc.n_paths.toLocaleString() + " paths" +
      (d.mc.engine === "rough_bergomi" ? " · rBergomi" : "");
    $("mc-ci").textContent = "95% CI  [" + d.mc.ci_low.toFixed(4) +
      ", " + d.mc.ci_high.toFixed(4) + "]";

    animateNumber($("speedup"), d.comparison.speedup,
      (v) => v >= 100 ? Math.round(v).toLocaleString() + "×" : v.toFixed(1) + "×");
    const bps = d.comparison.diff_bps_of_spot;
    const ok = bps < 10; // surrogate tolerance: 10 bps of spot
    const agr = $("agreement");
    agr.textContent = "Δ " + bps.toFixed(1) +
      " bps of spot vs MC" + (d.comparison.within_mc_ci ? " · inside 95% CI" : "");
    agr.className = "card-sub centered " + (ok ? "agreement-ok" : "agreement-warn");

    const g = d.nn.greeks;
    animateNumber($("g-delta"), g.delta, (v) => v.toFixed(4));
    animateNumber($("g-gamma"), g.gamma, (v) => v.toFixed(4));
    animateNumber($("g-vega"), g.vega, (v) => v.toFixed(4));
    animateNumber($("g-theta"), g.theta, (v) => v.toFixed(4));
    animateNumber($("g-rho"), g.rho, (v) => v.toFixed(4));
  } catch (err) {
    $("agreement").textContent = err.message;
    $("agreement").className = "card-sub centered agreement-warn";
  } finally {
    if (seq === priceSeq) document.querySelector(".results").classList.remove("updating");
  }
}

// ──────────────────────────────────────────────────────── convergence plot ──
async function updateConvergence() {
  const d = await api("/api/convergence", optionBody());
  clearShimmer("plot-convergence");

  const xs = d.mc_points.map((p) => p.n_paths);
  const traces = [
    { // CI band (upper then lower with fill)
      x: [...xs, ...xs.slice().reverse()],
      y: [...d.mc_points.map((p) => p.ci_high),
          ...d.mc_points.map((p) => p.ci_low).reverse()],
      fill: "toself", fillcolor: "rgba(255,92,168,0.12)",
      line: { width: 0 }, hoverinfo: "skip",
      name: "MC 95% CI", showlegend: true,
    },
    {
      x: xs, y: d.mc_points.map((p) => p.price),
      mode: "lines+markers",
      name: d.engine === "rough_bergomi"
        ? "Monte Carlo (rough Bergomi)" : "Monte Carlo",
      line: { color: COLORS.mc, width: 2.5, shape: "spline" },
      marker: { size: 7, color: COLORS.mc },
      customdata: d.mc_points.map((p) => fmtMs(p.latency_ms)),
      hovertemplate: "%{x:,} paths → $%{y:.4f}<br>%{customdata}<extra></extra>",
    },
    {
      x: [xs[0], xs[xs.length - 1]], y: [d.nn.price, d.nn.price],
      mode: "lines", name: "Neural net (" + fmtMs(d.nn.latency_ms) + ")",
      line: { color: COLORS.nn, width: 2.5, dash: "dash" },
      hovertemplate: "NN: $%{y:.4f}<extra></extra>",
    },
    {
      x: [xs[0], xs[xs.length - 1]],
      y: [d.reference.price, d.reference.price],
      mode: "lines",
      name: `Reference (${Math.round(d.reference.n_paths / 1000)}k paths)`,
      line: { color: "rgba(255,255,255,0.45)", width: 1.5, dash: "dot" },
      hovertemplate: "Reference: $%{y:.4f}<extra></extra>",
    },
  ];
  Plotly.react("plot-convergence", traces, {
    ...PLOT_BASE,
    xaxis: { type: "log", title: { text: "Monte Carlo paths" },
             gridcolor: COLORS.grid, zeroline: false },
    yaxis: { title: { text: "option price" }, gridcolor: COLORS.grid,
             zeroline: false, tickformat: ".3f" },
  }, PLOT_CONFIG);
}

// ─────────────────────────────────────────────────────────── latency plot ──
async function updateBenchmark() {
  const btn = $("btn-benchmark");
  btn.disabled = true;
  try {
    const d = await api("/api/benchmark", optionBody());
    clearShimmer("plot-latency");
    // Remove the "click Re-run" hint the initial-load block leaves in this
    // panel; Plotly renders into the same div without clearing it.
    $("plot-latency").querySelector(".latency-hint")?.remove();

    const rows = [
      ...d.mc.map((r) => ({ ...r, color: COLORS.mc })),
      ...d.nn.map((r) => ({ ...r, color: COLORS.nn })),
    ].sort((a, b) => b.latency_ms - a.latency_ms);

    Plotly.react("plot-latency", [{
      type: "bar", orientation: "h",
      y: rows.map((r) => r.label),
      x: rows.map((r) => Math.max(r.latency_ms, 0.001)),
      marker: { color: rows.map((r) => r.color), opacity: 0.85 },
      text: rows.map((r) => fmtMs(r.latency_ms)),
      textposition: "outside", textfont: { family: "JetBrains Mono", size: 11 },
      cliponaxis: false,
      hovertemplate: "%{y}: %{text}<extra></extra>",
    }], {
      ...PLOT_BASE, showlegend: false,
      margin: { l: 150, r: 60, t: 12, b: 42 },
      xaxis: { type: "log", title: { text: "wall-clock (ms, log)" },
               gridcolor: COLORS.grid, zeroline: false },
      yaxis: { gridcolor: "rgba(0,0,0,0)", automargin: true },
    }, PLOT_CONFIG);
  } finally { btn.disabled = false; }
}

// ─────────────────────────────────────────────────────────── surface plot ──
async function updateSurface() {
  const d = await api("/api/surface", {
    sigma: state.sigma, rate: state.rate, strike: state.strike,
    option_type: state.optionType,
  });
  clearShimmer("plot-surface");

  $("surface-stat").textContent =
    d.n_prices.toLocaleString() + " prices in " + fmtMs(d.latency_ms) +
    " · " + Math.round(d.prices_per_second / 1000).toLocaleString() + "k prices/s";

  const norm = d.prices.map((row) => row.map((v) => v / state.strike));
  Plotly.react("plot-surface", [{
    type: "surface", x: d.moneyness, y: d.maturity, z: norm,
    colorscale: [[0, "#0e1117"], [0.45, "#2a4a6b"], [0.75, "#5a8cc8"], [1, "#8891a3"]],
    showscale: false,
    contours: { z: { show: true, usecolormap: true,
                     highlightcolor: "#fff", project: { z: true } } },
    hovertemplate: "S/K %{x:.2f} · T %{y:.2f}y<br>price/K %{z:.4f}<extra></extra>",
    lighting: { specular: 0.4, roughness: 0.6 },
  }], {
    ...PLOT_BASE, showlegend: false,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      xaxis: { title: "moneyness S/K", gridcolor: COLORS.grid,
               color: COLORS.ink, showbackground: false },
      yaxis: { title: "maturity (y)", gridcolor: COLORS.grid,
               color: COLORS.ink, showbackground: false },
      zaxis: { title: "price / K", gridcolor: COLORS.grid,
               color: COLORS.ink, showbackground: false },
      camera: { eye: { x: -1.55, y: -1.6, z: 0.65 } },
    },
  }, PLOT_CONFIG);
}

// ────────────────────────────────────────────────── error-distribution plot ──
// Units are 1e-4 of the quantity: price errors are bps of strike; delta and
// vega errors are x10^-4 (per unit vol for vega).
const ERROR_METRA = {
  price: { label: "pricing error (bps of strike)", unit: "bps" },
  delta: { label: "delta error (×10⁻⁴)", unit: "×10⁻⁴" },
  vega: { label: "vega error (×10⁻⁴ per unit σ)", unit: "×10⁻⁴" },
};
let errorReport = null;
let errorMetric = "price";

function renderErrorDistribution() {
  const d = errorReport;
  if (!d) return;
  const meta = ERROR_METRA[errorMetric];
  const single = d.errors[errorMetric].single;
  const ens = d.errors[errorMetric].ensemble;

  $("error-sub").textContent =
    "signed " + errorMetric + " error on " + d.n_points.toLocaleString() +
    " independent test points vs " + (d.ref_paths / 1000).toFixed(0) +
    "k-path Monte Carlo references" +
    (d.differential_ml ? " · trained with Differential ML" : "");
  $("error-stat").textContent =
    "RMSE " + d.single[errorMetric].rmse_bps.toFixed(1) + " → " +
    d.ensemble[errorMetric].rmse_bps.toFixed(1) + " " + meta.unit +
    " · " + d.n_members + "-model ensemble";

  // Shared bins so the two histograms are directly comparable.
  const all = [...single, ...ens];
  const span = Math.max(Math.abs(Math.min(...all)), Math.abs(Math.max(...all)));
  const binSize = (2 * span) / 46;

  const traces = [
    {
      type: "histogram", x: single,
      name: "single model · RMSE " +
        d.single[errorMetric].rmse_bps.toFixed(1) + " " + meta.unit,
      marker: { color: "rgba(143,123,255,0.5)",
                line: { color: COLORS.violet, width: 1 } },
      xbins: { start: -span, end: span, size: binSize },
    },
    {
      type: "histogram", x: ens,
      name: "ensemble · RMSE " +
        d.ensemble[errorMetric].rmse_bps.toFixed(1) + " " + meta.unit,
      marker: { color: "rgba(90,140,200,0.45)",
                 line: { color: COLORS.nn, width: 1 } },
      xbins: { start: -span, end: span, size: binSize },
    },
  ];
  Plotly.react("plot-errors", traces, {
    ...PLOT_BASE, barmode: "overlay",
    xaxis: { title: { text: meta.label },
             gridcolor: COLORS.grid, zeroline: false },
    yaxis: { title: { text: "test points" }, gridcolor: COLORS.grid,
             zeroline: false },
    shapes: [{ type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper",
               line: { color: "rgba(255,255,255,0.35)", width: 1.5,
                       dash: "dot" } }],
  }, PLOT_CONFIG);
}

async function loadErrorDistribution() {
  try {
    errorReport = await api("/api/error-distribution");
    clearShimmer("plot-errors");
    renderErrorDistribution();
  } catch (err) {
    $("error-sub").textContent = err.message;
  }
}

// ──────────────────────────────────────────────────────────── model badge ──
async function loadModelInfo() {
  const dot = $("status-dot"), txt = $("model-badge-text");
  try {
    const health = await api("/api/health");
    if (!health.model_loaded) {
      dot.className = "status-dot bad";
      txt.textContent = "model not trained: run python -m backend.quant.train";
      return;
    }
    const m = await api("/api/model-info");
    dot.className = "status-dot ok";
    const members = m.n_members > 1 ? m.n_members + "× " : "";
    const acc = m.eval
      ? "RMSE " + m.eval.ensemble.price.rmse_bps.toFixed(1) + " bps vs " +
        (m.eval.ref_paths / 1000).toFixed(0) + "k-path MC"
      : "val RMSE " + m.val_rmse_bps_of_strike.toFixed(1) + " bps";
    txt.textContent = members + m.n_parameters.toLocaleString() +
      " params" + (m.differential_ml ? " · Differential ML" : "") +
      " · " + acc + " · " +
      m.n_samples.toLocaleString() + " MC-labelled samples";
  } catch {
    dot.className = "status-dot bad";
    txt.textContent = "backend unreachable";
  }
}

// ─────────────────────────────────────────────────────────────── wire up ──
const refreshFast = debounce(updatePrice, 220);
const refreshSlow = debounce(() => { updateConvergence(); updateSurface(); updateXAI(); }, 650);
const refreshAll = () => { refreshReadouts(); refreshFast(); refreshSlow(); };

bindSlider("spot", (v) => { state.spot = v; refreshAll(); });
bindSlider("strike", (v) => { state.strike = v; refreshAll(); });
bindSlider("maturity", (v) => { state.maturity = v; refreshAll(); });
bindSlider("sigma", (v) => { state.sigma = v / 100; refreshAll(); });
bindSlider("rate", (v) => { state.rate = v / 100; refreshAll(); });

bindSegmented("option-type", (v) => { state.optionType = v; refreshAll(); });
bindSegmented("mc-paths", (v) => { state.mcPaths = parseInt(v); refreshFast(); });
bindSegmented("error-metric", (v) => { errorMetric = v; renderErrorDistribution(); });

$("btn-benchmark").addEventListener("click", updateBenchmark);

// ───────────────────────────────────────────────────────────── Ticker API ──
// The pricer works in moneyness, so any spot level is exact - we rescale the
// spot/strike sliders around the live price instead of clamping into the
// demo range, and set the strike at-the-money.
let marketData = null;

function rescaleSpotSliders(spot) {
  const step = spot >= 500 ? 5 : spot >= 100 ? 1 : spot >= 20 ? 0.5 : 0.1;
  const lo = Math.ceil((0.55 * spot) / step) * step;
  const hi = Math.floor((1.95 * spot) / step) * step;
  const atm = Math.round(spot / step) * step;
  for (const id of ["in-spot", "in-strike"]) {
    const el = $(id);
    el.min = lo; el.max = hi; el.step = step; el.value = atm;
  }
}

async function fetchTicker() {
  const t = $("in-ticker").value.trim().toUpperCase();
  if (!t) return;
  const btn = $("btn-fetch-ticker");
  const chip = $("market-chip");
  btn.textContent = "..."; btn.disabled = true;
  try {
    const d = await api("/api/market/" + encodeURIComponent(t));
    marketData = d;
    rescaleSpotSliders(d.spot);
    $("in-sigma").value = d.sigma * 100;
    $("in-rate").value = d.rate * 100;
    for (const id of ["in-spot", "in-strike", "in-sigma", "in-rate"])
      $(id).dispatchEvent(new Event("input"));

    chip.innerHTML =
      "<b>" + d.ticker + "</b> $" + d.spot.toLocaleString(undefined,
        { maximumFractionDigits: 2 }) +
      " · σ̂<sub>1y</sub> " + (d.sigma_raw * 100).toFixed(1) + "%" +
      " · r " + (d.rate_raw * 100).toFixed(2) + "% (" + d.rate_source + ")" +
      "<br>as of " + d.as_of +
      (d.clamped ? " · <span class='warn'>clamped to trained domain</span>" : "");
    chip.classList.add("show");
  } catch (err) {
    marketData = null;
    chip.innerHTML = "<span class='warn'>" + err.message + "</span>";
    chip.classList.add("show");
  } finally {
    btn.textContent = "Fetch"; btn.disabled = false;
  }
}
$("btn-fetch-ticker").addEventListener("click", fetchTicker);
$("in-ticker").addEventListener("keydown", (e) => {
  if (e.key === "Enter") fetchTicker();
});

// ───────────────────────────────────────────────────────────── Tabs ──
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.style.display = "none");
    btn.classList.add("active");
    $(btn.dataset.tab).style.display = "flex";
    // charts drawn or window-resized while this pane was hidden need a nudge
    requestAnimationFrame(() => {
      document.querySelectorAll("#" + btn.dataset.tab + " .js-plotly-plot")
        .forEach((p) => Plotly.Plots.resize(p));
    });
  });
});

// ───────────────────────────────────────────────────────────── XAI ──
let lastAttributions = null;
async function updateXAI() {
  try {
    const d = await api("/api/explain", optionBody());
    clearShimmer("plot-xai");
    lastAttributions = d.attributions;

    const bT = d.baseline.maturity;
    $("xai-sub").textContent = "Integrated Gradients vs an ATM " +
      "minimal-option baseline (T=" + (bT < 13 / 252
        ? Math.round(bT * 252) + "d" : bT + "y") + ", σ=5%, r=0)" +
      (d.regime === "0dte_rough_bergomi" ? " · 0DTE rough-vol regime" : "");

    const rows = [
      { name: "Spot (moneyness)", v: d.attributions.spot },
      { name: "Maturity", v: d.attributions.maturity },
      { name: "Volatility", v: d.attributions.sigma },
      { name: "Rate", v: d.attributions.rate },
    ].sort((a, b) => Math.abs(a.v) - Math.abs(b.v));

    $("xai-stat").textContent =
      "baseline $" + d.baseline_price.toFixed(2) +
      " + Σ attributions → $" + d.target_price.toFixed(2) +
      " · completeness err " + Math.abs(d.completeness_error).toFixed(4);

    Plotly.react("plot-xai", [{
      type: "bar", orientation: "h",
      y: rows.map((r) => r.name),
      x: rows.map((r) => r.v),
      marker: { color: rows.map((r) => r.v >= 0
        ? "rgba(90,140,200,0.7)" : "rgba(196,92,92,0.7)") },
      text: rows.map((r) => (r.v >= 0 ? "+" : "−") + "$" +
        Math.abs(r.v).toFixed(3)),
      textposition: "outside",
      textfont: { family: "JetBrains Mono", size: 12 },
      cliponaxis: false,
      hovertemplate: "%{y}: %{x:+.4f}<extra></extra>",
    }], {
      ...PLOT_BASE, showlegend: false,
      margin: { l: 130, r: 70, t: 10, b: 40 },
      xaxis: { title: { text: "contribution to price ($)" },
               gridcolor: COLORS.grid, zeroline: true,
               zerolinecolor: "rgba(255,255,255,0.25)" },
      yaxis: { gridcolor: "rgba(0,0,0,0)", automargin: true },
    }, PLOT_CONFIG);
  } catch (err) {
    $("xai-sub").textContent = err.message;
  }
}

// ───────────────────────────────────────────────────────────── Hedging ──
// P&L is in strike units; scale by the current strike into dollars.
let lastHedge = null;
state.hedgeCost = 0.005;

$("in-cost").addEventListener("input", () => {
  state.hedgeCost = parseFloat($("in-cost").value) / 10000;
  $("val-cost").textContent = $("in-cost").value + " bps";
});

function hedgeStatChip(k, v, cls) {
  return "<div class='hedge-stat'><span class='k'>" + k +
    "</span><span class='v" + (cls ? " " + cls : "") + "'>" + v + "</span></div>";
}

async function runHedge() {
  const btn = $("btn-hedge");
  btn.textContent = "Simulating...";
  btn.disabled = true;
  try {
    const d = await api("/api/hedge",
      { sigma: state.sigma, rate: state.rate, cost: state.hedgeCost });
    clearShimmer("plot-hedge");
    clearShimmer("plot-holdings");
    lastHedge = d;
    const K = state.strike;
    const $$ = (v) => (v < 0 ? "−$" : "$") + Math.abs(v * K).toFixed(2);

    const improvement = (1 - d.deep.cvar95 / Math.max(d.delta.cvar95, 1e-9)) * 100;
    $("hedge-stats").innerHTML =
      hedgeStatChip("CVaR₉₅ deep hedge", $$(-d.deep.cvar95), "good") +
      hedgeStatChip("CVaR₉₅ delta hedge", $$(-d.delta.cvar95)) +
      hedgeStatChip("Tail-risk reduction",
        improvement.toFixed(0) + "%", improvement > 0 ? "good" : "") +
      hedgeStatChip("Avg costs deep vs delta",
        $$(d.deep.mean_costs) + " vs " + $$(d.delta.mean_costs));
    $("hedge-sub").textContent =
      d.n_paths.toLocaleString() + " simulated 30-day paths · short ATM call (premium " +
      $$(d.premium) + ") · σ " + (d.sigma * 100).toFixed(0) + "% · r " +
      (d.rate * 100).toFixed(1) + "% · cost " + (d.cost * 10000).toFixed(0) + " bps" +
      (d.clamped ? " · params clamped to hedger's trained box" : "");

    const allPnl = [...d.deep.pnl, ...d.delta.pnl].map((v) => v * K);
    const span = Math.max(Math.abs(Math.min(...allPnl)), Math.abs(Math.max(...allPnl)));
    const binSize = (2 * span) / 60;

    Plotly.react("plot-hedge", [
      {
        type: "histogram", x: d.delta.pnl.map((v) => v * K),
        name: "delta hedge · CVaR₉₅ " + $$(-d.delta.cvar95),
        marker: { color: "rgba(255,92,168,0.45)",
                  line: { color: COLORS.mc, width: 1 } },
        xbins: { start: -span, end: span, size: binSize },
      },
      {
        type: "histogram", x: d.deep.pnl.map((v) => v * K),
        name: "deep hedge · CVaR₉₅ " + $$(-d.deep.cvar95),
        marker: { color: "rgba(90,140,200,0.45)",
                  line: { color: COLORS.nn, width: 1 } },
        xbins: { start: -span, end: span, size: binSize },
      },
    ], {
      ...PLOT_BASE, barmode: "overlay",
      xaxis: { title: { text: "terminal hedging P&L ($, K = " + K + ")" },
               gridcolor: COLORS.grid, zeroline: false },
      yaxis: { title: { text: "paths" }, gridcolor: COLORS.grid, zeroline: false },
      shapes: [
        { type: "line", x0: -d.delta.cvar95 * K, x1: -d.delta.cvar95 * K,
          y0: 0, y1: 1, yref: "paper",
          line: { color: COLORS.mc, width: 2, dash: "dot" } },
        { type: "line", x0: -d.deep.cvar95 * K, x1: -d.deep.cvar95 * K,
          y0: 0, y1: 1, yref: "paper",
          line: { color: COLORS.nn, width: 2, dash: "dot" } },
      ],
    }, PLOT_CONFIG);

    // holdings along the illustrative path
    const days = d.example_path.deep_holdings.map((_, i) => i + 1);
    Plotly.react("plot-holdings", [
      {
        x: days.concat([]), y: d.example_path.spot.slice(1).map((s) => s * K),
        mode: "lines", name: "spot path ($)", yaxis: "y2",
        line: { color: "rgba(255,255,255,0.35)", width: 1.5 },
      },
      {
        x: days, y: d.example_path.delta_holdings,
        mode: "lines+markers", name: "delta-hedge holding",
        line: { color: COLORS.mc, width: 2 }, marker: { size: 4 },
      },
      {
        x: days, y: d.example_path.deep_holdings,
        mode: "lines+markers", name: "deep-hedge holding",
        line: { color: COLORS.nn, width: 2.5 }, marker: { size: 4 },
      },
    ], {
      ...PLOT_BASE,
      margin: { l: 52, r: 52, t: 12, b: 42 },
      xaxis: { title: { text: "trading day" }, gridcolor: COLORS.grid,
               zeroline: false },
      yaxis: { title: { text: "holding (shares per option)" },
               gridcolor: COLORS.grid, zeroline: false, range: [0, 1.1] },
      yaxis2: { overlaying: "y", side: "right", showgrid: false,
                tickfont: { color: "rgba(255,255,255,0.4)" } },
    }, PLOT_CONFIG);
  } catch (e) {
    $("hedge-sub").textContent = e.message;
  } finally {
    btn.textContent = "Run Simulation";
    btn.disabled = false;
  }
}
$("btn-hedge").addEventListener("click", runHedge);

// ───────────────────────────────────────────────────────────── LLM ──
$("btn-risk").addEventListener("click", async () => {
  const btn = $("btn-risk");
  btn.textContent = "Generating...";
  btn.disabled = true;
  const out = $("ai-report");
  try {
    // Auto-gather any missing inputs instead of bouncing the user around.
    if (!lastAttributions) { out.textContent = "Computing attributions..."; await updateXAI(); }
    if (!lastHedge) { out.textContent = "Running hedging simulation..."; await runHedge(); }
    if (lastNNPrice == null || !lastAttributions || !lastHedge)
      throw new Error("pricing/hedging inputs unavailable; is the backend up?");

    out.textContent = "";
    out.classList.add("streaming");
    const req = {
      ticker: marketData ? marketData.ticker
        : ($("in-ticker").value.trim().toUpperCase() || "a generic underlying"),
      nn_price: lastNNPrice,
      bs_cvar: -lastHedge.delta.cvar95 * state.strike,
      deep_cvar: -lastHedge.deep.cvar95 * state.strike,
      attributions: lastAttributions,
    };

    const response = await fetch("/api/risk-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      out.textContent += decoder.decode(value, { stream: true });
    }
  } catch (e) {
    out.textContent = "Error: " + e.message;
  } finally {
    out.classList.remove("streaming");
    btn.textContent = "Generate Report";
    btn.disabled = false;
  }
});

// ───────────────────────────────────────────────── WebSocket Live Stream ──
let ws = null;
let wsSpots = [];
let wsPrices = [];
let wsTicks = [];
const WS_MAX_POINTS = 400;

function wsConnect() {
  const btn = $("btn-stream");
  if (ws && ws.readyState <= WebSocket.OPEN) {
    ws.close();
    return;
  }

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(proto + "//" + location.host + "/ws/stream");
  wsSpots = [];
  wsPrices = [];
  wsTicks = [];

  ws.onopen = () => {
    btn.textContent = "Disconnect";
    btn.classList.add("btn-stream-active");
    $("stream-sub").textContent =
      "Connected. Streaming GBM ticks at 20 Hz with neural pricing.";

    ws.send(JSON.stringify({
      spot: state.spot, strike: state.strike, sigma: state.sigma,
      rate: state.rate, maturity: state.maturity,
      option_type: state.optionType, hz: 20,
    }));

    // Initialize the streaming chart
    Plotly.newPlot("plot-stream", [
      {
        y: [], mode: "lines", name: "Spot",
        line: { color: "rgba(255,255,255,0.5)", width: 1.5 },
      },
      {
        y: [], mode: "lines", name: "NN Price",
        line: { color: COLORS.nn, width: 2 }, yaxis: "y2",
      },
    ], {
      ...PLOT_BASE,
      margin: { l: 52, r: 60, t: 12, b: 42 },
      xaxis: { title: { text: "tick" }, gridcolor: COLORS.grid,
               zeroline: false },
      yaxis: { title: { text: "spot ($)" }, gridcolor: COLORS.grid,
               zeroline: false },
      yaxis2: { title: { text: "NN price ($)" }, overlaying: "y",
                side: "right", showgrid: false,
                tickfont: { color: COLORS.nn } },
    }, PLOT_CONFIG);
  };

  ws.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    if (d.error) return;

    $("ws-spot").textContent = "$" + d.spot.toFixed(2);
    $("ws-spot").className = "v mono live";
    $("ws-price").textContent = "$" + d.price.toFixed(4);
    $("ws-price").className = "v mono live";
    $("ws-delta").textContent = d.delta.toFixed(4);
    $("ws-gamma").textContent = d.gamma.toFixed(6);
    $("ws-vega").textContent = d.vega.toFixed(4);
    $("ws-theta").textContent = d.theta.toFixed(4);
    $("ws-latency").textContent = d.latency_us.toFixed(0) + " \u00b5s";
    $("ws-ticks").textContent = d.tick.toLocaleString();

    // Append to rolling buffers
    wsSpots.push(d.spot);
    wsPrices.push(d.price);
    wsTicks.push(d.tick);
    if (wsSpots.length > WS_MAX_POINTS) {
      wsSpots.shift();
      wsPrices.shift();
      wsTicks.shift();
    }

    // Throttle chart updates to ~10 fps to avoid layout thrashing
    if (d.tick % 2 === 0) {
      Plotly.extendTraces("plot-stream",
        { y: [[d.spot], [d.price]] }, [0, 1],
        WS_MAX_POINTS);
    }
  };

  ws.onclose = () => {
    btn.textContent = "Connect";
    btn.classList.remove("btn-stream-active");
    $("stream-sub").textContent =
      "Disconnected. Click Connect to resume streaming.";
    ws = null;
  };

  ws.onerror = () => {
    $("stream-sub").textContent = "WebSocket error. Is the server running?";
    ws = null;
    btn.textContent = "Connect";
    btn.classList.remove("btn-stream-active");
  };
}

$("btn-stream").addEventListener("click", wsConnect);

// Initial load. Cheap calls go out immediately; the simulation-heavy panels
// load one after another, because the server admits only one Monte Carlo /
// batch-inference job at a time (firing them in parallel would just queue
// them there while tying up connections). The latency benchmark is the whole
// convergence workload re-run for its wall-clock alone, so it loads on
// demand via its Re-run button instead of on every page view.
refreshReadouts();
loadModelInfo();
loadErrorDistribution();
(async () => {
  // The headline price lands first: /api/price and /api/convergence would
  // otherwise race for the server's single simulation slot, and losing that
  // race leaves the hero card blank while the convergence run finishes.
  await updatePrice().catch(() => {});
  await updateConvergence().catch(() => {});
  await updateSurface().catch(() => {});
  await updateXAI().catch(() => {});
})();
const latencyShimmer = $("plot-latency").querySelector(".shimmer");
if (latencyShimmer) {
  latencyShimmer.replaceWith(Object.assign(document.createElement("p"), {
    className: "card-sub centered latency-hint",
    textContent: "Click Re-run to measure latency on this instance.",
  }));
}


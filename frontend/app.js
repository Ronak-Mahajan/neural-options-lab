/* Neural Options Lab - dashboard logic
   Talks to the FastAPI backend, renders Plotly charts, animates numbers. */

"use strict";

// ─────────────────────────────────────────────── state & element handles ──
const $ = (id) => document.getElementById(id);

const state = {
  spot: 100, strike: 100, maturity: 1.0, sigma: 0.25, rate: 0.04,
  optionType: "call", mcPaths: 50000,
  // A position, so the premium and the Greeks describe something real.
  qty: 1, mult: 100,
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
const PLOT_CONFIG = { displayModeBar: false, responsive: true, scrollZoom: false };

// ───────────────────────────────────────────────────────────── utilities ──
const NL_CHAR = String.fromCharCode(10);

const debounce = (fn, ms) => {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};

// Backend exception text used to be written straight into panel subtitles,
// where it read as a stray sentence about the model's internals.
function friendlyError(status, detail) {
  const d = String(detail || "");
  if (status === 422 || /moneyness|domain|between|less than|greater than/i.test(d)) {
    return "This contract is outside the range the models were trained on. "
      + "Move spot and strike closer together, or pick another expiry.";
  }
  if (status === 503) return "The server is busy with another simulation. Try again in a moment.";
  if (status === 0) return "Could not reach the server.";
  return "This panel is unavailable right now.";
}

async function api(path, body) {
  let res;
  try {
    res = await fetch(path, {
      method: body ? "POST" : "GET",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error(friendlyError(0, ""));
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    const err = new Error(friendlyError(res.status, detail));
    err.status = res.status;
    throw err;
  }
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

function panelMessage(plotId, text) {
  const el = $(plotId);
  if (!el) return;
  clearShimmer(plotId);
  try { Plotly.purge(el); } catch { /* never had a plot */ }
  el.querySelector(".panel-message")?.remove();
  const box = document.createElement("div");
  box.className = "empty-state panel-message";
  box.innerHTML = "<p>" + text + "</p>";
  el.appendChild(box);
}

function clearPanelMessage(plotId) {
  $(plotId)?.querySelector(".panel-message")?.remove();
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

  // The matching readout is a typed field: a slider cannot express a strike
  // of 137.42, and a pricer that cannot take one is a demonstration.
  const box = $("val-" + id);
  if (!box) return;
  const commit = () => {
    const raw = box.value.trim().replace(/[%$,\s]/g, "");
    // "1.5y" and "30d" both mean something for maturity.
    const m = /^([0-9]*\.?[0-9]+)\s*([a-z]*)$/i.exec(raw);
    if (!m) { box.classList.add("invalid"); return; }
    let v = parseFloat(m[1]);
    const unit = m[2].toLowerCase();
    if (id === "maturity") {
      if (unit === "d") v = v / 252;
      else if (unit === "m") v = v / 12;
      else if (unit === "w") v = v / 52;
      // a bare number large enough to be days rather than years
      else if (!unit && v > 3) v = v / 252;
    }
    const lo = parseFloat(el.min), hi = parseFloat(el.max);
    // Volatility and rate are shown and typed in percent, and their sliders
    // are in percent too; only the state is a fraction.
    const sliderValue = v;
    if (!isFinite(sliderValue) || sliderValue < lo || sliderValue > hi) {
      box.classList.add("invalid");
      return;
    }
    box.classList.remove("invalid");
    // Typed values are exact: widen the step so the browser does not round
    // 137.42 to 137 on its way into the slider.
    el.step = "any";
    el.value = sliderValue;
    paint();
    onChange(parseFloat(el.value));
  };
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); commit(); box.blur(); }
    if (e.key === "Escape") { box.classList.remove("invalid"); refreshReadouts(); box.blur(); }
  });
  // While a field has focus, refreshReadouts must not overwrite what is
  // being typed.
  box.addEventListener("focus", () => { box.dataset.editing = "1"; });
  box.addEventListener("blur", () => {
    box.classList.remove("invalid");
    commit();
    delete box.dataset.editing;
    refreshReadouts();
  });
}

// refreshReadouts writes into these fields, so it must skip the one the user
// is typing in.
function setReadout(id, text) {
  const el = $(id);
  if (el && !el.dataset.editing) el.value = text;
}

// Maturities at or below 12 trading days route to the 0DTE rough-vol
// surrogate, which is trained on a narrower moneyness band.
const ZERO_DTE_CUTOFF = 12 / 252;
const is0dte = () => state.maturity <= ZERO_DTE_CUTOFF + 1e-9;

// A position is contracts x shares each; a negative count is a short, which
// flips the sign of the premium and of every Greek.
const positionSize = () => state.qty * state.mult;

function fmtSigned(v, digits) {
  const sign = v < 0 ? "\u2212" : "";
  return sign + "$" + Math.abs(v).toLocaleString(undefined, {
    minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// Whole-share and whole-dollar figures for a book; per-contract figures keep
// the four decimals the model's own accuracy supports.
function renderPosition() {
  const line = $("position-line");
  const hint = $("position-hint");
  if (!line) return;
  const n = positionSize();
  if (lastNNPrice == null || !isFinite(n) || n === 0) {
    line.textContent = "";
  } else {
    const total = lastNNPrice * n;
    line.textContent = (state.qty < 0 ? "Short " : "") +
      Math.abs(state.qty).toLocaleString() + " \u00d7 " +
      state.mult.toLocaleString() + " shares = " + fmtSigned(total, 2) +
      (state.qty < 0 ? " received" : " to pay");
  }
  if (hint) {
    hint.textContent = state.qty < 0
      ? "A short position: the premium is received and every Greek changes sign."
      : "Sets the scale of the premium and the Greeks below.";
  }
  renderGreeks();
}

// The Greeks strip shows either one contract or the whole position.
function renderGreeks() {
  if (!lastGreeks) return;
  const n = greekBasis === "position" ? positionSize() : 1;
  const digits = greekBasis === "position" ? 2 : 4;
  const map = { delta: "g-delta", gamma: "g-gamma", vega: "g-vega",
                theta: "g-theta", rho: "g-rho" };
  for (const [k, id] of Object.entries(map)) {
    const el = $(id);
    if (!el) continue;
    const v = lastGreeks[k] * n;
    el.textContent = greekBasis === "position"
      ? (v < 0 ? "\u2212" : "") + Math.abs(v).toLocaleString(undefined, {
          minimumFractionDigits: 2, maximumFractionDigits: 2 })
      : v.toFixed(digits);
  }
  document.querySelectorAll(".greek-unit").forEach((el) => {
    if (!el.dataset.unit) el.dataset.unit = el.textContent;
    el.textContent = greekBasis === "position"
      ? el.dataset.unit + ", whole position" : el.dataset.unit;
  });
}

// How far the contract sits from at-the-money, in words. "At the money"
// covers the ±2% band where the distinction stops being meaningful.
function moneynessWords() {
  const m = state.spot / state.strike;
  if (Math.abs(m - 1) < 0.02) return "at-the-money";
  const pct = Math.round(Math.abs(m - 1) * 100);
  const itm = state.optionType === "call" ? m > 1 : m < 1;
  return pct + "% " + (itm ? "in-the-money" : "out-of-the-money");
}

function maturityWords() {
  const T = state.maturity;
  const days = Math.max(1, Math.round(T * 252));
  if (days <= 45) return days === 1 ? "one-day" : days + "-day";
  if (T < 0.95) return Math.round(T * 12) + "-month";
  if (Math.abs(T - 1) < 0.03) return "one-year";
  if (Math.abs(T - 2) < 0.03) return "two-year";
  return T.toFixed(2) + "-year";
}

// The contract in one sentence. This is the page's answer to "what am I
// looking at, and did I need to type a ticker first?" - so it always says
// whether the underlying is hypothetical or a real one that was loaded.
// Just the instrument, for prose that continues after it.
function contractShort() {
  const kind = is0dte() ? "European " + state.optionType
                        : "Asian " + state.optionType;
  const head = maturityWords() + " " + moneynessWords() + " " + kind;
  return marketData ? head + " on " + marketData.ticker
                    : head + " on a $" + state.spot + " stock";
}

function contractSentence() {
  const kind = is0dte() ? "European " + state.optionType
                        : "Asian " + state.optionType;
  const head = maturityWords() + " " + moneynessWords() + " " + kind;
  const vol = $("val-sigma").value, rate = $("val-rate").value;
  if (marketData) {
    return "Pricing a " + head + " on " + marketData.ticker + " at $" +
      state.spot.toLocaleString() + ", volatility " + vol + ", rate " + rate +
      ", from market data loaded " + marketData.as_of.slice(0, 10) + ".";
  }
  return "Pricing a " + head + " on a $" + state.spot + " stock, volatility " +
    vol + ", rate " + rate +
    ". No market data needed: every number below is computed live for it.";
}

// What each tab does with the contract named above it. The Hedging tab in
// particular runs its own instrument, which nothing on screen used to say.
const CONTRACT_SCOPE = {
  hedging: "The hedging simulation runs its own 30-day at-the-money call, not the contract above.",
  ai: "The summary is written from whatever the Pricing and Hedging tabs last computed.",
  stream: "The feed prices this contract, tick by tick.",
};

function renderContractLine() {
  const pill = $("contract-pill");
  pill.textContent = marketData ? marketData.ticker : "Example";
  pill.classList.toggle("live", !!marketData);
  $("contract-text").textContent = contractSentence();
  const scope = $("contract-scope");
  const clause = CONTRACT_SCOPE[typeof currentTab === "string" ? currentTab : "pricing"];
  scope.textContent = clause || "";
  scope.hidden = !clause;
}

// Under rough volatility the hedging run supplies its own volatility and
// rate, so the two sliders that look like they drive it do nothing.
function paintRailScope() {
  const idle = currentTab === "hedging" && state.hedgeDynamics === "rough";
  for (const id of ["in-sigma", "in-rate"])
    $(id).closest(".param").classList.toggle("rail-inactive", idle);
  $("rail-inactive-note").hidden = !idle;
}

function refreshReadouts() {
  setReadout("val-spot", String(+state.spot.toFixed(4)));
  setReadout("val-strike", String(+state.strike.toFixed(4)));
  setReadout("val-maturity", is0dte()
    ? Math.max(1, Math.round(state.maturity * 252)) + "d"
    : state.maturity.toFixed(2) + "y");
  const sigPct = state.sigma * 100;
  setReadout("val-sigma",
    (Math.abs(sigPct - Math.round(sigPct)) < 0.05 ? Math.round(sigPct)
                                                  : sigPct.toFixed(1)) + "%");
  setReadout("val-rate", (state.rate * 100).toFixed(2).replace(/0$/, "") + "%");
  $("rail-summary").textContent = (marketData ? marketData.ticker + " · " : "") +
    maturityWords() + " " + moneynessWords() + " " + state.optionType +
    " · σ " + $("val-sigma").value + " · r " + $("val-rate").value;

  renderContractLine();

  // Above 12 trading days the payoff is an average; below it, a plain
  // European option under a different model. Saying "Asian" in both places
  // would be wrong, and the price is discontinuous across the boundary.
  $("rail-note").textContent = is0dte()
    ? "At or below 12 trading days the contract is a standard European "
      + "option priced by a second network trained on rough volatility, so "
      + "its price does not line up with the averaged contract above that "
      + "boundary. Greeks are exact derivatives of the network."
    : "Above 12 trading days the contract is a discrete arithmetic-average "
      + "Asian option with 50 monitoring dates. The simulation it is checked "
      + "against uses antithetic sampling and a geometric-Asian control "
      + "variate. Greeks are exact derivatives of the network.";

  const m = state.spot / state.strike;
  $("moneyness-val").textContent = m.toFixed(2) + (is0dte() ? " · short-dated" : "");
  const [lo, hi] = is0dte() ? [0.85, 1.15] : [0.5, 2.0];
  const outside = m < lo || m > hi;
  $("domain-warning").textContent = "Spot over strike is " + m.toFixed(2) +
    ", outside the range this model was trained on (" + lo + " to " + hi +
    "). Move spot or strike closer together.";
  $("domain-warning").classList.toggle("show", outside);

  document.querySelectorAll("#maturity-quickpick .pick").forEach((b) =>
    b.classList.toggle("active", Math.abs(+b.dataset.t - state.maturity) < 1e-6));

  // The short-dated volatility surface only describes contracts of 12
  // trading days or less, so open it when the reader moves into that regime.
  const short = is0dte();
  if (short !== wasShortDated) {
    const g = $("group-domain");
    if (short && g) g.open = true;
    wasShortDated = short;
  }
}
let wasShortDated = null;

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
let lastGreeks = null;
let lastCheck = null;
let greekBasis = "unit";
async function updatePrice() {
  document.querySelector(".results").classList.add("updating");
  const seq = ++priceSeq;
  try {
    const d = await api("/api/price", { ...optionBody(), mc_paths: state.mcPaths });
    if (seq !== priceSeq) return; // a newer request superseded this one

    lastNNPrice = d.nn.price;
    renderReportInputs();
    animateNumber($("nn-price"), d.nn.price, fmtMoney);
    animateNumber($("mc-price"), d.mc.price, fmtMoney);
    lastGreeks = d.nn.greeks;
    renderPosition();
    $("nn-sub").textContent = is0dte()
      ? "with all five Greeks, from one pass of the short-dated model"
      : "with all five Greeks, from one pass";
    lastCheck = { price: d.mc.price, n_paths: d.mc.n_paths,
                  half: (d.mc.ci_high - d.mc.ci_low) / 2 };
    $("mc-ci").textContent = "±$" +
      ((d.mc.ci_high - d.mc.ci_low) / 2).toFixed(4) +
      " at 95% confidence · fresh run, new seed each time";

    // The headline is how closely the network matches the simulation. The
    // old speedup ratio was two single-shot wall-clocks on a shared host,
    // so it swung several-fold between identical page loads; the timings
    // are still reported, just not as the claim.
    const diff = Math.abs(d.nn.price - d.mc.price);
    const bpsK = diff / state.strike * 1e4;
    animateNumber($("speedup"), bpsK, (v) => v.toFixed(1) + " bps");
    // Three states, not two. At 50,000 paths the simulation's error bar is
    // tighter than the network's own published error, so a gap can sit
    // outside the bar and still be exactly what the model promises. Calling
    // that a failure would misreport the result in the alarming direction.
    const inCI = d.comparison.within_mc_ci;
    const tol = modelInfo && modelInfo.eval
      ? modelInfo.eval.ensemble.price.p95_abs_bps : 2.5;
    const agr = $("agreement");
    if (inCI) {
      agr.textContent = "$" + diff.toFixed(4) + " apart · inside the simulation's error bar";
      agr.className = "card-sub centered agreement-ok";
    } else if (bpsK <= tol) {
      agr.textContent = "$" + diff.toFixed(4) + " apart · outside the simulation's " +
        "error bar, inside the model's own held-out range";
      agr.className = "card-sub centered agreement-neutral";
    } else {
      agr.textContent = "$" + diff.toFixed(4) + " apart · outside the simulation's error bar";
      agr.className = "card-sub centered agreement-warn";
    }
    $("timing-line").textContent = "On this server: network " +
      fmtMs(d.nn.latency_ms) + " for the price and all five Greeks, " +
      "simulation " + fmtMs(d.mc.latency_ms) + " for " +
      d.mc.n_paths.toLocaleString() + " paths" +
      (d.mc.engine === "rough_bergomi" ? ", rough-volatility engine" : "") + ".";

    document.querySelector(".results").classList.remove("errored");
    renderGreeks();
  } catch (err) {
    if (seq !== priceSeq) return;
    // Keep the last good numbers on screen but visibly stale: a red message
    // next to crisp prices read as if the prices belonged to the message.
    document.querySelector(".results").classList.add("errored");
    $("agreement").textContent = err.message;
    $("agreement").className = "card-sub centered agreement-warn";
    $("timing-line").textContent = "";
  } finally {
    if (seq === priceSeq) document.querySelector(".results").classList.remove("updating");
  }
}

// ──────────────────────────────────────────────────────── convergence plot ──
async function updateConvergence() {
  try {
    const d = await api("/api/convergence", optionBody());
    clearShimmer("plot-convergence");
    clearPanelMessage("plot-convergence");

    const xs = d.mc_points.map((p) => p.n_paths);
    const traces = [
      { // CI band (upper then lower with fill)
        x: [...xs, ...xs.slice().reverse()],
        y: [...d.mc_points.map((p) => p.ci_high),
            ...d.mc_points.map((p) => p.ci_low).reverse()],
        fill: "toself", fillcolor: "rgba(196,131,92,0.15)",
        line: { width: 0 }, hoverinfo: "skip",
        name: "95% confidence interval", showlegend: true,
      },
      {
        x: xs, y: d.mc_points.map((p) => p.price),
        mode: "lines+markers",
        name: d.engine === "rough_bergomi"
          ? "Simulation (rough volatility)" : "Simulation",
        line: { color: COLORS.mc, width: 2.5, shape: "spline" },
        marker: { size: 7, color: COLORS.mc },
        customdata: d.mc_points.map((p) => fmtMs(p.latency_ms)),
        hovertemplate: "%{x:,} paths → $%{y:.4f}<br>%{customdata}<extra></extra>",
      },
      {
        x: [xs[0], xs[xs.length - 1]], y: [d.nn.price, d.nn.price],
        mode: "lines", name: "Network price",
        line: { color: COLORS.nn, width: 2.5, dash: "dash" },
        hovertemplate: "NN: $%{y:.4f}<extra></extra>",
      },
      {
        x: [xs[0], xs[xs.length - 1]],
        y: [d.reference.price, d.reference.price],
        mode: "lines",
        name: `High-precision reference (${Math.round(d.reference.n_paths / 1000)}k paths)`,
        line: { color: "rgba(255,255,255,0.45)", width: 1.5, dash: "dot" },
        hovertemplate: "Reference: $%{y:.4f}<extra></extra>",
      },
    ];
    Plotly.react("plot-convergence", traces, {
      ...PLOT_BASE,
      xaxis: { type: "log", title: { text: "simulated paths" },
               tickvals: [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
               ticktext: ["500", "1k", "2k", "5k", "10k", "20k", "50k", "100k"],
               gridcolor: COLORS.grid, zeroline: false },
      yaxis: { title: { text: "option price" }, gridcolor: COLORS.grid,
               zeroline: false, tickformat: ".3f" },
    }, PLOT_CONFIG);
  } catch (err) {
    panelMessage("plot-convergence", err.message);
  }
}

// ─────────────────────────────────────────────────────── IV surface plot ──
// The 0DTE implied-vol surface depends only on (sigma, r): it is the
// no-arbitrage surrogate's smile at every maturity in the 0DTE box, with the
// Durrleman butterfly function g and the calendar slope dw/dT evaluated by
// autograd on the same grid so the "arbitrage-free" claim is checked live.
async function updateIVSurface() {
  const panel = $("panel-ivsurface");
  if (!panel) return;
  try {
    const d = await api("/api/iv-surface",
      { sigma: state.sigma, rate: state.rate, resolution: 41 });
    clearShimmer("plot-ivsurface");
    clearPanelMessage("plot-ivsurface");
    const fmtK = (v) => v.toFixed(3);
    const okB = d.g_min > 0, okC = d.calendar_min > 0;
    $("ivsurface-stats").innerHTML =
      hedgeStatChip("No-arbitrage check",
        okB && okC ? "passed at every point" : "violation found",
        okB && okC ? "good" : "") +
      hedgeStatChip("Distance from the pricing model",
        (d.fit && d.fit.iv_rmse_volpts_resolved != null
          ? d.fit.iv_rmse_volpts_resolved.toFixed(2) + " vol points" : "—"));
    $("ivsurface-stat").textContent =
      "Checked at " + d.n_points.toLocaleString() + " points across the grid, " +
      "in " + fmtMs(d.latency_ms) + ". Butterfly minimum " + d.g_min.toFixed(3) +
      " at log-moneyness " + fmtK(d.g_min_at[1]) + " and " +
      (d.g_min_at[0] * 252).toFixed(1) + " days; calendar slope minimum " +
      d.calendar_min.toExponential(2) + ". Both must stay above zero.";

    Plotly.react("plot-ivsurface", [{
      type: "surface",
      x: d.k, y: d.days, z: d.iv.map((row) => row.map((v) => v * 100)),
      colorscale: [[0, "#0a2a55"], [0.5, "#0A84FF"], [1, "#dbe9ff"]],
      showscale: false,
      contours: { z: { show: true, usecolormap: true, width: 1,
                       highlightcolor: "#fff" } },
      hovertemplate: "k %{x:.3f} · %{y:.1f}d → IV %{z:.2f}%<extra></extra>",
    }], {
      ...PLOT_BASE, showlegend: false,
      margin: { l: 0, r: 0, t: 6, b: 0 },
      scene: {
        xaxis: { title: { text: "log-moneyness k = ln(K/F)" }, gridcolor: COLORS.grid,
                 color: COLORS.ink },
        yaxis: { title: { text: "days to expiry" }, gridcolor: COLORS.grid,
                 color: COLORS.ink },
        zaxis: { title: { text: "implied vol (%)" }, gridcolor: COLORS.grid,
                 color: COLORS.ink },
        bgcolor: "rgba(0,0,0,0)",
        camera: { eye: { x: -1.7, y: -1.5, z: 0.9 } },
      },
    }, PLOT_CONFIG);
  } catch (err) {
    const sub = $("ivsurface-sub");
    if (sub) sub.textContent = err.message;
    panelMessage("plot-ivsurface", err.message);
  }
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
  try {
    const d = await api("/api/surface", {
      sigma: state.sigma, rate: state.rate, strike: state.strike,
      option_type: state.optionType,
    });
    clearShimmer("plot-surface");
    clearPanelMessage("plot-surface");

    $("surface-stat").textContent = "This grid is " +
      d.n_prices.toLocaleString() + " separate prices, computed in " +
      fmtMs(d.latency_ms) + " on this server: about " +
      Math.round(d.prices_per_second / 1000).toLocaleString() +
      ",000 prices per second in a batch.";

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
  } catch (err) {
    panelMessage("plot-surface", err.message);
  }
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

  const QUANTITY = { price: "price", delta: "delta", vega: "vega" };
  $("error-sub").textContent =
    "How far the network's " + QUANTITY[errorMetric] + " sits from a " +
    (d.ref_paths / 1000).toFixed(0) + ",000-path simulation, on " +
    d.n_points.toLocaleString() + " contracts it never saw in training.";
  const e = d.ensemble[errorMetric];
  $("error-stat").textContent =
    "Five averaged networks: mean " + (e.mean_bps >= 0 ? "+" : "") +
    e.mean_bps.toFixed(1) + " " + meta.unit + " · typical error " +
    e.rmse_bps.toFixed(1) + " · 95% of errors within " +
    e.p95_abs_bps.toFixed(1) + " " + meta.unit +
    (errorMetric === "price" ? " of strike" : "") + " (one network: " +
    d.single[errorMetric].rmse_bps.toFixed(1) + " typical).";

  // Shared bins so the two histograms are directly comparable.
  const all = [...single, ...ens];
  const span = Math.max(Math.abs(Math.min(...all)), Math.abs(Math.max(...all)));
  const binSize = (2 * span) / 46;

  const traces = [
    {
      type: "histogram", x: single,
      name: "one network · typical error " +
        d.single[errorMetric].rmse_bps.toFixed(1) + " " + meta.unit,
      marker: { color: "rgba(143,123,255,0.5)",
                line: { color: COLORS.violet, width: 1 } },
      xbins: { start: -span, end: span, size: binSize },
    },
    {
      type: "histogram", x: ens,
      name: "five averaged · typical error " +
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
    yaxis: { title: { text: "contracts" }, gridcolor: COLORS.grid,
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
    clearShimmer("plot-errors");
  }
}

// ──────────────────────────────────────────────────────────── model badge ──
let modelInfo = null;
async function loadModelInfo() {
  const dot = $("status-dot"), txt = $("model-badge-text");
  const body = $("model-card-body");
  const rows = (pairs) => pairs.map(([k, v]) =>
    "<dt>" + k + "</dt><dd>" + v + "</dd>").join("");
  try {
    const health = await api("/api/health");
    if (!health.model_loaded) {
      dot.className = "status-dot bad";
      txt.textContent = "Model unavailable";
      body.innerHTML = rows([["Status", "The pricing model is not loaded on this server."]]);
      return;
    }
    const m = await api("/api/model-info");
    modelInfo = m;
    dot.className = "status-dot ok";
    txt.textContent = "Model ready";
    const p = m.param_ranges || {};
    const pairs = [
      ["Architecture", (m.n_members > 1 ? m.n_members + " networks, " : "One network, ") +
        m.n_parameters.toLocaleString() + " parameters each"],
      ["Training data", m.n_samples.toLocaleString() +
        " contracts labelled by Monte Carlo" +
        (m.mc_paths_per_label ? " at " + m.mc_paths_per_label.toLocaleString() + " paths each" : "")],
    ];
    if (m.eval) {
      pairs.push(["Accuracy", "typical pricing error " +
        m.eval.ensemble.price.rmse_bps.toFixed(1) + " basis points of strike, on " +
        m.eval.n_points.toLocaleString() + " held-out contracts against " +
        (m.eval.ref_paths / 1000).toFixed(0) + ",000-path references"]);
    }
    if (p.moneyness && p.maturity && p.sigma) {
      pairs.push(["Trained range", "spot over strike " + p.moneyness[0] + " to " +
        p.moneyness[1] + ", expiry " + p.maturity[0] + " to " + p.maturity[1] +
        " years, volatility " + Math.round(p.sigma[0] * 100) + "% to " +
        Math.round(p.sigma[1] * 100) + "%"]);
    }
    body.innerHTML = rows(pairs);
    const acc = $("accuracy-stats");
    if (acc) {
      acc.innerHTML =
        hedgeStatChip("Ensemble", m.n_members + " networks") +
        hedgeStatChip("Parameters", m.n_parameters.toLocaleString() + " each") +
        hedgeStatChip("Training set",
          m.n_samples.toLocaleString() + " Monte Carlo-labelled contracts") +
        (m.eval ? hedgeStatChip("Typical error",
          m.eval.ensemble.price.rmse_bps.toFixed(1) + " bps of strike") : "");
    }
    const lede = $("report-lede");
    if (lede) {
      lede.textContent = m.report_writer === "model"
        ? "A short risk summary drafted by a language model from the current "
          + "price, its attribution and the hedging run. It is given the "
          + "numbers and nothing else."
        : "A short risk summary assembled from the current price, its "
          + "attribution and the hedging run, written by a rule-based "
          + "narrator on this server.";
    }
    const teaser = $("accuracy-teaser");
    if (teaser && m.eval) {
      teaser.textContent = "Typical error " +
        m.eval.ensemble.price.rmse_bps.toFixed(1) +
        " basis points of strike on " + m.eval.n_points.toLocaleString() +
        " held-out contracts. Measured once, so it does not change with the inputs.";
    }
  } catch {
    dot.className = "status-dot bad";
    txt.textContent = "Server unreachable";
    body.innerHTML = rows([["Status", "The server did not respond."]]);
  }
}

// The model card opens on click and closes on the next click outside it or
// on Escape, so it never sits over the page uninvited.
(() => {
  const btn = $("model-badge"), card = $("model-card");
  const setOpen = (open) => {
    card.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
  };
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(card.hidden);
  });
  document.addEventListener("click", (e) => {
    if (!card.hidden && !card.contains(e.target)) setOpen(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setOpen(false);
  });
})();

// ─────────────────────────────────────────────────────────────── wire up ──
const refreshFast = debounce(updatePrice, 220);
const refreshSlow = debounce(() => { updateConvergence(); updateSurface(); updateXAI(); updateIVSurface(); }, 650);
const refreshAll = () => { refreshReadouts(); refreshFast(); refreshSlow(); syncURL(); };

bindSlider("spot", (v) => { state.spot = v; refreshAll(); });
bindSlider("strike", (v) => { state.strike = v; refreshAll(); });
// Maturities strictly between the 0DTE cutoff (12/252) and the Asian net's
// 0.05y training floor are covered by neither model; the API rejects them
// with a 422. The slider's 0.001 grid has three such positions (0.048,
// 0.049 and, because of float rounding, 0.047619 itself is unreachable), so
// snap to whichever valid endpoint is nearer: 0.047 (serves as 12 trading
// days) or 0.05.
const ASIAN_FLOOR = 0.05;
function snapMaturity(v) {
  if (v > ZERO_DTE_CUTOFF && v < ASIAN_FLOOR) {
    v = (v - ZERO_DTE_CUTOFF) < (ASIAN_FLOOR - v) ? 0.047 : ASIAN_FLOOR;
    const el = $("in-maturity");
    el.value = v;
    el.style.setProperty("--fill",
      (el.value - el.min) / (el.max - el.min) * 100 + "%");
  }
  return v;
}
bindSlider("maturity", (v) => { state.maturity = snapMaturity(v); refreshAll(); });
bindSlider("sigma", (v) => { state.sigma = v / 100; refreshAll(); });
bindSlider("rate", (v) => { state.rate = v / 100; refreshAll(); });

bindSegmented("option-type", (v) => { state.optionType = v; refreshAll(); });
bindSegmented("mc-paths", (v) => { state.mcPaths = parseInt(v); refreshFast(); });
bindSegmented("error-metric", (v) => { errorMetric = v; renderErrorDistribution(); });

$("btn-benchmark").addEventListener("click", updateBenchmark);

// Size controls. A whole number of contracts; a negative count is a short.
function bindSizeField(id, key, { min, max, integer }) {
  const el = $(id);
  const commit = () => {
    const v = parseFloat(el.value.replace(/[,\s]/g, ""));
    if (!isFinite(v) || v < min || v > max || (integer && v !== Math.round(v))) {
      el.classList.add("invalid");
      return;
    }
    el.classList.remove("invalid");
    state[key] = v;
    renderPosition();
    renderContractLine();
    syncURL();
  };
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); commit(); el.blur(); }
    if (e.key === "Escape") { el.value = state[key]; el.classList.remove("invalid"); el.blur(); }
  });
  el.addEventListener("blur", () => { commit(); el.value = state[key]; });
}
bindSizeField("in-qty", "qty", { min: -100000, max: 100000, integer: true });
bindSizeField("in-mult", "mult", { min: 1, max: 10000, integer: true });

bindSegmented("greek-basis", (v) => { greekBasis = v; renderGreeks(); });

// A quote that can leave the page: the contract, every input, the price, the
// Greeks, the independent check and when it was produced.
function quoteRows() {
  const g = lastGreeks || {};
  const n = positionSize();
  const stamp = new Date().toISOString().replace("T", " ").slice(0, 19);
  const rows = [
    ["Produced", stamp + " local"],
    ["Instrument", contractShort()],
    ["Underlying", marketData ? marketData.ticker : "hypothetical"],
    ["Spot", state.spot],
    ["Strike", state.strike],
    ["Time to expiry (years)", +state.maturity.toFixed(6)],
    ["Volatility", (state.sigma * 100).toFixed(4) + "%"],
    ["Rate", (state.rate * 100).toFixed(4) + "%"],
    ["Type", state.optionType],
    ["Contracts", state.qty],
    ["Shares per contract", state.mult],
    ["Price per contract", lastNNPrice == null ? "" : lastNNPrice.toFixed(6)],
    ["Position value", lastNNPrice == null ? "" : (lastNNPrice * n).toFixed(2)],
    ["Delta per contract", g.delta == null ? "" : g.delta.toFixed(6)],
    ["Gamma per contract", g.gamma == null ? "" : g.gamma.toFixed(6)],
    ["Vega per contract", g.vega == null ? "" : g.vega.toFixed(6)],
    ["Theta per contract", g.theta == null ? "" : g.theta.toFixed(6)],
    ["Rho per contract", g.rho == null ? "" : g.rho.toFixed(6)],
    ["Delta, whole position", g.delta == null ? "" : (g.delta * n).toFixed(2)],
    ["Gamma, whole position", g.gamma == null ? "" : (g.gamma * n).toFixed(2)],
    ["Vega, whole position", g.vega == null ? "" : (g.vega * n).toFixed(2)],
    ["Theta, whole position", g.theta == null ? "" : (g.theta * n).toFixed(2)],
    ["Rho, whole position", g.rho == null ? "" : (g.rho * n).toFixed(2)],
  ];
  if (lastCheck) {
    rows.push(["Cross-check price", lastCheck.price.toFixed(6)]);
    rows.push(["Cross-check paths", lastCheck.n_paths]);
    rows.push(["Cross-check 95% half-width", lastCheck.half.toFixed(6)]);
  }
  if (modelInfo) {
    rows.push(["Model", modelInfo.n_members + " networks x " +
      modelInfo.n_parameters + " parameters"]);
    if (modelInfo.eval) {
      rows.push(["Model typical error",
        modelInfo.eval.ensemble.price.rmse_bps.toFixed(2) + " bps of strike"]);
    }
  }
  return rows;
}

function flashQuoteNote(text) {
  const note = $("quote-note");
  note.textContent = text;
  setTimeout(() => { note.textContent = ""; }, 2400);
}

$("btn-copy-quote").addEventListener("click", async () => {
  const text = quoteRows().map(([k, v]) => k + ": " + v).join(NL_CHAR);
  try {
    await navigator.clipboard.writeText(text);
    flashQuoteNote("Quote copied");
  } catch {
    flashQuoteNote("Clipboard unavailable; use Download CSV");
  }
});

$("btn-download-quote").addEventListener("click", () => {
  const csv = "field,value" + NL_CHAR + quoteRows()
    .map(([k, v]) => '"' + String(k).replace(/"/g, '""') + '","' +
                     String(v).replace(/"/g, '""') + '"').join(NL_CHAR);
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "quote-" + state.optionType + "-" + state.strike + ".csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
  flashQuoteNote("CSV downloaded");
});

// ──────────────────────────────────────────── URL state, presets, sharing ──
// Every slider, the contract type, the tab and the hedging cost are mirrored
// into the query string so a specific finding can be sent as a link, e.g.
//   /?tab=pricing&spot=160&strike=100&T=1&sigma=0.25&rate=0.04&type=put
const TAB_IDS = { pricing: "tab-pricing", stream: "tab-stream",
                  hedging: "tab-hedging", ai: "tab-ai" };
let currentTab = "pricing";

function serializeState() {
  const q = new URLSearchParams();
  q.set("tab", currentTab);
  q.set("spot", String(state.spot));
  q.set("strike", String(state.strike));
  q.set("T", String(+state.maturity.toFixed(4)));
  q.set("sigma", String(+state.sigma.toFixed(4)));
  q.set("rate", String(+state.rate.toFixed(4)));
  q.set("type", state.optionType);
  if (state.mcPaths !== 50000) q.set("paths", String(state.mcPaths));
  if (Math.round(state.hedgeCost * 1e4) !== 50)
    q.set("cost", String(Math.round(state.hedgeCost * 1e4)));
  if (state.qty !== 1) q.set("qty", String(state.qty));
  if (state.mult !== 100) q.set("mult", String(state.mult));
  if (state.hedgeDynamics !== "rough") q.set("dyn", state.hedgeDynamics);
  if (marketData) q.set("ticker", marketData.ticker);
  return q;
}
const syncURL = debounce(() => {
  history.replaceState(null, "", "?" + serializeState().toString());
}, 300);

function setSegmented(containerId, value) {
  $(containerId).querySelectorAll(".seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.value === String(value)));
}
function setSlider(id, value) {
  const el = $("in-" + id);
  el.value = value;
  el.style.setProperty("--fill",
    (el.value - el.min) / (el.max - el.min) * 100 + "%");
}

// Apply a partial state (from the URL or a preset) to the controls and the
// state object without firing per-control refreshes; the caller refreshes.
function applyState(p) {
  const num = (k) => (p[k] !== undefined && p[k] !== null && p[k] !== "" &&
                      isFinite(+p[k])) ? +p[k] : undefined;
  const spot = num("spot"), strike = num("strike");
  if (spot !== undefined || strike !== undefined) {
    const s = spot ?? state.spot, k = strike ?? state.strike;
    // Sliders default to 55..195; a ticker-scale spot needs the rail
    // rescaled around it first, otherwise the browser clamps the value.
    const el = $("in-spot");
    if (s < +el.min || s > +el.max || k < +el.min || k > +el.max)
      rescaleSpotSliders(Math.max(s, k));
    state.spot = s; state.strike = k;
    setSlider("spot", s); setSlider("strike", k);
  }
  const T = num("T");
  if (T !== undefined) {
    state.maturity = Math.min(2, Math.max(0.004, T));
    setSlider("maturity", state.maturity);
    state.maturity = snapMaturity(state.maturity);
  }
  const sigma = num("sigma");
  if (sigma !== undefined) { state.sigma = sigma; setSlider("sigma", sigma * 100); }
  const rate = num("rate");
  if (rate !== undefined) { state.rate = rate; setSlider("rate", rate * 100); }
  if (p.type === "call" || p.type === "put") {
    state.optionType = p.type; setSegmented("option-type", p.type);
  }
  const paths = num("paths");
  if (paths !== undefined && [10000, 50000, 100000].includes(paths)) {
    state.mcPaths = paths; setSegmented("mc-paths", paths);
  }
  const cost = num("cost");
  if (cost !== undefined) {
    const bps = Math.min(200, Math.max(0, Math.round(cost / 5) * 5));
    state.hedgeCost = bps / 1e4;
    $("in-cost").value = bps; $("val-cost").textContent = bps + " bps";
  }
  const qty = num("qty");
  if (qty !== undefined && Number.isInteger(qty)) {
    state.qty = qty; $("in-qty").value = qty;
  }
  const mult = num("mult");
  if (mult !== undefined && mult >= 1) {
    state.mult = mult; $("in-mult").value = mult;
  }
  if (p.dyn === "rough" || p.dyn === "gbm") {
    state.hedgeDynamics = p.dyn; setSegmented("hedge-dynamics", p.dyn);
  }
}

function showTab(key) {
  const id = TAB_IDS[key] || TAB_IDS.pricing;
  currentTab = TAB_IDS[key] ? key : "pricing";
  $("rail-note").hidden = currentTab === "hedging";
  if (currentTab === "ai") renderReportInputs();
  renderContractLine();
  paintRailScope();
  document.querySelectorAll(".tab-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === id));
  document.querySelectorAll(".tab-pane").forEach((p) =>
    p.style.display = p.id === id ? "flex" : "none");
  // Each pane's content starts at the top; landing mid-scroll shows a void.
  window.scrollTo(0, 0);
  // charts drawn or window-resized while this pane was hidden need a nudge
  requestAnimationFrame(() => {
    document.querySelectorAll("#" + id + " .js-plotly-plot")
      .forEach((p) => Plotly.Plots.resize(p));
  });
  syncURL();
}

// The Write summary button used to be a black box: it silently ran the
// hedging simulation and the attribution before writing anything.
function renderReportInputs() {
  const el = $("report-inputs");
  if (!el) return;
  const ready = (ok) => ok ? "ready" : "will be computed";
  el.innerHTML =
    hedgeStatChip("Price", ready(lastNNPrice != null), lastNNPrice != null ? "good" : "") +
    hedgeStatChip("Attribution", ready(!!lastAttributions), lastAttributions ? "good" : "") +
    hedgeStatChip("Hedging run", ready(!!lastHedge), lastHedge ? "good" : "");
  el.className = "hedge-stats";
}

function applyPreset(p) {
  applyState(p);
  refreshAll();
  showTab(p.tab || "pricing");
  if (p.tab === "hedging" && p.run) runHedge();
}
document.querySelectorAll(".chip-btn[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    try { applyPreset(JSON.parse(btn.dataset.preset)); } catch (e) { /* ignore */ }
  });
});

$("btn-share").addEventListener("click", async () => {
  const url = location.origin + location.pathname + "?" + serializeState().toString();
  const btn = $("btn-share");
  try {
    await navigator.clipboard.writeText(url);
    btn.textContent = "Copied"; btn.classList.add("copied");
  } catch {
    // Clipboard blocked (insecure context / permissions): fall back to the
    // address bar, which syncURL keeps current.
    history.replaceState(null, "", "?" + serializeState().toString());
    btn.textContent = "Link in address bar";
  }
  setTimeout(() => { btn.textContent = "Copy link"; btn.classList.remove("copied"); }, 1600);
});

// Parameter rail: on phones it starts collapsed so the results are the
// first thing on screen; the choice is remembered. Desktop never collapses
// (the toggle is display:none there), so the class is harmless.
(() => {
  const rail = $("controls"), btn = $("rail-toggle");
  const phone = window.matchMedia("(max-width: 800px)").matches;
  let collapsed = phone;
  try {
    const saved = localStorage.getItem("nol.rail");
    if (saved) collapsed = saved === "collapsed";
  } catch { /* private mode */ }
  const paint = () => {
    rail.classList.toggle("collapsed", collapsed);
    btn.setAttribute("aria-expanded", String(!collapsed));
  };
  paint();
  btn.addEventListener("click", () => {
    collapsed = !collapsed; paint();
    try { localStorage.setItem("nol.rail", collapsed ? "collapsed" : "open"); } catch { /* ignore */ }
    if (!collapsed) rail.scrollIntoView({ block: "start", behavior: "smooth" });
  });
})();

// "How to read this page": a disclosure beside the lede, so the explanation
// is always one click away instead of a paragraph that blocks the result
// once and then is dismissed forever.
(() => {
  const btn = $("btn-howto"), box = $("howto");
  btn.addEventListener("click", () => {
    const open = box.hidden;
    box.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
    btn.textContent = open ? "Hide this" : "How to read this page";
  });
})();

// One tap-to-reveal help primitive for every [data-help] control. Native
// title tooltips never appear on touch, so on a phone the page had no
// explanations at all.
(() => {
  const bubble = $("help-bubble");
  let anchor = null;
  const close = () => { bubble.hidden = true; anchor = null; };
  const open = (el) => {
    bubble.textContent = el.dataset.help;
    bubble.hidden = false;
    const r = el.getBoundingClientRect();
    const w = Math.min(300, window.innerWidth - 24);
    bubble.style.width = w + "px";
    let left = r.left + r.width / 2 - w / 2;
    left = Math.max(12, Math.min(left, window.innerWidth - w - 12));
    bubble.style.left = left + "px";
    const below = r.bottom + 10;
    const fitsBelow = below + bubble.offsetHeight < window.innerHeight - 12;
    bubble.style.top = (fitsBelow ? below
      : Math.max(12, r.top - bubble.offsetHeight - 10)) + window.scrollY + "px";
    anchor = el;
  };
  document.addEventListener("click", (e) => {
    const el = e.target.closest("[data-help]");
    if (!el) { if (!bubble.contains(e.target)) close(); return; }
    e.preventDefault(); e.stopPropagation();
    if (anchor === el) close(); else open(el);
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  window.addEventListener("scroll", () => { if (anchor) close(); }, { passive: true });
  window.addEventListener("resize", close);
})();

// Expiry quick-picks: the short-dated regime is 2% of the slider's track,
// so landing on it by dragging is luck.
document.querySelectorAll("#maturity-quickpick .pick").forEach((btn) => {
  btn.addEventListener("click", () => {
    const v = parseFloat(btn.dataset.t);
    setSlider("maturity", v);
    state.maturity = snapMaturity(v);
    refreshAll();
  });
});

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
  btn.textContent = "…"; btn.disabled = true;
  try {
    const d = await api("/api/market/" + encodeURIComponent(t));
    marketData = d;
    rescaleSpotSliders(d.spot);
    // A range input snaps its value to the step, so a 1% volatility step
    // turned a fetched 12.9% into 13% while the chip still advertised 12.9.
    // Fine steps keep the readout, the chip and the priced inputs identical.
    $("in-sigma").step = "0.1";
    $("in-rate").step = "0.01";
    $("in-sigma").value = (d.sigma * 100).toFixed(1);
    $("in-rate").value = (d.rate * 100).toFixed(2);
    for (const id of ["in-spot", "in-strike", "in-sigma", "in-rate"])
      $(id).dispatchEvent(new Event("input"));

    chip.innerHTML =
      "<b>" + d.ticker + "</b> $" + d.spot.toLocaleString(undefined,
        { maximumFractionDigits: 2 }) +
      " · one-year realised volatility " + (d.sigma_raw * 100).toFixed(1) +
      "% · 13-week Treasury bill " + (d.rate_raw * 100).toFixed(2) + "%" +
      "<br>as of " + d.as_of.slice(0, 16).replace("T", " ") +
      (Math.abs(state.spot - d.spot) > 0.005
        ? "<br>priced at $" + state.spot.toLocaleString() +
          ", the nearest step on the spot slider"
        : "") +
      (d.clamped
        ? "<br><span class='warn'>volatility and rate adjusted to the range the model was trained on</span>"
        : "");
    chip.classList.add("show");
  } catch (err) {
    marketData = null;
    chip.innerHTML = "<span class='warn'>No market data for \"" + t +
      "\". Check the symbol, or leave it blank to keep the example contract.</span>";
    chip.classList.add("show");
    refreshReadouts();
  } finally {
    btn.textContent = "Load"; btn.disabled = false;
  }
}
$("btn-fetch-ticker").addEventListener("click", fetchTicker);
$("in-ticker").addEventListener("keydown", (e) => {
  if (e.key === "Enter") fetchTicker();
});

// ───────────────────────────────────────────────────────────── Tabs ──
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const key = Object.keys(TAB_IDS).find((k) => TAB_IDS[k] === btn.dataset.tab);
    showTab(key || "pricing");
  });
});

// ───────────────────────────────────────────────────────────── XAI ──
let lastAttributions = null;
async function updateXAI() {
  try {
    const d = await api("/api/explain", optionBody());
    clearShimmer("plot-xai");
    clearPanelMessage("plot-xai");
    lastAttributions = d.attributions;
    renderReportInputs();

    const bT = d.baseline.maturity;

    const rows = [
      { name: "Spot level", v: d.attributions.spot },
      { name: "Time to expiry", v: d.attributions.maturity },
      { name: "Volatility", v: d.attributions.sigma },
      { name: "Interest rate", v: d.attributions.rate },
    ].sort((a, b) => Math.abs(a.v) - Math.abs(b.v));

    const top = rows[rows.length - 1];
    $("xai-sub").textContent = "Starting from a minimal at-the-money option " +
      "worth $" + d.baseline_price.toFixed(2) + ", the inputs add up to this " +
      "contract's $" + d.target_price.toFixed(2) + ". " +
      top.name.replace(" (moneyness)", "") + " contributes the most.";
    if (Math.abs(d.attributions.spot) < 0.005) {
      $("xai-sub").textContent += " The spot bar is near zero because this " +
        "contract is at the money, the same as the baseline option.";
    }
    $("xai-stat").textContent = "Integrated Gradients against a baseline option " +
      "at " + (bT < 13 / 252 ? Math.round(bT * 252) + " days" : bT + " years") +
      " to expiry, 5% volatility and a zero rate" +
      (d.regime === "0dte_rough_bergomi" ? ", in the short-dated regime" : "") +
      ". The four contributions sum to the price to within $" +
      Math.abs(d.completeness_error).toFixed(4) + ".";

    Plotly.react("plot-xai", [{
      type: "bar", orientation: "h",
      y: rows.map((r) => r.name),
      x: rows.map((r) => r.v),
      marker: { color: rows.map((r) => r.v >= 0
        ? "rgba(90,140,200,0.7)" : "rgba(196,92,92,0.7)") },
      text: rows.map((r) => (r.v >= 0 ? "+" : "−") + "$" +
        Math.abs(r.v).toFixed(2)),
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
    panelMessage("plot-xai", err.message);
  }
}

// ───────────────────────────────────────────────────────────── Hedging ──
// P&L is in strike units; scale by the current strike into dollars.
let lastHedge = null;
state.hedgeCost = 0.005;
state.hedgeDynamics = "rough";
bindSegmented("hedge-dynamics", (v) => {
  state.hedgeDynamics = v; paintRailScope(); syncURL();
});

$("in-cost").addEventListener("input", () => {
  state.hedgeCost = parseFloat($("in-cost").value) / 10000;
  $("val-cost").textContent = $("in-cost").value + " bps";
  syncURL();
});

const CHIP_HELP = {
  "Learned policy": "Average loss over the worst 5% of paths (the 95% conditional value at risk) when the neural policy hedges the short call. Closer to zero is better.",
  "Delta hedge": "The same measure for a Black-Scholes delta hedge that pays the same transaction costs on every trade.",
  "Whalley-Wilmott band": "The same measure for a delta hedge that only trades when it drifts outside a cost-aware no-trade band (Whalley and Wilmott, 1997). This is the strongest classical baseline.",
  "Trading cost per path": "Average transaction costs paid over one path by the learned policy and by the delta hedge.",
  "No-arbitrage check": "Whether a butterfly spread could ever have a negative price, and whether total variance ever falls as expiry lengthens. Either would be an arbitrage. Both are checked by automatic differentiation at every grid point.",
  "Distance from the pricing model": "How far this arbitrage-free surface sits from the pricing ensemble it was fitted to, in volatility points.",
};
function hedgeStatChip(k, v, cls) {
  const help = CHIP_HELP[k] ? " title='" + CHIP_HELP[k] + "'" : "";
  return "<div class='hedge-stat'" + help + "><span class='k'>" + k +
    "</span><span class='v" + (cls ? " " + cls : "") + "'>" + v + "</span></div>";
}

async function runHedge() {
  const btn = $("btn-hedge");
  btn.textContent = "Simulating…";
  btn.disabled = true;
  // The simulation takes seconds (tens of seconds on a small host); without
  // this the panel is a blank void with only the button label as feedback.
  $("hedge-verdict").textContent = "";
  $("hedge-sub").textContent =
    "Simulating 15,000 paths and hedging the same short call three ways. " +
    "A few seconds on this server.";
  $("hedge-empty")?.remove();
  $("holdings-empty")?.remove();
  try {
    const d = await api("/api/hedge",
      { sigma: state.sigma, rate: state.rate, cost: state.hedgeCost,
        dynamics: state.hedgeDynamics });
    clearShimmer("plot-hedge");
    clearShimmer("plot-holdings");
    lastHedge = d;
    renderReportInputs();
    const K = state.strike;
    const $$ = (v) => (v < 0 ? "−$" : "$") + Math.abs(v * K).toFixed(2);

    // cvar95 is a positive loss magnitude, so the SMALLER one is the better
    // hedge. The deep hedge does not always win (the honest out-of-sample
    // result under GBM often favors delta), so the green "good" highlight
    // and the reduction/increase label both follow the measurement instead
    // of assuming the deep policy won.
    const ww = d.whalley_wilmott;
    const improvement = (1 - d.deep.cvar95 / Math.max(d.delta.cvar95, 1e-9)) * 100;
    const best = Math.min(d.deep.cvar95, d.delta.cvar95,
                          ww ? ww.cvar95 : Infinity);
    // cvar95 is a positive loss magnitude, so the SMALLEST one is the best
    // hedge. Only that one is highlighted: two green chips pointing at
    // different winners is how a reader ends up unable to tell who won.
    const pm = (se) => se ? " ± " + (se * K).toFixed(2) : "";
    $("hedge-stats").innerHTML =
      hedgeStatChip("Worst-5% loss · learned policy",
        $$(-d.deep.cvar95) + pm(d.deep.cvar95_se),
        d.deep.cvar95 === best ? "good" : "") +
      hedgeStatChip("Worst-5% loss · delta hedge",
        $$(-d.delta.cvar95) + pm(d.delta.cvar95_se),
        d.delta.cvar95 === best ? "good" : "") +
      (ww ? hedgeStatChip("Worst-5% loss · Whalley-Wilmott band",
        $$(-ww.cvar95) + pm(ww.cvar95_se), ww.cvar95 === best ? "good" : "") : "") +
      hedgeStatChip("Trading cost per path",
        $$(d.deep.mean_costs) + " vs " + $$(d.delta.mean_costs) + " for delta");

    // Say who won, in a sentence, covering every ordering the run can produce.
    const names = [["the learned policy", d.deep.cvar95],
                   ["the delta hedge", d.delta.cvar95]];
    if (ww) names.push(["the Whalley-Wilmott band", ww.cvar95]);
    names.sort((a, b) => a[1] - b[1]);
    const costBps = (d.cost * 10000).toFixed(0);
    const market = d.dynamics === "gbm"
      ? "Black-Scholes paths" : "rough-volatility paths with jumps";
    let verdict = "Over " + d.n_paths.toLocaleString() + " " + market +
      " at " + costBps + " basis points a trade, " + names[0][0] +
      " has the smallest worst-5% loss, " + $$(-names[0][1]) + ", against " +
      $$(-names[1][1]) + " for " + names[1][0] +
      (names[2] ? " and " + $$(-names[2][1]) + " for " + names[2][0] : "") + ". ";
    verdict += improvement >= 0
      ? "The learned policy beats a plain delta hedge by " +
        Math.abs(improvement).toFixed(0) + "% on tail loss while paying " +
        $$(d.deep.mean_costs) + " a path in costs against " +
        $$(d.delta.mean_costs) + "."
      : "A plain delta hedge keeps the smaller tail loss here; the learned " +
        "policy trades less (" + $$(d.deep.mean_costs) + " a path against " +
        $$(d.delta.mean_costs) + ") but that saving does not cover the wider tail.";
    $("hedge-verdict").textContent = verdict;
    $("hedge-convention").textContent =
      "Worst-5% loss is the average profit or loss across the worst 5% of " +
      "simulated paths, in dollars per option at a $" + K + " strike. Closer " +
      "to zero is better; ± is a bootstrap standard error.";
    $("hedge-method").textContent = d.measure_note || "";

    $("hedge-sub").textContent =
      "Short one 30-day at-the-money call, hedged daily on " +
      d.n_paths.toLocaleString() + " simulated paths of " +
      (d.dynamics_label || "the selected market") + ". Premium " +
      $$(d.premium) + ", " + costBps + " basis points a trade. " +
      (d.sigma_source === "SPY calibration"
        ? "Volatility (" + (d.sigma * 100).toFixed(1) + "%) and rate (" +
          (d.rate * 100).toFixed(1) + "%) come from the SPY calibration this " +
          "market was fitted to, not from the sidebar."
        : "Volatility " + (d.sigma * 100).toFixed(1) + "% and rate " +
          (d.rate * 100).toFixed(1) + "%, from the sidebar.") +
      (d.clamped ? " Inputs were clamped to the policy's trained range." : "");

    const allPnl = [...d.deep.pnl, ...d.delta.pnl,
                    ...(ww && ww.pnl ? ww.pnl : [])].map((v) => v * K);
    const span = Math.max(Math.abs(Math.min(...allPnl)), Math.abs(Math.max(...allPnl)));
    const binSize = (2 * span) / 60;

    Plotly.react("plot-hedge", [
      {
        type: "histogram", x: d.delta.pnl.map((v) => v * K),
        name: "delta hedge · worst-5% loss " + $$(-d.delta.cvar95),
        marker: { color: "rgba(196,131,92,0.45)",
                  line: { color: COLORS.mc, width: 1 } },
        xbins: { start: -span, end: span, size: binSize },
      },
      {
        type: "histogram", x: d.deep.pnl.map((v) => v * K),
        name: "learned policy · worst-5% loss " + $$(-d.deep.cvar95),
        marker: { color: "rgba(90,140,200,0.45)",
                  line: { color: COLORS.nn, width: 1 } },
        xbins: { start: -span, end: span, size: binSize },
      },
      ...(ww && ww.pnl ? [{
        type: "histogram", x: ww.pnl.map((v) => v * K),
        name: "Whalley-Wilmott band · worst-5% loss " + $$(-ww.cvar95),
        marker: { color: "rgba(136,145,163,0.35)",
                  line: { color: COLORS.violet, width: 1 } },
        xbins: { start: -span, end: span, size: binSize },
      }] : []),
    ], {
      ...PLOT_BASE, barmode: "overlay",
      xaxis: { title: { text: "profit or loss at expiry ($, strike " + K + ")" },
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
        mode: "lines+markers", name: "delta hedge",
        line: { color: COLORS.mc, width: 2 }, marker: { size: 4 },
      },
      {
        x: days, y: d.example_path.deep_holdings,
        mode: "lines+markers", name: "learned policy",
        line: { color: COLORS.nn, width: 2.5 }, marker: { size: 4 },
      },
    ], {
      ...PLOT_BASE,
      margin: { l: 52, r: 52, t: 12, b: 42 },
      xaxis: { title: { text: "trading day" }, gridcolor: COLORS.grid,
               zeroline: false },
      yaxis: { title: { text: "shares held per option" },
               gridcolor: COLORS.grid, zeroline: false, range: [0, 1.1] },
      yaxis2: { title: { text: "spot ($)" },
                overlaying: "y", side: "right", showgrid: false,
                tickfont: { color: "rgba(255,255,255,0.4)" } },
    }, PLOT_CONFIG);
  } catch (e) {
    $("hedge-verdict").textContent = "";
    $("hedge-sub").textContent = e.message;
  } finally {
    btn.textContent = "Run simulation";
    btn.disabled = false;
  }
}
$("btn-hedge").addEventListener("click", runHedge);

// ───────────────────────────────────────────────────────────── LLM ──
$("btn-risk").addEventListener("click", async () => {
  const btn = $("btn-risk");
  btn.textContent = "Writing…";
  let step = 0;
  btn.disabled = true;
  const out = $("ai-report");
  try {
    // Auto-gather any missing inputs instead of bouncing the user around.
    if (!lastAttributions) {
      out.textContent = "Working out what drives the price… (1 of 3)";
      await updateXAI();
    }
    if (!lastHedge) {
      out.textContent = "Running the hedging simulation… (2 of 3)";
      await runHedge();
    }
    out.textContent = "Writing the summary… (3 of 3)";
    if (lastNNPrice == null || !lastAttributions || !lastHedge)
      throw new Error("pricing/hedging inputs unavailable; is the backend up?");

    out.classList.add("streaming");
    const K = state.strike;
    const req = {
      // Only a successfully fetched ticker names the underlying; a failed
      // lookup used to put strings like "ZZZZQQ" into the report.
      ticker: marketData ? marketData.ticker : "",
      contract: contractShort(),
      nn_price: lastNNPrice,
      bs_cvar: -lastHedge.delta.cvar95 * K,
      deep_cvar: -lastHedge.deep.cvar95 * K,
      ww_cvar: lastHedge.whalley_wilmott
        ? -lastHedge.whalley_wilmott.cvar95 * K : null,
      deep_cost: lastHedge.deep.mean_costs * K,
      delta_cost: lastHedge.delta.mean_costs * K,
      dynamics_label: lastHedge.dynamics_label || "",
      cost_bps: Math.round(lastHedge.cost * 1e4),
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
    let firstChunk = true;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (firstChunk) { out.textContent = ""; firstChunk = false; }
      out.textContent += decoder.decode(value, { stream: true });
    }
  } catch (e) {
    out.textContent = "The summary could not be written: " + e.message;
  } finally {
    out.classList.remove("streaming");
    btn.textContent = "Write summary";
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
    $("stream-empty")?.remove();
    $("stream-stats").classList.remove("idle");
    $("stream-sub").textContent = "Connected. Starting the feed…";

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
    if (d.status === "ready") {
      // The server caps the requested rate (MAX_STREAM_HZ); show the rate it
      // actually granted. This frame has no tick fields - falling through
      // used to throw a TypeError on every connect.
      $("stream-sub").textContent = "Live: " + d.hz +
        " simulated ticks a second, each priced by the network.";
      return;
    }

    $("ws-spot").textContent = "$" + d.spot.toFixed(2);
    $("ws-spot").className = "v mono live";
    $("ws-price").textContent = "$" + d.price.toFixed(4);
    $("ws-price").className = "v mono live";
    $("ws-delta").textContent = d.delta.toFixed(4);
    $("ws-gamma").textContent = d.gamma.toFixed(4);
    if (d.rho !== undefined) $("ws-rho").textContent = d.rho.toFixed(4);
    else $("ws-rho").closest(".stream-stat").hidden = true;
    $("ws-vega").textContent = d.vega.toFixed(4);
    $("ws-theta").textContent = d.theta.toFixed(4);
    $("ws-latency").textContent = fmtMs(d.latency_us / 1000);
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
    $("stream-sub").textContent = "Disconnected. Press Connect to resume.";
    $("stream-stats").classList.add("idle");
    ws = null;
  };

  ws.onerror = () => {
    $("stream-sub").textContent = "The feed could not be reached. Press Connect to retry.";
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
const urlParams = Object.fromEntries(new URLSearchParams(location.search));
applyState(urlParams);
if (urlParams.tab && TAB_IDS[urlParams.tab]) showTab(urlParams.tab);
refreshReadouts();
loadModelInfo();
loadErrorDistribution();
(async () => {
  // A deep link with a ticker replays the live fetch (which resets the
  // spot/strike/vol/rate sliders around the market), then prices.
  if (urlParams.ticker) {
    $("in-ticker").value = String(urlParams.ticker).slice(0, 10);
    await fetchTicker().catch(() => {});
  }
  // The headline price lands first: /api/price and /api/convergence would
  // otherwise race for the server's single simulation slot, and losing that
  // race leaves the hero card blank while the convergence run finishes.
  await updatePrice().catch(() => {});
  await updateConvergence().catch(() => {});
  await updateSurface().catch(() => {});
  await updateXAI().catch(() => {});
  await updateIVSurface().catch(() => {});
  if (currentTab === "hedging" && urlParams.run === "1") runHedge();
})();
const latencyShimmer = $("plot-latency").querySelector(".shimmer");
if (latencyShimmer) {
  latencyShimmer.replaceWith(Object.assign(document.createElement("p"), {
    className: "card-sub centered latency-hint",
    textContent: "Press Measure to time the network and the simulation on this server.",
  }));
}


/* Neural Options Lab dashboard.
   Calls the FastAPI backend, draws the Plotly charts and builds every
   sentence the page renders. */

"use strict";

// State and element handles.
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

// A horizontal legend sits on the top edge of the plot, anchored by its bottom
// edge, so Plotly's auto-margin grows the top margin by the legend's height.
// A legend that wraps to two or three rows at phone width pushes the plot down
// and stays clear of the traces.
const PLOT_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, -apple-system, SF Pro Text, sans-serif", color: COLORS.ink,
          size: 11.5 },
  margin: { l: 52, r: 16, t: 12, b: 42 },
  showlegend: true,
  legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom",
            font: { size: 11 } },
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true, scrollZoom: false };
// Phone width, where legend entries are shortened so the legend stays compact.
const isNarrow = () => window.matchMedia("(max-width: 480px)").matches;

// Utilities.
const NL_CHAR = String.fromCharCode(10);

const debounce = (fn, ms) => {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};

// Escapes text bound for innerHTML or a quoted attribute.
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => "&#" + c.charCodeAt(0) + ";");

// One request in flight per panel and at most one waiting behind it. The
// server admits one simulation at a time, and a sync handler runs to its end
// after a client abort, so the in-flight request is left to finish: its
// response is the signal that the slot is free. Slider stops superseded while
// it ran are never sent. fn receives isCurrent(), which turns false once a
// newer call is waiting, and drops a response that arrives stale.
function latestOnly(fn) {
  let running = null, rerun = false;
  const drain = async () => {
    do {
      rerun = false;
      await fn(() => !rerun).catch(() => {});
    } while (rerun);
    running = null;
  };
  return () => {
    if (running) { rerun = true; return running; }
    return (running = drain());
  };
}

// Backend exception text is never shown verbatim, because it describes the
// model's internals. Each status maps to one sentence about what to change.
// `detail` is a string from the domain gate or a list of {loc, msg} from
// pydantic, and either is searched for the field that was refused.
function friendlyError(status, detail) {
  const d = typeof detail === "string" ? detail
    : Array.isArray(detail)
      ? detail.map((e) => (e.loc || []).join(".") + " " + (e.msg || "")).join("; ")
      : "";
  if (status === 422) {
    if (/sigma|volatil/i.test(d)) return "Volatility must be between 5% and 80%.";
    if (/\brate\b/i.test(d)) return "The rate must be between 0% and 10%.";
    if (/maturity|expiry/i.test(d)) {
      return "Expiry must be between one trading day and two years. Expiries "
        + "between 12 trading days and 0.05 years are covered by neither model.";
    }
    return "This contract is outside the range the models were trained on. "
      + "Move spot and strike closer together, or pick another expiry.";
  }
  if (/moneyness|domain|between|less than|greater than/i.test(d)) {
    return "This contract is outside the range the models were trained on. "
      + "Move spot and strike closer together, or pick another expiry.";
  }
  if (status === 413) return "The request is too large.";
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

// Tween a numeric readout. The value is set directly when the tab is hidden,
// where requestAnimationFrame is throttled.
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
  const p = document.createElement("p");
  p.textContent = text;
  box.appendChild(p);
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

// Controls.
function bindSlider(id, onChange) {
  const el = $("in-" + id);
  const paint = () => {
    const pct = (el.value - el.min) / (el.max - el.min) * 100;
    el.style.setProperty("--fill", pct + "%");
  };
  el.addEventListener("input", () => { paint(); onChange(parseFloat(el.value)); });
  paint();

  // The matching readout is a typed field, because a slider cannot express a
  // strike of 137.42. commit() returns whether the typed value was accepted.
  const box = $("val-" + id);
  if (!box) return;
  const commit = () => {
    const raw = box.value.trim().replace(/[%$,\s]/g, "");
    // "1.5y" and "30d" both mean something for maturity.
    const m = /^([0-9]*\.?[0-9]+)\s*([a-z]*)$/i.exec(raw);
    if (!m) { box.classList.add("invalid"); return false; }
    let v = parseFloat(m[1]);
    const unit = m[2].toLowerCase();
    if (id === "maturity") {
      if (unit === "d") v = v / 252;
      else if (unit === "m") v = v / 12;
      else if (unit === "w") v = v / 52;
      // A bare number above 3 is read as days; the slider stops at 2 years.
      else if (!unit && v > 3) v = v / 252;
    }
    // The pricer is homogeneous in spot and strike, so a positive level off
    // the slider track is valid. Both tracks are rescaled around it, as they
    // are for a fetched ticker, and the other input keeps its value. The
    // moneyness check in refreshReadouts then reports the pair.
    if ((id === "spot" || id === "strike") && isFinite(v) && v > 0 &&
        (v < parseFloat(el.min) || v > parseFloat(el.max))) {
      const other = id === "spot" ? "strike" : "spot";
      rescaleSpotSliders(v);
      $("in-" + other).step = "any";
      setSlider(other, state[other]);
    }
    const lo = parseFloat(el.min);
    const hi = parseFloat(el.max);
    // One trading day is 1/252 = 0.003968 years, just under the slider's 0.004
    // floor, and the quick-pick offers it. Values within rounding of the floor
    // snap onto it, so a typed "1d" is accepted.
    if (id === "maturity" && v < lo && v >= lo - 1e-4) v = lo;
    // Volatility and rate are shown and typed in percent, and their sliders
    // are in percent too; only the state is a fraction.
    if (!isFinite(v) || v < lo || v > hi) {
      box.classList.add("invalid");
      return false;
    }
    box.classList.remove("invalid");
    // Typed values are exact: widen the step so the browser does not round
    // 137.42 to 137 on its way into the slider.
    el.step = "any";
    el.value = v;
    paint();
    onChange(parseFloat(el.value));
    return true;
  };
  box.addEventListener("keydown", (e) => {
    // A refused value keeps the focus and its red border until it is fixed.
    if (e.key === "Enter") { e.preventDefault(); if (commit()) box.blur(); }
    if (e.key === "Escape") { box.classList.remove("invalid"); refreshReadouts(); box.blur(); }
  });
  // While a field has focus, refreshReadouts must not overwrite what is
  // being typed.
  box.addEventListener("focus", () => { box.dataset.editing = "1"; });
  box.addEventListener("blur", () => {
    commit();
    delete box.dataset.editing;
    refreshReadouts();
    // The field shows the last accepted value again, so the flag comes off.
    box.classList.remove("invalid");
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
// The short-dated network's trained volatility band, transcribed from the
// sampling bounds in backend/quant/dataset_0dte.py; the checkpoint block on
// the wire carries the moneyness and maturity box only.
const ZERO_DTE_SIGMA = [0.05, 0.80];

// A position is contracts x shares each; a negative count is a short, which
// flips the sign of the premium and of every Greek.
const positionSize = () => state.qty * state.mult;

// The export stays disabled until a price exists. A quote without one would
// still carry a timestamp and the model's identity.
function paintQuoteActions() {
  const ok = lastNNPrice != null;
  const copy = $("btn-copy-quote"), dl = $("btn-download-quote");
  if (copy) copy.disabled = !ok;
  if (dl) dl.disabled = !ok;
}

function fmtSigned(v, digits) {
  const sign = v < 0 ? "\u2212" : "";
  return sign + "$" + Math.abs(v).toLocaleString(undefined, {
    minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// Whole-share and whole-dollar figures for a book. The per-share premium keeps
// four decimals because the cross-check beside it is a gap of a few
// ten-thousandths of a dollar. Those digits are finer than the model's
// measured error, so the card sub-line prints that error in dollars next to
// the price.
function renderPosition() {
  const value = $("pos-value"), sub = $("pos-sub");
  const hint = $("position-hint");
  if (!value) return;
  const n = positionSize();
  if (lastNNPrice == null || !isFinite(n) || n === 0) {
    value.textContent = "-";
    sub.textContent = "";
  } else {
    value.textContent = fmtSigned(lastNNPrice * n, 2);
    sub.textContent = n === 1
      ? "single option, unit size"
      : (state.qty < 0 ? "short " : "") + Math.abs(state.qty).toLocaleString() +
        (Math.abs(state.qty) === 1 ? " contract × " : " contracts × ") +
        state.mult.toLocaleString() + " shares × $" +
        lastNNPrice.toFixed(4) + (state.qty < 0 ? ", premium received" : "");
  }
  if (hint) {
    hint.textContent = state.qty < 0
      ? "A short position: the premium is received and every Greek changes sign."
      : "Sets the scale of the premium and the Greeks below.";
  }
  paintQuoteActions();
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
    // A value that rounds to zero prints without a sign, so a tiny negative
    // theta reads "0.0000" and never "-0.0000".
    const raw = lastGreeks[k] * n;
    const v = Math.abs(raw) < 0.5 * Math.pow(10, -digits) ? 0 : raw;
    el.textContent = greekBasis === "position"
      ? (v < 0 ? "\u2212" : "") + Math.abs(v).toLocaleString(undefined, {
          minimumFractionDigits: 2, maximumFractionDigits: 2 })
      : v.toFixed(digits);
  }
  // In position terms each Greek is a quantity (shares, or dollars per point),
  // so the unit strings change with the basis.
  const POSITION_UNITS = {
    "g-delta": "shares of the underlying",
    "g-gamma": "shares per $1 move",
    "g-vega": "$ per volatility point",
    "g-theta": "$ per trading day",
    "g-rho": "$ per rate point",
  };
  for (const [id, unit] of Object.entries(POSITION_UNITS)) {
    const cell = $(id);
    const el = cell && cell.closest(".greek").querySelector(".greek-unit");
    if (!el) continue;
    if (!el.dataset.unit) el.dataset.unit = el.textContent;
    el.textContent = greekBasis === "position" ? unit : el.dataset.unit;
  }
  const cash = $("greek-cash");
  if (cash) {
    cash.textContent = greekBasis === "position" && lastGreeks
      ? "Cash delta " + fmtSigned(lastGreeks.delta * positionSize() * state.spot, 2) +
        ": the position's equivalent exposure to the underlying at this spot, " +
        "to first order. Gamma above says how fast it changes."
      : "";
  }
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
  // Expiries on this page are trading days, 252 to the year, and theta is
  // quoted on the same clock; the word keeps the contract sentence and the
  // risk strip on one calendar.
  if (days <= 45) return days === 1 ? "one trading-day" : days + " trading-day";
  if (T < 0.95) return Math.round(T * 12) + "-month";
  if (Math.abs(T - 1) < 0.03) return "one-year";
  if (Math.abs(T - 2) < 0.03) return "two-year";
  return T.toFixed(2) + "-year";
}

// The instrument alone, for prose that continues after it.
function contractShort() {
  const kind = is0dte() ? "European " + state.optionType
                        : "Asian " + state.optionType;
  const head = maturityWords() + " " + moneynessWords() + " " + kind;
  return marketData ? head + " on " + marketData.ticker
                    : head + " on a $" + state.spot + " stock";
}

// The contract in one sentence. It names the underlying as a loaded ticker or
// as a hypothetical stock, and carries every priced input.
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
    vol + ", rate " + rate + ".";
}

// What each tab does with the contract named above it. The Hedge tab runs its
// own instrument, and this line is where the page says so.
const CONTRACT_SCOPE = {
  hedging: "The hedge bench trades its own 30-day at-the-money call, the contract its policies were trained on. The contract above does not enter it.",
  ai: "The note is written from the price, the attribution and the hedge run listed below.",
  stream: "The feed prices this contract, tick by tick.",
};

function renderContractLine() {
  const pill = $("contract-pill");
  // Loading a ticker sets the inputs from market data. The contract stays
  // hypothetical, and the pill says so.
  pill.textContent = marketData ? marketData.ticker + ", hypothetical" : "Example";
  pill.classList.toggle("live", !!marketData);
  $("contract-text").textContent = contractSentence();
  const scope = $("contract-scope");
  const clause = CONTRACT_SCOPE[typeof currentTab === "string" ? currentTab : "pricing"];
  scope.textContent = clause || "";
  scope.hidden = !clause;
}

// Under rough volatility the hedging run supplies its own volatility and
// rate, so those two sliders are marked inactive on that tab.
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
  $("rail-summary").textContent = (marketData ? marketData.ticker + ", " : "") +
    maturityWords() + " " + moneynessWords() + " " + state.optionType +
    ", σ " + $("val-sigma").value + ", r " + $("val-rate").value;

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
  $("moneyness-val").textContent = m.toFixed(2) + (is0dte() ? ", short-dated" : "");
  const [lo, hi] = is0dte() ? [0.85, 1.15] : [0.5, 2.0];
  const outside = m < lo || m > hi;
  $("domain-warning").textContent = "Spot over strike is " + m.toFixed(2) +
    ", outside the range this model was trained on (" + lo + " to " + hi +
    "). Move spot or strike closer together.";
  $("domain-warning").classList.toggle("show", outside);

  document.querySelectorAll("#maturity-quickpick .pick").forEach((b) =>
    b.classList.toggle("active", Math.abs(+b.dataset.t - state.maturity) < 1e-6));

  // The short-dated volatility surface describes contracts of 12 trading days
  // or less, so its group opens when the maturity moves into that regime.
  const short = is0dte();
  if (short !== wasShortDated) {
    const g = $("group-domain");
    if (short && g) g.open = true;
    wasShortDated = short;
    // The accuracy teaser, the accuracy chips and the model card all quote a
    // measured error, and the model that produces the price changes here. They
    // are repainted at the regime change so each carries the number of the
    // model it names.
    paintModelScope();
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

// Price and Greeks. Every updater below takes isCurrent from latestOnly and
// returns without painting once a newer request is waiting.
let lastNNPrice = null;
let lastGreeks = null;
let lastCheck = null;
let greekBasis = "unit";
async function updatePrice(isCurrent = () => true) {
  document.querySelector(".results").classList.add("updating");
  try {
    const d = await api("/api/price", { ...optionBody(), mc_paths: state.mcPaths });
    if (!isCurrent()) return;

    lastNNPrice = d.nn.price;
    renderReportInputs();
    animateNumber($("nn-price"), d.nn.price, fmtMoney);
    animateNumber($("mc-price"), d.mc.price, fmtMoney);
    lastGreeks = d.nn.greeks;
    renderPosition();
    $("nn-sub").textContent = nnSubText();
    lastCheck = { price: d.mc.price, n_paths: d.mc.n_paths,
                  half: (d.mc.ci_high - d.mc.ci_low) / 2 };
    $("mc-ci").textContent = "±$" +
      ((d.mc.ci_high - d.mc.ci_low) / 2).toFixed(4) + " at 95%, from " +
      d.mc.n_paths.toLocaleString() + " paths with a fresh seed each run";

    // The headline is the network-to-simulation gap in basis points of
    // strike. Single-shot wall-clocks on a shared host swing several-fold
    // between identical page loads, so the timings go in the timing line and
    // carry no headline ratio.
    const diff = Math.abs(d.nn.price - d.mc.price);
    const bpsK = diff / state.strike * 1e4;
    animateNumber($("speedup"), bpsK, (v) => v.toFixed(1) + " bps");
    // Four states. At 50,000 paths the simulation's error bar is tighter than
    // the network's measured error, so a gap outside the bar and inside that
    // error is reported as neutral. The band comes from the network that
    // priced this contract (crossCheckTolBps) and the sentence names it.
    // Where the premium is near zero the gap is reported as a share of the
    // price, because basis points of strike understate it there.
    const inCI = d.comparison.within_mc_ci;
    const tol = crossCheckTolBps();
    const measured = is0dte()
      ? "the short-dated model's measured validation error"
      : "the averaged-contract ensemble's measured error on held-out contracts";
    const rel = diff / Math.max(Math.abs(d.nn.price), 1e-9);
    const agr = $("agreement");
    if (rel > 0.02) {
      agr.textContent = "$" + diff.toFixed(4) + " from the simulation, " +
        (rel * 100).toFixed(0) + "% of the price. Near zero the network's " +
        "Softplus output floor dominates, so this gap is read in dollars " +
        "and as a share of the price.";
      agr.className = "card-sub agreement-neutral";
    } else if (inCI) {
      agr.textContent = "$" + diff.toFixed(4) +
        " from the simulation, inside its 95% error bar";
      agr.className = "card-sub agreement-ok";
    } else if (bpsK <= tol) {
      agr.textContent = "$" + diff.toFixed(4) + " from the simulation, wider " +
        "than its 95% error bar and inside " + measured;
      agr.className = "card-sub agreement-neutral";
    } else {
      agr.textContent = "$" + diff.toFixed(4) + " from the simulation, wider " +
        "than its 95% error bar and wider than " + measured + ". Treat this " +
        "price as indicative, or raise the cross-check precision in the sidebar.";
      agr.className = "card-sub agreement-warn";
    }
    $("hero-error").hidden = true;
    $("timing-line").textContent = "On this server: network " +
      fmtMs(d.nn.latency_ms) + " for the price and all five Greeks, " +
      "simulation " + fmtMs(d.mc.latency_ms) + " for " +
      d.mc.n_paths.toLocaleString() + " paths" +
      (d.mc.engine === "rough_bergomi" ? ", rough-volatility engine" : "") + ".";

    // Short-dated regime: the response carries the European no-arbitrage
    // floor (discounted intrinsic) alongside the price. Where the ensemble
    // prices under that floor the card says by how much and points at the
    // constrained surface, the arbitrage-free view of the same corner. The
    // price itself is shown unchanged.
    const sub = $("nn-sub");
    if (d.nn.below_intrinsic) {
      sub.textContent = d.nn.below_intrinsic_bps_of_strike.toFixed(1) +
        " bps of strike under the no-arbitrage floor; see the " +
        "arbitrage-free surface below";
      sub.title = "Discounted intrinsic, max(S \u2212 Ke^(\u2212rT), 0), is the " +
        "lowest price a European contract at this maturity can have without " +
        "an arbitrage. The ensemble price is shown unchanged; the short-dated " +
        "volatility surface prices the same corner with butterfly and calendar " +
        "conditions imposed.";
      sub.className = "card-sub agreement-warn";
    } else {
      sub.title = "";
      sub.className = "card-sub";
    }

    document.querySelector(".results").classList.remove("errored");
    renderGreeks();
  } catch (err) {
    if (!isCurrent()) return;
    // The last good numbers stay on screen, marked stale. A red message
    // beside unmarked prices reads as if the prices belong to it.
    document.querySelector(".results").classList.add("errored");
    const slot = $("hero-error");
    slot.textContent = err.message;
    slot.hidden = false;
    $("timing-line").textContent = "";
    paintQuoteActions();
  } finally {
    if (isCurrent()) document.querySelector(".results").classList.remove("updating");
  }
}

// Convergence plot.
async function updateConvergence(isCurrent = () => true) {
  try {
    const d = await api("/api/convergence", optionBody());
    if (!isCurrent()) return;
    clearShimmer("plot-convergence");
    clearPanelMessage("plot-convergence");

    // Four full-length legend entries take four rows at phone width and leave
    // the plot half its box, so the names are shortened there. The panel's
    // description carries the long forms.
    const narrow = isNarrow();
    const xs = d.mc_points.map((p) => p.n_paths);
    const traces = [
      { // CI band (upper then lower with fill)
        x: [...xs, ...xs.slice().reverse()],
        y: [...d.mc_points.map((p) => p.ci_high),
            ...d.mc_points.map((p) => p.ci_low).reverse()],
        fill: "toself", fillcolor: "rgba(196,131,92,0.15)",
        line: { width: 0 }, hoverinfo: "skip",
        name: narrow ? "95% band" : "95% confidence interval", showlegend: true,
      },
      {
        x: xs, y: d.mc_points.map((p) => p.price),
        mode: "lines+markers",
        name: d.engine === "rough_bergomi" && !narrow
          ? "Simulation (rough volatility)" : "Simulation",
        line: { color: COLORS.mc, width: 2.5, shape: "spline" },
        marker: { size: 7, color: COLORS.mc },
        customdata: d.mc_points.map((p) => fmtMs(p.latency_ms)),
        hovertemplate: "%{x:,} paths, $%{y:.4f}<br>%{customdata}<extra></extra>",
      },
      {
        x: [xs[0], xs[xs.length - 1]], y: [d.nn.price, d.nn.price],
        mode: "lines", name: "Model price",
        line: { color: COLORS.nn, width: 2.5, dash: "dash" },
        hovertemplate: "Model price: $%{y:.4f}<extra></extra>",
      },
      {
        x: [xs[0], xs[xs.length - 1]],
        y: [d.reference.price, d.reference.price],
        mode: "lines",
        name: (narrow ? "Reference (" : "High-precision reference (") +
          Math.round(d.reference.n_paths / 1000) + (narrow ? "k)" : "k paths)"),
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
    if (isCurrent()) panelMessage("plot-convergence", err.message);
  }
}

// The 0DTE implied-vol surface depends only on (sigma, r): it is the
// no-arbitrage surrogate's smile at every maturity in the 0DTE box. The
// Durrleman butterfly function g and the calendar slope dw/dT are evaluated by
// autograd on the same grid, so both conditions are checked on every draw.
async function updateIVSurface(isCurrent = () => true) {
  const panel = $("panel-ivsurface");
  if (!panel) return;
  try {
    const d = await api("/api/iv-surface",
      { sigma: state.sigma, rate: state.rate, resolution: 41 });
    if (!isCurrent()) return;
    clearShimmer("plot-ivsurface");
    clearPanelMessage("plot-ivsurface");
    const fmtK = (v) => v.toFixed(3);
    const okB = d.g_min > 0, okC = d.calendar_min > 0;
    $("ivsurface-stats").innerHTML =
      hedgeStatChip("No-arbitrage check",
        okB && okC ? "no violations on this grid" : "violation found on this grid",
        okB && okC ? "good" : "") +
      hedgeStatChip("Distance from the pricing model",
        (d.fit && d.fit.iv_rmse_volpts_resolved != null
          ? d.fit.iv_rmse_volpts_resolved.toFixed(2) + " vol points" : "n/a"));
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
      hovertemplate: "k %{x:.3f}, %{y:.1f}d, IV %{z:.2f}%<extra></extra>",
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
    // The message goes in the plot area. The panel's description stays, since
    // a later successful draw does not rewrite it.
    if (isCurrent()) panelMessage("plot-ivsurface", err.message);
  }
}

// Latency plot.
async function updateBenchmark() {
  const btn = $("btn-benchmark");
  btn.disabled = true;
  // The initial-load block leaves a hint in this panel, and Plotly renders
  // into the same div without clearing it.
  $("plot-latency").querySelector(".latency-hint")?.remove();
  try {
    const d = await api("/api/benchmark", optionBody());
    clearShimmer("plot-latency");
    clearPanelMessage("plot-latency");

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
  } catch (err) {
    // A refused contract or a busy server is reported in the panel; without
    // this the rejection surfaces only in the console.
    panelMessage("plot-latency", err.message);
  } finally { btn.disabled = false; }
}

// Price surface plot.
async function updateSurface(isCurrent = () => true) {
  try {
    const d = await api("/api/surface", {
      sigma: state.sigma, rate: state.rate, strike: state.strike,
      option_type: state.optionType,
    });
    if (!isCurrent()) return;
    clearShimmer("plot-surface");
    clearPanelMessage("plot-surface");

    $("surface-stat").textContent = "This grid is " +
      d.n_prices.toLocaleString() + " separate prices, computed in " +
      fmtMs(d.latency_ms) + " on this server, about " +
      Math.round(d.prices_per_second / 1000).toLocaleString() +
      ",000 prices per second in a batch.";

    const norm = d.prices.map((row) => row.map((v) => v / state.strike));
    Plotly.react("plot-surface", [{
      type: "surface", x: d.moneyness, y: d.maturity, z: norm,
      colorscale: [[0, "#0e1117"], [0.45, "#2a4a6b"], [0.75, "#5a8cc8"], [1, "#8891a3"]],
      showscale: false,
      contours: { z: { show: true, usecolormap: true,
                       highlightcolor: "#fff", project: { z: true } } },
      hovertemplate: "S/K %{x:.2f}, T %{y:.2f}y<br>price/K %{z:.4f}"
        + "<br><i>click to price this contract</i><extra></extra>",
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

    const surf = $("plot-surface");
    if (!surf.dataset.clickBound) {
      surf.dataset.clickBound = "1";
      surf.on("plotly_click", (ev) => {
        const pt = ev.points && ev.points[0];
        if (!pt) return;
        const spot = +(pt.x * state.strike).toFixed(4);
        const T = +pt.y.toFixed(4);
        setSlider("spot", Math.min(Math.max(spot, +$("in-spot").min),
                                   +$("in-spot").max));
        $("in-spot").step = "any";
        $("in-spot").value = spot;
        state.spot = spot;
        setSlider("maturity", T);
        state.maturity = snapMaturity(T);
        refreshAll();
      });
    }
  } catch (err) {
    if (isCurrent()) panelMessage("plot-surface", err.message);
  }
}

// Error-distribution plot. Units are 1e-4 of the quantity: price errors are
// bps of strike; delta and vega errors are x10^-4 (per unit vol for vega).
const ERROR_METRA = {
  price: { label: "pricing error (bps of strike)", unit: "bps" },
  delta: { label: "delta error (×10⁻⁴)", unit: "×10⁻⁴" },
  // The risk strip quotes vega per volatility POINT; evaluate.py measures it
  // per 1.00 of sigma, a hundred times larger, so the axis says which.
  vega: { label: "vega error (×10⁻⁴ of strike, per 1.00 of σ)", unit: "×10⁻⁴" },
  // Measured at unit strike, so this is d²(C/K)/d(S/K)²: the risk strip's
  // per-$ gamma at a $100 strike is a hundredth of it.
  gamma: { label: "gamma error (×10⁻⁴ of strike, per unit of (S/K)²)", unit: "×10⁻⁴" },
};
let errorReport = null;
let errorMetric = "price";

function renderErrorDistribution() {
  const d = errorReport;
  if (!d) return;
  const meta = ERROR_METRA[errorMetric];
  const single = d.errors[errorMetric].single;
  const ens = d.errors[errorMetric].ensemble;

  const QUANTITY = { price: "price", delta: "delta", vega: "vega", gamma: "gamma" };
  // artifacts/eval.json is the averaged-contract ensemble's held-out set, so
  // the panel says whose error it is drawing whatever the contract on screen.
  $("error-sub").textContent =
    "How far the averaged-contract ensemble's " + QUANTITY[errorMetric] +
    " sits from a " + (d.ref_paths / 1000).toFixed(0) + ",000-path simulation, on " +
    d.n_points.toLocaleString() + " averaged contracts held out of training, " +
    "drawn from the same parameter box, so this is its error inside that box. " +
    "The short-dated model is measured separately, on the methodology page.";
  const e = d.ensemble[errorMetric];
  // The mean is a signed bias. The sentence names it and gives the scatter
  // that remains once it is removed, sqrt(rmse² − bias²).
  const bias = e.mean_bps;
  const scatter = Math.sqrt(Math.max(e.rmse_bps * e.rmse_bps - bias * bias, 0));
  const ofStrike = errorMetric === "price" ? " of strike" : "";
  // A mean is called a systematic bias only when it exceeds two standard
  // errors of the mean. Inside that, the sample resolves no direction.
  const seOfMean = scatter / Math.sqrt(Math.max(d.n_points, 1));
  const biasResolved = Math.abs(bias) > 2 * seOfMean;
  const biasClause = biasResolved
    ? "A systematic bias of " + (bias >= 0 ? "+" : "−") +
      Math.abs(bias).toFixed(1) + " " + meta.unit + " runs through it" +
      (errorMetric === "price"
        ? " (the ensemble prices " + (bias >= 0 ? "rich" : "cheap") +
          " against the simulation)" : "") +
      ", with about " + scatter.toFixed(1) + " " + meta.unit +
      " of scatter around that bias."
    : "The mean error is " + (bias >= 0 ? "+" : "−") +
      Math.abs(bias).toFixed(1) + " " + meta.unit +
      ", inside two standard errors of the mean over these " +
      d.n_points.toLocaleString() +
      " points, so the sample resolves no bias in either direction.";
  // Gamma's reference is itself a Monte Carlo estimate (a conditional
  // density at the strike), so its own noise is part of the measured gap.
  const refClause = (errorMetric === "gamma" && d.gamma_reference_se_rms_bps != null)
    ? " The reference carries " + d.gamma_reference_se_rms_bps.toFixed(1) + " " +
      meta.unit + " of Monte Carlo noise of its own (RMS), which sets the floor " +
      "this comparison can resolve."
    : "";
  $("error-stat").textContent =
    "Five averaged networks: typical error " + e.rmse_bps.toFixed(1) + " " +
    meta.unit + ofStrike + ". " + biasClause + refClause + " 95% of errors fall within " +
    e.p95_abs_bps.toFixed(1) + " " + meta.unit + ofStrike + " (one network: " +
    d.single[errorMetric].rmse_bps.toFixed(1) + " typical).";

  // Shared bins so the two histograms are directly comparable.
  const all = [...single, ...ens];
  const span = Math.max(Math.abs(Math.min(...all)), Math.abs(Math.max(...all)));
  const binSize = (2 * span) / 46;

  // At phone width the legend names the two histograms and the sentence above
  // carries their typical errors.
  const legendName = (label, s) => isNarrow() ? label
    : label + ", typical error " + s.rmse_bps.toFixed(1) + " " + meta.unit;
  const traces = [
    {
      type: "histogram", x: single,
      name: legendName("one network", d.single[errorMetric]),
      marker: { color: "rgba(143,123,255,0.5)",
                line: { color: COLORS.violet, width: 1 } },
      xbins: { start: -span, end: span, size: binSize },
    },
    {
      type: "histogram", x: ens,
      name: legendName("five averaged", d.ensemble[errorMetric]),
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

// Model badge and model card.
let modelInfo = null;

// Two networks price this page. Above 12 trading days it is the
// averaged-contract (Asian) ensemble, whose held-out error is
// artifacts/eval.json. At or below, it is a rough-Bergomi network whose
// validation error arrives as zero_dte.val_rmse_bps_of_strike. Accuracy
// sentences are built through these helpers so each names the model it quotes.
function shortDatedRmseBps() {
  const z = modelInfo && modelInfo.zero_dte;
  return z && z.available && typeof z.val_rmse_bps_of_strike === "number"
    ? z.val_rmse_bps_of_strike : null;
}

// Typical error of the model that priced the contract on screen, in basis
// points of strike: held-out for the ensemble, validation for the short-dated
// network.
function activeModelRmseBps() {
  if (!modelInfo) return null;
  return is0dte() ? shortDatedRmseBps()
    : (modelInfo.eval ? modelInfo.eval.ensemble.price.rmse_bps : null);
}

// Band the cross-check card judges the network-to-simulation gap against: the
// 95th percentile of the averaged ensemble's held-out errors above the cutoff,
// the short-dated network's validation RMSE at or below it. Each regime is
// judged against the error of the model that priced it.
function crossCheckTolBps() {
  if (is0dte()) {
    const z = shortDatedRmseBps();
    return z == null ? 4 : z;
  }
  return modelInfo && modelInfo.eval
    ? modelInfo.eval.ensemble.price.p95_abs_bps : 2.5;
}

// The price card's sub-line: what the figure is per, which contract and model
// produced it, and what that model's measured error is worth in dollars at
// this strike - the four decimals above are finer than that band.
function nnSubText() {
  const base = is0dte()
    ? "per share, standard European contract, short-dated rough-volatility model"
    : "per share, average-price contract";
  const bps = activeModelRmseBps();
  if (bps == null) return base;
  const band = bps * state.strike / 1e4;
  return base + ", typical model error ±$" +
    (band >= 0.1 ? band.toFixed(2) : band.toFixed(3)) + " at this strike";
}

// Both teasers quote a benchmark measured on a fixed set, so the sentence
// that says so is written once.
const FIXED_BENCHMARK = " These figures come from a fixed benchmark set and do " +
  "not change with the contract on screen.";

// The short-dated figures against high-precision references are the six-smile
// re-pricing in docs/no_arbitrage_surface.md, section 4 (0.79 to 3.76 bps
// RMSE, largest single strike 14.4 bps), also tabulated on the methodology
// page. They are constants of that audit: update them with it when the
// short-dated checkpoint is retrained.
function accuracyTeaserText() {
  if (!modelInfo) return "";
  if (is0dte()) {
    const r = shortDatedRmseBps();
    return (r == null
      ? "Short-dated model: its measured error is on the methodology page."
      : "Short-dated model: typical error " + r.toFixed(1) + " basis points of "
        + "strike against its 20,000-path training labels, and 0.8 to 3.8 on "
        + "six smiles re-priced against 4 x 400,000-path references, with a "
        + "largest single-strike error of 14.4 bps.")
      + FIXED_BENCHMARK + " The chart below belongs to the averaged-contract "
      + "ensemble.";
  }
  const e = modelInfo.eval;
  if (!e) return "";
  return "Averaged-contract ensemble: typical error " +
    e.ensemble.price.rmse_bps.toFixed(1) + " basis points of strike on " +
    e.n_points.toLocaleString() + " held-out contracts against " +
    (e.ref_paths / 1000).toFixed(0) + ",000-path references. Individual " +
    "contracts run higher: " +
    e.ensemble.price.p95_abs_bps.toFixed(1) + " bps at the 95th percentile, " +
    e.ensemble.price.max_abs_bps.toFixed(1) + " bps at the worst point measured." +
    FIXED_BENCHMARK;
}

// Rows of the model-badge popover. The trained box differs by regime and the
// sidebar warning quotes the one in force, so that box is named first and
// both are on the card.
function modelCardRows() {
  const m = modelInfo;
  const p = m.param_ranges || {};
  const z = m.zero_dte;
  const zr = shortDatedRmseBps();
  const pairs = [
    ["Architecture, averaged contract",
      (m.n_members > 1 ? m.n_members + " networks, " : "One network, ") +
      m.n_parameters.toLocaleString() + " parameters each"],
    ["Training data, averaged contract", m.n_samples.toLocaleString() +
      " contracts labelled by Monte Carlo" + (m.mc_paths_per_label
        ? " at " + m.mc_paths_per_label.toLocaleString() + " paths each" : "")],
  ];
  if (m.eval) {
    pairs.push(["Accuracy, averaged contract", "typical pricing error " +
      m.eval.ensemble.price.rmse_bps.toFixed(1) + " basis points of strike, on " +
      m.eval.n_points.toLocaleString() + " held-out contracts against " +
      (m.eval.ref_paths / 1000).toFixed(0) + ",000-path references"]);
  }
  if (zr != null) {
    pairs.push(["Accuracy, short-dated model", "typical pricing error " +
      zr.toFixed(1) + " basis points of strike against its 20,000-path " +
      "training labels"]);
  }
  const asianBox = (p.moneyness && p.maturity && p.sigma)
    ? ["Trained range, averaged contract", "spot over strike " + p.moneyness[0] +
       " to " + p.moneyness[1] + ", expiry " + p.maturity[0] + " to " +
       p.maturity[1] + " years, volatility " + Math.round(p.sigma[0] * 100) +
       "% to " + Math.round(p.sigma[1] * 100) + "%"]
    : null;
  const shortBox = (z && z.available && z.moneyness &&
      typeof z.maturity_floor_years === "number" &&
      typeof z.maturity_cutoff_years === "number")
    ? ["Trained range, short-dated model", "spot over strike " + z.moneyness[0] +
       " to " + z.moneyness[1] + ", expiry " +
       Math.round(z.maturity_floor_years * 252) + " to " +
       Math.round(z.maturity_cutoff_years * 252) + " trading days, volatility " +
       Math.round(ZERO_DTE_SIGMA[0] * 100) + "% to " +
       Math.round(ZERO_DTE_SIGMA[1] * 100) + "%"]
    : null;
  for (const row of (is0dte() ? [shortBox, asianBox] : [asianBox, shortBox]))
    if (row) pairs.push(row);
  // The short-dated checkpoint carries its own provenance: whether its
  // rough-Bergomi parameters came from an accepted market calibration, and
  // which one. Every field comes from the checkpoint.
  if (z && z.available) {
    const hurst = typeof z.H === "number" ? ", Hurst index " + z.H.toFixed(3) : "";
    pairs.push(["Short-dated model", (z.calibrated
      ? "rough Bergomi calibrated to market option prices"
      : "rough Bergomi with default parameters, without a market calibration") + hurst +
      (z.calibration_note ? ". " + esc(z.calibration_note) : "")]);
  }
  return pairs;
}

// Everything that quotes a model's identity or its measured error. Called when
// the model info lands and again whenever the pricing regime changes.
function paintModelScope() {
  const m = modelInfo;
  if (!m) return;
  const body = $("model-card-body");
  if (body) body.innerHTML = modelCardRows().map(([k, v]) =>
    "<dt>" + k + "</dt><dd>" + v + "</dd>").join("");
  const teaser = $("accuracy-teaser");
  if (teaser) teaser.textContent = accuracyTeaserText();
  // The price card names its model and its error band too. Leave it alone
  // while it is carrying the no-arbitrage-floor warning, which owns the slot.
  const sub = $("nn-sub");
  if (sub && lastNNPrice != null && sub.className === "card-sub")
    sub.textContent = nnSubText();
  const acc = $("accuracy-stats");
  if (!acc) return;
  const z = m.zero_dte;
  const zr = shortDatedRmseBps();
  acc.innerHTML = is0dte()
    ? hedgeStatChip("Pricing this contract", "short-dated rough-volatility model") +
      (z && z.n_members ? hedgeStatChip("Ensemble", z.n_members + " networks") : "") +
      (zr != null ? hedgeStatChip("Typical error, short-dated model",
        zr.toFixed(1) + " bps of strike") : "") +
      hedgeStatChip("Chart below", "averaged-contract ensemble")
    : hedgeStatChip("Pricing this contract", "averaged-contract ensemble") +
      hedgeStatChip("Ensemble", m.n_members + " networks") +
      hedgeStatChip("Parameters", m.n_parameters.toLocaleString() + " each") +
      hedgeStatChip("Training set",
        m.n_samples.toLocaleString() + " Monte Carlo-labelled contracts") +
      (m.eval ? hedgeStatChip("Typical error, averaged contract",
        m.eval.ensemble.price.rmse_bps.toFixed(1) + " bps of strike") : "");
}

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
    paintModelScope();
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
  } catch {
    dot.className = "status-dot bad";
    txt.textContent = "Server unreachable";
    body.innerHTML = rows([["Status", "The server did not respond."]]);
  }
}

// The model card opens on click and closes on the next click outside it or
// on Escape.
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

// Wiring. Each panel gets a single-flight runner, and the debounced refreshes
// call the runners.
const runPrice = latestOnly(updatePrice);
const runConvergence = latestOnly(updateConvergence);
const runSurface = latestOnly(updateSurface);
const runXAI = latestOnly(updateXAI);
const runIVSurface = latestOnly(updateIVSurface);

// The two WebGL surfaces sit in the collapsed range group, and each redraw
// rebuilds a 3D scene on the main thread. While the group is closed a slider
// stop marks them stale, and the redraw happens when the group opens.
let surfacesStale = false;
function refreshSurfaces() {
  const g = $("group-domain");
  if (g && !g.open) { surfacesStale = true; return; }
  surfacesStale = false;
  runSurface();
  runIVSurface();
}

const refreshFast = debounce(runPrice, 220);
const refreshSlow = debounce(() => { runConvergence(); runXAI(); refreshSurfaces(); }, 650);
const refreshAll = () => { refreshReadouts(); refreshFast(); refreshSlow(); syncURL(); };

// A chart drawn while its group was collapsed is laid out against a box with
// no width, so it is resized when the group opens.
document.querySelectorAll("details.group-details").forEach((g) => {
  g.addEventListener("toggle", () => {
    if (!g.open) return;
    g.querySelectorAll(".js-plotly-plot").forEach((p) => Plotly.Plots.resize(p));
    if (g.id === "group-domain" && surfacesStale) refreshSurfaces();
  });
});

bindSlider("spot", (v) => { state.spot = v; refreshAll(); });
bindSlider("strike", (v) => { state.strike = v; refreshAll(); });
// Maturities strictly between the 0DTE cutoff (12/252) and the Asian net's
// 0.05y training floor are covered by neither model, and the API rejects them
// with a 422. The slider's 0.001 grid has two such positions (0.048 and
// 0.049), so the value snaps to the nearer valid endpoint: 0.047, which is
// priced as 12 trading days, or 0.05.
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

// Solves for the volatility input that reproduces a typed premium.
async function solveImpliedVol() {
  const box = $("in-target-price"), note = $("iv-solve-note");
  const btn = $("btn-solve-iv");
  const target = parseFloat(box.value.replace(/[$,\s]/g, ""));
  if (!isFinite(target) || target < 0) {
    box.classList.add("invalid");
    note.textContent = "Type the premium you want to match.";
    return;
  }
  box.classList.remove("invalid");
  btn.disabled = true;
  btn.textContent = "Solving";
  note.textContent = "";
  try {
    const d = await api("/api/implied-vol", {
      spot: state.spot, strike: state.strike, maturity: state.maturity,
      rate: state.rate, option_type: state.optionType, price: target,
    });
    if (!d.bracketed) {
      // No volatility in the trained range produces this premium. The note
      // gives the reachable price range and the volatility stays as it was.
      note.textContent = "No volatility between " +
        (d.search_range[0] * 100).toFixed(0) + "% and " +
        (d.search_range[1] * 100).toFixed(0) + "% prices this contract at $" +
        d.target_price.toFixed(4) + ". Across that range it spans $" +
        d.price_range[0].toFixed(4) + " to $" + d.price_range[1].toFixed(4) +
        "; the closest is $" + d.price_at_sigma.toFixed(4) + " at " +
        (d.sigma * 100).toFixed(1) + "%. Volatility left unchanged.";
      return;
    }
    const pct = d.sigma * 100;
    $("in-sigma").step = "any";
    $("in-sigma").value = pct;
    state.sigma = d.sigma;
    setSlider("sigma", pct);
    $("in-sigma").value = pct;
    refreshAll();
    // The solver returns the volatility input that reproduces the premium,
    // and its last digits are inside the pricer's error. Vega converts that
    // error into volatility points, and the note quotes that width.
    const bps = activeModelRmseBps();
    const vega = lastGreeks ? lastGreeks.vega : null;
    const volPts = (bps != null && vega && Math.abs(vega) > 1e-9)
      ? Math.abs(bps * state.strike / 1e4 / vega) : null;
    note.textContent = "$" + d.target_price.toFixed(4) + " is reproduced by " +
      pct.toFixed(1) + "% volatility " + (is0dte()
        ? "in the short-dated model, where σ sets the rough-volatility " +
          "forward variance (ξ₀ = σ²). It is a different quantity from a " +
          "Black-Scholes implied volatility."
        : "in this average-price model. It is this model's volatility input " +
          "and a different quantity from a market-quoted implied volatility.") +
      (volPts != null
        ? " The model's own " + bps.toFixed(1) + " bps price error moves it by " +
          "about " + volPts.toFixed(2) + " of a volatility point."
        : "");
  } catch (err) {
    note.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Solve";
  }
}
$("btn-solve-iv").addEventListener("click", solveImpliedVol);
$("in-target-price").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); solveImpliedVol(); }
});

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

// The exported quote: the contract, every input, the price, the Greeks, the
// Monte Carlo check and when it was produced. The check runs the same model's
// dynamics through a different numerical method, so the rows label it a
// cross-check. It is the same valuation model computed a second way.
function quoteRows() {
  const g = lastGreeks || {};
  const n = positionSize();
  // This row leads the clipboard quote and the downloaded CSV. The stamp is
  // the browser's local clock and names its zone. sv-SE formats the date in
  // ISO order.
  const stamp = new Date().toLocaleString("sv-SE", { hour12: false });
  const zone = Intl.DateTimeFormat().resolvedOptions().timeZone || "local time";
  const rows = [
    ["Produced", stamp + " " + zone],
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
    // The pricer values an option on one unit of underlying, so the price and
    // the five Greeks below are per share. The whole-position rows scale them
    // by contracts times shares per contract.
    ["Price per share", lastNNPrice == null ? "" : lastNNPrice.toFixed(6)],
    ["Position value", lastNNPrice == null ? "" : (lastNNPrice * n).toFixed(2)],
    ["Delta per share", g.delta == null ? "" : g.delta.toFixed(6)],
    ["Gamma per share", g.gamma == null ? "" : g.gamma.toFixed(6)],
    ["Vega per share", g.vega == null ? "" : g.vega.toFixed(6)],
    ["Theta per share, per trading day", g.theta == null ? "" : g.theta.toFixed(6)],
    ["Rho per share", g.rho == null ? "" : g.rho.toFixed(6)],
    ["Delta, whole position", g.delta == null ? "" : (g.delta * n).toFixed(2)],
    ["Gamma, whole position", g.gamma == null ? "" : (g.gamma * n).toFixed(2)],
    ["Vega, whole position", g.vega == null ? "" : (g.vega * n).toFixed(2)],
    ["Theta, whole position, per trading day", g.theta == null ? "" : (g.theta * n).toFixed(2)],
    ["Rho, whole position", g.rho == null ? "" : (g.rho * n).toFixed(2)],
  ];
  if (lastCheck) {
    rows.push(["Cross-check price", lastCheck.price.toFixed(6)]);
    rows.push(["Cross-check paths", lastCheck.n_paths]);
    rows.push(["Cross-check 95% half-width", lastCheck.half.toFixed(6)]);
  }
  if (modelInfo) {
    // The network that priced this quote, and that network's error. At or
    // below 12 trading days the short-dated ensemble produced the price row,
    // so its error is the one exported.
    rows.push(["Pricing model", is0dte()
      ? "short-dated rough-volatility ensemble"
      : modelInfo.n_members + " networks x " + modelInfo.n_parameters +
        " parameters, averaged-contract ensemble"]);
    const bps = activeModelRmseBps();
    if (bps != null) {
      rows.push(["Pricing model typical error",
        bps.toFixed(2) + " bps of strike"]);
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

// URL state, presets and sharing. Every slider, the contract type, the tab and
// the hedging cost are mirrored into the query string so a specific finding
// can be sent as a link, e.g.
//   /?tab=pricing&spot=160&strike=100&T=1&sigma=0.25&rate=0.04&type=put
const TAB_IDS = { pricing: "tab-pricing", stream: "tab-stream",
                  hedging: "tab-hedging", ai: "tab-ai" };
// The tab labels read Quote / Hedge / Live / Desk note, and the URL keys are
// pricing / hedging / stream / ai. Shared links carry those keys and
// serializeState writes them, so the label vocabulary is accepted on read only.
const TAB_ALIAS = { quote: "pricing", price: "pricing", hedge: "hedging",
                    live: "stream", monitor: "stream", note: "ai",
                    desk: "ai", report: "ai" };
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
  // Each pane's content starts at the top, so the scroll position resets
  // with the tab.
  window.scrollTo(0, 0);
  // Charts drawn or resized while this pane was hidden were laid out against
  // a box with no width; resize them once the pane is visible.
  requestAnimationFrame(() => {
    document.querySelectorAll("#" + id + " .js-plotly-plot")
      .forEach((p) => Plotly.Plots.resize(p));
  });
  syncURL();
}

// Write summary runs the attribution and the hedging simulation first when
// either is missing. These chips show which inputs are ready and which will
// be computed.
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

// Parameter rail. Up to 1100px wide, where the rail stacks above the stage, it
// starts collapsed so the results are the first thing on screen, and the
// choice is remembered. This width must equal the breakpoint of the
// .rail.collapsed rules in styles.css. Wider layouts hide the toggle, and the
// class has no effect there.
(() => {
  const rail = $("controls"), btn = $("rail-toggle");
  const stacked = window.matchMedia("(max-width: 1100px)").matches;
  let collapsed = stacked;
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

// "What this tool does" is a disclosure beside the lede, so the explanation
// stays one click away and takes no room above the result.
(() => {
  const btn = $("btn-howto"), box = $("howto");
  btn.addEventListener("click", () => {
    const open = box.hidden;
    box.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
    btn.textContent = open ? "Hide this" : "What this tool does";
  });
})();

// One tap-to-reveal help bubble for every [data-help] control. Native title
// tooltips do not appear on touch screens.
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

// Expiry quick-picks. The short-dated regime is 2% of the slider's track and
// hard to reach by dragging.
document.querySelectorAll("#maturity-quickpick .pick").forEach((btn) => {
  btn.addEventListener("click", () => {
    const v = parseFloat(btn.dataset.t);
    setSlider("maturity", v);
    state.maturity = snapMaturity(v);
    refreshAll();
  });
});

// Ticker lookup. The pricer works in moneyness, so any spot level is exact.
// The spot and strike sliders are rescaled around the fetched price and the
// strike is set at the money.
let marketData = null;

// backend/quant/market_data.py returns the symbol the risk-free rate came
// from: ^IRX first, ^TNX when that fails.
const RATE_SOURCE_NAMES = {
  "^IRX": "13-week Treasury bill (^IRX)",
  "^TNX": "10-year Treasury note (^TNX)",
};

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
    // A range input snaps its value to the step. At a 1% step a fetched 12.9%
    // volatility becomes 13% while the chip shows 12.9, so the steps are made
    // fine enough to keep the readout, the chip and the priced inputs equal.
    $("in-sigma").step = "0.1";
    $("in-rate").step = "0.01";
    $("in-sigma").value = (d.sigma * 100).toFixed(1);
    $("in-rate").value = (d.rate * 100).toFixed(2);
    for (const id of ["in-spot", "in-strike", "in-sigma", "in-rate"])
      $(id).dispatchEvent(new Event("input"));

    // market_data tries ^IRX and falls back to ^TNX, the 10-year, and reports
    // which in rate_source. The chip names the instrument from that field.
    // The quote enters the model as a continuously compounded rate without
    // conversion, hence "used as".
    const rateName = RATE_SOURCE_NAMES[d.rate_source] ||
      ("Treasury yield" + (d.rate_source ? " (" + d.rate_source + ")" : ""));
    chip.innerHTML =
      "<b>" + esc(d.ticker) + "</b> $" + d.spot.toLocaleString(undefined,
        { maximumFractionDigits: 2 }) +
      ", one-year realised volatility " + (d.sigma_raw * 100).toFixed(1) +
      "%, " + esc(rateName) + " " + (d.rate_raw * 100).toFixed(2) +
      "%, used as the model's continuously compounded rate" +
      "<br>as of " + esc(d.as_of.slice(0, 16).replace("T", " ")) + " " +
      esc(d.as_of_tz || "UTC") + ", the time the server fetched it" +
      // The spot is a trade only when the quote endpoint answered. Otherwise
      // it is the previous session's close, as it always is outside market
      // hours, and the chip says which.
      (d.spot_source === "last_close"
        ? "; the spot is the last daily close"
        : d.spot_source === "last_price" ? "; the spot is the last trade" : "") +
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
    chip.innerHTML = "<span class='warn'>No market data for \"" + esc(t) +
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

// Tabs.
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const key = Object.keys(TAB_IDS).find((k) => TAB_IDS[k] === btn.dataset.tab);
    showTab(key || "pricing");
  });
});

// Price attribution (Integrated Gradients).
let lastAttributions = null;
// The desk note quotes the Integrated Gradients baseline by value, so the
// baseline_price this panel prints is kept beside the attributions and the
// note needs no second /api/explain call.
let lastBaselinePrice = null;
// The panel's static copy, restored when a contract has no attribution.
const XAI_SUB_DEFAULT = $("xai-sub").textContent;
const XAI_STAT_DEFAULT = $("xai-stat").textContent;
async function updateXAI(isCurrent = () => true) {
  try {
    const d = await api("/api/explain", optionBody());
    if (!isCurrent()) return;
    clearShimmer("plot-xai");
    clearPanelMessage("plot-xai");
    lastAttributions = d.attributions;
    lastBaselinePrice = d.baseline_price;
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
      top.name + " contributes the most.";
    if (Math.abs(d.attributions.spot) < 0.005) {
      $("xai-sub").textContent += " The spot bar is near zero because this " +
        "contract is at the money, the same as the baseline option.";
    }
    // Integrated Gradients is complete against its baseline: the four
    // contributions sum to the price minus the baseline option, so the
    // baseline has to be in the sentence for the arithmetic to close.
    const bDays = Math.round(bT * 252);
    $("xai-stat").textContent = "Integrated Gradients against a baseline option " +
      "at " + (bT < 13 / 252 ? bDays + (bDays === 1 ? " trading day" : " trading days")
                             : bT + (bT === 1 ? " year" : " years")) +
      " to expiry, 5% volatility and a zero rate" +
      (d.regime === "0dte_rough_bergomi" ? ", in the short-dated regime" : "") +
      ". The four contributions plus that baseline's $" +
      d.baseline_price.toFixed(2) + " reproduce the price to within $" +
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
    if (!isCurrent()) return;
    // A refused contract has no attribution. The previous contract's is
    // dropped so that neither this panel nor the desk note can present it as
    // this contract's, and the plot area says why (for a 422, that the
    // contract is outside the trained range).
    lastAttributions = null;
    lastBaselinePrice = null;
    renderReportInputs();
    $("xai-sub").textContent = XAI_SUB_DEFAULT;
    $("xai-stat").textContent = XAI_STAT_DEFAULT;
    panelMessage("plot-xai", err.message);
  }
}

// Hedging. P&L is in strike units and is scaled by the current strike into
// dollars.
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

// Help text, keyed by the chip label as rendered.
const CHIP_HELP = {
  "Worst-5% loss, learned policy": "Average loss over the worst 5% of paths (the 95% conditional value at risk) when the neural policy hedges the short call. Closer to zero is better. The line beneath gives its mean profit or loss and its transaction costs per path.",
  "Worst-5% loss, delta hedge": "The same measure for a Black-Scholes delta hedge that pays the same transaction costs on every trade.",
  "Worst-5% loss, Whalley-Wilmott band": "The same measure for a delta hedge that trades only when it drifts outside a cost-aware no-trade band (Whalley and Wilmott, 1997). The band's risk aversion is tuned on a separate block of paths before the comparison.",
  "No-arbitrage check": "Whether a butterfly spread could ever have a negative price, and whether total variance ever falls as expiry lengthens. Either would be an arbitrage. Both are evaluated by automatic differentiation at every point of the displayed grid. The penalties that produced the surface are soft, so the check covers this grid and carries no guarantee between its points.",
  "Distance from the pricing model": "How far this arbitrage-free surface sits from the pricing ensemble it was fitted to, in volatility points.",
};
// A stat chip: label k, figure v, an optional class on the figure and an
// optional line of small print under it. Help uses the page's one affordance,
// a "?" button with data-help, so it opens in the tap bubble on touch screens.
function hedgeStatChip(k, v, cls, sub) {
  const help = CHIP_HELP[k]
    ? "<button type='button' class='help' data-help='" + esc(CHIP_HELP[k]) + "'>?</button>"
    : "";
  return "<div class='hedge-stat'><span class='k'>" + k + help +
    "</span><span class='v" + (cls ? " " + cls : "") + "'>" + v + "</span>" +
    (sub ? "<span class='k' style='display:block;margin-top:6px'>" + sub + "</span>" : "") +
    "</div>";
}

// Which hedger pairs the paired bootstrap separates, keyed by the short names
// the desk note uses, so both tabs rank the hedgers with one test. The backend
// keys pairs by strategy ("deep", "delta", "whalley_wilmott", "linear") in
// generation order; this normalises to a sorted key over the three policies
// the note covers. Returns null when the response carries no paired
// statistics, and the note then applies its two-standard-error test.
const PAIRED_SHORT = { deep: "deep", delta: "delta", whalley_wilmott: "band" };

// Policy names are written lower-case so they read inside a sentence; this
// lifts one that has to open its own.
const capFirst = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// Hedge P&L arrives in units of strike. These format it in dollars at strike K.
const hedgeDollars = (K) => (v) => (v < 0 ? "−$" : "$") + Math.abs(v * K).toFixed(2);
const hedgeError = (K) => (se) => (se ? " ± " + (se * K).toFixed(2) : "");

function pairedSeparationMap(hedge) {
  const pairs = hedge && hedge.paired_bootstrap && hedge.paired_bootstrap.pairs;
  if (!pairs) return null;
  const out = {};
  for (const key of Object.keys(pairs)) {
    const [a, b] = key.split("|");
    const sa = PAIRED_SHORT[a], sb = PAIRED_SHORT[b];
    if (!sa || !sb) continue;
    out[[sa, sb].sort().join("|")] = !!pairs[key].excludes_zero;
  }
  return Object.keys(out).length ? out : null;
}

// Returns the verdict paragraph and `winner`, the key of the hedger it names
// as best, or null when it names none. cvar95 is a positive loss magnitude, so
// the smallest is the best hedge. Every hedger ran on the same paths, so the
// backend resamples those paths once per replicate and reports the sampling
// error of each difference. Two hedgers are separated when the 95% paired
// bootstrap interval of their difference excludes zero. The opening sentence,
// the closing clause and the green chip all apply that one test, so the
// paragraph names a winner only where the chips show one.
function hedgeVerdict(d, K) {
  const usd = hedgeDollars(K), pm = hedgeError(K);
  const entry = (label, key, s) =>
    ({ label, key, cvar: s.cvar95, se: s.cvar95_se || 0 });
  const deep = entry("the learned policy", "deep", d.deep);
  const delta = entry("the delta hedge", "delta", d.delta);
  const ranked = [deep, delta];
  if (d.whalley_wilmott) {
    ranked.push(entry("the Whalley-Wilmott band", "whalley_wilmott",
                      d.whalley_wilmott));
  }
  ranked.sort((a, b) => a.cvar - b.cvar);

  const pairs = d.paired_bootstrap && d.paired_bootstrap.pairs;
  const pairStat = (a, b) =>
    (pairs && (pairs[a.key + "|" + b.key] || pairs[b.key + "|" + a.key])) || null;
  // Without paired statistics the test is two combined standard errors.
  const separated = (a, b) => {
    const s = pairStat(a, b);
    return s ? !!s.excludes_zero
             : Math.abs(a.cvar - b.cvar) > 2 * Math.hypot(a.se, b.se);
  };
  const quote = (h) => usd(-h.cvar) + pm(h.se);
  // The gap is quoted with the standard error of the paired difference.
  // Combining the two chips' error bars would overstate that error, because a
  // path that is bad for one hedger is usually bad for all of them.
  const gapSentence = (lead, a, b) => {
    const s = pairStat(a, b);
    if (!s) {
      return lead + (separated(a, b) ? " is wider than" : " is inside") +
        " two combined bootstrap standard errors.";
    }
    return lead + " is $" + Math.abs(s.diff * K).toFixed(2) + " ± " +
      (s.se * K).toFixed(2) + " on the same paths, and its 95% paired " +
      "bootstrap interval " + (s.excludes_zero ? "excludes" : "contains") +
      " zero.";
  };

  const [first, second, third] = ranked;
  const called = separated(first, second);
  const market = d.dynamics === "gbm"
    ? "Black-Scholes paths" : "rough-volatility paths with jumps";
  let text = "Over " + d.n_paths.toLocaleString() + " " + market + " at " +
    (d.cost * 10000).toFixed(0) + " basis points a trade, ";
  if (called) {
    text += first.label + " has the smallest worst-5% loss, " + quote(first) +
      ", against " + quote(second) + " for " + second.label +
      (third ? " and " + quote(third) + " for " + third.label : "") + ". " +
      gapSentence("The gap to " + second.label, first, second) + " ";
  } else {
    text += first.label + " and " + second.label + " are level on " +
      "worst-5% loss, " + quote(first) + " and " + quote(second) + ". " +
      gapSentence("The gap between them", first, second) + " ";
    if (third) {
      text += capFirst(third.label) + (separated(first, third)
        ? " is behind at " + quote(third) + ". "
        : ", at " + quote(third) + ", is not separated from " + first.label +
          " either. ");
    }
  }

  // The comparison with the delta hedge is gated on the same test, and the
  // closing clause places the learned policy against the leader with it too.
  const costs = usd(d.deep.mean_costs) + " a path in costs against " +
    usd(d.delta.mean_costs) + " for the delta hedge";
  const pct = Math.abs((1 - deep.cvar / Math.max(delta.cvar, 1e-9)) * 100).toFixed(0);
  if (!separated(deep, delta)) {
    text += "The learned policy and the delta hedge are level on worst-5% " +
      "loss, and the policy pays " + costs + ".";
  } else if (deep.cvar < delta.cvar) {
    text += "The learned policy's worst-5% loss is " + pct + "% smaller than " +
      "the delta hedge's, and it pays " + costs + ".";
    if (first !== deep) {
      text += separated(first, deep)
        ? " " + capFirst(first.label) + " is ahead of it at this cost level."
        : " It is level with " + first.label + " at this cost level.";
    }
  } else {
    // Worst-5% loss is net of costs, so a cheaper policy with the wider tail
    // has that saving counted already.
    text += "The learned policy's worst-5% loss is " + pct + "% larger than " +
      "the delta hedge's. It pays " + costs +
      (d.deep.mean_costs < d.delta.mean_costs
        ? ", and the worst-5% loss is already net of those costs." : ".");
  }
  return { text, winner: called ? first.key : null };
}

async function runHedge() {
  const btn = $("btn-hedge");
  btn.textContent = "Simulating...";
  btn.disabled = true;
  // The simulation takes seconds (tens of seconds on a small host), so the
  // panel says what is running while it waits.
  $("hedge-verdict").textContent = "";
  $("hedge-sub").textContent =
    "Simulating paths and hedging the same short call three ways. " +
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
    const $$ = hedgeDollars(K), pm = hedgeError(K);
    const ww = d.whalley_wilmott;
    const costBps = (d.cost * 10000).toFixed(0);

    // One chip per hedger. Its figure is green only when the verdict names
    // that hedger as best, so with the top two level no chip is green. The
    // line under each figure gives the hedger's mean P&L and trading cost: a
    // worst-5% loss is the mean plus the tail beyond it, and the two lines
    // together show which of the two a gap comes from.
    const verdict = hedgeVerdict(d, K);
    const tailChip = (label, key, s) => hedgeStatChip(
      "Worst-5% loss, " + label, $$(-s.cvar95) + pm(s.cvar95_se),
      verdict.winner === key ? "good" : "",
      "average P&L " + $$(s.mean) + ", trading cost " + $$(s.mean_costs) + " a path");
    $("hedge-stats").innerHTML =
      tailChip("learned policy", "deep", d.deep) +
      tailChip("delta hedge", "delta", d.delta) +
      (ww ? tailChip("Whalley-Wilmott band", "whalley_wilmott", ww) : "");

    $("hedge-verdict").textContent = verdict.text;
    $("hedge-convention").textContent =
      "Worst-5% loss is the average profit or loss across the worst 5% of " +
      "simulated paths, in dollars per option at a $" + K + " strike. Closer " +
      "to zero is better; ± is a bootstrap standard error. Average P&L is " +
      "the mean over all paths after costs, so a worst-5% loss is that mean " +
      "plus the tail beyond it.";
    // The served note states the vol-matching. This sentence adds its
    // consequence: the realised volatility is information a live hedger
    // lacks, so the protocol favours the baselines.
    $("hedge-method").textContent = (d.measure_note || "") +
      " The volatility these paths realise is information a live hedger " +
      "would not have, so the learned policy has to beat each baseline at " +
      "its strongest.";

    $("hedge-sub").textContent =
      "Short one 30-day at-the-money call, hedged daily on " +
      d.n_paths.toLocaleString() + " simulated paths of " +
      (d.dynamics_label || "the selected market") + ". Premium " +
      $$(d.premium) + " per option at a $" + K + " strike, and a proportional " +
      "cost of " + costBps + " basis points of the notional traded on every " +
      "trade. " +
      (d.sigma_source === "SPY calibration"
        ? "Volatility (" + (d.sigma * 100).toFixed(1) + "%) and rate (" +
          (d.rate * 100).toFixed(1) + "%) are those of the SPY calibration " +
          "this market was fitted to. Neither is read from the sidebar."
        : "Volatility " + (d.sigma * 100).toFixed(1) + "% and rate " +
          (d.rate * 100).toFixed(1) + "%, from the sidebar.") +
      (d.clamped ? " Inputs were clamped to the policy's trained range." : "");

    const allPnl = [...d.deep.pnl, ...d.delta.pnl,
                    ...(ww && ww.pnl ? ww.pnl : [])].map((v) => v * K);
    const span = Math.max(Math.abs(Math.min(...allPnl)), Math.abs(Math.max(...allPnl)));
    const binSize = (2 * span) / 60;

    // Plotly wraps a horizontal legend between entries, never inside one, and
    // gives every entry the width of the longest. At phone width the legend
    // names the hedger alone and abbreviates the band, which keeps it to two
    // rows; the chips above hold the figures and the full names.
    const narrow = isNarrow();
    const bandName = narrow ? "W-W band" : "Whalley-Wilmott band";
    const legendName = (label, s) =>
      narrow ? label : label + ", worst-5% loss " + $$(-s.cvar95);
    // One dotted guide per histogram, at that hedger's worst-5% loss.
    const guide = (s, color) => ({
      type: "line", x0: -s.cvar95 * K, x1: -s.cvar95 * K, y0: 0, y1: 1,
      yref: "paper", line: { color, width: 2, dash: "dot" } });

    Plotly.react("plot-hedge", [
      {
        type: "histogram", x: d.delta.pnl.map((v) => v * K),
        name: legendName("delta hedge", d.delta),
        marker: { color: "rgba(196,131,92,0.45)",
                  line: { color: COLORS.mc, width: 1 } },
        xbins: { start: -span, end: span, size: binSize },
      },
      {
        type: "histogram", x: d.deep.pnl.map((v) => v * K),
        name: legendName("learned policy", d.deep),
        marker: { color: "rgba(90,140,200,0.45)",
                  line: { color: COLORS.nn, width: 1 } },
        xbins: { start: -span, end: span, size: binSize },
      },
      ...(ww && ww.pnl ? [{
        type: "histogram", x: ww.pnl.map((v) => v * K),
        name: legendName(bandName, ww),
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
        guide(d.delta, COLORS.mc),
        guide(d.deep, COLORS.nn),
        ...(ww && ww.pnl ? [guide(ww, COLORS.violet)] : []),
      ],
    }, PLOT_CONFIG);

    // Holdings along the illustrative path. The band's holdings are drawn
    // when the response carries them.
    const days = d.example_path.deep_holdings.map((_, i) => i + 1);
    const bandHoldings = d.example_path.whalley_wilmott_holdings;
    Plotly.react("plot-holdings", [
      {
        x: days, y: d.example_path.spot.slice(1).map((s) => s * K),
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
      ...(bandHoldings ? [{
        x: days, y: bandHoldings,
        mode: "lines", name: bandName,
        line: { color: COLORS.violet, width: 2, dash: "dot" },
      }] : []),
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

// Desk note.
$("btn-risk").addEventListener("click", async () => {
  const btn = $("btn-risk");
  btn.textContent = "Writing...";
  btn.disabled = true;
  const out = $("ai-report");
  try {
    // Missing inputs are computed first, in the order the note uses them.
    if (!lastAttributions) {
      out.textContent = "Working out what drives the price... (1 of 3)";
      await runXAI();
    }
    if (!lastHedge) {
      out.textContent = "Running the hedging simulation... (2 of 3)";
      await runHedge();
    }
    out.textContent = "Writing the summary... (3 of 3)";
    if (lastNNPrice == null || !lastAttributions || !lastHedge) {
      throw new Error("It needs the price, the attribution and a hedging " +
        "run, and one of them is unavailable for this contract.");
    }

    out.classList.add("streaming");
    const K = state.strike;
    const req = {
      // Only a successfully fetched ticker names the underlying. Text left in
      // the ticker box after a failed lookup does not reach the report.
      ticker: marketData ? marketData.ticker : "",
      contract: contractShort(),
      nn_price: lastNNPrice,
      bs_cvar: -lastHedge.delta.cvar95 * K,
      deep_cvar: -lastHedge.deep.cvar95 * K,
      ww_cvar: lastHedge.whalley_wilmott
        ? -lastHedge.whalley_wilmott.cvar95 * K : null,
      // The bootstrap standard errors travel with the point estimates, so the
      // note can apply the two-standard-error test when the paired verdict
      // below is absent.
      bs_cvar_se: lastHedge.delta.cvar95_se != null
        ? lastHedge.delta.cvar95_se * K : null,
      deep_cvar_se: lastHedge.deep.cvar95_se != null
        ? lastHedge.deep.cvar95_se * K : null,
      ww_cvar_se: (lastHedge.whalley_wilmott
        && lastHedge.whalley_wilmott.cvar95_se != null)
        ? lastHedge.whalley_wilmott.cvar95_se * K : null,
      // Which pairs the paired bootstrap separates. The three hedgers ran on
      // the same paths, so their errors are correlated and the hypot of two
      // error bars misstates the error of a difference. Sending the paired
      // verdict keeps the note and the Hedge tab on one test for one run.
      paired_separated: pairedSeparationMap(lastHedge),
      baseline_price: lastBaselinePrice,
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
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail || detail; } catch { /* not json */ }
      throw new Error(friendlyError(response.status, detail));
    }

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
    out.textContent = "The summary could not be written. " + e.message;
  } finally {
    out.classList.remove("streaming");
    btn.textContent = "Write summary";
    btn.disabled = false;
  }
});

// Live stream over the websocket.
let ws = null;
let wsSpots = [];
let wsPrices = [];
let wsTicks = [];
const WS_MAX_POINTS = 400;
// The server sends one error frame and closes when it refuses a stream (a
// contract outside the trained range, or no free stream slot). The reason is
// kept so the close handler shows it in place of the generic line.
let wsRefusal = "";
function streamRefusalText(detail) {
  const d = String(detail || "");
  if (/capacity|saturated/i.test(d)) return "The live feed is at capacity. Try again shortly.";
  if (/not loaded/i.test(d)) return "The pricing model is not loaded on this server.";
  return friendlyError(422, d);
}

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
  wsRefusal = "";

  ws.onopen = () => {
    btn.textContent = "Disconnect";
    btn.classList.add("btn-stream-active");
    $("stream-empty")?.remove();
    $("stream-stats").classList.remove("idle");
    $("stream-sub").textContent = "Connected. Starting the feed...";

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
    if (d.error) {
      // An error frame without a tick is a refusal and the socket closes next.
      // One with a tick is a simulated spot outside the trained range, and the
      // stream continues with the following tick.
      if (d.tick === undefined) wsRefusal = streamRefusalText(d.error);
      return;
    }
    if (d.status === "ready") {
      // The server caps the requested rate (MAX_STREAM_HZ), so the caption
      // shows the granted rate. The ready frame carries no tick fields and
      // returns before the tick rendering below. The stat beside this caption
      // is the pricing wall-clock, and the caption says so because the tick
      // period implies a different rate.
      $("stream-sub").textContent = "Live: " + d.hz +
        " simulated ticks a second. Pricing time is the network's wall-clock " +
        "for the price and all five Greeks on this server.";
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

    // The chart is extended on every second tick to limit layout work.
    if (d.tick % 2 === 0) {
      Plotly.extendTraces("plot-stream",
        { y: [[d.spot], [d.price]] }, [0, 1],
        WS_MAX_POINTS);
    }
  };

  ws.onclose = () => {
    btn.textContent = "Connect";
    btn.classList.remove("btn-stream-active");
    $("stream-sub").textContent = wsRefusal || "Disconnected. Press Connect to resume.";
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

// Initial load. Cheap calls go out immediately. The simulation-heavy panels
// load in sequence, because the server admits one Monte Carlo or
// batch-inference job at a time and parallel requests would queue there while
// holding connections. The latency benchmark re-runs the convergence workload
// for its wall-clock alone, so it runs on demand from its button.
const urlParams = Object.fromEntries(new URLSearchParams(location.search));
applyState(urlParams);
if (urlParams.tab) {
  const t = String(urlParams.tab).toLowerCase();
  const key = TAB_IDS[t] ? t : TAB_ALIAS[t];
  if (key) showTab(key);
}
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
  // The surfaces are drawn once here even while their group is collapsed.
  // After that refreshSurfaces defers them until the group is open.
  await runPrice();
  await runConvergence();
  await runSurface();
  await runXAI();
  await runIVSurface();
  if (currentTab === "hedging" && urlParams.run === "1") runHedge();
})();
const latencyShimmer = $("plot-latency").querySelector(".shimmer");
if (latencyShimmer) {
  latencyShimmer.replaceWith(Object.assign(document.createElement("p"), {
    className: "card-sub centered latency-hint",
    textContent: "Press Time it to measure the network and the simulation on this server.",
  }));
}


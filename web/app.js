/* DownsideIQ web terminal — pure API client (no business logic here). */
const API = "";                       // same origin (served by FastAPI)
const state = { mode: "strict", lastSignalId: null, loaded: {} };

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmtPct = v => (typeof v === "number" && isFinite(v)) ? (v * 100).toFixed(1) + "%" : "—";
const fmtNum = (v, d = 3) => (typeof v === "number" && isFinite(v)) ? v.toFixed(d) : "—";
const cls = d => (d || "").replace(" ", "").toUpperCase().replace("NOTRADE", "NOTRADE");

async function getJSON(path) {
  try { const r = await fetch(API + path); return r.ok ? await r.json() : { _error: `${r.status}` }; }
  catch (e) { return { _error: String(e) }; }
}
async function postJSON(path) {
  try { const r = await fetch(API + path, { method: "POST" }); return await r.json(); }
  catch (e) { return { _error: String(e) }; }
}

const PLOTLY_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#8b93a7", family: "Inter, sans-serif", size: 11 },
  margin: { l: 48, r: 18, t: 10, b: 38 },
  xaxis: { gridcolor: "#1a2030", zerolinecolor: "#2e3950" },
  yaxis: { gridcolor: "#1a2030", zerolinecolor: "#2e3950" },
  showlegend: false,
};
const PLOTLY_CFG = { displayModeBar: false, responsive: true };

function metric(label, value, delta) {
  return `<div class="card metric"><div class="label">${label}</div>
    <div class="value">${value}</div>${delta ? `<div class="delta">${delta}</div>` : ""}</div>`;
}

/* ---------------- navigation ---------------- */
$$(".nav-item[data-view]").forEach(n => n.addEventListener("click", () => {
  $$(".nav-item").forEach(x => x.classList.remove("active"));
  n.classList.add("active");
  const v = n.dataset.view;
  $$(".view").forEach(x => x.classList.remove("active"));
  $("#view-" + v).classList.add("active");
  loadView(v);
}));

$("#modeswitch").addEventListener("click", e => {
  if (!e.target.dataset.mode) return;
  state.mode = e.target.dataset.mode;
  $$("#modeswitch button").forEach(b => b.classList.toggle("active", b.dataset.mode === state.mode));
  $("#signal-mode").textContent = state.mode;
  state.loaded = {};                              // force refresh on mode change
  const active = $(".nav-item.active").dataset.view;
  loadView(active);
});

function loadView(v) {
  if (v === "overview") return renderOverview();
  if (v === "signal") return;                      // on-demand via button
  if (v === "history") return renderHistory();
  if (v === "paper") return renderPaper();
  if (v === "risk") return renderRisk();
  if (v === "health") return renderHealth();
}

/* ---------------- API status poll ---------------- */
async function pollStatus() {
  const h = await getJSON("/health");
  const dot = $("#api-dot"), txt = $("#api-status");
  if (h._error) { dot.className = "dot bad"; txt.textContent = "API offline"; }
  else { dot.className = "dot ok"; txt.textContent = `API ${h.status} · ${Object.values(h.providers).filter(Boolean).length}/4 providers`; }
}

/* ---------------- Overview ---------------- */
async function renderOverview() {
  const [h, ms, hist] = await Promise.all([
    getJSON("/health"), getJSON("/model/status"),
    getJSON(`/signals/history?mode=${state.mode}&limit=500`)]);
  const nSig = hist.signals ? hist.signals.length : 0;
  const decided = (hist.signals || []).filter(s => s.prediction_correct !== null);
  const acc = decided.length ? decided.filter(s => s.prediction_correct).length / decided.length : null;
  $("#ov-stats").innerHTML =
    metric("API Status", h._error ? "OFFLINE" : h.status.toUpperCase(), h._error ? "" : "all systems") +
    metric("Providers Up", h._error ? "—" : `${Object.values(h.providers).filter(Boolean).length} / 4`, "tavily · gemini · av · finnhub") +
    metric("Logged Signals", nSig, `${state.mode} mode`) +
    metric("Resolved Accuracy", fmtPct(acc), `${decided.length} reconciled`);
  const perf = (ms.performance || []).slice(0, 8);
  $("#ov-models").innerHTML = perf.length
    ? perf.map(p => `<div class="kv"><span class="k">${p.model_name} <span class="mono" style="color:var(--faint)">v${p.model_version}</span></span><span>${p.metric_name}: <b>${fmtNum(p.metric_value)}</b></span></div>`).join("")
    : `<div class="empty" style="padding:20px">No trained models yet. Run <span class="mono">cli train -t NVDA</span>.</div>`;
}

/* ---------------- Live Signal ---------------- */
$("#analyze-btn").addEventListener("click", runAnalyze);
$("#ticker").addEventListener("keydown", e => { if (e.key === "Enter") runAnalyze(); });

async function runAnalyze() {
  const ticker = $("#ticker").value.trim().toUpperCase();
  if (!ticker) return;
  const btn = $("#analyze-btn"), status = $("#analyze-status");
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Running`;
  status.textContent = "queuing job…";
  const job = await postJSON(`/analyze/${ticker}?mode=${state.mode}`);
  if (job._error || !job.job_id) { status.textContent = "error: " + (job._error || job.detail); reset(); return; }
  status.textContent = "running councils + risk engine…";
  let result = null;
  for (let i = 0; i < 90; i++) {
    await new Promise(r => setTimeout(r, 1000));
    const j = await getJSON(`/jobs/${job.job_id}`);
    if (j.ready) { result = j.result; break; }
    status.textContent = `running… (${i + 1}s)`;
  }
  reset();
  if (!result) { status.textContent = "still running — try again shortly."; return; }
  status.textContent = "";
  state.lastSignalId = result.signal_id;
  renderSignal(result.signal_id);
  function reset() { btn.disabled = false; btn.textContent = "Analyze"; }
}

async function renderSignal(sid) {
  const sig = await getJSON(`/predictions/${sid}`);
  const box = $("#signal-result");
  if (sig._error) { box.innerHTML = `<div class="empty">Could not load signal: ${sig._error}</div>`; return; }
  const gov = sig.governance || {}, sc = gov.scores || {};
  const dec = sig.decision, c = cls(dec);
  const ap = sc.adjusted_p_downside ?? sig.adjusted_p_downside ?? 0;
  const risk = sc.adjusted_downside_risk_score ?? 0;
  const dqPill = `<span class="pill ${sig.data_quality}">${(sig.data_quality||"").toUpperCase()}</span>`;
  const gates = gov.gates || {};
  const news = gov.news_catalysts || {};
  const drivers = gov.top_drivers || [];
  const sizing = gov.position_sizing;

  box.innerHTML = `
    <div class="grid cols-2">
      <div class="card decision-card">
        <h3>Decision <span>${sig.ticker} · ${String(sig.timestamp).slice(0,10)} · ${dqPill}</span></h3>
        <div class="badge ${c}"><span class="glyph"></span> ${dec}</div>
        <div class="reasonbox" style="margin-top:14px">${gov.reason || sig.reason || "—"}</div>
        <div style="margin-top:18px">
          <div class="metric"><div class="label">Adjusted P(downside)</div>
            <div class="value">${fmtPct(ap)}</div></div>
          <div class="meter ${ap>=0.65?'red':''}"><span style="width:${Math.min(100,ap*100)}%"></span></div>
        </div>
        <div style="margin-top:14px">
          <div class="metric"><div class="label">Adjusted downside risk score</div>
            <div class="value">${fmtNum(risk,2)}</div></div>
          <div class="meter ${risk>=0.5?'red':''}"><span style="width:${Math.min(100,risk*100)}%"></span></div>
        </div>
      </div>
      <div class="card">
        <h3>Model breakdown</h3>
        ${kv("Base P(downside)", fmtPct(sig.p_downside))}
        ${kv("News overlay shift", fmtNum(sc.news_shift_logodds ?? gov.scores?.news_shift_logodds, 2) + " log-odds")}
        ${kv("Model agreement", fmtNum(sc.model_agreement_score,2))}
        ${kv("Data confidence", fmtNum(sc.data_confidence_score,2))}
        ${kv("Price/volume confirmation", fmtNum(sc.price_volume_confirmation,2))}
        ${kv("Expected edge", fmtNum(sc.expected_edge_bps,0) + " bps")}
        ${kv("Kill switch", (sig.kill_switch_active?'<span class="gate-no">ACTIVE</span>':'<span class="gate-yes">clear</span>'))}
      </div>
    </div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card"><h3>Signal-quality gates (${gov.mode||state.mode} · ${gov.threshold_mode||''})</h3>
        ${gateRow("Min probability", gates.min_probability)}
        ${gateRow("Min model agreement", gates.min_agreement)}
        ${gateRow("Min data confidence", gates.min_data_confidence)}
        ${kv("Requires price/vol confirm", gates.require_pv_confirmation ? "yes" : "no")}
        ${kv("Passed for SHORT", gates.gate_passed_for_short ? '<span class="gate-yes">✓ yes</span>' : '<span class="gate-no">✗ no</span>')}
      </div>
      <div class="card"><h3>News &amp; event intelligence</h3>
        ${kv("Company news risk", fmtNum(news.company_news_risk_score,2))}
        ${kv("Macro news risk", fmtNum(news.macro_risk_score,2))}
        ${kv("Negative catalyst", fmtNum(news.negative_catalyst_score,2))}
        ${kv("Abnormal news volume", news.abnormal_news_volume ? "yes ⚠" : "no")}
        ${gov.data_quality?.reasons?.length ? `<div class="reasonbox" style="border-left-color:var(--amber);margin-top:10px">Degraded: ${gov.data_quality.reasons.join("; ")}</div>` : ""}
      </div>
    </div>
    <div class="grid ${sizing?'cols-2':''}" style="margin-top:16px">
      <div class="card"><h3>Top drivers</h3><div id="drivers-chart" class="chart" style="height:260px"></div></div>
      ${sizing ? `<div class="card"><h3>Position sizing (paper)</h3>
        ${kv("Entry price", fmtNum(sizing.entry_price,2))}
        ${kv("Stop loss", fmtNum(sizing.stop_loss,2))}
        ${kv("Take profit", fmtNum(sizing.take_profit,2))}
        ${kv("Position notional", "$"+fmtNum(sizing.position_notional,0))}
        ${kv("Shares", fmtNum(sizing.position_size,1))}</div>` : ""}
    </div>`;

  if (drivers.length) {
    const d = [...drivers].reverse();
    Plotly.newPlot("drivers-chart", [{
      type: "bar", orientation: "h",
      x: d.map(x => x.weight), y: d.map(x => x.driver),
      marker: { color: "#4c8dff" },
    }], { ...PLOTLY_LAYOUT, margin: { l: 170, r: 18, t: 6, b: 30 } }, PLOTLY_CFG);
  }
}
const kv = (k, v) => `<div class="kv"><span class="k">${k}</span><span>${v}</span></div>`;
const gateRow = (k, v) => kv(k, v == null ? "—" : v);

/* ---------------- History ---------------- */
async function renderHistory() {
  const data = await getJSON(`/signals/history?mode=${state.mode}&limit=300`);
  const sigs = data.signals || [];
  const counts = sigs.reduce((a, s) => (a[s.decision] = (a[s.decision] || 0) + 1, a), {});
  $("#hist-stats").innerHTML =
    metric("Predictions", sigs.length, `${state.mode} mode`) +
    metric("NO TRADE", counts["NO TRADE"] || 0) +
    metric("WATCH", counts["WATCH"] || 0) +
    metric("SHORT", counts["SHORT"] || 0);
  const sc = sigs.filter(s => typeof s.actual_return_24h === "number");
  const colorMap = { SHORT: "#f2545b", WATCH: "#f5b41d", "NO TRADE": "#8b93a7" };
  if (sc.length) {
    Plotly.newPlot("hist-chart", [{
      type: "scatter", mode: "markers",
      x: sc.map(s => s.adjusted_p_downside), y: sc.map(s => s.actual_return_24h),
      marker: { size: 8, color: sc.map(s => colorMap[s.decision] || "#8b93a7"), opacity: .8 },
      text: sc.map(s => s.decision),
    }], { ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: "Adjusted P(downside)" },
          yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Realised 24h return", tickformat: ".0%" },
          shapes: [{ type: "line", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#2e3950", dash: "dash" } }] }, PLOTLY_CFG);
  } else { $("#hist-chart").innerHTML = `<div class="empty">No reconciled outcomes yet.</div>`; }
  $("#hist-table").innerHTML = sigs.length ? table(
    ["Time", "Decision", "Adj P", "Agreement", "Data", "Actual 24h", "Correct"],
    sigs.slice(0, 60).map(s => [
      String(s.timestamp).slice(0, 16),
      `<span class="pill ${cls(s.decision)}">${s.decision}</span>`,
      fmtPct(s.adjusted_p_downside), fmtNum(s.model_agreement_score, 2),
      `<span class="pill ${s.data_quality}">${s.data_quality}</span>`,
      s.actual_return_24h == null ? "—" : `<span class="${s.actual_return_24h<0?'neg':'pos'}">${fmtPct(s.actual_return_24h)}</span>`,
      s.prediction_correct == null ? "—" : (s.prediction_correct ? "✓" : "✗"),
    ])) : `<div class="empty">No signals yet. Run an analysis or <span class="mono">cli analyze -t NVDA</span>.</div>`;
}

/* ---------------- Paper ---------------- */
async function renderPaper() {
  const data = await getJSON(`/paper-trades?mode=research`);
  const p = data.performance || {}, trades = data.trades || [];
  if (!p.n_trades) {
    $("#paper-stats").innerHTML = ""; $("#paper-chart").innerHTML = "";
    $("#paper-table").innerHTML = `<div class="empty">No paper trades yet. Run <span class="mono">cli backtest -t NVDA -m research</span>.</div>`;
    return;
  }
  $("#paper-stats").innerHTML =
    metric("Trades", p.n_trades) + metric("Hit Rate", fmtPct(p.hit_rate)) +
    metric("Profit Factor", fmtNum(p.profit_factor, 2)) +
    metric("Total P&L", "$" + fmtNum(p.total_pnl, 2), `Sharpe ${fmtNum(p.sharpe_ratio,2)} · Sortino ${fmtNum(p.sortino_ratio,2)}`);
  const ordered = [...trades].reverse(); let cum = 0;
  const cumpnl = ordered.map(t => (cum += (t.pnl || 0)));
  Plotly.newPlot("paper-chart", [{
    type: "scatter", mode: "lines", y: cumpnl, line: { color: "#16c784", width: 2 },
    fill: "tozeroy", fillcolor: "rgba(22,199,132,0.08)",
  }], { ...PLOTLY_LAYOUT, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Cumulative P&L ($)" } }, PLOTLY_CFG);
  $("#paper-table").innerHTML = table(
    ["Entry", "Exit", "Reason", "Return", "P&L", "Result", "Driver"],
    trades.slice(0, 50).map(t => [
      String(t.entry_time).slice(0, 16), String(t.exit_time).slice(0, 16), t.exit_reason || "—",
      `<span class="${(t.return_pct||0)>=0?'pos':'neg'}">${fmtPct(t.return_pct)}</span>`,
      `<span class="${(t.pnl||0)>=0?'pos':'neg'}">$${fmtNum(t.pnl,2)}</span>`,
      `<span class="pill ${t.result==='win'?'ok':'blocked'}">${t.result||'—'}</span>`,
      t.attribution || "—",
    ]));
}

/* ---------------- Risk ---------------- */
async function renderRisk() {
  const d = await getJSON(`/risk/status?mode=${state.mode}`);
  const ks = d.kill_switch || {}, st = ks.state || {};
  $("#risk-killswitch").innerHTML = `<div class="card"><h3>Kill switch</h3>
    <div class="badge ${ks.active?'SHORT':'WATCH'}" style="font-size:22px">
      <span class="glyph" style="background:${ks.active?'var(--red)':'var(--green)'}"></span>
      ${ks.active ? "ACTIVE" : "CLEAR"}</div>
    ${ks.active ? `<div class="reasonbox" style="border-left-color:var(--red);margin-top:12px">${(ks.reasons||[]).join("; ")}</div>` : ""}</div>`;
  $("#risk-stats").innerHTML =
    metric("Daily P&L", fmtPct(st.daily_pnl_pct)) + metric("Weekly P&L", fmtPct(st.weekly_pnl_pct)) +
    metric("Consecutive Losses", st.consecutive_losses ?? 0) +
    metric("Rolling Accuracy", fmtPct(d.rolling_accuracy)) ;
  const dq = d.data_quality_distribution || {};
  $("#risk-dq").innerHTML = Object.keys(dq).length
    ? Object.entries(dq).map(([k, v]) => kv(`<span class="pill ${k}">${k}</span>`, v)).join("")
      + kv("False-positive short rate", fmtPct(d.false_positive_short_rate))
    : `<div class="empty" style="padding:20px">No signals yet.</div>`;
  $("#risk-limits").innerHTML =
    kv("Max risk / trade", "0.5%") + kv("Max daily loss", "2%") + kv("Max weekly drawdown", "5%") +
    kv("Max consecutive losses", "2") + kv("Live trading", '<span class="gate-no">disabled</span>');
}

/* ---------------- Health ---------------- */
async function renderHealth() {
  const [h, ms] = await Promise.all([getJSON("/health"), getJSON("/model/status")]);
  $("#health-stats").innerHTML =
    metric("Status", h._error ? "OFFLINE" : h.status.toUpperCase()) +
    metric("Database", h.database ? "✓ connected" : "✗") +
    metric("Providers Up", h._error ? "—" : `${Object.values(h.providers).filter(Boolean).length}/4`) +
    metric("Last Signal", h.last_signal_at ? String(h.last_signal_at).slice(0, 16) : "—");
  $("#health-providers").innerHTML = h.providers
    ? Object.entries(h.providers).map(([k, v]) => kv(k, v ? '<span class="gate-yes">● available</span>' : '<span class="gate-no">○ missing</span>')).join("")
    : `<div class="empty">API offline.</div>`;
  const perf = ms.performance || [];
  $("#health-models").innerHTML = perf.length
    ? table(["Model", "Ver", "Metric", "Value", "Mode"], perf.slice(0, 12).map(p =>
        [p.model_name, p.model_version, p.metric_name, fmtNum(p.metric_value), p.mode]))
    : `<div class="empty" style="padding:20px">No model metrics yet.</div>`;
}

/* ---------------- helpers ---------------- */
function table(headers, rows) {
  return `<table><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

/* ---------------- boot ---------------- */
pollStatus(); setInterval(pollStatus, 15000);
renderOverview();

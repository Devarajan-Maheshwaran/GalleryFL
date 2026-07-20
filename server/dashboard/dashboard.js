/* ============================================================
   GalleryFL dashboard — live bindings + self-contained charts
   No external chart library (works offline). Palette is read from
   CSS custom properties so it stays in sync with the design system.
   ============================================================ */
'use strict';

const PALETTE = {
  primary: getCSS('--primary', '#ef4d9b'),
  primarySoft: getCSS('--primary-soft', 'rgba(239,77,155,0.12)'),
  secondary: getCSS('--secondary', '#8b7fe8'),
  amber: getCSS('--amber', '#f5a623'),
  success: getCSS('--success', '#16b87a'),
  text: getCSS('--text-secondary', 'rgba(42,27,46,0.62)'),
  textStrong: getCSS('--text-primary', '#2a1b2e'),
  grid: 'rgba(42,27,46,0.08)',
  surface: getCSS('--surface-2', '#fff5fa'),
};
function getCSS(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

let ws;
let clients = {};
let isTraining = false;
let accessCodeValue = '';
let codeVisible = false;

const store = {
  roundLabels: [], acc: [], loss: [],
};

/* ---------------- Canvas chart engine ---------------- */
function prep(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(120, rect.width), h = Math.max(120, rect.height);
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawLine(canvas, labels, values, opts) {
  opts = opts || {};
  const { ctx, w, h } = prep(canvas);
  const padL = 38, padR = 12, padT = 14, padB = 26;
  const cw = w - padL - padR, ch = h - padT - padB;
  const yMin = opts.yMin != null ? opts.yMin : 0;
  const yMax = opts.yMax != null ? opts.yMax : 1;
  // gridlines
  ctx.strokeStyle = PALETTE.grid; ctx.lineWidth = 1; ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = PALETTE.text;
  for (let i = 0; i <= 4; i++) {
    const y = padT + ch * i / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + cw, y); ctx.stroke();
    const val = (yMax - (yMax - yMin) * i / 4);
    ctx.fillText(val.toFixed(2), 4, y + 3);
  }
  if (!values.length) return;
  const n = values.length;
  const xFor = i => padL + (n <= 1 ? cw / 2 : cw * i / (n - 1));
  const yFor = v => padT + ch * (1 - (v - yMin) / (yMax - yMin));
  // area fill
  const grad = ctx.createLinearGradient(0, padT, 0, padT + ch);
  grad.addColorStop(0, opts.fill || PALETTE.primarySoft);
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.beginPath();
  ctx.moveTo(xFor(0), yFor(values[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xFor(i), yFor(values[i]));
  ctx.lineTo(xFor(n - 1), padT + ch); ctx.lineTo(xFor(0), padT + ch); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  // line
  ctx.beginPath(); ctx.moveTo(xFor(0), yFor(values[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xFor(i), yFor(values[i]));
  ctx.strokeStyle = opts.color || PALETTE.primary; ctx.lineWidth = 2.5; ctx.stroke();
  // points
  ctx.fillStyle = opts.color || PALETTE.primary;
  for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.arc(xFor(i), yFor(values[i]), 3, 0, Math.PI * 2); ctx.fill(); }
  // x labels (first / last / mid)
  ctx.fillStyle = PALETTE.text;
  const ticks = n <= 6 ? labels : [labels[0], labels[Math.floor(n / 2)], labels[n - 1]];
  const idxs = n <= 6 ? labels.map((_, i) => i) : [0, Math.floor(n / 2), n - 1];
  idxs.forEach((i, k) => ctx.fillText(ticks[k], xFor(i) - 8, h - 8));
}

function drawGroupedBars(canvas, labels, series) {
  const { ctx, w, h } = prep(canvas);
  const padL = 38, padR = 12, padT = 14, padB = 30;
  const cw = w - padL - padR, ch = h - padT - padB;
  const yMax = 1;
  ctx.strokeStyle = PALETTE.grid; ctx.lineWidth = 1; ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = PALETTE.text;
  for (let i = 0; i <= 4; i++) {
    const y = padT + ch * i / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + cw, y); ctx.stroke();
    ctx.fillText((yMax - yMax * i / 4).toFixed(2), 4, y + 3);
  }
  const cats = labels.length;
  if (!cats) return;
  const groupW = cw / cats;
  const barW = Math.min(26, (groupW - 14) / series.length);
  ctx.font = '10px Outfit, sans-serif'; ctx.fillStyle = PALETTE.text;
  for (let c = 0; c < cats; c++) {
    const gx = padL + groupW * c + groupW / 2;
    series.forEach((s, si) => {
      const v = s.data[c] || 0;
      const bh = ch * (v / yMax);
      const x = gx - (series.length * barW) / 2 + si * barW;
      const y = padT + ch - bh;
      ctx.fillStyle = s.color;
      roundRect(ctx, x, y, barW - 3, bh, 4); ctx.fill();
    });
    ctx.fillStyle = PALETTE.textStrong; ctx.textAlign = 'center';
    ctx.fillText(labels[c], gx, h - 10); ctx.textAlign = 'left';
  }
  // legend
  let lx = padL;
  ctx.font = '11px Outfit, sans-serif';
  series.forEach(s => {
    ctx.fillStyle = s.color; roundRect(ctx, lx, 2, 10, 10, 3); ctx.fill();
    ctx.fillStyle = PALETTE.text; ctx.fillText(s.label, lx + 14, 11);
    lx += 16 + ctx.measureText(s.label).width + 14;
  });
}

function drawHBars(canvas, labels, values, color) {
  const { ctx, w, h } = prep(canvas);
  const padL = 92, padR = 16, padT = 10, padB = 10;
  const cw = w - padL - padR, ch = h - padT - padB;
  const rowH = ch / Math.max(1, labels.length);
  ctx.font = '11px Outfit, sans-serif';
  for (let i = 0; i < labels.length; i++) {
    const y = padT + rowH * i + rowH / 2;
    const v = values[i] || 0;
    ctx.fillStyle = PALETTE.textStrong; ctx.textAlign = 'right';
    ctx.fillText(labels[i], padL - 8, y + 4); ctx.textAlign = 'left';
    // track
    ctx.fillStyle = PALETTE.grid; roundRect(ctx, padL, y - rowH / 2 + 6, cw, 12, 6); ctx.fill();
    // value
    const bw = cw * v;
    const grad = ctx.createLinearGradient(padL, 0, padL + cw, 0);
    grad.addColorStop(0, color || PALETTE.primary); grad.addColorStop(1, PALETTE.secondary);
    ctx.fillStyle = grad; roundRect(ctx, padL, y - rowH / 2 + 6, Math.max(2, bw), 12, 6); ctx.fill();
    ctx.fillStyle = PALETTE.text; ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), padL + cw, y + 4); ctx.textAlign = 'left';
  }
}

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function renderAllCharts() {
  drawLine(document.getElementById('accChart'), store.roundLabels, store.acc, { color: PALETTE.primary, fill: PALETTE.primarySoft, yMin: 0, yMax: 1 });
  drawLine(document.getElementById('lossChart'), store.roundLabels, store.loss, { color: PALETTE.amber, fill: 'rgba(245,166,35,0.12)', yMin: 0, yMax: 1 });
  if (window.__cmp) {
    drawGroupedBars(document.getElementById('comparisonChart'), window.__cmp.categories, [
      { label: 'Baseline', data: window.__cmp.baseline, color: 'rgba(42,27,46,0.18)' },
      { label: 'Federated', data: window.__cmp.federated, color: PALETTE.primary },
    ]);
  }
  if (window.__f1) {
    drawHBars(document.getElementById('f1Chart'), window.__f1.labels, window.__f1.data, PALETTE.primary);
  }
}

/* ---------------- WebSocket ---------------- */
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/feed?client_id=dashboard`);
  ws.onopen = () => { setWs(true); };
  ws.onclose = () => { setWs(false); setTimeout(connectWS, 3000); };
  ws.onmessage = (e) => { try { const m = JSON.parse(e.data); route(m.type, m.data); } catch (_) {} };
}
function setWs(up) {
  document.getElementById('ws-dot').style.background = up ? PALETTE.success : PALETTE.error;
  document.getElementById('ws-text').innerText = up ? 'Connected' : 'Reconnecting…';
}
function route(type, data) {
  switch (type) {
    case 'round_started': onRoundStart(data); break;
    case 'round_completed': onRoundComplete(data); break;
    case 'training_complete': onTrainingComplete(data); break;
    case 'client_connected': clients[data.client_id] = data; renderClients(); break;
    case 'client_disconnected': delete clients[data.client_id]; renderClients(); break;
    case 'update_received': pulseClient(data.client_id); break;
    case 'clients_cleared': clients = {}; renderClients(); refreshStatus(); refreshLeaderboard(); break;
  }
}

function onRoundStart(d) {
  isTraining = true;
  document.getElementById('phase-badge').innerText = `Round ${d.round}`;
  document.getElementById('phase-text').innerText = `Training · R${d.round}`;
  document.getElementById('training-badge').innerText = 'Training';
  document.getElementById('round-num').innerText = d.round;
  document.getElementById('round-progress-label').innerText = `Round ${d.round} / ${d.total_rounds}`;
  document.getElementById('round-progress-fill').style.width = `${(d.round / d.total_rounds) * 100}%`;
  document.getElementById('btn-start').style.display = 'none';
  document.getElementById('btn-stop').style.display = '';
  setInputsDisabled(true);
}
function onRoundComplete(d) {
  store.roundLabels.push(`R${d.round}`); store.acc.push(d.global_accuracy); store.loss.push(d.global_loss);
  if (store.roundLabels.length > 50) { store.roundLabels.shift(); store.acc.shift(); store.loss.shift(); }
  renderAllCharts(); refreshLeaderboard(); refreshStatus(); refreshComparison();
  toast(`Round ${d.round} complete · loss ${d.global_loss.toFixed(3)}`, 'success');
}
function onTrainingComplete(d) {
  isTraining = false;
  document.getElementById('phase-badge').innerText = 'Complete';
  document.getElementById('phase-text').innerText = 'Idle';
  document.getElementById('training-badge').innerText = 'Ready';
  document.getElementById('round-progress-fill').style.width = '100%';
  document.getElementById('btn-start').style.display = '';
  document.getElementById('btn-stop').style.display = 'none';
  document.getElementById('btn-export-model').disabled = false;
  document.getElementById('btn-export-report').disabled = false;
  setInputsDisabled(false);
  toast('Training session complete', 'success');
}

/* ---------------- Clients ---------------- */
function renderClients() {
  const list = document.getElementById('client-list');
  const ids = Object.keys(clients);
  document.getElementById('clients-live').innerText = `${ids.length} live`;
  if (!ids.length) {
    list.innerHTML = `<div class="state" id="clients-state"><div class="state__icon">◎</div><div class="state__title">No clients yet</div><div class="state__sub">Devices appear here once they connect.</div></div>`;
    return;
  }
  list.innerHTML = '';
  ids.forEach(id => {
    const c = clients[id];
    const el = document.createElement('div');
    el.className = 'client'; el.id = `client-${id}`;
    el.innerHTML = `<div class="client__avatar">${(c.nickname || '?').slice(0, 1).toUpperCase()}</div>
      <div><div class="client__name">${c.nickname || id.slice(0, 8)}</div><div class="client__meta">${c.device_model || 'device'}</div></div>
      <div class="client__status"><span class="nav__dot" style="background:${PALETTE.success};margin:0;"></span>online</div>`;
    list.appendChild(el);
  });
}
function pulseClient(id) {
  const el = document.getElementById(`client-${id}`);
  if (!el) return;
  el.classList.add('pulse');
  setTimeout(() => el.classList.remove('pulse'), 800);
}

/* ---------------- REST refresh ---------------- */
async function refreshStatus() {
  try {
    const r = await fetch('/api/training/status');
    const d = await r.json();
    document.getElementById('model-version').innerText = `v${d.model_version}`;
    document.getElementById('model-ver-display').innerText = d.model_version;
    document.getElementById('client-count').innerText = d.connected_clients;
    document.getElementById('registered-count').innerText = d.registered_clients;
    document.getElementById('param-count').innerText = fmt(d.total_parameters);
    document.getElementById('round-num').innerText = d.current_round;
    if (accessCodeValue !== d.access_code) { accessCodeValue = d.access_code || ''; renderCode(); }
    const addr = d.server_address || (d.access_code && d.access_code.includes('@') ? d.access_code.split('@')[0] : null);
    if (addr) document.getElementById('lan-ip').innerText = addr;
    else document.getElementById('lan-ip').innerText = location.host;
    if (d.online_clients) { clients = {}; d.online_clients.forEach(c => clients[c.client_id] = c); renderClients(); }
  } catch (e) { toast('Status update failed', 'error'); }
}
async function refreshConfig() {
  try {
    const r = await fetch('/api/config');
    const c = await r.json();
    const rows = [
      ['Aggregation', c.aggregation],
      ['Local DP / round', `ε = ${c.dp_epsilon_per_round}, δ = ${c.dp_delta_per_round}`],
      ['Example Gradient Clip', `L2 norm ≤ ${c.max_grad_norm}`],
      ['Client Delta Clip', `L2 norm ≤ ${c.server_delta_clip_norm}`],
      ['Learning Rate', String(c.learning_rate)],
      ['FedProx μ', String(c.mu)],
      ['Pseudo-label Gate', `confidence ≥ ${c.pseudo_label_threshold}, weight ${c.pseudo_label_weight}`],
    ];
    const mk = arr => arr.map(([k, v]) => `<div class="priv__row"><span class="priv__k">${k}</span><span class="priv__v">${v}</span></div>`).join('');
    document.getElementById('privacy-config').innerHTML = mk(rows);
    document.getElementById('privacy-rows').innerHTML = mk(rows);
    document.getElementById('agg-hint').innerText = `Aggregation: ${c.aggregation} · local DP ε=${c.dp_epsilon_per_round}/round`;
  } catch (e) {}
}
async function refreshLeaderboard() {
  try {
    const r = await fetch('/api/metrics/leaderboard');
    const d = await r.json();
    const tb = document.querySelector('#leaderboard-table tbody');
    const state = document.getElementById('leaderboard-state');
    if (!d.leaderboard || !d.leaderboard.length) { state.style.display = ''; tb.innerHTML = ''; return; }
    state.style.display = 'none';
    tb.innerHTML = d.leaderboard.map((c, i) => `<tr class="${i === 0 ? 'rank-1' : ''}"><td>${i + 1}</td><td>${c.nickname || c.client_id.slice(0, 8)}</td><td class="mono">${c.images}</td><td class="mono">${c.rounds}</td><td class="mono">${c.score}</td><td class="mono">${c.epsilon}</td></tr>`).join('');
  } catch (e) {}
}
async function refreshComparison() {
  try {
    const r = await fetch('/api/metrics/comparison');
    const d = await r.json();
    if (!d.categories || !d.categories.length) return;
    window.__cmp = { categories: d.categories, baseline: d.baseline, federated: d.federated };
    window.__f1 = { labels: d.categories, data: d.federated };
    renderAllCharts();
    const note = document.getElementById('comparison-note');
    if (d.evaluated === false) { note.textContent = 'Awaiting server-side evaluation — run a session to populate FL vs. baseline.'; note.className = 'card__hint'; }
    else { note.textContent = 'Per-category F1: federated model vs. pre-trained baseline.'; note.className = 'card__hint'; }
  } catch (e) {}
}
async function refreshTaxonomy() {
  const wrap = document.getElementById('tax-list');
  try {
    const r = await fetch('/api/taxonomy');
    const d = await r.json();
    const cats = (d.categories || []);
    if (!cats.length) { wrap.innerHTML = `<div class="state"><div class="state__title">No taxonomy available</div></div>`; return; }
    let total = 0;
    wrap.innerHTML = cats.map(cat => {
      total += (cat.children || []).length;
      const chips = (cat.children || []).map(t => `<span class="chip chip--n">${t}</span>`).join('');
      return `<div class="tax__cat"><div class="tax__cat-head"><span class="tax__cat-name">${cat.name}</span><span class="badge badge--muted">${(cat.children || []).length}</span></div><div class="tax__chips">${chips}</div></div>`;
    }).join('');
    document.getElementById('tax-hint').innerText = `Live from server · ${cats.length} categories · ${total} tags`;
  } catch (e) { wrap.innerHTML = `<div class="state"><div class="state__title">Could not load taxonomy</div></div>`; }
}

async function refreshDemand() {
  const wrap = document.getElementById('demand-list');
  if (!wrap) return;
  try {
    const r = await fetch('/api/taxonomy/demand');
    const d = await r.json();
    const trend = (d.trending || []);
    if (!trend.length) {
      wrap.innerHTML = `<div class="state"><div class="state__title">No demand signals yet</div><div class="state__sub">Tags reported by clients (applied while organizing, or predicted with high confidence) appear here, ranked by recency-weighted demand.</div></div>`;
      const hint = document.getElementById('demand-hint');
      if (hint) hint.innerText = 'Recency-weighted demand from clients';
      return;
    }
    wrap.innerHTML = trend.map(t => `
      <div class="demand__row">
        <div class="demand__tag">${String(t.tag).replace(/_/g, ' ')}</div>
        <div class="demand__track"><div class="demand__fill" style="width:${(t.weight * 100).toFixed(1)}%"></div></div>
        <div class="demand__val">${(t.weight * 100).toFixed(0)}%</div>
      </div>`).join('');
    const hint = document.getElementById('demand-hint');
    if (hint) hint.innerText = `Recency-weighted demand · ${d.num_active_tags} active tags · ${d.total_signals} signals`;
  } catch (e) {
    wrap.innerHTML = `<div class="state"><div class="state__title">Could not load demand</div></div>`;
  }
}

async function loadHistory() {
  try {
    const r = await fetch('/api/metrics/history');
    const d = await r.json();
    if (d.history && d.history.length) {
      d.history.forEach(h => { store.roundLabels.push(`R${h.round}`); store.acc.push(h.global_accuracy); store.loss.push(h.global_loss); });
      document.getElementById('btn-export-model').disabled = false;
      document.getElementById('btn-export-report').disabled = false;
      renderAllCharts(); refreshComparison();
    }
  } catch (e) {}
}

/* ---------------- Training controls ---------------- */
function setInputsDisabled(dis) { document.querySelectorAll('.field .input').forEach(i => i.disabled = dis); }
async function startTraining() {
  const body = {
    min_clients: parseInt(document.getElementById('inp-min-clients').value) || 2,
    max_rounds: parseInt(document.getElementById('inp-max-rounds').value) || 10,
    local_epochs: parseInt(document.getElementById('inp-local-epochs').value) || 3,
  };
  try {
    const r = await fetch('/api/training/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const d = await r.json();
    if (d.status !== 'started') { toast(`Start: ${d.status}`, 'error'); document.getElementById('phase-badge').innerText = d.status; }
    else toast('Training started', 'success');
  } catch (e) { toast('Could not start training', 'error'); }
}
async function stopTraining() {
  try { await fetch('/api/training/stop', { method: 'POST' }); toast('Stop requested', 'success'); } catch (e) {}
}

/* ---------------- Token / access code ---------------- */
function renderCode() {
  const el = document.getElementById('access-code');
  if (!accessCodeValue) { el.textContent = '—'; el.classList.add('masked'); return; }
  // Show clean 8-char token when possible (new UX)
  let display = accessCodeValue;
  if (accessCodeValue.includes('@')) {
    const parts = accessCodeValue.split('@');
    display = parts[1] || accessCodeValue;
  }
  if (codeVisible) { el.textContent = display; el.classList.remove('masked'); }
  else { el.textContent = maskCode(display); el.classList.add('masked'); }
}
function maskCode(code) {
  const at = code.indexOf('@');
  const ip = at >= 0 ? code.slice(0, at) : '';
  const tok = at >= 0 ? code.slice(at + 1) : code;
  const shown = tok.length > 6 ? tok.slice(0, 4) + '••••' + tok.slice(-2) : tok;
  return ip ? `${ip}@${shown}` : shown;
}
async function copyCode() {
  if (!accessCodeValue) return;
  try { await navigator.clipboard.writeText(accessCodeValue); toast('Access code copied', 'success'); }
  catch (e) { toast('Copy failed', 'error'); }
}
async function regenerateCode() {
  try {
    const r = await fetch('/api/training/regenerate-token', { method: 'POST' });
    const d = await r.json();
    if (d.access_code) {
      accessCodeValue = d.access_code; codeVisible = false; renderCode();
      clients = {}; renderClients(); refreshStatus(); refreshLeaderboard();
      toast('Access code regenerated — clients revoked', 'success');
    }
  } catch (e) { toast('Regenerate failed', 'error'); }
}

/* ---------------- Export ---------------- */
let pendingExport = '';
function showExportModal(url, text) {
  pendingExport = url; document.getElementById('modal-export-text').innerText = text;
  document.getElementById('modal-export').style.display = 'flex';
}
function hideExportModal() { document.getElementById('modal-export').style.display = 'none'; }

/* ---------------- Toast ---------------- */
function toast(msg, kind) {
  const wrap = document.getElementById('toast-wrap');
  const t = document.createElement('div');
  t.className = `toast ${kind ? 'toast--' + kind : ''}`;
  t.textContent = msg;
  wrap.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 250); }, 2600);
}

/* ---------------- Misc ---------------- */
function fmt(n) { if (!n) return '—'; if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'; if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K'; return String(n); }
let startTime = Date.now();
setInterval(() => {
  const diff = Math.floor((Date.now() - startTime) / 1000);
  const p = n => String(n).padStart(2, '0');
  document.getElementById('uptime').innerText = `${p(diff / 3600 | 0)}:${p((diff % 3600) / 60 | 0)}:${p(diff % 60)}`;
}, 1000);

function initNav() {
  document.querySelectorAll('.nav__item').forEach(a => a.addEventListener('click', e => {
    e.preventDefault();
    const t = document.getElementById(a.getAttribute('href').slice(1));
    if (t) t.scrollIntoView({ behavior: 'smooth' });
  }));
  const obs = new IntersectionObserver(es => es.forEach(en => {
    if (en.isIntersecting) {
      document.querySelectorAll('.nav__item').forEach(x => x.classList.remove('active'));
      const a = document.querySelector(`.nav__item[href="#${en.target.id}"]`);
      if (a) a.classList.add('active');
    }
  }), { rootMargin: '-15% 0px -75% 0px' });
  document.querySelectorAll('.main > section').forEach(s => obs.observe(s));
}

/* ---------------- Boot ---------------- */
document.addEventListener('DOMContentLoaded', () => {
  connectWS();
  refreshStatus(); refreshConfig(); refreshLeaderboard(); refreshComparison(); refreshTaxonomy(); refreshDemand(); loadHistory();
  window.addEventListener('resize', renderAllCharts);

  document.getElementById('btn-start').addEventListener('click', startTraining);
  document.getElementById('btn-stop').addEventListener('click', stopTraining);
  document.getElementById('btn-copy-code').addEventListener('click', copyCode);
  document.getElementById('btn-toggle-code').addEventListener('click', () => { codeVisible = !codeVisible; renderCode(); document.getElementById('btn-toggle-code').textContent = codeVisible ? 'Hide' : 'Show'; });
  document.getElementById('btn-regenerate-code').addEventListener('click', regenerateCode);

  document.getElementById('btn-export-model').addEventListener('click', () => showExportModal('/api/export/model', 'Export the latest aggregated global model weights (TFLite)? This can be deployed to clients.'));
  document.getElementById('btn-export-report').addEventListener('click', () => showExportModal('/api/export/report', 'Download the training session metrics report (JSON)? It contains cumulative participant data.'));
  document.getElementById('modal-cancel-btn').addEventListener('click', hideExportModal);
  document.getElementById('modal-confirm-btn').addEventListener('click', () => { if (pendingExport) window.open(pendingExport, '_blank'); hideExportModal(); });

  initNav();
  setInterval(refreshStatus, 5000);
  setInterval(refreshDemand, 5000);
});

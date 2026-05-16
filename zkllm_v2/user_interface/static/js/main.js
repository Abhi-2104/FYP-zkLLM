// Initialize socket only if Socket.IO client is loaded
const socket = typeof io !== 'undefined' ? io() : null;

// ---------------------------------------------------------------------------
// Toast Notifications
// ---------------------------------------------------------------------------
function toast(msg, type = 'info', duration = 4500) {
  let c = document.getElementById('toast-container');
  if (!c) { 
    c = document.createElement('div'); c.id = 'toast-container'; 
    Object.assign(c.style, { position: 'fixed', bottom: '2rem', right: '2rem', zIndex: 9999, display: 'flex', flexDirection: 'column', gap: '0.5rem' });
    document.body.appendChild(c); 
  }
  const el = document.createElement('div');
  const colors = { success: 'var(--emerald)', error: 'var(--rose)', info: 'var(--sky)' };
  Object.assign(el.style, {
    background: 'var(--topbar-bg)', backdropFilter: 'blur(10px)',
    border: `1px solid var(--border)`, borderLeft: `3px solid ${colors[type]||colors.info}`,
    padding: '1rem 1.5rem', borderRadius: '8px', color: 'var(--text)', 
    fontSize: '0.9rem', fontWeight: 500, boxShadow: 'var(--card-shadow)',
    animation: 'slideIn 0.3s cubic-bezier(0.16,1,0.3,1)', transition: 'opacity 0.3s'
  });
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, duration);
}

if (socket) {
  socket.on('connect',    () => toast('Connected to VerifAI Core', 'success'));
  socket.on('disconnect', () => toast('Disconnected from Core', 'error'));
}

// ---------------------------------------------------------------------------
// Profile Icon & Global UI Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  // Profile Dropdown
  const profileBtn = document.getElementById('profile-btn');
  const profileDropdown = document.getElementById('profile-dropdown');
  if (profileBtn && profileDropdown) {
    profileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = profileDropdown.style.display === 'block';
      profileDropdown.style.display = isOpen ? 'none' : 'block';
    });
    document.addEventListener('click', (e) => {
      if (profileDropdown.style.display === 'block') profileDropdown.style.display = 'none';
    });
    profileDropdown.addEventListener('click', (e) => e.stopPropagation());
  }

  // Dashboard Role-Specific Init
  const path = location.pathname.toLowerCase();
  if (path.includes('/provider')) initProvider();
  else if (path.includes('/user')) initUser();
  else if (path.includes('/verifier')) initVerifier();
  else if (path.includes('/master')) { /* master specific init */ }

  // ---------------------------------------------------------------------------
  // Universal Theme Management
  // ---------------------------------------------------------------------------
  // Sync the UI if needed (though the inline script in header handles initial paint)
  const currentTheme = localStorage.getItem('verifai-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', currentTheme);

  const initThemeToggle = () => {
    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
      // Remove any existing listener to prevent double triggers if DOMContentLoaded fires twice (unlikely but safe)
      toggleBtn.replaceWith(toggleBtn.cloneNode(true));
      const newBtn = document.getElementById('theme-toggle');
      newBtn.addEventListener('click', () => {
        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        const nextTheme = isLight ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', nextTheme);
        localStorage.setItem('verifai-theme', nextTheme);
        if (window.updateParticleTheme) window.updateParticleTheme(nextTheme);
      });
    }
  };
  initThemeToggle();

  // Scroll Animations
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) entry.target.classList.add('visible');
    });
  }, { threshold: 0.05, rootMargin: "0px 0px -40px 0px" });
  document.querySelectorAll('.fade-up').forEach(el => observer.observe(el));

  // Interactive mouse glow background
  const cursorGlow = document.getElementById('cursor-glow');
  document.addEventListener('mousemove', (e) => {
    const x = e.clientX / window.innerWidth;
    const y = e.clientY / window.innerHeight;
    document.documentElement.style.setProperty('--mouse-x', x);
    document.documentElement.style.setProperty('--mouse-y', y);
    if (cursorGlow) {
      cursorGlow.style.left = e.clientX + 'px';
      cursorGlow.style.top = e.clientY + 'px';
    }
  });
});

async function loadModelsDropdown() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        window.__MODEL_MAP__ = {};
        data.available_models.forEach(m => {
          window.__MODEL_MAP__[m.id] = m;
        });
        const sel = document.getElementById('model-select');
        if (!sel) return;
        sel.innerHTML = '';
        const selected = data.selected_model;
        data.available_models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = `${m.name} (${m.size}B, Seq: ${m.seq_len})`;
          if (m.id === selected) opt.selected = true;
          sel.appendChild(opt);
        });
        
        sel.addEventListener('change', async (e) => {
            const mid = e.target.value;
            await fetch(`/api/models/${mid}`, { method: 'POST' });
            toast(`Switched focus to: ${mid}`, 'success');
            
            const path = location.pathname;
            if (path.includes('/provider') || path.includes('/verifier')) {
                if (typeof loadCommitments === 'function') loadCommitments(mid);
                if (typeof loadSessions === 'function') loadSessions(mid);
            }
        });
    } catch (_) {}
}

// ---------------------------------------------------------------------------
// Reusable Layer Modal Logic
// ---------------------------------------------------------------------------
let currentModalSid = null;
let currentModalCallback = null;

function getModelInfo(modelId) {
  const map = window.__MODEL_MAP__ || {};
  return map[modelId] || null;
}

function getSelectedModelId() {
  const sel = document.getElementById('model-select');
  return sel ? sel.value : null;
}

function getLayerCountForModel(modelId) {
  const info = getModelInfo(modelId);
  return info && Number.isFinite(info.layers) ? info.layers : 32;
}

function getLayerCountForSession(sid) {
  const sessions = window.__SESSIONS__ || [];
  const session = sessions.find(s => String(s.id) === String(sid));
  if (session && session.model_id) {
    return getLayerCountForModel(session.model_id);
  }
  const selected = getSelectedModelId();
  return getLayerCountForModel(selected);
}

function getActiveModelFilter() {
  const path = window.location.pathname.toLowerCase();
  const select = document.getElementById('model-select');
  const isModelScoped = path.includes('/provider') || path.includes('/verifier');
  if (!isModelScoped || !select || !select.value) return null;
  return select.value;
}

function parseBackendTimestamp(rawTs) {
  if (!rawTs) return null;
  if (rawTs instanceof Date) return Number.isNaN(rawTs.getTime()) ? null : rawTs;

  let ts = String(rawTs).trim();
  if (!ts) return null;

  // Normalize backend timestamp variants to ISO-8601.
  // Treat missing timezone as UTC (SQLite CURRENT_TIMESTAMP semantics).
  const sqliteLike = ts.match(/^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:\d{2})?$/);
  if (sqliteLike) {
    const [, ymd, hms, tz] = sqliteLike;
    ts = `${ymd}T${hms}${tz || 'Z'}`;
  }

  let d = new Date(ts);
  if (Number.isNaN(d.getTime())) {
    d = new Date(String(rawTs));
  }
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatTimestampIST(rawTs, options = {}) {
  const d = parseBackendTimestamp(rawTs);
  if (!d) {
    return {
      date: '--',
      time: '--:-- IST',
      datetime: '--',
    };
  }

  const showSeconds = options.showSeconds !== false;
  const dateFmt = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Kolkata',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
  const timeFmt = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Kolkata',
    hour: '2-digit',
    minute: '2-digit',
    second: showSeconds ? '2-digit' : undefined,
    hour12: false,
  });

  const dateParts = dateFmt.formatToParts(d);
  const day = dateParts.find(p => p.type === 'day')?.value || '00';
  const month = dateParts.find(p => p.type === 'month')?.value || '00';
  const year = dateParts.find(p => p.type === 'year')?.value || '0000';
  const date = `${day}-${month}-${year}`;

  let time = timeFmt.format(d);
  if (!showSeconds && /^\d{2}:\d{2}:\d{2}$/.test(time)) {
    time = time.slice(0, 5);
  }
  time = `${time} IST`;

  return {
    date,
    time,
    datetime: `${date} ${time}`,
  };
}

function refreshLedgerNow(options = {}) {
  const modelFilter = getActiveModelFilter();
  loadSessions(modelFilter);
  if (options.syncProofState && typeof restoreProofState === 'function') {
    restoreProofState();
  }
  if (options.syncVerifierState && typeof loadVerifierState === 'function') {
    loadVerifierState();
  }
}

function stopLedgerAutoRefresh() {
  if (window.__LEDGER_AUTO_REFRESH_TIMER__) {
    clearInterval(window.__LEDGER_AUTO_REFRESH_TIMER__);
    window.__LEDGER_AUTO_REFRESH_TIMER__ = null;
  }
}

function startLedgerAutoRefresh(durationMs = 60000, intervalMs = 2000, options = {}) {
  stopLedgerAutoRefresh();
  const startedAt = Date.now();
  refreshLedgerNow(options);
  window.__LEDGER_AUTO_REFRESH_TIMER__ = setInterval(() => {
    refreshLedgerNow(options);
    if (Date.now() - startedAt >= durationMs) {
      stopLedgerAutoRefresh();
    }
  }, intervalMs);
}

function showLayerModal(sid, type, callback) {
    currentModalSid = sid;
    currentModalCallback = callback;
    window._verifyType = type; // Store type globally for mode switching
    
    const modal = document.getElementById('layer-modal');
    const title = document.getElementById('modal-title-text');
    const sub   = document.getElementById('modal-sub-text');
    const confirmBtn = document.getElementById('modal-confirm-btn');
    const layerSection = document.getElementById('modal-layer-section');

    if (modal) modal.classList.add('active');
    if (title) {
        title.textContent = type === 'proof' ? 'Regenerate Proof' : 'Verify Proof';
        title.style.color = type === 'proof' ? 'var(--amber)' : 'var(--emerald)';
    }
    if (sub) sub.textContent = type === 'proof'
        ? `Regenerate cryptographic proof for Session #${sid}`
        : `Start cryptographic audit for Session #${sid}`;

    const completeLabel = type === 'proof' ? '▶ Regenerate All (Complete)' : '▶ Run Complete (all layers)';
    const partialLabel  = type === 'proof' ? '⚙ Regenerate Partial (select range)' : '⚙ Run Partial (select range)';

    // Inject Complete / Partial mode buttons
    const modeArea = document.getElementById('modal-mode-area');
    if (modeArea) {
        modeArea.innerHTML = `
          <div style="display:flex;gap:0.75rem;margin-bottom:1.5rem;">
            <button id="mode-complete" class="btn btn-brand" style="flex:1;padding:0.75rem;font-size:0.85rem;"
              onclick="selectVerifyMode('complete')">${completeLabel}</button>
            <button id="mode-partial" class="btn" style="flex:1;padding:0.75rem;font-size:0.85rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.12);"
              onclick="selectVerifyMode('partial')">${partialLabel}</button>
          </div>`;
    }
    if (layerSection) layerSection.style.display = 'none';
    if (confirmBtn)   confirmBtn.style.display   = 'none';

    window._verifyMode = 'complete';
    window._verifySid  = sid;
    window._verifyCallback = callback;

    const totalLayers = getLayerCountForSession(sid);
    const startInput = document.getElementById('modal-start-layer');
    const endInput = document.getElementById('modal-end-layer');
    const maxLayer = Math.max(totalLayers - 1, 0);
    if (startInput) {
      startInput.max = String(maxLayer);
      startInput.value = 0;
    }
    if (endInput) {
      endInput.max = String(maxLayer);
      endInput.value = maxLayer;
    }
}

window.selectVerifyMode = function(mode) {
    window._verifyMode = mode;
    const type = window._verifyType || 'verify';
    const layerSection = document.getElementById('modal-layer-section');
    const confirmBtn   = document.getElementById('modal-confirm-btn');
    const modeComplete = document.getElementById('mode-complete');
    const modePartial  = document.getElementById('mode-partial');

    const inactiveStyle = 'flex:1;padding:0.75rem;font-size:0.85rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.12);';

    if (mode === 'complete') {
        if (modeComplete) modeComplete.className = 'btn btn-brand';
        if (modePartial)  modePartial.className  = 'btn';
        if (modePartial)  modePartial.style.cssText = inactiveStyle;
        if (layerSection) layerSection.style.display = 'none';
        if (confirmBtn)   confirmBtn.style.display = 'none';
        
        // Immediately fire with full layer range for the selected model
        closeLayerModal();
        const totalLayers = getLayerCountForSession(window._verifySid);
        if (window._verifyCallback) window._verifyCallback(window._verifySid, 0, totalLayers - 1);
    } else {
        if (modePartial)  modePartial.className  = 'btn btn-brand';
        if (modeComplete) modeComplete.className = 'btn';
        if (modeComplete) modeComplete.style.cssText = inactiveStyle;
        if (layerSection) layerSection.style.display = 'block';
        if (confirmBtn) {
            confirmBtn.style.display = 'block';
            confirmBtn.textContent = type === 'proof' ? 'Start Regeneration' : 'Start Verification';
            confirmBtn.onclick = () => {
                const startVal = parseInt(document.getElementById('modal-start-layer')?.value) || 0;
                const endVal = parseInt(document.getElementById('modal-end-layer')?.value);
                const maxEnd = getLayerCountForSession(window._verifySid) - 1;
                const start = Math.max(0, Math.min(startVal, maxEnd));
                const end = Number.isFinite(endVal)
                  ? Math.max(start, Math.min(endVal, maxEnd))
                  : maxEnd;
                closeLayerModal();
                if (window._verifyCallback) window._verifyCallback(window._verifySid, start, end);
            };
        }
    }
};

function closeLayerModal() {
    const modal = document.getElementById('layer-modal');
    if (modal) modal.classList.remove('active');
    currentModalSid = null;
    currentModalCallback = null;
}

// ---------------------------------------------------------------------------
// Session Conversation Modal
// ---------------------------------------------------------------------------
function isVerifierDashboardView() {
  const p = (window.location.pathname || '').toLowerCase();
  return p.includes('/verifier');
}

function ensureSessionDetailsModal() {
  let overlay = document.getElementById('session-details-modal');
  if (overlay) return overlay;

  overlay = document.createElement('div');
  overlay.id = 'session-details-modal';
  overlay.className = 'modal-overlay';
  overlay.innerHTML = `
    <div class="modal card" style="max-width:760px; width:92%;">
      <h3 id="session-details-title" style="margin-bottom:0.35rem;">Session Trace</h3>
      <p id="session-details-sub" class="text-secondary" style="font-size:0.85rem; margin-bottom:1.25rem;"></p>

      <div id="session-panel-meta" style="margin-bottom:1rem;">
        <div style="font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-muted); margin-bottom:0.45rem;">Metadata</div>
        <div id="session-details-meta" style="display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:0.55rem; font-family:'JetBrains Mono', monospace; font-size:0.78rem;"></div>
      </div>

      <div id="session-details-tabs" style="display:flex; gap:0.5rem; margin-bottom:1rem;">
        <button id="session-tab-prompt" class="btn btn-sm" style="background:rgba(99,102,241,0.12); color:var(--indigo); border-color:rgba(99,102,241,0.28);">Prompt</button>
        <button id="session-tab-response" class="btn btn-sm btn-ghost" style="color:var(--text-muted);">Response</button>
        <button id="session-tab-proof" class="btn btn-sm btn-ghost" style="color:var(--text-muted);">Proof Log</button>
        <button id="session-tab-audit" class="btn btn-sm btn-ghost" style="color:var(--text-muted);">Verification Audit</button>
      </div>

      <div id="session-panel-prompt" style="margin-bottom:1rem;">
        <div style="font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-muted); margin-bottom:0.4rem;">Prompt</div>
        <div id="session-details-prompt" style="white-space:pre-wrap; font-family:'JetBrains Mono', monospace; font-size:0.85rem; padding:0.75rem; border-radius:var(--radius-sm); background:var(--code-bg); border:1px solid var(--border); min-height:80px;"></div>
      </div>

      <div id="session-panel-response" style="display:none; margin-bottom:1.5rem;">
        <div style="font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-muted); margin-bottom:0.4rem;">Response</div>
        <div id="session-details-response" style="white-space:pre-wrap; font-family:'JetBrains Mono', monospace; font-size:0.85rem; padding:0.75rem; border-radius:var(--radius-sm); background:var(--code-bg); border:1px solid var(--border); min-height:120px;"></div>
      </div>

      <div id="session-proof-toggle-wrap" style="display:none; margin:-0.8rem 0 1rem 0;">
        <button id="session-proof-toggle-btn" class="btn btn-sm btn-ghost" style="font-size:0.74rem; color:var(--amber); border-color:rgba(245,158,11,0.28);">Show Proof Generation Log</button>
      </div>

      <div id="session-panel-proof" style="display:none; margin-bottom:1.5rem;">
        <div class="flex-between" style="margin-bottom:0.45rem; gap:0.75rem;">
          <div style="font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-muted);">Proof Generation Log</div>
          <div id="session-proof-summary" class="badge badge-neutral">Loading…</div>
        </div>
        <div id="session-details-proof" style="max-height:320px; overflow:auto; font-family:'JetBrains Mono', monospace; font-size:0.8rem; padding:0.75rem; border-radius:var(--radius-sm); background:var(--code-bg); border:1px solid var(--border); min-height:120px;"></div>
      </div>

      <div id="session-panel-audit" style="display:none; margin-bottom:1.5rem;">
        <div class="flex-between" style="margin-bottom:0.45rem; gap:0.75rem;">
          <div style="font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-muted);">Layer/Parameter Verification Log</div>
          <div id="session-audit-summary" class="badge badge-neutral">Loading…</div>
        </div>
        <div id="session-details-audit" style="max-height:360px; overflow:auto; font-family:'JetBrains Mono', monospace; font-size:0.8rem; padding:0.75rem; border-radius:var(--radius-sm); background:var(--code-bg); border:1px solid var(--border); min-height:140px;"></div>
      </div>

      <div class="flex-between" style="gap:1rem;">
        <button class="btn btn-ghost btn-block" onclick="closeSessionDetailsModal()">Close</button>
      </div>
    </div>
  `;

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closeSessionDetailsModal();
  });

  document.body.appendChild(overlay);

  const tabPrompt = overlay.querySelector('#session-tab-prompt');
  const tabResponse = overlay.querySelector('#session-tab-response');
  const tabProof = overlay.querySelector('#session-tab-proof');
  const tabAudit = overlay.querySelector('#session-tab-audit');
  const proofToggleWrap = overlay.querySelector('#session-proof-toggle-wrap');
  const proofToggleBtn = overlay.querySelector('#session-proof-toggle-btn');

  const tabsWrap = overlay.querySelector('#session-details-tabs');
  const panelMeta = overlay.querySelector('#session-panel-meta');
  const panelPrompt = overlay.querySelector('#session-panel-prompt');
  const panelResponse = overlay.querySelector('#session-panel-response');
  const panelProof = overlay.querySelector('#session-panel-proof');
  const panelAudit = overlay.querySelector('#session-panel-audit');

  const switchTab = (tab) => {
    const isPrompt = tab === 'prompt';
    const isResponse = tab === 'response';
    const isProof = tab === 'proof';
    const isAudit = tab === 'audit';

    if (panelPrompt) panelPrompt.style.display = isPrompt ? '' : 'none';
    if (panelResponse) panelResponse.style.display = isResponse ? '' : 'none';
    if (panelProof) panelProof.style.display = isProof ? '' : 'none';
    if (panelAudit) panelAudit.style.display = isAudit ? '' : 'none';

    const styleTab = (btn, active, activeColor, activeBg, activeBorder) => {
      if (!btn) return;
      btn.className = active ? 'btn btn-sm' : 'btn btn-sm btn-ghost';
      btn.style.background = active ? activeBg : 'transparent';
      btn.style.color = active ? activeColor : 'var(--text-muted)';
      btn.style.borderColor = active ? activeBorder : 'var(--border)';
    };

    styleTab(tabPrompt, isPrompt, 'var(--indigo)', 'rgba(99,102,241,0.12)', 'rgba(99,102,241,0.28)');
    styleTab(tabResponse, isResponse, 'var(--sky)', 'rgba(14,165,233,0.12)', 'rgba(14,165,233,0.28)');
    styleTab(tabProof, isProof, 'var(--amber)', 'rgba(245,158,11,0.12)', 'rgba(245,158,11,0.28)');
    styleTab(tabAudit, isAudit, 'var(--emerald)', 'rgba(16,185,129,0.12)', 'rgba(16,185,129,0.28)');
  };

  const configureModal = (mode) => {
    const verifierMode = mode === 'verifier';
    if (panelMeta) panelMeta.style.display = verifierMode ? '' : 'none';
    if (tabsWrap) tabsWrap.style.display = verifierMode ? 'flex' : 'none';
    if (proofToggleWrap) proofToggleWrap.style.display = verifierMode ? 'none' : '';

    if (tabResponse) tabResponse.style.display = verifierMode ? 'none' : '';
    if (tabProof) tabProof.style.display = verifierMode ? '' : 'none';
    if (tabAudit) tabAudit.style.display = verifierMode ? '' : 'none';

    if (verifierMode) {
      switchTab('prompt');
    } else {
      // User/provider/master dashboard behavior: prompt + response only.
      if (panelPrompt) panelPrompt.style.display = '';
      if (panelResponse) panelResponse.style.display = '';
      if (panelProof) panelProof.style.display = 'none';
      if (panelAudit) panelAudit.style.display = 'none';
      if (proofToggleBtn) proofToggleBtn.textContent = 'Show Proof Generation Log';
    }
  };

  if (tabPrompt) tabPrompt.addEventListener('click', () => switchTab('prompt'));
  if (tabResponse) tabResponse.addEventListener('click', () => switchTab('response'));
  if (tabProof) tabProof.addEventListener('click', () => switchTab('proof'));
  if (tabAudit) tabAudit.addEventListener('click', () => switchTab('audit'));
  if (proofToggleBtn) {
    proofToggleBtn.addEventListener('click', () => {
      if (!panelProof) return;
      const opening = panelProof.style.display === 'none';
      panelProof.style.display = opening ? '' : 'none';
      proofToggleBtn.textContent = opening ? 'Hide Proof Generation Log' : 'Show Proof Generation Log';
    });
  }
  overlay.__switchSessionTab = switchTab;
  overlay.__configureSessionModal = configureModal;

  return overlay;
}

function closeSessionDetailsModal() {
  const overlay = document.getElementById('session-details-modal');
  if (overlay) overlay.classList.remove('active');
}

window.closeSessionDetailsModal = closeSessionDetailsModal;

window.showSessionDetailsModal = async function(sessionId) {
  if (!sessionId) {
    toast('Session ID missing', 'error');
    return;
  }

  const overlay = ensureSessionDetailsModal();
  const title = document.getElementById('session-details-title');
  const sub = document.getElementById('session-details-sub');
  const metaBox = document.getElementById('session-details-meta');
  const promptBox = document.getElementById('session-details-prompt');
  const responseBox = document.getElementById('session-details-response');
  const proofBox = document.getElementById('session-details-proof');
  const proofSummary = document.getElementById('session-proof-summary');
  const auditBox = document.getElementById('session-details-audit');
  const auditSummary = document.getElementById('session-audit-summary');
  const verifierMode = isVerifierDashboardView();

  const renderAuditEvent = (ev) => {
    const status = ev?.status || 'info';
    const icon = status === 'pass' ? '✅' : status === 'fail' ? '❌' : status === 'warn' ? '⚠️' : '•';
    const color = status === 'pass' ? 'var(--emerald)' : status === 'fail' ? 'var(--rose)' : status === 'warn' ? 'var(--amber)' : 'var(--text-secondary)';
    const layer = Number.isFinite(Number(ev?.layer)) ? `L${ev.layer}` : 'L?';
    const mod = ev?.module_label || ev?.module || 'module';
    const param = ev?.parameter ? ` · ${ev.parameter}` : '';
    const line = ev?.line || '';

    return `
      <div style="padding:0.5rem 0.25rem; border-bottom:1px dashed var(--border);">
        <div style="color:${color}; font-weight:700; font-size:0.75rem;">${icon} ${layer} · ${escapeHTML(String(mod))}${escapeHTML(param)}</div>
        <div style="color:var(--text-secondary); margin-top:0.15rem; line-height:1.45;">${escapeHTML(String(line))}</div>
      </div>
    `;
  };

  const deriveAuditEventsFromLines = (lines = []) => {
    return lines
      .filter(Boolean)
      .map((line) => {
        const lc = String(line).toLowerCase();
        let status = 'info';
        if (line.includes('✅') || lc.includes(' passed') || lc.includes(' successful')) status = 'pass';
        else if (line.includes('❌') || lc.includes(' failed') || lc.startsWith('error')) status = 'fail';
        else if (line.includes('⚠') || lc.includes('warning') || lc.includes('skip')) status = 'warn';
        const lm = String(line).match(/layer\s+(\d+)/i);
        return {
          layer: lm ? Number(lm[1]) : null,
          module: 'verification',
          module_label: 'Verification Runner',
          status,
          parameter: null,
          line,
        };
      })
      .filter((ev) => ev.status !== 'info' || /step\s+\d+|verify|proof|sumcheck|softmax|projection|zero-check/i.test(String(ev.line)));
  };

  const deriveProofEventsFromLines = (lines = []) => {
    return lines
      .filter(Boolean)
      .map((line) => {
        const lc = String(line).toLowerCase();
        let status = 'info';
        if (line.includes('✅') || lc.includes(' success')) status = 'pass';
        else if (line.includes('❌') || lc.includes(' failed') || lc.startsWith('error')) status = 'fail';
        else if (line.includes('⚠') || lc.includes('warning')) status = 'warn';

        const lm = String(line).match(/layer\s+(\d+)/i);
        const sm = String(line).match(/\[(\d+)\/5\]\s+(.+)/);
        return {
          layer: lm ? Number(lm[1]) : null,
          component: sm ? sm[2].trim() : null,
          status,
          line,
        };
      })
      .filter((ev) => ev.status !== 'info' || /\[(\d+)\/5\]|processing layer|completed|failed|success/i.test(String(ev.line)));
  };

  const renderProofEvent = (ev) => {
    const status = ev?.status || 'info';
    const icon = status === 'pass' ? '✅' : status === 'fail' ? '❌' : status === 'warn' ? '⚠️' : '•';
    const color = status === 'pass' ? 'var(--emerald)' : status === 'fail' ? 'var(--rose)' : status === 'warn' ? 'var(--amber)' : 'var(--text-secondary)';
    const layer = Number.isFinite(Number(ev?.layer)) ? `L${ev.layer}` : 'L?';
    const component = ev?.component || 'proof';
    return `
      <div style="padding:0.5rem 0.25rem; border-bottom:1px dashed var(--border);">
        <div style="color:${color}; font-weight:700; font-size:0.75rem;">${icon} ${layer} · ${escapeHTML(String(component))}</div>
        <div style="color:var(--text-secondary); margin-top:0.15rem; line-height:1.45;">${escapeHTML(String(ev?.line || ''))}</div>
      </div>
    `;
  };

  if (title) title.textContent = 'Session Trace';
  if (sub) sub.textContent = `Session #${sessionId}`;
  if (metaBox) metaBox.innerHTML = verifierMode ? '<div style="opacity:0.7; grid-column:1 / -1;">Loading metadata…</div>' : '';
  if (promptBox) promptBox.textContent = 'Loading prompt...';
  if (responseBox) responseBox.textContent = 'Loading response...';
  if (proofBox) proofBox.innerHTML = '<div style="opacity:0.7;">Loading proof generation log…</div>';
  if (proofSummary) {
    proofSummary.className = 'badge badge-neutral';
    proofSummary.textContent = 'Loading…';
  }
  if (auditBox) auditBox.innerHTML = '<div style="opacity:0.7;">Loading verification audit log…</div>';
  if (auditSummary) {
    auditSummary.className = 'badge badge-neutral';
    auditSummary.textContent = 'Loading…';
  }

  if (typeof overlay.__configureSessionModal === 'function') {
    overlay.__configureSessionModal(verifierMode ? 'verifier' : 'basic');
  } else if (typeof overlay.__switchSessionTab === 'function') {
    overlay.__switchSessionTab('prompt');
  }

  overlay.classList.add('active');

  try {
    const res = await fetch(`/api/sessions/${sessionId}`);
    const data = await res.json();
    if (!res.ok || data.error) {
      const msg = data.error || 'Session not found';
      if (metaBox) metaBox.innerHTML = `<div style="color:var(--rose); grid-column:1 / -1;">${escapeHTML(msg)}</div>`;
      if (promptBox) promptBox.textContent = msg;
      if (responseBox) responseBox.textContent = '';
      if (proofBox) proofBox.innerHTML = '<div style="opacity:0.7;">No proof log available.</div>';
      if (proofSummary) {
        proofSummary.className = 'badge badge-rose';
        proofSummary.textContent = 'Unavailable';
      }
      if (auditBox) auditBox.innerHTML = '<div style="opacity:0.7;">No audit details found.</div>';
      if (auditSummary) {
        auditSummary.className = 'badge badge-rose';
        auditSummary.textContent = 'Unavailable';
      }
      return;
    }

    const tsFmt = formatTimestampIST(data.timestamp, { showSeconds: true });
    const kv = (label, value) => `
      <div style="padding:0.45rem 0.55rem; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--hover-bg);">
        <div style="font-size:0.64rem; letter-spacing:0.06em; text-transform:uppercase; color:var(--text-muted); margin-bottom:0.2rem;">${escapeHTML(label)}</div>
        <div style="font-size:0.78rem; color:var(--text);">${escapeHTML(String(value ?? '—'))}</div>
      </div>
    `;

    if (metaBox) {
      if (verifierMode) {
        metaBox.innerHTML = [
          kv('Session', `#${data.id ?? sessionId}`),
          kv('User', data.username || (data.user_id ? `user-${data.user_id}` : '—')),
          kv('Model', data.model_id || '—'),
          kv('Occurred (IST)', tsFmt.datetime),
          kv('Inference Status', (data.inference_status || '—').toUpperCase()),
          kv('Proof Status', (data.proof_status || '—').toUpperCase()),
          kv('Proof Duration', Number.isFinite(Number(data.proof_duration)) ? `${data.proof_duration}s` : '—'),
          kv('Verification Status', (data.verify_status || '—').toUpperCase()),
          kv('Verification Duration', Number.isFinite(Number(data.verify_duration)) ? `${data.verify_duration}s` : '—'),
          kv('Verification Runs', Number.isFinite(Number(data.verify_runs)) ? data.verify_runs : 0),
          kv('Requested Verify', data.request_verify ? 'Yes' : 'No'),
          kv('C2PA State', data.c2pa_status || '—'),
        ].join('');
      } else {
        metaBox.innerHTML = '';
      }
    }

    if (promptBox) promptBox.textContent = data.prompt || '(no prompt stored)';
    if (responseBox) {
      if (data.response) responseBox.textContent = data.response;
      else if (data.inference_status === 'running') responseBox.textContent = 'Response not ready yet.';
      else responseBox.textContent = 'No response stored for this session.';
    }

    const proofStructured = Array.isArray(data.proof_audit) ? data.proof_audit : [];
    const proofFallback = proofStructured.length ? [] : (Array.isArray(data.proof_log) ? deriveProofEventsFromLines(data.proof_log) : []);
    const proofEvents = (proofStructured.length ? proofStructured : proofFallback).slice(-250);
    const proofPass = proofEvents.filter(e => e.status === 'pass').length;
    const proofFail = proofEvents.filter(e => e.status === 'fail').length;
    const proofWarn = proofEvents.filter(e => e.status === 'warn').length;

    if (proofSummary) {
      if (!proofEvents.length) {
        proofSummary.className = 'badge badge-neutral';
        proofSummary.textContent = 'No proof log';
      } else if (proofFail > 0) {
        proofSummary.className = 'badge badge-rose';
        proofSummary.textContent = `${proofPass} pass · ${proofFail} fail · ${proofWarn} warn`;
      } else {
        proofSummary.className = 'badge badge-success';
        proofSummary.textContent = `${proofPass} pass · ${proofWarn} warn`;
      }
    }

    if (proofBox) {
      if (!proofEvents.length) {
        proofBox.innerHTML = '<div style="opacity:0.7;">No proof generation log available for this session yet.</div>';
      } else {
        proofBox.innerHTML = proofEvents.map(renderProofEvent).join('');
      }
    }

    const structured = Array.isArray(data.verify_audit) ? data.verify_audit : [];
    const fallback = structured.length ? [] : (Array.isArray(data.verify_log) ? deriveAuditEventsFromLines(data.verify_log) : []);
    const events = (structured.length ? structured : fallback).slice(-250);
    const passCount = events.filter(e => e.status === 'pass').length;
    const failCount = events.filter(e => e.status === 'fail').length;
    const warnCount = events.filter(e => e.status === 'warn').length;

    if (auditSummary) {
      if (!verifierMode) {
        auditSummary.className = 'badge badge-neutral';
        auditSummary.textContent = 'Hidden';
      } else if (!events.length) {
        auditSummary.className = 'badge badge-neutral';
        auditSummary.textContent = 'No audit log';
      } else if (failCount > 0) {
        auditSummary.className = 'badge badge-rose';
        auditSummary.textContent = `${passCount} pass · ${failCount} fail · ${warnCount} warn`;
      } else {
        auditSummary.className = 'badge badge-success';
        auditSummary.textContent = `${passCount} pass · ${warnCount} warn`;
      }
    }

    if (auditBox) {
      if (!verifierMode) {
        auditBox.innerHTML = '<div style="opacity:0.7;">Verification audit details are available only in the verifier dashboard.</div>';
      } else if (!events.length) {
        auditBox.innerHTML = '<div style="opacity:0.7;">No layer/parameter verification log available for this session yet.</div>';
      } else {
        const byLayerParam = new Map();
        for (const ev of events) {
          const layer = Number.isFinite(Number(ev?.layer)) ? `L${ev.layer}` : 'L?';
          const bucket = ev?.parameter || ev?.module_label || ev?.module || 'general';
          const key = `${layer}::${bucket}`;
          if (!byLayerParam.has(key)) {
            byLayerParam.set(key, { layer, bucket, pass: 0, fail: 0, warn: 0 });
          }
          const row = byLayerParam.get(key);
          if (ev?.status === 'pass') row.pass += 1;
          else if (ev?.status === 'fail') row.fail += 1;
          else if (ev?.status === 'warn') row.warn += 1;
        }

        const summaryRows = Array.from(byLayerParam.values()).sort((a, b) => {
          const la = Number((a.layer || 'L0').replace('L', ''));
          const lb = Number((b.layer || 'L0').replace('L', ''));
          if (la !== lb) return la - lb;
          return String(a.bucket).localeCompare(String(b.bucket));
        }).slice(0, 120);

        const summaryHtml = `
          <div style="margin-bottom:0.75rem; padding:0.6rem; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--hover-bg);">
            <div style="font-size:0.66rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted); margin-bottom:0.35rem;">Layer/Parameter Pass-Fail Summary</div>
            ${summaryRows.map(r => `
              <div style="display:flex; justify-content:space-between; gap:0.75rem; font-size:0.74rem; padding:0.2rem 0; border-bottom:1px dashed var(--border);">
                <span style="color:var(--text);">${escapeHTML(r.layer)} · ${escapeHTML(String(r.bucket))}</span>
                <span style="white-space:nowrap; color:var(--text-secondary);">✅ ${r.pass} · ❌ ${r.fail} · ⚠️ ${r.warn}</span>
              </div>
            `).join('')}
          </div>
        `;

        auditBox.innerHTML = summaryHtml + events.map(renderAuditEvent).join('');
      }
    }
  } catch (err) {
    if (metaBox) metaBox.innerHTML = '<div style="color:var(--rose); grid-column:1 / -1;">Failed to load metadata.</div>';
    if (promptBox) promptBox.textContent = 'Failed to load session.';
    if (responseBox) responseBox.textContent = '';
    if (proofBox) proofBox.innerHTML = '<div style="color:var(--rose);">Failed to load proof log.</div>';
    if (proofSummary) {
      proofSummary.className = 'badge badge-rose';
      proofSummary.textContent = 'Load failed';
    }
    if (auditBox) auditBox.innerHTML = '<div style="color:var(--rose);">Failed to load audit details.</div>';
    if (auditSummary) {
      auditSummary.className = 'badge badge-rose';
      auditSummary.textContent = 'Load failed';
    }
  }
};

function stopVerifyFastWatcher() {
  if (window.__VERIFY_FAST_WATCHER__) {
    clearInterval(window.__VERIFY_FAST_WATCHER__);
    window.__VERIFY_FAST_WATCHER__ = null;
  }
}

function startVerifyFastWatcher() {
  stopVerifyFastWatcher();
  window.__VERIFY_FAST_WATCHER__ = setInterval(async () => {
    try {
      const resp = await fetch('/api/verify/status', { cache: 'no-store' });
      const data = await resp.json();
      if (typeof window._updateVerifyTrackerFromStatus === 'function') {
        window._updateVerifyTrackerFromStatus(data);
      }
      if (!data.active) {
        stopVerifyFastWatcher();
        setVerifierControls(false);
        refreshLedgerNow({ syncVerifierState: true });
        // Catch DB/write races when verification ends very quickly.
        setTimeout(() => refreshLedgerNow({ syncVerifierState: true }), 350);
        setTimeout(() => refreshLedgerNow({ syncVerifierState: true }), 1200);
        setTimeout(() => refreshLedgerNow({ syncVerifierState: true }), 2500);
      }
    } catch (_) {}
  }, 600);
}

// ---------------------------------------------------------------------------
// Proof Progress UI Helpers
// ---------------------------------------------------------------------------
const PROOF_MODULES = ['input_rmsnorm','self_attn','post_attn_rmsnorm','ffn','skip_connection'];
const PROOF_MODULE_KEYWORDS = {
  'input_rmsnorm':     ['Input RMSNorm', 'input rmsnorm', 'Input Norm'],
  'self_attn':         ['Self-Attention', 'self attention', 'Self Attn'],
  'post_attn_rmsnorm': ['Post-Attn', 'post_attention', 'Post Attn', 'Post-Attention'],
  'ffn':               ['Feed-Forward', 'FFN', 'Feed Forward'],
  'skip_connection':   ['Skip Connection', 'Skip'],
};
const PROOF_PILL_STYLES = {
  active: 'background:rgba(245,158,11,0.15);border-color:rgba(245,158,11,0.4);color:var(--amber);font-weight:700;',
  done:   'background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.35);color:var(--emerald);',
  failed: 'background:rgba(244,63,94,0.12);border-color:rgba(244,63,94,0.35);color:var(--rose);font-weight:700;'
};

function hasProofModulePills() {
  return !!document.getElementById('proof-module-pills');
}

function findProofModuleKey(component) {
  if (!component) return null;
  const lc = component.toLowerCase();
  for (const key of PROOF_MODULES) {
    const kws = PROOF_MODULE_KEYWORDS[key] || [];
    if (kws.some(k => lc.includes(k.toLowerCase()))) return key;
  }
  return null;
}

function setProofPillState(key, state) {
  const pill = document.getElementById(`pmp-${key}`);
  if (!pill) return;
  pill.style.cssText = state ? (PROOF_PILL_STYLES[state] || '') : '';
}

function resetProofModulePills() {
  if (!hasProofModulePills()) return;
  PROOF_MODULES.forEach(key => setProofPillState(key, null));
}

function updateProofModulePills(component) {
  if (!hasProofModulePills()) return;
  const key = findProofModuleKey(component);
  if (!key) return;
  let found = false;
  PROOF_MODULES.forEach(k2 => {
    const pill = document.getElementById(`pmp-${k2}`);
    if (!pill) return;
    if (k2 === key) {
      found = true;
      setProofPillState(k2, 'active');
    } else if (!found) {
      setProofPillState(k2, 'done');
    }
  });
}

function markProofModuleResult(component, success) {
  if (!hasProofModulePills()) return;
  const key = findProofModuleKey(component);
  if (!key) return;
  setProofPillState(key, success ? 'done' : 'failed');
}

function setProofBadgeState(state) {
  const gen = document.getElementById('bdg-gen');
  const ver = document.getElementById('bdg-ver');
  const fail = document.getElementById('bdg-fail');
  const showGen = state === 'running' || state === 'partial';
  const showFail = state === 'failed' || state === 'partial';
  if (gen) gen.classList.toggle('hidden', !showGen);
  if (ver) ver.classList.toggle('hidden', state !== 'success');
  if (fail) fail.classList.toggle('hidden', !showFail);
}

function formatProofStatusText(layer, total, component, showComponents) {
  if (showComponents && component) return `L${layer}/${total} — ${component}`;
  if (Number.isFinite(layer) && Number.isFinite(total)) return `Layer ${layer} / ${total} in progress`;
  if (Number.isFinite(layer)) return `Layer ${layer} in progress`;
  return 'Generating proof…';
}

function getProofRangeEnd(startLayer, totalLayers, explicitEndLayer) {
  if (Number.isFinite(explicitEndLayer)) return explicitEndLayer;
  if (Number.isFinite(startLayer) && Number.isFinite(totalLayers)) {
    return Math.max(startLayer + totalLayers - 1, startLayer);
  }
  const fallbackTotal = Number.isFinite(totalLayers) ? totalLayers : getLayerCountForModel(getSelectedModelId());
  return Math.max((fallbackTotal || 1) - 1, 0);
}

function renderProofState(data, options = {}) {
  if (!data) return;
  const showComponents = options.showComponents ?? false;
  const updateBadge = options.updateBadge ?? false;
  const badgeId = options.badgeId || 'verifier-status-badge';
  const showOnlyWhenActive = options.showOnlyWhenActive ?? false;

  const box = document.getElementById('proof-progress-box');
  const hasState = data.active || data.progress > 0 || data.success === false || data.success === true;
  if (!box) return;
  if (!hasState || (showOnlyWhenActive && !data.active)) {
    box.classList.add('hidden');
    return;
  }
  box.classList.remove('hidden');

  const total = Number.isFinite(data.total_layers) ? data.total_layers : getLayerCountForModel(getSelectedModelId());
  const startLayer = Number.isFinite(data.start_layer) ? data.start_layer : 0;
  const endLayer = getProofRangeEnd(startLayer, total, data.end_layer);
  const layer = Number.isFinite(data.current_layer) ? data.current_layer : 0;
  const component = data.current_component || '';

  const fill = document.getElementById('proof-pb-fill');
  const text = document.getElementById('proof-status-text');
  const layerInfo = document.getElementById('proof-layer-info');
  const pctLabel = document.getElementById('proof-pct-label');
  const pct = typeof data.progress === 'number' ? data.progress : 0;

  if (fill) fill.style.width = `${Math.max(0, Math.min(pct, 100))}%`;
  if (layerInfo) layerInfo.textContent = `Layer ${layer} / ${endLayer}`;
  if (pctLabel) pctLabel.textContent = pct >= 0 ? `${pct}%` : 'Failed';

  if (data.active) {
    setProofBadgeState('running');
    if (text) text.textContent = formatProofStatusText(layer, endLayer, component, showComponents);
    if (showComponents) updateProofModulePills(component);
    if (updateBadge) {
      const badge = document.getElementById(badgeId);
      if (badge) {
        badge.innerHTML = `<span class="dot" style="background:var(--amber)"></span> Generating Proof (${Math.round(pct)}%)`;
        badge.className = 'badge badge-warning';
      }
    }
  } else if (data.success === false || pct < 0) {
    setProofBadgeState('failed');
    if (text) text.textContent = '✗ Proof generation failed';
    if (updateBadge) {
      const badge = document.getElementById(badgeId);
      if (badge) {
        badge.innerHTML = `<span class="dot" style="background:var(--rose)"></span> Proof Failed`;
        badge.className = 'badge badge-rose';
      }
    }
  } else {
    setProofBadgeState('success');
    if (text) text.textContent = '✓ Proof complete';
    if (updateBadge) {
      const badge = document.getElementById(badgeId);
      if (badge) {
        badge.innerHTML = `<span class="dot" style="background:var(--emerald)"></span> Proof Ready`;
        badge.className = 'badge badge-success';
      }
    }
  }
}

function bindProofProgressHandlers(options = {}) {
  if (!socket) return;
  const showComponents = options.showComponents ?? false;
  const updateBadge = options.updateBadge ?? false;
  const badgeId = options.badgeId || 'verifier-status-badge';
  const startBtn = options.startBtn || null;
  const filterToSession = options.filterToSession ?? false;
  const showOnlyWhenActive = options.showOnlyWhenActive ?? false;
  let lastLayer = null;
  let lastPercent = -1;
  let hasProofFailure = false;

  const shouldHandleProofEvent = (data) => {
    if (!filterToSession) return true;
    if (!data) return false;
    if (!window.__PROOF_SESSION_ID__) {
      if (data.session_id) window.__PROOF_SESSION_ID__ = data.session_id;
      return true;
    }
    if (!data.session_id) return true;
    return String(window.__PROOF_SESSION_ID__) === String(data.session_id);
  };

  socket.off('proof_progress').on('proof_progress', (data) => {
    if (!shouldHandleProofEvent(data)) return;
    const box = document.getElementById('proof-progress-box');
    if (box) box.classList.remove('hidden');
    const total = Number.isFinite(data.total) ? data.total : getLayerCountForModel(getSelectedModelId());
    const startLayer = Number.isFinite(data.start_layer) ? data.start_layer : 0;
    const endLayer = getProofRangeEnd(startLayer, total, data.end_layer);
    const layer = Number.isFinite(data.layer) ? data.layer : 0;
    const component = data.component || '';

    const fill = document.getElementById('proof-pb-fill');
    const text = document.getElementById('proof-status-text');
    const layerInfo = document.getElementById('proof-layer-info');
    const pctLabel = document.getElementById('proof-pct-label');
    if (typeof data.percent === 'number' && data.percent < lastPercent) {
      hasProofFailure = false;
    }
    lastPercent = typeof data.percent === 'number' ? data.percent : lastPercent;
    if (fill) fill.style.width = `${data.percent}%`;
    let statusText = formatProofStatusText(layer, endLayer, component, showComponents);
    if (hasProofFailure) statusText += ' — errors detected';
    if (text) text.textContent = statusText;
    if (layerInfo) layerInfo.textContent = `Layer ${layer} / ${endLayer}`;
    if (pctLabel) pctLabel.textContent = `${data.percent}%`;
    setProofBadgeState(hasProofFailure ? 'partial' : 'running');

    if (startBtn) startBtn.disabled = true;
    if (updateBadge) {
      const badge = document.getElementById(badgeId);
      if (badge) {
        if (hasProofFailure) {
          badge.innerHTML = `<span class="dot" style="background:var(--rose)"></span> Proof Errors Detected`;
          badge.className = 'badge badge-rose';
        } else {
          badge.innerHTML = `<span class="dot" style="background:var(--amber)"></span> Generating Proof (${data.percent}%)`;
          badge.className = 'badge badge-warning';
        }
      }
    }

    if (showComponents && layer !== lastLayer) {
      lastLayer = layer;
      resetProofModulePills();
    }
    if (showComponents) updateProofModulePills(component);
  });

  socket.off('proof_component_done').on('proof_component_done', (data) => {
    if (!shouldHandleProofEvent(data)) return;
    if (!data.success) {
      hasProofFailure = true;
      setProofBadgeState('partial');
      if (updateBadge) {
        const badge = document.getElementById(badgeId);
        if (badge) {
          badge.innerHTML = `<span class="dot" style="background:var(--rose)"></span> Proof Errors Detected`;
          badge.className = 'badge badge-rose';
        }
      }
    }
    if (showComponents) markProofModuleResult(data.component, data.success);
  });

  socket.off('proof_complete').on('proof_complete', (data) => {
    if (!shouldHandleProofEvent(data)) return;
    hasProofFailure = false;
    lastPercent = -1;
    const success = !!data.success;
    setProofBadgeState(success ? 'success' : 'failed');
    const text = document.getElementById('proof-status-text');
    const pctLabel = document.getElementById('proof-pct-label');
    const fill = document.getElementById('proof-pb-fill');
    if (text) text.textContent = success
      ? `✓ Proof complete in ${(data.elapsed || data.duration || 0).toFixed(1)}s`
      : '✗ Proof generation failed';
    if (pctLabel) pctLabel.textContent = success ? '100%' : 'Failed';
    if (fill && success) fill.style.width = '100%';

    if (showComponents && success) {
      PROOF_MODULES.forEach(key => setProofPillState(key, 'done'));
    }
    if (startBtn) startBtn.disabled = false;
    if (updateBadge) {
      const badge = document.getElementById(badgeId);
      if (badge) {
        if (success) {
          badge.innerHTML = `<span class="dot" style="background:var(--emerald)"></span> Proof Ready`;
          badge.className = 'badge badge-success';
        } else {
          badge.innerHTML = `<span class="dot" style="background:var(--rose)"></span> Proof Failed`;
          badge.className = 'badge badge-rose';
        }
      }
    }
    if (showOnlyWhenActive) {
      const box = document.getElementById('proof-progress-box');
      if (box) box.classList.add('hidden');
    }
    if (typeof loadSessions === 'function') loadSessions();
  });
}


// =====================================================================
// PROVIDER PORTAL
// =====================================================================
async function initProvider() {
  await loadModelsDropdown();
  const modelSelect = document.getElementById('model-select');
  if (modelSelect) {
    const currentModel = modelSelect.value;
    loadCommitments(currentModel);
    loadSessions(currentModel);
    
    // Switch listener
    modelSelect.addEventListener('change', (e) => {
      loadCommitments(e.target.value);
      loadSessions(e.target.value);
    });
  } else {
    loadCommitments();
    loadSessions();
    restoreProofState();  // Restore progress bar / button state from server
  }
  
  const refBtn = document.getElementById('refresh-sessions-btn');
  if (refBtn) {
    refBtn.addEventListener('click', () => {
      refreshLedgerNow({ syncProofState: true });
    });
  }
  
  // Handle file upload
  const uploadInput = document.getElementById('upload-commitment-input');
  if (uploadInput) {
    uploadInput.addEventListener('change', handleUpload);
  }
  
  // Periodic activity update
  loadActivity();
  setInterval(loadActivity, 15000);
}

async function restoreProofState() {
  // Re-hydrate the proof progress bar and generate button from server state on page load
  try {
    const resp = await fetch('/api/proof/status');
    if (!resp.ok) return;
    const data = await resp.json();
    const hasState = data.active || data.progress > 0 || data.success === false || data.success === true;
    const isUserView = window.location.pathname.includes('/user') || window.location.pathname.includes('/dashboard/user');
    if (!hasState || (isUserView && !data.active)) {
      document.getElementById('proof-progress-box')?.classList.add('hidden');
      return;
    }

    if (isUserView && data.session_id) {
      try {
        const check = await fetch(`/api/sessions/${data.session_id}`);
        if (check.ok) window.__PROOF_SESSION_ID__ = data.session_id;
      } catch (_) {}
      if (!window.__PROOF_SESSION_ID__) window.__PROOF_SESSION_ID__ = data.session_id;
    }

    renderProofState(data, { showComponents: hasProofModulePills() });
    const genBtn = document.getElementById('generate-proof-btn');
    if (genBtn && data.active) {
      genBtn.disabled = true;
      genBtn.innerHTML = '<span class="spinner-sm"></span> Generating Proof...';
    }
  } catch (_) { /* silently ignore if endpoint unavailable */ }
}



async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const modelSelect = document.getElementById('model-select');
    const modelId = modelSelect ? modelSelect.value : null;
    const formData = new FormData();
    formData.append('file', file);
    if (modelId) formData.append('model', modelId);

    toast('Uploading cryptographic commitment...', 'info');

    try {
        const resp = await fetch('/api/commitments/upload', {
            method: 'POST',
            body: formData
        });
        const result = await resp.json();
        
        if (resp.ok) {
            toast('Commitment published successfully', 'success');
            loadCommitments(modelId);
        } else {
            toast(result.error || 'Upload failed', 'error');
        }
    } catch (err) {
        toast('Network error during upload', 'error');
    } finally {
        e.target.value = '';
    }
}

async function loadActivity() {
    try {
        const resp = await fetch('/api/provider/activity');
        if (!resp.ok) return;
        const data = await resp.json();
        const body = document.getElementById('activity-table-body');
        if (!body) return;
        
        if (!data.activity || data.activity.length === 0) {
            body.innerHTML = '<tr><td colspan="3" style="padding:2rem; text-align:center; opacity:0.5;">No user activity recorded yet.</td></tr>';
            return;
        }

        body.innerHTML = data.activity.map(a => {
          const ts = formatTimestampIST(a.timestamp, { showSeconds: false });
          return `
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:0.75rem; font-weight:600; font-size: 0.9rem;">@${a.username}</td>
            <td style="padding:0.75rem;">
              <div class="badge badge-${a.inference_status === 'verified' ? 'success' : (a.inference_status === 'running' ? 'warning' : 'neutral')}" style="font-size: 0.7rem;">
                ${a.inference_status}
              </div>
            </td>
            <td style="padding:0.75rem; font-size:0.75rem; opacity:0.8; text-align: right; line-height:1.25;">
              <div style="font-weight:700;">${ts.time}</div>
              <div style="opacity:0.65;">${ts.date}</div>
            </td>
          </tr>
        `;
        }).join('');
    } catch (err) {}
}

async function loadCommitments(modelId = null) {
  try {
    let url = '/api/commitments';
    if (modelId) url += '/' + modelId;
    const res = await fetch(url);
    const data = await res.json();
    const tbody = document.getElementById('commits-body');
    if (!tbody) return;
    tbody.innerHTML = '';
    
    if (data.commitments.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 2rem; color: var(--text-muted)">No commitments detected for this model.</td></tr>';
      return;
    }
    
    data.commitments.forEach(c => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-family:'JetBrains Mono',monospace; font-size: 0.85rem">${c.layer}</td>
        <td><span class="badge badge-neutral">${c.param}</span></td>
        <td>${c.size}</td>
        <td style="color:var(--emerald)">Published</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    toast('Failed to load commitments', 'error');
  }
}

// =====================================================================
// USER PORTAL
// =====================================================================
async function initUser() {
  await loadModelsDropdown();
  const form = document.getElementById('prompt-form');
  if (form) form.addEventListener('submit', handlePromptSubmit);

  ['refresh-sessions-btn', 'refresh-history-btn'].forEach((id) => {
    const btn = document.getElementById(id);
    if (btn) {
      btn.addEventListener('click', () => {
        refreshLedgerNow({ syncProofState: true });
      });
    }
  });

  loadSessions();
  restoreProofState();
  
  const genBtn = document.getElementById('generate-proof-btn');
  if (genBtn) {
    genBtn.addEventListener('click', () => {
      genBtn.disabled = true;
      genBtn.innerHTML = '<span class="spinner-sm"></span> Generating Proof...';
      toast('Proof generation initiated', 'success');
      const pbBox = document.getElementById('proof-progress-box');
      if (pbBox) pbBox.classList.remove('hidden');
      
      const activeModelId = document.getElementById('model-select').value;
      const totalLayers = getLayerCountForModel(activeModelId);
      const sessionId = window.__LAST_SESSION_ID__ || window.__PROOF_SESSION_ID__;
      if (sessionId) window.__PROOF_SESSION_ID__ = sessionId;
      
      fetch('/api/proof/start', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ 
          model: activeModelId,
          session_id: sessionId || undefined,
          start_layer: 0,
          end_layer: totalLayers - 1
        })
      }).then(res => res.json()).then(data => {
        if (data && data.session_id) window.__PROOF_SESSION_ID__ = data.session_id;
      }).catch(() => {});
    });
  }
  
  if (socket) {
    socket.on('inference_started', (data) => {
      console.log("Inference started in background", data);
    });

    socket.off('proof_log').on('proof_log', (data) => {
      appendTerminal('proof-console', data.line);
    });

    socket.off('inference_log').on('inference_log', (data) => {
      appendTerminal('inference-log-console', data.line);
    });

    socket.on('inference_progress', (data) => {
      const fill = document.getElementById('inference-pb-fill');
      const label = document.getElementById('inference-status-label');
      if (fill) fill.style.width = `${data.pct}%`;
      if (label) label.textContent = data.status;
    });

    socket.on('inference_complete', (data) => {
      const chatArea = document.getElementById('chat-area');
      const loadingBox = document.getElementById('ai-loading-box');
      const btn = document.getElementById('submit-btn');
      const genBtn = document.getElementById('generate-proof-btn');

      if (loadingBox) loadingBox.remove();

      if (data.success) {
        chatArea.innerHTML += `
          <div class="chat-msg chat-ai">
            <div class="chat-label">ZK-LLM Model</div>
            ${escapeHTML(data.response)}
          </div>
        `;
        if (genBtn) {
          genBtn.disabled = false;
          genBtn.className = 'btn btn-block';
          genBtn.style = 'padding: 1rem; background: rgba(99,102,241,0.1); color: var(--indigo); border: 1px solid rgba(99,102,241,0.3); transition: all 0.3s;';
          genBtn.innerHTML = '2. Generate Cryptographic Proof';
        }
        toast("Inference complete", "success");
      } else {
        chatArea.innerHTML += `
          <div class="chat-msg chat-ai" style="border-left-color: var(--rose);">
            <div class="chat-label" style="color: var(--rose);">System Error</div>
            Inference engine crashed. Error: ${data.error || 'Unknown process error'}.
          </div>
        `;
        toast("Inference failed", "error");
      }
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '1. Commence Inference &rarr;';
      }
      setTimeout(() => loadSessions(), 1000);
    });

    bindProofProgressHandlers({ showComponents: false, filterToSession: true });
  }
}


function promptForMaxNewTokens() {
  const minTokens = 1;
  const raw = window.prompt(`Max new tokens to generate (>= ${minTokens}):`, '1');
  if (raw === null) return null;

  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < minTokens) {
    toast(`Please enter a valid integer >= ${minTokens}.`, 'error');
    return null;
  }

  return parsed;
}


async function handlePromptSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('prompt-input').value;
  const btn = document.getElementById('submit-btn');
  const genBtn = document.getElementById('generate-proof-btn');
  
  if (!input.trim()) return;

  const maxNewTokens = promptForMaxNewTokens();
  if (maxNewTokens === null) {
    toast('Inference cancelled.', 'info', 2200);
    return;
  }

  const chatArea = document.getElementById('chat-area');
  
  chatArea.innerHTML = `
    <div class="chat-msg chat-user">
      <div class="chat-label">You</div>
      ${escapeHTML(input)}
    </div>
  `;
  btn.disabled = true;
  if(genBtn) genBtn.disabled = true;
  btn.innerHTML = '<span class="spinner-sm"></span> Initializing Engine...';
  
  try {
    const res = await fetch('/api/prompt', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ prompt: input, request_verify: true, max_new_tokens: maxNewTokens })
    });
    
    if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || 'Inference failed to start');
    }

    const data = await res.json();
    if (data && data.session_id) {
      window.__LAST_SESSION_ID__ = data.session_id;
    }
    toast("Inference started in background", "info");
    
    // Add logic to show a loading state in chat
    chatArea.innerHTML += `
      <div id="ai-loading-box" class="chat-msg chat-ai">
        <div class="chat-label">ZK-LLM Model</div>
        <div class="engine-log" id="inference-log-console" style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.5rem; max-height: 100px; overflow-y: auto;">
          Starting background engine (max new tokens: ${maxNewTokens})...
        </div>
        <div class="progress-container" style="height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px;">
          <div id="inference-pb-fill" style="width: 0%; height: 100%; background: var(--indigo); transition: width 0.3s;"></div>
        </div>
        <div id="inference-status-label" style="font-size: 11px; margin-top: 4px; opacity: 0.5;">Loading weights...</div>
      </div>
    `;

  } catch (err) {
    toast(`Inference Failed: ${err.message}`, 'error');
    chatArea.innerHTML += `
      <div class="chat-msg chat-ai" style="border-left-color: var(--rose);">
        <div class="chat-label" style="color: var(--rose);">System Error</div>
        ${escapeHTML(err.message)}.
      </div>
    `;
    btn.disabled = false;
    btn.innerHTML = '1. Commence Inference &rarr;';
  }
}

function appendTerminal(id, line) {
  const el = document.getElementById(id);
  if (!el) return;
  const div = document.createElement('div');
  div.className = 'ter-line';
  
  if (line.includes('[AUDIT] SUCCESS') || line.includes('✅')) {
    div.classList.add('ter-audit-success');
  } else if (line.includes('[AUDIT] FAIL') || line.includes('❌')) {
    div.classList.add('ter-audit-fail');
  } else if (line.includes('[AUDIT]') || line.includes('───')) {
    div.classList.add('ter-audit-info');
  }
  
  div.textContent = line.includes('>') ? line : `> ${line}`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

// =====================================================================
// VERIFIER PORTAL
// =====================================================================
function setVerifierControls(isVerifying, { preserveIdleLabel = false } = {}) {
  const startBtn = document.getElementById('start-verify-btn');
  const stopBtn = document.getElementById('stop-verify-btn');

  if (startBtn) {
    startBtn.disabled = !!isVerifying;
    if (isVerifying) {
      startBtn.innerHTML = '<span class="spinner-sm"></span> Verifying...';
    } else if (!preserveIdleLabel) {
      startBtn.innerHTML = 'Trigger Verification';
    }
  }

  if (stopBtn) {
    stopBtn.disabled = !isVerifying;
    stopBtn.classList.toggle('hidden', !isVerifying);
  }
}

async function initVerifier() {
  await loadModelsDropdown();
  const startBtn = document.getElementById('start-verify-btn');
  const stopBtn  = document.getElementById('stop-verify-btn');
  
  if (startBtn) startBtn.addEventListener('click', startVerification);
  if (stopBtn)  stopBtn.addEventListener('click', () => stopProcess('verify'));
  setVerifierControls(false, { preserveIdleLabel: true });
  
  const refBtn = document.getElementById('refresh-sessions-btn');
  if (refBtn) {
    refBtn.addEventListener('click', () => {
      refreshLedgerNow({ syncVerifierState: true });
    });
  }
  
  loadVerifierState();
  loadSessions();

  if (socket) {
    // --- Proof Log Handling ---
    socket.on('proof_log', (data) => {
      const term = document.getElementById('proof-terminal');
      if (term) appendTerminal('proof-terminal', `[TRACE] ${data.line}`);
    });

    bindProofProgressHandlers({ showComponents: true, updateBadge: true, startBtn, showOnlyWhenActive: true });

    // --- Verify Log Handling ---
    socket.on('verify_log', (data) => {
      const term = document.getElementById('verify-terminal');
      if (term) appendTerminal('verify-terminal', `[AUDIT] ${data.line}`);
    });

    socket.on('verify_component', (data) => {
      const term = document.getElementById('verify-terminal');
      if (term) appendTerminal('verify-terminal', `[RESULT] ${data.line || (data.component + ': ' + (data.passed ? 'PASSED' : 'FAILED'))}`, data.passed ? 'success' : 'error');
    });
    socket.on('verify_complete',  (data) => {
      setVerifierControls(false);
      stopVerifyFastWatcher();
      
      const elapsed = Number(data?.elapsed ?? data?.duration ?? 0);
      if (data.success) toast(`Verification Passed! (${elapsed.toFixed(1)}s)`, 'success');
      else toast('Verification Failed!', 'error');
      if (typeof loadSessions === 'function') loadSessions();
      refreshLedgerNow({ syncVerifierState: true });
      setTimeout(() => refreshLedgerNow({ syncVerifierState: true }), 300);
      setTimeout(() => refreshLedgerNow({ syncVerifierState: true }), 1200);
      setTimeout(() => refreshLedgerNow({ syncVerifierState: true }), 2500);
    });
  }
}

// Global helper for verifier dashboard
window.selectForVerify = (id) => {
    const input = document.getElementById('verify-session-id');
    if (input) {
        input.value = id;
        window.scrollTo({ top: 0, behavior: 'smooth' });
        toast(`Session #${id} selected for audit`, 'info');
    }
};

async function loadVerifierState() {
  try {
    const prRes = await fetch('/api/proof/status');
    const pData = await prRes.json();
    const startBtn = document.getElementById('start-verify-btn');
    const proofRunning = !!pData.active;

    renderProofState(pData, { showComponents: true, updateBadge: true, showOnlyWhenActive: true });
    if (startBtn) startBtn.disabled = proofRunning;
    if (pData.log_tail) {
        clearTerminal('proof-terminal');
        pData.log_tail.forEach(l => appendTerminal('proof-terminal', `[TRACE] ${l}`));
    }
    
    const vRes = await fetch('/api/verify/status');
    const vData = await vRes.json();
    if (typeof window._updateVerifyTrackerFromStatus === 'function') {
      window._updateVerifyTrackerFromStatus(vData);
    }
    const wasRunning = !!window.__VERIFY_ACTIVE_LAST__;
    if (vData.active || vData.success !== null) {
      if (vData.log_tail) {
        clearTerminal('verify-terminal');
        vData.log_tail.forEach(l => appendTerminal('verify-terminal', `[AUDIT] ${l}`));
      }
      if (vData.active) {
        window.__VERIFY_ACTIVE_LAST__ = true;
        setVerifierControls(true);
        startVerifyFastWatcher();
      } else {
        window.__VERIFY_ACTIVE_LAST__ = false;
        setVerifierControls(false);
        stopVerifyFastWatcher();
        if (wasRunning) {
          refreshLedgerNow({ syncVerifierState: true });
          setTimeout(() => refreshLedgerNow({ syncVerifierState: true }), 400);
        }
        if (proofRunning && startBtn) startBtn.disabled = true;
      }
    } else {
      window.__VERIFY_ACTIVE_LAST__ = false;
      setVerifierControls(false, { preserveIdleLabel: true });
      stopVerifyFastWatcher();
      if (proofRunning && startBtn) startBtn.disabled = true;
    }
  } catch (_) {}
}

async function startVerification(passedSid = null) {
  const sidInput = document.getElementById('verify-session-id');
  const sid = passedSid || (sidInput ? sidInput.value : null);
  if (!sid) return toast('Please select or enter a Session ID', 'info');

  showLayerModal(sid, 'verify', async (sessionId, start, end) => {
    setVerifierControls(true);

    clearTerminal('proof-terminal');
    clearTerminal('verify-terminal');

    // Initialize the per-layer/module progress tracker
    if (typeof window._initVerifyTracker === 'function') {
      window._initVerifyTracker(start, end);
    }

    try {
      const res = await fetch('/api/verify/start', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ start_layer: start, end_layer: end, session_id: sessionId }),
      });
      const data = await res.json();
      if (!data.success) {
          toast(data.error || 'Failed to start verification', 'error');
          setVerifierControls(false);
          stopVerifyFastWatcher();
      } else {
          toast('Verification started', 'success');
          window.__VERIFY_ACTIVE_LAST__ = true;
          startVerifyFastWatcher();
          startLedgerAutoRefresh(90000, 2000, { syncVerifierState: true });
      }
    } catch (e) {
      toast('Network error starting verification', 'error');
      setVerifierControls(false);
      stopVerifyFastWatcher();
    }
  });
}


async function startRegeneration(sid) {
    showLayerModal(sid, 'proof', async (sessionId, start, end) => {
        toast(`Regenerating proof for session #${sessionId}...`, 'info');
        
        let proofBox = document.getElementById('proof-progress-box');
        if (proofBox) {
            proofBox.classList.remove('hidden');
            proofBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            const template = `
                <div id="proof-progress-box" class="card" style="position:fixed; top:20%; left:50%; transform:translateX(-50%); width:400px; z-index:9999; box-shadow:0 10px 30px rgba(0,0,0,0.5); border-color: rgba(99,102,241,0.5);">
                    <div class="flex-between" style="margin-bottom: 0.75rem;">
                        <div style="display:flex; align-items:center; gap:0.5rem; font-weight: 600; font-size:0.9rem;">
                            <svg viewBox="0 0 24 24" style="color:var(--indigo); width: 1.1rem; flex-shrink:0;" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 2v6h-6M3 12a9 9 0 0115-6.7L21 8M3 22v-6h6M21 12a9 9 0 01-15 6.7L3 16"/></svg>
                            ZK Proof Generation
                        </div>
                        <div style="display:flex; gap:0.5rem; align-items:center;">
                            <div id="bdg-gen" class="badge badge-warning"><span class="dot" style="background:currentColor"></span> Generating</div>
                            <div id="bdg-ver" class="badge badge-success hidden">✓ Complete</div>
                          <div id="bdg-fail" class="badge badge-rose hidden">✗ Failed</div>
                        </div>
                    </div>
                    <div id="proof-module-pills" style="display:flex; gap:0.4rem; flex-wrap:wrap; margin-bottom:0.75rem;">
                        <span class="proof-module-pill" id="pmp-input_rmsnorm" data-label="Input Norm">📐 Input Norm</span>
                        <span class="proof-module-pill" id="pmp-self_attn" data-label="Self-Attn">🧠 Self-Attn</span>
                        <span class="proof-module-pill" id="pmp-post_attn_rmsnorm" data-label="Post Norm">📏 Post Norm</span>
                        <span class="proof-module-pill" id="pmp-ffn" data-label="FFN">⚡ FFN</span>
                        <span class="proof-module-pill" id="pmp-skip_connection" data-label="Skip">🔗 Skip</span>
                    </div>
                    <p id="proof-status-text" class="text-secondary" style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; margin-bottom: 0.5rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">Waiting for proof generation…</p>
                    <div class="progress-wrap">
                        <div class="progress-bar">
                            <div id="proof-pb-fill" class="progress-fill" style="width:0%;"></div>
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:var(--text-muted); margin-top:0.4rem;">
                        <span id="proof-layer-info">Layer — / —</span>
                        <span id="proof-pct-label">0%</span>
                    </div>
                    <button class="btn btn-sm btn-ghost btn-block" style="margin-top:1rem;" onclick="this.closest('#proof-progress-box').remove()">✕ Hide Tracker</button>
                 </div>
            `;
            const div = document.createElement('div');
            div.innerHTML = template;
            document.body.appendChild(div.firstElementChild);
        }

        try {
            const res = await fetch('/api/proof/start', {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({ 
                    session_id: sessionId,
                    start_layer: start,
                    end_layer: end
                })
            });
            const data = await res.json();
            if (!data.success) {
                toast(data.error || 'Failed to start regeneration', 'error');
            } else {
                toast('Proof regeneration started', 'success');
              startLedgerAutoRefresh(90000, 2000, { syncProofState: true });
            }
        } catch (err) {
            toast('Network error starting regeneration', 'error');
        }
    });
}

// ---------------------------------------------------------------------------
// Unified Session Management
// ---------------------------------------------------------------------------
async function loadSessions(modelFilter = null) {
    if (typeof modelFilter !== 'string') {
  modelFilter = getActiveModelFilter();
    }
    const path = window.location.pathname;
    const isVerifier = path.includes('verifier');
    const isProvider = path.includes('provider');
    let endpoint = (isVerifier || isProvider) ? '/api/all_sessions' : '/api/sessions';
    if (modelFilter) {
        endpoint += `?model=${modelFilter}`;
    }
    const bodyId = isVerifier ? 'verifier-sessions-body' : 'sessions-table-body';
    
    const body = document.getElementById(bodyId);
    if (!body) return;

    try {
        const cacheBust = endpoint.includes('?') ? '&' : '?';
        const requestUrl = `${endpoint}${cacheBust}_ts=${Date.now()}`;
        const resp = await fetch(requestUrl, {
          cache: 'no-store',
          headers: { 'Cache-Control': 'no-cache' }
        });
        if (!resp.ok) throw new Error(`Failed to load sessions (${resp.status})`);
        const data = await resp.json();
        
        if (!data.sessions || data.sessions.length === 0) {
          const colspan = isVerifier ? 9 : (isProvider ? 8 : 7);
            body.innerHTML = `<tr><td colspan="${colspan}" style="padding:3rem; text-align:center; opacity:0.5;">No execution records found.</td></tr>`;
            return;
        }

        window.__SESSIONS__ = data.sessions;
        
        body.innerHTML = data.sessions.map(s => {
            const getStatusBadge = (st, dur) => {
                if (!st || st === 'pending') return `<span class="badge badge-neutral">PENDING</span>`;
                if (st === 'running') return `<span class="badge badge-warning">RUNNING</span>`;
                
                let cls = 'badge-neutral';
                let label = st.toUpperCase();
                let icon = '';
                
                if (st === 'completed' || st === 'verified' || st === 'passed') {
                    cls = 'badge-success';
                    icon = '✓ ';
                } else if (st === 'failed') {
                    cls = 'badge-rose';
                    icon = '✗ ';
                }
                
                const timeStr = dur ? `<span style="opacity:0.6; font-size:0.7rem; margin-left:0.4rem;">(${dur}s)</span>` : '';
                return `<div style="display:flex; align-items:center;"><span class="badge ${cls}">${icon}${label}</span>${timeStr}</div>`;
            };

            const infoBtnTitle = isVerifier ? 'View Prompt, Proof & Audit' : 'View Prompt & Response';
            const infoBtn = `<button class="btn btn-sm btn-ghost" onclick="showSessionDetailsModal('${s.id}')" style="color:var(--text-muted); font-size:0.95rem; line-height:1; padding:0.1rem 0.4rem;" title="${infoBtnTitle}">ℹ️</button>`;
            const c2paActionBtn = s.c2pa_status === 'signed'
                ? `<a href="/api/c2pa/receipt/${s.id}" class="btn btn-sm" style="background:rgba(16,185,129,0.1); color:var(--emerald); border-color:rgba(16,185,129,0.2); text-decoration:none;" title="Download signed JPEG receipt">🔏 Receipt</a>`
                : '';

            const actions = isVerifier ? `
              <div class="session-actions">
                    ${infoBtn}
                    <button class="btn btn-sm" onclick="startVerification('${s.id}')" style="background:rgba(16,185,129,0.1); color:var(--emerald); border-color:rgba(16,185,129,0.2);">Verify</button>
                    ${s.verify_status === 'running' ? `<button class="btn btn-sm" onclick="stopProcess('${s.id}', 'verify')" style="background:rgba(244,63,94,0.1); color:var(--rose); border-color:rgba(244,63,94,0.25);">Stop</button>` : ''}
                    <button class="btn btn-sm btn-ghost" onclick="deleteSession('${s.id}')" style="color:var(--rose);">Purge</button>
                </div>
            ` : (isProvider ? `
              <div class="session-actions">
                    ${infoBtn}
                    ${c2paActionBtn}
                    <button class="btn btn-sm" onclick="startRegeneration('${s.id}')" style="background:rgba(99,102,241,0.1); color:var(--indigo); border-color:rgba(99,102,241,0.2);">Regen</button>
                    <button class="btn btn-sm btn-ghost" onclick="deleteSession('${s.id}')" style="color:var(--rose);">Purge</button>
                </div>
            ` : `
              <div class="session-actions">
                    ${infoBtn}
                    ${c2paActionBtn}
                    ${s.proof_status === 'running' ? `<button class="btn btn-sm" onclick="stopProcess('${s.id}', 'proof')" style="background:rgba(244,63,94,0.1); color:var(--rose);">Stop</button>` : ''}
                    ${s.verify_status === 'running' ? `<button class="btn btn-sm" onclick="stopProcess('${s.id}', 'verify')" style="background:rgba(244,63,94,0.1); color:var(--rose);">Abort</button>` : ''}
                    <button class="btn btn-sm" onclick="startRegeneration('${s.id}')" style="background:rgba(99,102,241,0.1); color:var(--indigo); border-color:rgba(99,102,241,0.2);">Regen</button>
                    <button class="btn btn-sm btn-ghost" onclick="deleteSession('${s.id}')" style="color:var(--rose);"><svg style="width:1rem;height:1rem;" viewBox="0 0 24 24"><path d="M19 4h-3.5l-1-1h-5l-1 1H5v2h14V4zM6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12z" fill="currentColor"/></svg></button>
                </div>
            `);

            const tsFmt = formatTimestampIST(s.timestamp, { showSeconds: true });
            const timestamp = tsFmt.time;
            const dateStr = tsFmt.date;

            if (isVerifier || isProvider) {
                const c2paBadge = s.c2pa_status === 'signed'
                    ? `<a href="/api/c2pa/receipt/${s.id}" class="badge badge-success" style="text-decoration:none; cursor:pointer; font-size:0.68rem;" title="Download C2PA Receipt">🔏 Receipt</a>`
                    : `<span class="badge badge-neutral" style="opacity:0.4; font-size:0.68rem;">—</span>`;
              const c2paCell = isVerifier ? `<td style="padding:1rem; text-align:center;">${c2paBadge}</td>` : '';
                return `
                    <tr class="fade-in">
                        <td style="padding:1rem; font-family:var(--font-mono); font-size:0.75rem; font-weight:700; opacity:0.7;">#${s.id}</td>
                        <td style="padding:1rem; font-size:0.8rem; line-height:1.2;">
                            <div style="font-weight:700;">${timestamp}</div>
                            <div style="opacity:0.6; font-size:0.7rem;">${dateStr}</div>
                        </td>
                        <td style="padding:1rem; font-weight:800; font-size:0.85rem; color:var(--indigo);">@${s.username || 'unknown'}</td>
                        <td style="padding:1rem; font-size:0.85rem; font-weight:700;">${s.model_id || 'llama-2'}</td>
                        <td style="padding:1rem; max-width:200px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:0.85rem; font-weight:600;" title="${escapeHTML(s.prompt)}">${escapeHTML(s.prompt)}</td>
                        <td style="padding:1rem;">${getStatusBadge(s.proof_status, s.proof_duration)}</td>
                        <td style="padding:1rem;">${getStatusBadge(s.verify_status, s.verify_duration)}</td>
                        ${c2paCell}
                        <td style="padding:1rem; text-align:right; white-space:normal;">${actions}</td>
                    </tr>
                `;
            } else {
                return `
                    <tr class="fade-in">
                        <td style="padding:1rem; font-family:var(--font-mono); font-size:0.75rem; font-weight:700; opacity:0.7;">#${s.id}</td>
                  <td style="padding:1rem; font-size:0.8rem; line-height:1.2;">
                    <div style="font-weight:700;">${timestamp}</div>
                    <div style="opacity:0.6; font-size:0.7rem;">${dateStr}</div>
                  </td>
                        <td style="padding:1rem; max-width:250px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:0.9rem; font-weight:600;" title="${escapeHTML(s.prompt)}">${escapeHTML(s.prompt)}</td>
                        <td style="padding:1rem; font-size:0.85rem; font-weight:700;">${s.model_id || 'llama-2'}</td>
                        <td style="padding:1rem;">${getStatusBadge(s.proof_status, s.proof_duration)}</td>
                        <td style="padding:1rem;">${getStatusBadge(s.verify_status, s.verify_duration)}</td>
                        <td style="padding:1rem; text-align:right; white-space:normal;">${actions}</td>
                    </tr>
                `;
            }
        }).join('');
    } catch (err) {
        console.error('Session load failure', err);
    }
}


async function stopProcess(id, type) {
  let sessionId = id;
  let processType = type;
  if (!processType && (id === 'proof' || id === 'verify')) {
    processType = id;
    sessionId = null;
  }
  if (!processType) return;
  if (!confirm(`Are you sure you want to stop this ${processType} process?`)) return;
    try {
    const endpoint = sessionId ? `/api/${processType}/stop/${sessionId}` : `/api/${processType}/stop`;
    const resp = await fetch(endpoint, { method: 'POST' });
        const data = await resp.json();
        if (data.status === 'success' || data.success) {
            toast('Process termination signal sent', 'success');
      const opts = processType === 'verify'
        ? { syncVerifierState: true }
        : { syncProofState: true };
      if (processType === 'verify') {
        setVerifierControls(false);
        stopVerifyFastWatcher();
      }
      startLedgerAutoRefresh(30000, 2000, opts);
        } else {
            toast(data.error || 'Failed to stop process', 'error');
        }
    } catch (err) {
        toast('API connection error', 'error');
    }
}

async function deleteSession(id) {
    if (!confirm('Permanently remove this session from the ledger? This cannot be undone.')) return;
    try {
        const resp = await fetch(`/api/sessions/delete/${id}`, { method: 'DELETE' });
        const data = await resp.json();
        if (data.status === 'success' || data.success) {
            toast('Session purged from ledger', 'success');
      refreshLedgerNow();
        } else {
            toast(data.error || 'Purge failed', 'error');
        }
    } catch (err) {
        toast('Connection error', 'error');
    }
}

// ---------------------------------------------------------------------------
// Helpers & Terminals
// ---------------------------------------------------------------------------
function appendTerminal(tid, line, type = 'info') {
  const t = typeof tid === 'string' ? document.getElementById(tid) : tid;
  if (!t) return;
  if (t.tagName === 'TEXTAREA') {
      const textLine = line.includes('>') ? line : `> ${line}`;
      if (t.value && !t.value.endsWith('\n')) t.value += '\n';
      t.value += textLine;
      t.scrollTop = t.scrollHeight;
      return;
  }
  const d = document.createElement('div');
  d.className = `ter-line ter-${type}`;
  
  if (line.toLowerCase().includes('error') || line.toLowerCase().includes('fail')) {
      d.classList.add('ter-error');
  } else if (line.toLowerCase().includes('success') || line.toLowerCase().includes('verified') || line.toLowerCase().includes('pass')) {
      d.classList.add('ter-success');
  }

  // Escape HTML first to prevent XSS
  let f = escapeHTML(line);
  
  // Highlighting key terms for legibility
  f = f.replace(/(\b\d+(?:\.\d+)?(?:s|ms)\b)/g, '<span style="color:var(--emerald);font-weight:700;">$1</span>'); // time
  f = f.replace(/(Layer \d+|L\d+)/gi, '<span style="color:var(--indigo);font-weight:700;">$1</span>'); // layers
  f = f.replace(/(0x[a-fA-F0-9]{8,})/g, '<span style="color:var(--amber);font-family:var(--font-mono);font-size:0.9em;opacity:0.8;">$1</span>'); // hashes
  f = f.replace(/(\[SUCCESS\]|Verified|Passed|Complete|Done|Authentic|Valid)/gi, '<span style="color:var(--emerald);font-weight:800;">$1</span>');
  f = f.replace(/(\[ERROR\]|Failed|OOM|Exception|Invalid)/gi, '<span style="color:var(--rose);font-weight:800;">$1</span>');

  // De-emphasize generic log wrapper text to make highlights pop
  if (!f.includes('<span')) {
      d.style.opacity = '0.7';
  } else {
      d.style.color = 'var(--text)'; // reset color if highlighted
  }

  d.innerHTML = f;
  t.appendChild(d);
  t.scrollTop = t.scrollHeight;
}

function clearTerminal(tid) {
  const t = document.getElementById(tid);
  if (!t) return;
  if (t.tagName === 'TEXTAREA') t.value = '';
  else t.innerHTML = '';
}

function downloadTerminalLog(tid, filenamePrefix = 'log') {
  const t = document.getElementById(tid);
  if (!t) return;
  const text = t.tagName === 'TEXTAREA' ? t.value : t.innerText;
  const blob = new Blob([text || ''], { type: 'text/plain' });
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `${filenamePrefix}-${ts}.txt`;
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(link.href);
}

function addVerificationResult(data) {
  const g = document.getElementById('results-grid');
  if (!g) return;
  const d = document.createElement('div');
  d.className = `res-box ${data.passed ? 'res-pass' : 'res-fail'} fade-up visible`;
  d.innerHTML = `<strong>L${data.layer}</strong> ${data.component} <span style="float:right">${data.passed ? '✓' : '✗'}</span>`;
  g.appendChild(d);
}

function escapeHTML(s) {
  if (!s) return '';
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function toggleExpand(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  // Ensure zen-overlay exists
  let overlay = document.getElementById('zen-overlay');
  if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'zen-overlay';
      overlay.className = 'zen-overlay';
      document.body.appendChild(overlay);
      
      // Close all expanded containers when clicking the overlay
      overlay.addEventListener('click', () => {
          document.querySelectorAll('.terminal-container.expanded').forEach(c => toggleExpand(c.id));
      });
  }

  const isExpanded = container.classList.toggle('expanded');
  document.body.classList.toggle('zen-active', isExpanded);
  
  if (isExpanded) {
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden'; // Prevent scrolling background
  } else {
    overlay.classList.remove('active');
    document.body.style.overflow = '';
  }
}

// Close expanded view on ESC key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const expanded = document.querySelector('.terminal-container.expanded');
    if (expanded) {
      toggleExpand(expanded.id);
    }
  }
});

// Close when clicking overlay
document.getElementById('zen-overlay')?.addEventListener('click', () => {
  const expanded = document.querySelector('.terminal-container.expanded');
  if (expanded) {
    toggleExpand(expanded.id);
  }
});

// ---------------------------------------------------------------------------
// Auto-Initialization
// ---------------------------------------------------------------------------
// NOTE: role-specific initialization is already handled in the primary
// DOMContentLoaded handler near the top of this file.

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
        
        // Immediately fire with full layer range (default 32 layers)
        closeLayerModal();
        const totalLayers = 32; 
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
                const start = parseInt(document.getElementById('modal-start-layer')?.value) || 0;
                const end   = parseInt(document.getElementById('modal-end-layer')?.value)   ?? 31;
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
      const mid = document.getElementById('model-select')?.value;
      loadSessions(mid);
      restoreProofState();  // Also re-check running state on manual refresh
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
    if (!data.active) return;

    // A proof job is running — restore the UI
    const pbBox = document.getElementById('proof-progress-box');
    const fill  = document.getElementById('proof-pb-fill');
    const text  = document.getElementById('proof-status-text');
    const pct   = document.getElementById('proof-pct-label');
    const li    = document.getElementById('proof-layer-info');
    const genBtn = document.getElementById('generate-proof-btn');

    if (pbBox) pbBox.classList.remove('hidden');
    if (fill)  fill.style.width = `${data.progress || 0}%`;
    if (text)  text.textContent  = `L${data.current_layer || 0}/32 — ${data.current_component || '...'}`;
    if (pct)   pct.textContent   = `${data.progress || 0}%`;
    if (li)    li.textContent    = `Layer ${data.current_layer || 0} / 32`;
    if (genBtn) {
      genBtn.disabled = true;
      genBtn.innerHTML = '<span class="spinner-sm"></span> Generating Proof...';
    }
    document.getElementById('bdg-gen')?.classList.remove('hidden');
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

        body.innerHTML = data.activity.map(a => `
            <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                <td style="padding:0.75rem; font-weight:600; font-size: 0.9rem;">@${a.username}</td>
                <td style="padding:0.75rem;">
                    <div class="badge badge-${a.inference_status === 'verified' ? 'success' : (a.inference_status === 'running' ? 'warning' : 'neutral')}" style="font-size: 0.7rem;">
                        ${a.inference_status}
                    </div>
                </td>
                <td style="padding:0.75rem; font-size:0.75rem; opacity:0.6; text-align: right;">${new Date(a.timestamp).toLocaleTimeString()}</td>
            </tr>
        `).join('');
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
  
  const refBtn = document.getElementById('refresh-sessions-btn');
  if (refBtn) refBtn.addEventListener('click', () => loadSessions());

  loadSessions();
  
  const genBtn = document.getElementById('generate-proof-btn');
  if (genBtn) {
    genBtn.addEventListener('click', () => {
      genBtn.disabled = true;
      genBtn.innerHTML = '<span class="spinner-sm"></span> Generating Proof...';
      toast('Proof generation initiated', 'success');
      const pbBox = document.getElementById('proof-progress-box');
      if (pbBox) pbBox.classList.remove('hidden');
      
      const activeModelId = document.getElementById('model-select').value;
      
      fetch('/api/proof/start', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ 
          model: activeModelId,
          start_layer: 0,
          end_layer: 31  // always run all 32 layers
        })
      });
    });
  }
  
  if (socket) {
    socket.on('inference_started', (data) => {
      console.log("Inference started in background", data);
    });

    socket.off('proof_log').on('proof_log', (data) => {
      appendTerminal('proof-console', data.line);
    });

    socket.on('inference_log', (data) => {
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

    const PROOF_MODULES = ['input_rmsnorm','self_attn','post_attn_rmsnorm','ffn','skip_connection'];
    const PROOF_MODULE_KEYWORDS = {
      'input_rmsnorm':     ['Input RMSNorm', 'input rmsnorm', 'Input Norm'],
      'self_attn':         ['Self-Attention', 'self attention', 'Self Attn'],
      'post_attn_rmsnorm': ['Post-Attn', 'post_attention', 'Post Attn', 'Post-Attention'],
      'ffn':               ['Feed-Forward', 'FFN', 'Feed Forward'],
      'skip_connection':   ['Skip Connection', 'Skip'],
    };

    function updateProofModulePills(component) {
      if (!component) return;
      PROOF_MODULES.forEach(key => {
        const pill = document.getElementById(`pmp-${key}`);
        if (!pill) return;
        const keywords = PROOF_MODULE_KEYWORDS[key] || [];
        const isMatch = keywords.some(k => component.toLowerCase().includes(k.toLowerCase()));
        if (isMatch) {
          // Mark previous ones done
          let found = false;
          PROOF_MODULES.forEach(k2 => {
            const p2 = document.getElementById(`pmp-${k2}`);
            if (!p2) return;
            if (k2 === key) { found = true; p2.style.cssText = 'background:rgba(245,158,11,0.15);border-color:rgba(245,158,11,0.4);color:var(--amber);font-weight:700;'; }
            else if (!found) { p2.style.cssText = 'background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.35);color:var(--emerald);'; }
          });
        }
      });
    }

    let _lastProofLayer = -1;
    socket.on('proof_progress', (data) => {
      document.getElementById('proof-progress-box')?.classList.remove('hidden');
      const fill = document.getElementById('proof-pb-fill');
      const text = document.getElementById('proof-status-text');
      const layerInfo = document.getElementById('proof-layer-info');
      const pctLabel = document.getElementById('proof-pct-label');
      if (fill) fill.style.width = `${data.percent}%`;
      if (text) text.textContent = `L${data.layer}/${data.total} — ${data.component}`;
      if (layerInfo) layerInfo.textContent = `Layer ${data.layer} / ${data.total}`;
      if (pctLabel) pctLabel.textContent = `${data.percent}%`;
      document.getElementById('bdg-gen')?.classList.remove('hidden');

      // Reset pills on new layer
      if (data.layer !== _lastProofLayer) {
        _lastProofLayer = data.layer;
        PROOF_MODULES.forEach(k => {
          const pill = document.getElementById(`pmp-${k}`);
          if (pill) pill.style.cssText = '';
        });
      }
      updateProofModulePills(data.component);
    });

    socket.on('proof_component_done', (data) => {
      if (data.success) {
        const key = PROOF_MODULES.find(k => {
          const kws = PROOF_MODULE_KEYWORDS[k] || [];
          return kws.some(kw => (data.component||'').toLowerCase().includes(kw.toLowerCase()));
        });
        if (key) {
          const pill = document.getElementById(`pmp-${key}`);
          if (pill) pill.style.cssText = 'background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.35);color:var(--emerald);';
        }
      }
    });

    socket.on('proof_complete', (data) => {
      document.getElementById('bdg-gen')?.classList.add('hidden');
      document.getElementById('bdg-ver')?.classList.remove('hidden');
      const text = document.getElementById('proof-status-text');
      const pctLabel = document.getElementById('proof-pct-label');
      const fill = document.getElementById('proof-pb-fill');
      if (text) text.textContent = data.success
        ? `✓ Proof complete in ${(data.elapsed||data.duration||0).toFixed(1)}s`
        : '✗ Proof generation failed';
      if (pctLabel) pctLabel.textContent = data.success ? '100%' : 'Failed';
      if (fill && data.success) fill.style.width = '100%';
      // All pills green on success
      if (data.success) {
        PROOF_MODULES.forEach(key => {
          const pill = document.getElementById(`pmp-${key}`);
          if (pill) pill.style.cssText = 'background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.35);color:var(--emerald);';
        });
      }
      if (typeof loadSessions === 'function') loadSessions();
    });
  }
}


async function handlePromptSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('prompt-input').value;
  const btn = document.getElementById('submit-btn');
  const genBtn = document.getElementById('generate-proof-btn');
  
  if (!input.trim()) return;
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
      body: JSON.stringify({ prompt: input, request_verify: true })
    });
    
    if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || 'Inference failed to start');
    }

    const data = await res.json();
    toast("Inference started in background", "info");
    
    // Add logic to show a loading state in chat
    chatArea.innerHTML += `
      <div id="ai-loading-box" class="chat-msg chat-ai">
        <div class="chat-label">ZK-LLM Model</div>
        <div class="engine-log" id="inference-log-console" style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.5rem; max-height: 100px; overflow-y: auto;">
          Starting background engine...
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
async function initVerifier() {
  await loadModelsDropdown();
  const startBtn = document.getElementById('start-verify-btn');
  const stopBtn  = document.getElementById('stop-verify-btn');
  
  if (startBtn) startBtn.addEventListener('click', startVerification);
  if (stopBtn)  stopBtn.addEventListener('click', () => stopProcess('verify'));
  
  const refBtn = document.getElementById('refresh-sessions-btn');
  if (refBtn) refBtn.addEventListener('click', loadSessions);
  
  loadVerifierState();
  loadSessions();

  if (socket) {
    // --- Proof Log Handling ---
    socket.on('proof_log', (data) => {
      const term = document.getElementById('proof-terminal');
      if (term) appendTerminal('proof-terminal', `[TRACE] ${data.line}`);
    });

    socket.on('proof_progress', (data) => {
      const badge = document.getElementById('verifier-status-badge');
      if (badge) badge.innerHTML = `<span class="dot" style="background:var(--amber)"></span> Generating Proof (${data.percent}%)`;
      if (startBtn) startBtn.disabled = true;
    });

    socket.on('proof_complete', (data) => {
      const badge = document.getElementById('verifier-status-badge');
      if (badge) {
          badge.innerHTML = `<span class="dot" style="background:var(--emerald)"></span> Proof Ready`;
          badge.className = 'badge badge-success';
      }
      if (startBtn) startBtn.disabled = false;
    });

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
      if (startBtn) {
          startBtn.disabled = false;
          startBtn.innerHTML = 'Trigger Verification Sync';
      }
      if (stopBtn) stopBtn.disabled = true;
      
      if (data.success) toast(`Verification Passed! (${data.duration.toFixed(1)}s)`, 'success');
      else toast('Verification Failed!', 'error');
      if (typeof loadSessions === 'function') loadSessions();
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
    const badge = document.getElementById('verifier-status-badge');
    const startBtn = document.getElementById('start-verify-btn');

    if (pData.active || pData.progress > 0) {
      if (pData.active) {
        if (badge) badge.innerHTML = `<span class="dot" style="background:var(--amber)"></span> Generating Proof (${Math.round(pData.progress)}%)`;
        if (startBtn) startBtn.disabled = true;
      } else {
        if (badge) {
            badge.innerHTML = `<span class="dot" style="background:var(--emerald)"></span> Proof Ready`;
            badge.className = 'badge badge-success';
        }
        if (startBtn) startBtn.disabled = false;
      }
      if (pData.log_tail) {
          clearTerminal('proof-terminal');
          pData.log_tail.forEach(l => appendTerminal('proof-terminal', `[TRACE] ${l}`));
      }
    }
    
    const vRes = await fetch('/api/verify/status');
    const vData = await vRes.json();
    if (vData.active || vData.success !== null) {
      if (vData.log_tail) {
        clearTerminal('verify-terminal');
        vData.log_tail.forEach(l => appendTerminal('verify-terminal', `[AUDIT] ${l}`));
      }
      if (startBtn) {
        if (vData.active) {
            startBtn.disabled = true;
            startBtn.innerHTML = '<span class="spinner-sm"></span> Verifying...';
            document.getElementById('stop-verify-btn')?.removeAttribute('disabled');
        } else {
            startBtn.disabled = false;
            startBtn.innerHTML = 'Trigger Verification Sync';
            document.getElementById('stop-verify-btn')?.setAttribute('disabled', 'true');
        }
      }
    }
  } catch (_) {}
}

async function startVerification(passedSid = null) {
  const sidInput = document.getElementById('verify-session-id');
  const sid = passedSid || (sidInput ? sidInput.value : null);
  if (!sid) return toast('Please select or enter a Session ID', 'info');

  showLayerModal(sid, 'verify', async (sessionId, start, end) => {
    const startBtn = document.getElementById('start-verify-btn');
    const stopBtn  = document.getElementById('stop-verify-btn');
    
    if (startBtn) { startBtn.disabled = true; startBtn.innerHTML = '<span class="spinner-sm"></span> Auditing...'; }
    if (stopBtn) stopBtn.disabled = false;

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
          if (startBtn) { startBtn.disabled = false; startBtn.innerHTML = 'Trigger Verification'; }
      }
    } catch (e) {
      toast('Network error starting verification', 'error');
      if (startBtn) { startBtn.disabled = false; startBtn.innerHTML = 'Trigger Verification'; }
    }
  });
}


async function startRegeneration(sid) {
    showLayerModal(sid, 'proof', async (sessionId, start, end) => {
        toast(`Regenerating proof for session #${sessionId}...`, 'info');
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
        const resp = await fetch(endpoint);
        const data = await resp.json();
        
        if (!data.sessions || data.sessions.length === 0) {
            const colspan = (isVerifier || isProvider) ? 8 : 7;
            body.innerHTML = `<tr><td colspan="${colspan}" style="padding:3rem; text-align:center; opacity:0.5;">No execution records found.</td></tr>`;
            return;
        }

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

            const actions = isVerifier ? `
                <div style="display:flex; justify-content:flex-end; gap:0.5rem;">
                    <button class="btn btn-sm" onclick="startVerification('${s.id}')" style="background:rgba(16,185,129,0.1); color:var(--emerald); border-color:rgba(16,185,129,0.2);">Verify</button>
                    <button class="btn btn-sm btn-ghost" onclick="deleteSession('${s.id}')" style="color:var(--rose);">Purge</button>
                </div>
            ` : (isProvider ? `
                <div style="display:flex; justify-content:flex-end; gap:0.5rem;">
                    <button class="btn btn-sm" onclick="startRegeneration('${s.id}')" style="background:rgba(99,102,241,0.1); color:var(--indigo); border-color:rgba(99,102,241,0.2);">Regen</button>
                    <button class="btn btn-sm btn-ghost" onclick="deleteSession('${s.id}')" style="color:var(--rose);">Purge</button>
                </div>
            ` : `
                <div style="display:flex; justify-content:flex-end; gap:0.5rem;">
                    ${s.proof_status === 'running' ? `<button class="btn btn-sm" onclick="stopProcess('${s.id}', 'proof')" style="background:rgba(244,63,94,0.1); color:var(--rose);">Stop</button>` : ''}
                    ${s.verify_status === 'running' ? `<button class="btn btn-sm" onclick="stopProcess('${s.id}', 'verify')" style="background:rgba(244,63,94,0.1); color:var(--rose);">Abort</button>` : ''}
                    <button class="btn btn-sm" onclick="startRegeneration('${s.id}')" style="background:rgba(99,102,241,0.1); color:var(--indigo); border-color:rgba(99,102,241,0.2);">Regen</button>
                    <button class="btn btn-sm btn-ghost" onclick="deleteSession('${s.id}')" style="color:var(--rose);"><svg style="width:1rem;height:1rem;" viewBox="0 0 24 24"><path d="M19 4h-3.5l-1-1h-5l-1 1H5v2h14V4zM6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12z" fill="currentColor"/></svg></button>
                </div>
            `);

            const timestamp = s.timestamp ? s.timestamp.split(' ')[1] || s.timestamp : '--:--';
            const dateStr = s.timestamp ? s.timestamp.split(' ')[0] : '';

            if (isVerifier || isProvider) {
                const c2paBadge = s.c2pa_status === 'signed'
                    ? `<a href="/api/c2pa/receipt/${s.id}" class="badge badge-success" style="text-decoration:none; cursor:pointer; font-size:0.68rem;" title="Download C2PA Receipt">🔏 Receipt</a>`
                    : `<span class="badge badge-neutral" style="opacity:0.4; font-size:0.68rem;">—</span>`;
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
                        <td style="padding:1rem; text-align:center;">${c2paBadge}</td>
                        <td style="padding:1rem;">${actions}</td>
                    </tr>
                `;
            } else {
                const c2paBadge = s.c2pa_status === 'signed'
                    ? `<a href="/api/c2pa/receipt/${s.id}" class="badge badge-success" style="text-decoration:none; cursor:pointer; font-size:0.68rem;" title="Download Signed JPEG Receipt">🔏 Signed JPEG</a>`
                    : `<span class="badge badge-neutral" style="opacity:0.4; font-size:0.68rem;">—</span>`;
                
                let manifestInfo = '<span style="opacity:0.5; font-size:0.75rem;">None</span>';
                if (s.c2pa_manifest) {
                    try {
                        const m = JSON.parse(s.c2pa_manifest);
                        manifestInfo = `<div style="font-size:0.7rem; color:var(--emerald); font-weight:600; line-height:1.1;">Verified AI<br><span style="opacity:0.7; font-weight:400; font-size:0.65rem;">${m.claim_generator || 'C2PA-sys'}</span></div>`;
                    } catch(e) {
                        manifestInfo = '<span style="opacity:0.5; font-size:0.75rem;">Valid</span>';
                    }
                }

                return `
                    <tr class="fade-in">
                        <td style="padding:1rem; font-family:var(--font-mono); font-size:0.75rem; font-weight:700; opacity:0.7;">#${s.id}</td>
                        <td style="padding:1rem; font-size:0.85rem; font-weight:700;">${timestamp}</td>
                        <td style="padding:1rem; max-width:250px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:0.9rem; font-weight:600;" title="${escapeHTML(s.prompt)}">${escapeHTML(s.prompt)}</td>
                        <td style="padding:1rem; font-size:0.85rem; font-weight:700;">${s.model_id || 'llama-2'}</td>
                        <td style="padding:1rem;">${getStatusBadge(s.proof_status, s.proof_duration)}</td>
                        <td style="padding:1rem;">${manifestInfo}</td>
                        <td style="padding:1rem; text-align:center;">${c2paBadge}</td>
                        <td style="padding:1rem;">${actions}</td>
                    </tr>
                `;
            }
        }).join('');
    } catch (err) {
        console.error('Session load failure', err);
    }
}


async function stopProcess(id, type) {
    if (!confirm(`Are you sure you want to stop this ${type} process?`)) return;
    try {
        const resp = await fetch(`/api/${type}/stop/${id}`, { method: 'POST' });
        const data = await resp.json();
        if (data.status === 'success' || data.success) {
            toast('Process termination signal sent', 'success');
            setTimeout(loadSessions, 1000);
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
            setTimeout(loadSessions, 500);
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
  if (t) t.innerHTML = '';
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
document.addEventListener('DOMContentLoaded', async () => {
    const path = window.location.pathname;
    if (path.includes('provider')) {
        await initProvider();
    } else if (path.includes('verifier')) {
        await initVerifier();
    } else if (path.includes('dashboard') || path.includes('user')) {
        await initUser();
    }
});

#!/usr/bin/env python3
"""
VerifAI — Zero-Knowledge LLM Verification Interface
Flask backend with real-time SocketIO progress streaming.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os, subprocess, json, threading, time, re, sqlite3, random, datetime as _dt
from functools import wraps
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'verifai-secret-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Make session info available in all templates
@app.context_processor
def inject_user():
    return dict(
        logged_in = 'user_id' in session,
        username = session.get('username'),
        user_role = session.get('role')
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).parent.parent          # zkllm_v2/
MODEL_STORE  = BASE_DIR / "model-storage"
ACT_DIR      = BASE_DIR / "activations"
DB_PATH      = BASE_DIR / "zkllm.db"

# Use the dedicated Conda environment for all backend scripts
PYTHON_EXE = "/home/abhi/miniconda3/envs/zkllm-env/bin/python3"

def get_db_conn():
    """Helper to get a DB connection with WAL mode and timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    with get_db_conn() as conn:
        # User-linked sessions
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                prompt TEXT,
                model_id TEXT,
                request_verify BOOLEAN,
                inference_status TEXT,
                proof_status TEXT,
                verify_status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        # Secure user table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                approved INTEGER DEFAULT 0
            )
        ''')
        
        # Check if we need to migrate user_id column if it doesn't exist (rough migration)
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER REFERENCES users(id)")
        except sqlite3.OperationalError:
            pass # Already exists
        
        # Check if we need to migrate password to password_hash
        try:
             conn.execute("ALTER TABLE users RENAME COLUMN password TO password_hash")
        except sqlite3.OperationalError:
            pass

        # C2PA provenance columns
        for col, ctype in [("c2pa_status", "TEXT"), ("c2pa_receipt", "BLOB"), ("c2pa_manifest", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {ctype} DEFAULT NULL")
            except sqlite3.OperationalError:
                pass
        
            # Store model response for conversation playback
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN response TEXT")
            except sqlite3.OperationalError:
                pass

        # Track how many times verification has been triggered for each session.
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN verify_runs INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("UPDATE sessions SET verify_runs = 0 WHERE verify_runs IS NULL")
        except sqlite3.OperationalError:
            pass

        # Persist verifier telemetry for post-run audit inspection in UI.
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN verify_log TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN verify_audit TEXT")
        except sqlite3.OperationalError:
            pass

        # Persist proof-generation telemetry for session detail popup.
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN proof_log TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN proof_audit TEXT")
        except sqlite3.OperationalError:
            pass

init_db()

# Initialize C2PA certificate at startup
from c2pa_signer import init_certs as _c2pa_init, sign_inference as _c2pa_sign, inspect_file as _c2pa_inspect
_c2pa_init()

# ----------------------
# Auth helpers
# ----------------------
def login_required(roles=None):
    if isinstance(roles, str):
        roles = [roles]
        
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            if roles and session.get('role') not in roles and session.get('role') != 'master':
                flash('Access denied: insufficient permissions.', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def get_optimized_env():
    """Get environment with optimized CUDA memory settings for 13B models."""
    env = os.environ.copy()
    # High-pressure memory tuning: help prevent fragmentation crashes
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure subprocesses don't over-subscribe CPU threads during ZKP
    env["OMP_NUM_THREADS"] = "4" 
    env["MKL_NUM_THREADS"] = "4"
    # Favor managed memory for very large CUDA allocations when needed
    env.setdefault("ZKLLM_MANAGED_THRESHOLD_MB", "2048")
    return env

def get_user_by_username(username):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE username = ?', (username,))
        return cur.fetchone()

def create_user(username, password, role):
    # Master and provider need manual approval usually, but for this demo:
    # 'user' is auto-approved, others need master approval
    approved = 1 if role == 'user' else 0
    password_hash = generate_password_hash(password)
    with get_db_conn() as conn:
        try:
            conn.execute('INSERT INTO users (username, password_hash, role, approved) VALUES (?, ?, ?, ?)', 
                         (username, password_hash, role, approved))
            return True
        except sqlite3.IntegrityError:
            return False

def verify_user(username, password):
    user = get_user_by_username(username)
    # user[2] is password_hash
    if user and check_password_hash(user[2], password):
        # user[4] is approved
        if user[3] == 'user' or user[3] == 'master' or user[4] == 1:
            return user
        else:
            return 'not_approved'
    return None

# ---------------------------------------------------------------------------
# Model detection — auto-discover from model-storage/
# ---------------------------------------------------------------------------
# Known model specs (extend as needed)
MODEL_SPECS = {
    7:  {"layers": 32, "hidden": 4096, "heads": 32},
    13: {"layers": 40, "hidden": 5120, "heads": 40},
}

def detect_models():
    """Scan model-storage/ for HuggingFace model cache directories."""
    models = {}
    if not MODEL_STORE.exists():
        return models
    for d in sorted(MODEL_STORE.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"models--(.+?)--(.+)", d.name)
        if not m:
            continue
        org  = m.group(1).replace("-", "-")
        name = m.group(2)  # e.g. Llama-2-7b-hf
        # Extract size from name
        size_m = re.search(r"(\d+)b", name, re.IGNORECASE)
        size   = int(size_m.group(1)) if size_m else 0
        specs  = MODEL_SPECS.get(size, {"layers": 32, "hidden": 4096, "heads": 32})
        mid    = f"llama-2-{size}b"
        workdir = BASE_DIR / "zkllm-workdir" / f"Llama-2-{size}b"
        models[mid] = {
            "id":      mid,
            "name":    f"Llama-2-{size}B",
            "card":    f"{org}/{name}",
            "size":    size,
            "params":  f"{size}B",
            "layers":  specs["layers"],
            "hidden":  specs["hidden"],
            "heads":   specs["heads"],
            "seq_len": 128,
            "curve":   "BLS12-381",
            "proof":   "Sumcheck",
            "workdir": str(workdir),
            "status":  "available" if workdir.exists() else "no workdir",
        }
    return models

MODELS = detect_models()

def get_workdir(model_id=None):
    """Return workdir Path for a model id."""
    if model_id and model_id in MODELS:
        return Path(MODELS[model_id]["workdir"])
    # Fallback to first available
    if MODELS:
        return Path(next(iter(MODELS.values()))["workdir"])
    return BASE_DIR / "zkllm-workdir" / "Llama-2-7b"

# ---------------------------------------------------------------------------
# Shared state (simple in-memory — fine for single-server demo)
# ---------------------------------------------------------------------------
state = {
    "proof_job":    None,
    "verify_job":   None,
    "inference_job": None,
    "last_prompt":  None,
    "last_response": None,
    "verify_results": [],
    "selected_model": next(iter(MODELS), None),
    "current_session_id": None,
}

# ---------------------------------------------------------------------------
# Route: pages
# ---------------------------------------------------------------------------

# ----------------------
# Page routes
# ----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        if create_user(username, password, role):
            flash('Signup successful. Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username already exists.', 'danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = verify_user(username, password)
        if user == 'not_approved':
            flash('Account not approved by master admin.', 'danger')
        elif user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            flash('Login successful.', 'success')
            # Redirect to dashboard based on role
            # Redirect to unified dashboard route
            return redirect(url_for('dashboard_router', role=user[3]))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

# ----------------------
# Dashboards (role-based)
# ----------------------
@app.route('/dashboard')
@login_required()
def dashboard():
    return redirect(url_for('dashboard_router', role=session.get('role', 'user')))

# Unified Dashboard Router
@app.route('/dashboard/<role>')
@login_required()
def dashboard_router(role):
    # Ensure user can only access their assigned role's dashboard (except Master)
    if session.get('role') != role and session.get('role') != 'master':
        return redirect(url_for('dashboard_router', role=session.get('role')))
    
    template_map = {
        'master': 'master_dashboard.html',
        'provider': 'provider_dashboard.html',
        'user': 'user_dashboard.html',
        'verifier': 'verifier_dashboard.html'
    }
    
    pending_approvals = 0
    if role == 'master':
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM users WHERE approved = 0')
            pending_approvals = cur.fetchone()[0]

    return render_template(template_map.get(role, 'index.html'), 
                           user_role=session.get('role'),
                           username=session.get('username'),
                           pending_approvals=pending_approvals)

@app.route('/manage_users', methods=['GET', 'POST'])
@login_required(roles='master')
def manage_users():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if user_id:
            with get_db_conn() as conn:
                conn.execute('UPDATE users SET approved=1 WHERE id=?', (user_id,))
            flash('User approved.', 'success')
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute('SELECT id, username, role, approved FROM users WHERE role != "master"')
        users = cur.fetchall()
    return render_template('manage_users.html', 
                           users=users, 
                           user_role=session.get('role'), 
                           username=session.get('username'))


# Legacy direct pages (now redirect to dashboards)
@app.route('/provider')
@login_required(roles='provider')
def provider():
    return redirect(url_for('dashboard_router', role='provider'))

@app.route('/user')
@login_required(roles='user')
def user():
    return redirect(url_for('dashboard_router', role='user'))

@app.route('/verifier')
@login_required(roles='verifier')
def verifier():
    return redirect(url_for('dashboard_router', role='verifier'))

# ---------------------------------------------------------------------------
# API — Models
# ---------------------------------------------------------------------------
@app.route('/api/models')
def api_models():
    available = []
    for m in MODELS.values():
        available.append(dict(m))
        
    return jsonify({
        "available_models": available,
        "selected_model": state.get("selected_model", "llama-2-7b")
    })

@app.route('/api/models/<model_id>', methods=['GET', 'POST'])
def api_model_detail(model_id):
    m = MODELS.get(model_id)
    if not m:
        return jsonify({"error": "not found"}), 404
        
    if request.method == 'POST':
        state['selected_model'] = model_id
        return jsonify({"success": True, "model": m})
        
    return jsonify(m)

# ---------------------------------------------------------------------------
# API — Provider: commitments
# ---------------------------------------------------------------------------
@app.route('/api/commitments')
@app.route('/api/commitments/<model_id>')
def api_commitments(model_id=None):
    """List commitment files grouped by layer for a model."""
    model_id = model_id or request.args.get('model') or state.get('selected_model')
    workdir = get_workdir(model_id)
    if not workdir or not workdir.exists():
        return jsonify({"layers": [], "total": 0, "model": model_id})

    files = sorted(workdir.glob("*-commitment.bin"))
    flat_commitments = []
    for f in files:
        cm = re.match(r"layer-(\d+)-(.+)-commitment\.bin", f.name)
        if cm:
            ln = int(cm.group(1))
            comp = cm.group(2).replace('.weight', '')
            flat_commitments.append({
                "layer": f"Layer {ln}",
                "param": comp,
                "file": f.name,
                "size": f"{f.stat().st_size / 1024 / 1024:.2f} MB"
            })
            
    # Sort by layer number then param name
    flat_commitments.sort(key=lambda x: (int(x["layer"].split(" ")[1]), x["param"]))
    
    return jsonify({"commitments": flat_commitments, "total": len(files), "model": model_id})

@app.route('/api/provider/activity')
@login_required(roles='provider')
def api_provider_activity():
    """Show which users are using models in this environment."""
    with get_db_conn() as conn:
        conn.row_factory = sqlite3.Row
        # In a real system, we'd filter by models owned by the provider.
        # Here we show all sessions but hide non-relevant info.
        cursor = conn.execute("""
            SELECT s.id, u.username, s.model_id, s.timestamp, s.inference_status,
                   s.proof_status, s.verify_status, s.proof_duration, s.verify_duration
            FROM sessions s 
            JOIN users u ON s.user_id = u.id 
            ORDER BY s.id DESC
        """)
        activity = [dict(row) for row in cursor.fetchall()]
    return jsonify({"activity": activity})

@app.route('/api/models/commitment_status/<model_id>')
def api_model_commitment_status(model_id):
    """Check if model has any commitment files."""
    workdir = get_workdir(model_id)
    has_commitments = False
    if workdir.exists():
        files = list(workdir.glob("*-commitment.bin"))
        has_commitments = len(files) > 0
    return jsonify({"model_id": model_id, "has_commitments": has_commitments})

@app.route('/api/commitments/upload', methods=['POST'])
def api_upload_commitment():
    """Upload a new commitment file for the active model."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    model_id = request.form.get('model', state.get('selected_model'))
    workdir = get_workdir(model_id)
    workdir.mkdir(parents=True, exist_ok=True)
    
    # Save the file
    filename = secure_filename(file.filename)
    if not filename.endswith("-commitment.bin"):
        return jsonify({"error": "File must end with -commitment.bin"}), 400
        
    filepath = workdir / filename
    file.save(filepath)
    return jsonify({"success": True, "filename": filename})

# ---------------------------------------------------------------------------
# API — User: prompt
# ---------------------------------------------------------------------------
@app.route('/api/prompt', methods=['POST'])
def api_prompt():
    """Accept user prompt, run real inference to capture activations."""
    data = request.json or {}
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({"error": "empty prompt"}), 400

    raw_max_new_tokens = data.get('max_new_tokens', 1)
    try:
        max_new_tokens = int(raw_max_new_tokens)
    except (TypeError, ValueError):
        return jsonify({"error": "max_new_tokens must be an integer"}), 400
    if max_new_tokens < 1:
        return jsonify({"error": "max_new_tokens must be >= 1"}), 400

    # Server-side policy: keep token-k hidden from end users.
    # - Single token request: capture prefill pass.
    # - Multi-token request: hybrid mode — capture prefill (all prompt tokens)
    #   AND one random decode step for strongest cryptographic guarantee.
    capture_mode = 'hybrid' if max_new_tokens > 1 else 'prefill'

    selected_model_id = state.get("selected_model", "llama-2-7b")
    model_info = MODELS.get(selected_model_id) or MODELS.get("llama-2-7b") or {"layers":32, "size":7, "seq_len":128}
    model_size = 13 if "13b" in selected_model_id.lower() else 7
    seq = model_info["seq_len"]
    
    request_verify = data.get('request_verify', False)
    user_id = session.get('user_id')
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO sessions (user_id, prompt, model_id, request_verify, inference_status, proof_status, verify_status)
            VALUES (?, ?, ?, ?, 'running', 'pending', 'pending')
        ''', (user_id, prompt, selected_model_id, request_verify))
        session_id = cur.lastrowid
        conn.commit()
    state["current_session_id"] = session_id
    
    # Run the activation capture script
    cmd = [
        PYTHON_EXE, str(BASE_DIR / "capture_activations.py"),
        "--text", prompt,
        "--model_size", str(model_size),
        "--output_dir", str(ACT_DIR / str(session_id)),
        "--max_seq_len", str(seq),
        "--max_new_tokens", str(max_new_tokens),
        "--capture_mode", str(capture_mode),
    ]
    if model_size > 7:
        cmd.append("--cpu")  # Still force CPU for 13B on 6GB card to prevent hard OOM




    
    try:
        # Launch background inference
        uname = session.get('username', 'anonymous')
        run_inference_task(session_id, cmd, uname)
        return jsonify({
            "success": True, 
            "session_id": session_id,
            "message": "Inference engine started in background"
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to start inference: {e}"}), 500

def run_inference_task(sid, cmd, username):
    """Background task to run capture_activations.py and stream logs."""
    def run():
        state["inference_job"] = {
            "running": True,
            "session_id": sid,
            "started_at": time.time(),
            "log": [],
            "progress": 0
        }
        
        socketio.emit("inference_started", {"session_id": sid})
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR),
            env=get_optimized_env()
        )
        
        pred_token = "unknown"
        generated_text = ""
        generated_token_count = 1
        for line in proc.stdout:
            line = line.strip()
            if not line: continue
            
            state["inference_job"]["log"].append(line)
            socketio.emit("inference_log", {"sid": sid, "line": line})
            
            # Progress heuristics
            if "Loading LLaMA-2" in line:
                socketio.emit("inference_progress", {"sid": sid, "pct": 20, "status": "Loading weights..."})
            elif "Registering Hooks" in line:
                socketio.emit("inference_progress", {"sid": sid, "pct": 50, "status": "Configuring ZK layers..."})
            elif "Executing forward pass" in line:
                socketio.emit("inference_progress", {"sid": sid, "pct": 70, "status": "Synthesing proofs..."})
            elif "Generating continuation" in line:
                socketio.emit("inference_progress", {"sid": sid, "pct": 85, "status": "Generating continuation..."})
            elif "Predicted next token:" in line:
                m = re.match(r"^Predicted next token:\s*'(.*)'$", line)
                pred_token = m.group(1) if m else line.split(":", 1)[1].strip().strip("'")
            elif "Generated token count:" in line:
                try:
                    generated_token_count = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif "Generated text JSON:" in line:
                payload = line.split(":", 1)[1].strip()
                try:
                    generated_text = json.loads(payload)
                except Exception:
                    generated_text = payload.strip().strip("'")
                socketio.emit("inference_progress", {"sid": sid, "pct": 95, "status": "Finalizing..."})

            socketio.sleep(0) # Yield for event loop

        proc.wait()
        state["inference_job"]["running"] = False
        
        if proc.returncode == 0:
            final_text = generated_text if generated_text else pred_token
            if generated_token_count > 1:
                resp_text = f"[Inference complete.]\n\nGenerated continuation ({generated_token_count} tokens): {final_text}"
            else:
                resp_text = f"[Inference complete.]\n\nPredicted next token: {final_text}"
            with get_db_conn() as conn:
                conn.execute(
                        "UPDATE sessions SET inference_status = 'completed', response = ? WHERE id = ?",
                        (resp_text, sid)
                )
            
            # C2PA Signing Hook
            try:
                # Prepare timestamp
                ts = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                # Get the model_id from DB
                with get_db_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT prompt, model_id FROM sessions WHERE id = ?", (sid,))
                    row = cur.fetchone()
                    if row:
                        db_prompt, db_model = row
                        # sign_inference(session_id, prompt, response, model_id, username, timestamp)
                        receipt_blob, manifest_summary = _c2pa_sign(
                            sid, db_prompt, resp_text, db_model, username, ts
                        )
                        conn.execute(
                            "UPDATE sessions SET c2pa_status='signed', c2pa_receipt=?, c2pa_manifest=? WHERE id=?",
                            (sqlite3.Binary(receipt_blob), json.dumps(manifest_summary), sid)
                        )
            except Exception as e:
                print(f"C2PA Signing failed: {e}")

            socketio.emit("inference_complete", {
                "sid": sid, 
                "success": True, 
                "prediction": final_text,
                "generated_text": final_text,
                "generated_token_count": generated_token_count,
                "response": resp_text
            })
        else:
            # Unmask true tracebacks from stdout/stderr piped buffers
            raw_logs = state["inference_job"].get("log", [])
            last_err = raw_logs[-1] if raw_logs else f"RC: {proc.returncode}"
            err_msg = f"Process crashed (likely OOM). Reason: {str(last_err)}"
            with get_db_conn() as conn:
                conn.execute("UPDATE sessions SET inference_status = 'failed', response = ? WHERE id = ?", (err_msg, sid))
            socketio.emit("inference_complete", {"sid": sid, "success": False, "error": err_msg})

    threading.Thread(target=run, daemon=True).start()

# ---------------------------------------------------------------------------
# API — Sessions
# ---------------------------------------------------------------------------
@app.route('/api/sessions/<int:id>', methods=['GET'])
@login_required()
def api_session_detail(id):
    """Retrieve detailed state for a specific session."""
    with get_db_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Avoid returning binary blobs (e.g., c2pa_receipt) which are not JSON-serializable.
        schema = conn.execute("PRAGMA table_info(sessions)").fetchall()
        available = {row[1] for row in schema}
        wanted = [
            'id', 'user_id', 'prompt', 'response', 'model_id', 'request_verify',
            'inference_status', 'proof_status', 'verify_status', 'timestamp',
            'proof_duration', 'verify_duration', 'verify_runs',
            'proof_log', 'proof_audit', 'verify_log', 'verify_audit',
            'c2pa_status', 'c2pa_manifest'
        ]
        selected = [c for c in wanted if c in available]
        cols = ", ".join(selected) if selected else "id, prompt, model_id, inference_status, proof_status, verify_status, timestamp"

        if session.get('role') in ['verifier', 'master', 'provider']:
            cur.execute(f"SELECT {cols} FROM sessions WHERE id = ?", (id,))
        else:
            cur.execute(f"SELECT {cols} FROM sessions WHERE id = ? AND user_id = ?", (id, session.get('user_id')))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "Session not found"}), 404
        payload = dict(row)
        payload.setdefault('response', None)

        # Attach username for richer metadata display in session details popup.
        uid = payload.get('user_id')
        if uid is not None:
            urow = conn.execute("SELECT username FROM users WHERE id = ?", (uid,)).fetchone()
            if urow:
                payload['username'] = urow['username'] if isinstance(urow, sqlite3.Row) else urow[0]

        # Decode serialized JSON payloads where available.
        for key in ('proof_log', 'proof_audit', 'verify_log', 'verify_audit', 'c2pa_manifest'):
            if key in payload and isinstance(payload[key], str) and payload[key].strip():
                try:
                    payload[key] = json.loads(payload[key])
                except Exception:
                    pass

        # Fallback to current in-memory verifier buffers for the active/last run.
        active_vjob = state.get("verify_job") or {}
        if active_vjob.get("session_id") == id:
            if not payload.get("verify_log"):
                payload["verify_log"] = active_vjob.get("log", [])[-300:]
            if not payload.get("verify_audit"):
                payload["verify_audit"] = active_vjob.get("results", [])[-300:]

        active_pjob = state.get("proof_job") or {}
        if active_pjob.get("session_id") == id:
            if not payload.get("proof_log"):
                payload["proof_log"] = active_pjob.get("log", [])[-300:]
            if not payload.get("proof_audit"):
                payload["proof_audit"] = active_pjob.get("results", [])[-300:]
        return jsonify(payload)

@app.route('/api/sessions', methods=['GET'])
@login_required()
def api_sessions():
    user_id = session.get('user_id')
    role = session.get('role')
    
    with get_db_conn() as conn:
        conn.row_factory = sqlite3.Row
        if role == 'master':
            # Master sees everything
            cursor = conn.execute("SELECT s.*, u.username FROM sessions s LEFT JOIN users u ON s.user_id = u.id ORDER BY s.id DESC")
        else:
            # Others see only their own
            cur = conn.cursor()
            cur.execute("""
                SELECT s.id, s.user_id, s.prompt, s.model_id, s.request_verify, 
                       s.inference_status, s.proof_status, s.verify_status, 
                       s.timestamp, s.c2pa_status, s.c2pa_manifest, u.username,
                       s.proof_duration, s.verify_duration, s.verify_runs
                FROM sessions s 
                JOIN users u ON s.user_id = u.id 
                WHERE s.user_id = ? 
                ORDER BY s.timestamp DESC
            """, (session['user_id'],))
            sessions_list = [dict(ix) for ix in cur.fetchall()]
    return jsonify({"sessions": sessions_list})

@app.route('/api/all_sessions')
@login_required()
def api_all_sessions():
    if session.get('role') not in ['verifier', 'master', 'provider']:
        return jsonify({"error": "Unauthorized"}), 403
        
    model_filter = request.args.get('model')
    with get_db_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        query = """
            SELECT s.id, s.user_id, s.prompt, s.model_id, s.request_verify, 
                   s.inference_status, s.proof_status, s.verify_status, 
                   s.timestamp, s.c2pa_status, s.c2pa_manifest, u.username,
                   s.proof_duration, s.verify_duration, s.verify_runs
            FROM sessions s 
            LEFT JOIN users u ON s.user_id = u.id 
        """
        params = []
        if model_filter:
            query += " WHERE s.model_id = ?"
            params.append(model_filter)
        
        query += " ORDER BY s.timestamp DESC"
        cur.execute(query, params)
        sessions_list = [dict(ix) for ix in cur.fetchall()]
    return jsonify({"sessions": sessions_list})

@app.route('/api/sessions/delete/<int:id>', methods=['DELETE'])
@login_required()
def api_session_delete(id):
    with get_db_conn() as conn:
        # If not verifier/master, only allow deleting own sessions
        if session.get('role') not in ['verifier', 'master']:
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM sessions WHERE id = ?", (id,))
            row = cur.fetchone()
            if not row or row[0] != session.get('user_id'):
                return jsonify({"error": "Unauthorized"}), 403
                
        conn.execute("DELETE FROM sessions WHERE id = ?", (id,))
    return jsonify({"success": True})

# ---------------------------------------------------------------------------
# API — C2PA Content Provenance
# ---------------------------------------------------------------------------
@app.route('/api/c2pa/receipt/<int:session_id>')
@login_required()
def api_c2pa_receipt(session_id):
    """Download the C2PA-signed JPEG receipt for a session."""
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT c2pa_receipt, c2pa_status FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        return jsonify({"error": "No C2PA receipt for this session"}), 404
    from io import BytesIO
    return send_file(
        BytesIO(row[0]),
        mimetype="image/jpeg",
        as_attachment=True,
        download_name=f"verifai_receipt_{session_id}.jpg",
    )

@app.route('/api/c2pa/inspect', methods=['POST'])
@login_required()
def api_c2pa_inspect():
    """Upload a file and extract its C2PA manifest."""
    uploaded = request.files.get("file")
    if not uploaded or uploaded.filename == "":
        return jsonify({"error": "No file uploaded"}), 400
    data = uploaded.read()
    result = _c2pa_inspect(data)
    return jsonify(result)

@app.route('/api/c2pa/verify_match/<int:session_id>', methods=['POST'])
@login_required()
def api_c2pa_verify_match(session_id):
    """Upload a receipt and verify it against a specific session's ledger manifest."""
    uploaded = request.files.get("file")
    if not uploaded or uploaded.filename == "":
        return jsonify({"error": "No file uploaded"}), 400
    
    data = uploaded.read()
    uploaded_manifest = _c2pa_inspect(data)
    
    if "error" in uploaded_manifest:
        return jsonify(uploaded_manifest), 400
        
    with get_db_conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT c2pa_manifest FROM sessions WHERE id = ?", (session_id,)).fetchone()
        
    if not row or not row['c2pa_manifest']:
        return jsonify({"error": "No C2PA manifest found for this session ID"}), 404
        
    import json
    try:
        db_manifest = json.loads(row['c2pa_manifest'])
    except Exception:
        db_manifest = {}
        
    # Determine if they match (in real world, requires deep cryptographic payload dict diff)
    is_match = (uploaded_manifest.get('claim_generator') == db_manifest.get('claim_generator') and 
                uploaded_manifest.get('model_id') == db_manifest.get('model_id') and
                uploaded_manifest.get('prompt') == db_manifest.get('prompt'))
                
    return jsonify({
        "success": True,
        "match": is_match,
        "uploaded_manifest": uploaded_manifest,
        "database_manifest": db_manifest
    })

# ---------------------------------------------------------------------------
# API — Proof generation (master script)
# ---------------------------------------------------------------------------
@app.route('/api/proof/start', methods=['POST'])
def api_proof_start():
    """Launch generate_proofs_v2.py in a background thread."""
    if state["proof_job"] and state["proof_job"].get("running"):
        return jsonify({"error": "proof job already running"}), 409

    data   = request.json or {}
    sid    = data.get("session_id") or state.get("current_session_id")
    start  = data.get("start_layer", 0)
    
    selected_model_id = data.get("model")
    if not selected_model_id and sid:
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT model_id FROM sessions WHERE id = ?", (sid,)).fetchone()
            if row:
                selected_model_id = row['model_id']
    
    if not selected_model_id:
        selected_model_id = state.get("selected_model", "llama-2-7b")

    model_info = MODELS.get(selected_model_id) or MODELS.get("llama-2-7b") or {"layers":32, "size":7, "seq_len":128}
    try:
        start = int(data.get("start_layer", 0))
    except (ValueError, TypeError):
        start = 0
        
    try:
        layers_total = int(model_info.get("layers", 32))
        end = int(data.get("end_layer", layers_total - 1))
    except (ValueError, TypeError):
        end = int(model_info.get("layers", 32)) - 1

    layers_total = int(model_info.get("layers", 32))
    start = max(0, min(start, layers_total - 1))
    end = max(start, min(end, layers_total - 1))
        
    model_size = data.get("model_size", model_info["size"])
    seq = data.get("seq_len", model_info["seq_len"])

    state["proof_job"] = {
        "running": True, "progress": 0, "current_layer": start,
        "current_component": "", "total_layers": max(1, end - start + 1),
        "start_layer": start, "end_layer": end,
        "log": [], "results": [], "started_at": time.time(), "success": None,
        "session_id": sid
    }
    state["verify_results"] = []
    
    if sid:
        state["current_session_id"] = sid
        with get_db_conn() as conn:
            conn.execute(
                "UPDATE sessions SET proof_status = 'running', proof_log = NULL, proof_audit = NULL WHERE id = ?",
                (sid,)
            )

    def run():
        embed_dim = 4096  # LLaMA-2 embed_dim

        # ------- Phase detection: check if prefill activations exist ----------
        prefill_act_dir = ACT_DIR / str(sid) / "prefill"
        has_prefill = prefill_act_dir.is_dir() and (prefill_act_dir / "layer-0-block-input.bin").exists()

        # Auto-detect prefill seq_len from file size
        prefill_seq = None
        if has_prefill:
            try:
                fsize = (prefill_act_dir / "layer-0-block-input.bin").stat().st_size
                prefill_seq = fsize // (4 * embed_dim)  # int32 = 4 bytes
            except Exception:
                has_prefill = False

        # --- MODE SELECTION (Uncomment the desired strategy) ---

        # # [A] HYBRID MODE (Default: Decode Step + Prefill)
        # phases = [("decode", str(ACT_DIR / str(sid)), str(seq), str(sid) if sid else None)]
        # if has_prefill and prefill_seq and prefill_seq > 0:
        #      phases.append(("prefill", str(prefill_act_dir), str(prefill_seq), f"{sid}/prefill" if sid else "prefill"))

        # [B] DECODE ONLY MODE (Single token proof only)
        phases = [("decode", str(ACT_DIR / str(sid)), str(seq), str(sid) if sid else None)]

        # [C] PREFILL ONLY MODE (Whole prompt proof only)
        # if has_prefill and prefill_seq and prefill_seq > 0:
        #     phases = [("prefill", str(prefill_act_dir), str(prefill_seq), f"{sid}/prefill" if sid else "prefill")]


        total_phases = len(phases)
        total_layers_per_phase = max(1, end - start + 1)
        grand_total = total_phases * total_layers_per_phase

        all_success = True
        grand_started = time.time()

        for phase_idx, (phase_name, act_dir, phase_seq, run_id) in enumerate(phases):
            phase_label = f"[Phase {phase_idx+1}/{total_phases}: {phase_name.upper()}]"
            phase_msg = f"\n{'='*70}\n  {phase_label} Proof generation (seq_len={phase_seq})\n{'='*70}"
            state["proof_job"]["log"].append(phase_msg)
            socketio.emit("proof_log", {"line": phase_msg})
            socketio.emit("proof_phase", {
                "phase": phase_idx + 1, "total_phases": total_phases,
                "phase_name": phase_name, "seq_len": int(phase_seq),
                "session_id": state["proof_job"].get("session_id"),
            })

            cmd = [
                PYTHON_EXE, "-u", str(BASE_DIR / "generate_proofs_v2.py"),
                "--model_size", str(model_size),
                "--seq_len", str(phase_seq),
                "--start_layer", str(start),
                "--end_layer", str(end),
                "--model_card", str(model_info.get("card", f"meta-llama/Llama-2-{model_size}b-hf")),
                "--act_dir", act_dir
            ]
            if run_id:
                cmd += ["--run_id", str(run_id)]

            # Add workdir if it exists in model_info
            if "workdir" in model_info:
                cmd += ["--workdir", str(model_info["workdir"])]

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(BASE_DIR), bufsize=1, env=get_optimized_env()
            )
            state["proof_job"]["process"] = proc
            component_map = {
                "Input RMSNorm": 0, "Self-Attention": 1,
                "Post-Attn RMSNorm": 2, "Feed-Forward": 3,
                "Skip Connection": 4,
            }
            for line in proc.stdout:
                line = line.rstrip()
                tagged_line = f"{phase_label} {line}" if total_phases > 1 else line
                state["proof_job"]["log"].append(tagged_line)
                socketio.emit("proof_log", {"line": tagged_line})

                # Parse layer progress
                lm = re.search(r"\[(\d+)/5\]\s+(.+)", line)
                if lm:
                    step = int(lm.group(1))
                    comp = lm.group(2).strip()
                    state["proof_job"]["current_component"] = comp
                    state["proof_job"]["results"].append({
                        "ts": round(time.time(), 3),
                        "layer": state["proof_job"].get("current_layer", start),
                        "component": comp,
                        "status": "info",
                        "line": tagged_line,
                        "phase": phase_name,
                    })
                    # Compute progress across all phases
                    layer_progress = step / 5
                    layers_done_this_phase = state["proof_job"]["current_layer"] - start + layer_progress
                    grand_done = phase_idx * total_layers_per_phase + layers_done_this_phase
                    state["proof_job"]["progress"] = min(int(grand_done / grand_total * 100), 99)
                    socketio.emit("proof_progress", {
                        "layer": state["proof_job"]["current_layer"],
                        "total": state["proof_job"]["total_layers"],
                        "start_layer": state["proof_job"].get("start_layer", start),
                        "end_layer": state["proof_job"].get("end_layer", end),
                        "component": comp,
                        "percent": state["proof_job"]["progress"],
                        "session_id": state["proof_job"].get("session_id"),
                        "phase": phase_name,
                        "phase_idx": phase_idx + 1,
                        "total_phases": total_phases,
                    })

                # Detect layer start
                pm = re.search(r"PROCESSING LAYER (\d+)", line)
                if pm:
                    state["proof_job"]["current_layer"] = int(pm.group(1))

                # Detect layer completion
                sm = re.search(r"✅ Layer (\d+) completed", line)
                if sm:
                    state["proof_job"]["current_layer"] = int(sm.group(1)) + 1

                # Detect SUCCESS / FAILURE per component
                if "- SUCCESS" in line:
                    state["proof_job"]["results"].append({
                        "ts": round(time.time(), 3),
                        "layer": state["proof_job"].get("current_layer", start),
                        "component": state["proof_job"].get("current_component", ""),
                        "status": "pass",
                        "line": tagged_line,
                        "phase": phase_name,
                    })
                    socketio.emit("proof_component_done", {
                        "layer": state["proof_job"]["current_layer"],
                        "component": state["proof_job"]["current_component"],
                        "success": True,
                        "session_id": state["proof_job"].get("session_id"),
                        "phase": phase_name,
                    })
                elif "- FAILED" in line:
                    state["proof_job"]["results"].append({
                        "ts": round(time.time(), 3),
                        "layer": state["proof_job"].get("current_layer", start),
                        "component": state["proof_job"].get("current_component", ""),
                        "status": "fail",
                        "line": tagged_line,
                        "phase": phase_name,
                    })
                    socketio.emit("proof_component_done", {
                        "layer": state["proof_job"]["current_layer"],
                        "component": state["proof_job"]["current_component"],
                        "success": False,
                        "session_id": state["proof_job"].get("session_id"),
                        "phase": phase_name,
                    })

                if len(state["proof_job"]["results"]) > 1200:
                    state["proof_job"]["results"] = state["proof_job"]["results"][-1200:]

                # Yield to event loop
                socketio.sleep(0)

            proc.wait()
            if proc.returncode != 0:
                all_success = False
                # Log phase failure but continue to next phase
                fail_msg = f"{phase_label} ❌ FAILED (exit code {proc.returncode})"
                state["proof_job"]["log"].append(fail_msg)
                socketio.emit("proof_log", {"line": fail_msg})

            # Reset current_layer for next phase
            state["proof_job"]["current_layer"] = start

        # ---- All phases complete ----
        state["proof_job"]["running"]  = False
        duration = time.time() - grand_started
        state["proof_job"]["success"] = all_success
        state["proof_job"]["progress"] = 100 if all_success else -1

        if sid:
            with get_db_conn() as conn:
                status = "completed" if all_success else "failed"
                conn.execute(
                    "UPDATE sessions SET proof_status = ?, proof_duration = ?, proof_log = ?, proof_audit = ? WHERE id = ?",
                    (
                        status,
                        round(duration, 1),
                        json.dumps(state["proof_job"].get("log", [])[-1500:]),
                        json.dumps(state["proof_job"].get("results", [])[-1500:]),
                        sid,
                    )
                )

        socketio.emit("proof_complete", {
            "success": all_success,
            "elapsed": round(duration, 1),
            "start_layer": state["proof_job"].get("start_layer", start),
            "end_layer": state["proof_job"].get("end_layer", end),
            "session_id": state["proof_job"].get("session_id"),
            "phases_run": total_phases,
        })

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"success": True, "message": "Proof generation started", "session_id": sid, "hybrid": bool((ACT_DIR / str(sid) / "prefill").is_dir())})

@app.route('/api/proof/status')
def api_proof_status():
    if not state["proof_job"]:
        return jsonify({"active": False})
    job = state["proof_job"]
    return jsonify({
        "active": job["running"],
        "progress": job["progress"],
        "current_layer": job.get("current_layer"),
        "current_component": job.get("current_component", ""),
        "total_layers": job.get("total_layers"),
        "start_layer": job.get("start_layer"),
        "end_layer": job.get("end_layer"),
        "success": job.get("success"),
        "session_id": job.get("session_id"),
        "log_tail": job["log"][-100:],
        "elapsed": round(time.time() - job["started_at"], 1) if job["running"] else None,
    })

@app.route('/api/proof/stop', methods=['POST'])
@app.route('/api/proof/stop/<int:session_id>', methods=['POST'])
def api_proof_stop(session_id=None):
    if state["proof_job"] and state["proof_job"].get("running"):
        proc = state["proof_job"].get("process")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        state["proof_job"]["running"] = False
        state["proof_job"]["progress"] = -1
        state["proof_job"]["success"] = False
        socketio.emit("proof_complete", {
            "success": False,
            "elapsed": 0.0,
            "session_id": state["proof_job"].get("session_id")
        })
        
        # Use URL-supplied session ID first, fall back to current_session_id
        sid = session_id or state.get("current_session_id")
        if sid:
            with get_db_conn() as conn:
                conn.execute(
                    "UPDATE sessions SET proof_status = 'aborted', proof_log = ?, proof_audit = ? WHERE id = ?",
                    (
                        json.dumps(state["proof_job"].get("log", [])[-1500:]),
                        json.dumps(state["proof_job"].get("results", [])[-1500:]),
                        sid,
                    )
                )
                
        return jsonify({"success": True, "message": "Proof generation aborted"})
    # Even if no running process, mark the session aborted if we got a session_id
    if session_id:
        with get_db_conn() as conn:
            conn.execute("UPDATE sessions SET proof_status = 'aborted' WHERE id = ? AND proof_status = 'running'", (session_id,))
        return jsonify({"success": True, "message": "Session marked aborted"})
    return jsonify({"error": "No proof job running"}), 400

# ---------------------------------------------------------------------------
# API — Verification (master script)
# ---------------------------------------------------------------------------
@app.route('/api/verify/start', methods=['POST'])
def api_verify_start():
    """Launch verify_proofs_v2.py in a background thread."""
    if state["verify_job"] and state["verify_job"].get("running"):
        return jsonify({"error": "verify job already running"}), 409

    # Avoid concurrent proof/verify races that can cause nondeterministic verifier failures.
    if state["proof_job"] and state["proof_job"].get("running"):
        return jsonify({"error": "proof generation is still running; wait for completion before verification"}), 409

    data   = request.json or {}
    sid    = data.get("session_id") or state.get("current_session_id")
    
    selected_model_id = data.get("model")
    if not selected_model_id and sid:
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT model_id FROM sessions WHERE id = ?", (sid,)).fetchone()
            if row:
                selected_model_id = row['model_id']
    
    if not selected_model_id:
        selected_model_id = state.get("selected_model", "llama-2-7b")

    model_info = MODELS.get(selected_model_id) or MODELS.get("llama-2-7b") or {"layers":32, "size":7, "seq_len":128}
    try:
        start = int(data.get("start_layer", 0))
    except (ValueError, TypeError):
        start = 0
        
    try:
        layers_total = int(model_info.get("layers", 32))
        end = int(data.get("end_layer", layers_total - 1))
    except (ValueError, TypeError):
        end = int(model_info.get("layers", 32)) - 1

    layers_total = int(model_info.get("layers", 32))
    start = max(0, min(start, layers_total - 1))
    end = max(start, min(end, layers_total - 1))

    if sid:
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            srow = conn.execute(
                "SELECT proof_status FROM sessions WHERE id = ?",
                (sid,)
            ).fetchone()
        if not srow:
            return jsonify({"error": f"session {sid} not found"}), 404
        if srow["proof_status"] != "completed":
            return jsonify({
                "error": f"cannot verify session {sid}: proof_status is '{srow['proof_status']}', expected 'completed'"
            }), 409

    # Verify required proof artifacts exist for each selected layer before launching verifier.
    base_workdir = Path(model_info.get("workdir", str(get_workdir(selected_model_id))))
    proof_dir = (base_workdir / str(sid)) if sid else base_workdir
    required_suffixes = [
        "input-rmsnorm-proof.bin",
        "self-attn-proof.bin",
        "post-attn-rmsnorm-proof.bin",
        "ffn-proof.bin",
        "skip-proof.bin",
    ]
    missing = []
    for layer in range(start, end + 1):
        for suffix in required_suffixes:
            p = proof_dir / f"layer-{layer}-{suffix}"
            if not p.exists():
                missing.append(str(p))
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break

    if missing:
        return jsonify({
            "error": "verification artifacts missing; regenerate proofs for the selected layer range",
            "missing_examples": missing
        }), 409

    state["verify_job"] = {
        "running": True, "log": [], "results": [],
        "started_at": time.time(),
        "success": None,
        "session_id": sid,
        "progress": {
            "start_layer": start,
            "end_layer": end,
            "layer": start,
            "module": "input_rmsnorm",
            "module_label": "Input RMSNorm",
            "module_idx": 0,
            "percent": 0,
        },
        "pass_count": 0,
        "fail_count": 0,
        "last_update": round(time.time(), 3),
    }
    
    if sid:
        state["current_session_id"] = sid
        with get_db_conn() as conn:
            conn.execute(
                "UPDATE sessions SET verify_status = 'running', verify_runs = COALESCE(verify_runs, 0) + 1, verify_log = NULL, verify_audit = NULL WHERE id = ?",
                (sid,)
            )

    def run():
        proc = None
        success = False
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            model_size = model_info.get("size", 7)
            seq_len    = model_info.get("seq_len", 128)
            embed_dim = 4096  # LLaMA-2 embed_dim

            # ------- Phase detection for hybrid verify ----------
            prefill_act_dir = ACT_DIR / str(sid) / "prefill"
            base_workdir = Path(model_info.get("workdir", str(get_workdir(selected_model_id))))
            prefill_proof_dir = (base_workdir / f"{sid}_prefill") if sid else None

            has_prefill = (
                prefill_proof_dir is not None
                and prefill_proof_dir.is_dir()
                and prefill_act_dir.is_dir()
                and (prefill_act_dir / "layer-0-block-input.bin").exists()
            )

            # Auto-detect prefill seq_len from file size
            prefill_seq = None
            if has_prefill:
                try:
                    fsize = (prefill_act_dir / "layer-0-block-input.bin").stat().st_size
                    prefill_seq = fsize // (4 * embed_dim)
                except Exception:
                    has_prefill = False

            # --- MODE SELECTION (Uncomment the desired strategy) ---

            # [A] HYBRID MODE (Default: Decode Step + Prefill)
            phases = [("decode", str(ACT_DIR / str(sid)), str(seq_len), str(sid) if sid else None)]
            if has_prefill and prefill_seq and prefill_seq > 0:
                phases.append(("prefill", str(prefill_act_dir), str(prefill_seq), f"{sid}/prefill" if sid else "prefill"))

            # [B] DECODE ONLY MODE (Single token verification only)
            # phases = [("decode", str(ACT_DIR / str(sid)), str(seq_len), str(sid) if sid else None)]

            # [C] PREFILL ONLY MODE (Whole prompt verification only)
            # if has_prefill and prefill_seq and prefill_seq > 0:
            #     phases = [("prefill", str(prefill_act_dir), str(prefill_seq), f"{sid}/prefill" if sid else "prefill")]


            total_phases = len(phases)

            # Verification pipeline modules per layer (in order)
            MODULES = ["input_rmsnorm", "self_attn", "post_attn_rmsnorm", "ffn", "skip_connection"]
            MODULE_LABELS = {
                "input_rmsnorm":    "Input RMSNorm",
                "self_attn":        "Self-Attention",
                "post_attn_rmsnorm":"Post-Attn RMSNorm",
                "ffn":              "Feed-Forward (FFN)",
                "skip_connection":  "Skip Connection",
            }
            total_layers = max(end - start + 1, 1)
            _v_layer = [start]
            _v_mod = [0]
            _phase_label = [""]

            def emit_verify_progress():
                pct = int(((_v_layer[0] - start) * 5 + _v_mod[0]) / (total_layers * 5) * 100)
                progress_payload = {
                    "layer": _v_layer[0],
                    "total_layers": total_layers,
                    "start_layer": start,
                    "end_layer": end,
                    "module": MODULES[min(_v_mod[0], 4)],
                    "module_label": MODULE_LABELS.get(MODULES[min(_v_mod[0], 4)], ""),
                    "module_idx": _v_mod[0],
                    "percent": min(pct, 99),
                    "phase": _phase_label[0],
                }
                state["verify_job"]["progress"] = progress_payload
                state["verify_job"]["last_update"] = round(time.time(), 3)
                socketio.emit("verify_progress", progress_payload)

            def infer_parameter(raw_line):
                ll = raw_line.lower()
                mapping = [
                    ("q projection", "self_attn.q_proj"),
                    ("k projection", "self_attn.k_proj"),
                    ("v projection", "self_attn.v_proj"),
                    ("o projection", "self_attn.o_proj"),
                    ("q @ k", "self_attn.scores"),
                    ("pooling", "self_attn.pooling"),
                    ("softmax", "self_attn.softmax"),
                    ("up projection", "mlp.up_proj"),
                    ("gate projection", "mlp.gate_proj"),
                    ("down projection", "mlp.down_proj"),
                    ("swiglu", "mlp.swiglu"),
                    ("hadamard", "rmsnorm.hadamard"),
                    ("weight commitment", "weights.commitment"),
                    ("rescaling", "rmsnorm.rescaling"),
                    ("zero-check", "skip.zero_check"),
                    ("sumcheck", "sumcheck.transcript"),
                    ("claimed output", "claim.binding"),
                ]
                for needle, label in mapping:
                    if needle in ll:
                        return label
                return None

            def infer_level(raw_line):
                ll = raw_line.lower()
                if "❌" in raw_line or " failed" in ll or ll.startswith("error"):
                    return "fail"
                if "✅" in raw_line or " passed" in ll or " successful" in ll:
                    return "pass"
                if "⚠" in raw_line or "warning" in ll or "skipping" in ll or "bypassed" in ll:
                    return "warn"
                return "info"

            all_success = True

            for phase_idx, (phase_name, act_dir, phase_seq, run_id) in enumerate(phases):
                _phase_label[0] = phase_name
                phase_label_str = f"[Phase {phase_idx+1}/{total_phases}: {phase_name.upper()}]"

                phase_msg = f"\n{'='*70}\n  {phase_label_str} Verification (seq_len={phase_seq})\n{'='*70}"
                state["verify_job"]["log"].append(phase_msg)
                socketio.emit("verify_log", {"line": phase_msg})
                socketio.emit("verify_phase", {
                    "phase": phase_idx + 1, "total_phases": total_phases,
                    "phase_name": phase_name, "seq_len": int(phase_seq),
                    "session_id": state["verify_job"].get("session_id"),
                })

                # Reset layer tracking for this phase
                _v_layer[0] = start
                _v_mod[0] = 0

                cmd = [
                    PYTHON_EXE, "-u", str(BASE_DIR / "verify_proofs_v2.py"),
                    "--model_size",  str(model_size),
                    "--seq_len",     str(phase_seq),
                    "--start_layer", str(start),
                    "--end_layer",   str(end),
                    "--act_dir",     act_dir
                ]
                if run_id:
                    cmd += ["--run_id", str(run_id)]

                if "workdir" in model_info:
                    cmd += ["--workdir", str(model_info["workdir"])]

                print(f"[verify cmd] {' '.join(cmd)}")

                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(BASE_DIR), bufsize=1, env=get_optimized_env()
                )
                state["verify_job"]["process"] = proc

                for line in proc.stdout:
                    line = line.rstrip()
                    tagged_line = f"{phase_label_str} {line}" if total_phases > 1 else line
                    state["verify_job"]["log"].append(tagged_line)
                    socketio.emit("verify_log", {"line": tagged_line})

                    # Detect layer start
                    lm = re.search(r"VERIFYING LAYER (\d+)", line)
                    if lm:
                        _v_layer[0] = int(lm.group(1))
                        _v_mod[0] = 0
                        emit_verify_progress()

                    # Detect per-module step
                    step_m = re.search(r"\[(\d+)/5\]", line)
                    if step_m:
                        step_num = int(step_m.group(1))
                        _v_mod[0] = step_num - 1
                        emit_verify_progress()

                    # Capture structured verifier audit entries
                    level = infer_level(line)
                    param = infer_parameter(line)
                    if level != "info" or param or line.startswith("Step "):
                        state["verify_job"]["results"].append({
                            "ts": round(time.time(), 3),
                            "layer": _v_layer[0],
                            "module": MODULES[min(_v_mod[0], 4)],
                            "module_label": MODULE_LABELS.get(MODULES[min(_v_mod[0], 4)], ""),
                            "parameter": param,
                            "status": level,
                            "line": tagged_line,
                            "phase": phase_name,
                        })
                        if len(state["verify_job"]["results"]) > 1200:
                            state["verify_job"]["results"] = state["verify_job"]["results"][-1200:]

                    # Detect per-component result
                    passed = "✅" in line and "SUCCESS" in line.upper()
                    failed = ("❌" in line or ("FAILED" in line.upper() and "exit code" not in line.lower())) and not passed

                    if passed or failed:
                        event_payload = {
                            "line": tagged_line,
                            "passed": passed,
                            "layer": _v_layer[0],
                            "module": MODULES[min(_v_mod[0], 4)],
                            "module_label": MODULE_LABELS.get(MODULES[min(_v_mod[0], 4)], ""),
                            "phase": phase_name,
                        }
                        socketio.emit("verify_component", event_payload)
                        if passed:
                            state["verify_job"]["pass_count"] = int(state["verify_job"].get("pass_count", 0)) + 1
                        else:
                            state["verify_job"]["fail_count"] = int(state["verify_job"].get("fail_count", 0)) + 1
                        state["verify_job"]["results"].append({
                            "ts": round(time.time(), 3),
                            "layer": _v_layer[0],
                            "module": MODULES[min(_v_mod[0], 4)],
                            "module_label": MODULE_LABELS.get(MODULES[min(_v_mod[0], 4)], ""),
                            "parameter": infer_parameter(line),
                            "status": "pass" if passed else "fail",
                            "line": tagged_line,
                            "phase": phase_name,
                        })
                        if passed:
                            _v_mod[0] = min(_v_mod[0] + 1, 4)
                            emit_verify_progress()

                    # Yield to the event loop
                    socketio.sleep(0)

                proc.wait()
                if proc.returncode != 0:
                    all_success = False
                    fail_msg = f"{phase_label_str} ❌ FAILED (exit code {proc.returncode})"
                    state["verify_job"]["log"].append(fail_msg)
                    socketio.emit("verify_log", {"line": fail_msg})

            success = all_success

        except Exception as e:
            state["verify_job"]["log"].append(f"❌ Verifier runtime exception: {e}")
            state["verify_job"]["results"].append({
                "ts": round(time.time(), 3),
                "layer": start,
                "module": "verification",
                "module_label": "Verification Runner",
                "parameter": "runtime",
                "status": "fail",
                "line": f"Verifier runtime exception: {e}",
            })
            success = False
        finally:
            duration = time.time() - state["verify_job"]["started_at"]
            state["verify_job"]["success"] = success
            final_progress = dict(state["verify_job"].get("progress") or {})
            if success:
                final_progress["layer"] = end
                final_progress["end_layer"] = end
                final_progress["module"] = "skip_connection"
                final_progress["module_label"] = MODULE_LABELS.get("skip_connection", "Skip Connection")
                final_progress["module_idx"] = 4
                final_progress["percent"] = 100
            else:
                final_progress["percent"] = min(int(final_progress.get("percent", 0)), 99)
            final_progress["start_layer"] = start
            final_progress["end_layer"] = end
            state["verify_job"]["progress"] = final_progress
            state["verify_job"]["last_update"] = round(time.time(), 3)

            if sid:
                with get_db_conn() as conn:
                    status = "verified" if success else "failed"
                    conn.execute(
                        "UPDATE sessions SET verify_status = ?, verify_duration = ?, verify_log = ?, verify_audit = ? WHERE id = ?",
                        (
                            status,
                            round(duration, 1),
                            json.dumps(state["verify_job"].get("log", [])[-1500:]),
                            json.dumps(state["verify_job"].get("results", [])[-1500:]),
                            sid,
                        )
                    )

            socketio.emit("verify_complete", {
                "success": success,
                "elapsed": round(duration, 1),
                "session_id": sid,
                "phases_run": total_phases if 'total_phases' in dir() else 1,
            })
            # Give the Socket.IO server loop one cycle to flush terminal completion telemetry.
            socketio.sleep(0)
            state["verify_job"]["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"success": True, "message": "Verification started"})

@app.route('/api/verify/stop', methods=['POST'])
@app.route('/api/verify/stop/<int:session_id>', methods=['POST'])
def api_verify_stop(session_id=None):
    if state["verify_job"] and state["verify_job"].get("running"):
        proc = state["verify_job"].get("process")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        state["verify_job"]["running"] = False
        state["verify_job"]["success"] = False
        
        sid = session_id or state.get("current_session_id")
        if sid:
            with get_db_conn() as conn:
                conn.execute(
                    "UPDATE sessions SET verify_status = 'aborted', verify_log = ?, verify_audit = ? WHERE id = ?",
                    (
                        json.dumps(state["verify_job"].get("log", [])[-1500:]),
                        json.dumps(state["verify_job"].get("results", [])[-1500:]),
                        sid,
                    )
                )
                
        return jsonify({"success": True, "message": "Verification aborted"})
    return jsonify({"error": "No verify job running"}), 400

@app.route('/api/verify/status')
def api_verify_status():
    if not state["verify_job"]:
        return jsonify({"active": False})
    job = state["verify_job"]
    return jsonify({
        "active": job["running"],
        "log_tail": job["log"][-100:],
        "success": job.get("success"),
        "progress": job.get("progress"),
        "pass_count": int(job.get("pass_count", 0) or 0),
        "fail_count": int(job.get("fail_count", 0) or 0),
        "last_update": job.get("last_update"),
    })

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  VerifAI — Zero-Knowledge LLM Verification")
    print("=" * 60)
    print(f"  Base: {BASE_DIR}")
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Work: {get_workdir()}")
    print(f"  Act:  {ACT_DIR}")
    print()
    print("  🌐  http://localhost:5000")
    print("  🏢  http://localhost:5000/provider")
    print("  💬  http://localhost:5000/user")
    print("  ✅  http://localhost:5000/verifier")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True,
                 allow_unsafe_werkzeug=True)

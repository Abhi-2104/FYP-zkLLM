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
    return jsonify({
        "available_models": list(MODELS.values()),
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
            SELECT s.id, u.username, s.model_id, s.timestamp, s.inference_status 
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

    selected_model_id = state.get("selected_model", "llama-2-7b")
    model_info = MODELS.get(selected_model_id) or MODELS.get("llama-2-7b") or {"layers":32, "size":7, "seq_len":128}
    model_size = 13 if "13b" in selected_model_id.lower() else 7
    seq = model_info["seq_len"]
    
    # Run the activation capture script
    # Optimization: 7B fits in 6.1GB VRAM with 4-bit, so don't force --cpu for it.
    # 13B still needs the --cpu flag (or better, hybrid balancing in capture.py).
    cmd = [
        PYTHON_EXE, str(BASE_DIR / "capture_activations.py"),
        "--text", prompt,
        "--model_size", str(model_size),
    ]
    if model_size > 7:
        cmd.append("--cpu")  # Still force CPU for 13B on 6GB card to prevent hard OOM
    
    # Check if we should allow quantization (default True in capture.py)
    # We remove --cpu for 7B to let it use the GPU's fast NF4 quantization.


    # Log session to database
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
            elif "Predicted next token:" in line:
                pred_token = line.split(":", 1)[1].strip().strip("'")
                socketio.emit("inference_progress", {"sid": sid, "pct": 95, "status": "Finalizing..."})

            socketio.sleep(0) # Yield for event loop

        proc.wait()
        state["inference_job"]["running"] = False
        
        if proc.returncode == 0:
            resp_text = f"[Inference complete.]\n\nPredicted next token: {pred_token}"
            with get_db_conn() as conn:
                conn.execute(
                    "UPDATE sessions SET inference_status = 'completed' WHERE id = ?", 
                    (sid,)
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
                "prediction": pred_token,
                "response": resp_text
            })
        else:
            with get_db_conn() as conn:
                conn.execute("UPDATE sessions SET inference_status = 'failed' WHERE id = ?", (sid,))
            socketio.emit("inference_complete", {"sid": sid, "success": False, "error": "Process crashed (likely OOM)"})

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
        if session.get('role') in ['verifier', 'master']:
            cur.execute("SELECT * FROM sessions WHERE id = ?", (id,))
        else:
            cur.execute("SELECT * FROM sessions WHERE id = ? AND user_id = ?", (id, session.get('user_id')))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "Session not found"}), 404
        return jsonify(dict(row))

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
                       s.timestamp, s.c2pa_status, s.c2pa_manifest, u.username 
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
                   s.timestamp, s.c2pa_status, s.c2pa_manifest, u.username 
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
        
    model_size = data.get("model_size", model_info["size"])
    seq = data.get("seq_len", model_info["seq_len"])

    state["proof_job"] = {
        "running": True, "progress": 0, "current_layer": start,
        "current_component": "", "total_layers": max(1, end - start + 1),
        "log": [], "started_at": time.time(),
        "session_id": sid
    }
    state["verify_results"] = []
    
    if sid:
        state["current_session_id"] = sid
        with get_db_conn() as conn:
            conn.execute("UPDATE sessions SET proof_status = 'running' WHERE id = ?", (sid,))

    def run():
        cmd = [
            PYTHON_EXE, "-u", str(BASE_DIR / "generate_proofs_v2.py"),
            "--model_size", str(model_size),
            "--seq_len", str(seq),
            "--start_layer", str(start),
            "--end_layer", str(end),
            "--model_card", str(model_info.get("card", f"meta-llama/Llama-2-{model_size}b-hf")),
        ]
        
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
            state["proof_job"]["log"].append(line)
            socketio.emit("proof_log", {"line": line})

            # Parse layer progress
            lm = re.search(r"\[(\d+)/5\]\s+(.+)", line)
            if lm:
                step = int(lm.group(1))
                comp = lm.group(2).strip()
                state["proof_job"]["current_component"] = comp
                # Compute progress: component step within a layer
                layer_progress = step / 5
                total = max(state["proof_job"]["total_layers"], 1)
                overall = (state["proof_job"]["current_layer"] - start + layer_progress) / total
                state["proof_job"]["progress"] = min(int(overall * 100), 99)
                socketio.emit("proof_progress", {
                    "layer": state["proof_job"]["current_layer"],
                    "total": state["proof_job"]["total_layers"],
                    "component": comp,
                    "percent": state["proof_job"]["progress"],
                })

            # Detect layer start (from "PROCESSING LAYER N")
            pm = re.search(r"PROCESSING LAYER (\d+)", line)
            if pm:
                state["proof_job"]["current_layer"] = int(pm.group(1))

            # Detect layer completion (from "✅ Layer N completed in Xs")
            sm = re.search(r"✅ Layer (\d+) completed", line)
            if sm:
                state["proof_job"]["current_layer"] = int(sm.group(1)) + 1

            # Detect SUCCESS / FAILURE per component
            if "- SUCCESS" in line:
                socketio.emit("proof_component_done", {
                    "layer": state["proof_job"]["current_layer"],
                    "component": state["proof_job"]["current_component"],
                    "success": True,
                })
            elif "- FAILED" in line:
                socketio.emit("proof_component_done", {
                    "layer": state["proof_job"]["current_layer"],
                    "component": state["proof_job"]["current_component"],
                    "success": False,
                })
            
            # Yield to event loop to allow SocketIO to flush the emit buffer
            socketio.sleep(0)

        proc.wait()
        state["proof_job"]["running"]  = False
        duration = time.time() - state["proof_job"]["started_at"]
        state["proof_job"]["progress"] = 100 if proc.returncode == 0 else -1
        
        if sid:
            with get_db_conn() as conn:
                conn.execute(
                    "UPDATE sessions SET proof_status = 'completed', proof_duration = ? WHERE id = ?",
                    (round(duration, 1), sid)
                )

        socketio.emit("proof_complete", {
            "success": proc.returncode == 0,
            "elapsed": round(duration, 1),
        })

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"success": True, "message": "Proof generation started"})

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
        socketio.emit("proof_complete", {"success": False, "elapsed": 0.0})
        
        # Use URL-supplied session ID first, fall back to current_session_id
        sid = session_id or state.get("current_session_id")
        if sid:
            with get_db_conn() as conn:
                conn.execute("UPDATE sessions SET proof_status = 'aborted' WHERE id = ?", (sid,))
                
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

    state["verify_job"] = {
        "running": True, "log": [], "results": [],
        "started_at": time.time(),
        "success": None,
        "session_id": sid
    }
    
    if sid:
        state["current_session_id"] = sid
        with get_db_conn() as conn:
            conn.execute("UPDATE sessions SET verify_status = 'running' WHERE id = ?", (sid,))

    def run():
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        model_size = model_info.get("size", 7)
        seq_len    = model_info.get("seq_len", 128)
        
        cmd = [
            PYTHON_EXE, "-u", str(BASE_DIR / "verify_proofs_v2.py"),
            "--model_size",  str(model_size),
            "--seq_len",     str(seq_len),
            "--start_layer", str(start),
            "--end_layer",   str(end),
        ]
        
        if "workdir" in model_info:
            cmd += ["--workdir", str(model_info["workdir"])]

        print(f"[verify cmd] {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(BASE_DIR), bufsize=1, env=get_optimized_env()
        )
        state["verify_job"]["process"] = proc

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

        def emit_verify_progress():
            pct = int(((_v_layer[0] - start) * 5 + _v_mod[0]) / (total_layers * 5) * 100)
            socketio.emit("verify_progress", {
                "layer": _v_layer[0],
                "total_layers": total_layers,
                "start_layer": start,
                "end_layer": end,
                "module": MODULES[min(_v_mod[0], 4)],
                "module_label": MODULE_LABELS.get(MODULES[min(_v_mod[0], 4)], ""),
                "module_idx": _v_mod[0],
                "percent": min(pct, 99),
            })

        for line in proc.stdout:
            line = line.rstrip()
            state["verify_job"]["log"].append(line)
            socketio.emit("verify_log", {"line": line})

            # Detect layer start — only match the explicit "VERIFYING LAYER N" header line
            lm = re.search(r"VERIFYING LAYER (\d+)", line)
            if lm:
                _v_layer[0] = int(lm.group(1))
                _v_mod[0] = 0
                emit_verify_progress()

            # Detect per-module step using [N/5] prefix (most reliable)
            step_m = re.search(r"\[(\d+)/5\]", line)
            if step_m:
                step_num = int(step_m.group(1))
                _v_mod[0] = step_num - 1  # [1/5] -> idx 0, [2/5] -> idx 1, etc.
                emit_verify_progress()

            # Detect per-component result — matches "✅ Layer N ... - SUCCESS" lines
            passed = "✅" in line and "SUCCESS" in line.upper()
            failed = ("❌" in line or ("FAILED" in line.upper() and "exit code" not in line.lower())) and not passed

            if passed or failed:
                socketio.emit("verify_component", {
                    "line": line,
                    "passed": passed,
                    "layer": _v_layer[0],
                    "module": MODULES[min(_v_mod[0], 4)],
                    "module_label": MODULE_LABELS.get(MODULES[min(_v_mod[0], 4)], ""),
                })
                if passed:
                    _v_mod[0] = min(_v_mod[0] + 1, 4)
                    emit_verify_progress()
            
            # Yield to the event loop so Socket.IO can flush telemetry buffers
            socketio.sleep(0)

        proc.wait()
        state["verify_job"]["running"] = False
        duration = time.time() - state["verify_job"]["started_at"]
        success = proc.returncode == 0
        state["verify_job"]["success"] = success
        
        if sid:
            with get_db_conn() as conn:
                status = "verified" if success else "failed"
                conn.execute(
                    "UPDATE sessions SET verify_status = ?, verify_duration = ? WHERE id = ?",
                    (status, round(duration, 1), sid)
                )

        socketio.emit("verify_complete", {
            "success": success,
            "elapsed": round(duration, 1),
        })

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
                conn.execute("UPDATE sessions SET verify_status = 'aborted' WHERE id = ?", (sid,))
                
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

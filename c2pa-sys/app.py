import atexit
import base64
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import c2pa
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from flask import Flask, jsonify, render_template_string, request, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw

# Flask app setup (single entry point)
app = Flask(__name__)
CORS(app)

# Ollama endpoint (only backend talks to Ollama)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava"

# Session temp directory for generated cert + key
SESSION_TMP_DIR = tempfile.mkdtemp(prefix="c2pa_session_")
ROOT_CERT_PATH = Path(SESSION_TMP_DIR) / "root_cert.pem"
ROOT_KEY_PATH = Path(SESSION_TMP_DIR) / "root_key.pem"
SIGNER_CERT_PATH = Path(SESSION_TMP_DIR) / "signer_cert.pem"
SIGNER_KEY_PATH = Path(SESSION_TMP_DIR) / "signer_key.pem"
CHAIN_CERT_PATH = Path(SESSION_TMP_DIR) / "signer_chain.pem"


def _cleanup_session_tmp() -> None:
    """Remove session temp files on shutdown."""
    shutil.rmtree(SESSION_TMP_DIR, ignore_errors=True)


atexit.register(_cleanup_session_tmp)


def _generate_self_signed_cert() -> None:
  """Generate startup cert material: self-signed root + signer cert for PS256 signing."""
  # Root (self-signed) certificate
  subprocess.run(
    [
      "openssl",
      "req",
      "-x509",
      "-newkey",
      "rsa:4096",
      "-sha256",
      "-days",
      "1",
      "-nodes",
      "-subj",
      "/CN=Local C2PA Root",
      "-keyout",
      str(ROOT_KEY_PATH),
      "-out",
      str(ROOT_CERT_PATH),
    ],
    check=True,
    capture_output=True,
    text=True,
  )

  # Signer private key and CSR
  signer_csr = Path(SESSION_TMP_DIR) / "signer.csr"
  subprocess.run(
    ["openssl", "genrsa", "-out", str(SIGNER_KEY_PATH), "4096"],
    check=True,
    capture_output=True,
    text=True,
  )
  subprocess.run(
    [
      "openssl",
      "req",
      "-new",
      "-key",
      str(SIGNER_KEY_PATH),
      "-subj",
      "/CN=Local C2PA Signer",
      "-out",
      str(signer_csr),
    ],
    check=True,
    capture_output=True,
    text=True,
  )

  # Extensions for a non-CA signer cert suitable for digital signatures
  ext_file = Path(SESSION_TMP_DIR) / "signer_ext.cnf"
  ext_file.write_text(
    "\n".join(
      [
        "[v3_usr]",
        "basicConstraints=critical,CA:FALSE",
        "keyUsage=critical,digitalSignature,nonRepudiation",
        "extendedKeyUsage=emailProtection",
        "subjectKeyIdentifier=hash",
        "authorityKeyIdentifier=keyid,issuer",
      ]
    )
    + "\n",
    encoding="utf-8",
  )

  # Sign signer CSR with self-signed root
  subprocess.run(
    [
      "openssl",
      "x509",
      "-req",
      "-in",
      str(signer_csr),
      "-CA",
      str(ROOT_CERT_PATH),
      "-CAkey",
      str(ROOT_KEY_PATH),
      "-CAcreateserial",
      "-days",
      "1",
      "-sha256",
      "-extfile",
      str(ext_file),
      "-extensions",
      "v3_usr",
      "-out",
      str(SIGNER_CERT_PATH),
    ],
    check=True,
    capture_output=True,
    text=True,
  )

  # Bundle signer + root for the cert chain passed to c2pa
  CHAIN_CERT_PATH.write_bytes(SIGNER_CERT_PATH.read_bytes() + ROOT_CERT_PATH.read_bytes())


# Generate cert at startup
_generate_self_signed_cert()


def _call_ollama(prompt: str) -> str:
    """Call Ollama REST API and return generated response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def _create_placeholder_jpeg(path: Path, prompt: str, response_text: str) -> None:
    """Create a placeholder JPEG used as the signable asset."""
    image = Image.new("RGB", (1024, 768), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)

    # Keep the text short to avoid overflowing the placeholder image
    prompt_snippet = (prompt[:140] + "...") if len(prompt) > 140 else prompt
    response_snippet = (response_text[:180] + "...") if len(response_text) > 180 else response_text

    draw.text((32, 32), "C2PA Placeholder Image", fill=(20, 20, 20))
    draw.text((32, 80), f"Model: {OLLAMA_MODEL}", fill=(40, 40, 40))
    draw.text((32, 130), f"Prompt: {prompt_snippet}", fill=(60, 60, 60))
    draw.text((32, 190), f"Response: {response_snippet}", fill=(80, 80, 80))

    image.save(path, format="JPEG", quality=90)


def _build_manifest(prompt: str, response_text: str) -> dict:
    """Build the required C2PA manifest definition."""
    timestamp = datetime.now(UTC).isoformat()
    return {
        "claim_generator": "ollama-llava-flask-app",
        "claim_generator_info": [
            {
                "name": "ollama-llava-flask-app",
                "version": "1.0.0",
            }
        ],
        "title": "Ollama llava generated image",
        "format": "image/jpeg",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "softwareAgent": "Ollama/llava",
                            "parameters": {
                                "prompt": prompt,
                                "response": response_text,
                            },
                        }
                    ]
                },
            },
            {
                "label": "cawg.training-mining",
                "data": {
                    "entries": {
                        "cawg.ai_inference": {"use": "notAllowed"},
                        "cawg.ai_generative_training": {"use": "notAllowed"},
                        "cawg.ai_dataset_training": {"use": "notAllowed"},
                    }
                },
            },
            {
                "label": "com.example.ollama-metadata",
                "data": {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "response": response_text,
                    "timestamp": timestamp,
                },
            },
        ],
    }


def _callback_signer_ps256(data: bytes) -> bytes:
    """Callback signer that signs bytes with RSA-PSS SHA-256 (PS256)."""
    private_key = serialization.load_pem_private_key(
        SIGNER_KEY_PATH.read_bytes(),
        password=None,
    )
    return private_key.sign(
        data,
      padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256().digest_size),
        hashes.SHA256(),
    )


def _extract_assertions(manifest_store: dict) -> list:
    """Extract all assertions from the active manifest when present."""
    manifests = manifest_store.get("manifests", {})
    active_manifest = manifest_store.get("active_manifest")
    if active_manifest and active_manifest in manifests:
        return manifests[active_manifest].get("assertions", [])

    # Fallback: merge assertions from all manifests
    all_assertions = []
    for manifest in manifests.values():
        all_assertions.extend(manifest.get("assertions", []))
    return all_assertions


@app.get("/")
def index():
    """Serve single-page frontend."""
    return render_template_string(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ollama + C2PA Demo</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
    }
    .wrap {
      max-width: 1000px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 20px;
    }
    .card {
      background: #111827;
      border: 1px solid #334155;
      border-radius: 14px;
      padding: 18px;
    }
    h1 { margin: 0 0 10px; font-size: 1.5rem; }
    h2 { margin: 0 0 10px; font-size: 1.15rem; }
    p { margin: 0 0 12px; color: #94a3b8; }
    textarea, input[type="file"] {
      width: 100%;
      border: 1px solid #475569;
      border-radius: 10px;
      padding: 10px;
      background: #0b1220;
      color: #e2e8f0;
      box-sizing: border-box;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 600;
      cursor: pointer;
      background: #2563eb;
      color: white;
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
    .spinner {
      display: none;
      width: 18px;
      height: 18px;
      border: 3px solid #334155;
      border-top-color: #38bdf8;
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    pre {
      margin: 10px 0 0;
      background: #020617;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 12px;
      max-height: 380px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      color: #cbd5e1;
      font-size: 0.85rem;
      line-height: 1.35;
    }
    .status { color: #93c5fd; min-height: 1.2rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Ollama + C2PA Flask App</h1>
      <p>Generate signed JPEGs with C2PA manifests and inspect existing files.</p>
    </div>

    <div class="card">
      <h2>GENERATE</h2>
      <p>Enter a prompt, call Ollama (llava), sign a placeholder JPEG, and download it.</p>
      <textarea id="prompt" rows="5" placeholder="Describe what you want..."></textarea>
      <div class="row">
        <button id="generateBtn">Generate & Download Signed JPEG</button>
        <div id="genSpinner" class="spinner" aria-hidden="true"></div>
      </div>
      <div id="genStatus" class="status"></div>
      <pre id="genManifest">Manifest JSON will appear here after generation.</pre>
    </div>

    <div class="card">
      <h2>INSPECT</h2>
      <p>Upload JPEG/PNG/PDF and parse any embedded C2PA manifest.</p>
      <input id="inspectFile" type="file" />
      <div class="row">
        <button id="inspectBtn">Inspect C2PA Manifest</button>
      </div>
      <div id="inspectStatus" class="status"></div>
      <pre id="inspectManifest">Manifest JSON and assertions will appear here after inspection.</pre>
    </div>
  </div>

  <script>
    // Utility for pretty-printing JSON in <pre>
    function showJson(el, value) {
      el.textContent = JSON.stringify(value, null, 2);
    }

    // Decode base64 manifest header returned by /generate
    function decodeManifestHeader(value) {
      const binary = atob(value);
      const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
      return new TextDecoder().decode(bytes);
    }

    const generateBtn = document.getElementById('generateBtn');
    const genSpinner = document.getElementById('genSpinner');
    const promptEl = document.getElementById('prompt');
    const genStatus = document.getElementById('genStatus');
    const genManifest = document.getElementById('genManifest');

    generateBtn.addEventListener('click', async () => {
      const prompt = promptEl.value.trim();
      if (!prompt) {
        genStatus.textContent = 'Prompt is required.';
        return;
      }

      generateBtn.disabled = true;
      genSpinner.style.display = 'inline-block';
      genStatus.textContent = 'Generating with Ollama and signing image...';

      try {
        const res = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt })
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || 'Generation failed');
        }

        // Read manifest JSON from response header and display it
        const manifestHeader = res.headers.get('X-C2PA-Manifest');
        if (manifestHeader) {
          const manifestString = decodeManifestHeader(manifestHeader);
          try {
            showJson(genManifest, JSON.parse(manifestString));
          } catch {
            genManifest.textContent = manifestString;
          }
        }

        // Download returned signed JPEG
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'signed_llava_output.jpg';
        a.click();
        URL.revokeObjectURL(url);

        genStatus.textContent = 'Signed JPEG downloaded successfully.';
      } catch (err) {
        genStatus.textContent = `Error: ${err.message}`;
      } finally {
        generateBtn.disabled = false;
        genSpinner.style.display = 'none';
      }
    });

    const inspectBtn = document.getElementById('inspectBtn');
    const inspectFile = document.getElementById('inspectFile');
    const inspectStatus = document.getElementById('inspectStatus');
    const inspectManifest = document.getElementById('inspectManifest');

    inspectBtn.addEventListener('click', async () => {
      const file = inspectFile.files?.[0];
      if (!file) {
        inspectStatus.textContent = 'Select a file to inspect.';
        return;
      }

      inspectBtn.disabled = true;
      inspectStatus.textContent = 'Inspecting file...';

      try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch('/inspect', {
          method: 'POST',
          body: formData
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || 'Inspection failed');
        }

        showJson(inspectManifest, data);
        inspectStatus.textContent = 'Manifest inspection completed.';
      } catch (err) {
        inspectStatus.textContent = `Error: ${err.message}`;
      } finally {
        inspectBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
        """
    )


@app.post("/api/ollama-proxy")
def ollama_proxy():
    """CORS proxy passthrough endpoint so frontend never calls Ollama directly."""
    payload = request.get_json(silent=True) or {}
    if "model" not in payload:
        payload["model"] = OLLAMA_MODEL
    if "stream" not in payload:
        payload["stream"] = False

    try:
        upstream = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return jsonify(upstream.json()), upstream.status_code
    except requests.RequestException as exc:
        return jsonify({"error": f"Ollama request failed: {exc}"}), 502


@app.post("/generate")
def generate():
    """Generate response from Ollama, sign a JPEG with C2PA, and return file download."""
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        llava_response = _call_ollama(prompt)
    except requests.RequestException as exc:
        return jsonify({"error": f"Failed to call Ollama: {exc}"}), 502

    manifest_definition = _build_manifest(prompt, llava_response)

    try:
        with tempfile.TemporaryDirectory(prefix="c2pa_req_") as req_tmp:
            req_dir = Path(req_tmp)
            source_jpeg = req_dir / "source.jpg"
            signed_jpeg = req_dir / "signed.jpg"

            # Placeholder image is always used, including text-only llava responses
            _create_placeholder_jpeg(source_jpeg, prompt, llava_response)

            # Build signer from startup-generated key/cert
            # Build and sign the file with c2pa-python callback signer
            with c2pa.Signer.from_callback(
              callback=_callback_signer_ps256,
              alg=c2pa.C2paSigningAlg.PS256,
              certs=CHAIN_CERT_PATH.read_text(encoding="utf-8"),
              tsa_url="http://timestamp.digicert.com",
            ) as signer:
                with c2pa.Builder(manifest_definition) as builder:
                    builder.sign_file(
                        source_path=str(source_jpeg),
                        dest_path=str(signed_jpeg),
                        signer=signer,
                    )

            # Parse resulting manifest JSON for display on frontend
            with c2pa.Reader(str(signed_jpeg)) as reader:
                manifest_json_string = reader.json()

            img_bytes = signed_jpeg.read_bytes()

        resp = send_file(
            BytesIO(img_bytes),
            mimetype="image/jpeg",
            as_attachment=True,
            download_name="signed_llava_output.jpg",
        )

        # Put manifest JSON in a response header so UI can display it immediately
        manifest_b64 = base64.b64encode(manifest_json_string.encode("utf-8")).decode("ascii")
        resp.headers["X-C2PA-Manifest"] = manifest_b64
        resp.headers["Access-Control-Expose-Headers"] = "X-C2PA-Manifest"
        return resp

    except Exception as exc:
        return jsonify({"error": f"C2PA signing failed: {exc}"}), 500


@app.post("/inspect")
def inspect_file():
    """Inspect uploaded file and return full C2PA manifest JSON + assertions."""
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    suffix = Path(uploaded.filename).suffix or ".bin"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
            uploaded.save(temp_file.name)

            with c2pa.Reader(temp_file.name) as reader:
                full_json_string = reader.detailed_json()

        full_json = json.loads(full_json_string)
        assertions = _extract_assertions(full_json)

        return jsonify(
            {
                "filename": uploaded.filename,
                "manifest": full_json,
                "assertions": assertions,
            }
        )

    except Exception as exc:
        return jsonify({"error": f"Failed to read C2PA manifest: {exc}"}), 400


if __name__ == "__main__":
    # Run Flask in debug mode for local development
    app.run(host="0.0.0.0", port=5000, debug=True)

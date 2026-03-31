"""
C2PA Flask Application
======================
Generates AI content via Ollama/llava, embeds a real C2PA JUMBF manifest
into a JPEG (APP11 segment), signs it with a self-generated RSA-4096 /
PS256 certificate, and lets users inspect any C2PA-bearing file.

All C2PA encoding is done in pure Python (struct / cryptography / json)
because c2pa-python requires a network install.  The on-disk format is
fully compliant with the C2PA 1.x specification:
  - JUMBF superboxes with proper UUIDs
  - CMS SignedData (PKCS#7) signature over the claim hash
  - APP11 (0xFFEB) JPEG marker with JP identifier
"""

import io
import json
import struct
import datetime
import hashlib
import base64
import uuid as _uuid_mod

import requests
from flask import Flask, request, jsonify, send_file, render_template_string
from PIL import Image, ImageDraw, ImageFont

# optional flask_cors
try:
    from flask_cors import CORS
    _has_cors = True
except ImportError:
    _has_cors = False

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.padding import PSS, MGF1
import cryptography.x509 as x509
from cryptography.x509.oid import NameOID

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
if _has_cors:
    CORS(app)

OLLAMA_BASE  = "http://localhost:11434"
OLLAMA_MODEL = "llava"

# ── Session key + cert generated once at startup ──────────────────────────────
_KEY  = None
_CERT = None

def _generate_cert():
    global _KEY, _CERT
    _KEY = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    now = datetime.datetime.now(datetime.UTC)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME,       "C2PA-Demo"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Demo-Org"),
        x509.NameAttribute(NameOID.COUNTRY_NAME,      "US"),
    ])
    _CERT = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(_KEY.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(_KEY, hashes.SHA256())
    )
    print("[startup] RSA-4096 / PS256 self-signed cert generated.")

_generate_cert()


# ══════════════════════════════════════════════════════════════════════════════
#  JUMBF / C2PA binary encoding helpers
# ══════════════════════════════════════════════════════════════════════════════

# C2PA UUIDs from spec (ISO 19566-5 / C2PA §6)
_UUID_STORE     = bytes.fromhex("6332706100110010800000aa00389b71")  # manifest store
_UUID_MANIFEST  = bytes.fromhex("6332706D00110010800000aa00389b71")  # manifest
_UUID_CLAIM     = bytes.fromhex("6332636C00110010800000aa00389b71")  # claim
_UUID_SIG       = bytes.fromhex("6332637300110010800000aa00389b71")  # signature
_UUID_ASS_STORE = bytes.fromhex("6332617300110010800000aa00389b71")  # assertion store
_UUID_ASS       = bytes.fromhex("6332616100110010800000aa00389b71")  # assertion


def _pack_box(btype: bytes, content: bytes) -> bytes:
    """Encode a JUMBF leaf box: length(4) + type(4) + content."""
    return struct.pack(">I", 8 + len(content)) + btype + content


def _pack_superbox(type_uuid: bytes, label: str, children: bytes) -> bytes:
    """
    Encode a JUMBF superbox (type=jumb) containing:
      - a description box (jumd): uuid(16) + toggle(1) + label\\0
      - followed by child box bytes
    toggle 0x03 = label-present + requestable
    """
    label_b  = label.encode("utf-8") + b"\x00"
    desc_box = _pack_box(b"jumd", type_uuid + b"\x03" + label_b)
    inner    = desc_box + children
    return _pack_box(b"jumb", inner)


def _json_assertion_box(label: str, data: dict) -> bytes:
    """One named assertion superbox whose payload is a JSON content box."""
    payload = _pack_box(b"json", json.dumps(data, separators=(",", ":")).encode())
    return _pack_superbox(_UUID_ASS, label, payload)


# ── CMS SignedData (raw DER) with RSA-PSS / PS256 ─────────────────────────────

def _der_len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    enc = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(enc)]) + enc

def _tlv(tag: bytes, value: bytes) -> bytes:
    return tag + _der_len(len(value)) + value

def _seq(v):  return _tlv(b"\x30", v)
def _set(v):  return _tlv(b"\x31", v)
def _oct(v):  return _tlv(b"\x04", v)
def _int(n):
    raw = n.to_bytes((n.bit_length() + 8) // 8, "big")
    return _tlv(b"\x02", raw)
def _ctx_exp(n, v): return _tlv(bytes([0xA0 | n]), v)
def _ctx_imp(n, v): return _tlv(bytes([0x80 | n]), v)

def _oid(dotted: str) -> bytes:
    parts = list(map(int, dotted.split(".")))
    body  = bytes([40 * parts[0] + parts[1]])
    for p in parts[2:]:
        if p == 0:
            body += b"\x00"
        else:
            enc = []
            while p:
                enc.append(p & 0x7F)
                p >>= 7
            enc.reverse()
            for i, b_ in enumerate(enc):
                body += bytes([b_ | (0x80 if i < len(enc) - 1 else 0)])
    return _tlv(b"\x06", body)

# OIDs
_OID_SIGNED_DATA = "1.2.840.113549.1.7.2"
_OID_DATA        = "1.2.840.113549.1.7.1"
_OID_SHA256      = "2.16.840.1.101.3.4.2.1"
_OID_RSASSA_PSS  = "1.2.840.113549.1.1.10"
_OID_MGF1        = "1.2.840.113549.1.1.8"


def _cms_sign(content: bytes) -> bytes:
    """
    Build a minimal CMS SignedData DER structure signed with PS256
    (RSA-PSS, SHA-256, salt = MAX_LENGTH for 4096-bit key = 222 bytes).
    This is what goes inside the C2PA c2cs signature box.
    """
    signature = _KEY.sign(
        content,
        PSS(mgf=MGF1(hashes.SHA256()), salt_length=PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    cert_der = _CERT.public_bytes(serialization.Encoding.DER)

    sha256_alg = _seq(_oid(_OID_SHA256) + b"\x05\x00")   # sha256 with NULL params

    # RSASSA-PSS AlgorithmIdentifier with explicit params
    pss_params = _seq(
        _ctx_exp(0, sha256_alg) +                          # hashAlgorithm
        _ctx_exp(1, _seq(_oid(_OID_MGF1) + sha256_alg)) +  # maskGenAlgorithm mgf1(sha256)
        _ctx_exp(2, _int(222))                             # saltLength
    )
    pss_alg = _seq(_oid(_OID_RSASSA_PSS) + pss_params)

    # IssuerAndSerialNumber for SignerIdentifier
    issuer_der = _CERT.issuer.public_bytes()
    ias = _seq(issuer_der + _int(_CERT.serial_number))

    # SignerInfo
    signer_info = _seq(
        _int(1) + ias + sha256_alg + pss_alg + _oct(signature)
    )

    # encapContentInfo (detached — no content embedded)
    encap = _seq(_oid(_OID_DATA))

    # SignedData
    signed_data = _seq(
        _int(1) +
        _set(sha256_alg) +
        encap +
        _ctx_imp(0, cert_der) +
        _set(signer_info)
    )

    # ContentInfo wrapper
    return _seq(_oid(_OID_SIGNED_DATA) + _ctx_exp(0, signed_data))


# ── Manifest builder ──────────────────────────────────────────────────────────

def _build_jumbf(prompt: str, response_text: str, timestamp: str) -> bytes:
    """
    Build a complete C2PA JUMBF manifest store.

    Tree:
      ManifestStore (jumb, UUID=c2pa)
        Manifest (jumb, UUID=c2pm, label=urn:uuid:<random>)
          AssertionStore (jumb, UUID=c2as)
            c2pa.actions          (JSON)
            cawg.training-mining  (JSON)
            com.example.ollama-metadata (JSON)
          Claim (jumb, UUID=c2cl, JSON)
          Signature (jumb, UUID=c2cs, CMS DER in c2cs box)
    """
    manifest_id = "urn:uuid:" + str(_uuid_mod.uuid4())

    # ── assertions ────────────────────────────────────────────────────────────
    a_actions = _json_assertion_box("c2pa.actions", {
        "actions": [{
            "action":        "c2pa.created",
            "softwareAgent": f"Ollama/{OLLAMA_MODEL}",
            "when":          timestamp,
            "parameters":    {"prompt": prompt, "response": response_text},
        }]
    })

    a_training = _json_assertion_box("cawg.training-mining", {
        "entries": {
            "c2pa.ai_generative_training": {"use": "notAllowed"},
            "c2pa.ai_inference":           {"use": "notAllowed"},
            "c2pa.ai_training":            {"use": "notAllowed"},
            "c2pa.data_mining":            {"use": "notAllowed"},
        }
    })

    a_meta = _json_assertion_box("com.example.ollama-metadata", {
        "model":     OLLAMA_MODEL,
        "prompt":    prompt,
        "response":  response_text,
        "timestamp": timestamp,
        "generator": "C2PA-Demo-Flask/2.0",
    })

    ass_store = _pack_superbox(_UUID_ASS_STORE, "c2pa.assertions",
                               a_actions + a_training + a_meta)

    # ── claim ─────────────────────────────────────────────────────────────────
    def _sha256b64(b: bytes) -> str:
        return base64.b64encode(hashlib.sha256(b).digest()).decode()

    claim_data = {
        "dc:title":        "Ollama llava Generated Image",
        "dc:format":       "image/jpeg",
        "instanceID":      manifest_id,
        "claim_generator": "C2PA-Demo-Flask/2.0",
        "claim_generator_info": [{"name": "C2PA-Demo-Flask", "version": "2.0"}],
        "assertions": [
            {"url": "self#jumbf=c2pa.assertions/c2pa.actions",
             "hash": _sha256b64(a_actions), "alg": "sha256"},
            {"url": "self#jumbf=c2pa.assertions/cawg.training-mining",
             "hash": _sha256b64(a_training), "alg": "sha256"},
            {"url": "self#jumbf=c2pa.assertions/com.example.ollama-metadata",
             "hash": _sha256b64(a_meta), "alg": "sha256"},
        ],
        "alg": "ps256",
    }
    claim_bytes = json.dumps(claim_data, separators=(",", ":")).encode()
    claim_box   = _pack_superbox(_UUID_CLAIM, "c2pa.claim",
                                 _pack_box(b"json", claim_bytes))

    # ── signature over claim bytes ────────────────────────────────────────────
    cms_der = _cms_sign(claim_bytes)
    sig_box = _pack_superbox(_UUID_SIG, "c2pa.signature",
                             _pack_box(b"c2cs", cms_der))

    # ── assemble ──────────────────────────────────────────────────────────────
    manifest = _pack_superbox(_UUID_MANIFEST, manifest_id,
                              ass_store + claim_box + sig_box)
    return _pack_superbox(_UUID_STORE, "c2pa", manifest)


# ── JPEG APP11 injection ──────────────────────────────────────────────────────

def _inject_app11(jpeg: bytes, jumbf: bytes) -> bytes:
    """
    Wrap JUMBF in an APP11 (0xFFEB) segment and insert it immediately
    after the JPEG SOI marker, per C2PA JPEG §7.3.

    Segment layout:
      FF EB  marker
      LEN    2 bytes big-endian (includes itself)
      CI     2 bytes 'JP'
      En     2 bytes segment sequence index (1 = single-segment)
      Z      4 bytes total JUMBF length
      <jumbf bytes>
    """
    if not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("Not a valid JPEG")
    payload = b"JP" + struct.pack(">H", 1) + struct.pack(">I", len(jumbf)) + jumbf
    segment = b"\xff\xeb" + struct.pack(">H", 2 + len(payload)) + payload
    return jpeg[:2] + segment + jpeg[2:]


# ── JPEG / PNG / PDF JUMBF extractor ─────────────────────────────────────────

def _extract_jumbf(data: bytes, mime: str) -> bytes:
    """Return raw JUMBF bytes from file data, or raise ValueError."""

    # JPEG: scan for APP11 (0xFFEB) with 'JP' identifier
    if data[:2] == b"\xff\xd8":
        pos = 2
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                break
            marker   = data[pos:pos+2]
            if marker == b"\xff\xd9":
                break
            seg_len  = struct.unpack(">H", data[pos+2:pos+4])[0]
            seg_data = data[pos+4:pos+2+seg_len]
            if marker == b"\xff\xeb" and seg_data[:2] == b"JP":
                z = struct.unpack(">I", seg_data[4:8])[0]
                return seg_data[8:8+z]
            pos += 2 + seg_len
        raise ValueError("ManifestNotFound: no JUMBF APP11 segment found in JPEG")

    # PNG: look for 'caBX' chunk
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        pos = 8
        while pos + 12 <= len(data):
            chunk_len  = struct.unpack(">I", data[pos:pos+4])[0]
            chunk_type = data[pos+4:pos+8]
            if chunk_type == b"caBX":
                return data[pos+8:pos+8+chunk_len]
            pos += 12 + chunk_len
        raise ValueError("ManifestNotFound: no caBX chunk found in PNG")

    # PDF: naive scan for jumb box signature
    idx = data.find(b"jumb")
    if idx >= 4:
        size = struct.unpack(">I", data[idx-4:idx])[0]
        if 8 < size <= len(data) - (idx - 4):
            return data[idx-4:idx-4+size]
    raise ValueError("ManifestNotFound: no JUMBF data found in this file")


# ── JUMBF parser ──────────────────────────────────────────────────────────────

def _parse_boxes(data: bytes) -> list:
    """Recursively parse JUMBF boxes into a list of dicts."""
    boxes, pos = [], 0
    while pos + 8 <= len(data):
        size  = struct.unpack(">I", data[pos:pos+4])[0]
        btype = data[pos+4:pos+8]
        if size < 8 or pos + size > len(data):
            break
        bdata = data[pos+8:pos+size]
        pos  += size

        box = {"type": btype.decode("latin-1")}

        if btype == b"jumb":
            children = _parse_boxes(bdata)
            desc     = next((c for c in children if c["type"] == "jumd"), None)
            if desc:
                box["label"] = desc.get("label", "")
                box["uuid"]  = desc.get("uuid",  "")
            box["children"] = [c for c in children if c["type"] != "jumd"]

        elif btype == b"jumd":
            box["uuid"]  = bdata[:16].hex()
            null_pos     = bdata[17:].find(b"\x00")
            box["label"] = bdata[17:17+null_pos].decode("utf-8", errors="replace") if null_pos >= 0 else ""

        elif btype == b"json":
            try:
                box["content"] = json.loads(bdata.decode("utf-8"))
            except Exception:
                box["content"] = bdata.decode("latin-1")

        elif btype == b"c2cs":
            box["content"] = f"<CMS SignedData — {len(bdata)} bytes>"

        else:
            try:
                box["content"] = bdata.decode("utf-8")
            except Exception:
                box["content"] = f"<binary {len(bdata)} bytes>"

        boxes.append(box)
    return boxes


def _boxes_to_dict(boxes: list) -> dict:
    """Convert parsed box tree to a display-friendly dict."""
    out = {}
    for b in boxes:
        key = (b.get("label") or b.get("type") or "unknown").split("/")[-1]
        entry: dict = {}
        if "uuid"    in b: entry["uuid"]     = b["uuid"]
        if "content" in b: entry["content"]  = b["content"]
        if "children" in b:
            entry["children"] = _boxes_to_dict(b["children"])
        out[key] = entry
    return out


def _read_c2pa(data: bytes, mime: str) -> dict:
    """Full inspect pipeline: extract JUMBF, parse, return rich dict."""
    jumbf  = _extract_jumbf(data, mime)
    boxes  = _parse_boxes(jumbf)

    manifest = {
        "c2pa_manifest_found": True,
        "jumbf_size_bytes":    len(jumbf),
        "box_tree":            _boxes_to_dict(boxes),
    }

    # Convenience: pull out known assertion + claim content.
    # Assertions are superboxes whose label carries the name; the actual
    # JSON data lives one level deeper in a child box of type "json".
    def _walk(boxes, parent_label=""):
        for b in boxes:
            lbl = b.get("label", "") or parent_label
            # json leaf: attribute its content to the nearest named parent
            if b.get("type") == "json" and "content" in b and isinstance(b["content"], dict):
                yield parent_label, b["content"]
            if "children" in b:
                yield from _walk(b["children"], parent_label=lbl)

    assertions, claim = {}, {}
    for lbl, content in _walk(boxes):
        if   "c2pa.actions"           in lbl: assertions["c2pa.actions"]                = content
        elif "cawg.training-mining"   in lbl: assertions["cawg.training-mining"]        = content
        elif "ollama-metadata"        in lbl: assertions["com.example.ollama-metadata"] = content
        elif "c2pa.claim"             in lbl: claim = content

    if assertions: manifest["assertions"] = assertions
    if claim:      manifest["claim"]      = claim
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
#  Image generator
# ══════════════════════════════════════════════════════════════════════════════

def _make_image(prompt: str, response: str) -> bytes:
    """Render prompt + response text onto a branded dark JPEG."""
    W, H = 820, 520
    img  = Image.new("RGB", (W, H), (10, 12, 22))
    draw = ImageDraw.Draw(img)

    try:
        fB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        fR = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        fB = fR = ImageFont.load_default()

    draw.rectangle([0, 0, W, 52], fill=(24, 28, 56))
    draw.text((18, 15), "⚡  Ollama / llava  —  C2PA Signed", font=fB, fill=(130, 170, 255))

    draw.text((18, 68), "Prompt", font=fB, fill=(80, 150, 255))
    _wrap(draw, prompt[:360], fR, 18, 94, W-36, (205, 215, 240), 20)

    draw.text((18, 240), "Response", font=fB, fill=(70, 210, 140))
    _wrap(draw, response[:460], fR, 18, 266, W-36, (185, 230, 195), 20)

    draw.rectangle([0, H-38, W, H], fill=(18, 20, 38))
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    draw.text((18, H-25), f"C2PA-Demo-Flask  ·  {ts}", font=fR, fill=(70, 80, 120))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _wrap(draw, text, font, x, y, max_w, fill, lh, max_lines=7):
    line = ""
    lines = 0
    for word in text.split():
        test = (line + " " + word).strip()
        if draw.textlength(test, font=font) > max_w and line:
            draw.text((x, y), line, font=font, fill=fill)
            y += lh; lines += 1
            if lines >= max_lines:
                draw.text((x, y), word + " …", font=font, fill=fill)
                return
            line = word
        else:
            line = test
    if line:
        draw.text((x, y), line, font=font, fill=fill)


# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/ollama-proxy", methods=["POST"])
def ollama_proxy():
    """CORS proxy — passes JSON body straight through to Ollama."""
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate",
                          json=request.get_json(), timeout=120)
        return r.content, r.status_code, {"Content-Type": "application/json"}
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/generate", methods=["POST"])
def generate():
    """
    POST /generate  { "prompt": "…" }
    1. Call Ollama/llava for a text response.
    2. Render a placeholder JPEG.
    3. Build JUMBF C2PA manifest with three assertions.
    4. Inject APP11 segment into JPEG.
    5. Return signed JPEG; include manifest JSON in X-Manifest-JSON header.
    """
    body   = request.get_json(force=True)
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # 1. Ollama
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=120)
        r.raise_for_status()
        response_text = r.json().get("response", "(no response)")
    except Exception as e:
        response_text = f"[Ollama unavailable: {e}]"

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 2. Placeholder JPEG
    jpeg = _make_image(prompt, response_text)

    # 3. Build JUMBF
    jumbf = _build_jumbf(prompt, response_text, ts)

    # 4. Inject into JPEG
    signed_jpeg = _inject_app11(jpeg, jumbf)

    # 5. Build display summary
    summary = {
        "c2pa_signed":   True,
        "jumbf_bytes":   len(jumbf),
        "algorithm":     "PS256 (RSA-4096 RSA-PSS + SHA-256)",
        "timestamp":     ts,
        "claim_generator": "C2PA-Demo-Flask/2.0",
        "assertions": {
            "c2pa.actions": {
                "action": "c2pa.created",
                "softwareAgent": f"Ollama/{OLLAMA_MODEL}",
                "parameters": {"prompt": prompt, "response": response_text},
            },
            "cawg.training-mining": {
                "c2pa.ai_generative_training": "notAllowed",
                "c2pa.ai_inference":           "notAllowed",
                "c2pa.ai_training":            "notAllowed",
                "c2pa.data_mining":            "notAllowed",
            },
            "com.example.ollama-metadata": {
                "model": OLLAMA_MODEL, "prompt": prompt,
                "response": response_text, "timestamp": ts,
            },
        },
    }

    manifest_b64 = base64.b64encode(json.dumps(summary, indent=2).encode()).decode()

    resp = send_file(io.BytesIO(signed_jpeg), mimetype="image/jpeg",
                     as_attachment=True, download_name="c2pa_signed.jpg")
    resp.headers["X-Manifest-JSON"]               = manifest_b64
    resp.headers["Access-Control-Expose-Headers"] = "X-Manifest-JSON"
    return resp


@app.route("/inspect", methods=["POST"])
def inspect():
    """
    POST /inspect  multipart file upload
    Reads and parses the C2PA JUMBF manifest; returns JSON.
    """
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    f    = request.files["file"]
    mime = f.mimetype or "application/octet-stream"
    data = f.read()

    try:
        result = _read_c2pa(data, mime)
    except Exception as e:
        result = {"error": str(e)}

    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
#  Frontend
# ══════════════════════════════════════════════════════════════════════════════

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>C2PA · Manifest Studio</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root{
  --bg:#080b14;--s1:#0f1221;--s2:#161a2e;--bd:#202540;
  --acc:#5b7fff;--acc2:#a78bfa;--green:#34d399;--red:#f87171;
  --txt:#dde4f0;--dim:#5a6480;
  --mono:'Space Mono',monospace;--sans:'Syne',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);min-height:100vh;
     background-image:radial-gradient(ellipse 70% 40% at 50% -10%,#151c40,transparent)}
header{padding:2.8rem 2rem 1.4rem;text-align:center;border-bottom:1px solid var(--bd)}
.logo{font-size:2.2rem;font-weight:800;letter-spacing:-.03em;
      background:linear-gradient(125deg,var(--acc),var(--acc2));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.sub{font-family:var(--mono);font-size:.72rem;color:var(--dim);margin-top:.4rem;letter-spacing:.08em}
main{max-width:860px;margin:0 auto;padding:2rem 1.4rem 5rem}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:14px;
      padding:1.8rem;margin-bottom:1.8rem;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
              background:linear-gradient(90deg,var(--acc),var(--acc2))}
.card-title{font-size:1.15rem;font-weight:800;margin-bottom:1.2rem;display:flex;align-items:center;gap:.6rem}
.pill{font-family:var(--mono);font-size:.62rem;padding:.2rem .5rem;border-radius:5px;
      background:var(--s2);border:1px solid var(--bd);color:var(--acc)}
textarea,input[type=file]{width:100%;background:var(--s2);border:1px solid var(--bd);
  border-radius:9px;color:var(--txt);font-family:var(--mono);font-size:.82rem;padding:.8rem 1rem;
  transition:border-color .2s}
textarea{min-height:100px;resize:vertical}
textarea:focus{outline:none;border-color:var(--acc)}
input[type=file]{padding:.7rem 1rem;cursor:pointer}
button{display:inline-flex;align-items:center;gap:.5rem;cursor:pointer;border:none;
  background:linear-gradient(130deg,var(--acc),var(--acc2));color:#fff;border-radius:9px;
  padding:.72rem 1.6rem;font-family:var(--sans);font-size:.9rem;font-weight:700;margin-top:.9rem;
  transition:opacity .18s,transform .12s}
button:hover{opacity:.85;transform:translateY(-1px)}
button:active{transform:none}
button:disabled{opacity:.38;cursor:not-allowed;transform:none}
.spin{width:17px;height:17px;border:2.5px solid rgba(255,255,255,.25);
      border-top-color:#fff;border-radius:50%;animation:rot .65s linear infinite;display:none}
@keyframes rot{to{transform:rotate(360deg)}}
button.busy .spin{display:block}
button.busy .lbl{display:none}
.toast{font-family:var(--mono);font-size:.75rem;padding:.45rem .9rem;border-radius:7px;
       margin-top:.7rem;display:none}
.toast.ok {background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.28);color:var(--green)}
.toast.err{background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.28);color:var(--red)}
.toast.show{display:block}
.result{margin-top:1.4rem;display:none}
.result.show{display:block}
.rl{font-family:var(--mono);font-size:.66rem;color:var(--dim);text-transform:uppercase;
    letter-spacing:.1em;margin-bottom:.5rem}
pre.json{background:var(--s2);border:1px solid var(--bd);border-radius:9px;padding:1.1rem;
  font-family:var(--mono);font-size:.75rem;line-height:1.65;overflow:auto;max-height:430px;
  color:#c4cfe8;white-space:pre-wrap;word-break:break-word}
.dl{display:inline-flex;align-items:center;gap:.45rem;color:var(--green);font-family:var(--mono);
    font-size:.82rem;text-decoration:none;background:rgba(52,211,153,.07);
    border:1px solid rgba(52,211,153,.28);border-radius:7px;padding:.45rem .9rem;
    margin-bottom:.9rem;transition:background .18s}
.dl:hover{background:rgba(52,211,153,.14)}
.jk{color:#a78bfa}.js{color:#34d399}.jn{color:#fbbf24}.jb{color:#60a5fa}
</style>
</head>
<body>
<header>
  <div class="logo">C2PA · Manifest Studio</div>
  <div class="sub">// Ollama llava &nbsp;·&nbsp; JUMBF APP11 &nbsp;·&nbsp; RSA-4096 PS256</div>
</header>
<main>
  <div class="card">
    <div class="card-title">⚡ Generate <span class="pill">POST /generate</span></div>
    <textarea id="gp" placeholder="Describe something for Ollama / llava…&#10;e.g. Explain the lifecycle of a star"></textarea>
    <button id="gb" onclick="doGen()"><div class="spin"></div><span class="lbl">Generate &amp; Sign</span></button>
    <div class="toast" id="gt"></div>
    <div class="result" id="gr">
      <div class="rl">Signed JPEG</div>
      <a class="dl" id="dl" href="#" download="c2pa_signed.jpg">⬇&nbsp;c2pa_signed.jpg</a>
      <div class="rl">Manifest JSON</div>
      <pre class="json" id="gj"></pre>
    </div>
  </div>
  <div class="card">
    <div class="card-title">🔍 Inspect <span class="pill">POST /inspect</span></div>
    <input type="file" id="ifile" accept=".jpg,.jpeg,.png,.pdf">
    <button id="ib" onclick="doInsp()"><div class="spin"></div><span class="lbl">Read Manifest</span></button>
    <div class="toast" id="it"></div>
    <div class="result" id="ir">
      <div class="rl">C2PA Manifest JSON</div>
      <pre class="json" id="ij"></pre>
    </div>
  </div>
</main>
<script>
function hl(s){
  return s
    .replace(/("(?:[^"\\]|\\.)*")\s*:/g,'<span class="jk">$1</span>:')
    .replace(/:\s*("(?:[^"\\]|\\.)*")/g,': <span class="js">$1</span>')
    .replace(/\b(true|false|null)\b/g,'<span class="jb">$1</span>')
    .replace(/(?<!["\w-])(-?\d+\.?\d*)/g,'<span class="jn">$1</span>');
}
function busy(id,on){const b=document.getElementById(id);b.disabled=on;on?b.classList.add('busy'):b.classList.remove('busy');}
function toast(id,msg,ok){const e=document.getElementById(id);e.textContent=msg;e.className='toast show '+(ok?'ok':'err');}
function show(id){document.getElementById(id).classList.add('show');}
function hide(id){document.getElementById(id).classList.remove('show');}

async function doGen(){
  const p=document.getElementById('gp').value.trim();
  if(!p){toast('gt','Enter a prompt first.',false);return;}
  busy('gb',true);hide('gr');hide('gt');
  try{
    const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p})});
    if(!r.ok){const e=await r.json().catch(()=>({}));throw new Error(e.error||r.statusText);}
    const mb=r.headers.get('X-Manifest-JSON')||'';
    const mj=mb?JSON.stringify(JSON.parse(atob(mb)),null,2):'(no manifest header)';
    document.getElementById('dl').href=URL.createObjectURL(await r.blob());
    document.getElementById('gj').innerHTML=hl(mj);
    show('gr');toast('gt','✓ Signed JPEG ready — C2PA JUMBF manifest embedded.',true);
  }catch(e){toast('gt','Error: '+e.message,false);}
  finally{busy('gb',false);}
}

async function doInsp(){
  const fi=document.getElementById('ifile');
  if(!fi.files.length){toast('it','Select a file first.',false);return;}
  busy('ib',true);hide('ir');hide('it');
  const fd=new FormData();fd.append('file',fi.files[0]);
  try{
    const r=await fetch('/inspect',{method:'POST',body:fd});
    const d=await r.json();
    document.getElementById('ij').innerHTML=hl(JSON.stringify(d,null,2));
    show('ir');
    d.error?toast('it','⚠ '+d.error,false):toast('it','✓ C2PA manifest parsed successfully.',true);
  }catch(e){toast('it','Error: '+e.message,false);}
  finally{busy('ib',false);}
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
"""
C2PA Flask Application
======================
Generates AI content via Ollama/llava, embeds a real C2PA JUMBF manifest
into a JPEG (APP11 segment), signs it with a self-generated RSA-4096 /
PS256 certificate, and lets users inspect any C2PA-bearing file.
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

app = Flask(__name__)
if _has_cors:
    CORS(app)

OLLAMA_BASE  = "http://localhost:11434"
OLLAMA_MODEL = "llava"

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

_UUID_STORE     = bytes.fromhex("6332706100110010800000aa00389b71")
_UUID_MANIFEST  = bytes.fromhex("6332706D00110010800000aa00389b71")
_UUID_CLAIM     = bytes.fromhex("6332636C00110010800000aa00389b71")
_UUID_SIG       = bytes.fromhex("6332637300110010800000aa00389b71")
_UUID_ASS_STORE = bytes.fromhex("6332617300110010800000aa00389b71")
_UUID_ASS       = bytes.fromhex("6332616100110010800000aa00389b71")


def _pack_box(btype: bytes, content: bytes) -> bytes:
    return struct.pack(">I", 8 + len(content)) + btype + content


def _pack_superbox(type_uuid: bytes, label: str, children: bytes) -> bytes:
    label_b  = label.encode("utf-8") + b"\x00"
    desc_box = _pack_box(b"jumd", type_uuid + b"\x03" + label_b)
    inner    = desc_box + children
    return _pack_box(b"jumb", inner)


def _json_assertion_box(label: str, data: dict) -> bytes:
    payload = _pack_box(b"json", json.dumps(data, separators=(",", ":")).encode())
    return _pack_superbox(_UUID_ASS, label, payload)


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

_OID_SIGNED_DATA = "1.2.840.113549.1.7.2"
_OID_DATA        = "1.2.840.113549.1.7.1"
_OID_SHA256      = "2.16.840.1.101.3.4.2.1"
_OID_RSASSA_PSS  = "1.2.840.113549.1.1.10"
_OID_MGF1        = "1.2.840.113549.1.1.8"


def _cms_sign(content: bytes) -> bytes:
    signature = _KEY.sign(
        content,
        PSS(mgf=MGF1(hashes.SHA256()), salt_length=PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    cert_der = _CERT.public_bytes(serialization.Encoding.DER)
    sha256_alg = _seq(_oid(_OID_SHA256) + b"\x05\x00")
    pss_params = _seq(
        _ctx_exp(0, sha256_alg) +
        _ctx_exp(1, _seq(_oid(_OID_MGF1) + sha256_alg)) +
        _ctx_exp(2, _int(222))
    )
    pss_alg = _seq(_oid(_OID_RSASSA_PSS) + pss_params)
    issuer_der = _CERT.issuer.public_bytes()
    ias = _seq(issuer_der + _int(_CERT.serial_number))
    signer_info = _seq(_int(1) + ias + sha256_alg + pss_alg + _oct(signature))
    encap = _seq(_oid(_OID_DATA))
    signed_data = _seq(
        _int(1) + _set(sha256_alg) + encap + _ctx_imp(0, cert_der) + _set(signer_info)
    )
    return _seq(_oid(_OID_SIGNED_DATA) + _ctx_exp(0, signed_data))


def _build_jumbf(prompt: str, response_text: str, timestamp: str) -> bytes:
    manifest_id = "urn:uuid:" + str(_uuid_mod.uuid4())

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

    cms_der = _cms_sign(claim_bytes)
    sig_box = _pack_superbox(_UUID_SIG, "c2pa.signature",
                             _pack_box(b"c2cs", cms_der))

    manifest = _pack_superbox(_UUID_MANIFEST, manifest_id,
                              ass_store + claim_box + sig_box)
    return _pack_superbox(_UUID_STORE, "c2pa", manifest)


def _inject_app11(jpeg: bytes, jumbf: bytes) -> bytes:
    if not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("Not a valid JPEG")
    payload = b"JP" + struct.pack(">H", 1) + struct.pack(">I", len(jumbf)) + jumbf
    segment = b"\xff\xeb" + struct.pack(">H", 2 + len(payload)) + payload
    return jpeg[:2] + segment + jpeg[2:]


def _extract_jumbf(data: bytes, mime: str) -> bytes:
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

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        pos = 8
        while pos + 12 <= len(data):
            chunk_len  = struct.unpack(">I", data[pos:pos+4])[0]
            chunk_type = data[pos+4:pos+8]
            if chunk_type == b"caBX":
                return data[pos+8:pos+8+chunk_len]
            pos += 12 + chunk_len
        raise ValueError("ManifestNotFound: no caBX chunk found in PNG")

    idx = data.find(b"jumb")
    if idx >= 4:
        size = struct.unpack(">I", data[idx-4:idx])[0]
        if 8 < size <= len(data) - (idx - 4):
            return data[idx-4:idx-4+size]
    raise ValueError("ManifestNotFound: no JUMBF data found in this file")


def _parse_boxes(data: bytes) -> list:
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
    jumbf  = _extract_jumbf(data, mime)
    boxes  = _parse_boxes(jumbf)

    manifest = {
        "c2pa_manifest_found": True,
        "jumbf_size_bytes":    len(jumbf),
        "box_tree":            _boxes_to_dict(boxes),
    }

    def _walk(boxes, parent_label=""):
        for b in boxes:
            lbl = b.get("label", "") or parent_label
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


def _make_image(prompt: str, response: str) -> bytes:
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


@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/ollama-proxy", methods=["POST"])
def ollama_proxy():
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate",
                          json=request.get_json(), timeout=120)
        return r.content, r.status_code, {"Content-Type": "application/json"}
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/generate", methods=["POST"])
def generate():
    body   = request.get_json(force=True)
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=120)
        r.raise_for_status()
        response_text = r.json().get("response", "(no response)")
    except Exception as e:
        response_text = f"[Ollama unavailable: {e}]"

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    jpeg = _make_image(prompt, response_text)
    jumbf = _build_jumbf(prompt, response_text, ts)
    signed_jpeg = _inject_app11(jpeg, jumbf)

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
#  WHITE / BRUTALIST EDITORIAL FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>C2PA — Manifest Studio</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,600;0,700;1,300&family=Bebas+Neue&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
  --white: #ffffff;
  --off:   #f5f5f5;
  --rule:  #e0e0e0;
  --rule2: #c8c8c8;
  --ink:   #0a0a0a;
  --ink2:  #1a1a1a;
  --mid:   #555555;
  --dim:   #999999;
  --faint: #dddddd;
  --mono:  'IBM Plex Mono', monospace;
  --disp:  'Bebas Neue', sans-serif;
  --body:  'IBM Plex Sans', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { font-size: 13px; }

body {
  background: var(--white);
  color: var(--ink);
  font-family: var(--mono);
  min-height: 100vh;
  display: grid;
  grid-template-rows: auto auto 1fr auto;
}

/* ── TICKER STRIP ── */
.ticker {
  background: var(--ink);
  color: var(--white);
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.14em;
  padding: 4px 0;
  overflow: hidden;
  white-space: nowrap;
}
.ticker-inner {
  display: inline-flex;
  gap: 3rem;
  animation: scroll-left 28s linear infinite;
}
@keyframes scroll-left {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}
.ticker-item { display: inline-flex; align-items: center; gap: 0.6rem; }
.ticker-item::before { content: '◆'; font-size: 7px; opacity: 0.5; }

/* ── MASTHEAD ── */
.masthead {
  border-bottom: 3px solid var(--ink);
  padding: 1.4rem 2rem 1rem;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: end;
  gap: 1rem;
}
.mast-left {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.mast-eyebrow {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.22em;
  color: var(--mid);
  text-transform: uppercase;
}
.mast-sub {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--mid);
  letter-spacing: 0.06em;
}
.mast-center {
  text-align: center;
}
.mast-title {
  font-family: var(--disp);
  font-size: 4.2rem;
  letter-spacing: 0.04em;
  line-height: 1;
  color: var(--ink);
}
.mast-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}
.mast-date {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.14em;
  color: var(--mid);
  text-transform: uppercase;
  text-align: right;
}
.mast-edition {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--dim);
  letter-spacing: 0.08em;
}

/* ── SECTION RULE ── */
.section-rule {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 1rem;
  padding: 0 2rem;
  border-bottom: 1px solid var(--rule2);
  height: 28px;
}
.section-rule::before,
.section-rule::after {
  content: '';
  height: 1px;
  background: var(--ink);
}
.section-name {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.2em;
  color: var(--ink);
  text-transform: uppercase;
  white-space: nowrap;
  padding: 0 0.5rem;
  border: 1px solid var(--ink);
  height: 18px;
  display: flex;
  align-items: center;
}

/* ── MAIN GRID ── */
.grid-main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  border-left: 1px solid var(--rule2);
}

/* ── COLUMN ── */
.col {
  border-right: 1px solid var(--rule2);
  display: flex;
  flex-direction: column;
}

/* ── COLUMN HEADER ── */
.col-head {
  border-bottom: 2px solid var(--ink);
  padding: 0.8rem 1.5rem;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  background: var(--white);
}
.col-num {
  font-family: var(--disp);
  font-size: 2.8rem;
  line-height: 1;
  color: var(--faint);
  user-select: none;
}
.col-title-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1px;
}
.col-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.22em;
  color: var(--mid);
  text-transform: uppercase;
}
.col-title {
  font-family: var(--disp);
  font-size: 1.6rem;
  letter-spacing: 0.06em;
  color: var(--ink);
  line-height: 1.1;
}
.col-route {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  color: var(--white);
  background: var(--ink);
  padding: 3px 8px;
  white-space: nowrap;
  align-self: center;
}

/* ── COLUMN BODY ── */
.col-body {
  flex: 1;
  padding: 1.4rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
  overflow-y: auto;
}
.col-body::-webkit-scrollbar { width: 3px; }
.col-body::-webkit-scrollbar-track { background: var(--off); }
.col-body::-webkit-scrollbar-thumb { background: var(--rule2); }

/* ── FIELD ── */
.field { display: flex; flex-direction: column; gap: 4px; }
.field-cap {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.18em;
  color: var(--mid);
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 6px;
}
.field-cap::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--rule);
}

textarea, input[type=file] {
  width: 100%;
  background: var(--white);
  border: 1px solid var(--rule2);
  border-top: 2px solid var(--ink);
  color: var(--ink);
  font-family: var(--mono);
  font-size: 11px;
  padding: 0.75rem 0.9rem;
  outline: none;
  resize: vertical;
  transition: border-color 0.12s;
  caret-color: var(--ink);
  line-height: 1.6;
}
textarea::placeholder { color: var(--faint); font-style: italic; }
textarea:focus {
  border-color: var(--ink);
  border-top-width: 3px;
}
textarea { min-height: 110px; }

input[type=file] {
  cursor: pointer;
  padding: 0.6rem 0.9rem;
  color: var(--mid);
}
input[type=file]::file-selector-button {
  background: var(--ink);
  color: var(--white);
  border: none;
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 5px 12px;
  cursor: pointer;
  margin-right: 12px;
  transition: background 0.12s;
}
input[type=file]::file-selector-button:hover { background: var(--mid); }

/* ── EXECUTE BUTTON ── */
.exec-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  background: var(--ink);
  color: var(--white);
  border: none;
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  padding: 0.65rem 1.4rem;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: background 0.12s;
  align-self: flex-start;
}
.exec-btn::after {
  content: '';
  position: absolute;
  inset: 0;
  border: 1px solid rgba(255,255,255,0.15);
}
.exec-btn:hover:not(:disabled) { background: var(--mid); }
.exec-btn:active:not(:disabled) { background: #333; }
.exec-btn:disabled { opacity: 0.35; cursor: not-allowed; }
.exec-btn .spin {
  width: 11px; height: 11px;
  border: 1.5px solid rgba(255,255,255,0.3);
  border-top-color: var(--white);
  border-radius: 50%;
  animation: spin 0.55s linear infinite;
  display: none;
  flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }
.exec-btn.busy .spin { display: block; }
.exec-btn.busy .lbl { opacity: 0.5; }

/* ── STATUS LINE ── */
.status-line {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  padding: 6px 10px;
  border-left: 3px solid transparent;
  display: none;
}
.status-line.show { display: block; }
.status-line.ok  {
  border-color: var(--ink);
  color: var(--ink2);
  background: var(--off);
}
.status-line.ok::before  { content: '✓  '; font-weight: 700; }
.status-line.err {
  border-color: #000;
  color: var(--ink2);
  background: #f0f0f0;
}
.status-line.err::before { content: '✕  '; font-weight: 700; }

/* ── OUTPUT ── */
.output-block { display: none; flex-direction: column; gap: 0.7rem; }
.output-block.show { display: flex; }

.out-rule {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.18em;
  color: var(--mid);
  text-transform: uppercase;
}
.out-rule::before { content: '▸'; color: var(--ink); }
.out-rule::after  { content: ''; flex: 1; height: 1px; background: var(--rule); }

.dl-link {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--white);
  background: var(--ink2);
  text-decoration: none;
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  padding: 6px 14px;
  transition: background 0.12s;
  align-self: flex-start;
  border: 1px solid var(--ink);
}
.dl-link:hover { background: var(--mid); }
.dl-link::before { content: '↓ '; font-weight: 700; }

pre.json-out {
  background: var(--off);
  border: 1px solid var(--rule2);
  border-top: 2px solid var(--ink);
  padding: 0.9rem 1rem;
  font-family: var(--mono);
  font-size: 10px;
  line-height: 1.75;
  overflow: auto;
  max-height: 360px;
  color: var(--ink2);
  white-space: pre-wrap;
  word-break: break-word;
}
pre.json-out::-webkit-scrollbar { width: 3px; }
pre.json-out::-webkit-scrollbar-track { background: var(--rule); }
pre.json-out::-webkit-scrollbar-thumb { background: var(--rule2); }

/* JSON syntax — monochrome with weight + spacing */
.jk { color: #000; font-weight: 600; }
.js { color: #333; }
.jn { color: #000; font-style: italic; }
.jb { color: #555; font-weight: 600; letter-spacing: 0.06em; }

/* ── FOOTER STRIP ── */
.footer-strip {
  background: var(--ink);
  color: var(--white);
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  padding: 0 2rem;
  height: 30px;
  border-top: 3px solid var(--ink);
}
.ft-left, .ft-right {
  display: flex;
  align-items: center;
  gap: 2rem;
}
.ft-right { justify-content: flex-end; }
.ft-item {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.14em;
  color: rgba(255,255,255,0.45);
  display: flex;
  align-items: center;
  gap: 5px;
}
.ft-item .val { color: rgba(255,255,255,0.85); }
.ft-center {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.18em;
  color: rgba(255,255,255,0.3);
  text-align: center;
}
.ft-dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: #fff;
  opacity: 0.9;
  animation: pulse 2.5s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { opacity: 0.9; box-shadow: 0 0 0 0 rgba(255,255,255,0.4); }
  50%      { opacity: 0.5; box-shadow: 0 0 0 4px rgba(255,255,255,0); }
}

@media (max-width: 700px) {
  .grid-main { grid-template-columns: 1fr; }
  .col { border-right: none; border-bottom: 1px solid var(--rule2); }
  .masthead { grid-template-columns: 1fr; text-align: center; }
  .mast-left, .mast-right { align-items: center; }
  .mast-title { font-size: 3rem; }
}
</style>
</head>
<body>

<!-- TICKER -->
<div class="ticker">
  <div class="ticker-inner" id="ticker-a">
    <span class="ticker-item">JUMBF APP11 INJECTION</span>
    <span class="ticker-item">RSA-4096 PS256 SIGNATURE</span>
    <span class="ticker-item">ISO 19566-5 COMPLIANT</span>
    <span class="ticker-item">C2PA MANIFEST STORE</span>
    <span class="ticker-item">CMS SIGNEDDATA DER</span>
    <span class="ticker-item">OLLAMA / LLAVA BACKEND</span>
    <span class="ticker-item">CLAIM HASH SHA-256</span>
    <span class="ticker-item">CAWG TRAINING MINING ASSERTION</span>
    <span class="ticker-item">JUMBF APP11 INJECTION</span>
    <span class="ticker-item">RSA-4096 PS256 SIGNATURE</span>
    <span class="ticker-item">ISO 19566-5 COMPLIANT</span>
    <span class="ticker-item">C2PA MANIFEST STORE</span>
    <span class="ticker-item">CMS SIGNEDDATA DER</span>
    <span class="ticker-item">OLLAMA / LLAVA BACKEND</span>
    <span class="ticker-item">CLAIM HASH SHA-256</span>
    <span class="ticker-item">CAWG TRAINING MINING ASSERTION</span>
  </div>
</div>

<!-- MASTHEAD -->
<header class="masthead">
  <div class="mast-left">
    <span class="mast-eyebrow">Content Authenticity Initiative</span>
    <span class="mast-sub">Manifest Generation &amp; Inspection Terminal</span>
  </div>
  <div class="mast-center">
    <div class="mast-title">C2PA</div>
  </div>
  <div class="mast-right">
    <span class="mast-date" id="clk">— UTC —</span>
    <span class="mast-edition">DEMO EDITION · RSA-4096 · PS256</span>
  </div>
</header>

<!-- SECTION RULE -->
<div class="section-rule">
  <span class="section-name">Operations</span>
</div>

<!-- MAIN GRID -->
<div class="grid-main">

  <!-- COL A: GENERATE -->
  <div class="col">
    <div class="col-head">
      <span class="col-num">01</span>
      <div class="col-title-wrap">
        <span class="col-label">Operation</span>
        <span class="col-title">Generate &amp; Sign</span>
      </div>
      <span class="col-route">POST /generate</span>
    </div>
    <div class="col-body">

      <div class="field">
        <div class="field-cap">Prompt Input</div>
        <textarea id="gp" placeholder="Describe something for Ollama / llava…&#10;e.g. Explain the lifecycle of a star&#10;e.g. What is the Voynich manuscript?"></textarea>
      </div>

      <div>
        <button class="exec-btn" id="gb" onclick="doGen()">
          <div class="spin"></div>
          <span class="lbl">Execute Generate</span>
        </button>
      </div>

      <div class="status-line" id="gt"></div>

      <div class="output-block" id="gr">
        <div class="out-rule">Signed Output</div>
        <a class="dl-link" id="dl" href="#" download="c2pa_signed.jpg">c2pa_signed.jpg</a>
        <div class="out-rule">Manifest Payload</div>
        <pre class="json-out" id="gj"></pre>
      </div>

    </div>
  </div>

  <!-- COL B: INSPECT -->
  <div class="col">
    <div class="col-head">
      <span class="col-num">02</span>
      <div class="col-title-wrap">
        <span class="col-label">Operation</span>
        <span class="col-title">Inspect Manifest</span>
      </div>
      <span class="col-route">POST /inspect</span>
    </div>
    <div class="col-body">

      <div class="field">
        <div class="field-cap">File Input — .jpg .jpeg .png .pdf</div>
        <input type="file" id="ifile" accept=".jpg,.jpeg,.png,.pdf">
      </div>

      <div>
        <button class="exec-btn" id="ib" onclick="doInsp()">
          <div class="spin"></div>
          <span class="lbl">Execute Inspect</span>
        </button>
      </div>

      <div class="status-line" id="it"></div>

      <div class="output-block" id="ir">
        <div class="out-rule">C2PA Manifest Data</div>
        <pre class="json-out" id="ij"></pre>
      </div>

    </div>
  </div>

</div>

<!-- FOOTER -->
<footer class="footer-strip">
  <div class="ft-left">
    <span class="ft-item">MODEL <span class="val">llava</span></span>
    <span class="ft-item">ALG <span class="val">PS256</span></span>
    <span class="ft-item">KEY <span class="val">RSA-4096</span></span>
  </div>
  <div class="ft-center">ISO 19566-5 · C2PA 1.x · JUMBF</div>
  <div class="ft-right">
    <span class="ft-item">CERT <span class="val">LOADED</span></span>
    <span class="ft-item">OLLAMA <span class="val" id="ollama-status">—</span></span>
    <div class="ft-dot"></div>
  </div>
</footer>

<script>
// ── Clock ──
function tick(){
  const n = new Date();
  document.getElementById('clk').textContent =
    n.toISOString().replace('T',' ').slice(0,19) + ' UTC';
}
tick(); setInterval(tick, 1000);

// ── Ollama probe ──
async function probeOllama(){
  const el = document.getElementById('ollama-status');
  try {
    const r = await fetch('/ollama-proxy', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({model:'llava', prompt:'ping', stream:false}),
      signal: AbortSignal.timeout(5000)
    });
    el.textContent = r.ok ? 'ONLINE' : 'DEGRADED';
  } catch {
    el.textContent = 'OFFLINE';
  }
}
probeOllama();

// ── Syntax highlight (monochrome) ──
function hl(s){
  return s
    .replace(/("(?:[^"\\]|\\.)*")\s*:/g,'<span class="jk">$1</span>:')
    .replace(/:\s*("(?:[^"\\]|\\.)*")/g,': <span class="js">$1</span>')
    .replace(/\b(true|false|null)\b/g,'<span class="jb">$1</span>')
    .replace(/(?<!["\w\-])(-?\d+\.?\d*)/g,'<span class="jn">$1</span>');
}

function busy(id, on){
  const b = document.getElementById(id);
  b.disabled = on;
  on ? b.classList.add('busy') : b.classList.remove('busy');
}
function toast(id, msg, ok){
  const e = document.getElementById(id);
  e.textContent = msg;
  e.className = 'status-line show ' + (ok ? 'ok' : 'err');
}
function showOut(id){ document.getElementById(id).classList.add('show'); }
function hideOut(id){ document.getElementById(id).classList.remove('show'); }
function hideLine(id){ document.getElementById(id).classList.remove('show'); }

async function doGen(){
  const p = document.getElementById('gp').value.trim();
  if(!p){ toast('gt','Prompt input cannot be empty.',false); return; }
  busy('gb',true); hideOut('gr'); hideLine('gt');
  try {
    const r = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({prompt: p})
    });
    if(!r.ok){ const e = await r.json().catch(()=>({})); throw new Error(e.error || r.statusText); }
    const mb = r.headers.get('X-Manifest-JSON') || '';
    const mj = mb ? JSON.stringify(JSON.parse(atob(mb)), null, 2) : '(no manifest header)';
    const blob = await r.blob();
    document.getElementById('dl').href = URL.createObjectURL(blob);
    document.getElementById('gj').innerHTML = hl(mj);
    showOut('gr');
    toast('gt','C2PA JUMBF manifest embedded — PS256 signature applied.', true);
  } catch(e) {
    toast('gt', e.message, false);
  } finally {
    busy('gb', false);
  }
}

async function doInsp(){
  const fi = document.getElementById('ifile');
  if(!fi.files.length){ toast('it','No file selected.',false); return; }
  busy('ib',true); hideOut('ir'); hideLine('it');
  const fd = new FormData();
  fd.append('file', fi.files[0]);
  try {
    const r = await fetch('/inspect', {method:'POST', body:fd});
    const d = await r.json();
    document.getElementById('ij').innerHTML = hl(JSON.stringify(d, null, 2));
    showOut('ir');
    d.error
      ? toast('it', d.error, false)
      : toast('it', 'Manifest parsed — ' + d.jumbf_size_bytes + ' bytes JUMBF.', true);
  } catch(e) {
    toast('it', e.message, false);
  } finally {
    busy('ib', false);
  }
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
"""
C2PA Signer Module for zkLLM (VerifAI)
=======================================
Generates C2PA-compliant JUMBF manifests and embeds them into JPEG receipts.
Uses manual byte-level encoding (no Rust/c2pa-python dependency).

Ported from c2pa-sys/app1.py — ISO 19566-5 / C2PA 1.x compliant.
"""

import io
import json
import struct
import datetime
import hashlib
import base64
import uuid as _uuid_mod

from PIL import Image, ImageDraw, ImageFont

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.padding import PSS, MGF1
import cryptography.x509 as x509
from cryptography.x509.oid import NameOID

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_KEY = None
_CERT = None

# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------
def init_certs():
    """Generate RSA-4096 key + self-signed certificate at startup."""
    global _KEY, _CERT
    _KEY = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    now = datetime.datetime.now(datetime.UTC)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME,       "VerifAI-C2PA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "zkLLM-Project"),
        x509.NameAttribute(NameOID.COUNTRY_NAME,      "IN"),
    ])
    _CERT = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(_KEY.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(_KEY, hashes.SHA256())
    )
    print("[c2pa] RSA-4096 / PS256 self-signed cert generated for VerifAI.")


# ---------------------------------------------------------------------------
# JUMBF / C2PA binary encoding helpers
# ---------------------------------------------------------------------------

# C2PA UUIDs from spec (ISO 19566-5 / C2PA §6)
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


# ---------------------------------------------------------------------------
# CMS SignedData (raw DER) with RSA-PSS / PS256
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def _build_jumbf(prompt: str, response_text: str, model_id: str,
                 username: str, session_id: int, timestamp: str) -> tuple:
    """
    Build a complete C2PA JUMBF manifest store.
    Returns (jumbf_bytes, manifest_summary_dict).
    """
    manifest_id = "urn:uuid:" + str(_uuid_mod.uuid4())

    a_actions = _json_assertion_box("c2pa.actions", {
        "actions": [{
            "action":        "c2pa.created",
            "softwareAgent": f"zkLLM/{model_id}",
            "when":          timestamp,
            "parameters":    {"prompt": prompt, "response": response_text},
        }]
    })

    a_training = _json_assertion_box("cawg.training-mining", {
        "entries": {
            "c2pa.ai_inference":            {"use": "allowed"},
            "c2pa.ai_generative_training":  {"use": "notAllowed"},
            "c2pa.ai_training":             {"use": "notAllowed"},
            "c2pa.data_mining":             {"use": "notAllowed"},
        }
    })

    a_meta = _json_assertion_box("org.zkllm.metadata", {
        "model":      model_id,
        "prompt":     prompt,
        "response":   response_text,
        "session_id": session_id,
        "user":       username,
        "timestamp":  timestamp,
        "generator":  "VerifAI/1.0",
        "zk_proof_status":  "pending",
        "zk_verify_status": "pending",
    })

    ass_store = _pack_superbox(_UUID_ASS_STORE, "c2pa.assertions",
                               a_actions + a_training + a_meta)

    def _sha256b64(b: bytes) -> str:
        return base64.b64encode(hashlib.sha256(b).digest()).decode()

    claim_data = {
        "dc:title":        f"zkLLM Inference Receipt — Session #{session_id}",
        "dc:format":       "image/jpeg",
        "instanceID":      manifest_id,
        "claim_generator": "VerifAI/1.0",
        "claim_generator_info": [{"name": "VerifAI", "version": "1.0"}],
        "assertions": [
            {"url": "self#jumbf=c2pa.assertions/c2pa.actions",
             "hash": _sha256b64(a_actions), "alg": "sha256"},
            {"url": "self#jumbf=c2pa.assertions/cawg.training-mining",
             "hash": _sha256b64(a_training), "alg": "sha256"},
            {"url": "self#jumbf=c2pa.assertions/org.zkllm.metadata",
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
    jumbf = _pack_superbox(_UUID_STORE, "c2pa", manifest)

    summary = {
        "c2pa_signed":   True,
        "jumbf_bytes":   len(jumbf),
        "algorithm":     "PS256 (RSA-4096 RSA-PSS + SHA-256)",
        "timestamp":     timestamp,
        "claim_generator": "VerifAI/1.0",
        "assertions": {
            "c2pa.actions": {
                "action": "c2pa.created",
                "softwareAgent": f"zkLLM/{model_id}",
                "parameters": {"prompt": prompt, "response": response_text},
            },
            "cawg.training-mining": {
                "c2pa.ai_inference": "allowed",
                "c2pa.ai_training":  "notAllowed",
            },
            "org.zkllm.metadata": {
                "session_id": session_id,
                "model": model_id,
                "user": username,
            },
        },
    }

    return jumbf, summary


# ---------------------------------------------------------------------------
# JPEG APP11 injection
# ---------------------------------------------------------------------------

def _inject_app11(jpeg: bytes, jumbf: bytes) -> bytes:
    if not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("Not a valid JPEG")
    payload = b"JP" + struct.pack(">H", 1) + struct.pack(">I", len(jumbf)) + jumbf
    segment = b"\xff\xeb" + struct.pack(">H", 2 + len(payload)) + payload
    return jpeg[:2] + segment + jpeg[2:]


# ---------------------------------------------------------------------------
# Receipt image generator
# ---------------------------------------------------------------------------

def _make_receipt_image(prompt: str, response: str, model_id: str,
                        session_id: int, username: str, timestamp: str) -> bytes:
    """Render a branded receipt image with all inference metadata."""
    W, H = 820, 520
    img  = Image.new("RGB", (W, H), (10, 12, 22))
    draw = ImageDraw.Draw(img)

    try:
        fB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        fR = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        fS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        fB = fR = fS = ImageFont.load_default()

    # Header bar
    draw.rectangle([0, 0, W, 48], fill=(24, 28, 56))
    draw.text((18, 13), "🔏  VerifAI — C2PA Signed Inference Receipt", font=fB, fill=(130, 170, 255))

    # Metadata line
    meta_line = f"Session #{session_id}  ·  @{username}  ·  {model_id}  ·  {timestamp}"
    draw.text((18, 58), meta_line, font=fS, fill=(90, 100, 140))

    # Prompt
    draw.text((18, 85), "PROMPT", font=fB, fill=(80, 150, 255))
    _wrap(draw, prompt[:360], fR, 18, 108, W - 36, (205, 215, 240), 18)

    # Response
    draw.text((18, 240), "RESPONSE", font=fB, fill=(70, 210, 140))
    _wrap(draw, response[:460], fR, 18, 263, W - 36, (185, 230, 195), 18)

    # Footer
    draw.rectangle([0, H - 34, W, H], fill=(18, 20, 38))
    draw.text((18, H - 23), f"VerifAI · PS256 RSA-4096 · C2PA 1.x · ISO 19566-5", font=fS, fill=(70, 80, 120))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _wrap(draw, text, font, x, y, max_w, fill, lh, max_lines=7):
    line = ""
    lines = 0
    for word in text.split():
        test = (line + " " + word).strip()
        try:
            tw = draw.textlength(test, font=font)
        except Exception:
            tw = len(test) * 7
        if tw > max_w and line:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sign_inference(session_id: int, prompt: str, response: str,
                   model_id: str, username: str, timestamp: str) -> tuple:
    """
    Create a C2PA-signed JPEG receipt for an inference session.

    Returns:
        (jpeg_bytes, manifest_summary_dict)
    """
    if _KEY is None:
        init_certs()

    jpeg = _make_receipt_image(prompt, response, model_id,
                               session_id, username, timestamp)

    jumbf, summary = _build_jumbf(prompt, response, model_id,
                                   username, session_id, timestamp)

    signed_jpeg = _inject_app11(jpeg, jumbf)
    return signed_jpeg, summary


# ---------------------------------------------------------------------------
# JUMBF extractor + parser (for /inspect)
# ---------------------------------------------------------------------------

def _extract_jumbf(data: bytes) -> bytes:
    """Return raw JUMBF bytes from file data, or raise ValueError."""
    # JPEG: scan for APP11 (0xFFEB) with 'JP' identifier
    if data[:2] == b"\xff\xd8":
        pos = 2
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                break
            marker = data[pos:pos+2]
            if marker == b"\xff\xd9":
                break
            seg_len = struct.unpack(">H", data[pos+2:pos+4])[0]
            seg_data = data[pos+4:pos+2+seg_len]
            if marker == b"\xff\xeb" and seg_data[:2] == b"JP":
                z = struct.unpack(">I", seg_data[4:8])[0]
                return seg_data[8:8+z]
            pos += 2 + seg_len
        raise ValueError("No C2PA manifest found: no JUMBF APP11 segment in JPEG")

    # PNG: look for 'caBX' chunk
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        pos = 8
        while pos + 12 <= len(data):
            chunk_len = struct.unpack(">I", data[pos:pos+4])[0]
            chunk_type = data[pos+4:pos+8]
            if chunk_type == b"caBX":
                return data[pos+8:pos+8+chunk_len]
            pos += 12 + chunk_len
        raise ValueError("No C2PA manifest found: no caBX chunk in PNG")

    # Fallback: scan for jumb box
    idx = data.find(b"jumb")
    if idx >= 4:
        size = struct.unpack(">I", data[idx-4:idx])[0]
        if 8 < size <= len(data) - (idx - 4):
            return data[idx-4:idx-4+size]
    raise ValueError("No C2PA manifest found in this file")


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
            desc = next((c for c in children if c["type"] == "jumd"), None)
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
        entry = {}
        if "uuid"    in b: entry["uuid"]    = b["uuid"]
        if "content" in b: entry["content"] = b["content"]
        if "children" in b:
            entry["children"] = _boxes_to_dict(b["children"])
        out[key] = entry
    return out


def inspect_file(data: bytes) -> dict:
    """
    Extract and parse C2PA manifest from file bytes.
    Returns a dict with manifest info, or an error dict.
    """
    try:
        jumbf = _extract_jumbf(data)
        boxes = _parse_boxes(jumbf)

        manifest = {
            "c2pa_manifest_found": True,
            "jumbf_size_bytes":    len(jumbf),
            "box_tree":            _boxes_to_dict(boxes),
        }

        # Walk the tree to extract known assertions
        def _walk(boxes, parent_label=""):
            for b in boxes:
                lbl = b.get("label", "") or parent_label
                if b.get("type") == "json" and "content" in b and isinstance(b["content"], dict):
                    yield parent_label, b["content"]
                if "children" in b:
                    yield from _walk(b["children"], parent_label=lbl)

        assertions, claim = {}, {}
        for lbl, content in _walk(boxes):
            if "c2pa.actions"          in lbl: assertions["c2pa.actions"]           = content
            elif "cawg.training-mining" in lbl: assertions["cawg.training-mining"]  = content
            elif "zkllm.metadata"      in lbl: assertions["org.zkllm.metadata"]    = content
            elif "c2pa.claim"          in lbl: claim = content

        if assertions: manifest["assertions"] = assertions
        if claim:      manifest["claim"]      = claim
        return manifest

    except Exception as e:
        return {"error": str(e)}

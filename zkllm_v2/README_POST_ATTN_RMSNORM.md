# Post-Attention RMSNorm in zkLLM v2

## Code and Implementation README

## Scope

This README documents only the Post-Attention RMSNorm proving and verification path in this repository.

Focus areas:

- Actual code flow (Python wrapper -> CUDA prover -> serialized proof -> CUDA verifier)
- Post-attention specific file naming and orchestration
- Proof data model and binary layout
- Mathematical foundations required to understand the implementation
- Baselines and caveats for debugging and extension

This is intentionally implementation-first and verifier-centric.

---

## 1. Where Post-Attention RMSNorm Sits in the Layer Pipeline

Per transformer layer, Post-Attention RMSNorm is executed after self-attention output and before FFN input:

- Input to Post-Attention RMSNorm:
  - `activations/layer-N-self-attn-output.bin`
- Output from Post-Attention RMSNorm:
  - `activations/layer-N-ffn-activation.bin`
- Proof file:
  - `zkllm-workdir/Llama-2-<size>b/layer-N-post-attn-rmsnorm-proof.bin`

Orchestration references:

- `generate_proofs_v2.py` -> `generate_post_attn_rmsnorm(...)`
- `verify_proofs_v2.py` -> `verify_post_attn_rmsnorm(...)`

---

## 2. File-Level Implementation Map

- `llama-post-attn-rmsnorm_v2.py`
  - Python wrapper for post-attention RMSNorm proof generation.
  - Compiles `rmsnorm_v2`, computes and stores per-layer `rms_inv`, invokes CUDA prover.

- `rmsnorm_v2.cu`
  - Unified RMSNorm prover binary for both `input` and `post_attention`.
  - Produces serialized `RMSNormProof` and normalized output activation.

- `verify_rmsnorm_v2.cu`
  - Unified RMSNorm verifier binary for both `input` and `post_attention`.
  - Loads proof + commitments and performs structural/cryptographic checks.

- `proof_io_v2.{cuh,cu}`
  - Defines `RMSNormProof` and binary save/load logic.

- `proof_v2.cuh`
  - Declares `hadamard_product_sumcheck(...)` and proof-related primitives.

- `zkfc_v2.{cuh,cu}`
  - Used by RMSNorm path for proving/validating the weight-scaled RMS term as a linear map.

---

## 3. Mathematical Foundations Needed for This Code Path

## 3.1 RMSNorm equation

For sequence position `i` and feature index `j`:

$$
\mathrm{RMSNorm}(X)_{i,j} = \gamma_j \cdot \frac{X_{i,j}}{\sqrt{\frac{1}{d}\sum_{k=1}^{d} X_{i,k}^2 + \epsilon}}
$$

In this implementation:

- `rms_inv[i] = 1 / sqrt(mean(X_i^2) + epsilon)` is precomputed in Python.
- The CUDA prover computes:
  - `g_inv_rms = zkFC(gamma)(rms_inv)`
  - `g_inv_rms_ = Rescaling(2^16)(g_inv_rms)`
  - `Y = g_inv_rms_ * X`
  - `Y_ = Rescaling(2^16)(Y)`

All tensor arithmetic is in `Fr` (BLS12-381 scalar field), with fixed-point encoded integers loaded from `.bin`.

## 3.2 Hadamard product sumcheck relation

The prover emits a hadamard sumcheck transcript binding:

$$
\widetilde{Y}(u) = \widetilde{g\_inv\_rms\_}(u) \cdot \widetilde{X}(u)
$$

where `u` is a random challenge vector stored in proof (`random_u`, and `random_v = random_u` in this path).

Verifier-side structural check currently enforces transcript length:

$$
|\mathrm{hp\_proof}| = 3\cdot|u| + 2
$$

## 3.3 Commitment and claim binding intuition

`gamma` (RMSNorm weight) is commitment-backed through files loaded by `create_weight(...)`.

Prover also stores `claimed_output` evaluated at random point `u`:

$$
\mathrm{claimed\_output} = \widetilde{g\_inv\_rms\_}(u) \cdot \widetilde{X}(u)
$$

Verifier recomputes that value from loaded commitments + input activation + `rms_inv`, then checks equality.

---

## 4. Proof Generation Flow (Post-Attention)

## 4.1 Python wrapper: `llama-post-attn-rmsnorm_v2.py`

1. Compiles `rmsnorm_v2` via:
   - `make -f Makefile_v2 rmsnorm_v2`
2. Resolves `embed_dim` and `variance_epsilon`:
   - either from CLI `--precomputed`
   - or by loading LLaMA model metadata
3. Loads input activation from `--input_file` (`int32`) and dequantizes by `2^16`.
4. Computes row-wise:
   - `rms_inv = 1 / sqrt(mean(X^2, dim=1) + epsilon)`
5. Saves per-layer, per-type file:
   - `workdir/layer-N-post_attention-rms_inv.bin`
6. Executes CUDA prover:
   - `./rmsnorm_v2 post_attention <input_file> <seq_len> <embed_dim> <workdir> <layer_prefix> <output_file> <rms_inv_file>`

Important: post-attention wrapper persists `rms_inv` with `post_attention` in filename, avoiding collision with input RMSNorm.

## 4.2 CUDA prover: `rmsnorm_v2.cu`

The binary is shared for `input` and `post_attention` paths and branches by `which`.

### Step A: resolve paths and load commitment-backed weight

- `proof_suffix = input-rmsnorm` or `post-attn-rmsnorm`
- proof file:
  - `workdir/layer-N-<proof_suffix>-proof.bin`
- load weight bundle:
  - `workdir/<which>_layernorm.weight-pp.bin`
  - `workdir/layer-N-<which>_layernorm.weight-int.bin`
  - `workdir/layer-N-<which>_layernorm.weight-commitment.bin`

### Step B: load tensors and compute forward

- `X = FrTensor::from_int_bin(input_file_name)`
- `rms_inv_temp = FrTensor::from_int_bin(rms_inv_file)`
- build `zkFC g(1, embed_dim, rmsnorm_weight.weight)`
- compute:
  - `g_inv_rms = g(rms_inv_temp)`
  - `g_inv_rms_ = rs1(g_inv_rms)`
  - `Y = g_inv_rms_ * X`
  - `Y_ = rs2(Y)`

### Step C: generate proof transcript

- sample random challenge vector:
  - `u = random_vec(ceilLog2(Y.size))`
- compute claim value before proof operations:
  - `claimed_output = g_inv_rms_(u) * X(u)`
- generate components:
  - `hp_proof_fr = hadamard_product_sumcheck(g_inv_rms_, X, u, u)`
  - `weight_proof_poly` via `g.prove(rms_inv_temp, g_inv_rms, weight_proof_poly)`
  - `rs1_proof` and `rs2_proof` remain empty in this path (internal checks only)

### Step D: serialize and write output

Populates `RMSNormProof` and writes by `save_rmsnorm_proof(...)`.

Also writes normalized activation:

- `Y_.save_int(output_file_name)`

---

## 5. RMSNormProof Data Model and Binary Layout

`RMSNormProof` fields (`proof_io_v2.cuh`):

- `hadamard_product_proof : vector<Fr_t>`
- `weight_proof : vector<Polynomial>`
- `rs1_proof : vector<Polynomial>`
- `rs2_proof : vector<Polynomial>`
- `random_u : vector<Fr_t>`
- `random_v : vector<Fr_t>`
- `claimed_output : Fr_t`

Serialization order (`save_rmsnorm_proof` in `proof_io_v2.cu`):

1. `hp_size` + hadamard transcript bytes
2. `weight_proof`
3. `rs1_proof`
4. `rs2_proof`
5. `u_size` + `random_u`
6. `v_size` + `random_v`
7. `claimed_output`

Deserializer (`load_rmsnorm_proof`) is backward compatible when random challenge fields are absent.

---

## 6. Verification Flow (Post-Attention-Centric)

`verify_rmsnorm_v2.cu` is a unified verifier invoked with `which=post_attention` for post-attention checks.

CLI:

```text
./verify_rmsnorm_v2 <proof_file> <workdir> <layer_prefix> <which> <input_activation_file>
```

For post-attention:

```text
./verify_rmsnorm_v2 workdir/layer-N-post-attn-rmsnorm-proof.bin workdir layer-N post_attention activations/layer-N-self-attn-output.bin
```

## 6.1 Step 1: load proof

- `proof = load_rmsnorm_proof(proof_file)`
- logs sizes for hadamard transcript, weight proof, RS proofs, challenge vectors
- if `random_u/random_v` missing, falls back to structural-only mode

## 6.2 Step 2: load commitment-backed weight

Loads post-attention layernorm weight from:

- `workdir/post_attention_layernorm.weight-pp.bin`
- `workdir/layer-N-post_attention_layernorm.weight-int.bin`
- `workdir/layer-N-post_attention_layernorm.weight-commitment.bin`

## 6.3 Step 2.5: cryptographic claimed-output check

If random challenges exist and input activation file is provided:

1. load input activation `X`
2. derive `rms_inv` path:
   - `workdir/layer-N-post_attention-rms_inv.bin`
3. fallback to `rms_inv_temp.bin` for legacy runs
4. recompute:
   - `g_inv_rms = zkFC(1, embed_dim, weight)(rms_inv)`
   - `g_inv_rms_ = Rescaling(2^16)(g_inv_rms)`
   - `computed_claim = g_inv_rms_(u) * X(u)`
5. enforce:
   - `computed_claim == proof.claimed_output`

Failure here is treated as cryptographic binding failure and verifier exits non-zero.

## 6.4 Step 3: hadamard transcript structural verification

- ensures hadamard transcript is non-empty
- if cryptographic mode:
  - checks `proof_size == 3*|u| + 2`
- old format mode:
  - checks legacy expected size (59) as structural heuristic

## 6.5 Step 4: weight-proof status

Current verifier behavior:

- if `weight_proof` empty: accepts structural validity message
- if non-empty: reports not fully implemented for full weight-proof verification path

This is an important baseline when interpreting verification guarantees.

## 6.6 Step 5: rescaling proof status

Current behavior expects RS proofs to be empty (internal prover checks only).

---

## 7. Post-Attention Naming and Path Baselines

Post-attention branch uses two different suffix conventions by design:

- proof file suffix in CUDA prover:
  - `post-attn-rmsnorm-proof.bin`
- `rms_inv` helper tensor suffix in Python wrapper and verifier:
  - `post_attention-rms_inv.bin`

So both are expected and valid in current code.

Typical per-layer files:

- `activations/layer-N-self-attn-output.bin`
- `activations/layer-N-ffn-activation.bin`
- `zkllm-workdir/Llama-2-7b/layer-N-post-attn-rmsnorm-proof.bin`
- `zkllm-workdir/Llama-2-7b/layer-N-post_attention-rms_inv.bin`

---

## 8. End-to-End Example Commands

## 8.1 Generate post-attention RMSNorm proof

```bash
python3 llama-post-attn-rmsnorm_v2.py 7 0 128 \
  --input_file ./activations/layer-0-self-attn-output.bin \
  --output_file ./activations/layer-0-ffn-activation.bin \
  --precomputed --embed_dim 4096 --variance_epsilon 1e-05
```

## 8.2 Verify post-attention RMSNorm proof

```bash
./verify_rmsnorm_v2 \
  ./zkllm-workdir/Llama-2-7b/layer-0-post-attn-rmsnorm-proof.bin \
  ./zkllm-workdir/Llama-2-7b \
  layer-0 \
  post_attention \
  ./activations/layer-0-self-attn-output.bin
```

## 8.3 Pipeline-level verification

`verify_proofs_v2.py` internally invokes the same verifier command in its layer loop.

---

## 9. Important Implementation Caveats for Post-Attention Verification

1. Verifier embed dimension is currently hardcoded in one path

- In `verify_rmsnorm_v2.cu`, Step 2 weight loading and Step 2.5 recomputation use `embed_dim = 4096`.
- This is correct for 7B but not generic for 13B unless adapted.

2. Full weight-proof replay is not fully enforced

- `weight_proof` can be loaded and counted, but full verification of that polynomial path is not completed in this verifier.

3. RS proofs are format-checked, not replay-verified

- `rs1_proof` and `rs2_proof` are expected empty by design for this path.

4. Random challenge availability changes verification strength

- If `random_u/random_v` are missing (legacy proof format), verifier downgrades to structural validation.

5. Claimed-output check depends on `rms_inv` availability

- Without per-layer `rms_inv` (or legacy fallback), binding check is skipped.

---

## 10. Minimal Mental Model for Post-Attention RMSNorm Verification

- Prover computes post-attention RMS normalization and emits:
  - output activation,
  - hadamard transcript,
  - weight proof artifacts,
  - random challenges,
  - claimed output value.

- Verifier loads commitment-backed post-attention layernorm weights and proof transcript.

- Verifier confirms transcript shape and, when data is present, recomputes random-point claim to cryptographically bind the proof to:
  - the committed post-attention weight,
  - the provided self-attention activation,
  - the exact random challenge point.

That is the implementation flow of Post-Attention RMSNorm proof generation and verification in this codebase.

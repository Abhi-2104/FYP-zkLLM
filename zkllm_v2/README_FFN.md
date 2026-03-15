# FFN Proof Generation and Verification in zkLLM v2

## Scope

This document is a code-first guide to the Feed-Forward Network (FFN) proving and verification path in the v2 pipeline.

It focuses on implementation details for:

- FFN proof generation (CUDA prover)
- FFN proof verification (CUDA verifier)
- Proof serialization format (`FFNProof`)
- Mathematical foundations needed to read and modify the implementation
- Practical baselines for running, debugging, and extending the FFN path

This is intentionally scoped to FFN only.

---

## 1. FFN in the Layer Pipeline

In each transformer layer, FFN runs after post-attention RMSNorm.

Conceptual FFN computation:

1. Up projection: `U = X * W_up`
2. Gate projection: `G = X * W_gate`
3. SwiGLU: `S = swiglu(G)`
4. Hidden combine: `H = S ⊙ U`
5. Down projection: `Y = H * W_down`

In this project, all values are represented in BLS12-381 scalar field form (`Fr_t`) using fixed-point encodings and explicit rescaling between phases.

---

## 2. File-Level Implementation Map

- `llama-ffn_v2.py`
  - Python wrapper and orchestration for FFN proving
  - Compiles `ffn_v2`, prepares SwiGLU table, resolves model dimensions, invokes CUDA binary

- `ffn_v2.cu`
  - FFN prover binary
  - Runs phased computation and emits `FFNProof`

- `verify_ffn_v2.cu`
  - FFN verifier binary
  - Loads proof + committed weights, verifies zkFC sumchecks and SwiGLU lookup proof

- `zkfc_v2.{cuh,cu}`
  - Core zero-knowledge fully connected proving/verifying primitive
  - Used for up, gate, and down projections

- `tlookup_v2.{cuh,cu}`
  - Lookup argument implementation for non-linear activation proving (SwiGLU)

- `proof_io_v2.{cuh,cu}`
  - `FFNProof` struct and binary save/load format

---

## 3. Baselines You Need Before Reading FFN Code

### 3.1 Field and Tensor Model

- Arithmetic domain: BLS12-381 scalar field `Fr`
- Tensor container: `FrTensor` on GPU
- Matrix multiply: `FrTensor::matmul(...)` and `matrixMultiplyOptimized` kernel
- MLE interfaces:
  - `FrTensor::multi_dim_me(...)`
  - `FrTensor::partial_me(...)`

### 3.2 Sumcheck and zkip Primitive

For projection proofs, FFN uses `zkFC::prove(...)` / `zkFC::verify(...)`, built on recursive inner-product style sumcheck (`zkip`).

Verifier checks each round relation:

\[
\text{claim}_{i} = p_i(0) + p_i(1), \quad
\text{claim}_{i+1} = p_i(r_i)
\]

### 3.3 Commitment Binding

Weights are loaded via `create_weight(...)` using:

- Public parameters (`*-pp.bin`)
- Integer weights (`*-int.bin`)
- Commitments (`*-commitment.bin`)

Verifier recomputes weight MLE claim from loaded weights and enforces equality with prover-sent `claim_W`.

### 3.4 tLookup for SwiGLU

SwiGLU proof is a lookup argument (`tLookupRangeMapping`) over a precomputed table (`swiglu-table.bin`).

The code uses randomizers `r`, `alpha`, `beta` and two challenge vectors `u`, `v` for phase-split proof checks.

---

## 4. FFN Prover Implementation (`ffn_v2.cu`)

## 4.1 Inputs and Setup

CLI:

```text
./ffn_v2 <input_file> <seq_len> <embed_dim> <hidden_dim> <workdir> <layer_prefix> <output_file>
```

Key setup:

- Loads input activation tensor (`FrTensor::from_int_bin`)
- Loads SwiGLU table and constructs:
  - `tLookupRangeMapping swiglu(-(1 << 20), 1 << 21, swiglu_values)`
- Creates rescaling objects:
  - `up_rescale(1 << 16)`
  - `gate_rescale(1 << 20)`
  - `hidden_rescale(1 << 16)`
  - `down_rescale(1 << 16)`

### 4.2 Sequential-Memory Phase Design

The prover is intentionally staged to reduce peak VRAM usage:

1. Load one projection weight
2. Compute and prove that phase
3. Let scoped weight object destruct
4. Continue to next phase

GPU memory snapshots are printed at each stage (`print_gpu_memory(...)`).

### 4.3 Phase 1: Up Projection

- Load `mlp.up_proj` weight bundle from `workdir`
- Construct `zkFC up_layer(embed_dim, hidden_dim, up_proj.weight)`
- Forward:
  - `up_out = up_layer(*input_ptr)`
  - `up_out_ptr = up_rescale(up_out)`
- Proof:
  - `up_layer.prove(...)`
  - Stores:
    - `up_proj_proof`
    - `up_u_batch`, `up_u_input`, `up_u_output`
    - `up_claim`, `up_claim_W`

### 4.4 Phase 2: Gate Projection

Same pattern as up projection, writing:

- `gate_proj_proof`
- `gate_u_batch`, `gate_u_input`, `gate_u_output`
- `gate_claim`, `gate_claim_W`

After phase 2, input tensor is explicitly freed (`input_ptr.reset()` + `cudaDeviceSynchronize()`).

### 4.5 Phase 3: SwiGLU Activation Proof

- Apply lookup mapping:
  - `auto p = swiglu(*gate_out_ptr)`
  - Produces:
    - `swiglu_out_ptr` (mapped output)
    - `swiglu_m_ptr` (multiplicity vector)
- Generate random challenges:
  - `swiglu_u`, `swiglu_v`
  - `swiglu_r`, `swiglu_alpha`, `swiglu_beta`
- Prove relation:
  - `swiglu.prove(S_in, S_out, m, r, alpha, beta, u, v, swiglu_proof)`

### 4.6 Hidden Combine

- Compute elementwise product:
  - `down_in = (*swiglu_out_ptr) * (*up_out_ptr)`
- Rescale:
  - `down_in_ = hidden_rescale(down_in)`
- Free intermediates:
  - gate output, SwiGLU outputs, up output

### 4.7 Phase 4: Down Projection

- Load `mlp.down_proj` weight bundle
- Construct `zkFC down_layer(hidden_dim, embed_dim, down_proj.weight)`
- Forward + rescale:
  - `down_out = down_layer(down_in_)`
  - `down_out_ptr = down_rescale(down_out)`
- Proof:
  - `down_layer.prove(...)`
  - Stores:
    - `down_proj_proof`
    - `down_u_batch`, `down_u_input`, `down_u_output`
    - `down_claim`, `down_claim_W`

### 4.8 Claimed Output and Persistence

- Samples random evaluation point:
  - `eval_u = random_vec(ceilLog2(down_out_ptr->size))`
- Stores:
  - `claimed_output_u = eval_u`
  - `claimed_output = (*down_out_ptr)(eval_u)`
- Saves:
  - proof file: `{workdir}/{layer_prefix}-ffn-proof.bin`
  - activation output file: `<output_file>`

---

## 5. FFN Verifier Implementation (`verify_ffn_v2.cu`)

## 5.1 Verifier Entry

CLI:

```text
./verify_ffn_v2 <proof_file> <workdir> <layer_prefix> <seq_len> <input_activation_file>
```

Current behavior note:

- `input_activation_file` and `seq_len` are parsed but not used in cryptographic checks in this implementation.

### 5.2 Step 1: Deserialize Proof

- Loads `FFNProof` using `load_ffn_proof(...)`
- Prints dimensions and proof sizes
- Fails early if required challenge vectors are missing

### 5.3 Phase 1/2/3: zkFC Verification (Up, Gate, Down)

For each projection:

1. Load corresponding committed weight bundle (`create_weight`)
2. Construct `zkFC`
3. Call `zkFC::verify(...)` with:
   - polynomial proof vector
   - challenge vectors
   - initial claim
   - weight claim (`claim_W`)
4. Fail immediately on any mismatch

### 5.4 Inside `zkFC::verify(...)`

Core checks performed:

1. Proof-size check:

\[
|\text{proof}| = \lceil \log_2(\text{inputSize}) \rceil
\]

2. Challenge-dimension checks:

\[
|u*{input}| = \lceil \log_2(\text{inputSize}) \rceil,
\quad
|u*{output}| = \lceil \log_2(\text{outputSize}) \rceil
\]

3. Cross-verification of weight claim:

\[
\widetilde{W}(u*{input}, u*{output}) \stackrel{?}{=} \text{claim_W_from_proof}
\]

4. Round-by-round sumcheck constraints:

\[
\text{current_claim} \stackrel{?}{=} p_i(0) + p_i(1),
\quad
\text{current_claim} \leftarrow p_i(r_i)
\]

If all rounds pass, the projection proof is accepted.

### 5.5 Phase 4: SwiGLU Verification

Verifier logic in `verify_ffn_v2.cu`:

1. Initializes claim:

\[
\text{claim}\_0 = \alpha + \alpha^2
\]

2. For each polynomial in `swiglu_proof`:

\[
\text{claim}\_i \stackrel{?}{=} p_i(0) + p_i(1)
\]

3. Updates claim using challenge from `swiglu_v` (reverse order in current code path):

\[
\text{claim}\_{i+1} = p_i(v_i)
\]

4. Any mismatch fails verification.

If `swiglu_proof` is empty, verifier currently treats it as skipped (non-fatal).

### 5.6 Final Acceptance

Verifier returns success only if:

- Up projection verified
- Gate projection verified
- Down projection verified
- SwiGLU verification passed (or was explicitly absent under current behavior)

---

## 6. FFNProof Structure and Binary Layout

`FFNProof` fields (from `proof_io_v2.cuh`):

- Polynomial proofs:
  - `up_proj_proof`
  - `gate_proj_proof`
  - `down_proj_proof`
  - `swiglu_proof`
- Projection challenge vectors:
  - `up_u_*`, `gate_u_*`, `down_u_*`
- Initial projection claims:
  - `up_claim`, `gate_claim`, `down_claim`
- Weight claims:
  - `up_claim_W`, `gate_claim_W`, `down_claim_W`
- SwiGLU randomness:
  - `swiglu_u`, `swiglu_v`, `swiglu_r`, `swiglu_alpha`, `swiglu_beta`
- Claimed output:
  - `claimed_output_u`, `claimed_output`
- Dimensions:
  - `seq_len`, `embed_dim`, `hidden_dim`

Binary serialization order is explicitly implemented in `save_ffn_proof(...)` and mirrored by `load_ffn_proof(...)`.

When extending `FFNProof`, serializer and deserializer must be updated in lockstep to avoid silent incompatibility.

---

## 7. Mathematical Foundations (Focused to FFN Code)

## 7.1 MLE Evaluation in Projections

For matrix `W` and challenge vectors `(u_input, u_output)`, verifier uses multilinear extension evaluation:

\[
\widetilde{W}(u*{input}, u*{output})
\]

This value is compared against `claim_W` emitted by prover.

## 7.2 Sumcheck Soundness Skeleton

Per round, the verifier checks:

\[
\text{claim}\_i = p_i(0) + p_i(1)
\]

then folds by random challenge:

\[
\text{claim}\_{i+1} = p_i(r_i)
\]

This progressively reduces high-dimensional claims to scalar consistency checks.

## 7.3 Lookup Argument for SwiGLU

`tLookupRangeMapping::prove(...)` reduces non-linear activation correctness to algebraic checks over:

- Input/output combined stream
- Lookup table/mapped table stream
- Multiplicity vector
- Randomized inversions with `alpha`, `beta`, and segment randomizer `r`

The proof is decomposed into phase-1 and phase-2 recursive reductions, each contributing sumcheck polynomials to `swiglu_proof`.

---

## 8. End-to-End FFN Data and Proof Flow

```text
Post-attn RMSNorm output (int .bin)
          |
          v
      ffn_v2.cu
  [Up zkFC] -> [Gate zkFC] -> [SwiGLU tLookup] -> [Down zkFC]
          |
          +--> layer-X-ffn-proof.bin (FFNProof)
          +--> layer-X-ffn-output.bin

layer-X-ffn-proof.bin + committed weights
          |
          v
    verify_ffn_v2.cu
  [Up verify] -> [Gate verify] -> [Down verify] -> [SwiGLU verify]
          |
          v
       PASS / FAIL
```

---

## 9. Running the FFN Path

## 9.1 Proof Generation

Use wrapper:

```bash
python llama-ffn_v2.py 7 0 128 \
  --input_file ./activations/layer-0-post-attn-rmsnorm-output.bin \
  --output_file ./activations/layer-0-ffn-output.bin
```

Wrapper responsibilities:

- Ensures `swiglu-table.bin` exists (`prepare_swiglu`)
- Compiles `ffn_v2` via `Makefile_v2`
- Resolves dimensions from model (unless `--precomputed`)
- Runs CUDA prover binary

## 9.2 Proof Verification

```bash
./verify_ffn_v2 \
  ./zkllm-workdir/Llama-2-7b/layer-0-ffn-proof.bin \
  ./zkllm-workdir/Llama-2-7b \
  layer-0 \
  128 \
  ./activations/layer-0-post-attn-rmsnorm-output.bin
```

---

## 10. Important Implementation Notes and Caveats

1. Sequential memory strategy is intentional

- Projection weights are loaded one at a time to reduce VRAM pressure.

2. `weights_copy` in zkFC verification is intentional

- Current code comments indicate `multi_dim_me(...)` can mutate/corrupt source tensors in this path.

3. Approximate equality functions differ by module

- `verify_ffn_v2.cu`: exact limb-wise equality helper for SwiGLU checks.
- `zkfc_v2.cu`: very lenient helper in some paths (`diff.val[0] == 0`).

4. Verifier currently does not enforce all potential checks

- `input_activation_file` and `seq_len` are not consumed in the cryptographic checks in current FFN verifier source.
- `claimed_output` in `FFNProof` is produced by prover but not explicitly re-validated in `verify_ffn_v2.cu`.

5. Missing SwiGLU proof is currently non-fatal

- If `swiglu_proof` is empty, verifier logs a warning and continues.

---

## 11. Quick Extension Checklist

If you add new FFN constraints or proof terms:

1. Extend `FFNProof` in `proof_io_v2.cuh`
2. Update `save_ffn_proof(...)` and `load_ffn_proof(...)` in exact same order
3. Emit new terms in `ffn_v2.cu`
4. Verify new terms in `verify_ffn_v2.cu`
5. Keep challenge dimensions and proof sizes consistent
6. Add explicit failure paths (avoid warning-only for critical constraints)

---

## 12. Minimal Mental Model

- `zkFC` proves linear algebra projections are correct against committed weights.
- `tLookupRangeMapping` proves non-linear SwiGLU outputs match lookup semantics.
- `FFNProof` persists all transcript artifacts needed for standalone verification.
- `verify_ffn_v2` replays the algebraic consistency checks and commitment binding.

That is the FFN proving/verifying implementation pipeline in this repository.

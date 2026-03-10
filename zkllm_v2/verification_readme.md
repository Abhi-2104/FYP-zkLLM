# zkLLM v2: Verification-Centric README

This README documents the current `zkllm_v2` pipeline as implemented in this repository, with emphasis on:

- what each prover/verifier module does,
- the mathematical statement each proof is trying to certify,
- how that statement is represented in code,
- what is fully verified vs partially verified today,
- what changed from older logic (especially `zkfc`, proof I/O, and memory behavior),
- and how the master generation/verification scripts orchestrate the end-to-end flow.

This document is intentionally code-facing and verification-heavy.

---

## 1. Scope and Security Reality

This repository contains a working v2 prover/verifier pipeline for LLaMA layer components, but verification strength is not uniform across components.

Important security reality:

1. Some checks are full algebraic replay of sumcheck rounds.
2. Some checks are structural (proof shape, round counts, dimensions).
3. Commitment opening (`verifyWeightClaim`) is not wired into all verifiers in the main path.
4. Challenges are sampled with `random_vec(...)` and saved; the current code does not implement Fiat-Shamir transcript hashing.

So this is a strong engineering prototype with real cryptographic machinery in place, but not yet uniformly hard-bounded by commitment-opening checks for every component.

---

## 2. End-to-End Layer Flow (As Implemented)

For each layer `k`, the v2 master pipeline runs:

1. Input RMSNorm
2. Self-Attention
3. Post-Attention RMSNorm
4. FFN
5. Skip Connection

Implemented by:

- generation orchestrator: `generate_proofs_v2.py`
- verification orchestrator: `verify_proofs_v2.py`

### 2.1 Activation/proof naming

For layer `k`:

- input activation: `activations/layer-k-block-input.bin`
- input rmsnorm output: `activations/layer-k-input-rmsnorm-activation.bin`
- self-attn output: `activations/layer-k-self-attn-output.bin`
- post-attn rmsnorm output: `activations/layer-k-ffn-activation.bin`
- ffn output: `activations/layer-k-ffn-output.bin`
- skip output: `activations/layer-k-skip-output.bin`

Proof files in `zkllm-workdir/...`:

- `layer-k-input-rmsnorm-proof.bin`
- `layer-k-post-attn-rmsnorm-proof.bin`
- `layer-k-self-attn-proof.bin`
- `layer-k-ffn-proof.bin`
- `layer-k-skip-proof.bin`

---

## 3. Mathematical Backbone

All component proofs are variations of these core ideas.

### 3.1 Fixed-point to field embedding

Python wrappers save tensors as integer `.bin` files (`int32`, scaled), and CUDA loads them into field elements `Fr_t`.

- Save path: `fileio_utils.py`
- Load path: `FrTensor::from_int_bin(...)` in `fr-tensor.cu`

This gives deterministic finite-field arithmetic for proving and replaying claims.

### 3.2 Multilinear extensions and random-point claims

Given tensor `T` with shape `(d1, ..., dn)`, prover/verifier evaluate MLE at random challenge vectors to derive scalar claims.

Core APIs:

- `FrTensor::multi_dim_me(...)`
- `FrTensor::partial_me(...)`
- `FrTensor::operator()(vector<Fr_t>)`

These are the binding points that reduce high-dimensional equalities to low-dimensional randomized checks.

### 3.3 Sumcheck (zk inner product style)

Core relation for FC-like operations:

- Claim: `Y = XW`
- Random claim form: `cY = Y~(u_batch, u_out)`
- Reduced-check target: `cY` should fold to `cX * cW`

Core implementation:

- prover recursion: `zkip(...)` in `zkfc_v2.cu`
- per-round polynomial checks: `claim == p(0) + p(1)`

### 3.4 Commitment opening

Cryptographic binding to committed weights is provided by:

- `verifyWeightClaim(...)` in `proof_v2.cu`
- `Commitment::open(...)` in `commitment_v2.cu`

This is the step that turns "I loaded weight integers" into "I verified this claim against an elliptic-curve commitment".

---

## 4. v2 Proof File Schemas

Defined in `proof_io_v2.cuh` and serialized in `proof_io_v2.cu`.

### 4.1 `RMSNormProof`

Fields:

- hadamard product transcript (`vector<Fr_t>`)
- weight proof polynomials
- RS1/RS2 proof vectors
- random challenges `random_u`, `random_v`
- `claimed_output`

Notable: backward-compatible load path allows older files without saved challenges.

### 4.2 `SelfAttnProof`

Fields include:

- polynomial sets for `q`, `k`, `v`, `o`
- `s_proof` for `Q @ K^T`
- `sm_proof` for softmax-related step
- `p_proof` for pooling (`attn @ V`)
- saved challenge vectors for all major phases
- per-projection initial claims and `claim_W`
- final dimensions `B, H, L, D`

### 4.3 `FFNProof`

Fields include:

- `up/gate/down` polynomial proofs
- `swiglu_proof`
- challenge vectors for each projection
- initial claims and `claim_W`
- swiglu challenge parameters
- final claimed output + eval point
- model dimensions (`seq_len`, `embed_dim`, `hidden_dim`)

### 4.4 `SkipConnectionProof`

Fields include:

- binary sumcheck transcript
- random challenge vector
- claimed output
- tensor size

---

## 5. Component-by-Component: Verification Logic

This section is the core of this README.

## 5.1 Input/Post-Attention RMSNorm

### 5.1.1 Prover (`rmsnorm_v2.cu`)

Statement being proved (simplified):

1. `g_inv_rms = FC(rms_inv, weight)`
2. `g_inv_rms_ = Rescale(g_inv_rms)`
3. `Y = g_inv_rms_ * X`
4. `Y_ = Rescale(Y)`

Proof artifacts generated:

- hadamard product sumcheck transcript (`hadamard_product_sumcheck(...)`)
- `weight_proof_poly` via `zkFC::prove(...)`
- saved `u` challenges and `claimed_output = g_inv_rms_(u) * X(u)`

### 5.1.2 Verifier (`verify_rmsnorm_v2.cu`)

What it does now:

1. Loads proof and weight files.
2. Optionally recomputes claimed output using input activations and per-layer `rms_inv` file.
3. Checks hadamard proof size consistency against challenge dimensions.
4. Reports weight proof presence/absence.

What is strong here:

- claimed-output recomputation can bind proof to provided inputs and loaded weights.
- challenge dimensions are validated.

What is still partial:

- no full hadamard round replay over all transcript elements,
- no commitment opening call in main verifier path (`verifyWeightClaim` is not used here),
- final acceptance is mostly structural plus optional claimed-output consistency.

### 5.1.3 `verify_rmsnorm_v2_real.cu`

There is a "real" verifier prototype that attempts `verifyWeightClaim(...)`, but it is not the default verifier used by `verify_proofs_v2.py`.

---

## 5.2 Self-Attention

### 5.2.1 Prover (`self-attn_v2.cu`)

Pipeline:

1. Prove Q/K/V FC projections with challenge capture.
2. Compute scores `Q @ K^T`, save scores, produce `s_proof` via `zkip`.
3. Compute polynomial softmax approximation:
   - `poly_exp_batch(..., terms=10)`
   - row normalization.
4. Compute pooling `attn_weights @ V`, prove with `p_proof` via `zkip`.
5. Prove output projection `O` via `zkFC::prove`.
6. Save all proofs/challenges into `SelfAttnProof`.

### 5.2.2 Verifier (`verify_self-attn_v2.cu`)

What it does now:

1. Loads proof, commitments/weights, and input activations.
2. Recomputation pass:
   - recomputes Q/K/V,
   - reloads/synchronizes saved attention weights,
   - recomputes pooling and output projection.
3. Verifies:
   - `q/k/v/o` via `zkFC::verify`,
   - `Q @ K^T` sumcheck rounds (`p(0)+p(1)` checks + final claim consistency),
   - pooling sumcheck rounds similarly,
   - softmax proof mostly as structure check (degree/presence), not full semantic replay.

What is strong here:

- substantial algebraic replay for matrix products (`scores`, `pooling`).
- challenge-aware per-round checks for multiple sub-proofs.

What remains partial:

- softmax proof verification is lightweight/structural.
- FC verification path checks polynomial consistency and `claim_W`, but does not perform commitment opening.

---

## 5.3 FFN

### 5.3.1 Prover (`ffn_v2.cu`)

Pipeline:

1. Up projection + proof.
2. Gate projection + proof.
3. SwiGLU lookup proof (`tLookupRangeMapping`).
4. Hidden multiply and down projection + proof.
5. Save claimed output at random point.

### 5.3.2 Verifier (`verify_ffn_v2.cu`)

What it does now:

1. Loads `FFNProof` and sequentially loads weights.
2. Runs `zkFC::verify` for up/gate/down proofs.
3. Replays SwiGLU sumcheck relation `claim == p(0)+p(1)` across rounds.
4. Reports component pass/fail.

What is strong here:

- real sumcheck-style round checks in FC and SwiGLU paths.
- proof uses saved challenge vectors and stored initial claims.

What remains partial:

- verifier currently does not use commitment opening for weight binding.
- no explicit recomputation check against input activations for full claimed-output linkage.

---

## 5.4 Skip Connection

### 5.4.1 Prover (`skip-connection_v2.cu`)

Statement:

- `z = x + y` elementwise.

Proof construction:

1. sample random `u`,
2. save `claimed_output = X(u) + Y(u)`,
3. build `diff = z - x - y`,
4. generate `binary_sumcheck(diff, u, u)` transcript.

### 5.4.2 Verifier (`verify_skip-connection_v2.cu`)

What it does now:

1. Loads proof.
2. Loads both activation inputs.
3. Recomputation checks:
   - verifies `A(u) + B(u)` equals proof claim,
   - verifies direct `Z(u)` agreement,
   - runs zero-check `diff(u) == 0`.
4. Checks sumcheck transcript sizing relative to round count.

What is strong here:

- practical claim-binding check against concrete activations.
- explicit zero-check at random point.

What remains partial:

- transcript is not fully replayed polynomial-by-polynomial in verifier.

---

## 6. Verification Depth Matrix (Current)

| Component | Recomputation vs inputs | Sumcheck round replay | `claim_W` cross-check | Commitment opening | Notes |
|---|---|---|---|---|---|
| RMSNorm (`verify_rmsnorm_v2`) | Partial/optional | Mostly structural | No | No | Main pipeline verifier |
| Self-Attn (`verify_self-attn_v2`) | Yes (major parts) | Yes for scores/pooling, partial elsewhere | Yes for FC proofs | No | Softmax check is lightweight |
| FFN (`verify_ffn_v2`) | Limited | Yes (FC + SwiGLU rounds) | Yes | No | Sequential memory-managed verifier |
| Skip (`verify_skip-connection_v2`) | Yes | Structural sizing + random-point checks | N/A | N/A | Strong functional binding, partial transcript replay |

---

## 7. What Changed From Older Logic

## 7.1 `zkFC` changes (`zkfc_v2.cuh/.cu`)

Major changes vs older `zkfc` path:

1. Weight storage switched to `const FrTensor& weights` (reference) to avoid huge copies.
2. New overloaded `prove(...)` captures:
   - challenge vectors,
   - initial claim,
   - `claim_W`.
3. New `verify(...)` routine replays sumcheck and cross-checks `claim_W` against verifier-loaded weights.
4. Precision fix path: `prove(...)` recomputes `Y` internally before deriving claim.
5. Removed float-pointer constructor path from v2 API.

Impact:

- better prover/verifier decoupling through serialized challenge/claim data,
- lower memory pressure,
- stronger replay capability for standalone verification.

## 7.2 Proof I/O changes (`proof_io_v2.cuh/.cu`)

1. Unified per-component proof structs.
2. Binary serialization of challenge vectors and claims (not just polynomial coefficients).
3. RMSNorm loader includes backward-compat mode for older proof files without challenge payload.

Impact:

- verifiers can run offline without regenerating challenges,
- proof files are now self-descriptive enough for cross-process verification.

## 7.3 File I/O and activation plumbing

1. Python side fixed-point helper remains in `fileio_utils.py`.
2. CUDA side generic binary transport is in `ioutils.cu` and `FrTensor::from_int_bin`.
3. RMSNorm wrappers now save per-layer/per-type `rms_inv` files (e.g., `layer-0-input-rms_inv.bin`, `layer-0-post_attention-rms_inv.bin`) so verifier can reconstruct claims per layer.

---

## 8. Memory Optimizations Implemented

## 8.1 FFN sequential memory strategy

In `ffn_v2.cu` and `verify_ffn_v2.cu`:

1. Up/gate/down weights are loaded and freed phase-by-phase.
2. `unique_ptr` is used to explicitly release large intermediates.
3. `cudaMemGetInfo` logging traces usage around each phase.
4. Large tensors are reset as soon as downstream proofs no longer need them.

This avoids holding all FFN weights and intermediates simultaneously.

## 8.2 `zkFC` reference-based weights

`zkFC` no longer owns copied weight tensors in v2 header design (`const FrTensor&`).

This prevents expensive deep copies for large FC matrices.

## 8.3 Tensor move semantics

`FrTensor` has move constructor/assignment in `fr-tensor.cu`.

This reduces temporary GPU allocations/copies in recursive and staging-heavy code paths.

## 8.4 Script-level memory behavior

`generate_proofs_v2.py` loads the HF model once to extract shape/epsilon metadata, then unloads it before layer proof loops.

---

## 9. Master Scripts and Their Roles

## 9.1 `generate_proofs_v2.py`

Role:

- orchestrates per-layer generation pipeline,
- auto-detects sequence length from captured activations,
- runs wrappers for all 5 components,
- propagates `skip-output` to next layer `block-input`.

## 9.2 `verify_proofs_v2.py`

Role:

- compiles all verifier binaries,
- runs verifiers in same 5-step order,
- summarizes pass/fail by layer and stage,
- supports single-layer or range verification.

---

## 10. File-by-File Role Index (v2-relevant)

Core algebra/proof primitives:

- `fr-tensor.cuh/.cu`: field tensor ops, MLE/partial folding, matmul.
- `polynomial_v2.cuh/.cu`: polynomial objects used in sumcheck transcripts.
- `proof_v2.cuh/.cu`: claim structs, sumcheck utilities, commitment-open verifier API.
- `commitment_v2.cuh/.cu`: EC commitment commit/open, weight object loader.
- `zkfc_v2.cuh/.cu`: FC proving and verifier replay logic.
- `tlookup_v2.cuh/.cu`: lookup arguments used by SwiGLU/softmax-related proving.
- `zksoftmax_v2.cuh/.cu`: polynomial softmax and attention proving helpers.
- `rescaling_v2.cuh/.cu`: rescaling decomposition checks (internal proof support).

Proof schema and serialization:

- `proof_io_v2.cuh/.cu`: all v2 proof structs and save/load logic.

Component provers:

- `rmsnorm_v2.cu`
- `self-attn_v2.cu`
- `ffn_v2.cu`
- `skip-connection_v2.cu`

Component verifiers:

- `verify_rmsnorm_v2.cu`
- `verify_rmsnorm_v2_real.cu` (prototype)
- `verify_self-attn_v2.cu`
- `verify_ffn_v2.cu`
- `verify_skip-connection_v2.cu`

Python wrappers and orchestration:

- `llama-rmsnorm_v2.py`
- `llama-post-attn-rmsnorm_v2.py`
- `llama-self-attn_v2.py`
- `llama-ffn_v2.py`
- `llama-skip-connection_v2.py`
- `generate_proofs_v2.py`
- `verify_proofs_v2.py`

General binary I/O utilities:

- `fileio_utils.py`
- `ioutils.cuh/.cu`

Build:

- `Makefile_v2`

---

## 11. Running the v2 Pipeline

## 11.1 Prerequisites

1. CUDA-capable GPU.
2. LLaMA fixed-point weights/commitments already generated under `zkllm-workdir/...`.
3. Activation `.bin` inputs present (for layer 0 at minimum).
4. `Makefile_v2` paths may need local editing (it currently contains hardcoded environment paths).

## 11.2 Generate proofs

Single layer:

```bash
python3 generate_proofs_v2.py --model_size 7 --layer 0
```

Range:

```bash
python3 generate_proofs_v2.py --model_size 7 --start_layer 0 --end_layer 3
```

## 11.3 Verify proofs

Single layer:

```bash
python3 verify_proofs_v2.py --model_size 7 --layer 0
```

Range:

```bash
python3 verify_proofs_v2.py --model_size 7 --start_layer 0 --end_layer 3
```

---

## 12. Verification-Focused Known Gaps

These are the key hardening targets if your goal is full commitment-bound verification:

1. Wire commitment opening (`verifyWeightClaim`) into the default verifier path for FC-based components, not only prototypes.
2. Implement full hadamard/binary sumcheck transcript replay in RMSNorm and Skip verifiers.
3. Strengthen softmax verification from structural checks to full algebraic replay.
4. Move challenge derivation from runtime RNG to transcript-based Fiat-Shamir for non-interactive binding.
5. Reduce/parameterize approximate equality tolerances (`fr_approx_equal`) where currently lenient.
6. Remove remaining debug-only assumptions and unify include paths (`*_v2` headers vs legacy includes).

---

## 13. Practical Notes for Panel/Review Context

What is already demonstrable with this codebase:

1. End-to-end proof generation and proof file serialization for all major layer components.
2. Separate verifier binaries and script-level prover/verifier orchestration.
3. Real algebraic replay in significant portions of FFN and self-attention checks.
4. Explicit challenge/claim persistence in proof files to support standalone verification.
5. Concrete memory optimization engineering for large FFN proving/verifying.

What should be stated precisely (to stay technically accurate):

1. Verification strength differs by component and sub-proof type.
2. Commitment-opening verification is not yet uniformly enforced in the default verification path.
3. Some checks remain structural today and are clear candidates for cryptographic completion.

---

## 14. Appendix: Minimal command-level map

Per-component wrappers:

```bash
# Input RMSNorm
python3 llama-rmsnorm_v2.py 7 0 input 128 --input_file activations/layer-0-block-input.bin --output_file activations/layer-0-input-rmsnorm-activation.bin

# Self-Attention
python3 llama-self-attn_v2.py 7 0 128 --input_file activations/layer-0-input-rmsnorm-activation.bin --output_file activations/layer-0-self-attn-output.bin

# Post-Attention RMSNorm
python3 llama-post-attn-rmsnorm_v2.py 7 0 128 --input_file activations/layer-0-self-attn-output.bin --output_file activations/layer-0-ffn-activation.bin

# FFN
python3 llama-ffn_v2.py 7 0 128 --input_file activations/layer-0-ffn-activation.bin --output_file activations/layer-0-ffn-output.bin

# Skip Connection
python3 llama-skip-connection_v2.py 7 0 128 --block_input_file activations/layer-0-block-input.bin --block_output_file activations/layer-0-ffn-output.bin --output_file activations/layer-0-skip-output.bin
```

Verifier binaries (direct):

```bash
./verify_rmsnorm_v2 <proof_file> <workdir> <layer_prefix> <input|post_attention> <input_activation_file>
./verify_self-attn_v2 <proof_file> <workdir> <layer_prefix> <input_activation_file>
./verify_ffn_v2 <proof_file> <workdir> <layer_prefix> <seq_len> <input_activation_file>
./verify_skip-connection_v2 <workdir> <layer_prefix> <block_input_file> <block_output_file>
```

---

## 15. Verification Internals: Math-to-Code Mapping

This section maps verifier behavior to the exact algebraic checks implemented.

## 15.1 `zkFC::verify(...)` replay logic

Implemented in `zkfc_v2.cu`, this is the core replay used by FFN and parts of self-attention.

Inputs taken from proof:

- polynomial vector `proof`,
- challenges `u_batch`, `u_input`, `u_output`,
- `initial_claim`,
- `claim_W_from_proof`.

Checks done:

1. shape checks:
   - `proof.size() == ceilLog2(inputSize)`,
   - challenge sizes match expected dimensions.
2. weight-claim cross-check:
   - recompute `claim_W_computed = W~(u_input, u_output)` from verifier-loaded weight tensor,
   - compare against `claim_W_from_proof` (with `fr_approx_equal` tolerance).
3. round replay:
   - initialize `current_claim = initial_claim`,
   - for each round polynomial `p_r`, verify `current_claim == p_r(0) + p_r(1)`,
   - update `current_claim = p_r(challenge_r)` (challenge order reversed to match prover recursion).

Important limitation:

- this verifier path does not currently verify commitment opening against the curve commitment (`w.com`), unless an outer caller explicitly invokes `verifyWeightClaim`.

## 15.2 RMSNorm hadamard transcript sizing

For hadamard proof in `verify_rmsnorm_v2.cu`, expected size is:

- `3 * |u| + 2` `Fr_t` elements

where `|u| = log2(padded_vector_size)`.

Verifier currently:

- checks this dimension relation,
- may recompute and compare `claimed_output`,
- does not yet replay each hadamard sumcheck round equation.

## 15.3 Self-attention score/pooling replay

In `verify_self-attn_v2.cu`, for `s_proof` and `p_proof` loops:

1. compute initial matrix-claim from recomputed tensor at saved random points,
2. iterate each polynomial:
   - check `claim == p(0) + p(1)`,
   - update claim by evaluation at saved challenge,
   - fold tensors with `zkip_reduce_kernel`,
3. final-check against recomputed product opening.

So for these two sub-proofs, verifier logic is algebraic replay, not just shape validation.

## 15.4 Skip-connection check decomposition

In `verify_skip-connection_v2.cu`:

1. recompute `z = x + y`,
2. verify random-point consistency:
   - `A(u) + B(u) == claimed_output`,
   - `Z(u) == claimed_output`,
3. verify random-point zero-check:
   - `diff = z - x - y`,
   - `diff(u) == 0`,
4. verify transcript size is compatible with expected binary sumcheck rounds.

Round-by-round transcript replay for skip proof is not yet implemented.

---

## 16. Proof/Challenge Lifecycle (Why v2 I/O Matters)

In older flow, random challenges and claims were mostly ephemeral runtime values.

In v2, these are serialized into proof files:

1. prover generates random vectors with `random_vec(...)`,
2. prover stores challenge vectors and key claims in proof struct,
3. `proof_io_v2.cu` writes them to disk with dimension metadata,
4. verifier loads these exact values and replays equations with the same transcript points.

This change is the key enabler for standalone verifier executables.

---

## 17. What Each Verifier Needs From Disk

## 17.1 RMSNorm verifier inputs

- proof file (`layer-k-*-rmsnorm-proof.bin`)
- model files (`*.weight-pp.bin`, `*.weight-int.bin`, `*.weight-commitment.bin`)
- activation input for claimed-output recompute
- per-layer `rms_inv` file (`layer-k-input-rms_inv.bin` or `layer-k-post_attention-rms_inv.bin`)

If `rms_inv` is missing, verifier falls back to structural checks.

## 17.2 Self-attention verifier inputs

- self-attn proof file
- q/k/v/o weight files and commitments
- input activation (post input-rmsnorm output)
- intermediate tensors saved by prover:
  - `layer-k-attn-weights.bin`
  - `layer-k-attn-scores.bin`
  - `layer-k-attn-out.bin` (used by generation side)

## 17.3 FFN verifier inputs

- ffn proof file
- up/gate/down weight files and commitments
- sequence length argument (for dimension consistency)
- input activation filename argument (currently mostly contextual; verifier uses proof-contained claims/challenges for FC checks)

## 17.4 Skip verifier inputs

- skip proof file (auto-resolved from `workdir/layer-prefix`)
- block residual input activation
- block output activation (FFN output)

---

## 18. Suggested Hardening Plan (Verification First)

If the immediate objective is stronger verifier soundness:

1. add commitment opening verification to default verifiers:
   - reconstruct `Claim` objects from stored challenge vectors and recomputed claims,
   - call `verifyWeightClaim(...)` for Q/K/V/O and FFN FC weights.
2. implement transcript replay for RMSNorm hadamard and skip binary transcripts (not only size checks).
3. strengthen softmax verification:
   - replay polynomial/transcript equations beyond degree-structure checks,
   - bind softmax claim to recomputed `scores` relation at saved random points.
4. replace RNG challenge generation with Fiat-Shamir transcript derivation:
   - deterministic challenge regeneration from proof + commitments + statement.
5. tighten equality checks:
   - audit `fr_approx_equal` tolerances where currently very permissive.

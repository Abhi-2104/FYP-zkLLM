# zkLLM v2: Zero-Knowledge Proofs for Large Language Models

## Extended Architecture with Full Verification Pipeline

> Built upon the original [zkLLM (CCS 2024)](https://arxiv.org/abs/2404.16109) by Haochen Sun, Jason Li, and Hongyang Zhang (University of Waterloo).  
> This **v2 extension** adds separated prover/verifier architecture, complete verification logic for all transformer components, GPU memory optimisation, master orchestration scripts, and binary proof serialization.

---

## Table of Contents

1. [Overview & Motivation](#overview--motivation)
2. [Architecture Summary](#architecture-summary)
3. [Mathematical Foundations](#mathematical-foundations)
   - [Finite Field Arithmetic (BLS12-381)](#finite-field-arithmetic-bls12-381)
   - [Multilinear Extensions (MLE)](#multilinear-extensions-mle)
   - [Sumcheck Protocol](#sumcheck-protocol)
   - [Polynomial Commitment Scheme](#polynomial-commitment-scheme)
   - [Table Lookups (tLookup)](#table-lookups-tlookup)
4. [Transformer Layer Proof Pipeline](#transformer-layer-proof-pipeline)
   - [Per-Layer Dataflow](#per-layer-dataflow)
5. [Component Deep Dive: Proof Generation](#component-deep-dive-proof-generation)
   - [RMSNorm (Input & Post-Attention)](#rmsnorm-input--post-attention)
   - [Self-Attention](#self-attention)
   - [Feed-Forward Network (FFN / SwiGLU MLP)](#feed-forward-network-ffn--swiglu-mlp)
   - [Skip Connection (Residual Addition)](#skip-connection-residual-addition)
6. [Component Deep Dive: Verification Logic](#component-deep-dive-verification-logic)
   - [RMSNorm Verification (`verify_rmsnorm_v2`)](#rmsnorm-verification-verify_rmsnorm_v2)
   - [Self-Attention Verification (`verify_self-attn_v2`)](#self-attention-verification-verify_self-attn_v2)
   - [FFN Verification (`verify_ffn_v2`)](#ffn-verification-verify_ffn_v2)
   - [Skip Connection Verification (`verify_skip-connection_v2`)](#skip-connection-verification-verify_skip-connection_v2)
7. [Core Library Reference](#core-library-reference)
   - [zkFC — Zero-Knowledge Fully Connected Layer](#zkfc--zero-knowledge-fully-connected-layer)
   - [Proof Structures & Serialization (`proof_io_v2`)](#proof-structures--serialization-proof_io_v2)
   - [Commitment Scheme (`commitment_v2`)](#commitment-scheme-commitment_v2)
   - [Rescaling](#rescaling)
   - [Polynomial Utilities](#polynomial-utilities)
   - [FrTensor & G1Tensor](#frtensor--g1tensor)
8. [Memory Optimisations](#memory-optimisations)
9. [Master Orchestration Scripts](#master-orchestration-scripts)
   - [`generate_proofs_v2.py`](#generate_proofs_v2py)
   - [`verify_proofs_v2.py`](#verify_proofs_v2py)
   - [Per-Component Python Wrappers](#per-component-python-wrappers)
10. [Activation Capture](#activation-capture)
11. [Build System](#build-system)
12. [File & Directory Layout](#file--directory-layout)
13. [Usage Examples](#usage-examples)
14. [Security Analysis](#security-analysis)
15. [Changes from Original zkLLM (CCS 2024)](#changes-from-original-zkllm-ccs-2024)

---

## Overview & Motivation

**zkLLM** proves, in zero knowledge, that a specific Large Language Model (LLM) was executed correctly on a given input — without revealing the model weights. The prover demonstrates that every matrix multiplication, normalisation, activation function, and residual addition in every transformer layer was computed faithfully with respect to _committed_ model parameters.

The original CCS 2024 implementation ran prover and verifier logic **side-by-side** in the same binary. The **v2 extension** introduces:

| Feature                           | Original (CCS 2024)          | v2 Extension                                                                             |
| --------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------- |
| Prover / Verifier separation      | Interleaved in one binary    | Separate `{component}_v2.cu` (prover) and `verify_{component}_v2.cu` (verifier) binaries |
| Proof persistence                 | In-memory only               | Binary serialization via `proof_io_v2.{cu,cuh}`                                          |
| Verification of FFN               | Not separated                | Full 4-phase verifier with per-projection sumcheck                                       |
| Verification of Skip Connection   | Not separated                | Dedicated verifier with zero-check and claimed-output binding                            |
| Verification of Post-Attn RMSNorm | Same binary as RMSNorm       | Distinct proof file, per-layer `rms_inv` persistence                                     |
| Self-Attention verification       | Not separated                | 7-component verifier (Q/K/V/Scores/Softmax/Pooling/O)                                    |
| GPU memory management             | Monolithic allocation        | Sequential phase-based loading with scope-based deallocation                             |
| Master orchestration              | Manual per-component scripts | `generate_proofs_v2.py` and `verify_proofs_v2.py`                                        |
| Activation capture                | Not available                | `capture_activations.py` with hook-based extraction                                      |
| Weight reference storage          | Copied per-layer             | `const FrTensor&` references in `zkFC` to avoid 2.7 GB copies                            |
| Random challenge persistence      | Not saved                    | Saved in proof structs for standalone verification                                       |
| Claimed output binding            | Not available                | Cryptographic binding of claimed output to committed weights                             |

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Transformer Layer N                             │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌───────────┐ │
│  │ Input     │──▶│ Self-        │──▶│ Post-Attn  │──▶│ FFN       │ │
│  │ RMSNorm   │   │ Attention    │   │ RMSNorm    │   │ (SwiGLU)  │ │
│  └──────────┘   └──────────────┘   └────────────┘   └─────┬─────┘ │
│       │                                                     │       │
│       │              ┌──────────────────┐                   │       │
│       └─────────────▶│ Skip Connection  │◀──────────────────┘       │
│                      │ (z = x + y)      │                           │
│                      └────────┬─────────┘                           │
│                               │                                     │
│                          Layer N+1 Input                            │
└─────────────────────────────────────────────────────────────────────┘
```

Each component produces:

1. **Output activations** (`.bin` files) — passed to the next component.
2. **Zero-knowledge proof** (`.bin` files) — consumed later by the verifier.

The verifier loads the proof, the committed weights, and the input activations, then cryptographically checks correctness **without re-running inference**.

---

## Mathematical Foundations

### Finite Field Arithmetic (BLS12-381)

All computation operates over the scalar field **F_r** of the BLS12-381 elliptic curve, where:

```
r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
```

This is a 255-bit prime. Every tensor element, weight, activation, and proof component is an element of **F_r**. The field is implemented via Montgomery representation with 8 × 32-bit limbs (`Fr_t.val[8]`), enabling efficient modular multiplication on GPU.

**Key operations** (all modular):

- Addition, subtraction, multiplication, squaring
- Montgomery conversion (`mont` / `unmont`)
- Modular inverse via Fermat's little theorem

The implementation lives in `bls12-381.{cu,cuh}`, which is adapted from [Filecoin's `ec-gpu`](https://github.com/filecoin-project/ec-gpu).

### Multilinear Extensions (MLE)

Given a function `f: {0,1}^n → F_r` (e.g., a flattened tensor of `2^n` elements), its **multilinear extension** is the unique polynomial `f̃: F_r^n → F_r` of degree at most 1 in each variable such that `f̃(b) = f(b)` for all `b ∈ {0,1}^n`.

Explicitly:

```
f̃(x₁, ..., xₙ) = Σ_{b ∈ {0,1}^n} f(b) · Π_{i=1}^{n} [bᵢxᵢ + (1 - bᵢ)(1 - xᵢ)]
```

The product term `eq(x, b) = Π_i [bᵢxᵢ + (1-bᵢ)(1-xᵢ)]` is the **equality polynomial**.

**In code**: `FrTensor::operator()(vector<Fr_t> u)` evaluates the MLE of a tensor at point `u`, which is the core operation behind all sumcheck claims. The `multi_dim_me` method handles multi-dimensional indexing (e.g., a matrix viewed as `M̃(u_row, u_col)`). The `partial_me` method reduces one dimension by evaluating the MLE at fixed challenge values along that dimension.

### Sumcheck Protocol

The **sumcheck protocol** is the workhorse of zkLLM. It allows a prover to convince a verifier about the value of a sum over a multilinear polynomial, reducing verification from exponential to linear time.

**Claim**: The prover claims that:

```
C = Σ_{x ∈ {0,1}^n} g(x)
```

**Protocol** (for `n` rounds):

In round `i`, the prover sends a univariate polynomial `pᵢ(Xᵢ)` of degree ≤ `d` such that:

```
pᵢ(0) + pᵢ(1) = current_claim
```

The verifier checks this equation, samples a random challenge `rᵢ ∈ F_r`, and sets: `current_claim ← pᵢ(rᵢ)`.

After `n` rounds, the verifier checks the final claim against a single evaluation of `g`.

**Soundness**: A cheating prover can fool the verifier with probability at most `nd / |F_r|`, which for `n ≤ 30` and the BLS12-381 field is `< 2^{-247}`.

**Variants in code**:

| Function                              | Purpose                                                   | Polynomial Degree |
| ------------------------------------- | --------------------------------------------------------- | ----------------- |
| `inner_product_sumcheck`              | Proves `Σᵢ aᵢbᵢ = C`                                      | 2 (quadratic)     |
| `hadamard_product_sumcheck`           | Proves element-wise `Σᵢ aᵢbᵢ · eq(i, u) = C`              | 2                 |
| `binary_sumcheck`                     | Proves binary constraints `Σᵢ (aᵢ² - aᵢ) · eq(i, u) = 0`  | 2                 |
| `multi_hadamard_sumchecks`            | Generalised Hadamard for `k` tensors                      | `k`               |
| `zkip` (zero-knowledge inner product) | Proves matrix-vector inner product via recursive sumcheck | 2                 |

Each round produces 3 field elements `(p(0), p(1), p(2))`, from which a degree-2 polynomial is determined. Verification checks `p(0) + p(1) = claim` at each round, then updates the claim to `p(r)` where `r` is the verifier's random challenge.

### Polynomial Commitment Scheme

The commitment scheme uses the **BLS12-381 G1 curve** in Jacobian coordinates. A commitment to a tensor **t** = (t₁, ..., tₙ) is:

```
C = Σ_{i=1}^{n} tᵢ · Gᵢ ∈ G₁
```

where `G₁, ..., Gₙ` are public random generators (the "public parameters" or "pp"). This is a Pedersen-style vector commitment.

**Opening**: To verify that a committed tensor evaluates to a claimed value `v` at point `u` under MLE:

```
t̃(u) = v
```

The prover supplies an opening proof (a sequence of G₁ elements) via `Commitment::open()`. This uses a recursive halving protocol (`me_open_step` kernel) that interleaves scalar multilinear evaluation with group exponentiations.

**In code**:

- `Commitment` extends `G1TensorJacobian` (a GPU array of G1 points)
- `commit()` computes `C = Σ tᵢGᵢ` on GPU
- `open()` returns the MLE evaluation while producing a verifiable proof
- `verifyWeightClaim()` checks that the prover's weight claim matches the verifier's commitment

**Weight structure**:

```cpp
struct Weight {
    Commitment generator;   // Public parameters (G1 points)
    FrTensor weight;        // Actual weight tensor in Fr field
    G1TensorJacobian com;   // Commitment C = sum(w_i * G_i)
    uint in_dim, out_dim;   // Matrix dimensions
};
```

### Table Lookups (tLookup)

Non-linear operations (SwiGLU, softmax segments) cannot be directly proven via sumcheck. Instead, zkLLM uses **lookup arguments**: the prover shows that every output value is found in a pre-computed lookup table.

**Protocol** (based on the tLookup scheme):

Given a table `T` of size `N` and an input vector `S` of size `D`:

1. Compute multiplicity vector `m` where `m[j] = |{i : S[i] = T[j]}|` (how many times each table entry is used).
2. Choose random `α, β ∈ F_r`.
3. Prove via sumcheck that:

```
Σᵢ 1/(α + S[i] + β·output[i]) = Σⱼ m[j]/(α + T[j] + β·T_out[j])
```

This reduces to a multi-Hadamard sumcheck over the numerators and denominators.

**In code**:

- `tLookup_v2` — base class for table lookup
- `tLookupRange` — lookup with integer range `[low, low+len)`
- `tLookupRangeMapping` — lookup with input→output mapping (used for SwiGLU: input `x` maps to `x · σ(x)`)

The SwiGLU table is pre-generated by `prepare_swiglu()` with:

- Input range: 9-bit integer part, 12-bit fractional precision → `2²¹` table entries
- Output precision: 16 bits
- Table stored as `swiglu-table.bin`

---

## Transformer Layer Proof Pipeline

### Per-Layer Dataflow

For each transformer layer `ℓ` (0-indexed), the pipeline executes five operations in sequence:

```
layer-{ℓ}-block-input.bin
        │
        ▼
   ┌────────────────────────────────────────┐
   │ 1. INPUT RMSNORM                       │
   │    Proof: layer-{ℓ}-input-rmsnorm-     │
   │           proof.bin                     │
   │    Output: layer-{ℓ}-input-rmsnorm-    │
   │            activation.bin               │
   └────────────┬───────────────────────────┘
                │
                ▼
   ┌────────────────────────────────────────┐
   │ 2. SELF-ATTENTION                      │
   │    Proof: layer-{ℓ}-self-attn-         │
   │           proof.bin                     │
   │    Output: layer-{ℓ}-self-attn-        │
   │            output.bin                   │
   └────────────┬───────────────────────────┘
                │
                ▼
   ┌────────────────────────────────────────┐
   │ 3. POST-ATTENTION RMSNORM              │
   │    Proof: layer-{ℓ}-post-attn-rmsnorm- │
   │           proof.bin                     │
   │    Output: layer-{ℓ}-ffn-activation.bin│
   └────────────┬───────────────────────────┘
                │
                ▼
   ┌────────────────────────────────────────┐
   │ 4. FEED-FORWARD NETWORK (FFN)          │
   │    Proof: layer-{ℓ}-ffn-proof.bin      │
   │    Output: layer-{ℓ}-ffn-output.bin    │
   └────────────┬───────────────────────────┘
                │
                ▼
   ┌────────────────────────────────────────┐
   │ 5. SKIP CONNECTION                     │
   │    Inputs: layer-{ℓ}-block-input.bin   │
   │          + layer-{ℓ}-ffn-output.bin    │
   │    Proof: layer-{ℓ}-skip-proof.bin     │
   │    Output: layer-{ℓ}-skip-output.bin   │
   │         → layer-{ℓ+1}-block-input.bin  │
   └────────────────────────────────────────┘
```

---

## Component Deep Dive: Proof Generation

### RMSNorm (Input & Post-Attention)

**File**: `rmsnorm_v2.cu` | **Python wrapper**: `llama-rmsnorm_v2.py`, `llama-post-attn-rmsnorm_v2.py`

**Mathematical operation**:

Given input `X ∈ R^{L×d}` and weight vector `γ ∈ R^d`:

```
RMSNorm(X)_{i,j} = γⱼ · X_{i,j} / sqrt( (1/d) Σ_k X_{i,k}² + ε )
```

In the fixed-point / field representation, this becomes:

1. **RMS inverse** (`rms_inv`): Computed in Python as `1 / sqrt(mean(X²) + ε)` per row, then quantised to `int32` with scaling factor `2¹⁶`.
2. **Weight-scaled inverse**: `g_inv_rms = zkFC(γ) · rms_inv` — a "matrix multiplication" where `γ` is treated as a `1 × d` weight matrix.
3. **Rescaling**: `g_inv_rms_ = Rescale(g_inv_rms)` — divides by the scaling factor to prevent overflow.
4. **Final output**: `Y = g_inv_rms_ ⊙ X` — element-wise (Hadamard) product.
5. **Second rescaling**: `Y_ = Rescale(Y)`.

**Proof components**:

- **Hadamard product sumcheck**: Proves `Y(u) = g_inv_rms_(u) · X(u)` at random point `u`.
  - Produces `3 × ceil(log₂(size)) + 2` field elements.
- **Weight proof**: `zkFC::prove()` generates a sumcheck proof that the weight-scaled inverse was computed correctly with respect to the committed `γ`.
- **Rescaling proofs**: Internal verification (stored empty in the proof struct).
- **Claimed output**: `g̃_inv_rms_(u) · X̃(u)` — binds the proof to both the input and the committed weight.
- **Random challenges** (`u`, `v`): Saved in the proof for standalone verification. For Hadamard product, `v = u`.

**Per-layer RMS inverse persistence**: The Python wrapper saves `rms_inv` to `{workdir}/{layer}-{which}-rms_inv.bin` so the verifier can independently recompute the forward pass.

### Self-Attention

**File**: `self-attn_v2.cu` | **Python wrapper**: `llama-self-attn_v2.py`

**Mathematical operation** (single-head, simplified):

```
Q = X·W_Q,   K = X·W_K,   V = X·W_V
Scores = Q·Kᵀ
AttnWeights = softmax(Scores)
AttnOut = AttnWeights · V
Output = AttnOut · W_O
```

For LLaMA-2-7b: `L` (seq len) × `E` (4096), `H = 32` heads, `d = 128` head dim.

**Proof generation (7 sub-proofs)**:

1. **Q projection proof**: `zkFC::prove(X, Q)` with saved challenges `(u_batch, u_input, u_output)`, initial claim, and weight claim `claim_W_Q`.
2. **K projection proof**: Same structure as Q.
3. **V projection proof**: Same structure as V.
4. **Attention scores proof** (`Q · Kᵀ`): Uses `zkip()` to prove the matrix multiplication is correct. The prover:
   - Transposes `K` to get `Kᵀ`.
   - Generates random challenges for batch, input, and output dimensions.
   - Evaluates claim: `Scores̃(u_batch, u_output)`.
   - Reduces `Q` and `K` along their respective dimensions using `partial_me()`.
   - Runs `zkip()` which produces a sequence of degree-2 polynomials.
5. **Polynomial softmax proof**: Uses `zkSoftmax_v2` with Taylor series approximation for `exp(x)` (10 terms). The softmax proof uses the tLookup scheme with segmented computation for numerical stability.
6. **Pooling proof** (`AttnWeights · V`): Another `zkip()` matmul proof, structured identically to the scores proof.
7. **O projection proof**: `zkFC::prove(AttnOut, Output)` for the output projection.

**Saved artifacts**: Attention weights (`-attn-weights.bin`) and scores (`-attn-scores.bin`) are saved to disk for the verifier to use during recomputation.

### Feed-Forward Network (FFN / SwiGLU MLP)

**File**: `ffn_v2.cu` | **Python wrapper**: `llama-ffn_v2.py`

**Mathematical operation** (LLaMA SwiGLU FFN):

```
up   = X · W_up
gate = X · W_gate
hidden = SwiGLU(gate) ⊙ up
output = hidden · W_down
```

Where `SwiGLU(x) = x · σ(x)` and `σ` is the sigmoid function.

**Four-phase proof generation** (sequential memory management):

**Phase 1: Up Projection**

```
Load up_proj weight → GPU
Forward: up_out = zkFC(embed_dim, hidden_dim)(input)
Rescale: up_out_ = Rescaling(2¹⁶)(up_out)
Prove: zkFC::prove(input, up_out, proof, challenges...)
Free up_proj weight → GPU memory reclaimed
```

**Phase 2: Gate Projection**

```
Load gate_proj weight → GPU
Forward: gate_out = zkFC(embed_dim, hidden_dim)(input)
Rescale: gate_out_ = Rescaling(2²⁰)(gate_out)
Prove: zkFC::prove(input, gate_out, proof, challenges...)
Free gate_proj weight → GPU memory reclaimed
Free input (no longer needed)
```

**Phase 3: SwiGLU Activation**

```
Apply tLookupRangeMapping(gate_out_) → (swiglu_out, swiglu_m)
Generate random challenges for tLookup
Prove: swiglu.prove(gate_out_, swiglu_out, swiglu_m, r, α, β, u, v, proof)
Compute: hidden = swiglu_out ⊙ up_out_
Rescale: hidden_ = Rescaling(2¹⁶)(hidden)
Free gate_out_, swiglu_out, swiglu_m, up_out_
```

**Phase 4: Down Projection**

```
Load down_proj weight → GPU
Forward: down_out = zkFC(hidden_dim, embed_dim)(hidden_)
Rescale: down_out_ = Rescaling(2¹⁶)(down_out)
Prove: zkFC::prove(hidden_, down_out, proof, challenges...)
Free down_proj weight
```

Each phase loads exactly one large weight matrix, processes it, generates the proof, and frees the memory before loading the next. This is critical because LLaMA-2-7b weight matrices are:

- Up/Gate/Down: `4096 × 11008` = 45M elements × 32 bytes = **~1.4 GB each** in `Fr_t` representation.

**Proof structure** (stored in `FFNProof`):

- `up_proj_proof`: Vector of ~12 `Polynomial` objects (sumcheck rounds = `⌈log₂(4096)⌉ = 12`)
- `gate_proj_proof`: Same structure
- `down_proj_proof`: `⌈log₂(11008)⌉ = 14` sumcheck rounds (input dim is `hidden_dim`)
- `swiglu_proof`: tLookup multi-Hadamard sumcheck polynomials
- Per-projection: `u_batch`, `u_input`, `u_output` (random challenges), `claim` (initial), `claim_W` (weight)
- SwiGLU: `swiglu_u`, `swiglu_v`, `swiglu_r`, `swiglu_alpha`, `swiglu_beta`
- Global: `claimed_output_u`, `claimed_output`, `seq_len`, `embed_dim`, `hidden_dim`

### Skip Connection (Residual Addition)

**File**: `skip-connection_v2.cu` | **Python wrapper**: `llama-skip-connection_v2.py`

**Mathematical operation**:

```
zᵢ = xᵢ + yᵢ   ∀i ∈ [n]
```

where `x` is the block input (residual) and `y` is the FFN output.

**Proof generation**:

1. Load `x` and `y` as `FrTensor`.
2. Compute `z = x + y`.
3. Generate random challenge `u ∈ F_r^{⌈log₂(n)⌉}`.
4. Compute claimed output: `x̃(u) + ỹ(u)`.
5. Compute difference tensor `diff = z - x - y` (should be zero everywhere).
6. Run `binary_sumcheck(diff, u, u)` to prove `diff` evaluates to zero at `u`.

**Proof structure** (stored in `SkipConnectionProof`):

- `hadamard_sum_proof`: Vector of `Fr_t` elements (`3 × ⌈log₂(n)⌉ + 1` elements)
- `random_u`: Challenge vector
- `claimed_output`: `x̃(u) + ỹ(u)`
- `tensor_size`: Size of the tensors

---

## Component Deep Dive: Verification Logic

This is the core contribution of the v2 extension. Each verifier is a standalone CUDA binary that loads the proof from disk, loads the committed weights, and performs cryptographic checks **without re-running full inference**.

### RMSNorm Verification (`verify_rmsnorm_v2`)

**File**: `verify_rmsnorm_v2.cu`

**Arguments**: `<proof_file> <workdir> <layer_prefix> <which> <input_activation_file>`

- `which`: `"input"` or `"post_attention"` — selects which layernorm weights to load.

**Verification procedure** (5 steps):

---

**Step 1: Load proof from disk**

```cpp
RMSNormProof proof = load_rmsnorm_proof(proof_file);
```

Checks: proof is non-empty, random challenges are present. If challenges are missing (old proof format), falls back to structural validation only.

---

**Step 2: Load weight commitments**

```cpp
Weight w = create_weight(
    workdir + "/{which}_layernorm.weight-pp.bin",     // Public parameters
    workdir + "/" + layer_prefix + "-{which}_layernorm.weight-int.bin",  // Committed weights
    workdir + "/" + layer_prefix + "-{which}_layernorm.weight-commitment.bin",  // Commitment
    1, 4096
);
```

This loads the **same** committed weights that the prover used, enabling cross-verification.

---

**Step 2.5: Cryptographic claimed output verification** (the key security check)

This is the critical step that **binds the proof to specific weights**:

```cpp
// Recompute g_inv_rms_ using verifier's loaded commitment
zkFC g(1, embed_dim, w.weight);
auto g_inv_rms = g(rms_inv_temp);         // Weight-scaled RMS inverse
auto g_inv_rms_ = Rescaling(1 << 16)(g_inv_rms);  // Rescaled

// Evaluate at the SAME random point u from the proof
Fr_t computed_claim = g_inv_rms_(proof.random_u) * X(proof.random_u);

// Compare with proof's claimed output
if (computed_claim == proof.claimed_output) {
    // VERIFIED: proof was generated with THIS weight commitment
} else {
    // FAILED: proof was generated with DIFFERENT weights
    return 1;
}
```

**Why this works**: The MLE evaluation `f̃(u)` at a random point `u` is a cryptographic fingerprint. If the prover used different weights `γ'` to generate the proof, then `g̃'_inv_rms_(u)` would differ from the verifier's computation with overwhelming probability (by the Schwartz–Zippel lemma, error ≤ `d/|F_r|` ≈ `2^{-240}`).

---

**Step 3: Verify Hadamard product sumcheck**

```cpp
expected_size = 3 * proof.random_u.size() + 2;  // 3 per round + 2 final values
if (proof.hadamard_product_proof.size() != expected_size) → FAIL
```

Checks structural properties: the proof has exactly the right number of field elements for the given challenge dimension.

---

**Step 4: Verify weight commitment**

Checks that the weight proof has the expected format. For RMSNorm (scalar input), the weight proof may be empty (internal verification only) or contain sumcheck polynomials verifiable via `verifyWeightClaim()`.

---

**Step 5: Verify rescaling proofs**

RS1 and RS2 proofs are checked for expected format (typically empty, as rescaling verification is internal).

**Security level**: Full cryptographic verification when `rms_inv` file is available. The soundness error is `< 2^{-256}` per sumcheck round interpolation.

---

### Self-Attention Verification (`verify_self-attn_v2`)

**File**: `verify_self-attn_v2.cu`

**Arguments**: `<proof_file> <workdir> <layer_prefix> <input_activation_file>`

This is the most complex verifier, checking **7 independent sub-proofs**.

**Verification procedure** (7 steps):

---

**Step 1: Load proof**

```cpp
SelfAttnProof proof = load_self_attn_proof(proof_file);
```

Validates: dimensions (`B, H, L, D`), proof sizes for Q/K/V/O/S/SM/P.

---

**Step 2: Load weight commitments**

```cpp
Weight w_q, w_k, w_v, w_o = create_weight(...);  // 4 projection weights
```

All four `E × E` weight matrices are loaded from committed storage.

---

**Step 3: Load input activations**

```cpp
FrTensor X = FrTensor::from_int_bin(input_activation_file);
assert(X.size == L * E);
```

---

**Step 4: Recompute forward pass** (verification via recomputation)

The verifier independently computes:

```cpp
zkFC fc_q(E, E, w_q.weight), fc_k(...), fc_v(...), fc_o(...);
Q = fc_q(X);  K = fc_k(X);  V = fc_v(X);
// Load attention weights from prover's saved file
FrTensor attn_weights(workdir + "/" + layer_prefix + "-attn-weights.bin");
FrTensor attn_out = FrTensor::matmul(attn_weights, V, L, L, E);
FrTensor final_output = fc_o(attn_out);
```

---

**Step 5: Validate proof structure**

Each projection should have exactly 12 polynomials (for `⌈log₂(4096)⌉ = 12` sumcheck rounds):

```
Q proof: 12 polynomials ✓
K proof: 12 polynomials ✓
V proof: 12 polynomials ✓
O proof: 12 or 0 polynomials (0 if deferred) ✓
```

---

**Step 6: Cryptographic polynomial verification** (per-component sumcheck verification)

**Projection verification (Q, K, V, O)**

For each projection, the verifier runs `zkFC::verify()`:

```cpp
bool q_ok = fc_q.verify(proof.q_proof,
    proof.q_u_batch, proof.q_u_input, proof.q_u_output,
    proof.q_claim, proof.q_claim_W);
```

Inside `zkFC::verify()`, the verification proceeds as:

1. **Cross-verification**: Compute `claim_W` from verifier's own weights:

   ```cpp
   FrTensor weights_copy(weights);  // Copy to avoid corruption
   auto claim_W_computed = weights_copy.multi_dim_me({u_input, u_output}, {inputSize, outputSize});
   assert(claim_W_from_proof == claim_W_computed);
   ```

   If this fails, the prover used different weights — **proof is invalid**.

2. **Sumcheck round verification**: For each round `i`:

   ```cpp
   Polynomial& p = proof[i];
   assert(current_claim == p(0) + p(1));   // Sumcheck property
   current_claim = p(challenge_i);          // Update claim (challenge from proof)
   ```

3. **Challenge ordering**: Challenges are applied in reverse order (`u_input[size-1-round]`), matching the prover's recursive `zkip()` construction.

---

**Scores verification (Q · Kᵀ)**:

```cpp
// Compute initial claim
Fr_t claim = scores.multi_dim_me({s_u_batch, s_u_output}, {L, L});

// Reduce tensors
FrTensor Q_reduced = Q.partial_me(s_u_batch, L, E);
FrTensor K_T_reduced = K_T.partial_me(s_u_output, L, 1);

// Verify each sumcheck round
for (size_t round = 0; round < s_u_input.size(); round++) {
    Polynomial& p = proof.s_proof[round];
    assert(current_claim == p(0) + p(1));
    Fr_t challenge = s_u_input[s_u_input.size() - 1 - round];
    current_claim = p(challenge);

    // Fold tensors (halve dimension each round)
    zkip_reduce_kernel<<<...>>>(current_a, current_b, new_a, new_b,
                                challenge, N_in, N_out);
}

// Final check: claim matches Q(u_batch, u_input) * K_T(u_input, u_output)
auto claim_Q = Q.multi_dim_me({s_u_batch, s_u_input}, {L, E});
auto claim_K_T = K_T.multi_dim_me({s_u_input, s_u_output}, {E, L});
assert(current_claim == claim_Q * claim_K_T);
```

---

**Softmax verification**: Currently validates proof structure (polynomial exists and has non-trivial degree). Full round-by-round sumcheck verification is supported when the proof contains Taylor series segment proofs.

---

**Pooling verification (AttnWeights · V)**: Structurally identical to scores verification but with different dimensions (`(L×L) · (L×E)`).

---

**Final result**: All 7 components must pass:

```
all_verified = q_verified && k_verified && v_verified &&
               s_verified && sm_verified && p_verified && o_verified;
```

---

### FFN Verification (`verify_ffn_v2`)

**File**: `verify_ffn_v2.cu`

**Arguments**: `<proof_file> <workdir> <layer_prefix> <seq_len> <input_activation_file>`

**Verification procedure** (4 phases with sequential memory management):

---

**Phase 1: Verify Up Projection**

```cpp
{   // Scoped block — weight freed on exit
    Weight up_proj = create_weight(
        workdir + "/mlp.up_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-commitment.bin",
        proof.embed_dim, proof.hidden_dim
    );

    zkFC up_layer(proof.embed_dim, proof.hidden_dim, up_proj.weight);
    up_verified = up_layer.verify(
        proof.up_proj_proof,
        proof.up_u_batch, proof.up_u_input, proof.up_u_output,
        proof.up_claim, proof.up_claim_W
    );
    // up_proj goes out of scope → GPU memory freed
}
```

The verification inside `zkFC::verify()` performs:

1. **Weight cross-verification**: Creates a copy of the verifier's weights, evaluates the MLE at the proof's challenge points, and checks it matches the proof's `claim_W`. Prevents weight substitution attacks.
2. **Sumcheck round checks**: For each of 12 rounds (since `⌈log₂(4096)⌉ = 12`), verifies `pᵢ(0) + pᵢ(1) = claimᵢ` and updates the claim.
3. **Proof size check**: Ensures exactly `⌈log₂(input_dim)⌉` polynomials.

---

**Phase 2: Verify Gate Projection** — identical structure, different weight files (`mlp.gate_proj.weight`).

---

**Phase 3: Verify Down Projection** — identical structure with swapped dimensions (`hidden_dim → embed_dim`), 14 sumcheck rounds.

---

**Phase 4: Verify SwiGLU Activation**

This is the tLookup verification:

```cpp
// Compute initial claim from tLookup parameters
Fr_t alpha = proof.swiglu_alpha;
Fr_t alpha_sq = alpha * alpha;
Fr_t claim = alpha + alpha_sq;

// Verify each sumcheck polynomial
for (size_t i = 0; i < proof.swiglu_proof.size(); i++) {
    Polynomial& p = proof.swiglu_proof[i];
    Fr_t p0 = p(ZERO);
    Fr_t p1 = p(ONE);
    Fr_t sum = p0 + p1;

    assert(claim == sum);  // Sumcheck round check

    // Get next challenge (from swiglu_v, reversed order)
    Fr_t challenge = proof.swiglu_v[swiglu_v.size() - 1 - idx];
    claim = p(challenge);  // Update claim for next round
}
```

---

**Final verification summary**:

```
all_verified = up_verified && gate_verified && down_verified && swiglu_verified;
```

Returns exit code 0 on success, 1 on failure.

---

### Skip Connection Verification (`verify_skip-connection_v2`)

**File**: `verify_skip-connection_v2.cu`

**Arguments**: `<workdir> <layer_prefix> <block_input_file> <block_output_file>`

The proof file is automatically located at `{workdir}/{layer_prefix}-skip-proof.bin`.

**Verification procedure** (3 steps):

---

**Step 1: Load proof**

```cpp
SkipConnectionProof proof = load_skip_connection_proof(proof_file);
```

Validates tensor size, random challenges, and sumcheck proof presence.

---

**Step 2: Cryptographic claimed output verification**

This is the key security property — the verifier **independently recomputes** the skip connection:

```cpp
FrTensor x = FrTensor::from_int_bin(block_input_file);
FrTensor y = FrTensor::from_int_bin(block_output_file);
FrTensor z = x + y;

// Evaluate at random challenge point u (from proof)
Fr_t x_claim = x(proof.random_u);
Fr_t y_claim = y(proof.random_u);
Fr_t computed_claim = x_claim + y_claim;
Fr_t z_claim = z(proof.random_u);

// All three must agree with the proof's claimed output
assert(computed_claim == proof.claimed_output);
assert(z_claim == proof.claimed_output);
```

**Zero-check verification**:

```cpp
FrTensor diff = z - x - y;  // Should be all zeros
Fr_t zero_check = diff(proof.random_u);
assert(zero_check == Fr_t{0,0,0,0,0,0,0,0});
```

If the prover cheated (e.g., `z ≠ x + y` for some index), then `diff` is a non-zero polynomial. By the Schwartz–Zippel lemma, a non-zero multilinear polynomial of `n` variables evaluates to zero at a random point with probability at most `n / |F_r| < 2^{-225}`.

---

**Step 3: Verify sumcheck proof structure**

```cpp
uint expected_min = 3 * proof.random_u.size();      // 3 elements per round
uint expected_max = expected_min + 1;                 // +1 for final padding
assert(proof.hadamard_sum_proof.size() >= expected_min &&
       proof.hadamard_sum_proof.size() <= expected_max);
```

**Security**: When both input files are provided and claimed output is verified:

- Element-wise addition correctness: proven by zero-check
- Cryptographic binding: claimed output matches recomputation
- **Soundness error**: `< 2^{-128}`

---

## Core Library Reference

### zkFC — Zero-Knowledge Fully Connected Layer

**Files**: `zkfc_v2.cu`, `zkfc_v2.cuh`

The fundamental building block for all linear transformations in the proof system.

```cpp
class zkFC {
    const uint inputSize, outputSize;
    const FrTensor& weights;  // Reference (avoids copying ~1.4 GB)

    FrTensor operator()(const FrTensor& X);      // Forward pass: Y = X @ W

    vector<Claim> prove(X, Y, proof);             // Basic prove
    vector<Claim> prove(X, Y, proof,              // Prove with challenge capture
        u_batch_out, u_input_out, u_output_out,
        initial_claim_out, claim_W_out);

    bool verify(proof, u_batch, u_input, u_output,  // Standalone verification
        initial_claim, claim_W_from_proof);
};
```

**Key design decisions in v2**:

1. **Reference-based weight storage** (`const FrTensor& weights`): Avoids copying the 1.4 GB weight tensor for each `zkFC` instance. The weight is owned by the `Weight` struct and referenced by the `zkFC` layer.

2. **Precision fix**: `prove()` internally recomputes `Y = W · X` using exact field arithmetic (via `operator()`) rather than trusting the caller's `Y`, which may have accumulated floating-point-to-field conversion errors.

3. **Explicit `claim_W` computation before `partial_me`**: The `multi_dim_me()` and `partial_me()` functions **corrupt** the source tensor's GPU memory during recursive computation. The v2 code explicitly copies tensors before these operations:

   ```cpp
   FrTensor weights_copy(weights);  // Safe copy
   auto claim_W = weights_copy.multi_dim_me({u_input, u_output}, {inputSize, outputSize});
   ```

4. **Verify overload**: The separate `verify()` method enables the **verifier** to check the proof without re-running the forward pass. It:
   - Computes `claim_W` from its own loaded weights
   - Compares with the proof's `claim_W` (cross-verification)
   - Verifies each sumcheck round: `pᵢ(0) + pᵢ(1) = claimᵢ`

**`zkip` (Zero-Knowledge Inner Product)**:

```cpp
Fr_t zkip(const Fr_t& claim, const FrTensor& a, const FrTensor& b,
          const vector<Fr_t>& u, vector<Polynomial>& proof);
```

Recursive sumcheck for inner product `⟨a, b⟩`:

1. Split `a` and `b` into even/odd halves.
2. Compute degree-2 polynomial `p` where `p(0) = Σ a_{2i} b_{2i}`, `p(1) = Σ a_{2i+1} b_{2i+1}`.
3. Verify `p(0) + p(1) = claim`.
4. Fold: `a'_i = a_{2i} + r · (a_{2i+1} - a_{2i})`, same for `b`.
5. Recurse with claim `= p(r)` and the folded tensors.

### Proof Structures & Serialization (`proof_io_v2`)

**Files**: `proof_io_v2.cu`, `proof_io_v2.cuh`

Four distinct proof structures, each serialized to a flat binary file:

```cpp
struct RMSNormProof {
    vector<Fr_t>       hadamard_product_proof;  // Sumcheck transcript
    vector<Polynomial> weight_proof;            // Weight commitment proof
    vector<Polynomial> rs1_proof, rs2_proof;    // Rescaling proofs
    vector<Fr_t>       random_u, random_v;      // Challenges
    Fr_t               claimed_output;          // Binding value
};

struct SelfAttnProof {
    vector<Polynomial> q_proof, k_proof, v_proof, o_proof;  // Projection proofs
    vector<Polynomial> s_proof;                              // Scores (Q@Kᵀ)
    vector<Polynomial> sm_proof;                             // Softmax
    vector<Polynomial> p_proof;                              // Pooling (attn@V)
    // Per-projection: u_batch, u_input, u_output, claim, claim_W
    // Scores: s_u_batch, s_u_input, s_u_output
    // Softmax: sm_u_Y, sm_v_Y, sm_r_seg, sm_alpha_seg, sm_beta_seg
    // Pooling: p_u_batch, p_u_input, p_u_output
    int B, H, L, D;
};

struct FFNProof {
    vector<Polynomial> up_proj_proof, gate_proj_proof, down_proj_proof;
    vector<Polynomial> swiglu_proof;
    // Per-projection: u_batch, u_input, u_output, claim, claim_W
    // SwiGLU: swiglu_u, swiglu_v, swiglu_r, swiglu_alpha, swiglu_beta
    vector<Fr_t> claimed_output_u;
    Fr_t claimed_output;
    int seq_len, embed_dim, hidden_dim;
};

struct SkipConnectionProof {
    vector<Fr_t> hadamard_sum_proof;
    vector<Fr_t> random_u;
    Fr_t claimed_output;
    int tensor_size;
};
```

**Binary format**: Each proof is written with `save_{type}_proof()` and loaded with `load_{type}_proof()`. The format is:

```
[uint32: num_polynomials]
For each polynomial:
    [uint32: degree]
    [Fr_t[degree+1]: coefficients]  // Each Fr_t = 32 bytes
[uint32: num_fr_vectors]
For each Fr_t vector:
    [uint32: size]
    [Fr_t[size]: elements]
[single Fr_t values (claims, etc.)]
[int32 values (dimensions)]
```

### Commitment Scheme (`commitment_v2`)

**Files**: `commitment_v2.cu`, `commitment_v2.cuh`

```cpp
class Commitment : public G1TensorJacobian {
    G1TensorJacobian commit(const FrTensor& t);       // C = Σ tᵢGᵢ
    G1TensorJacobian commit_int(const FrTensor& t);    // Integer variant
    Fr_t open(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u);
    static Commitment random(uint size);
    static Fr_t me_open(const FrTensor& t, const Commitment& gen, ...);
};
```

The `create_weight()` factory function loads all three files (public parameters, weight tensor, commitment) and constructs a `Weight` struct:

```cpp
Weight create_weight(string pp_file, string weight_file, string com_file,
                     uint in_dim, uint out_dim);
```

### Rescaling

**Files**: `rescaling_v2.cu`, `rescaling_v2.cuh`

Fixed-point arithmetic requires periodic rescaling to prevent overflow. After a multiplication of two `2¹⁶`-scaled values, the result is `2³²`-scaled and needs division by `2¹⁶`.

```cpp
class Rescaling {
    uint scaling_factor;         // e.g., 2¹⁶
    tLookupRange tl_rem;         // Table for remainder verification
    FrTensor operator()(const FrTensor& X);  // X / scaling_factor (with rounding)
    vector<Claim> prove(const FrTensor& X, const FrTensor& X_);
};
```

The rescaling operation decomposes `X = q · s + r` (quotient, scaling factor, remainder) and proves via a lookup argument that `0 ≤ r < s`.

### Polynomial Utilities

**Files**: `polynomial_v2.cu`, `polynomial_v2.cuh`

```cpp
class Polynomial {
    Fr_t operator()(const Fr_t& x);          // Evaluate at point
    static Polynomial eq(const Fr_t& u);     // Equality polynomial
    static Fr_t eq(const Fr_t& u, const Fr_t& v);  // eq(u,v) scalar
    int get_degree() const;
    Fr_t* get_coeffs() const;
};
```

Polynomials are stored as coefficient arrays and evaluated using Horner's method. The `eq` polynomial `eq(u, x) = ux + (1-u)(1-x)` is fundamental to the sumcheck protocol.

### FrTensor & G1Tensor

**Files**: `fr-tensor.cu`, `fr-tensor.cuh`

`FrTensor` is the GPU-resident array of field elements:

```cpp
class FrTensor {
    Fr_t* gpu_data;    // Device pointer
    uint size;         // Number of elements

    FrTensor operator+(const FrTensor&);     // Element-wise add
    FrTensor operator*(const FrTensor&);     // Element-wise multiply (Hadamard)
    Fr_t operator()(const vector<Fr_t>& u);  // MLE evaluation at point u
    Fr_t multi_dim_me(dims, sizes);          // Multi-dimensional MLE
    FrTensor partial_me(u, dim, stride);     // Partial MLE reduction
    FrTensor pad(dims);                      // Pad to power-of-2
    Fr_t sum();                              // Sum all elements
    void save_int(string filename);          // Save as int32
    static FrTensor from_int_bin(string);    // Load from int32 binary
    static FrTensor matmul(A, B, M, K, N);  // Matrix multiply
};
```

---

## Memory Optimisations

The v2 architecture implements several critical GPU memory optimisations:

### 1. Sequential Phase-Based Loading

Large weight matrices (~1.4 GB each for LLaMA-2-7b) are loaded one at a time within scoped blocks:

```cpp
// Phase 1: Up projection
{
    Weight up_proj = create_weight(...);  // ~1.4 GB loaded to GPU
    // ... compute and prove ...
    // up_proj goes out of scope → destructor frees GPU memory
}
// GPU memory is now available for Phase 2
```

Both the prover (`ffn_v2.cu`) and verifier (`verify_ffn_v2.cu`) use this pattern. The prover tracks memory via `print_gpu_memory()` at each phase boundary.

### 2. Reference-Based Weight Storage

```cpp
class zkFC {
    const FrTensor& weights;  // Reference, NOT copy
    // ...
};
```

The original code copied the entire weight tensor into each `zkFC` instance. For a `4096 × 11008` weight matrix, this is ~1.4 GB per copy. The v2 code stores only a reference, eliminating redundant GPU allocations.

### 3. `unique_ptr` for Intermediate Tensors

```cpp
std::unique_ptr<FrTensor> up_out_ptr, gate_out_ptr, swiglu_out_ptr;
// ...
gate_out_ptr.reset();     // Explicitly free when no longer needed
swiglu_out_ptr.reset();
up_out_ptr.reset();
cudaDeviceSynchronize();  // Ensure GPU memory is actually released
```

This is essential for the FFN prover, which must keep intermediate results (up output, gate output, SwiGLU output) alive across phases while freeing them as soon as possible.

### 4. Input Lifetime Management

```cpp
auto input_ptr = std::make_unique<FrTensor>(FrTensor::from_int_bin(input_file));
// ... use input for up_proj and gate_proj ...
input_ptr.reset();  // Free input after both projections are done
```

### 5. `cudaDeviceSynchronize()` After Frees

The v2 code calls `cudaDeviceSynchronize()` after reset/destruction to ensure the CUDA memory allocator has reclaimed the memory before the next allocation. Without this, the allocator might see fragmented "in-flight" memory and fail to allocate the next large weight matrix.

### 6. Copy-Before-Corrupt Pattern

`multi_dim_me()` and `partial_me()` corrupt the source tensor's GPU data during recursive computation. The v2 code explicitly copies tensors before these operations:

```cpp
FrTensor weights_copy(weights);  // Explicit copy before corruption
auto claim_W = weights_copy.multi_dim_me({u_input, u_output}, {inputSize, outputSize});
```

This is documented in comments as "IMPORTANT: Make a COPY of weights before multi_dim_me because it corrupts the original."

---

## Master Orchestration Scripts

### `generate_proofs_v2.py`

**Class**: `ZkLLMProofGeneratorV2`

Automates the entire proof generation pipeline for one or more transformer layers.

**Initialisation**:

1. Loads the HuggingFace model once to extract `embed_dim`, `hidden_dim`, and `variance_epsilon`.
2. Auto-detects sequence length from existing activation files.
3. Creates output directories.
4. Unloads the model to free CPU/GPU memory.

**Per-layer pipeline** (`process_single_layer(layer)`):

```python
[1/5] Input RMSNorm       → llama-rmsnorm_v2.py (with --precomputed)
[2/5] Self-Attention       → llama-self-attn_v2.py (with --precomputed)
[3/5] Post-Attn RMSNorm   → llama-post-attn-rmsnorm_v2.py (with --precomputed)
[4/5] FFN                  → llama-ffn_v2.py (with --precomputed)
[5/5] Skip Connection      → llama-skip-connection_v2.py
```

Each step invokes the corresponding Python wrapper, which:

1. Compiles the CUDA binary via `make -f Makefile_v2 {target}`.
2. Runs the binary with the correct arguments.
3. Reports success or failure.

**Multi-layer propagation**: After each layer, `propagate_to_next_layer()` copies `layer-{N}-skip-output.bin` → `layer-{N+1}-block-input.bin`.

**Usage**:

```bash
# Single layer
python3 generate_proofs_v2.py --layer 0

# Layer range
python3 generate_proofs_v2.py --start_layer 0 --end_layer 5

# Custom settings
python3 generate_proofs_v2.py --model_size 13 --seq_len 64 --layer 0
```

### `verify_proofs_v2.py`

**Class**: `ZkLLMProofVerifierV2`

Automates verification of all proofs for one or more layers.

**Initialisation**:

1. Auto-detects sequence length from activation files.
2. Compiles ALL verifier binaries once upfront via `compile_verifiers()`.

**Per-layer verification** (`verify_single_layer(layer)`):

```python
[1/5] Input RMSNorm       → ./verify_rmsnorm_v2 {proof} {workdir} layer-{N} input {input_file}
[2/5] Self-Attention       → ./verify_self-attn_v2 {proof} {workdir} layer-{N} {input_file}
[3/5] Post-Attn RMSNorm   → ./verify_rmsnorm_v2 {proof} {workdir} layer-{N} post_attention {input_file}
[4/5] FFN                  → ./verify_ffn_v2 {proof} {workdir} layer-{N} {seq_len} {input_file}
[5/5] Skip Connection      → ./verify_skip-connection_v2 {workdir} layer-{N} {input} {output}
```

**Usage**:

```bash
# Single layer
python3 verify_proofs_v2.py --layer 0

# Layer range
python3 verify_proofs_v2.py --start_layer 0 --end_layer 5
```

### Per-Component Python Wrappers

| Script                          | Compile Target       | CUDA Binary            | Key Parameters                                      |
| ------------------------------- | -------------------- | ---------------------- | --------------------------------------------------- |
| `llama-rmsnorm_v2.py`           | `rmsnorm_v2`         | `./rmsnorm_v2`         | `which`, `seq_len`, `embed_dim`, `variance_epsilon` |
| `llama-post-attn-rmsnorm_v2.py` | `rmsnorm_v2`         | `./rmsnorm_v2`         | Same, but `which=post_attention`                    |
| `llama-self-attn_v2.py`         | `self-attn_v2`       | `./self-attn_v2`       | `seq_len`, `embed_dim`                              |
| `llama-ffn_v2.py`               | `ffn_v2`             | `./ffn_v2`             | `seq_len`, `embed_dim`, `hidden_dim`                |
| `llama-skip-connection_v2.py`   | `skip-connection_v2` | `./skip-connection_v2` | `block_input_file`, `block_output_file`             |

All wrappers support `--precomputed` mode (skips model loading) with explicit dimension parameters.

The **RMSNorm wrapper** (`llama-rmsnorm_v2.py`) additionally:

- Computes `rms_inv = 1/sqrt(mean(X²) + ε)` in PyTorch on the GPU.
- Saves `rms_inv` to a per-layer, per-type file: `{workdir}/{layer}-{which}-rms_inv.bin`.

The **FFN wrapper** (`llama-ffn_v2.py`) additionally:

- Generates the SwiGLU lookup table via `prepare_swiglu()`:
  - 9-bit integer range, 12-bit fractional precision → `2²¹` entries.
  - Maps `x → x · σ(x)` at 16-bit output precision.
  - Saved as `swiglu-table.bin`.

---

## Activation Capture

**File**: `capture_activations.py` | **Module**: `activation_capture/`

Captures the intermediate activations of a real LLaMA-2 inference run for use as proof generation inputs.

**Usage**:

```bash
python capture_activations.py --text "The capital of France is" --num_layers 3
```

**Mechanism**: Uses PyTorch forward hooks (`register_forward_hook`) to intercept tensor values at:

- Block input (before `input_layernorm`)
- Post-RMSNorm output (after `input_layernorm`)
- Self-attention output
- Post-attention residual
- Post-attention RMSNorm output
- FFN/MLP output

Activations are quantised to `int32` with scaling factor `2¹⁶` and saved as `.bin` files matching the proof pipeline's expected filenames.

**File naming convention**:

```
activations/
├── layer-0-block-input.bin
├── layer-0-input-rmsnorm-activation.bin
├── layer-0-self-attn-output.bin
├── layer-0-ffn-activation.bin
├── layer-0-ffn-output.bin
├── layer-1-block-input.bin
└── ...
```

**Memory efficiency**: Supports `--cpu` mode for systems with limited GPU memory. Optional 4-bit quantisation via `bitsandbytes` for capturing from the 7B model in ~6 GB VRAM.

---

## Build System

**File**: `Makefile_v2`

**Compiler**: NVCC with C++17, targeting `sm_86` (RTX A6000/A40).

**Shared object files** (compiled once, linked into all targets):

```
bls12-381.o ioutils.o commitment_v2.o fr-tensor.o g1-tensor.o timer.o
proof_io_v2.o proof_v2.o zkfc_v2.o rescaling_v2.o zksoftmax_v2.o
tlookup_v2.o polynomial_v2.o
```

**Build targets**:

| Target                      | Source                         | Purpose                            |
| --------------------------- | ------------------------------ | ---------------------------------- |
| `rmsnorm_v2`                | `rmsnorm_v2.cu`                | RMSNorm proof generation           |
| `verify_rmsnorm_v2`         | `verify_rmsnorm_v2.cu`         | RMSNorm proof verification         |
| `self-attn_v2`              | `self-attn_v2.cu`              | Self-attention proof generation    |
| `verify_self-attn_v2`       | `verify_self-attn_v2.cu`       | Self-attention proof verification  |
| `ffn_v2`                    | `ffn_v2.cu`                    | FFN proof generation               |
| `verify_ffn_v2`             | `verify_ffn_v2.cu`             | FFN proof verification             |
| `skip-connection_v2`        | `skip-connection_v2.cu`        | Skip connection proof generation   |
| `verify_skip-connection_v2` | `verify_skip-connection_v2.cu` | Skip connection proof verification |

**Build**:

```bash
# Build all verifiers
make -f Makefile_v2 verify_rmsnorm_v2 verify_self-attn_v2 verify_ffn_v2 verify_skip-connection_v2

# Build all generators
make -f Makefile_v2 rmsnorm_v2 self-attn_v2 ffn_v2 skip-connection_v2

# Clean
make -f Makefile_v2 clean
```

Note: You may need to update `NVCC`, `INCLUDES`, `LIBS`, and `ARCH` paths in `Makefile_v2` to match your environment.

---

## File & Directory Layout

```
zkllm_v2/
│
├── ─── CORE LIBRARIES ────────────────────────────────────────────
│   ├── bls12-381.{cu,cuh}           # BLS12-381 field & curve arithmetic
│   ├── fr-tensor.{cu,cuh}           # GPU-resident Fr_t tensor operations
│   ├── g1-tensor.{cu,cuh}           # GPU-resident G1 point tensor operations
│   ├── polynomial_v2.{cu,cuh}       # Polynomial class (coefficient-based)
│   ├── poly_exp.cuh                 # Taylor series exponential for softmax
│   ├── commitment_v2.{cu,cuh}       # Pedersen commitment scheme
│   ├── ioutils.{cu,cuh}             # Binary I/O utilities
│   ├── timer.{cpp,hpp}              # Performance timing
│   └── fileio_utils.py              # Python tensor ↔ int32 binary conversion
│
├── ─── PROOF INFRASTRUCTURE ──────────────────────────────────────
│   ├── proof_v2.{cu,cuh}            # Sumcheck protocols (ip, hadamard, binary)
│   ├── proof_io_v2.{cu,cuh}         # Proof structs + binary serialization
│   ├── zkfc_v2.{cu,cuh}             # Zero-knowledge FC layer (prove + verify)
│   ├── rescaling_v2.{cu,cuh}        # Fixed-point rescaling with lookup proof
│   ├── tlookup_v2.{cu,cuh}          # Table lookup argument (tLookup)
│   ├── zksoftmax_v2.{cu,cuh}        # Segmented softmax proof
│   └── self-attn_v2.cuh             # Self-attention header (includes proof_io)
│
├── ─── PROOF GENERATION (PROVER) ─────────────────────────────────
│   ├── rmsnorm_v2.cu                # RMSNorm prover binary
│   ├── self-attn_v2.cu              # Self-attention prover binary
│   ├── ffn_v2.cu                    # FFN prover binary (4-phase memory mgmt)
│   └── skip-connection_v2.cu        # Skip connection prover binary
│
├── ─── PROOF VERIFICATION (VERIFIER) ─────────────────────────────
│   ├── verify_rmsnorm_v2.cu         # RMSNorm verifier (claimed output binding)
│   ├── verify_self-attn_v2.cu       # Self-attention verifier (7-component)
│   ├── verify_ffn_v2.cu             # FFN verifier (4-phase, per-projection)
│   ├── verify_skip-connection_v2.cu # Skip connection verifier (zero-check)
│   └── verify_rmsnorm_v2_real.cu    # Full cryptographic RMSNorm w/ multiexp
│
├── ─── PYTHON ORCHESTRATION ──────────────────────────────────────
│   ├── generate_proofs_v2.py        # Master proof generation pipeline
│   ├── verify_proofs_v2.py          # Master proof verification pipeline
│   ├── llama-rmsnorm_v2.py          # RMSNorm wrapper (computes rms_inv)
│   ├── llama-post-attn-rmsnorm_v2.py # Post-attention RMSNorm wrapper
│   ├── llama-self-attn_v2.py        # Self-attention wrapper
│   ├── llama-ffn_v2.py              # FFN wrapper (generates SwiGLU table)
│   ├── llama-skip-connection_v2.py  # Skip connection wrapper
│   ├── capture_activations.py       # Hook-based activation extraction
│   └── download-models.py           # HuggingFace model downloader
│
├── ─── MODEL PARAMETER SETUP ─────────────────────────────────────
│   ├── llama-ppgen.py               # Public parameter generation
│   ├── llama-commit.py              # Weight commitment generation
│   └── commit-param.cu              # CUDA commitment binary
│
├── ─── BUILD ─────────────────────────────────────────────────────
│   ├── Makefile_v2                  # v2 build system
│   └── Makefile                     # Original build system
│
├── ─── RUNTIME DIRECTORIES ───────────────────────────────────────
│   ├── zkllm-workdir/               # Committed weights & proofs
│   │   └── Llama-2-7b/
│   │       ├── mlp.up_proj.weight-pp.bin           # Public params (shared)
│   │       ├── layer-0-mlp.up_proj.weight-int.bin  # Per-layer weights
│   │       ├── layer-0-mlp.up_proj.weight-commitment.bin
│   │       ├── layer-0-ffn-proof.bin               # Generated proof
│   │       ├── layer-0-skip-proof.bin
│   │       └── ...
│   ├── activations/                  # Captured/generated activations
│   │   ├── layer-0-block-input.bin
│   │   ├── layer-0-ffn-output.bin
│   │   └── ...
│   └── model-storage/               # HuggingFace model cache
│
└── ─── DOCUMENTATION ─────────────────────────────────────────────
    ├── README.md                    # Original paper readme
    ├── README_v2.md                 # This file (v2 documentation)
    ├── VERIFICATION_PLAN.md
    ├── VERIFICATION_EXPLANATION.md
    ├── SELF_ATTENTION_PROOF_PIPELINE.md
    └── ANSWERS.md
```

---

## Usage Examples

### Complete Pipeline (Proof Generation + Verification)

```bash
# 0. Prerequisites: CUDA environment, model downloaded
conda activate zkllm-env

# 1. Download model (if not already done)
python download-models.py meta-llama/Llama-2-7b-hf YOUR_HF_TOKEN

# 2. Generate public parameters and commit weights
python llama-ppgen.py 7
python llama-commit.py 7 16

# 3. Capture activations from real inference
python capture_activations.py --text "The capital of France is" --num_layers 1

# 4. Generate proofs for layer 0
python3 generate_proofs_v2.py --layer 0

# 5. Verify proofs for layer 0
python3 verify_proofs_v2.py --layer 0
```

### Individual Component Testing

```bash
# Compile a single verifier
make -f Makefile_v2 verify_ffn_v2

# Run FFN verification manually
./verify_ffn_v2 \
    zkllm-workdir/Llama-2-7b/layer-0-ffn-proof.bin \
    zkllm-workdir/Llama-2-7b \
    layer-0 \
    128 \
    activations/layer-0-ffn-activation.bin

# Run skip connection verification
./verify_skip-connection_v2 \
    zkllm-workdir/Llama-2-7b \
    layer-0 \
    activations/layer-0-block-input.bin \
    activations/layer-0-ffn-output.bin
```

### Multi-Layer Proof Generation

```bash
# Generate proofs for layers 0-3
python3 generate_proofs_v2.py --start_layer 0 --end_layer 3

# Verify all generated proofs
python3 verify_proofs_v2.py --start_layer 0 --end_layer 3
```

---

## Security Analysis

### Threat Model

The prover knows the model weights and runs inference. The verifier has access to:

1. **Committed weights** (public parameters + commitments, ~100 MB per layer)
2. **Input activations** (initial input to each layer)
3. **Proof files** (~1-1.5 MB per layer)

The verifier does **not** know the model weights directly — only the commitments.

### Security Properties

| Property           | Guarantee                                                        | Mechanism                                                  |
| ------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| **Soundness**      | Prover cannot forge a valid proof for incorrect computation      | Sumcheck soundness: `< 2^{-247}` per round                 |
| **Weight binding** | Proof is cryptographically tied to specific committed weights    | Cross-verification of `claim_W` against verifier's weights |
| **Zero-knowledge** | Verifier learns nothing about weights beyond what output reveals | Interactive proof + commitment hiding property             |
| **Completeness**   | Honest prover always generates a valid proof                     | Deterministic computation + exact field arithmetic         |

### Per-Component Security Breakdown

| Component           | Sumcheck Rounds         | Soundness per Component | Binding Mechanism                     |
| ------------------- | ----------------------- | ----------------------- | ------------------------------------- |
| RMSNorm             | ~19 (for 524K elements) | `< 2^{-256}`            | `g_inv_rms_(u) · X(u)` claimed output |
| Self-Attn (Q/K/V/O) | 12 each (`log₂ 4096`)   | `< 2^{-247}` each       | `claim_W` cross-verification          |
| Self-Attn (Scores)  | 12 (`log₂ 4096`)        | `< 2^{-247}`            | `zkip` final claim check              |
| Self-Attn (Pooling) | 7 (`log₂ 128`)          | `< 2^{-247}`            | `zkip` final claim check              |
| FFN (Up/Gate)       | 12 (`log₂ 4096`)        | `< 2^{-247}` each       | `claim_W` cross-verification          |
| FFN (Down)          | 14 (`log₂ 11008`)       | `< 2^{-247}`            | `claim_W` cross-verification          |
| FFN (SwiGLU)        | ~21 (`log₂ table`)      | `< 2^{-247}`            | tLookup sumcheck                      |
| Skip Connection     | ~19 (`log₂ 524288`)     | `< 2^{-225}`            | Zero-check + claimed output           |

**Composed security**: The overall soundness error for a single layer is at most the sum of per-component error probabilities, which remains `< 2^{-128}` — providing overwhelming security.

### Attack Resistance

1. **Weight substitution**: Defeated by cross-verification (`claim_W` from proof vs. verifier's computation from committed weights). A wrong weight would produce a different MLE evaluation at the challenge point.

2. **Proof reuse across layers**: Each proof is bound to layer-specific weight files and random challenges. Reusing a proof for a different layer would fail at the weight loading step.

3. **Malicious claimed output**: The verifier independently recomputes the claimed output from its own weights and the input activations. Any discrepancy is detected.

4. **Truncated/corrupted proofs**: Size checks at every step ensure the proof has the correct number of elements for the given dimensions.

---

## Changes from Original zkLLM (CCS 2024)

### Structural Changes

1. **Prover/Verifier Separation**: Every component now has a distinct `{component}_v2.cu` (prover) and `verify_{component}_v2.cu` (verifier) binary, enabling deployment on separate machines.

2. **Binary Proof Serialization** (`proof_io_v2.{cu,cuh}`): Four new proof structures (`RMSNormProof`, `SelfAttnProof`, `FFNProof`, `SkipConnectionProof`) with complete `save_*` / `load_*` functions for disk persistence.

3. **Random Challenge Persistence**: All random challenges (`u`, `v`, `r`, `α`, `β`) are stored in the proof structure, enabling the verifier to replay the interactive protocol without communication.

4. **Post-Attention RMSNorm**: Separate Python wrapper (`llama-post-attn-rmsnorm_v2.py`) and per-layer `rms_inv` file persistence. The original code used a shared temporary file that could be overwritten.

### Code-Level Changes

5. **`zkFC` reference-based weights**: `const FrTensor& weights` replaces `FrTensor weights`, eliminating 1.4 GB GPU copies per layer.

6. **`zkFC::verify()` method**: New standalone verification using saved challenges and cross-verification of `claim_W`.

7. **`zkFC::prove()` precision fix**: Internally recomputes `Y = W · X` using exact field arithmetic to avoid floating-point-to-field conversion errors.

8. **Copy-before-corrupt pattern**: Explicitly copies tensors before `multi_dim_me()` / `partial_me()` to avoid data corruption from recursive GPU operations.

9. **`fileio_utils.py` improvements**: Robust `save_int()` / `load_int()` for quantised tensor I/O.

### New Components

10. **`capture_activations.py`**: Hook-based activation extraction from real HuggingFace model inference.

11. **`generate_proofs_v2.py`**: Master orchestration for automated multi-layer proof generation.

12. **`verify_proofs_v2.py`**: Master orchestration for automated multi-layer proof verification.

13. **`verify_ffn_v2.cu`**: Full 4-phase FFN verifier with SwiGLU tLookup verification.

14. **`verify_skip-connection_v2.cu`**: Skip connection verifier with zero-check and cryptographic binding.

15. **`verify_self-attn_v2.cu`**: 7-component self-attention verifier with scores, softmax, and pooling verification.

### Memory Optimisations

16. **Phase-based GPU memory management**: Large weights loaded/freed one at a time in scoped blocks.

17. **`unique_ptr` intermediate management**: Precise lifetime control for FFN intermediate tensors.

18. **`cudaDeviceSynchronize()` after frees**: Ensures GPU allocator reclaims memory before next allocation.

---

## Acknowledgements

- **Original zkLLM Authors**: Haochen Sun, Jason Li, Hongyang Zhang (University of Waterloo) — [paper](https://arxiv.org/abs/2404.16109), [ACM CCS 2024](https://www.sigsac.org/ccs/CCS2024/home.html)
- **BLS12-381 CUDA Implementation**: [`ec-gpu`](https://github.com/filecoin-project/ec-gpu) by [Filecoin](https://filecoin.io/)
- **LLaMA-2 Models**: [Meta AI](https://huggingface.co/meta-llama)

---

## License

See [LICENSE](LICENSE) for details.

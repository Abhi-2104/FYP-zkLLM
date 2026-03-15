# Skip Connection Proof Generation and Verification in zkLLM v2

## Implementation-Focused README

## Scope

This document covers only the Skip Connection component in zkLLM v2, with emphasis on code behavior and verification flow.

It explains:

- How skip proof generation is implemented
- How skip proof verification is implemented
- The exact proof data model and serialized binary layout
- Mathematical foundations required to understand the implementation
- Operational baselines, caveats, and extension guidance

This README is intentionally code-first and verifier-centric.

---

## 1. Component Role in the Layer Pipeline

Skip Connection computes the residual merge at the end of each transformer block:

$$
z = x + y
$$

Where:

- $x$ is block input residual stream
- $y$ is FFN output stream
- $z$ is the merged output forwarded to next layer input

In this repository the relevant per-layer files are:

- Input A: activations/layer-N-block-input.bin
- Input B: activations/layer-N-ffn-output.bin
- Output: activations/layer-N-skip-output.bin
- Proof: zkllm-workdir/Llama-2-<size>b/layer-N-skip-proof.bin

---

## 2. Implementation Map

- llama-skip-connection_v2.py
  - Python wrapper for skip proof generation
  - Compiles skip binary and invokes prover command

- skip-connection_v2.cu
  - Skip prover implementation
  - Computes z, builds SkipConnectionProof, serializes proof, saves skip output

- verify_skip-connection_v2.cu
  - Skip verifier implementation
  - Loads proof and inputs, recomputes claims and zero-check, validates transcript shape

- proof_io_v2.cuh
  - SkipConnectionProof structure definition

- proof_io_v2.cu
  - save_skip_connection_proof and load_skip_connection_proof binary format

- proof_v2.cu
  - binary_sumcheck implementation used by skip prover

---

## 3. Mathematical Foundations Required for This Code Path

## 3.1 Field and tensor model

All values are represented as Fr elements (BLS12-381 scalar field), loaded from int32 activation binaries into FrTensor.

The addition in skip path is field addition, not floating-point addition.

## 3.2 Core relation proved

The skip correctness target is:

$$
\forall i, \; z_i = x_i + y_i
$$

The prover builds:

$$
\mathrm{diff} = z - x - y
$$

So correctness becomes:

$$
\forall i, \; \mathrm{diff}_i = 0
$$

Verifier evaluates this via random-point check:

$$
\widetilde{\mathrm{diff}}(u) = 0
$$

## 3.3 Claimed output binding check

Prover stores:

$$
\mathrm{claimed\_output} = \widetilde{x}(u) + \widetilde{y}(u)
$$

Verifier recomputes:

- computed_claim = x(u) + y(u)
- z_claim = z(u)

and checks both equal claimed_output.

## 3.4 Binary sumcheck transcript in skip path

The skip prover calls binary_sumcheck(diff, u, u).

In proof_v2.cu, each recursive round contributes 3 Fr values from Fr_bin_sc_step, and terminal round contributes one final value. This is why verifier expects transcript size in:

$$
3\cdot |u| \; \text{to} \; 3\cdot |u| + 1
$$

Fr_bin_sc_step computes per split pair terms:

- out0 = a0^2 - a0
- out1 from linearized cross term
- out2 = (a1 - a0)^2

which is aligned with a binary-constraint style sumcheck transcript over diff.

---

## 4. Proof Generation Flow

## 4.1 Wrapper flow: llama-skip-connection_v2.py

1. Compiles skip prover:
   - make -f Makefile_v2 skip-connection_v2
2. Validates input files:
   - block_input_file
   - block_output_file
3. Builds command:
   - ./skip-connection_v2 <block_input_file> <block_output_file> <workdir> <layer_prefix> <output_file>
4. Exits non-zero on prover failure.

## 4.2 CUDA prover flow: skip-connection_v2.cu

### Step 1: Load tensors

- x = FrTensor::from_int_bin(block_input_fn)
- y = FrTensor::from_int_bin(block_output_fn)
- Enforces x.size == y.size

### Step 2: Compute skip output

- z = x + y

### Step 3: Build proof object

- proof.tensor_size = z.size
- proof.random_u = random_vec(ceilLog2(z.size))
- x_claim = x(proof.random_u)
- y_claim = y(proof.random_u)
- proof.claimed_output = x_claim + y_claim

Then builds zero-check transcript:

- diff = z - x - y
- zero_check = diff(proof.random_u)
- proof.hadamard_sum_proof = binary_sumcheck(diff, proof.random_u, proof.random_u)

### Step 4: Persist proof

- proof file path = workdir + "/" + layer_prefix + "-skip-proof.bin"
- save_skip_connection_proof(proof, proof_filename)

### Step 5: Persist output activation

- z.save_int(output_fn)

---

## 5. SkipConnectionProof Data Model and Binary Layout

## 5.1 Struct fields

Defined in proof_io_v2.cuh:

- hadamard_sum_proof : vector<Fr_t>
- random_u : vector<Fr_t>
- claimed_output : Fr_t
- tensor_size : int

## 5.2 Serialization order

Implemented in proof_io_v2.cu save_skip_connection_proof:

1. tensor_size
2. proof_size + hadamard_sum_proof bytes
3. u_size + random_u bytes
4. claimed_output

Deserializer load_skip_connection_proof reads in same order.

Any struct extension must update both functions in lockstep.

---

## 6. Verification Flow

Verifier binary:

- verify_skip-connection_v2.cu

CLI:

- ./verify_skip-connection_v2 <workdir> <layer_prefix> <block_input_file> <block_output_file>

Proof file is auto-located as:

- <workdir>/<layer_prefix>-skip-proof.bin

## 6.1 Step 1: Load proof

- proof = load_skip_connection_proof(proof_file)
- Prints tensor size, challenge count, transcript size
- If random_u empty, downgrades to structural-only mode

## 6.2 Step 2: Recompute claims and zero-check

When random_u exists and input files are provided:

1. Load x and y activations
2. Check size compatibility with proof.tensor_size
3. Recompute z = x + y
4. Evaluate random-point values:
   - x_claim = x(u)
   - y_claim = y(u)
   - computed_claim = x_claim + y_claim
   - z_claim = z(u)
5. Enforce:
   - computed_claim == proof.claimed_output
   - z_claim == proof.claimed_output
6. Recompute diff = z - x - y and enforce:
   - diff(u) == 0

Any mismatch returns failure immediately.

## 6.3 Step 3: Transcript shape validation

Verifier currently checks transcript size bounds:

- expected_min = 3 \* |u|
- expected_max = expected_min + 1

and reports structural validity if transcript length falls in range.

---

## 7. Pipeline Integration Baselines

## 7.1 Generation pipeline

generate_proofs_v2.py uses:

- python3 llama-skip-connection_v2.py <model_size> <layer> <seq_len> --block_input_file ... --block_output_file ... --output_file ...

## 7.2 Verification pipeline

verify_proofs_v2.py uses:

- ./verify_skip-connection_v2 <workdir> layer-N activations/layer-N-block-input.bin activations/layer-N-ffn-output.bin

So standalone and pipeline execution call the same verifier logic.

---

## 8. End-to-End Commands

## 8.1 Generate skip proof

Example:

```bash
python3 llama-skip-connection_v2.py 7 0 128 \
  --block_input_file ./activations/layer-0-block-input.bin \
  --block_output_file ./activations/layer-0-ffn-output.bin \
  --output_file ./activations/layer-0-skip-output.bin
```

## 8.2 Verify skip proof

Example:

```bash
./verify_skip-connection_v2 \
  ./zkllm-workdir/Llama-2-7b \
  layer-0 \
  ./activations/layer-0-block-input.bin \
  ./activations/layer-0-ffn-output.bin
```

---

## 9. Important Verification Caveats

1. Transcript replay is not fully reconstructed in verifier

- Current verifier performs strong random-point binding checks plus transcript size checks.
- It does not currently replay and enforce every round identity from hadamard_sum_proof values.

2. Structural fallback mode exists

- If proof.random_u is missing (older proofs), verifier reports structural validation only.

3. Cryptographic mode needs both input files

- Claimed-output and zero-check validation requires loading block input and block output activations.

4. Tensor size mismatch is warned

- Verifier warns if activation tensor size differs from proof.tensor_size and can fail later during checks.

---

## 10. Minimal Mental Model

- Prover computes z = x + y and creates a proof object containing:
  - random challenge vector
  - random-point claimed output
  - binary sumcheck transcript for diff = z - x - y
- Verifier reloads x and y, recomputes z, and checks random-point identities and zero-check.
- Transcript size checks ensure expected proof shape from binary_sumcheck recursion.

This is the current skip-connection proving and verification implementation in zkllm_v2.

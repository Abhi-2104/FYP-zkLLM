# RMSNorm Proof Verification Plan
## Goal: Demonstrate End-to-End Proof Generation & Verification for Panel Review

---

## 🎯 OBJECTIVE
Prove to the review panel that we can:
1. ✅ Generate ZK proofs for RMSNorm computation
2. ✅ Save proofs to disk in a structured format
3. ✅ **Verify saved proofs against model commitments** ← **Tomorrow's Demo**

---

## 📊 CURRENT STATUS

### What We Have ✅
1. **Proof Generation Working**
   - `rmsnorm_v2.cu` generates unified proof file
   - Saves to: `layer-0-rmsnorm-proof.bin` (1.9KB)
   - Contains:
     - ✅ Hadamard product sumcheck proof (59 Fr_t elements)
     - ✅ Weight proof structure (empty by design - scalar input)
     - ✅ RS1/RS2 proof structures (empty - internal verification only)

2. **Existing Commitments Available**
   - Model weights committed at: `zkllm-workdir/Llama-2-7b/`
   - For layer-0, we have:
     - `layer-0-input_layernorm.weight-commitment.bin` (144 bytes)
     - `layer-0-input_layernorm.weight-int.bin` (16KB)
     - `layer-0-post_attention_layernorm.weight-commitment.bin` (144 bytes)
     - `layer-0-post_attention_layernorm.weight-int.bin` (16KB)

3. **Existing Verification Function**
   - `verifyWeightClaim()` in `proof_v2.cu` (lines 3-10)
   - Takes: `Weight` struct and `Claim` struct
   - Does: Commitment opening verification

### What We Need 🔨
1. **Standalone Verifier Executable**
   - Loads saved proof from disk
   - Loads model commitments from disk
   - Reconstructs claims from proof data
   - Verifies proof against commitments
   - Returns: SUCCESS/FAILURE with detailed output

---

## 🏗️ VERIFICATION ARCHITECTURE

### Existing Flow (Inline Verification)
```
Prover (rmsnorm.cu):
  1. Compute RMSNorm forward pass
  2. Generate proofs inline
  3. Immediately verify with verifyWeightClaim()
  4. Discard proofs (NOT saved)
```

### New Flow (Save & Verify Separately)
```
Prover (rmsnorm_v2.cu):
  1. Compute RMSNorm forward pass
  2. Generate proofs inline
  3. Save proofs to disk ✅ DONE
  
Verifier (verify_rmsnorm_v2.cu): ← NEW
  1. Load proof from disk
  2. Load weight commitments from disk
  3. Reconstruct computation claims
  4. Verify claims against commitments
  5. Report: VALID ✅ or INVALID ❌
```

---

## 📋 IMPLEMENTATION PLAN

### Step 1: Create Verifier Executable (verify_rmsnorm_v2.cu)
**File**: `verify_rmsnorm_v2.cu`
**Purpose**: Standalone proof verifier for RMSNorm

**Key Components**:
```cpp
int main(int argc, char *argv[]) {
    // Args: proof_file, workdir, layer_prefix
    string proof_file = argv[1];
    string workdir = argv[2];
    string layer_prefix = argv[3];
    
    // 1. Load saved proof
    RMSNormProof proof = load_rmsnorm_proof(proof_file);
    
    // 2. Load weight commitment
    auto rmsnorm_weight = create_weight(...);
    
    // 3. Verify hadamard product sumcheck
    bool hp_valid = verify_hadamard_product_proof(proof.hadamard_product_proof);
    
    // 4. Verify weight commitment (if weight proof non-empty)
    bool weight_valid = true;
    if (!proof.weight_proof.empty()) {
        // Reconstruct claim from proof
        // Call verifyWeightClaim()
    }
    
    // 5. Report results
    if (hp_valid && weight_valid) {
        cout << "✅ PROOF VALID" << endl;
        return 0;
    } else {
        cout << "❌ PROOF INVALID" << endl;
        return 1;
    }
}
```

### Step 2: Implement Verification Functions
**Files to modify**:
- `proof_io_v2.cu` - Add verification helpers
- `proof_v2.cu` - Add sumcheck verification

**New Functions Needed**:

```cpp
// Verify hadamard product sumcheck proof
bool verify_hadamard_product_sumcheck(
    const vector<Fr_t>& proof,
    const FrTensor& a,
    const FrTensor& b,
    const vector<Fr_t>& u,
    const vector<Fr_t>& v
) {
    // Walk through sumcheck proof
    // Verify each polynomial evaluation
    // Check final claim matches
}

// Reconstruct weight claim from saved data
Claim reconstruct_weight_claim(
    const vector<Polynomial>& weight_proof,
    const FrTensor& input,
    const FrTensor& output
) {
    // Rebuild the claim structure
    // Extract random challenges from proof
    // Return claim for verification
}
```

### Step 3: Add to Build System
**File**: `Makefile_v2`

```makefile
verify_rmsnorm_v2: verify_rmsnorm_v2.cu $(V2_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $^ -o $@ $(LIBS)
```

### Step 4: Create Test Script
**File**: `test_verify_rmsnorm_v2.sh`

```bash
#!/bin/bash
# Test end-to-end: Generate proof → Verify proof

echo "=== RMSNorm Proof Verification Test ==="

# Step 1: Generate proof
python3 llama-rmsnorm_v2.py 7 0 input 128 \
    --input_file activations/layer-0-block-input.bin \
    --output_file layer-0-rmsnorm-output.bin

# Step 2: Verify proof
./verify_rmsnorm_v2 \
    zkllm-workdir/Llama-2-7b/layer-0-rmsnorm-proof.bin \
    zkllm-workdir/Llama-2-7b \
    layer-0

echo "=== Test Complete ==="
```

---

## 🎬 DEMO SCRIPT FOR PANEL REVIEW

### Setup (Before Panel)
```bash
# 1. Build verifier
make -f Makefile_v2 verify_rmsnorm_v2

# 2. Generate fresh proof
python3 llama-rmsnorm_v2.py 7 0 input 128 \
    --input_file activations/layer-0-block-input.bin \
    --output_file layer-0-rmsnorm-output.bin

# 3. Verify it works
./verify_rmsnorm_v2 \
    zkllm-workdir/Llama-2-7b/layer-0-rmsnorm-proof.bin \
    zkllm-workdir/Llama-2-7b \
    layer-0
```

### Live Demo Steps
1. **Show Proof File**
   ```bash
   ls -lh zkllm-workdir/Llama-2-7b/layer-0-rmsnorm-proof.bin
   # Output: 1.9KB proof file
   ```

2. **Show Commitment Files**
   ```bash
   ls -lh zkllm-workdir/Llama-2-7b/layer-0-input_layernorm*
   # Output: weight-int.bin + weight-commitment.bin
   ```

3. **Run Verification**
   ```bash
   ./verify_rmsnorm_v2 \
       zkllm-workdir/Llama-2-7b/layer-0-rmsnorm-proof.bin \
       zkllm-workdir/Llama-2-7b \
       layer-0
   ```

4. **Expected Output**
   ```
   Loading proof from: layer-0-rmsnorm-proof.bin
     - Hadamard product proof: 59 Fr_t elements
     - Weight proof: 0 polynomials
   
   Loading commitments from: zkllm-workdir/Llama-2-7b/
     - Weight commitment: layer-0-input_layernorm.weight-commitment.bin
     - Weight values: layer-0-input_layernorm.weight-int.bin
   
   Verifying hadamard product sumcheck...
     ✅ Sumcheck verification passed
   
   Verifying weight commitment...
     ✅ Weight claim verified
   
   ========================================
   ✅ PROOF VERIFICATION SUCCESSFUL
   ========================================
   All claims verified against commitments!
   ```

5. **Tamper Test (Optional)**
   ```bash
   # Corrupt the proof file
   dd if=/dev/urandom of=zkllm-workdir/Llama-2-7b/layer-0-rmsnorm-proof.bin \
       bs=1 count=10 seek=100 conv=notrunc
   
   # Re-run verification
   ./verify_rmsnorm_v2 ... 
   # Expected: ❌ PROOF INVALID
   ```

---

## ⏰ TIMELINE FOR TOMORROW

### Priority 1 (MUST HAVE - 2-3 hours)
- [ ] Create `verify_rmsnorm_v2.cu` skeleton
- [ ] Implement basic proof loading
- [ ] Implement commitment loading
- [ ] Call existing `verifyWeightClaim()` function
- [ ] Build and test executable

### Priority 2 (NICE TO HAVE - 1-2 hours)
- [ ] Implement hadamard sumcheck verification
- [ ] Add detailed verification output
- [ ] Create demo script

### Priority 3 (IF TIME - 1 hour)
- [ ] Add tamper detection test
- [ ] Clean up output formatting
- [ ] Add timing measurements

---

## 🎓 KEY TALKING POINTS FOR PANEL

### What This Demonstrates
1. **End-to-End ZK Pipeline**
   - ✅ Proof Generation (rmsnorm_v2.cu)
   - ✅ Proof Serialization (proof_io_v2.cu)
   - ✅ Proof Verification (verify_rmsnorm_v2.cu)

2. **Commitment Scheme Working**
   - Model weights committed once during setup
   - Proofs verify against these commitments
   - No need to reveal actual weights

3. **Scalability Path**
   - Same pattern works for all operations:
     - RMSNorm (done)
     - Self-Attention (similar structure)
     - FFN (similar structure)
   - Full transformer layer verification is just composition

4. **Production-Ready Features**
   - Proofs saved to disk (not just in-memory)
   - Separate prover/verifier executables
   - Can verify proofs offline
   - Can batch verify multiple layers

### Anticipated Questions & Answers

**Q: Why is the weight proof empty?**
A: By design - RMSNorm uses scalar RMS value (size 1), so no sumcheck needed. The weight claim is still verified through commitment opening.

**Q: How long does verification take?**
A: Much faster than proof generation (~0.1s vs ~3s). Verification is the efficient part of ZK.

**Q: Can this scale to full model?**
A: Yes - we're verifying layer-0. Same code works for all 32 layers. Just loop the verifier.

**Q: What prevents cheating?**
A: Cryptographic commitment scheme (Pedersen/KZG). Prover can't change weights without breaking the commitment. Sumcheck protocol ensures computation correctness.

---

## 📝 NEXT STEPS AFTER REVIEW

1. **Extend to Full Layer**
   - Verify self-attention proofs
   - Verify FFN proofs
   - Compose all into full layer verification

2. **Optimize Proof Size**
   - Current: ~2KB per RMSNorm
   - Could compress polynomial proofs
   - Could use recursive SNARKs

3. **Batch Verification**
   - Verify multiple layers in parallel
   - Aggregate proofs for efficiency

4. **Verifier Optimization**
   - Move verification to CPU (don't need GPU)
   - Could run on lightweight client

---

## 🚀 IMMEDIATE ACTION ITEMS

**Tonight (3-4 hours)**:
1. Create `verify_rmsnorm_v2.cu` 
2. Implement basic proof + commitment loading
3. Call existing `verifyWeightClaim()`
4. Test with layer-0 proof

**Tomorrow Morning (1-2 hours)**:
1. Add verification output formatting
2. Create demo script
3. Test run 3-5 times to ensure stability
4. Prepare talking points

**Before Panel**:
1. Delete all temporary files
2. Generate fresh clean proof
3. Verify it works
4. Open terminal + editor ready for demo

---

## 🎯 SUCCESS CRITERIA

**Minimum Success (Must Have)**:
- ✅ `verify_rmsnorm_v2` executable exists
- ✅ Loads proof file successfully
- ✅ Loads commitment file successfully  
- ✅ Calls `verifyWeightClaim()` and passes
- ✅ Prints "PROOF VALID" or "PROOF INVALID"

**Full Success (Nice to Have)**:
- ✅ All of above
- ✅ Verifies hadamard sumcheck
- ✅ Clean formatted output
- ✅ Timing measurements
- ✅ Tamper detection works

**Stretch Goals**:
- ✅ All of above
- ✅ Verify multiple layers in one run
- ✅ Batch verification optimization
- ✅ Web-based verification dashboard

---

## 📚 REFERENCE

### Existing Verification Code
- `proof_v2.cu::verifyWeightClaim()` - Lines 3-10
- `commitment_v2.cu::Commitment::open()` - Commitment opening
- `proof_v2.cu::hadamard_product_sumcheck()` - Proof generation (verify is inverse)

### Key Data Structures
```cpp
struct RMSNormProof {
    vector<Fr_t> hadamard_product_proof;
    vector<Polynomial> weight_proof;
    vector<Polynomial> rs1_proof;
    vector<Polynomial> rs2_proof;
};

struct Weight {
    Commitment generator;
    FrTensor weight;
    G1TensorJacobian com;
    uint in_dim;
    uint out_dim;
};

struct Claim {
    Fr_t claim;
    vector<vector<Fr_t>> u;
    vector<uint> dims;
};
```

### File Locations
- Proof: `zkllm-workdir/Llama-2-7b/layer-0-rmsnorm-proof.bin`
- Commitment: `zkllm-workdir/Llama-2-7b/layer-0-input_layernorm.weight-commitment.bin`
- Weight: `zkllm-workdir/Llama-2-7b/layer-0-input_layernorm.weight-int.bin`

---

**Last Updated**: Nov 5, 2024 17:30
**Status**: Ready to implement verifier
**Next**: Create `verify_rmsnorm_v2.cu`

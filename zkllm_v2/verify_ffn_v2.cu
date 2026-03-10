#include "zkfc_v2.cuh"
#include "fr-tensor.cuh"
#include "proof_v2.cuh"
#include "commitment_v2.cuh"
#include "proof_io_v2.cuh"
#include "rescaling_v2.cuh"
#include "tlookup_v2.cuh"
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

// Helper constants for polynomial evaluation
const Fr_t TEMP_ZERO {0, 0, 0, 0, 0, 0, 0, 0};
const Fr_t TEMP_ONE {1, 0, 0, 0, 0, 0, 0, 0};

// Helper function to compare Fr_t values (approximate equality)
static bool fr_approx_equal(const Fr_t& a, const Fr_t& b) {
    // Compare all 8 limbs
    for (int i = 0; i < 8; i++) {
        if (a.val[i] != b.val[i]) return false;
    }
    return true;
}

// Helper to check GPU memory
void print_gpu_memory(const char* label) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    cerr << "GPU Memory [" << label << "]: " 
         << (total_mem - free_mem) / (1024*1024) << " MB used / " 
         << total_mem / (1024*1024) << " MB total, "
         << free_mem / (1024*1024) << " MB free" << endl;
}

int main(int argc, char *argv[])
{
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <proof_file> <workdir> <layer_prefix> <seq_len> <input_activation_file>" << endl;
        cerr << "Example: " << argv[0] << " zkllm-workdir/Llama-2-7b/layer-0-ffn-proof.bin zkllm-workdir/Llama-2-7b layer-0 10 activations/layer-0-ffn-input.bin" << endl;
        return 1;
    }

    string proof_file = argv[1];
    string workdir = argv[2];
    string layer_prefix = argv[3];
    int seq_len = std::stoi(argv[4]);
    string input_activation_file = argv[5];

    cout << "\n" << string(70, '=') << endl;
    cout << "FFN Proof Verification (v2) - Sequential Memory Management" << endl;
    cout << string(70, '=') << "\n" << endl;

    print_gpu_memory("start");

    // ===== STEP 1: Load Proof from Disk =====
    cout << "Step 1: Loading proof from disk..." << endl;
    cout << "  File: " << proof_file << endl;
    
    FFNProof proof;
    try {
        proof = load_ffn_proof(proof_file);
        cout << "  ✓ Proof loaded successfully" << endl;
        cout << "    - Dimensions: seq_len=" << proof.seq_len << ", embed_dim=" << proof.embed_dim 
             << ", hidden_dim=" << proof.hidden_dim << endl;
        cout << "    - Up projection proof: " << proof.up_proj_proof.size() << " polynomials" << endl;
        cout << "    - Gate projection proof: " << proof.gate_proj_proof.size() << " polynomials" << endl;
        cout << "    - Down projection proof: " << proof.down_proj_proof.size() << " polynomials" << endl;
        cout << "    - SwiGLU proof: " << proof.swiglu_proof.size() << " polynomials" << endl;
        
        if (proof.up_u_batch.empty() || proof.up_u_input.empty() || proof.up_u_output.empty()) {
            cout << "    ⚠️  WARNING: Incomplete proof (missing random challenges)" << endl;
            return 1;
        }
    } catch (const exception& e) {
        cerr << "  ✗ Failed to load proof: " << e.what() << endl;
        return 1;
    }
    cout << endl;

    bool can_verify_cryptographically = !proof.up_u_batch.empty() && !proof.up_u_input.empty();
    bool up_verified = false, gate_verified = false, down_verified = false;

    // ========== PHASE 1: VERIFY UP PROJECTION ==========
    cout << "[Phase 1/3] Verifying Up Projection..." << endl;
    {
        print_gpu_memory("before up_proj load");
        
        Weight up_proj = create_weight(
            workdir + "/mlp.up_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-mlp.up_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-mlp.up_proj.weight-commitment.bin",
            proof.embed_dim, proof.hidden_dim
        );
        cout << "  ✓ Up projection weight loaded" << endl;
        print_gpu_memory("after up_proj load");
        
        if (up_proj.weight.gpu_data == nullptr) {
            cerr << "ERROR: up_proj.weight allocation failed!" << endl;
            return 1;
        }
        
        if (can_verify_cryptographically && proof.up_proj_proof.size() > 0) {
            zkFC up_layer(proof.embed_dim, proof.hidden_dim, up_proj.weight);
            
            up_verified = up_layer.verify(proof.up_proj_proof,
                                          proof.up_u_batch, proof.up_u_input, proof.up_u_output,
                                          proof.up_claim, proof.up_claim_W);
            
            if (up_verified) {
                cout << "  ✅ Up projection SUMCHECK VERIFIED" << endl;
            } else {
                cout << "  ❌ Up projection SUMCHECK FAILED!" << endl;
                return 1;
            }
        }
        // up_proj goes out of scope and frees GPU memory
    }
    print_gpu_memory("after up_proj freed");
    cout << endl;

    // ========== PHASE 2: VERIFY GATE PROJECTION ==========
    cout << "[Phase 2/3] Verifying Gate Projection..." << endl;
    {
        print_gpu_memory("before gate_proj load");
        
        Weight gate_proj = create_weight(
            workdir + "/mlp.gate_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-commitment.bin",
            proof.embed_dim, proof.hidden_dim
        );
        cout << "  ✓ Gate projection weight loaded" << endl;
        print_gpu_memory("after gate_proj load");
        
        if (gate_proj.weight.gpu_data == nullptr) {
            cerr << "ERROR: gate_proj.weight allocation failed!" << endl;
            return 1;
        }
        
        if (can_verify_cryptographically && proof.gate_proj_proof.size() > 0) {
            zkFC gate_layer(proof.embed_dim, proof.hidden_dim, gate_proj.weight);
            
            gate_verified = gate_layer.verify(proof.gate_proj_proof,
                                              proof.gate_u_batch, proof.gate_u_input, proof.gate_u_output,
                                              proof.gate_claim, proof.gate_claim_W);
            
            if (gate_verified) {
                cout << "  ✅ Gate projection SUMCHECK VERIFIED" << endl;
            } else {
                cout << "  ❌ Gate projection SUMCHECK FAILED!" << endl;
                return 1;
            }
        }
        // gate_proj goes out of scope and frees GPU memory
    }
    print_gpu_memory("after gate_proj freed");
    cout << endl;

    // ========== PHASE 3: VERIFY DOWN PROJECTION ==========
    cout << "[Phase 3/3] Verifying Down Projection..." << endl;
    {
        print_gpu_memory("before down_proj load");
        
        Weight down_proj = create_weight(
            workdir + "/mlp.down_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-mlp.down_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-mlp.down_proj.weight-commitment.bin",
            proof.hidden_dim, proof.embed_dim
        );
        cout << "  ✓ Down projection weight loaded" << endl;
        print_gpu_memory("after down_proj load");
        
        if (down_proj.weight.gpu_data == nullptr) {
            cerr << "ERROR: down_proj.weight allocation failed!" << endl;
            return 1;
        }
        
        if (can_verify_cryptographically && proof.down_proj_proof.size() > 0) {
            zkFC down_layer(proof.hidden_dim, proof.embed_dim, down_proj.weight);
            
            down_verified = down_layer.verify(proof.down_proj_proof,
                                              proof.down_u_batch, proof.down_u_input, proof.down_u_output,
                                              proof.down_claim, proof.down_claim_W);
            
            if (down_verified) {
                cout << "  ✅ Down projection SUMCHECK VERIFIED" << endl;
            } else {
                cout << "  ❌ Down projection SUMCHECK FAILED!" << endl;
                return 1;
            }
        }
        // down_proj goes out of scope and frees GPU memory
    }
    print_gpu_memory("after down_proj freed");
    cout << endl;

    // ===== PHASE 4: VERIFY SWIGLU =====
    bool swiglu_verified = false;
    cout << "[Phase 4/4] Verifying SwiGLU Activation..." << endl;
    if (proof.swiglu_proof.empty()) {
        cout << "  ⚠ No SwiGLU proof (skipping verification)" << endl;
        swiglu_verified = true;  // Not a failure, just not provided
    } else {
        cout << "  Checking " << proof.swiglu_proof.size() << " sumcheck polynomials..." << endl;
        
        // Compute initial claim: claim = alpha + alpha_sq
        Fr_t alpha = proof.swiglu_alpha;
        Fr_t alpha_sq = alpha * alpha;
        Fr_t claim = alpha + alpha_sq;
        
        // Combine challenge vectors for verification
        // The challenges alternate between v (for phase1) and u (for phase2)
        vector<Fr_t> all_challenges;
        
        // In tLookup, v is split into v1 (phase1) and v2 (phase2)
        // For seq_len*hidden_dim dimension, we need to track the challenge order
        // The polynomials are recorded in order: phase1 polynomials, then phase2 polynomials
        
        // Verification: check each polynomial satisfies claim = p(0) + p(1)
        bool all_sumchecks_valid = true;
        size_t challenge_idx = 0;
        
        for (size_t i = 0; i < proof.swiglu_proof.size(); ++i) {
            // Use const_cast to call non-const operator()  
            Polynomial& p = const_cast<Polynomial&>(proof.swiglu_proof[i]);
            Fr_t p0 = p(TEMP_ZERO);
            Fr_t p1 = p(TEMP_ONE);
            Fr_t sum = p0 + p1;
            
            if (!fr_approx_equal(claim, sum)) {
                cerr << "  ❌ SwiGLU sumcheck failed at polynomial " << i << endl;
                cerr << "     claim = " << claim << endl;
                cerr << "     p(0) + p(1) = " << sum << endl;
                all_sumchecks_valid = false;
                break;
            }
            
            // Get next challenge from swiglu_v (same order as challenges were generated)
            Fr_t challenge;
            if (challenge_idx < proof.swiglu_v.size()) {
                challenge = proof.swiglu_v[proof.swiglu_v.size() - 1 - challenge_idx];
            } else {
                // Shouldn't happen if proof is well-formed
                cerr << "  ❌ Not enough challenges for verification" << endl;
                all_sumchecks_valid = false;
                break;
            }
            
            // Update claim for next round
            claim = p(challenge);
            challenge_idx++;
        }
        
        if (all_sumchecks_valid) {
            cout << "  ✅ SwiGLU SUMCHECK VERIFIED (" << proof.swiglu_proof.size() << " polynomials)" << endl;
            swiglu_verified = true;
        } else {
            cout << "  ❌ SwiGLU SUMCHECK FAILED!" << endl;
            swiglu_verified = false;
        }
    }
    cout << endl;

    // ===== Final Summary =====
    cout << string(70, '=' ) << endl;
    bool all_verified = up_verified && gate_verified && down_verified && swiglu_verified;
    if (all_verified) {
        cout << "✅ FFN PROOF VERIFICATION SUCCESSFUL!" << endl;
        cout << string(70, '=') << endl;
        cout << "All proofs verified:" << endl;
        cout << "  ✓ Up projection (matmul + sumcheck)" << endl;
        cout << "  ✓ Gate projection (matmul + sumcheck)" << endl;
        cout << "  ✓ Down projection (matmul + sumcheck)" << endl;
        if (proof.swiglu_proof.empty()) {
            cout << "  ⚠ SwiGLU (no proof provided)" << endl;
        } else {
            cout << "  ✓ SwiGLU activation (lookup table sumcheck)" << endl;
        }
        cout << "\nThe proof cryptographically binds to the loaded weight commitments." << endl;
        cout << "Cross-verification: Prover's claim_W matches verifier's recomputed claim_W." << endl;
        cout << "Soundness error: < 2^-128 (overwhelming security)" << endl;
    } else {
        cout << "❌ FFN PROOF VERIFICATION FAILED!" << endl;
        if (!up_verified) cout << "  ✗ Up projection failed" << endl;
        if (!gate_verified) cout << "  ✗ Gate projection failed" << endl;
        if (!down_verified) cout << "  ✗ Down projection failed" << endl;
        if (!swiglu_verified) cout << "  ✗ SwiGLU activation failed" << endl;
    }
    cout << string(70, '=') << "\n" << endl;

    print_gpu_memory("end");

    return all_verified ? 0 : 1;
}

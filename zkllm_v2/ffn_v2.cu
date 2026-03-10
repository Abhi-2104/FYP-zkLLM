#include "zkfc_v2.cuh"
#include "fr-tensor.cuh"
#include "proof_v2.cuh"
#include "commitment_v2.cuh"
#include "rescaling_v2.cuh"
#include "proof_io_v2.cuh"
#include "tlookup_v2.cuh"
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>

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
    if (argc != 8) {
        cerr << "Usage: " << argv[0] << " <input_file> <seq_len> <embed_dim> <hidden_dim> <workdir> <layer_prefix> <output_file>" << endl;
        return 1;
    }

    string input_file_name = argv[1];
    int seq_len = std::stoi(argv[2]);
    int embed_dim = std::stoi(argv[3]);
    int hidden_dim = std::stoi(argv[4]);
    string workdir = argv[5];
    string layer_prefix = argv[6];
    string output_file_name = argv[7];

    cout << "\n" << string(70, '=') << endl;
    cout << "FFN Proof Generation (v2) - Sequential Memory Management" << endl;
    cout << string(70, '=') << "\n" << endl;

    print_gpu_memory("start");

    FFNProof ffn_proof;
    ffn_proof.seq_len = seq_len;
    ffn_proof.embed_dim = embed_dim;
    ffn_proof.hidden_dim = hidden_dim;

    // ===== Load Input (needed for up_proj and gate_proj only) =====
    cout << "Loading input..." << endl;
    auto input_ptr = std::make_unique<FrTensor>(FrTensor::from_int_bin(input_file_name));
    cout << "  ✓ Input loaded: " << input_ptr->size << " elements" << endl;
    print_gpu_memory("after input");

    // ===== Load SwiGLU Table (needed for activation) =====
    cout << "Loading SwiGLU table..." << endl;
    FrTensor swiglu_values = FrTensor::from_int_bin("swiglu-table.bin");
    tLookupRangeMapping swiglu(-(1 << 20), 1 << 21, swiglu_values);
    cout << "  ✓ SwiGLU table loaded" << endl;
    print_gpu_memory("after swiglu table");

    // Rescaling objects (small memory footprint)
    Rescaling up_rescale(1 << 16);
    Rescaling gate_rescale(1 << 20);
    Rescaling hidden_rescale(1 << 16);
    Rescaling down_rescale(1 << 16);

    // Intermediate tensors we need to keep (using unique_ptr to avoid default construction)
    std::unique_ptr<FrTensor> up_out_ptr, gate_out_ptr, swiglu_out_ptr, swiglu_m_ptr;

    // ========== PHASE 1: UP PROJECTION ==========
    cout << "\n[Phase 1/4] Up Projection" << endl;
    {
        print_gpu_memory("before up_proj load");
        
        auto up_proj = create_weight(
            workdir + "/mlp.up_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-mlp.up_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-mlp.up_proj.weight-commitment.bin",
            embed_dim, hidden_dim
        );
        cout << "  ✓ Up projection weight loaded" << endl;
        print_gpu_memory("after up_proj load");
        
        // Check allocation succeeded
        if (up_proj.weight.gpu_data == nullptr) {
            cerr << "ERROR: up_proj.weight allocation failed!" << endl;
            return 1;
        }
        
        zkFC up_layer(embed_dim, hidden_dim, up_proj.weight);
        
        // Forward pass
        auto up_out = up_layer(*input_ptr);
        up_out_ptr = std::make_unique<FrTensor>(up_rescale(up_out));
        cout << "  ✓ Up projection computed" << endl;
        
        // Generate proof
        up_layer.prove(*input_ptr, up_out, ffn_proof.up_proj_proof, 
                       ffn_proof.up_u_batch, ffn_proof.up_u_input, ffn_proof.up_u_output,
                       ffn_proof.up_claim, ffn_proof.up_claim_W);
        cout << "  ✓ Up projection proof: " << ffn_proof.up_proj_proof.size() << " polynomials" << endl;
        
        // up_proj goes out of scope and frees GPU memory
    }
    print_gpu_memory("after up_proj freed");

    // ========== PHASE 2: GATE PROJECTION ==========
    cout << "\n[Phase 2/4] Gate Projection" << endl;
    {
        print_gpu_memory("before gate_proj load");
        
        auto gate_proj = create_weight(
            workdir + "/mlp.gate_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-commitment.bin",
            embed_dim, hidden_dim
        );
        cout << "  ✓ Gate projection weight loaded" << endl;
        print_gpu_memory("after gate_proj load");
        
        // Check allocation succeeded
        if (gate_proj.weight.gpu_data == nullptr) {
            cerr << "ERROR: gate_proj.weight allocation failed!" << endl;
            return 1;
        }
        
        zkFC gate_layer(embed_dim, hidden_dim, gate_proj.weight);
        
        // Forward pass
        auto gate_out = gate_layer(*input_ptr);
        gate_out_ptr = std::make_unique<FrTensor>(gate_rescale(gate_out));
        cout << "  ✓ Gate projection computed" << endl;
        
        // Generate proof
        gate_layer.prove(*input_ptr, gate_out, ffn_proof.gate_proj_proof,
                         ffn_proof.gate_u_batch, ffn_proof.gate_u_input, ffn_proof.gate_u_output,
                         ffn_proof.gate_claim, ffn_proof.gate_claim_W);
        cout << "  ✓ Gate projection proof: " << ffn_proof.gate_proj_proof.size() << " polynomials" << endl;
        
        // gate_proj goes out of scope and frees GPU memory
    }
    
    // === Free input - no longer needed after up_proj and gate_proj ===
    input_ptr.reset();
    cudaDeviceSynchronize();  // Ensure GPU memory is released
    print_gpu_memory("after gate_proj freed");

    // ========== PHASE 3: SWIGLU ACTIVATION ==========
    cout << "\n[Phase 3/4] SwiGLU Activation" << endl;
    {
        auto p = swiglu(*gate_out_ptr);
        swiglu_out_ptr = std::make_unique<FrTensor>(std::move(p.first));
        swiglu_m_ptr = std::make_unique<FrTensor>(std::move(p.second));
        cout << "  ✓ SwiGLU activation computed" << endl;

        // Generate SwiGLU proof parameters
        ffn_proof.swiglu_u = random_vec(ceilLog2(seq_len * hidden_dim));
        ffn_proof.swiglu_v = random_vec(ceilLog2(seq_len * hidden_dim));
        auto temp_rand = random_vec(3);
        ffn_proof.swiglu_r = temp_rand[0];
        ffn_proof.swiglu_alpha = temp_rand[1];
        ffn_proof.swiglu_beta = temp_rand[2];
        
        // Generate SwiGLU proof using tLookupRangeMapping
        // S_in = gate_out, S_out = swiglu_out, m = swiglu_m
        swiglu.prove(*gate_out_ptr, *swiglu_out_ptr, *swiglu_m_ptr,
                     ffn_proof.swiglu_r, ffn_proof.swiglu_alpha, ffn_proof.swiglu_beta,
                     ffn_proof.swiglu_u, ffn_proof.swiglu_v, ffn_proof.swiglu_proof);
        cout << "  ✓ SwiGLU proof: " << ffn_proof.swiglu_proof.size() << " polynomials" << endl;
    }
    print_gpu_memory("after swiglu");

    // Compute hidden layer (element-wise multiplication)
    auto down_in = (*swiglu_out_ptr) * (*up_out_ptr);
    auto down_in_ = hidden_rescale(down_in);
    cout << "  ✓ Hidden layer computed (swiglu_out * up_out)" << endl;

    // Free tensors no longer needed
    gate_out_ptr.reset();
    swiglu_out_ptr.reset();
    swiglu_m_ptr.reset();
    up_out_ptr.reset();
    print_gpu_memory("after freeing intermediates");

    // ========== PHASE 4: DOWN PROJECTION ==========
    cout << "\n[Phase 4/4] Down Projection" << endl;
    std::unique_ptr<FrTensor> down_out_ptr;
    {
        print_gpu_memory("before down_proj load");
        
        auto down_proj = create_weight(
            workdir + "/mlp.down_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-mlp.down_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-mlp.down_proj.weight-commitment.bin",
            hidden_dim, embed_dim
        );
        cout << "  ✓ Down projection weight loaded" << endl;
        print_gpu_memory("after down_proj load");
        
        // Check allocation succeeded
        if (down_proj.weight.gpu_data == nullptr) {
            cerr << "ERROR: down_proj.weight allocation failed!" << endl;
            return 1;
        }
        
        zkFC down_layer(hidden_dim, embed_dim, down_proj.weight);
        
        // Forward pass
        auto down_out = down_layer(down_in_);
        down_out_ptr = std::make_unique<FrTensor>(down_rescale(down_out));
        cout << "  ✓ Down projection computed" << endl;
        
        // Generate proof
        down_layer.prove(down_in_, down_out, ffn_proof.down_proj_proof,
                         ffn_proof.down_u_batch, ffn_proof.down_u_input, ffn_proof.down_u_output,
                         ffn_proof.down_claim, ffn_proof.down_claim_W);
        cout << "  ✓ Down projection proof: " << ffn_proof.down_proj_proof.size() << " polynomials" << endl;
        
        // down_proj goes out of scope and frees GPU memory
    }
    print_gpu_memory("after down_proj freed");

    // ===== Compute Claimed Output =====
    cout << "\nComputing claimed output for verification..." << endl;
    auto eval_u = random_vec(ceilLog2(down_out_ptr->size));
    ffn_proof.claimed_output_u = eval_u;
    ffn_proof.claimed_output = (*down_out_ptr)(eval_u);
    cout << "  ✓ Claimed output computed at random point" << endl;

    // ===== Save Proof =====
    cout << "\nSaving proof to disk..." << endl;
    string proof_filename = workdir + "/" + layer_prefix + "-ffn-proof.bin";
    save_ffn_proof(ffn_proof, proof_filename);
    cout << "  ✓ Proof saved to: " << proof_filename << endl;

    // ===== Save Output Activations =====
    cout << "Saving output activations..." << endl;
    down_out_ptr->save_int(output_file_name);
    cout << "  ✓ Output saved to: " << output_file_name << endl;

    // ===== Summary =====
    cout << "\n" << string(70, '=') << endl;
    cout << "✅ FFN Proof Generation Complete!" << endl;
    cout << string(70, '=') << endl;
    cout << "Proof Components:" << endl;
    cout << "  • Up projection:   " << ffn_proof.up_proj_proof.size() << " polynomials" << endl;
    cout << "  • Gate projection: " << ffn_proof.gate_proj_proof.size() << " polynomials" << endl;
    cout << "  • SwiGLU:          " << ffn_proof.swiglu_proof.size() << " polynomials" << endl;
    cout << "  • Down projection: " << ffn_proof.down_proj_proof.size() << " polynomials" << endl;
    cout << string(70, '=') << "\n" << endl;

    print_gpu_memory("end");

    return 0;
}

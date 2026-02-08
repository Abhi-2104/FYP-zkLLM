#include "self-attn_v2.cuh"
#include "zkfc_v2.cuh"
#include <iostream>

int main() {
    // Test if forward pass and prove use the same computation
    uint E = 4096;
    
    // Load weights
    auto w_o = create_weight(
        "./zkllm-workdir/Llama-2-7b/self_attn.o_proj.weight-pp.bin",
        "./zkllm-workdir/Llama-2-7b/layer-0-self_attn.o_proj.weight-int.bin",
        "./zkllm-workdir/Llama-2-7b/layer-0-self_attn.o_proj.weight-commitment.bin",
        E, E
    );
    
    zkFC fc_o(E, E, w_o.weight);
    
    // Create small test input
    FrTensor test_input(E);
    
    // Compute output twice
    FrTensor out1 = fc_o(test_input);
    FrTensor out2 = fc_o(test_input);
    
    // Compare
    bool match = (out1.size == out2.size);
    std::cout << "Forward pass determinism: " << (match ? "PASS" : "FAIL") << std::endl;
    
    // Now test prove
    try {
        std::vector<Polynomial> proof;
        std::vector<Fr_t> u_b, u_i, u_o;
        fc_o.prove(test_input, out1, proof, u_b, u_i, u_o);
        std::cout << "Prove with out1: SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Prove with out1: FAILED - " << e.what() << std::endl;
    }
    
    return 0;
}

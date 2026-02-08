#include "proof_io_v2.cuh"
#include <iostream>

int main() {
    std::cout << "Testing SelfAttnProof I/O..." << std::endl;
    
    // Create a dummy proof
    SelfAttnProof proof;
    proof.B = 1;
    proof.H = 32;
    proof.L = 128;
    proof.D = 128;
    
    // Add some dummy data
    proof.q_proof.push_back(Fr_t());
    proof.k_proof.push_back(Fr_t());
    proof.v_proof.push_back(Fr_t());
    proof.o_proof.push_back(Fr_t());
    proof.s_proof.push_back(Fr_t());
    proof.sm_proof.push_back(Fr_t());
    proof.p_proof.push_back(Fr_t());
    proof.random_challenges.push_back(Fr_t());
    proof.claimed_output = Fr_t();
    
    std::string test_path = "test-self-attn-proof.bin";
    
    // Save
    std::cout << "Saving proof..." << std::endl;
    save_self_attn_proof(proof, test_path);
    
    // Load
    std::cout << "Loading proof..." << std::endl;
    SelfAttnProof loaded_proof = load_self_attn_proof(test_path);
    
    // Verify dimensions match
    if (loaded_proof.B == proof.B && 
        loaded_proof.H == proof.H && 
        loaded_proof.L == proof.L && 
        loaded_proof.D == proof.D) {
        std::cout << "✅ SUCCESS: Proof I/O works!" << std::endl;
        std::cout << "   Dimensions: B=" << loaded_proof.B 
                  << ", H=" << loaded_proof.H 
                  << ", L=" << loaded_proof.L 
                  << ", D=" << loaded_proof.D << std::endl;
        return 0;
    } else {
        std::cout << "❌ FAILURE: Dimensions don't match!" << std::endl;
        return 1;
    }
}

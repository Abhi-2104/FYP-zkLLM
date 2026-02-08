#ifndef PROOF_IO_V2_CUH
#define PROOF_IO_V2_CUH

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "proof_v2.cuh"
#include "ioutils.cuh"

// Saves a proof (vector of Polynomials) to a binary file.
void save_proof(const std::vector<Polynomial>& proof, const std::string& filename);

// Loads a proof from a binary file into a vector of Polynomials.
std::vector<Polynomial> load_proof(const std::string& filename);

// Saves a proof (vector of Polynomials) to a binary file.
void save_proof(const std::vector<Polynomial>& proof, std::ofstream& out);

// Loads a proof from a binary file into a vector of Polynomials.
std::vector<Polynomial> load_proof(std::ifstream& in);

// Saves a vector of Fr_t to a binary file.
void save_proof_fr(const std::vector<Fr_t>& proof, const std::string& filename);

// Loads a vector of Fr_t from a binary file.
std::vector<Fr_t> load_proof_fr(const std::string& filename);

// Unified RMSNorm proof structure
struct RMSNormProof {
    std::vector<Fr_t> hadamard_product_proof;  // Hadamard product sumcheck proof
    std::vector<Polynomial> weight_proof;       // Weight commitment proof
    std::vector<Polynomial> rs1_proof;          // Rescaling proof 1
    std::vector<Polynomial> rs2_proof;          // Rescaling proof 2
    
    // Random challenges for verification (enables standalone verifier)
    std::vector<Fr_t> random_u;                 // Random challenges for hadamard sumcheck
    std::vector<Fr_t> random_v;                 // Random challenges for hadamard sumcheck
    Fr_t claimed_output;                        // Claimed inner product value
};

// Saves a complete RMSNorm proof to a binary file
void save_rmsnorm_proof(const RMSNormProof& proof, const std::string& filename);

// Loads a complete RMSNorm proof from a binary file
RMSNormProof load_rmsnorm_proof(const std::string& filename);

// Self-attention proof structure - follows RMSNorm v2 pattern
struct SelfAttnProof {
    std::vector<Polynomial> q_proof, k_proof, v_proof, o_proof;
    std::vector<Polynomial> s_proof;   // Attention scores proof (Q @ K^T)
    std::vector<Polynomial> sm_proof;  // Softmax proof
    std::vector<Polynomial> p_proof;   // Pooling proof (SM @ V)
    
    // Random challenges from prove() calls - enables polynomial verification
    std::vector<Fr_t> q_u_batch, q_u_input, q_u_output;  // Q projection challenges
    std::vector<Fr_t> k_u_batch, k_u_input, k_u_output;  // K projection challenges
    std::vector<Fr_t> v_u_batch, v_u_input, v_u_output;  // V projection challenges
    std::vector<Fr_t> o_u_batch, o_u_input, o_u_output;  // O projection challenges
    
    // Q @ K^T challenges for verifying attention scores computation
    std::vector<Fr_t> s_u_batch, s_u_input, s_u_output;  // Scores (Q @ K^T) challenges
    
    // Softmax challenges for verifying softmax(scores) computation
    std::vector<Fr_t> sm_u_Y, sm_v_Y;  // Softmax input/output challenges
    Fr_t sm_r_seg, sm_alpha_seg, sm_beta_seg;  // Softmax segment challenges
    
    // Pooling challenges for verifying attn_weights @ V
    std::vector<Fr_t> p_u_batch, p_u_input, p_u_output;  // Pooling (attn @ V) challenges
    
    Fr_t claimed_output;
    int B, H, L, D;  // Batch, Heads, Sequence Length, Head Dimension
};

// Saves a complete Self-Attention proof to a binary file
void save_self_attn_proof(const SelfAttnProof& proof, const std::string& filename);

// Loads a complete Self-Attention proof from a binary file
SelfAttnProof load_self_attn_proof(const std::string& filename);

#endif // PROOF_IO_V2_CUH

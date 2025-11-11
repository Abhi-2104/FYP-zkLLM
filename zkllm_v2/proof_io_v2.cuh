#ifndef PROOF_IO_V2_CUH
#define PROOF_IO_V2_CUH

#include "polynomial_v2.cuh"
#include "bls12-381.cuh"
#include <vector>
#include <string>

// Saves a proof (vector of Polynomials) to a binary file.
void save_proof(const std::vector<Polynomial>& proof, const std::string& filename);

// Loads a proof from a binary file into a vector of Polynomials.
std::vector<Polynomial> load_proof(const std::string& filename);

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

#endif // PROOF_IO_V2_CUH

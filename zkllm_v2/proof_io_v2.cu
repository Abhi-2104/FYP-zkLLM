#include "proof_io_v2.cuh"
#include "fr-tensor.cuh"
#include "polynomial_v2.cuh"
#include <fstream>
#include <stdexcept>

void save_proof(const std::vector<Polynomial>& proof, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    save_proof(proof, out);
}

// Loads a proof from a binary file into a vector of Polynomials.
std::vector<Polynomial> load_proof(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    return load_proof(in);
}

// Saves a vector of Fr_t to a binary file.
void save_proof_fr(const std::vector<Fr_t>& proof, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    uint64_t num_elements = proof.size();
    out.write(reinterpret_cast<const char*>(&num_elements), sizeof(num_elements));
    out.write(reinterpret_cast<const char*>(proof.data()), num_elements * sizeof(Fr_t));
}

// Loads a vector of Fr_t from a binary file.
std::vector<Fr_t> load_proof_fr(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    uint64_t num_elements;
    in.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
    
    std::vector<Fr_t> proof(num_elements);
    in.read(reinterpret_cast<char*>(proof.data()), num_elements * sizeof(Fr_t));
    
    return proof;
}

// Saves a complete RMSNorm proof to a binary file
void save_rmsnorm_proof(const RMSNormProof& proof, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Save hadamard product proof (Fr_t vector)
    uint64_t hp_size = proof.hadamard_product_proof.size();
    out.write(reinterpret_cast<const char*>(&hp_size), sizeof(hp_size));
    out.write(reinterpret_cast<const char*>(proof.hadamard_product_proof.data()), hp_size * sizeof(Fr_t));

    save_proof(proof.weight_proof, out);
    save_proof(proof.rs1_proof, out);
    save_proof(proof.rs2_proof, out);

    // Save random challenges for verification
    uint64_t u_size = proof.random_u.size();
    out.write(reinterpret_cast<const char*>(&u_size), sizeof(u_size));
    out.write(reinterpret_cast<const char*>(proof.random_u.data()), u_size * sizeof(Fr_t));

    uint64_t v_size = proof.random_v.size();
    out.write(reinterpret_cast<const char*>(&v_size), sizeof(v_size));
    out.write(reinterpret_cast<const char*>(proof.random_v.data()), v_size * sizeof(Fr_t));

    // Save claimed output
    out.write(reinterpret_cast<const char*>(&proof.claimed_output), sizeof(Fr_t));
}

// Loads a complete RMSNorm proof from a binary file
RMSNormProof load_rmsnorm_proof(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    RMSNormProof proof;

    // Load hadamard product proof
    uint64_t hp_size;
    in.read(reinterpret_cast<char*>(&hp_size), sizeof(hp_size));
    proof.hadamard_product_proof.resize(hp_size);
    in.read(reinterpret_cast<char*>(proof.hadamard_product_proof.data()), hp_size * sizeof(Fr_t));

    proof.weight_proof = load_proof(in);
    proof.rs1_proof = load_proof(in);
    proof.rs2_proof = load_proof(in);

    // Load random challenges (if present - backward compatibility)
    if (in.peek() != EOF) {
        uint64_t u_size;
        in.read(reinterpret_cast<char*>(&u_size), sizeof(u_size));
        proof.random_u.resize(u_size);
        in.read(reinterpret_cast<char*>(proof.random_u.data()), u_size * sizeof(Fr_t));

        uint64_t v_size;
        in.read(reinterpret_cast<char*>(&v_size), sizeof(v_size));
        proof.random_v.resize(v_size);
        in.read(reinterpret_cast<char*>(proof.random_v.data()), v_size * sizeof(Fr_t));

        // Load claimed output
        in.read(reinterpret_cast<char*>(&proof.claimed_output), sizeof(Fr_t));
    }

    return proof;
}

void save_proof(const std::vector<Polynomial>& proof, std::ofstream& out) {
    uint64_t num_polynomials = proof.size();
    out.write(reinterpret_cast<const char*>(&num_polynomials), sizeof(num_polynomials));

    for (const auto& poly : proof) {
        int degree = poly.get_degree();
        Fr_t* coeffs_gpu = poly.get_coeffs();
        uint64_t num_coeffs = degree + 1;
        
        // Copy coefficients from GPU to CPU before writing to file
        std::vector<Fr_t> coeffs_cpu(num_coeffs);
        cudaMemcpy(coeffs_cpu.data(), coeffs_gpu, num_coeffs * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        
        out.write(reinterpret_cast<const char*>(&num_coeffs), sizeof(num_coeffs));
        out.write(reinterpret_cast<const char*>(coeffs_cpu.data()), num_coeffs * sizeof(Fr_t));
    }
}

std::vector<Polynomial> load_proof(std::ifstream& in) {
    uint64_t num_polynomials;
    in.read(reinterpret_cast<char*>(&num_polynomials), sizeof(num_polynomials));

    std::vector<Polynomial> proof;
    proof.reserve(num_polynomials);

    for (uint64_t i = 0; i < num_polynomials; ++i) {
        uint64_t num_coeffs;
        in.read(reinterpret_cast<char*>(&num_coeffs), sizeof(num_coeffs));
        
        std::vector<Fr_t> coeffs(num_coeffs);
        in.read(reinterpret_cast<char*>(coeffs.data()), num_coeffs * sizeof(Fr_t));
        proof.emplace_back(std::move(coeffs));
    }

    return proof;
}

// Saves a complete Self-Attention proof to a binary file
void save_self_attn_proof(const SelfAttnProof& proof, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // Dimensions
    out.write(reinterpret_cast<const char*>(&proof.B), sizeof(int));
    out.write(reinterpret_cast<const char*>(&proof.H), sizeof(int));
    out.write(reinterpret_cast<const char*>(&proof.L), sizeof(int));
    out.write(reinterpret_cast<const char*>(&proof.D), sizeof(int));

    // Save polynomial proof vectors
    save_proof(proof.q_proof, out);
    save_proof(proof.k_proof, out);
    save_proof(proof.v_proof, out);
    save_proof(proof.o_proof, out);
    save_proof(proof.s_proof, out);
    save_proof(proof.sm_proof, out);
    save_proof(proof.p_proof, out);

    // Save Q projection random challenges
    uint64_t q_u_batch_size = proof.q_u_batch.size();
    out.write(reinterpret_cast<const char*>(&q_u_batch_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.q_u_batch.data()), q_u_batch_size * sizeof(Fr_t));
    
    uint64_t q_u_input_size = proof.q_u_input.size();
    out.write(reinterpret_cast<const char*>(&q_u_input_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.q_u_input.data()), q_u_input_size * sizeof(Fr_t));
    
    uint64_t q_u_output_size = proof.q_u_output.size();
    out.write(reinterpret_cast<const char*>(&q_u_output_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.q_u_output.data()), q_u_output_size * sizeof(Fr_t));

    // Save K projection random challenges
    uint64_t k_u_batch_size = proof.k_u_batch.size();
    out.write(reinterpret_cast<const char*>(&k_u_batch_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.k_u_batch.data()), k_u_batch_size * sizeof(Fr_t));
    
    uint64_t k_u_input_size = proof.k_u_input.size();
    out.write(reinterpret_cast<const char*>(&k_u_input_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.k_u_input.data()), k_u_input_size * sizeof(Fr_t));
    
    uint64_t k_u_output_size = proof.k_u_output.size();
    out.write(reinterpret_cast<const char*>(&k_u_output_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.k_u_output.data()), k_u_output_size * sizeof(Fr_t));

    // Save V projection random challenges
    uint64_t v_u_batch_size = proof.v_u_batch.size();
    out.write(reinterpret_cast<const char*>(&v_u_batch_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.v_u_batch.data()), v_u_batch_size * sizeof(Fr_t));
    
    uint64_t v_u_input_size = proof.v_u_input.size();
    out.write(reinterpret_cast<const char*>(&v_u_input_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.v_u_input.data()), v_u_input_size * sizeof(Fr_t));
    
    uint64_t v_u_output_size = proof.v_u_output.size();
    out.write(reinterpret_cast<const char*>(&v_u_output_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.v_u_output.data()), v_u_output_size * sizeof(Fr_t));

    // Save O projection random challenges
    uint64_t o_u_batch_size = proof.o_u_batch.size();
    out.write(reinterpret_cast<const char*>(&o_u_batch_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.o_u_batch.data()), o_u_batch_size * sizeof(Fr_t));
    
    uint64_t o_u_input_size = proof.o_u_input.size();
    out.write(reinterpret_cast<const char*>(&o_u_input_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.o_u_input.data()), o_u_input_size * sizeof(Fr_t));
    
    uint64_t o_u_output_size = proof.o_u_output.size();
    out.write(reinterpret_cast<const char*>(&o_u_output_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.o_u_output.data()), o_u_output_size * sizeof(Fr_t));

    // Save Q @ K^T (scores) random challenges
    uint64_t s_u_batch_size = proof.s_u_batch.size();
    out.write(reinterpret_cast<const char*>(&s_u_batch_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.s_u_batch.data()), s_u_batch_size * sizeof(Fr_t));
    
    uint64_t s_u_input_size = proof.s_u_input.size();
    out.write(reinterpret_cast<const char*>(&s_u_input_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.s_u_input.data()), s_u_input_size * sizeof(Fr_t));
    
    uint64_t s_u_output_size = proof.s_u_output.size();
    out.write(reinterpret_cast<const char*>(&s_u_output_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.s_u_output.data()), s_u_output_size * sizeof(Fr_t));

    // Save Softmax random challenges
    uint64_t sm_u_Y_size = proof.sm_u_Y.size();
    out.write(reinterpret_cast<const char*>(&sm_u_Y_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.sm_u_Y.data()), sm_u_Y_size * sizeof(Fr_t));
    
    uint64_t sm_v_Y_size = proof.sm_v_Y.size();
    out.write(reinterpret_cast<const char*>(&sm_v_Y_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.sm_v_Y.data()), sm_v_Y_size * sizeof(Fr_t));
    
    out.write(reinterpret_cast<const char*>(&proof.sm_r_seg), sizeof(Fr_t));
    out.write(reinterpret_cast<const char*>(&proof.sm_alpha_seg), sizeof(Fr_t));
    out.write(reinterpret_cast<const char*>(&proof.sm_beta_seg), sizeof(Fr_t));

    // Save Pooling (attn @ V) random challenges
    uint64_t p_u_batch_size = proof.p_u_batch.size();
    out.write(reinterpret_cast<const char*>(&p_u_batch_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.p_u_batch.data()), p_u_batch_size * sizeof(Fr_t));
    
    uint64_t p_u_input_size = proof.p_u_input.size();
    out.write(reinterpret_cast<const char*>(&p_u_input_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.p_u_input.data()), p_u_input_size * sizeof(Fr_t));
    
    uint64_t p_u_output_size = proof.p_u_output.size();
    out.write(reinterpret_cast<const char*>(&p_u_output_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(proof.p_u_output.data()), p_u_output_size * sizeof(Fr_t));

    // Claimed output
    out.write(reinterpret_cast<const char*>(&proof.claimed_output), sizeof(Fr_t));

    out.close();
}

// Loads a complete Self-Attention proof from a binary file
SelfAttnProof load_self_attn_proof(const std::string& filename) {
    SelfAttnProof proof;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        exit(1);
    }

    // Dimensions
    in.read(reinterpret_cast<char*>(&proof.B), sizeof(int));
    in.read(reinterpret_cast<char*>(&proof.H), sizeof(int));
    in.read(reinterpret_cast<char*>(&proof.L), sizeof(int));
    in.read(reinterpret_cast<char*>(&proof.D), sizeof(int));

    // Load polynomial proof vectors
    proof.q_proof = load_proof(in);
    proof.k_proof = load_proof(in);
    proof.v_proof = load_proof(in);
    proof.o_proof = load_proof(in);
    proof.s_proof = load_proof(in);
    proof.sm_proof = load_proof(in);
    proof.p_proof = load_proof(in);

    // Load Q projection random challenges
    uint64_t q_u_batch_size;
    in.read(reinterpret_cast<char*>(&q_u_batch_size), sizeof(uint64_t));
    proof.q_u_batch.resize(q_u_batch_size);
    in.read(reinterpret_cast<char*>(proof.q_u_batch.data()), q_u_batch_size * sizeof(Fr_t));
    
    uint64_t q_u_input_size;
    in.read(reinterpret_cast<char*>(&q_u_input_size), sizeof(uint64_t));
    proof.q_u_input.resize(q_u_input_size);
    in.read(reinterpret_cast<char*>(proof.q_u_input.data()), q_u_input_size * sizeof(Fr_t));
    
    uint64_t q_u_output_size;
    in.read(reinterpret_cast<char*>(&q_u_output_size), sizeof(uint64_t));
    proof.q_u_output.resize(q_u_output_size);
    in.read(reinterpret_cast<char*>(proof.q_u_output.data()), q_u_output_size * sizeof(Fr_t));

    // Load K projection random challenges
    uint64_t k_u_batch_size;
    in.read(reinterpret_cast<char*>(&k_u_batch_size), sizeof(uint64_t));
    proof.k_u_batch.resize(k_u_batch_size);
    in.read(reinterpret_cast<char*>(proof.k_u_batch.data()), k_u_batch_size * sizeof(Fr_t));
    
    uint64_t k_u_input_size;
    in.read(reinterpret_cast<char*>(&k_u_input_size), sizeof(uint64_t));
    proof.k_u_input.resize(k_u_input_size);
    in.read(reinterpret_cast<char*>(proof.k_u_input.data()), k_u_input_size * sizeof(Fr_t));
    
    uint64_t k_u_output_size;
    in.read(reinterpret_cast<char*>(&k_u_output_size), sizeof(uint64_t));
    proof.k_u_output.resize(k_u_output_size);
    in.read(reinterpret_cast<char*>(proof.k_u_output.data()), k_u_output_size * sizeof(Fr_t));

    // Load V projection random challenges
    uint64_t v_u_batch_size;
    in.read(reinterpret_cast<char*>(&v_u_batch_size), sizeof(uint64_t));
    proof.v_u_batch.resize(v_u_batch_size);
    in.read(reinterpret_cast<char*>(proof.v_u_batch.data()), v_u_batch_size * sizeof(Fr_t));
    
    uint64_t v_u_input_size;
    in.read(reinterpret_cast<char*>(&v_u_input_size), sizeof(uint64_t));
    proof.v_u_input.resize(v_u_input_size);
    in.read(reinterpret_cast<char*>(proof.v_u_input.data()), v_u_input_size * sizeof(Fr_t));
    
    uint64_t v_u_output_size;
    in.read(reinterpret_cast<char*>(&v_u_output_size), sizeof(uint64_t));
    proof.v_u_output.resize(v_u_output_size);
    in.read(reinterpret_cast<char*>(proof.v_u_output.data()), v_u_output_size * sizeof(Fr_t));

    // Load O projection random challenges
    uint64_t o_u_batch_size;
    in.read(reinterpret_cast<char*>(&o_u_batch_size), sizeof(uint64_t));
    proof.o_u_batch.resize(o_u_batch_size);
    in.read(reinterpret_cast<char*>(proof.o_u_batch.data()), o_u_batch_size * sizeof(Fr_t));
    
    uint64_t o_u_input_size;
    in.read(reinterpret_cast<char*>(&o_u_input_size), sizeof(uint64_t));
    proof.o_u_input.resize(o_u_input_size);
    in.read(reinterpret_cast<char*>(proof.o_u_input.data()), o_u_input_size * sizeof(Fr_t));
    
    uint64_t o_u_output_size;
    in.read(reinterpret_cast<char*>(&o_u_output_size), sizeof(uint64_t));
    proof.o_u_output.resize(o_u_output_size);
    in.read(reinterpret_cast<char*>(proof.o_u_output.data()), o_u_output_size * sizeof(Fr_t));

    // Load Q @ K^T (scores) random challenges
    uint64_t s_u_batch_size;
    in.read(reinterpret_cast<char*>(&s_u_batch_size), sizeof(uint64_t));
    proof.s_u_batch.resize(s_u_batch_size);
    in.read(reinterpret_cast<char*>(proof.s_u_batch.data()), s_u_batch_size * sizeof(Fr_t));
    
    uint64_t s_u_input_size;
    in.read(reinterpret_cast<char*>(&s_u_input_size), sizeof(uint64_t));
    proof.s_u_input.resize(s_u_input_size);
    in.read(reinterpret_cast<char*>(proof.s_u_input.data()), s_u_input_size * sizeof(Fr_t));
    
    uint64_t s_u_output_size;
    in.read(reinterpret_cast<char*>(&s_u_output_size), sizeof(uint64_t));
    proof.s_u_output.resize(s_u_output_size);
    in.read(reinterpret_cast<char*>(proof.s_u_output.data()), s_u_output_size * sizeof(Fr_t));

    // Load Softmax random challenges
    uint64_t sm_u_Y_size;
    in.read(reinterpret_cast<char*>(&sm_u_Y_size), sizeof(uint64_t));
    proof.sm_u_Y.resize(sm_u_Y_size);
    in.read(reinterpret_cast<char*>(proof.sm_u_Y.data()), sm_u_Y_size * sizeof(Fr_t));
    
    uint64_t sm_v_Y_size;
    in.read(reinterpret_cast<char*>(&sm_v_Y_size), sizeof(uint64_t));
    proof.sm_v_Y.resize(sm_v_Y_size);
    in.read(reinterpret_cast<char*>(proof.sm_v_Y.data()), sm_v_Y_size * sizeof(Fr_t));
    
    in.read(reinterpret_cast<char*>(&proof.sm_r_seg), sizeof(Fr_t));
    in.read(reinterpret_cast<char*>(&proof.sm_alpha_seg), sizeof(Fr_t));
    in.read(reinterpret_cast<char*>(&proof.sm_beta_seg), sizeof(Fr_t));

    // Load Pooling (attn @ V) random challenges
    uint64_t p_u_batch_size;
    in.read(reinterpret_cast<char*>(&p_u_batch_size), sizeof(uint64_t));
    proof.p_u_batch.resize(p_u_batch_size);
    in.read(reinterpret_cast<char*>(proof.p_u_batch.data()), p_u_batch_size * sizeof(Fr_t));
    
    uint64_t p_u_input_size;
    in.read(reinterpret_cast<char*>(&p_u_input_size), sizeof(uint64_t));
    proof.p_u_input.resize(p_u_input_size);
    in.read(reinterpret_cast<char*>(proof.p_u_input.data()), p_u_input_size * sizeof(Fr_t));
    
    uint64_t p_u_output_size;
    in.read(reinterpret_cast<char*>(&p_u_output_size), sizeof(uint64_t));
    proof.p_u_output.resize(p_u_output_size);
    in.read(reinterpret_cast<char*>(proof.p_u_output.data()), p_u_output_size * sizeof(Fr_t));

    // Claimed output
    in.read(reinterpret_cast<char*>(&proof.claimed_output), sizeof(Fr_t));

    in.close();
    return proof;
}

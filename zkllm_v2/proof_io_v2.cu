#include "proof_io_v2.cuh"
#include "fr-tensor.cuh"
#include "polynomial_v2.cuh"
#include <fstream>
#include <stdexcept>

// Saves a proof (vector of Polynomials) to a binary file.
void save_proof(const std::vector<Polynomial>& proof, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    uint64_t num_polynomials = proof.size();
    out.write(reinterpret_cast<const char*>(&num_polynomials), sizeof(num_polynomials));

    for (const auto& poly : proof) {
        int degree = poly.get_degree();
        Fr_t* coeffs = poly.get_coeffs();
        uint64_t num_coeffs = degree + 1;
        out.write(reinterpret_cast<const char*>(&num_coeffs), sizeof(num_coeffs));
        out.write(reinterpret_cast<const char*>(coeffs), num_coeffs * sizeof(Fr_t));
    }
}

// Loads a proof from a binary file into a vector of Polynomials.
std::vector<Polynomial> load_proof(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

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

    // Save weight proof (Polynomial vector)
    uint64_t weight_size = proof.weight_proof.size();
    out.write(reinterpret_cast<const char*>(&weight_size), sizeof(weight_size));
    for (const auto& poly : proof.weight_proof) {
        int degree = poly.get_degree();
        Fr_t* coeffs = poly.get_coeffs();
        uint64_t num_coeffs = degree + 1;
        out.write(reinterpret_cast<const char*>(&num_coeffs), sizeof(num_coeffs));
        out.write(reinterpret_cast<const char*>(coeffs), num_coeffs * sizeof(Fr_t));
    }

    // Save rs1 proof (Polynomial vector)
    uint64_t rs1_size = proof.rs1_proof.size();
    out.write(reinterpret_cast<const char*>(&rs1_size), sizeof(rs1_size));
    for (const auto& poly : proof.rs1_proof) {
        int degree = poly.get_degree();
        Fr_t* coeffs = poly.get_coeffs();
        uint64_t num_coeffs = degree + 1;
        out.write(reinterpret_cast<const char*>(&num_coeffs), sizeof(num_coeffs));
        out.write(reinterpret_cast<const char*>(coeffs), num_coeffs * sizeof(Fr_t));
    }

    // Save rs2 proof (Polynomial vector)
    uint64_t rs2_size = proof.rs2_proof.size();
    out.write(reinterpret_cast<const char*>(&rs2_size), sizeof(rs2_size));
    for (const auto& poly : proof.rs2_proof) {
        int degree = poly.get_degree();
        Fr_t* coeffs = poly.get_coeffs();
        uint64_t num_coeffs = degree + 1;
        out.write(reinterpret_cast<const char*>(&num_coeffs), sizeof(num_coeffs));
        out.write(reinterpret_cast<const char*>(coeffs), num_coeffs * sizeof(Fr_t));
    }

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

    // Load weight proof
    uint64_t weight_size;
    in.read(reinterpret_cast<char*>(&weight_size), sizeof(weight_size));
    for (uint64_t i = 0; i < weight_size; ++i) {
        uint64_t num_coeffs;
        in.read(reinterpret_cast<char*>(&num_coeffs), sizeof(num_coeffs));
        std::vector<Fr_t> coeffs(num_coeffs);
        in.read(reinterpret_cast<char*>(coeffs.data()), num_coeffs * sizeof(Fr_t));
        proof.weight_proof.emplace_back(std::move(coeffs));
    }

    // Load rs1 proof
    uint64_t rs1_size;
    in.read(reinterpret_cast<char*>(&rs1_size), sizeof(rs1_size));
    for (uint64_t i = 0; i < rs1_size; ++i) {
        uint64_t num_coeffs;
        in.read(reinterpret_cast<char*>(&num_coeffs), sizeof(num_coeffs));
        std::vector<Fr_t> coeffs(num_coeffs);
        in.read(reinterpret_cast<char*>(coeffs.data()), num_coeffs * sizeof(Fr_t));
        proof.rs1_proof.emplace_back(std::move(coeffs));
    }

    // Load rs2 proof
    uint64_t rs2_size;
    in.read(reinterpret_cast<char*>(&rs2_size), sizeof(rs2_size));
    for (uint64_t i = 0; i < rs2_size; ++i) {
        uint64_t num_coeffs;
        in.read(reinterpret_cast<char*>(&num_coeffs), sizeof(num_coeffs));
        std::vector<Fr_t> coeffs(num_coeffs);
        in.read(reinterpret_cast<char*>(coeffs.data()), num_coeffs * sizeof(Fr_t));
        proof.rs2_proof.emplace_back(std::move(coeffs));
    }

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

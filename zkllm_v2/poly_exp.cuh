#ifndef POLY_EXP_CUH
#define POLY_EXP_CUH

#include "fr-tensor.cuh"
#include "polynomial_v2.cuh"
#include <array>
#include <vector>

// Polynomial exponential using Taylor series
// exp(x) ≈ 1 + x + x²/2! + x³/3! + ... + x^n/n!
// Returns the exponential value in the field
Fr_t poly_exp(const Fr_t& x, int terms = 10) {
    // Precomputed factorial inverses (1 / i!) cached once to avoid per-element inversions.
    static const uint factorials[] = {
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800
    };
    static const std::array<Fr_t, 12> factorial_invs = [] {
        std::array<Fr_t, 12> invs{};
        invs[0] = {1, 0, 0, 0, 0, 0, 0, 0};
        invs[1] = {1, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 2; i < 12; ++i) {
            Fr_t f = {factorials[i], 0, 0, 0, 0, 0, 0, 0};
            invs[i] = inv(f);
        }
        return invs;
    }();

    Fr_t result = {1, 0, 0, 0, 0, 0, 0, 0};  // Start with 1
    Fr_t x_power = {1, 0, 0, 0, 0, 0, 0, 0};  // x^0 = 1

    const int max_terms = (terms < 11) ? terms : 11;
    for (int i = 1; i <= max_terms; i++) {
        x_power = x_power * x;  // x^i
        Fr_t term = x_power * factorial_invs[i];  // x^i / i!
        result = result + term;
    }

    return result;
}

// Batch exponential computation for a tensor
void poly_exp_batch(const FrTensor& input, FrTensor& output, int terms = 10) {
    uint size = input.size;
    
    // Download input to CPU
    Fr_t* input_cpu = new Fr_t[size];
    cudaMemcpy(input_cpu, input.gpu_data, size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    
    // Compute exp for each element
    Fr_t* output_cpu = new Fr_t[size];
    for (uint i = 0; i < size; i++) {
        output_cpu[i] = poly_exp(input_cpu[i], terms);
    }
    
    // Upload result to GPU
    cudaMemcpy(output.gpu_data, output_cpu, size * sizeof(Fr_t), cudaMemcpyHostToDevice);
    
    delete[] input_cpu;
    delete[] output_cpu;
}

// Compute sum of all elements in a tensor (reduction)
Fr_t tensor_sum(const FrTensor& tensor) {
    uint size = tensor.size;
    
    // Download to CPU
    Fr_t* data = new Fr_t[size];
    cudaMemcpy(data, tensor.gpu_data, size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    
    // Sum all elements
    Fr_t sum = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint i = 0; i < size; i++) {
        sum = sum + data[i];
    }
    
    delete[] data;
    return sum;
}

#endif // POLY_EXP_CUH

#ifndef ZKSOFTMAX_V2_CUH 
#define ZKSOFTMAX_V2_CUH

#include "tlookup_v2.cuh"
#include "zkfc_v2.cuh"
#include "polynomial_v2.cuh"
#include "proof_v2.cuh"
#include <vector>

class zkSoftmax_v2 {
    public:
    zkSoftmax_v2(const std::vector<uint>& bs, uint L, uint M, unsigned long scaling_factor_in, 
                 const std::vector<double>& thetas, uint m, uint n, uint d, uint E);

    FrTensor compute(const FrTensor& X, FrTensor& shift, FrTensor& X_shifted, 
                     std::vector<FrTensor>& X_segments, std::vector<FrTensor>& Y_segments, 
                     std::vector<FrTensor>& m_segments);

    Fr_t prove(const FrTensor& Y, const FrTensor& X, const FrTensor& shift, const FrTensor& X_shifted,
        const std::vector<FrTensor>& X_segments, const std::vector<FrTensor>& Y_segments, 
        const std::vector<FrTensor>& m_segments,
        const std::vector<Fr_t>& u_Y, const std::vector<Fr_t>& v_Y,
        const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg, 
        std::vector<Polynomial>& proof);

    protected:
    const std::vector<uint> bs;
    const uint K, L, M; // the number of most and least significant segments
    const unsigned long scaling_factor_in; // the input of scaling factor (gamma**2)
    const std::vector<double> thetas; // the output scaling factor for each segment
    const uint m, n, d; // the dimensions of the input
    const uint E; // the error of the output
    
    std::vector<tLookupRange> least_significant_segments; // the lookup table for the least significant segments
    std::vector<tLookupRangeMapping> other_segments; // the lookup table for other segments
};

class zkAttn_v2 : public zkSoftmax_v2 {
    public:
    zkAttn_v2(unsigned long sf_Q, unsigned long sf_K, const std::vector<uint>& bs, uint L, uint M, 
              const std::vector<double>& thetas, uint m, uint n, uint d, uint E);

    // Q: m * d, K: n * d, V: n * d
    FrTensor compute(const FrTensor& Q, const FrTensor& K, const FrTensor& V, FrTensor& sm_in, FrTensor& sm_out,
        FrTensor& sm_shift, FrTensor& sm_in_shifted, std::vector<FrTensor>& sm_in_segments, 
        std::vector<FrTensor>& sm_out_segments, std::vector<FrTensor>& sm_m_segments);

    Fr_t prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const std::vector<FrTensor>& sm_in_segments, const std::vector<FrTensor>& sm_out_segments, 
        const std::vector<FrTensor>& sm_m_segments,
        const std::vector<Fr_t>& u_matmul_out, const std::vector<Fr_t>& v_matmul_out, 
        const std::vector<Fr_t>& w_matmul_out, 
        const std::vector<Fr_t>& v_sm, const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg, 
        const std::vector<Fr_t>& v_matmul_in,
        std::vector<Polynomial>& proof);

    std::vector<Claim> prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const std::vector<FrTensor>& sm_in_segments, const std::vector<FrTensor>& sm_out_segments, 
        const std::vector<FrTensor>& sm_m_segments);
};

class zkAttnStacked_v2 : public zkAttn_v2 {
    public:
    zkAttnStacked_v2(uint num, unsigned long sf_Q, unsigned long sf_K, const std::vector<uint>& bs, uint L, uint M, 
                     const std::vector<double>& thetas, uint m, uint n, uint d, uint E);

    Fr_t prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const std::vector<FrTensor>& sm_in_segments, const std::vector<FrTensor>& sm_out_segments, 
        const std::vector<FrTensor>& sm_m_segments,
        const std::vector<Fr_t>& u_matmul_out_num, const std::vector<Fr_t>& v_matmul_out_num,
        const std::vector<Fr_t>& u_matmul_out, const std::vector<Fr_t>& v_matmul_out, 
        const std::vector<Fr_t>& w_matmul_out, 
        const std::vector<Fr_t>& v_sm, const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg, 
        const std::vector<Fr_t>& v_matmul_in_num, const std::vector<Fr_t>& v_matmul_in,
        std::vector<Polynomial>& proof);

    std::vector<Claim> prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const std::vector<FrTensor>& sm_in_segments, const std::vector<FrTensor>& sm_out_segments, 
        const std::vector<FrTensor>& sm_m_segments);

    protected:
    const uint num;
};

#endif

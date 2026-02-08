#ifndef TLOOKUP_V2_CUH
#define TLOOKUP_V2_CUH

#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "polynomial_v2.cuh"
#include "proof_v2.cuh"
#include <vector>

class tLookup_v2
{
    public:
    FrTensor table;
    tLookup_v2(const FrTensor& table);
    tLookup_v2(uint size);  // Constructor that creates empty table of given size
    
    // We do not directly use the values from the tensors. Instead, we assume that the tensors have been elementwisely converted to the indices of the table.
    FrTensor prep(const uint* indices, const uint D); // D - dimension of the tensor

    Fr_t prove(const FrTensor& S, const FrTensor& m, const Fr_t& alpha, const Fr_t& beta,
        const std::vector<Fr_t>& u, const std::vector<Fr_t>& v, std::vector<Polynomial>& proof);
};

class tLookupRange : public tLookup_v2
{
    public:
    int low;
    tLookupRange(int low, uint len);
    FrTensor prep(const int* vals, const uint D);
    FrTensor prep(const FrTensor& vals);
};

class tLookupRangeMapping : public tLookupRange
{
    public:
    FrTensor mapped_vals;
    tLookupRangeMapping(int low, uint len, const FrTensor& mvals);
    std::pair<FrTensor, FrTensor> operator()(const int* vals, const uint D);
    std::pair<FrTensor, FrTensor> operator()(const FrTensor& vals);
    Fr_t prove(const FrTensor& S_in, const FrTensor& S_out, const FrTensor& m, 
        const Fr_t& r, const Fr_t& alpha, const Fr_t& beta, 
        const std::vector<Fr_t>& u, const std::vector<Fr_t>& v, std::vector<Polynomial>& proof);
};


#endif
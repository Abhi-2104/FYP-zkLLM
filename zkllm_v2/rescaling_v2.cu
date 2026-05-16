#include "rescaling_v2.cuh"

Rescaling::Rescaling(uint scaling_factor): scaling_factor(scaling_factor), tl_rem(-static_cast<int>(scaling_factor>>1), scaling_factor), rem_tensor_ptr(nullptr)
{
}

// void decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem, FrTensor& rem_ind);
KERNEL void rescaling_kernel(Fr_t* in_ptr, Fr_t* out_ptr, Fr_t* rem_ptr, long scaling_factor, uint N)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {   
        long hsf = scaling_factor >> 1;
        long x = scalar_to_long(in_ptr[tid]);
        long temp = (x + hsf) % scaling_factor;
        long x_rem = (temp < 0 ? temp + scaling_factor : temp) - hsf;
        long x_rescaled = (x - x_rem) / scaling_factor;
        out_ptr[tid] = long_to_scalar(x_rescaled);
        rem_ptr[tid] = long_to_scalar(x_rem);
    }
}

FrTensor Rescaling::operator()(const FrTensor& X)
{
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    rem_tensor_ptr = new FrTensor(X.size);

    FrTensor out(X.size);
    uint block_size = 256;
    rescaling_kernel<<<(X.size + block_size - 1) / block_size, block_size>>>(X.gpu_data, out.gpu_data, rem_tensor_ptr->gpu_data, scaling_factor, X.size);
    cudaDeviceSynchronize();
    
    return out;
}

Rescaling::~Rescaling()
{
    if (rem_tensor_ptr) delete rem_tensor_ptr;
}

vector<Claim> Rescaling::prove(const FrTensor& X, const FrTensor& X_)
{
    if (X.size != X_.size)
    {
        throw std::runtime_error("Error: the size of X and X_ should be the same.");
    }

    // Challenge for the sanity check
    auto u_check = random_vec(ceilLog2(X.size));

    auto rem = rem_tensor_ptr -> pad({rem_tensor_ptr -> size});
    
    // Ensure rem.size >= table.size and is a multiple of table.size (for tLookup)
    uint table_size = tl_rem.table.size;
    if (rem.size < table_size) {
        uint padded_size = table_size; // table_size is a power of 2
        FrTensor rem_padded(padded_size);
        cudaMemcpy(rem_padded.gpu_data, rem.gpu_data, sizeof(Fr_t) * rem.size, cudaMemcpyDeviceToDevice);
        cudaMemset(rem_padded.gpu_data + rem.size, 0, sizeof(Fr_t) * (padded_size - rem.size));
        rem = std::move(rem_padded);
    }

    // Challenges for the tLookup proof
    auto u_proof = random_vec(ceilLog2(rem.size));
    auto v_proof = random_vec(ceilLog2(rem.size));

    auto rand_temp = random_vec(2);
    vector<Polynomial> proof;

    auto m = tl_rem.prep(rem);

    // cout << X << endl;
    // cout << X_ << endl;
    // cout << rem << endl;
    // cout << m << endl;
    // cout << tl_rem.table << endl;
    
    if (X(u_check) != X_(u_check) * Fr_t({scaling_factor, 0, 0, 0, 0, 0, 0, 0}) + (*rem_tensor_ptr)(u_check))
    {
        throw std::runtime_error("Error: the rem is not correct.");
    }
    // cout << rem << endl;
    // cout << rem.sum() << endl;
    // cout << m << endl;
    // cout << tl_rem.table << endl;
    // cout << m*tl_rem.table << endl;
    // cout << (m*tl_rem.table).sum() << endl;
    tl_rem.prove(rem, m, rand_temp[0], rand_temp[1], u_proof, v_proof, proof);

    
    
    cout << "Rescaling proof complete." << endl;
    return {};
}
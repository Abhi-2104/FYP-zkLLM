#include "fr-tensor.cuh"
#include "commitment.cuh"

// Custom extract kernel to pad on-the-fly for a specific chunk, saving gigabytes of VRAM
KERNEL void Extract_chunk_kernel(const GLOBAL Fr_t* arr_in, GLOBAL Fr_t* arr_out, uint chunk_elements, unsigned long long offset, uint last_dim_in, uint last_dim_out, Fr_t pad_val)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= chunk_elements) return;
    
    unsigned long long global_idx = offset + gid;
    unsigned long long gid0 = global_idx / last_dim_out;
    uint gid1 = global_idx % last_dim_out;
    
    if (gid1 >= last_dim_in) arr_out[gid] = pad_val;
    else arr_out[gid] = arr_in[gid0 * last_dim_in + gid1];
}

int main(int argc, char *argv[])
{
    string generator_filename = argv[1];
    string param_filename = argv[2];
    string output_filename = argv[3];
    uint in_dim = std::stoi(argv[4]);
    uint out_dim = std::stoi(argv[5]);

    Commitment generator(generator_filename);
    if (generator.size != (1 << ceilLog2(generator.size))) throw std::runtime_error("Generator size has to be a power of 2");

    FrTensor param = FrTensor::from_int_bin(param_filename);
    cout << "Param size: " << param.size << endl;
    cout << "In dim: " << in_dim << endl;
    cout << "Out dim: " << out_dim << endl;
    
    // MEMORY OPTIMIZATION: On-the-fly chunk padding to keep VRAM < 3GB
    Fr_t pad_val = {0, 0, 0, 0, 0, 0, 0, 0};
    uint last_dim = out_dim;
    uint last_dim_padded = 1 << ceilLog2(last_dim);
    
    uint step1_size = (param.size / last_dim) * last_dim_padded;
    uint step2_size = 1 << ceilLog2(step1_size);
    uint m_total = step2_size / generator.size;
    uint m_real = step1_size / generator.size;
    
    G1TensorJacobian com(m_total);
    
    cout << "Generator size: " << generator.size << endl;
    cout << "Unpadded param size: " << param.size << endl;
    cout << "Padded param size: " << step2_size << " (optimized in memory with streaming chunks)" << endl;
    
    // Process in chunks of rows
    uint chunk_size = 8; 
    for (uint i = 0; i < m_real; i += chunk_size) {
        uint current_chunk = std::min(chunk_size, m_real - i);
        unsigned long long offset = (unsigned long long)i * generator.size;
        uint chunk_elements = current_chunk * generator.size;
        
        FrTensor chunk_fr(chunk_elements);
        Extract_chunk_kernel<<<(chunk_elements+255)/256, 256>>>(param.gpu_data, chunk_fr.gpu_data, chunk_elements, offset, last_dim, last_dim_padded, pad_val);
        cudaDeviceSynchronize();
        
        G1TensorJacobian chunk_com = generator.commit_int(chunk_fr);
        
        cudaDeviceSynchronize();
        cudaError_t err = cudaMemcpy(com.gpu_data + i, chunk_com.gpu_data, current_chunk * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            cerr << "Error: cudaMemcpy failed in chunk copy: " << cudaGetErrorString(err) << endl;
            return 1;
        }
        cudaDeviceSynchronize();
    }
    
    if (m_total > m_real) {
        std::vector<G1Jacobian_t> zeros(m_total - m_real, blstrs__g1__G1Affine_ZERO);
        cudaError_t err = cudaMemcpy(com.gpu_data + m_real, zeros.data(), zeros.size() * sizeof(G1Jacobian_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cerr << "Error: cudaMemcpy failed in zero pad copy: " << cudaGetErrorString(err) << endl;
            return 1;
        }
    }
    
    cout << "Commitment size: " << com.size << endl;
    com.save(output_filename);
    cout << "Commitment computed successfully." << endl;
    return 0;
}



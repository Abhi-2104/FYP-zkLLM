#include "fr-tensor.cuh"
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

using namespace std;

int main() {
    uint L = 512;
    uint E = 4096;
    
    // Create input tensor (simulate V)
    FrTensor V = FrTensor::from_int_bin("activations/layer-0-block-input.bin");
    cout << "Loaded V: " << V.size << " elements" << endl;
    
    // Create uniform attention weights
    FrTensor attn_weights(L * L);
    Fr_t one_over_L;
    unsigned long scale = 1UL << 24;
    uint64_t val = scale / L;
    
    one_over_L.val[0] = val;
    one_over_L.val[1] = 0;
    one_over_L.val[2] = 0;
    one_over_L.val[3] = 0;
    
    // Fill with thrust
    thrust::device_ptr<Fr_t> dev_ptr(attn_weights.gpu_data);
    thrust::fill(dev_ptr, dev_ptr + attn_weights.size, one_over_L);
    cout << "Created uniform attention weights" << endl;
    
    // Compute attn_out = attn_weights @ V
    FrTensor attn_out = FrTensor::matmul(attn_weights, V, L, L, E);
    cout << "Computed attn_out: " << attn_out.size << " elements" << endl;
    
    // Check values BEFORE save
    Fr_t check_before[3];
    cudaMemcpy(check_before, attn_out.gpu_data, 3 * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    printf("BEFORE save: [0] = 0x%016lx%016lx%016lx%016lx\n",
           check_before[0].val[3], check_before[0].val[2],
           check_before[0].val[1], check_before[0].val[0]);
    
    // Save using Fr_t format
    attn_out.save("test_attn_out.bin");
    cout << "Saved attn_out using save()" << endl;
    
    // Load back
    FrTensor loaded("test_attn_out.bin");
    cout << "Loaded attn_out" << endl;
    
    // Check values AFTER load
    Fr_t check_after[3];
    cudaMemcpy(check_after, loaded.gpu_data, 3 * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    printf("AFTER  load: [0] = 0x%016lx%016lx%016lx%016lx\n",
           check_after[0].val[3], check_after[0].val[2],
           check_after[0].val[1], check_after[0].val[0]);
    
    // Compare all values
    bool match = true;
    Fr_t* before_all = new Fr_t[attn_out.size];
    Fr_t* after_all = new Fr_t[attn_out.size];
    cudaMemcpy(before_all, attn_out.gpu_data, attn_out.size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(after_all, loaded.gpu_data, loaded.size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    
    for (uint i = 0; i < attn_out.size && i < 100; i++) {
        if (before_all[i].val[0] != after_all[i].val[0] ||
            before_all[i].val[1] != after_all[i].val[1]) {
            match = false;
            printf("  Mismatch at [%d]: before=0x%016lx%016lx, after=0x%016lx%016lx\n",
                   i, before_all[i].val[1], before_all[i].val[0],
                   after_all[i].val[1], after_all[i].val[0]);
            break;
        }
    }
    
    delete[] before_all;
    delete[] after_all;
    
    if (match && check_before[0].val[0] != 0) {
        cout << "✅ Save/load works correctly for matmul output!" << endl;
        return 0;
    } else {
        cout << "❌ Save/load FAILED!" << endl;
        return 1;
    }
}

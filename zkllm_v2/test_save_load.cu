#include "fr-tensor.cuh"
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

using namespace std;

int main() {
    uint L = 512;
    
    // Create tensor with uniform values
    FrTensor test(L * L);
    Fr_t uniform_val;
    uniform_val.val[0] = 0x12345678;
    uniform_val.val[1] = 0xabcdef00;
    uniform_val.val[2] = 0;
    uniform_val.val[3] = 0;
    
    // Fill with thrust
    thrust::device_ptr<Fr_t> dev_ptr(test.gpu_data);
    thrust::fill(dev_ptr, dev_ptr + test.size, uniform_val);
    
    // Check values BEFORE save
    Fr_t check_before[3];
    cudaMemcpy(check_before, test.gpu_data, 3 * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    printf("BEFORE save: [0] = 0x%08x%08x%08x%08x\n",
           check_before[0].val[3], check_before[0].val[2],
           check_before[0].val[1], check_before[0].val[0]);
    
    // Save as FR_T (not int)
    test.save("test_tensor_frt.bin");
    
    // Load back
    FrTensor loaded_frt("test_tensor_frt.bin");
    
    // Check values AFTER load
    Fr_t check_frt[3];
    cudaMemcpy(check_frt, loaded_frt.gpu_data, 3 * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    printf("AFTER  load (Fr_t): [0] = 0x%08x%08x%08x%08x\n",
           check_frt[0].val[3], check_frt[0].val[2],
           check_frt[0].val[1], check_frt[0].val[0]);
    
    if (check_before[0].val[0] == check_frt[0].val[0] &&
        check_before[0].val[1] == check_frt[0].val[1]) {
        cout << "✅ Save/load (Fr_t) works correctly!" << endl;
    } else {
        cout << "❌ Save/load (Fr_t) FAILED!" << endl;
    }
    
    // NOW test save_int
    test.save_int("test_tensor.bin");
    
    // Load back
    FrTensor loaded = FrTensor::from_int_bin("test_tensor.bin");
    
    // Check values AFTER load
    Fr_t check_after[3];
    cudaMemcpy(check_after, loaded.gpu_data, 3 * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    printf("AFTER  load (int):  [0] = 0x%08x%08x%08x%08x\n",
           check_after[0].val[3], check_after[0].val[2],
           check_after[0].val[1], check_after[0].val[0]);
    
    if (check_before[0].val[0] == check_after[0].val[0] &&
        check_before[0].val[1] == check_after[0].val[1]) {
        cout << "✅ Save/load works correctly!" << endl;
    } else {
        cout << "❌ Save/load FAILED - data corrupted!" << endl;
    }
    
    return 0;
}

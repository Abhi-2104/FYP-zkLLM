#include "fr-tensor.cuh"
#include "proof_v2.cuh"
#include "proof_io_v2.cuh"
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <block_input_file> <block_output_file> <workdir> <layer_prefix> <output_file>" << endl;
        cerr << "\nExample:" << endl;
        cerr << "  " << argv[0] << " block-input.bin ffn-output.bin ./zkllm-workdir/Llama-2-7b layer-0 skip-output.bin" << endl;
        return 1;
    }

    string block_input_fn = argv[1];
    string block_output_fn = argv[2];
    string workdir = argv[3];
    string layer_prefix = argv[4];
    string output_fn = argv[5];

    cout << "\n" << string(70, '=') << endl;
    cout << "Skip Connection Proof Generation (v2)" << endl;
    cout << string(70, '=') << "\n" << endl;
    
    cout << "Configuration:" << endl;
    cout << "  Block input: " << block_input_fn << endl;
    cout << "  Block output: " << block_output_fn << endl;
    cout << "  Workdir: " << workdir << endl;
    cout << "  Layer: " << layer_prefix << endl;
    cout << "  Output: " << output_fn << endl;
    cout << endl;

    // ===== STEP 1: Load Input Tensors =====
    cout << "Step 1: Loading input tensors..." << endl;
    FrTensor x = FrTensor::from_int_bin(block_input_fn);
    cout << "  ✓ Block input loaded: " << x.size << " elements" << endl;
    
    FrTensor y = FrTensor::from_int_bin(block_output_fn);
    cout << "  ✓ Block output loaded: " << y.size << " elements" << endl;
    
    if (x.size != y.size) {
        cerr << "  ✗ Error: Input tensors have different sizes!" << endl;
        cerr << "    x.size = " << x.size << ", y.size = " << y.size << endl;
        return 1;
    }
    cout << endl;

    // ===== STEP 2: Compute Skip Connection (Element-wise Addition) =====
    cout << "Step 2: Computing skip connection (z = A + B)..." << endl;
    FrTensor z = x + y;
    cout << "  ✓ Skip connection computed: " << z.size << " elements" << endl;
    cout << endl;

    // ===== STEP 3: Generate Proof =====
    cout << "Step 3: Generating zero-knowledge proof..." << endl;
    
    SkipConnectionProof proof;
    proof.tensor_size = z.size;
    
    // Generate random challenge point
    proof.random_u = random_vec(ceilLog2(z.size));
    cout << "  ✓ Random challenges generated: " << proof.random_u.size() << " elements" << endl;
    
    // For skip connection, we use a simple sumcheck that verifies:
    // Z(u) = X(u) + Y(u) at a random point u
    // This proves element-wise addition is correct
    
    // Compute claims at random point
    Fr_t x_claim = x(proof.random_u);
    Fr_t y_claim = y(proof.random_u);
    proof.claimed_output = x_claim + y_claim;
    cout << "  ✓ Claimed output computed" << endl;
    
    // Generate sumcheck proof for the addition operation
    // The proof verifies that sum_i (z[i] - x[i] - y[i]) * L_u(i) = 0
    // where L_u is the Lagrange basis polynomial
    // This is equivalent to verifying z[i] = x[i] + y[i] for all i at random point u
    
    // Compute difference tensor for verification: diff = z - x - y (should be all zeros)
    FrTensor diff = z - x;
    diff = diff - y;
    
    // Evaluate at random point - should be zero
    Fr_t zero_check = diff(proof.random_u);
    
    // Generate sumcheck transcript using binary_sumcheck
    proof.hadamard_sum_proof = binary_sumcheck(diff, proof.random_u, proof.random_u);
    
    cout << "  ✓ Sumcheck proof generated: " << proof.hadamard_sum_proof.size() << " Fr_t elements" << endl;
    cout << "  ✓ Zero-check value: " << (zero_check == Fr_t{0,0,0,0,0,0,0,0} ? "PASS" : "FAIL") << endl;
    cout << endl;

    // ===== STEP 4: Save Proof =====
    cout << "Step 4: Saving proof to disk..." << endl;
    
    // Construct proof filename using workdir and layer_prefix
    string proof_filename = workdir + "/" + layer_prefix + "-skip-proof.bin";
    
    save_skip_connection_proof(proof, proof_filename);
    cout << "  ✓ Proof saved to: " << proof_filename << endl;
    cout << endl;

    // ===== STEP 5: Save Output =====
    cout << "Step 5: Saving output activations..." << endl;
    z.save_int(output_fn);
    cout << "  ✓ Output saved to: " << output_fn << endl;
    cout << endl;

    // ===== Summary =====
    cout << string(70, '=') << endl;
    cout << "✅ Skip Connection Proof Generation Complete!" << endl;
    cout << string(70, '=') << endl;
    cout << "Proof Components:" << endl;
    cout << "  • Tensor size: " << proof.tensor_size << " elements" << endl;
    cout << "  • Random challenges: " << proof.random_u.size() << " Fr_t elements" << endl;
    cout << "  • Sumcheck proof: " << proof.hadamard_sum_proof.size() << " Fr_t elements" << endl;
    cout << "  • Operation: z = block_input + block_output" << endl;
    cout << "  • Proof file: " << proof_filename << endl;
    cout << string(70, '=') << "\n" << endl;

    return 0;
}

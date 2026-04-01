#include <iostream>
#include <vector>
#include "fr-tensor.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <tensor_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::cout << "Inspecting tensor: " << filename << std::endl;

    try {
        FrTensor X = FrTensor::from_int_bin(filename);
        std::cout << "Tensor size: " << X.size << " elements" << std::endl;

        int num_to_print = std::min((uint)20, X.size);
        std::cout << "First " << num_to_print << " elements:" << std::endl;
        for (int i = 0; i < num_to_print; i++) {
            std::cout << " [" << i << "]: " << X(i) << std::endl;
        }

        // Check for non-zero elements
        uint nonzero_count = 0;
        for (uint i = 0; i < X.size; i++) {
            Fr_t val = X(i);
            for (int j = 0; j < 8; j++) {
                if (val.val[j] != 0) {
                    nonzero_count++;
                    break;
                }
            }
        }
        std::cout << "Total non-zero elements: " << nonzero_count << " / " << X.size << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

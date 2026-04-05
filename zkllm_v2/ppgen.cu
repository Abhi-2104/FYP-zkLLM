#include "commitment.cuh"
#include <string>

int main(int argc, char *argv[]) {
    unsigned long long size = std::stoull(argv[1]);
    string filename = argv[2];

    Commitment commitment = Commitment::random(1ULL << ceilLog2(size));
    commitment.save(filename);
    std::cout << "Parameters generated successfully." << std::endl;
    return 0;
}
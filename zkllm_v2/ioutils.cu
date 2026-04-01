#include "ioutils.cuh"
#include <iostream>
#include <cstdio>
#include <cstdlib>

void savebin(const string& filename, const void* gpudata, uint size)
{
    // Copy data from GPU to CPU
    void* data = malloc(size);
    if (!data) {
        cerr << "Error: Could not allocate host memory for saving " << filename << endl;
        return;
    }
    
    cudaError_t err = cudaMemcpy(data, gpudata, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed in savebin for " << filename << ": " << cudaGetErrorString(err) << endl;
        free(data);
        return;
    }

    // Write data to file
    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) {
        cerr << "Error: Could not open file for writing: " << filename << endl;
        free(data);
        return;
    }
    
    size_t written = fwrite(data, 1, size, file);
    if (written != size) {
        cerr << "Error: Expected to write " << size << " bytes, but wrote " << written << " for " << filename << endl;
    }
    fclose(file);
    
    // Free memory
    free(data);
}

uint findsize(const string& filename)
{
    // Read data from file
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        cerr << "Error: Could not open file for reading (size check): " << filename << endl;
        return 0;
    }
    
    if (fseek(file, 0, SEEK_END) != 0) {
        cerr << "Error: fseek failed for " << filename << endl;
        fclose(file);
        return 0;
    }
    
    long size = ftell(file);
    if (size < 0) {
        cerr << "Error: ftell failed for " << filename << endl;
        fclose(file);
        return 0;
    }
    
    fclose(file);
    return (uint)size;
}

void loadbin(const string& filename, void* gpudata, uint size)
{
    if (size == 0) return;

    // Allocate memory
    void* data = malloc(size);
    if (!data) {
        cerr << "Error: Could not allocate host memory for loading " << filename << endl;
        return;
    }

    // Read data from file
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        cerr << "Error: Could not open file for reading: " << filename << endl;
        free(data);
        return;
    }
    
    size_t read = fread(data, 1, size, file);
    if (read != size) {
        cerr << "Error: Expected to read " << size << " bytes, but read " << read << " for " << filename << endl;
    }
    fclose(file);
    
    // Copy data from CPU to GPU
    cudaError_t err = cudaMemcpy(gpudata, data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed in loadbin for " << filename << ": " << cudaGetErrorString(err) << endl;
    }

    // Free memory
    free(data);
}
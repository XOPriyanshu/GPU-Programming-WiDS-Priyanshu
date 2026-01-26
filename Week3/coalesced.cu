#include <stdio.h>
#include <cuda_runtime.h>

// Task 1.1: Coalesced Access
// Threads access memory consecutively: Thread i reads Data[i]
__global__ void coalesced_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Coalesced read and write
        data[idx] = data[idx] * 2.0f; 
    }
}

int main() {
    int n = 1 << 24; // 16M floats
    size_t bytes = n * sizeof(float);
    float *d_data;

    cudaMalloc(&d_data, bytes);
    
    // Warmup
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    coalesced_kernel<<<gridSize, blockSize>>>(d_data, n);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    coalesced_kernel<<<gridSize, blockSize>>>(d_data, n);
    cudaEventRecord(stop);
    
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Coalesced Access Time: %.3f ms\n", milliseconds);
    printf("Effective Bandwidth: %.2f GB/s\n", (2.0f * bytes) / (milliseconds * 1e6));

    cudaFree(d_data);
    return 0;
}

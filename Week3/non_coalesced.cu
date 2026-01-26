#include <stdio.h>
#include <cuda_runtime.h>

// Task 1.2: Non-Coalesced Access (Strided)
// Threads access memory with a stride: Thread i reads Data[i * stride]
__global__ void strided_kernel(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check accounts for the stride to prevent out-of-bounds
    if (idx * stride < n) {
        // Strided read and write (Performance Killer)
        data[idx * stride] = data[idx * stride] * 2.0f; 
    }
}

int main() {
    int n = 1 << 24; // 16M floats
    size_t bytes = n * sizeof(float);
    float *d_data;
    int stride = 32; // Stride of 32 breaks coalescing completely

    cudaMalloc(&d_data, bytes);
    
    // Kernel configuration
    int blockSize = 256;
    // Grid must cover the *indices*, not the total array size
    int num_elements_accessed = n / stride;
    int gridSize = (num_elements_accessed + blockSize - 1) / blockSize;

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    strided_kernel<<<gridSize, blockSize>>>(d_data, n, stride);
    cudaEventRecord(stop);
    
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Non-Coalesced (Stride %d) Time: %.3f ms\n", stride, milliseconds);
    // Note: We transfer 32x less data, but check the bandwidth utilization!
    size_t active_bytes = (n / stride) * sizeof(float); 
    printf("Effective Bandwidth: %.2f GB/s\n", (2.0f * active_bytes) / (milliseconds * 1e6));

    cudaFree(d_data);
    return 0;
}

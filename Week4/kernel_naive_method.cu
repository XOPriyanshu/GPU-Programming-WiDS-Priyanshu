#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 4096
#define K 4096

// Error checking macro
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// NAIVE KERNEL
__global__ void gemv_naive(const float *A, const float *x, float *y, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * x[i];
        }
        y[row] = sum;
    }
}

int main() {
    printf("--- Benchmarking Naive GEMV (%dx%d) ---\n", N, K);
    
    size_t bytes_A = N * K * sizeof(float);
    size_t bytes_x = K * sizeof(float);
    size_t bytes_y = N * sizeof(float);

    // Host Memory
    float *h_A = (float*)malloc(bytes_A);
    float *h_x = (float*)malloc(bytes_x);
    float *h_y = (float*)malloc(bytes_y);

    // Initialize
    for(int i=0; i<N*K; i++) h_A[i] = 1.0f;
    for(int i=0; i<K; i++) h_x[i] = 1.0f;

    // Device Memory
    float *d_A, *d_x, *d_y;
    CHECK(cudaMalloc(&d_A, bytes_A));
    CHECK(cudaMalloc(&d_x, bytes_x));
    CHECK(cudaMalloc(&d_y, bytes_y));

    // Copy to Device
    CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    gemv_naive<<<gridSize, blockSize>>>(d_A, d_x, d_y, N, K);
    CHECK(cudaDeviceSynchronize());

    // Record
    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        gemv_naive<<<gridSize, blockSize>>>(d_A, d_x, d_y, N, K);
    }
    cudaEventRecord(stop);
    CHECK(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Naive Average Time: %.3f ms\n", ms / 100.0f);

    // Cleanup
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    free(h_A); free(h_x); free(h_y);
    
    return 0;
}

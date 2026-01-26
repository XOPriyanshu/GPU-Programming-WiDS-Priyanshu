#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4096
#define K 4096

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

// OPTIMIZED KERNEL (Shared Memory)
__global__ void gemv_optimized(const float *A, const float *x, float *y, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Shared Memory to cache the vector X
    __shared__ float s_x[4096];

    // Collaborative loading: All threads load a piece of X into Shared Mem
    for (int i = tid; i < k; i += blockDim.x) {
        s_x[i] = x[i];
    }
    __syncthreads(); // Wait until s_x is fully loaded

    if (row < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            // Read A from Global, Read X from Shared
            sum += A[row * k + i] * s_x[i];
        }
        y[row] = sum;
    }
}

int main() {
    printf("--- Benchmarking Optimized GEMV (%dx%d) ---\n", N, K);
    
    size_t bytes_A = N * K * sizeof(float);
    size_t bytes_x = K * sizeof(float);
    size_t bytes_y = N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_x = (float*)malloc(bytes_x);
    float *h_y = (float*)malloc(bytes_y);

    for(int i=0; i<N*K; i++) h_A[i] = 1.0f;
    for(int i=0; i<K; i++) h_x[i] = 1.0f;

    float *d_A, *d_x, *d_y;
    CHECK(cudaMalloc(&d_A, bytes_A));
    CHECK(cudaMalloc(&d_x, bytes_x));
    CHECK(cudaMalloc(&d_y, bytes_y));

    CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    gemv_optimized<<<gridSize, blockSize>>>(d_A, d_x, d_y, N, K);
    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for(int i=0; i<100; i++) {
        gemv_optimized<<<gridSize, blockSize>>>(d_A, d_x, d_y, N, K);
    }
    cudaEventRecord(stop);
    CHECK(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Optimized Average Time: %.3f ms\n", ms / 100.0f);

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    free(h_A); free(h_x); free(h_y);
    
    return 0;
}

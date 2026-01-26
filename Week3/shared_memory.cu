#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define TILE_WIDTH 32

// Baseline: Global Memory Only
__global__ void matmul_naive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized: Shared Memory Tiling
__global__ void matmul_shared(float *A, float *B, float *C, int N) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int col_A = t * TILE_WIDTH + threadIdx.x;
        int row_B = t * TILE_WIDTH + threadIdx.y;
        
        // Load A tile
        if (row < N && col_A < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + col_A];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        if (row_B < N && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[row_B * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads(); // Wait for load
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads(); // Wait for compute before next load
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 2048; // Large enough to see difference
    size_t bytes = N * N * sizeof(float);
    printf("Matrix Multiplication (N=%d)\n", N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Initialize random data (omitted for brevity, assume populated)

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Timing Naive
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float naive_time = 0;
    cudaEventElapsedTime(&naive_time, start, stop);
    printf("Naive Time: %.3f ms\n", naive_time);

    // Timing Shared
    cudaEventRecord(start);
    matmul_shared<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float shared_time = 0;
    cudaEventElapsedTime(&shared_time, start, stop);
    printf("Shared Time: %.3f ms\n", shared_time);
    
    printf("Speedup: %.2fx\n", naive_time / shared_time);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

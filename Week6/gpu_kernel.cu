#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
 * Naive CUDA GEMM kernel
 * C = A x B
 */
__global__ void gemm_kernel(const float *A, const float *B, float *C, int N) {
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

int main(int argc, char *argv[]) {
    // -----------------------------
    // Matrix size
    // -----------------------------
    int N = 512;   // default
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("Running GPU GEMM for N = %d\n", N);

    size_t size = (size_t)N * N * sizeof(float);

    // -----------------------------
    // Host memory
    // -----------------------------
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    if (!h_A || !h_B || !h_C) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // -----------------------------
    // Device memory
    // -----------------------------
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // -----------------------------
    // Kernel launch config
    // -----------------------------
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    // -----------------------------
    // CUDA timing events
    // -----------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("GPU GEMM Time: %.3f ms\n", ms);

    // -----------------------------
    // Cleanup
    // -----------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

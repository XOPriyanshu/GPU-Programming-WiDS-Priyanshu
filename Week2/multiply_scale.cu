#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GPU kernel
__global__ void multiply_scale(float *a, float *b, float *c, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = alpha * a[i] * b[i];
    }
}

int main() {
    int n = 1000;
    float alpha = 0.5f;
    size_t size = n * sizeof(float);

    // Host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize input
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Kernel launch
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    multiply_scale<<<gridSize, blockSize>>>(d_a, d_b, d_c, alpha, n);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < n; i++) {
        float expected = alpha * h_a[i] * h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            printf("Mismatch at %d\n", i);
            break;
        }
    }
    printf("Multiply & Scale SUCCESS\n");

    // Free memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

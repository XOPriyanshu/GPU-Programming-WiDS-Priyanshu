#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GPU kernel
__global__ void relu(float *a, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
    }
}

int main() {
    int n = 1000;
    size_t size = n * sizeof(float);

    // Host memory
    float *h_a = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize input
    for (int i = 0; i < n; i++) {
        h_a[i] = (i % 2 == 0) ? i * 1.0f : -i * 1.0f;
    }

    // Device memory
    float *d_a, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, size);

    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Kernel launch
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    relu<<<gridSize, blockSize>>>(d_a, d_c, n);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < n; i++) {
        float expected = (h_a[i] > 0.0f) ? h_a[i] : 0.0f;
        if (h_c[i] != expected) {
            printf("Mismatch at %d\n", i);
            break;
        }
    }
    printf("ReLU SUCCESS\n");

    // Free memory
    free(h_a); free(h_c);
    cudaFree(d_a); cudaFree(d_c);

    return 0;
}

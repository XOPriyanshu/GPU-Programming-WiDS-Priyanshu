#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GPU kernel
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 10000000;
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

    // Copy data to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Kernel launch
 int blockSize = 512;
int gridSize = (n + blockSize - 1) / blockSize;

// ---- TIMING START ----
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);

printf("Input:%d | Block size: %d | Kernel time: %f ms\n",n, blockSize, ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
// ---- TIMING END ----


    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Mismatch at %d\n", i);
            break;
        }
    }
    printf("Vector Add SUCCESS\n");

    // Free memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

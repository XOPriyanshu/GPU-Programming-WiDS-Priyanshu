import numpy as np

def cpu_gemm(A, B):
    """
    Naive CPU reference GEMM
    """
    N = A.shape[0]
    C = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]

    return C


if __name__ == "__main__":
    # -----------------------------
    # Use SMALL N for correctness
    # -----------------------------
    N = 128
    print(f"Running correctness check for N = {N}")

    # -----------------------------
    # Same initialization as CUDA
    # -----------------------------
    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)

    # -----------------------------
    # CPU reference computation
    # -----------------------------
    C_cpu = cpu_gemm(A, B)

    # -----------------------------
    # Expected analytical result
    # -----------------------------
    # For A=1 and B=1:
    # C[i][j] = sum_k (1 * 1) = N
    expected_value = float(N)
    C_expected = np.full((N, N), expected_value, dtype=np.float32)

    # -----------------------------
    # Error analysis
    # -----------------------------
    max_error = np.max(np.abs(C_cpu - C_expected))

    print(f"Max absolute error: {max_error:e}")

    if max_error < 1e-5:
        print("✅ CORRECTNESS CHECK PASSED")
    else:
        print("❌ CORRECTNESS CHECK FAILED")

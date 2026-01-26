import numpy as np
import time
import sys

def cpu_gemm(A, B, show_progress=False):
    N = A.shape[0]
    C = np.zeros((N, N), dtype=np.float32)

    start_time = time.perf_counter()

    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

        # Progress update per row (low overhead)
        if show_progress:
            completed = (i + 1) / N * 100
            elapsed = time.perf_counter() - start_time
            eta = (elapsed / (i + 1)) * (N - i - 1) if i > 0 else 0.0

            print(
                f"\rProgress: {completed:6.2f}% | "
                f"Elapsed: {elapsed:6.1f}s | "
                f"ETA: {eta:6.1f}s",
                end=""
            )

    if show_progress:
        print()  # newline after completion

    return C


if __name__ == "__main__":
    # -----------------------------
    # Input size
    # -----------------------------
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 512

    print(f"Running CPU GEMM for N = {N}")

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # -----------------------------
    # Warm-up (no progress shown)
    # -----------------------------
    cpu_gemm(A, B, show_progress=False)

    # -----------------------------
    # Timed runs
    # -----------------------------
    runs = 1  # keep 1 when progress is enabled
    times = []

    for r in range(runs):
        print(f"\nRun {r+1}:")
        start = time.perf_counter()
        cpu_gemm(A, B, show_progress=True)
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        print(f"Run {r+1} time: {elapsed:.2f} seconds")

    avg_time = sum(times) / runs

    print("\n==== CPU Baseline Results ====")
    print(f"Matrix size     : {N} x {N}")
    print(f"Average runtime : {avg_time:.2f} seconds")
    print(f"Time complexity : O(N^3)")

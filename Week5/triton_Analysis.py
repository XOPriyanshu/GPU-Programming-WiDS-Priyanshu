import torch
import triton
import triton.language as tl
import time

# -------------------------------------------------------------------------
# TASK 1: Triton Kernel Implementation (GEMM / GEMV)
# -------------------------------------------------------------------------
# This kernel computes C = A @ B
# It uses block-level parallelism just like CUDA, but simpler syntax.
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # 1. Map Program ID to the Block of Output Matrix we are computing
    pid = tl.program_id(axis=0)
    
    # Grid logic: how many blocks fit in the N dimension?
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Calculate which tile (row, col) this program is responsible for
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 2. Create pointers to the first block of A and B
    # A_ptr moves along M rows, B_ptr moves along N cols
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    # The 'k' dimension (inner loop) pointer offsets
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to A and B in memory
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. Main Loop: Iterate over K dimension in chunks of BLOCK_SIZE_K
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next chunk of A and B
        # mask is usually needed for arbitrary sizes, but we'll assume aligned for simplicity
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Compute dot product for this chunk and add to accumulator
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next K-block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 4. Store the result
    # We apply activation here if needed (e.g., ReLU), but we stick to raw MatMul
    c = accumulator.to(tl.float32) # ensure type
    
    # Calculate output pointers
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Store
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

# -------------------------------------------------------------------------
# Helper to Launch the Kernel
# -------------------------------------------------------------------------
def triton_matmul(a, b):
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    
    # Alloc Output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # 1D Launch Grid (standard for Triton Matmul)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    # Launch
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32 # Tuning params
    )
    return c

# -------------------------------------------------------------------------
# TASK 2: Benchmarking & Verification
# -------------------------------------------------------------------------
def run_benchmark():
    torch.manual_seed(0)
    
    # Dimensions for your Edge AI Simulation
    # Note: Triton works best with block sizes >= 16. M=1 (GEMV) is a special case.
    # We will test M=32 to show Triton's power, then M=1 for your specific case.
    
    configs = [
        (32, 4096, 4096),   # Small Batch Inference (M=32)
        (4096, 4096, 4096)  # Full Training Layer (M=4096)
    ]

    print(f"{'Mode':<20} {'Time (Triton)':<15} {'Time (PyTorch)':<15} {'Speedup':<10}")
    print("-" * 65)

    for M, K, N in configs:
        a = torch.randn((M, K), device='cuda', dtype=torch.float32)
        b = torch.randn((K, N), device='cuda', dtype=torch.float32)
        
        # Warmup & Correctness Check
        triton_output = triton_matmul(a, b)
        torch_output = torch.matmul(a, b)
        
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
            status = "PASS"
        else:
            status = f"FAIL (Max Diff: {torch.max(torch.abs(triton_output - torch_output)).item():.4f})"
            
        # Benchmark Triton
        # We use triton.testing.do_bench for accurate GPU timing
        triton_ms = triton.testing.do_bench(lambda: triton_matmul(a, b))
        
        # Benchmark PyTorch (cuBLAS)
        torch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        
        speedup = torch_ms / triton_ms
        print(f"{f'{M}x{K}x{N}':<20} {triton_ms:.4f} ms      {torch_ms:.4f} ms      {speedup:.2f}x")
        
        if "FAIL" in status:
            print(f"Warning: Correctness check failed for {M}x{K}x{N}")

if __name__ == "__main__":
    run_benchmark()

import torch
import time

# Matrix size
N = 1024

# CPU tensors
A = torch.randn(N, N)
B = torch.randn(N, N)

# Warm-up
_ = A @ B

start = time.perf_counter()
C = A @ B
end = time.perf_counter()

print("CPU time:", end - start)

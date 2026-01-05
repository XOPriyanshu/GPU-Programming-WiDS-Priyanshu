# **Week 4 — Real Compute Kernels & Performance Engineering**

---

## **Learning Goals**

By the end of Week 4, you should be able to:

* Implement non-trivial GPU kernels used in real workloads
* Apply **tiling and shared memory** to structured computations
* Distinguish between **compute-bound** and **memory-bound** kernels in practice
* Benchmark GPU kernels against optimized CPU and PyTorch baselines
* Reason about performance bottlenecks using measured data

This week prepares you for:

* Triton (Week 5)
* The final mini-project (Week 6)

---

## **Required Resources**

### **1. CUDA C++ Programming Guide — Performance & Execution**

Review relevant sections from:

* **Chapter 8 — Performance Guidelines**
  * Section 8.2 (Maximize Utilization) — occupancy concepts
  * Section 8.3 (Maximize Memory Throughput)
  * Section 8.4 (Maximize Instruction Throughput)

* **Chapter 10 — C++ Language Extensions**
  * Section 10.37 (Execution Configuration)
  * Section 10.38 (Launch Bounds) — for controlling occupancy

 Link:
[https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

Focus on:

* Occupancy (conceptual, not formula-heavy)
* Instruction throughput vs memory throughput
* When more threads stop helping

---

### **2. Matrix Multiplication on GPUs (Canonical Pattern)**

Read **any one** of the following:

* **NVIDIA Blog — An Efficient Matrix Transpose in CUDA C/C++**
  
  Link:
  [https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

* **CUDA Sample: `matrixMul`** (inspect structure only)
  
  GitHub:
  [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)

Focus on:

* Tiling
* Shared memory reuse
* Thread cooperation

---

### **3. Real-World Kernel Examples**

You are not expected to understand every line.

* **tiny-cuda-nn (NVIDIA)**
  
  GitHub:
  [https://github.com/NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

* **FlashAttention (Dao et al.)**
  
 GitHub:
  [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

Look for:

* How kernels are structured
* How memory reuse is emphasized
* How complexity is managed

---

## **Concepts Covered This Week**

* Structured parallelism (tiles, blocks, subproblems)
* Shared memory reuse across threads
* Compute intensity vs memory traffic
* Benchmarking methodology
* Performance vs correctness trade-offs

---

## **Optional but Highly Recommended**

### **1. CUDA Best Practices Guide — Performance Section**

 Link:
[https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

Focus on:

* Memory Optimizations
* Execution Configuration Optimizations
* Instruction Optimization

---

### **2. Nsight Compute — Profiling Deep Dive**

 Official Documentation:
[https://docs.nvidia.com/nsight-compute/NsightCompute/index.html](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)

 Tutorial:
[https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/)

---

### **3. NVIDIA Blogs on GEMM and Tiling Strategies**

 Cutlass GEMM Overview:
[https://github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

Matrix Multiplication Background:
[https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

# **Week 3 — GPU Memory Hierarchy & Performance Optimization**

This week focuses on understanding the GPU memory hierarchy, identifying bottlenecks, and writing kernels based on how GPUs actually move data.

---

##  **Learning Goals**

* Explain the GPU memory hierarchy and its performance implications
* Distinguish between **memory-bound** and **compute-bound** kernels
* Understand **coalesced global memory access**
* Use **shared memory** correctly and avoid common pitfalls
* Identify **bank conflicts** and understand how to mitigate them
* Profile GPU kernels and interpret basic performance metrics

This week is foundational for all serious GPU work, including ML kernels and scientific simulations.

---

## **Required Resources**

### **1. NVIDIA CUDA C++ Programming Guide — Memory Model & Performance**

Read the following sections carefully:

* **Section 5.3 — Memory Hierarchy**
* **Chapter 8 — Performance Guidelines**
  * Section 8.2 (Maximize Utilization) — covers occupancy
  * Section 8.3 (Maximize Memory Throughput) — covers coalescing and access patterns

📄 Link:
[https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

Focus on:

* Global vs shared vs constant memory
* Latency vs bandwidth
* Memory access patterns and coalescing
* Occupancy 

---

### **2. Mark Harris — Coalesced Memory Access (GTC Talk)**

🎥 Video:
[https://www.nvidia.com/en-us/on-demand/session/gtc24-s62550/](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62550/)

Focus on:

* What "coalesced access" means
* Why strided access is slow
* How warps access memory

---

### **3. GPU Gems 3 — Parallel Prefix Sum (Scan)**

Read **Chapter 39**.

📄 Link:
[https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

Focus on:

* Using shared memory to reduce global memory traffic
* Avoiding bank conflicts
* Structuring parallel algorithms around memory constraints

---

##  **Concepts Covered This Week**

* GPU memory hierarchy:

  * Global memory
  * Shared memory
  * Registers
  * Constant memory 
* Memory latency vs bandwidth
* Coalesced vs non-coalesced access
* Shared memory tiling
* Bank conflicts
* Synchronization (`__syncthreads()`)
* Intro to profiling and timing

---

## **What You Will Build This Week**

You will take simple kernels from Week 2 and **make them faster** by:

* Changing memory access patterns
* Introducing shared memory
* Reducing redundant global loads
* Measuring performance improvements

This is the first time we will see **order-of-magnitude speedups**.

---

## **Week 3 Assignment (Summary)**

### **Task 1 — Memory Access Pattern Experiment**

* Implement two versions of a kernel:

  * One with **coalesced** global memory access
  * One with **non-coalesced** (strided) access
* Measure and compare performance
* Explain the difference using warp-level reasoning

---

### **Task 2 — Shared Memory Optimization**

* Implement a kernel that:

  * First reads data directly from global memory
  * Then reimplements the same computation using shared memory
* Use `__syncthreads()` correctly
* Compare runtime and memory behavior

---

### **Task 3 — CPU Baseline Submission (Deadline)**

This is the **final deadline** to submit your CPU baseline for the workload you identified in Week 1.

You must include:

* `cpu_baseline.py`
* Input sizes
* Measured runtime

This baseline will be used to evaluate GPU speedups in Weeks 4–6.

---

## 📁 **Submission Folder**

```
week3/
 ├── assignment.pdf
 ├── coalesced.cu
 ├── non_coalesced.cu
 ├── shared_memory.cu
 ├── cpu_baseline.py
```
---

## Optional but Highly Recommended (Week 3)

### **1. CUDA Best Practices Guide — Memory Optimizations**

📄 Link:
[https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

Relevant sections:

* *Memory Optimizations*
* *Occupancy*
* *Performance Guidelines*

---

### **2. Nsight Compute — Basic Walkthrough**

Learn how to profile kernels and interpret key GPU metrics.

📄 Official Documentation:
[https://docs.nvidia.com/nsight-compute/NsightCompute/index.html](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)

🎥 Introductory Tutorial (NVIDIA):
[https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/)

---

### **3. NVIDIA Blog — Shared Memory & Bank Conflicts**

📄 Shared Memory Overview:
[https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

📄 Bank Conflicts Explained:
[https://github.com/Kobzol/hardware-effects-gpu/blob/master/bank-conflicts/README.md](https://github.com/Kobzol/hardware-effects-gpu/blob/master/bank-conflicts/README.md)

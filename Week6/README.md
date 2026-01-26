# Final Mini-Project — GPU Acceleration of a Real Workload
---

## Overview

The goal of this final project is to **accelerate a real computational workload** using GPU programming.

Unlike earlier assignments, this project is **open-ended and self-directed**. You will identify a bottleneck in a workload you care about (research, engineering, or ML), port it to the GPU using **CUDA and/or Triton**, and **quantitatively evaluate** the performance improvement over a CPU baseline.

This project is meant to resemble **real systems or ML research work**, not a toy assignment.

---

## Project Objective

You will:

1. Select a real compute-heavy workload (from your own work or a public codebase)
2. Identify the performance-critical section
3. Implement a GPU-accelerated version using:
   - CUDA, or
   - Triton, or
   - a combination of both
4. Benchmark CPU vs GPU performance
5. Analyze speedup, bottlenecks, and design trade-offs

---

## Suitable Project Types

Your project may fall into (but is not limited to) one of the following categories:

### Scientific & Numerical Computing
- Simulation step (finite difference, stencil, particle update)
- Numerical kernels (matrix ops, reductions, iterative solvers)
- Signal or image processing kernels

### Machine Learning & Data
- Softmax, normalization, attention-like operations
- Custom PyTorch operations or layers
- Feature extraction or preprocessing kernels

### Systems & Data Processing
- Large-scale array transformations
- Pairwise computations
- Histogramming, filtering, aggregation

If in doubt: **pick something that was slow on CPU**.

---

## Project Requirements

### 1. CPU Baseline
You must provide:
- A CPU implementation
- Input sizes used
- Measured runtime (average over multiple runs)

This may reuse the CPU baseline submitted earlier in the course.

---

### 2. GPU Implementation
You must implement **at least one GPU version** of the bottleneck:

- CUDA kernel(s), and/or
- Triton kernel(s)

Requirements:
- Correctness matching CPU/PyTorch reference
- Proper bounds checking
- Clear kernel structure

---

### 3. Benchmarking
You must benchmark:
- CPU version
- GPU version(s)

Report:
- Runtime
- Speedup factor
- GPU model and environment

---

### 4. Analysis & Discussion
Your analysis should address:
- Where the speedup comes from
- Whether the kernel is memory-bound or compute-bound
- Trade-offs between CUDA and Triton (if applicable)
- Limitations or remaining bottlenecks

---

## Deliverables

Your project directory should look like:

```

final-project/
├── README.md
├── report.pdf
├── cpu_baseline.py
├── gpu_kernel.cu        # if using CUDA
├── triton_kernel.py     # if using Triton
├── correctness_check.py

```

---

## Report Guidelines

Submit a **PDF report (4–6 pages)** including:

1. Problem description
2. CPU baseline
3. GPU design & implementation
4. Benchmark results
5. Performance analysis
6. Lessons learned

---

## Notes & Policies

- You may use CUDA documentation, Triton docs, and public repositories for reference
- You may be asked to orally explain your design and results

---

## Outcome

By the end of this project, you should have:

- A GPU-accelerated version of a real workload
- Quantitative evidence of performance gains
- A reusable codebase
- A project suitable for research or systems interviews


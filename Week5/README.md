# **Week 5 — Triton: Modern GPU Programming for Researchers**

The goal of this week is to understand:

* when Triton is the right tool over CUDA,
* how it maps to GPU hardware,
* and how to rapidly prototype performant kernels for research and ML workloads.

---

##  **Learning Goals**

By the end of Week 5, you should be able to:

* Understand Triton’s programming model and abstractions
* Write GPU kernels in Triton for real compute workloads
* Map Triton concepts (programs, blocks) to CUDA concepts (threads, blocks)
* Reimplement a CUDA kernel from Week 4 in Triton
* Compare **developer productivity vs performance** between CUDA and Triton
* Decide when Triton is sufficient and when raw CUDA is necessary

---

##  **Required Resources**

### **1. Triton Official Documentation**

Work through the tutorials carefully.

[https://github.com/openai/triton](https://github.com/openai/triton)
[https://triton-lang.org/main/index.html](https://triton-lang.org/main/index.html)

Focus on:

* Programming model
* `tl.load`, `tl.store`
* Block pointers
* Autotuning basics

You should understand:

* Vector addition
* Matrix multiplication
* Softmax

These examples mirror exactly what you did in CUDA in Weeks 2–4.

---
---

## **What You Will Build This Week**

You will take a **real kernel from Week 3/4** and:

* Reimplement it in Triton
* Verify correctness
* Benchmark performance
* Compare:

  * Lines of code
  * Development complexity
  * Runtime performance

---

##  **Week 5 Assignment (Summary)**

### **Task 1 — Triton Kernel Implementation**

Choose **one** kernel you implemented in Week 3/4:

* Softmax
* GEMM
* Custom PyTorch operation

Reimplement it using Triton.

Requirements:

* Correctness matching CPU/PyTorch
* Clear mapping from CUDA → Triton logic

---

### **Task 2 — Benchmarking & Comparison**

Benchmark:

* CUDA optimized kernel (Week 4)
* Triton kernel

Report:

* Runtime
* Speedup or slowdown
* Code complexity (qualitative)

---

### **Task 3 — Analysis: CUDA vs Triton**

In your PDF, answer:

* When is Triton preferable?
* When is CUDA unavoidable?
* How would you choose between them for a new research project?

---

##  **Submission Folder**

```
week5/
 ├── assignment.pdf
 ├── triton_kernel.py
 ├── correctness_check.py
```

---

## **Optional but Highly Recommended**

* Triton autotuning examples
* FlashAttention Triton implementation (read-only)
* PyTorch custom ops using Triton

---

##  Looking Ahead

**Week 6** is the **final mini-project**:

* Accelerate *your own* research or engineering workload
* Choose CUDA, Triton, or both
* Quantify the gains
* Present your work like a real systems/ML paper

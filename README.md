### Accelerating Scientific & AI Workflows with CUDA & Triton




Most researchers and engineers already have heavy Python code-simulations, numerical solvers, matrix workloads, data transforms, ML ops—that run slowly on CPUs.
This course teaches you to **accelerate your own real project** using **GPU programming** with **CUDA** and **Triton**.

Over 6 weeks, we will go from **GPU intuition → CUDA fundamentals → memory optimization → real kernels → Triton → a final mini-project**, where you implement and benchmark a GPU-accelerated version of a problem you personally care about.



## Goals:

By the end of this course, you will be able to:

* Understand *when and why* GPUs outperform CPUs
* Map your compute to the CUDA execution model (threads, blocks, grids)
* Use the GPU memory hierarchy effectively (global, shared, constant memory)
* Write and optimize CUDA kernels for math-heavy workloads
* Use **Triton** to write concise, high-performance GPU kernels in Python
* Profile your compute and quantify speedups vs CPU implementations
* Accelerate a real Python-based research problem—or your own project



## Who This Course Is For

* Researchers (CS, physics, engineering, biology, economics, etc.)
* Engineers and developers with compute-heavy workloads
* ML practitioners building custom layers or operations
* Anyone who wants their Python code to run **10×–100× faster**

**Prerequisites:**

* Python and NumPy (required)
* Some C/C++ helpful but **not required**
* Basic linear algebra knowledge



## Course Structure (6 Weeks)

Each week contains:

*  **Concepts & readings** from NVIDIA docs and academic lectures
*  **Hands-on coding assignments**
*  **Weekly submission**
*  **Progress towards the mini-project**

Weekly breakdown:

1. **GPU Intuition & Compute Foundations**
2. **Parallel Thinking & CUDA Basics**
3. **Memory Hierarchy & Performance Optimization**
4. **Real Kernels: GEMM, softmax, compute patterns**
5. **Modern GPU Programming with Triton**
6. **Mini-Project: Accelerate Your Own Research Code**

Each week will have its own markdown file under `weekX.md`.



## Final Mini-Project

You will:

1. Choose a real compute-heavy Python task from your research or interests
2. Profile the CPU implementation
3. Rewrite the bottleneck(s) using **CUDA and/or Triton**
4. Benchmark and visualize the speedup
5. Submit a short write-up or slide deck summarizing the problem, GPU approach, and results

Examples of suitable problems:

* PDE or simulation step (finite-difference, Monte-Carlo, particle update)
* Numerical algorithms (scan, reduction, iterative solvers)
* Custom ML ops (attention block, loss function, activation)
* Data transformations (pairwise distances, feature extraction, filtering)

You finish the course with:

* A **GPU-accelerated version** of something meaningful
* A clean codebase
* Techniquees you can use in research papers or engineering work



## Tools You Will Use

* **CUDA Toolkit** and the official **CUDA Programming Guide**
* **Python + PyTorch/NumPy**
* **Triton** for high-level GPU programming
* **NVCC**, **Nsight Systems**, **Nsight Compute**, and timing via `torch.cuda.Event`
* Git & GitHub for submissions



## Key References

* **NVIDIA CUDA Programming Guide:**
  [https://docs.nvidia.com/cuda/cuda-programming-guide/](https://docs.nvidia.com/cuda/cuda-programming-guide/)
* **CUDA Samples:**
  [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
* **GPU Gems 3 (Scan chapter):**
  [https://developer.nvidia.com/gpugems/gpugems3](https://developer.nvidia.com/gpugems/gpugems3)
* **Triton (OpenAI):**
  [https://github.com/openai/triton](https://github.com/openai/triton)



## Getting Started

1. Fork this repository
2. Ensure access to an NVIDIA GPU (local or cloud)
3. Install the CUDA Toolkit + Python environment
4. Start with **Week 1 → `weeks/week1.md`**

Here is a **clean, concise section** you can paste directly into the root `README.md`.
It only includes **two options**: your own NVIDIA laptop, or Google Colab — and gives setup steps for both.



## GPU Access for This Course

To complete this course, you need access to an NVIDIA GPU.
You have **two simple options**:



### **Option 1 — Use Your Own Laptop (Recommended if you have an NVIDIA GPU)**

If your laptop/PC has an NVIDIA GPU (e.g., GTX/RTX series), you can run all CUDA and Triton code locally.

**Setup steps:**

1. **Install NVIDIA GPU drivers**
   [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

2. **Install the CUDA Toolkit** (version 12.x or latest)
   [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

3. Verify CUDA is installed:

   ```bash
   nvcc --version
   ```

4. Use a Python environment:

   ```bash
   conda create -n gpu python=3.10
   conda activate gpu
   ```

5. Install PyTorch with CUDA support (example for CUDA 12.1):

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

If your GPU is recent (RTX 20-series or newer), you’ll be able to run everything comfortably.



### **Option 2 — Use Google Colab (No GPU Laptop Needed)**

If you don’t have a machine with an NVIDIA GPU, you can still do **100% of the course** on Google Colab.

1. Open a new Colab notebook
   [https://colab.research.google.com/](https://colab.research.google.com/)

2. Enable the GPU runtime:

   ```
   Runtime → Change runtime type → GPU
   ```

3. Verify GPU access in a notebook cell:

   ```python
   !nvidia-smi
   ```

Colab provides free access to NVIDIA T4/P100/V100 GPUs and works fully with CUDA and Triton.






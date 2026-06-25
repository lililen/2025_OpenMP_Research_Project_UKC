# OpenMP/AVX2 Image Rotation Optimization: Up to 117x Speedup with Near-Perfect Accuracy Over SciPy

> A high-performance C++ image rotation kernel using OpenMP multi-threading and AVX2 SIMD vectorization, exposed to Python via pybind11. Achieves up to **117x speedup** over SciPy's `ndimage.rotate()` while maintaining near-perfect accuracy (avg. SSIM = 0.998).

## Google Drive Files:

https://drive.google.com/drive/folders/1YpiamVYoMAawjELfPiM3Zi5hyg_l2jzD?usp=sharing

**Uyen Le**  
Department of Mathematics and Computer Science, Santa Clara University, USA

> **UKC 2025 Poster Award — 3rd Place**  
> Korean-American Scientists and Engineers Association (KSEA) · August 9, 2025

---

## Abstract

This project introduces an optimized high-performance image rotation method that combines **OpenMP multi-threading** with **AVX2 SIMD vectorization** in C++, exposed to Python via pybind11. Benchmarks across 20 independent runs demonstrate peak speedups of **117.3x** over SciPy's `ndimage.rotate()` while maintaining near-perfect accuracy (average SSIM = 0.998, with most configurations achieving SSIM = 1.000).

---

## Motivation

SciPy's `ndimage.rotate()` has three compounding bottlenecks in production environments:

| Bottleneck | Description |
|---|---|
| **Single-threaded execution** | Runs on a single CPU core, leaving parallel resources idle. On systems with 8+ cores, ≥87.5% of compute capacity is unused. |
| **Lack of vectorization** | Processes pixels sequentially with no SIMD. Fails to exploit AVX2 instruction sets that can process 8 single-precision floats simultaneously. |
| **Memory access inefficiency** | Non-optimized access patterns cause cache misses and inefficient data movement, especially at high resolutions. |

---

## Architecture

```
SciPy ndimage.rotate() — Single-Threaded Bottleneck
───────────────────────────────────────────────────
IMAGE INPUT          COORDINATE TRANSFORM    INTERPOLATION           OUTPUT WRITING
Sequential memory  → Pixel-by-pixel        → Sequential/slower    → Single-threaded
loading              processing               interpolation           memory copy
Single Core          No vectorization         Cache misses            Under-utilized

OpenMP/AVX2 Optimized — Parallel Processing Solution
─────────────────────────────────────────────────────
PARALLEL INPUT       THREAD DISTRIBUTION     SIMD PROCESSING         SYNCHRONIZED OUTPUT
Memory-aligned     → Work divided across   → 8 pixels processed    → Thread-safe parallel
data loading         8+ cores                simultaneously           writing
Multi-Core           OPENMP Parallel          AVX2 Vectorized          Cache Optimized
```

---

## Results

Benchmarks ran each configuration **20 times** for statistical confidence. All rotations were performed at a standardized **30-degree angle**. Image quality was validated using SSIM.

### Performance Summary

| Test Category | Image Size | Avg. Speedup | Best Speedup | Avg. SSIM | Avg. PSNR (dB) |
|---|---|---|---|---|---|
| Grayscale AVX2 | 256×256 | 46.0x | **117.3x** | 1.000 | 124.2 |
| Grayscale AVX2 | 512×512 | 18.2x | 26.7x | 1.000 | 126.9 |
| Grayscale AVX2 | 1024×1024 | 24.1x | 34.9x | 1.000 | 124.5 |
| RGB Standard | 512×512 | 20.5x | 32.7x | 1.000 | inf |
| Boundary Modes | 512×512 | 16.8x | 38.1x | 0.993* | 158.2* |
| **Overall Outcome** | — | **27.2x** | **117.3x** | **0.998** | **140.0** |

### Key Observations

- Performance scales positively with image size — larger images benefit more from parallel processing and efficient workload distribution across CPU cores.
- Grayscale images consistently outperform RGB configurations due to reduced memory bandwidth requirements.
- All boundary handling modes (Reflect, Wrap, Constant, Clamp) maintained consistent performance with standard deviation around below 5% for processing times.

### Test Configurations

- **RGB rotation testing:** 256×256, 512×512, 1024×768, 768×1024 pixels
- **Grayscale AVX2 validation:** 256×256, 512×512, 1024×1024 pixels
- **Boundary modes:** Reflect, Wrap, Constant, Clamp

---

## Project Structure

```
.
├── rotate.cpp          # OpenMP + AVX2 parallel rotation kernel (pybind11 extension)
├── scipy_benchmark.py  # Benchmarks OpenMP vs SciPy rotation across image sizes
├── baseline.py         # PyTorch DataLoader benchmark (0–8 workers, CIFAR-10)
├── test_rotate.py      # Visual + metric validation of the rotation output
├── data_analysis.py    # Environment smoke test
├── setup.py            # setuptools build for the pybind11 extension
├── CMakeLists.txt      # CMake build alternative
└── environment.yml     # Conda environment definition
```

---

## Requirements

- Python 3.10+
- A C++ compiler with OpenMP and AVX2 support  
  (GCC/Clang on Linux/macOS; MSVC with `/openmp:experimental` and `/arch:AVX2` on Windows)
- CMake 3.14+ (optional, alternative to setuptools)

Python dependencies:
```
numpy
scipy
scikit-image
matplotlib
torch
torchvision
pybind11
```

---

## Building the Extension

### Using setuptools (recommended on Windows)

```bash
pip install pybind11
python setup.py build_ext --inplace
```

The compiled module (`rotate.*.pyd` / `rotate.*.so`) will appear in the project root.

### Using CMake

```bash
cmake -B build
cmake --build build --config Release
```

---

## Usage

```python
import numpy as np
from rotate import rotate

img = np.random.rand(512, 512, 3).astype(np.float32)
rotate(img, 30.0)   # rotates in-place by 30 degrees
```

---

## Running Benchmarks

**OpenMP vs SciPy rotation benchmark:**
```bash
python scipy_benchmark.py
```
Tests image sizes `256×256`, `512×512`, `1024×768`, and `768×1024`, reporting wall-clock time, speedup, and mean absolute pixel difference.

**PyTorch DataLoader baseline (CIFAR-10):**
```bash
python baseline.py
```
Measures epoch throughput for `num_workers` in `[0, 2, 4, 8]` over 3 epochs. Downloads CIFAR-10 automatically on first run.

**Visual validation:**
```bash
python test_rotate.py
```
Rotates a synthetic 64×64 image, prints SSIM and mean absolute difference, and displays a side-by-side comparison.

---

## Implementation Notes

- The outer row loop is parallelised with `#pragma omp parallel for schedule(static)`.
- `collapse(2)` is omitted for MSVC compatibility (requires `/openmp:experimental` or OpenMP 5+).
- **Bilinear interpolation** is applied when the source coordinate is at least 1 pixel from the image boundary; nearest-neighbour fallback handles the remaining edge pixels.
- **AVX2 vectorization** processes 8 single-precision floats per instruction cycle, dramatically increasing per-core throughput.
- The output buffer is written back to the input array via `memcpy` so the caller sees the result in-place.

---

## References

1. Krishnan, P.T., Krishnadoss, P., Khandelwal, M., Gupta, D., Nihaal, A., and Kumar, T.S., "Enhancing brain tumor detection in MRI with a rotation invariant Vision Transformer," *Frontiers in Neuroinformatics*, Vol. 18, p. 1414925, 2024.
2. Kozlowski, M., Racewicz, S., and Wierzchicki, S., "Medical Kozlowski," *Applied Sciences*, Vol. 14, No. 8, p. 8150, 2024.
3. Abu-raddaha, A., El-Shair, Z.A., and Rawashdeh, S., "Leveraging Perspective Transformation for Enhanced Pothole Detection in Autonomous Vehicles," *Journal of Imaging*, Vol. 10, No. 9, pp. 227, 2024.

---

## Award

This work received the **UKC 2025 Poster Award — 3rd Place** at the Korean-American Scientists and Engineers Association (KSEA) poster session on August 9, 2025, recognized by the 54th KSEA President Jae Hyeon Ryu.

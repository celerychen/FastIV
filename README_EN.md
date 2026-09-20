# FastIV

<div align="center">

**High-performance heterogeneous-computing algorithm library — C23 · pure C · zero third-party runtime dependencies**

<br>

[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](License/LICENSE)
[![Language](https://img.shields.io/badge/Language-C23-59666c.svg)](#)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20NEON-brightgreen.svg)](#)
[![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](#)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-9cf.svg)](README.md)

</div>

> A high-performance algorithm library for speech, image & computer vision, SLAM, machine learning & neural networks, SPICE simulation, and financial time-series prediction, with targeted SIMD optimizations for x86 (AVX2) and ARM (NEON).

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Module overview](#module-overview)
- [Core modules](#core-modules)
- [Tests](#tests)
- [Linear programming (PDLP)](#linear-programming-pdlp)
- [License](#license)
- [中文](#中文)

## Features

- ✅ **Pure C23**, zero third-party runtime dependencies; needs only `libm`, embeddable as a whole static library
- ⚡ **Dual SIMD kernels**: x86 / AVX2 and ARM / NEON dispatched automatically by architecture, with a scalar fallback
- 🧮 **Unified tensor abstraction** + dense / sparse matrices, SVD, Cholesky / LU, symmetric eigendecomposition
- 🧠 **Computation-graph neural nets**: forward inference + backpropagation training, with conv / attention SIMD kernels
- ➗ **Math operators**: Sigmoid · SiLU (SwiGLU) · Softmax · RMSNorm, dual single / double precision paths with AVX2 / NEON vectorization on the hot operators
- 🖼️ **End-to-end vision apps**: face (BlazeFace · 478-pt Landmark · YuNet), YOLO26 detection / segmentation / pose / classification
- 📐 **First-order PDLP LP solver**: no simplex basis, no factorization; scales to large, sparse problems

## Quick start

### Build

**Windows** (C23-capable GCC or MSVC):

```bat
build\build.bat              # default GCC
build\build.bat msvc         # MSVC
```

**macOS / Linux**:

```sh
make -C build                # build all tests (AVX2 on x86, NEON on ARM)
make -C build test_mat_svd   # build a single target
make -C build run            # build and run every test
make -C build clean
```

> Run test binaries from the `build/` directory (relative paths read images and weights). Clang before 17 spells C23 as `-std=c2x`; `build/Makefile` detects the usable spelling automatically.

### Minimal example

```c
#include "fiv_matrix.h"   /* matrix / vector API */

int main(void) {
    /* 4x3 matrix A, and two vectors x(3), y(4) */
    fiv_mat* A = fiv_create_tensor2d((size_t[]){4, 3}, FIV_32F1);
    fiv_vec* x = fiv_create_tensor1d(3, FIV_32F1);
    fiv_vec* y = fiv_create_tensor1d(4, FIV_32F1);

    /* fill inputs (data.fl is a contiguous float32 buffer) */
    for (size_t i = 0; i < 12; i++) A->data.fl[i] = (ivf32)(i + 1);
    for (size_t i = 0; i <  3; i++) x->data.fl[i] = 1.0f;

    fiv_matrix_mul_vec(y, A, x, 0);   /* y = A·x, auto-routes to AVX2 / NEON SIMD */

    fiv_release_tensor2d(&A);
    fiv_release_tensor1d(&x);
    fiv_release_tensor1d(&y);
    return 0;
}
```

Link with `src/ctensor/*.o` and `src/mat/*.o`, and add `api/` and `src/ctensor/` to the header search path.

## Module overview

| Module | Path | Highlights |
|---|---|---|
| Tensor & dynamic array | `src/ctensor` | Unified N-D tensor abstraction, `fiv_darray`; element-wise binary ops, zero-copy view / reshape |
| Matrix ops | `src/mat` | Transpose, mat-vec / mat-mat mul (dispatched on `FIV_32F1`/`FIV_64F1`), Cholesky / LU, symmetric eigendecomposition, SVD (one-sided Jacobi and blocked bidiagonalization + QR); dense GEMM SIMD on x86/ARM |
| Math operators | `src/math` | Sigmoid, SiLU (SwiGLU), row-wise Softmax, RMSNorm; single / double precision, AVX2 / NEON kernels on the hot operators, Softmax reused by the `src/nn` Attention node |
| Linear programming PDLP | `src/lp` | First-order primal-dual solver for large-scale LP (see [section](#linear-programming-pdlp)) |
| Neural networks | `src/nn` | Computation graph, forward inference + backprop; Linear / ReLU·ReLU6·PReLU·Sigmoid·SiLU / 2D conv (standard / depthwise / pointwise / separable) / Max2D·MaxPool / Flatten / Add / Pad·SpatialPad / Concat / Slice / Upsample / Attention; model save/load |
| Image | `src/image` | I/O (`fiv_image_io`), color-space conversion, Gaussian blur (8U fixed-point and 32F float), resize (`fiv_image_resize`: nearest / bilinear), dtype normalization |
| Face vision | `app/face` | BlazeFace, 478-pt Landmark, YuNet; weights in `app/face/models/` |
| YOLO26 | `app/yolo26` | Detection / instance segmentation / pose / classification inference (NMS-free end-to-end decode); weights in `app/yolo26/models/` |

## Core modules

**Neural networks (`src/nn`)** — convolutions are optimized for x86 (AVX2) and ARM (NEON); standard / depthwise / pointwise / separable modes with SIMD kernels for 1×1, 2×2, 3×3, 5×5, plus explicit per-side padding (`zero` / `replicate`). Attention is a composite node carrying the YOLO26 C2PSA self-attention (it calls the row-wise Softmax from `src/math`). Training uses SGD + backpropagation with cross-entropy or MSE loss, and models can be saved to / loaded from a binary blob.

**Math operators (`src/math`)** — Sigmoid, SiLU (SwiGLU), row-wise Softmax and RMSNorm in both single (`FIV_32F1`) and double (`FIV_64F1`) precision. Sigmoid and Softmax carry AVX2 + NEON kernels and SwiGLU an AVX2 one, while RMSNorm is a scalar implementation; the row-wise Softmax also serves as the operator base of the `src/nn` Attention node.

**Image (`src/image`)** — decoding/encoding uses the bundled stb (`import/image_io/`, header-only); resizing offers nearest and bilinear paths; Gaussian blur ships both an 8U fixed-point implementation (16-bit intermediates, single rounding) and a 32F float one; color-space conversion (BGR↔RGB, RGB / BGR→GRAY) and dtype normalization (`8U → [0,1] / [-1,1] / mu-sigma`) are orthogonal and compose freely.

**Face / YOLO26 (`app/`)** — each is an end-to-end app exposing only a few construct / infer / release entry points:

| App | Public API |
|---|---|
| BlazeFace | `fiv_create_face_detetor` / `fiv_face_detector_on_image` / `fiv_release_face_detector` |
| Landmark | `fiv_create_landmark_graph` / `fiv_landmark_graph_run_inference` / `fiv_release_landmark_graph` |
| YuNet | `fiv_yunet_build_graph` / `fiv_yunet_preprocess` / `fiv_yunet_detect` / `fiv_yunet_release_graph` |
| YOLO26 | `fiv_create_yolo26_{detector,segmenter,pose,classifier}` / `fiv_yolo26_<task>_on_image` / `fiv_release_yolo26_<task>`; depth branches can be diffed against PyTorch via `YOLO26_REF_DIR` |

## Tests

Each feature builds one standalone test binary in `build/`; **on success it prints `PASS=n FAIL=0` and exits 0** (a few print `ALL PASS`). Grouped by domain:

| Domain | Targets |
|---|---|
| Basics | `test_darray`, `test_ctensor`, `bench_vec_dot` |
| Matrix | `test_mat_transpose` `test_mat_vec` `test_mat_mul_vec_db` `test_mat_mul` `test_mat_mul_db` `test_mat_reduce_db` `test_mat_cholesky` `test_mat_lu` `test_mat_eig_sym` `test_mat_svd` `test_mat_svd_jacobi` |
| Neural nets | `test_nn`, `test_nn_conv2d` (incl. numeric-gradient check), `test_nn_mnist` `test_nn_mnist_conv` (skip if data absent) |
| Image / face | `test_blazeface` (needs `src/reference/`, not published), `test_image_io` `test_color_space_perf` `test_data_convert` `test_face_api` `test_landmark_ops` `test_landmark_net` `test_landmark_api` `test_landmark_e2e` `test_yunet_api` `test_yunet_resize` |
| YOLO26 | `test_yolo26_ref` `test_yolo26_ops` `test_yolo26_attn` `test_yolo26_blocks` `test_yolo26_net` `test_yolo26_e2e` `test_yolo26_seg` `test_yolo26_pose` `test_yolo26_cls` `test_yolo26_model` (all but `model` need a `YOLO26_REF_DIR` truth directory, regenerable with `app/yolo26/test/gen_ref*.py`) |
| Linear programming | `test_lp_sp_matrix` `test_lp_ops` `test_lp_rescale` `test_lp_pdhg` `test_lp_solve` `test_lp_solve13` |

## Linear programming (PDLP)

`src/lp/` solves LPs with a first-order primal-dual method (no simplex basis, no factorization), scaling to large, sparse problems:

```
minimize    cᵀ x
subject to  G x >= h        (first num_inequality rows of K / q)
            A x  = b        (remaining rows)
            l <= x <= u
```

Constraints are passed as the stacked matrix `K = [G ; A]` (m×n) and RHS `q = [h ; b]` (length m). Bounds may carry `±INF` for free variables; `x_out` / `y_out` receive the unscaled primal / dual solutions.

| Layer | Source | Responsibility |
|---|---|---|
| Sparse storage | `fiv_sp_matrix.c` | CSR / CSC / COO matrices, SpMV, row/col reductions, CSRL packed view, dense↔sparse conversion |
| Unified matrix | `fiv_lp_mat.c` | `fiv_lp_mat` abstraction hiding dense vs. sparse |
| Vector ops | `fiv_lp_vec.c` | Box projection, finite-term products; norms/dot/axpy delegate to FastIV |
| Preconditioning | `fiv_lp_rescale.c` | Ruiz equilibration + Pock-Chambolle rescaling, unscaling |
| Algorithm core | `fiv_lp_pdhg.c` | Adaptive-step PDHG, primal-weight update, box normal-cone λ, squared KKT error |
| Driver | `fiv_lp_solve.c` | Rescale → initialize → restart loop → termination (optimal / primal infeasible / dual infeasible) → unscale |

Only `api/fiv_sp_matrix.h` and `api/fiv_lp_solve.h` are public (`fiv_lp_mat` is opaque; build with `fiv_lp_mat_wrap_dense` / `fiv_lp_mat_wrap_sparse` / `fiv_create_lp_mat_from_coo`, release with `fiv_release_lp_mat`). Entry `fiv_lp_solve()` takes a `fiv_lp_solve_params` that may be NULL for PDLP defaults (10 Ruiz iters, Pock-Chambolle α=1.0, tol 1e-4, 10000-iter cap); `fiv_lp_solve_info` reports objective, duality gap, residuals and iteration count. All arithmetic is `FIV_64F1` (double).

```sh
make -C build test_lp_sp_matrix test_lp_ops test_lp_rescale test_lp_pdhg test_lp_solve test_lp_solve13
cd build && ./test_lp_solve
```

- `test_lp_sp_matrix` — sparse storage, SpMV, reductions, CSRL packing
- `test_lp_ops` — vector primitives and dense/sparse matrix abstraction
- `test_lp_rescale` — Ruiz and Pock-Chambolle preconditioning
- `test_lp_pdhg` — PDHG step, primal-weight update, KKT error
- `test_lp_solve` — end-to-end solver on the general LP suite
- `test_lp_solve13` — the 13 standard problems (infeasible, unbounded, degenerate)

`make oracle` and `make oracle13` additionally diff the C solver against torch `pdlp.py` (scripts under `src/lp/extra_test/`).

## License

GPL v3. See [License/LICENSE](License/LICENSE). Copyright (C) 2026 Celery Chen.

---

## 中文

请参阅 **[README.md](README.md)** 获取完整中文版本。

FastIV 是一套面向语音、图像与计算机视觉、SLAM、机器学习与神经网络、SPICE 仿真、金融时序预测等场景的高性能算法库，在 x86（AVX2）与 ARM（NEON）等异构架构下做针对性优化。代码以 C23 编写，纯 C、零第三方运行时依赖。

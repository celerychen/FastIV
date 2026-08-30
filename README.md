# FastIV

## 中文

本项目致力于语音、图像与计算机视觉，SLAM、机器学习与神经网络、大模型、SPICE仿真，金融时序预测等相关的高性能算法，在各类异构体系架构下的最优化实现。

### 模块简介

- **张量与基础数据结构**：提供统一的张量抽象与动态数组，作为上层算法的基础数据容器，支持任意维度张量及其逐元素二元运算。
- **矩阵运算**：涵盖矩阵转置、矩阵-向量乘法与矩阵-矩阵乘法、Cholesky / LU 分解、对称特征分解与 SVD（分块双对角化 + Golub-Kahan QR 与 Jacobi 两条路径），面向 x86 / ARM 等异构架构做针对性优化。
- **线性规划（PDLP）**：一阶算法求解大规模线性规划的完整 C 实现（移植自 Google OR-Tools PDLP），位于 `src/lp/`。含 CSR / CSC / COO 稀疏矩阵与 CSRL 打包视图、Ruiz 平衡与 Pock-Chambolle 重缩放、自适应步长 PDHG 迭代、重启策略与原始 / 对偶不可行判据；算法核心对稠密与稀疏后端统一抽象，全程双精度运算，可与 torch 参考实现逐题交叉验证。
- **神经网络**：提供计算图构建、前向推理与反向传播训练能力。算子涵盖线性层（Linear）、激活函数（ReLU / ReLU6）、二维卷积（标准 / 深度可分离 / 逐点 / 可分离，含 3×3 与 5×5 stride-2 的 SIMD 优化内核）、最大池化（Max2D）、展平（Flatten）、逐元素相加（Add）与填充（Pad）；支持模型保存与加载。卷积在 x86（AVX2）/ ARM（NEON）架构下做了针对性优化。
- **高性能人脸检测器**：基于 BlazeFace 的端到端人脸检测应用，封装于 `app/face/`，对外仅暴露 `fiv_create_face_detetor` / `fiv_face_detector_on_image` / `fiv_release_face_detector` 三个接口，权重随仓库发布于 `app/face/models/`。

## English

FastIV is dedicated to high-performance algorithms for speech, image and computer vision, SLAM, machine learning and neural networks, large models, SPICE simulation, and financial time-series prediction, with optimized implementations across a variety of heterogeneous architectures.

### Modules

- **Tensor & data structures**: A unified tensor abstraction and dynamic array that serve as the foundational data containers for higher-level algorithms, supporting N-dimensional tensors and element-wise binary operations.
- **Matrix operations**: Matrix transpose, matrix-vector and matrix-matrix multiplication, Cholesky / LU factorization, symmetric eigendecomposition and SVD (blocked bidiagonalization + Golub-Kahan QR, plus a Jacobi path), with targeted optimizations for heterogeneous architectures such as x86 and ARM.
- **Linear programming (PDLP)**: A complete C port of PDLP (Primal-Dual Hybrid Gradient for Linear Programming) from Google OR-Tools, in `src/lp/`. It ships CSR / CSC / COO sparse storage with an optional SIMD-friendly CSRL packed view, Ruiz equilibration and Pock-Chambolle rescaling, adaptive-step PDHG with restarts, and primal / dual infeasibility detection. The algorithm core is written against a backend-agnostic dense/sparse abstraction, runs entirely in double precision, and is cross-validated problem-by-problem against the torch reference implementation.
- **Neural networks**: Computation-graph construction, forward inference and backpropagation training. Operators include linear layers (Linear), activations (ReLU / ReLU6), 2D convolutions (standard / depthwise / pointwise / separable, with SIMD-optimized 3x3 and 5x5 stride-2 kernels), max pooling (Max2D), flatten, element-wise add and pad; with model save/load support. Convolutions are optimized for x86 (AVX2) and ARM (NEON).
- **High-performance face detector**: An end-to-end BlazeFace-based face detector in `app/face/`, exposing only three public APIs — `fiv_create_face_detetor` / `fiv_face_detector_on_image` / `fiv_release_face_detector`, with weights shipped in the repo under `app/face/models/`.

## Linear programming (PDLP)

`src/lp/` solves linear programs with a first-order primal-dual method (no simplex basis, no factorization), which scales to large, sparse problems. The solver handles the form

```
minimize    cᵀ x
subject to  G x >= h        (first num_inequality rows of K / q)
            A x  = b        (remaining rows)
            l <= x <= u
```

with the constraints passed as the stacked matrix `K = [G ; A]` (m x n) and right-hand side `q = [h ; b]` (length m). Bounds may carry `±INF` for free variables, and `x_out` / `y_out` receive the unscaled primal / dual solutions.

### Module layout

| Layer | Source | Responsibility |
|---|---|---|
| Sparse storage | `fiv_sp_matrix.c` | CSR / CSC / COO matrices, SpMV, row/column reductions, optional CSRL packed view (SIMD-friendly), dense <-> sparse conversion |
| Unified matrix | `fiv_lp_mat.c` | `fiv_lp_mat` abstraction hiding dense vs. sparse from the algorithm core |
| Vector ops | `fiv_lp_vec.c` | Box projection and finite-term products; norms / dot / axpy delegate to FastIV |
| Preconditioning | `fiv_lp_rescale.c` | Ruiz equilibration + Pock-Chambolle rescaling, and unscaling of the solution |
| Algorithm core | `fiv_lp_pdhg.c` | Adaptive-step PDHG, primal-weight update, box normal-cone lambda, squared KKT error |
| Driver | `fiv_lp_solve.c` | Rescale -> initialize -> restart loop -> termination criteria (optimal / primal infeasible / dual infeasible) -> unscale |

Following the project convention (one public header per module, internals next to their sources), only two LP headers are public: `api/fiv_sp_matrix.h` (the general sparse-matrix structure) and `api/fiv_lp_solve.h` (the solver entry point). The layered internals — `fiv_lp_vec.h`, `fiv_lp_mat.h`, `fiv_lp_rescale.h`, `fiv_lp_pdhg.h` — live in `src/lp/` alongside their `.c` files, as do the `mat` / `nn` / `image` modules. The constraint matrix type `fiv_lp_mat` is opaque to callers: build one with `fiv_lp_mat_wrap_dense` / `fiv_lp_mat_wrap_sparse` / `fiv_create_lp_mat_from_coo` and release it with `fiv_release_lp_mat`.

The entry point is `fiv_lp_solve()`; its `fiv_lp_solve_params` may be NULL to use the PDLP defaults (10 Ruiz iterations, Pock-Chambolle alpha 1.0, tolerance 1e-4, 10000-iteration limit), and `fiv_lp_solve_info` reports objectives, duality gap, residuals and iteration count. All arithmetic is `FIV_64F1` (double precision).

### Tests

```
make -C build test_lp_sp_matrix test_lp_ops test_lp_rescale test_lp_pdhg test_lp_solve test_lp_solve13
cd build && ./test_lp_solve
```

- `test_lp_sp_matrix` — sparse storage, SpMV, reductions, CSRL packing
- `test_lp_ops` — vector primitives and the dense/sparse matrix abstraction
- `test_lp_rescale` — Ruiz and Pock-Chambolle preconditioning
- `test_lp_pdhg` — PDHG step, primal-weight update and KKT error
- `test_lp_solve` — end-to-end solver on the general LP suite
- `test_lp_solve13` — the 13 standard problems, including infeasible, unbounded and degenerate cases

Together they run 321 checks, all passing.

### Cross-validation against the reference implementation

Beyond self-consistency, the C port is checked against the real torch `pdlp.py` it was ported from: both implementations solve the same problems and their termination status and objectives are compared.

```
make -C build oracle      # general LP suite (9 comparisons)
make -C build oracle13    # the 13 standard problems, incl. objective and optimal-vertex checks (26 comparisons)
```

These targets need PyTorch and a local OR-Tools source tree that contains `pdlp.py`; set `FIV_PDLP_DIR` to that directory (it defaults to the author's checkout). They are optional — the plain `test_lp_*` binaries have no Python dependency.

## Build

Windows (C23-capable GCC or MSVC):

```
build\build.bat              # default: GCC
build\build.bat msvc         # MSVC
```

macOS / Linux:

```
make -C build          # build all tests (AVX2 on x86, NEON on ARM)
make -C build run      # build and run
make -C build clean
```

## Run

The build produces one standalone test binary per feature in `build/`:

- `test_darray` — dynamic array (`fiv_darray`, std::vector-style)
- `test_ctensor` — N-D tensor and binary ops
- `test_mat_transpose`, `test_mat_vec`, `test_mat_mul` — matrix ops
- `test_mat_cholesky`, `test_mat_lu`, `test_mat_eig_sym` — matrix factorizations and decompositions
- `test_mat_svd`, `test_mat_svd_jacobi` — SVD via blocked bidiagonalization + QR, and via Jacobi
- `test_lp_sp_matrix`, `test_lp_ops`, `test_lp_rescale`, `test_lp_pdhg` — linear-programming (PDLP) building blocks
- `test_lp_solve`, `test_lp_solve13` — end-to-end PDLP solver and the 13 standard LP problems
- `test_nn` — neural-network unit and integration tests (no dataset required)
- `test_nn_conv2d` — convolution forward/backward tests incl. numeric-gradient check (no dataset required)
- `test_nn_mnist` — end-to-end MLP MNIST accuracy test (skips if data is absent)
- `test_nn_mnist_conv` — end-to-end CNN MNIST test (conv→ReLU→Max2D→flatten→Linear), requires >98% accuracy (skips if data is absent)
- `test_blazeface` — BlazeFace short-range face-detector inference (5x5 stride-2 stem conv), uses the SIMD conv kernels
- `test_face_api` — black-box contract test of the public face-detector API (create / on_image / release) over the bundled model in `app/face/models/`

Each prints `PASS=n FAIL=0` on success and exits non-zero on failure.

## License

GPL v3. See [License/LICENSE](License/LICENSE). Copyright (C) 2026 Celery Chen.

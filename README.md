# FastIV

## 中文

FastIV 是一套面向语音、图像与计算机视觉、SLAM、机器学习与神经网络、SPICE 仿真、金融时序预测等场景的高性能算法库，在 x86（AVX2）与 ARM（NEON）等异构架构下做针对性优化。代码以 C23 编写，纯 C、零第三方运行时依赖。

### 模块简介

- **张量与基础数据结构（`src/ctensor`）**：统一的 N 维张量抽象与动态数组（`fiv_darray`），支持任意维度张量与逐元素二元运算，是上层算法的基础容器。
- **矩阵运算（`src/mat`）**：矩阵转置、矩阵-向量 / 矩阵-矩阵乘法（按 `FIV_32F1`/`FIV_64F1` 分派），Cholesky / LU 分解、对称特征分解、SVD（分块双对角化 + Golub-Kahan QR 与单边 Jacobi 两条路径）。稠密 GEMM 在 x86/ARM 下均有 SIMD 加速。
- **线性规划 PDLP（`src/lp`）**：一阶原始-对偶混合梯度法求解大规模 LP 的完整 C 实现。详见下方「线性规划（PDLP）」章节。
- **神经网络（`src/nn`）**：计算图构建、前向推理与反向传播训练。算子含 Linear、ReLU / ReLU6 / PReLU / Sigmoid、2D 卷积（标准 / 深度可分离 / 逐点 / 可分离，含 1×1、2×2、3×3、5×5 的 SIMD 内核）、Max2D、Flatten、Add、Pad、Concat、Upsample；支持模型保存与加载。卷积在 x86（AVX2）/ ARM（NEON）下做针对性优化。
- **图像（`src/image`）**：图像读写（`fiv_image_io`）、颜色空间转换、高斯模糊、双线性 / 最近邻缩放（`fiv_image_resize`）等前后处理算子。
- **人脸视觉应用（`app/face`）**：三个端到端应用，权重随仓库发布于 `app/face/models/`。
  - **BlazeFace 人脸检测**（`app/face/blazeFace`）：短距人脸检测器，对外接口 `fiv_create_face_detetor` / `fiv_face_detector_on_image` / `fiv_release_face_detector`。
  - **人脸网格 landmark**（`app/face/landmark`）：输入检测框经仿射 warp 后回归 478 点 3D 人脸网格与语义分割，对外接口 `fiv_create_landmark_graph` / `fiv_landmark_graph_run_inference` / `fiv_release_landmark_graph`。
  - **YuNet 人脸检测**（`app/face/yunet`）：高召回人脸检测器，对外接口 `fiv_yunet_build_graph` / `fiv_yunet_preprocess` / `fiv_yunet_detect` / `fiv_yunet_release_graph`，含最长边 320 的图像缩放预处理。

### 构建

Windows（支持 C23 的 GCC 或 MSVC）：

```
build\build.bat              # 默认 GCC
build\build.bat msvc         # MSVC
```

macOS / Linux：

```
make -C build                # 构建全部测试（x86 上 AVX2，ARM 上 NEON）
make -C build test_blazeface  # 只构建单个目标
make -C build clean
```

> 测试二进制从 `build/` 目录运行（相对路径读取图片与权重）。

### 运行

`build/` 下每个功能生成一个独立测试二进制：

- `test_darray` — 动态数组
- `test_ctensor` — N 维张量与二元算子
- `test_mat_transpose` / `test_mat_vec` / `test_mat_mul` / `test_mat_mul_vec_db` / `test_mat_mul_db` / `test_mat_reduce_db` — 矩阵运算
- `test_mat_cholesky` / `test_mat_lu` / `test_mat_eig_sym` — 矩阵分解
- `test_mat_svd` / `test_mat_svd_jacobi` — SVD（分块双对角化 + QR / 单边 Jacobi）
- `bench_vec_dot` — 向量点积性能基准
- `test_nn` — 神经网络单元与集成测试
- `test_nn_conv2d` — 卷积前向/反向（含数值梯度校验）
- `test_nn_mnist` / `test_nn_mnist_conv` — MNIST 端到端（无数据则跳过）
- `test_blazeface` — BlazeFace 推理（使用 5×5 stride-2 SIMD 卷积内核）
- `test_image_io` / `test_color_space_perf` / `test_data_convert` — 图像模块
- `test_face_api` — BlazeFace 公开接口契约测试
- `test_landmark_ops` / `test_landmark_net` / `test_landmark_api` / `test_landmark_e2e` — 人脸网格 landmark（算子 / 网络 / API / 端到端）
- `test_yunet_api` / `test_yunet_resize` — YuNet 检测与缩放
- `test_lp_*` — 线性规划构建块与端到端求解（见下）

所有二进制成功时打印 `PASS=n FAIL=0` 并以 0 退出。

### 线性规划（PDLP）

`src/lp/` 用一阶原始-对偶方法（无单纯形基、无矩阵分解）求解 LP，可扩展到大规模稀疏问题：

```
minimize    cᵀ x
subject to  G x >= h        (K / q 的前 num_inequality 行)
            A x  = b        (其余行)
            l <= x <= u
```

约束以堆叠矩阵 `K = [G ; A]` (m×n) 与右端 `q = [h ; b]` (长度 m) 传入。边界可用 `±INF` 表示自由变量，`x_out` / `y_out` 接收未缩放的原始 / 对偶解。

| 层 | 源文件 | 职责 |
|---|---|---|
| 稀疏存储 | `fiv_sp_matrix.c` | CSR / CSC / COO 矩阵、SpMV、行列归约、CSRL 打包视图、稠密↔稀疏转换 |
| 统一矩阵 | `fiv_lp_mat.c` | `fiv_lp_mat` 抽象，对算法核心隐藏稠密/稀疏 |
| 向量运算 | `fiv_lp_vec.c` | 盒投影、有限项乘积；范数/点积/axpy 委托给 FastIV |
| 预处理 | `fiv_lp_rescale.c` | Ruiz 平衡 + Pock-Chambolle 重缩放、解的反缩放 |
| 算法核心 | `fiv_lp_pdhg.c` | 自适应步长 PDHG、原始权重更新、盒法锥 λ、平方 KKT 误差 |
| 驱动 | `fiv_lp_solve.c` | 重缩放 → 初始化 → 重启循环 → 终止判据（最优 / 原始不可行 / 对偶不可行）→ 反缩放 |

公开头只有 `api/fiv_sp_matrix.h` 与 `api/fiv_lp_solve.h`（`fiv_lp_mat` 对调用方不透明，用 `fiv_lp_mat_wrap_dense` / `fiv_lp_mat_wrap_sparse` / `fiv_create_lp_mat_from_coo` 构造，用 `fiv_release_lp_mat` 释放）。入口 `fiv_lp_solve()` 的 `fiv_lp_solve_params` 传 NULL 即用 PDLP 默认（10 次 Ruiz、Pock-Chambolle α=1.0、容差 1e-4、上限 10000 迭代）；`fiv_lp_solve_info` 回报目标值、对偶间隙、残差与迭代次数。全程 `FIV_64F1`（双精度）。

```
make -C build test_lp_sp_matrix test_lp_ops test_lp_rescale test_lp_pdhg test_lp_solve test_lp_solve13
cd build && ./test_lp_solve
```

- `test_lp_sp_matrix` — 稀疏存储、SpMV、归约、CSRL 打包
- `test_lp_ops` — 向量原语与稠密/稀疏矩阵抽象
- `test_lp_rescale` — Ruiz 与 Pock-Chambolle 预处理
- `test_lp_pdhg` — PDHG 步、原始权重更新、KKT 误差
- `test_lp_solve` — 通用 LP 套件端到端
- `test_lp_solve13` — 13 个标准问题（含不可行、无界、退化情形）

### 许可

GPL v3。见 [License/LICENSE](License/LICENSE)。Copyright (C) 2026 Celery Chen。

---

## English

FastIV is a high-performance algorithm library for speech, image & computer vision, SLAM, machine learning & neural networks, SPICE simulation, and financial time-series prediction, with targeted optimizations for heterogeneous architectures such as x86 (AVX2) and ARM (NEON). It is written in C23, pure C, with no third-party runtime dependencies.

### Modules

- **Tensor & data structures (`src/ctensor`)**: A unified N-D tensor abstraction and dynamic array (`fiv_darray`), supporting arbitrary-dimension tensors and element-wise binary ops.
- **Matrix operations (`src/mat`)**: Matrix transpose, matrix-vector / matrix-matrix multiplication (dispatched on `FIV_32F1`/`FIV_64F1`), Cholesky / LU factorization, symmetric eigendecomposition, and SVD (blocked bidiagonalization + Golub-Kahan QR, plus a one-sided Jacobi path). Dense GEMM is SIMD-accelerated on x86/ARM.
- **Linear programming (PDLP) (`src/lp`)**: A complete C implementation of first-order PDLP. See the "Linear programming (PDLP)" section below.
- **Neural networks (`src/nn`)**: Computation-graph construction, forward inference and backpropagation training. Operators include Linear, ReLU / ReLU6 / PReLU / Sigmoid, 2D convolutions (standard / depthwise / pointwise / separable, with SIMD kernels for 1×1, 2×2, 3×3, 5×5), Max2D, Flatten, Add, Pad, Concat, Upsample; with model save/load. Convolutions are optimized for x86 (AVX2) and ARM (NEON).
- **Image (`src/image`)**: Image I/O (`fiv_image_io`), color-space conversion, Gaussian blur, bilinear / nearest-neighbor resizing (`fiv_image_resize`), and other pre/post-processing operators.
- **Face vision apps (`app/face`)**: Three end-to-end apps; weights ship in `app/face/models/`.
  - **BlazeFace detector** (`app/face/blazeFace`): short-range face detector; public API `fiv_create_face_detetor` / `fiv_face_detector_on_image` / `fiv_release_face_detector`.
  - **Face mesh landmarks** (`app/face/landmark`): takes a detection box, affine-warps the ROI, and regresses a 478-point 3D face mesh plus segmentation; public API `fiv_create_landmark_graph` / `fiv_landmark_graph_run_inference` / `fiv_release_landmark_graph`.
  - **YuNet detector** (`app/face/yunet`): high-recall face detector; public API `fiv_yunet_build_graph` / `fiv_yunet_preprocess` / `fiv_yunet_detect` / `fiv_yunet_release_graph`, with longest-edge-320 image resizing.

### Build

Windows (C23-capable GCC or MSVC):

```
build\build.bat              # default GCC
build\build.bat msvc         # MSVC
```

macOS / Linux:

```
make -C build                # build all tests (AVX2 on x86, NEON on ARM)
make -C build test_blazeface  # build a single target
make -C build clean
```

> Run test binaries from the `build/` directory (relative paths read images and weights).

### Run

The build produces one standalone test binary per feature in `build/`:

- `test_darray` — dynamic array
- `test_ctensor` — N-D tensor and binary ops
- `test_mat_transpose` / `test_mat_vec` / `test_mat_mul` / `test_mat_mul_vec_db` / `test_mat_mul_db` / `test_mat_reduce_db` — matrix ops
- `test_mat_cholesky` / `test_mat_lu` / `test_mat_eig_sym` — factorizations
- `test_mat_svd` / `test_mat_svd_jacobi` — SVD (blocked bidiagonalization + QR / one-sided Jacobi)
- `bench_vec_dot` — vector-dot product micro-benchmark
- `test_nn` — neural-network unit & integration tests
- `test_nn_conv2d` — conv forward/backward (incl. numeric-gradient check)
- `test_nn_mnist` / `test_nn_mnist_conv` — end-to-end MNIST (skip if data absent)
- `test_blazeface` — BlazeFace inference (uses the 5×5 stride-2 SIMD conv kernel)
- `test_image_io` / `test_color_space_perf` / `test_data_convert` — image module
- `test_face_api` — BlazeFace public-API contract test
- `test_landmark_ops` / `test_landmark_net` / `test_landmark_api` / `test_landmark_e2e` — face mesh landmarks
- `test_yunet_api` / `test_yunet_resize` — YuNet detection and resizing
- `test_lp_*` — PDLP building blocks and end-to-end solver (below)

Each prints `PASS=n FAIL=0` on success and exits non-zero on failure.

### Linear programming (PDLP)

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

```
make -C build test_lp_sp_matrix test_lp_ops test_lp_rescale test_lp_pdhg test_lp_solve test_lp_solve13
cd build && ./test_lp_solve
```

- `test_lp_sp_matrix` — sparse storage, SpMV, reductions, CSRL packing
- `test_lp_ops` — vector primitives and dense/sparse matrix abstraction
- `test_lp_rescale` — Ruiz and Pock-Chambolle preconditioning
- `test_lp_pdhg` — PDHG step, primal-weight update, KKT error
- `test_lp_solve` — end-to-end solver on the general LP suite
- `test_lp_solve13` — the 13 standard problems (infeasible, unbounded, degenerate)

### License

GPL v3. See [License/LICENSE](License/LICENSE). Copyright (C) 2026 Celery Chen.

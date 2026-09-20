# FastIV

<div align="center">

**高性能异构计算算法库 —— C23 · 纯 C · 零第三方运行时依赖**

<br>

[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](License/LICENSE)
[![Language](https://img.shields.io/badge/Language-C23-59666c.svg)](#)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20NEON-brightgreen.svg)](#)
[![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](#)
[![English](https://img.shields.io/badge/README-English-9cf.svg)](README_EN.md)

</div>

> 面向**语音、图像与计算机视觉、SLAM、机器学习与神经网络、SPICE 仿真、金融时序预测**的高性能算法库，在 x86（AVX2）与 ARM（NEON）等异构架构下做针对性 SIMD 优化。

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [模块总览](#模块总览)
- [核心模块说明](#核心模块说明)
- [测试](#测试)
- [线性规划（PDLP）](#线性规划pdlp)
- [许可](#许可)
- [English](#english)

## 特性

- ✅ **纯 C23**，零第三方运行时依赖，仅需 `libm`，可整体静态嵌入宿主项目
- ⚡ **双路 SIMD 内核**：x86 / AVX2 与 ARM / NEON 自动按架构分派，标量回退保底
- 🧮 **统一张量抽象** + 稠密 / 稀疏矩阵、SVD、Cholesky / LU、对称特征分解
- 🧠 **计算图神经网络**：前向推理 + 反向传播训练，卷积 / 注意力 SIMD 内核
- ➗ **数学算子**：Sigmoid · SiLU（SwiGLU）· Softmax · RMSNorm，单精度 / 双精度双路，热点算子 AVX2 / NEON 向量化
- 🖼️ **端到端视觉应用**：人脸（BlazeFace · 478 点 Landmark · YuNet）、YOLO26 检测 / 分割 / 姿态 / 分类
- 📐 **一阶 PDLP 线性规划**：无单纯形基、无矩阵分解，可扩展到大规模稀疏问题

## 快速开始

### 构建

**Windows**（支持 C23 的 GCC 或 MSVC）：

```bat
build\build.bat              # 默认 GCC
build\build.bat msvc         # MSVC
```

**macOS / Linux**：

```sh
make -C build                # 构建全部测试（x86 上 AVX2，ARM 上 NEON）
make -C build test_mat_svd   # 只构建单个目标
make -C build run            # 构建并逐个运行全部测试
make -C build clean
```

> 测试二进制从 `build/` 目录运行（相对路径读取图片与权重）。Clang 17 之前把 C23 拼作 `-std=c2x`，`build/Makefile` 会自动探测。

### 最小示例

```c
#include "fiv_matrix.h"   /* 引入矩阵 / 向量 API */

int main(void) {
    /* 4x3 矩阵 A，以及两个向量 x(3)、y(4) */
    fiv_mat* A = fiv_create_tensor2d((size_t[]){4, 3}, FIV_32F1);
    fiv_vec* x = fiv_create_tensor1d(3, FIV_32F1);
    fiv_vec* y = fiv_create_tensor1d(4, FIV_32F1);

    /* 填充输入（data.fl 是连续的 float32 缓冲） */
    for (size_t i = 0; i < 12; i++) A->data.fl[i] = (ivf32)(i + 1);
    for (size_t i = 0; i <  3; i++) x->data.fl[i] = 1.0f;

    fiv_matrix_mul_vec(y, A, x, 0);   /* y = A·x，自动走 AVX2 / NEON SIMD 内核 */

    fiv_release_tensor2d(&A);
    fiv_release_tensor1d(&x);
    fiv_release_tensor1d(&y);
    return 0;
}
```

链接时把 `src/ctensor/*.o` 与 `src/mat/*.o` 加入工程，并包含头文件搜索路径 `api/` 与 `src/ctensor/`。

## 模块总览

| 模块 | 路径 | 能力要点 |
|---|---|---|
| 张量与动态数组 | `src/ctensor` | 统一 N 维张量抽象、`fiv_darray` 动态数组；逐元素二元运算、零拷贝 view / reshape，上层算法的基础容器 |
| 矩阵运算 | `src/mat` | 转置、矩阵-向量 / 矩阵-矩阵乘（按 `FIV_32F1`/`FIV_64F1` 分派）、Cholesky / LU、对称特征分解、SVD（单边 Jacobi 与分块双对角化 + QR 双后端）；稠密 GEMM 在 x86/ARM 均有 SIMD |
| 数学算子 | `src/math` | Sigmoid、SiLU（SwiGLU）、行内 Softmax、RMSNorm；单精度 / 双精度双路，热点算子带 AVX2 / NEON 内核，Softmax 供 `src/nn` 的 Attention 节点复用 |
| 线性规划 PDLP | `src/lp` | 一阶原始-对偶混合梯度法求解大规模 LP（见[专章](#线性规划pdlp)） |
| 神经网络 | `src/nn` | 计算图、前向推理 + 反向传播训练；Linear / ReLU·ReLU6·PReLU·Sigmoid·SiLU / 2D 卷积（标准 / 深度可分离 / 逐点 / 可分离）/ Max2D·MaxPool / Flatten / Add / Pad·SpatialPad / Concat / Slice / Upsample / Attention；模型保存加载 |
| 图像 | `src/image` | 读写（`fiv_image_io`）、颜色空间转换、高斯模糊（8U 定点与 32F 浮点）、缩放（`fiv_image_resize`，最近邻 / 双线性）、数据类型归一化 |
| 人脸视觉 | `app/face` | BlazeFace 检测、478 点 Landmark 网格、YuNet 检测；权重见 `app/face/models/` |
| YOLO26 | `app/yolo26` | 检测 / 实例分割 / 姿态 / 分类四任务推理（NMS-free 端到端解码）；权重见 `app/yolo26/models/` |

## 核心模块说明

**神经网络（`src/nn`）** — 卷积在 x86（AVX2）/ ARM（NEON）下做针对性优化；标准 / 深度可分离 / 逐点 / 可分离四种模式，覆盖 1×1、2×2、3×3、5×5 的 SIMD 内核，并支持显式四边 padding（`zero` / `replicate` 两种填充）。Attention 为复合节点，承载 YOLO26 的 C2PSA 自注意力（内部调用 `src/math` 的行内 Softmax）。训练侧为 SGD + 反向传播，损失支持交叉熵与 MSE，模型可 `save` / `load` 为二进制。

**数学算子（`src/math`）** — Sigmoid、SiLU（SwiGLU）、行内 Softmax、RMSNorm 的单精度（`FIV_32F1`）与双精度（`FIV_64F1`）实现。其中 Sigmoid 与 Softmax 带 AVX2 + NEON 双路内核，SwiGLU 带 AVX2 内核，RMSNorm 为标量实现；行内 Softmax 同时是 `src/nn` 中 Attention 复合节点的算子底座。

**图像（`src/image`）** — 编解码内置 stb（`import/image_io/`，header-only）；缩放提供最近邻与双线性两条路径；高斯模糊同时提供 8U 定点（16 位中间量、单次舍入）与 32F 浮点实现；色彩空间转换（BGR↔RGB、RGB / BGR→GRAY）与数据类型归一化（`8U → [0,1] / [-1,1] / mu-sigma`）相互正交、可独立组合。

**人脸 / YOLO26（`app/`）** — 两者均为端到端应用，对外仅暴露少量构造 / 推理 / 释放接口：

| 应用 | 公开接口 |
|---|---|
| BlazeFace | `fiv_create_face_detetor` / `fiv_face_detector_on_image` / `fiv_release_face_detector` |
| Landmark | `fiv_create_landmark_graph` / `fiv_landmark_graph_run_inference` / `fiv_release_landmark_graph` |
| YuNet | `fiv_yunet_build_graph` / `fiv_yunet_preprocess` / `fiv_yunet_detect` / `fiv_yunet_release_graph` |
| YOLO26 | `fiv_create_yolo26_{detector,segmenter,pose,classifier}` / `fiv_yolo26_<task>_on_image` / `fiv_release_yolo26_<task>`；深度分支可用 `YOLO26_REF_DIR` 指向参考目录与 PyTorch 输出对拍 |

## 测试

每个功能在 `build/` 下生成一个独立测试二进制，**成功时打印 `PASS=n FAIL=0` 并以 0 退出**（部分用例打印 `ALL PASS`）。按领域分组：

| 领域 | 测试目标 |
|---|---|
| 基础 | `test_darray`、`test_ctensor`、`bench_vec_dot` |
| 矩阵 | `test_mat_transpose` `test_mat_vec` `test_mat_mul_vec_db` `test_mat_mul` `test_mat_mul_db` `test_mat_reduce_db` `test_mat_cholesky` `test_mat_lu` `test_mat_eig_sym` `test_mat_svd` `test_mat_svd_jacobi` |
| 神经网络 | `test_nn`、`test_nn_conv2d`（含数值梯度校验）、`test_nn_mnist` `test_nn_mnist_conv`（无数据则跳过） |
| 图像 / 人脸 | `test_blazeface`（需 `src/reference/` 参考实现，未随仓库发布）、`test_image_io` `test_color_space_perf` `test_data_convert` `test_face_api` `test_landmark_ops` `test_landmark_net` `test_landmark_api` `test_landmark_e2e` `test_yunet_api` `test_yunet_resize` |
| YOLO26 | `test_yolo26_ref` `test_yolo26_ops` `test_yolo26_attn` `test_yolo26_blocks` `test_yolo26_net` `test_yolo26_e2e` `test_yolo26_seg` `test_yolo26_pose` `test_yolo26_cls` `test_yolo26_model`（除 `model` 外需 `YOLO26_REF_DIR` 真值目录，可由 `app/yolo26/test/gen_ref*.py` 重新生成） |
| 线性规划 | `test_lp_sp_matrix` `test_lp_ops` `test_lp_rescale` `test_lp_pdhg` `test_lp_solve` `test_lp_solve13` |

## 线性规划（PDLP）

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

```sh
make -C build test_lp_sp_matrix test_lp_ops test_lp_rescale test_lp_pdhg test_lp_solve test_lp_solve13
cd build && ./test_lp_solve
```

- `test_lp_sp_matrix` — 稀疏存储、SpMV、归约、CSRL 打包
- `test_lp_ops` — 向量原语与稠密/稀疏矩阵抽象
- `test_lp_rescale` — Ruiz 与 Pock-Chambolle 预处理
- `test_lp_pdhg` — PDHG 步、原始权重更新、KKT 误差
- `test_lp_solve` — 通用 LP 套件端到端
- `test_lp_solve13` — 13 个标准问题（含不可行、无界、退化情形）

`make oracle` 与 `make oracle13` 另提供与 torch `pdlp.py` 的跨语言对拍（脚本在 `src/lp/extra_test/`）。

## 许可

GPL v3。见 [License/LICENSE](License/LICENSE)。Copyright (C) 2026 Celery Chen。

---

## English

Please see **[README_EN.md](README_EN.md)** for the full English version.

FastIV is a high-performance algorithm library for speech, image & computer vision, SLAM, machine learning & neural networks, SPICE simulation, and financial time-series prediction, with targeted SIMD optimizations for x86 (AVX2) and ARM (NEON). Written in C23, pure C, no third-party runtime dependencies.

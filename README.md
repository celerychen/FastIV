# FastIV

## 中文

本项目致力于语音、图像与计算机视觉，SLAM、机器学习与神经网络、大模型、SPICE仿真，金融时序预测等相关的高性能算法，在各类异构体系架构下的最优化实现。

### 模块简介

- **张量与基础数据结构**：提供统一的张量抽象与动态数组，作为上层算法的基础数据容器，支持任意维度张量及其逐元素二元运算。
- **矩阵运算**：涵盖矩阵转置、矩阵-向量乘法与矩阵-矩阵乘法，面向 x86 / ARM 等异构架构做针对性优化。
- **神经网络**：提供计算图构建、前向推理与反向传播训练能力。算子涵盖线性层（Linear）、激活函数（ReLU / ReLU6）、二维卷积（标准 / 深度可分离 / 逐点 / 可分离，含 3×3 与 5×5 stride-2 的 SIMD 优化内核）、最大池化（Max2D）、展平（Flatten）、逐元素相加（Add）与填充（Pad）；支持模型保存与加载。卷积在 x86（AVX2）/ ARM（NEON）架构下做了针对性优化。
- **高性能人脸检测器**：基于 BlazeFace 的端到端人脸检测应用，封装于 `app/face/`，对外仅暴露 `fiv_create_face_detetor` / `fiv_face_detector_on_image` / `fiv_release_face_detector` 三个接口，权重随仓库发布于 `app/face/models/`。

## English

FastIV is dedicated to high-performance algorithms for speech, image and computer vision, SLAM, machine learning and neural networks, large models, SPICE simulation, and financial time-series prediction, with optimized implementations across a variety of heterogeneous architectures.

### Modules

- **Tensor & data structures**: A unified tensor abstraction and dynamic array that serve as the foundational data containers for higher-level algorithms, supporting N-dimensional tensors and element-wise binary operations.
- **Matrix operations**: Matrix transpose, matrix-vector multiplication and matrix-matrix multiplication, with targeted optimizations for heterogeneous architectures such as x86 and ARM.
- **Neural networks**: Computation-graph construction, forward inference and backpropagation training. Operators include linear layers (Linear), activations (ReLU / ReLU6), 2D convolutions (standard / depthwise / pointwise / separable, with SIMD-optimized 3x3 and 5x5 stride-2 kernels), max pooling (Max2D), flatten, element-wise add and pad; with model save/load support. Convolutions are optimized for x86 (AVX2) and ARM (NEON).
- **High-performance face detector**: An end-to-end BlazeFace-based face detector in `app/face/`, exposing only three public APIs — `fiv_create_face_detetor` / `fiv_face_detector_on_image` / `fiv_release_face_detector`, with weights shipped in the repo under `app/face/models/`.

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
- `test_nn` — neural-network unit and integration tests (no dataset required)
- `test_nn_conv2d` — convolution forward/backward tests incl. numeric-gradient check (no dataset required)
- `test_nn_mnist` — end-to-end MLP MNIST accuracy test (skips if data is absent)
- `test_nn_mnist_conv` — end-to-end CNN MNIST test (conv→ReLU→Max2D→flatten→Linear), requires >98% accuracy (skips if data is absent)
- `test_blazeface` — BlazeFace short-range face-detector inference (5x5 stride-2 stem conv), uses the SIMD conv kernels
- `test_face_api` — black-box contract test of the public face-detector API (create / on_image / release) over the bundled model in `app/face/models/`

Each prints `PASS=n FAIL=0` on success and exits non-zero on failure.

## License

GPL v3. See [License/LICENSE](License/LICENSE). Copyright (C) 2026 Celery Chen.

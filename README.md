# FastIV

## 中文

本项目致力于语音、图像与计算机视觉、机器学习与神经网络、大模型、SPICE 仿真相关的高性能算法，在各类异构体系架构下的最优化实现。

### 模块简介

- **张量与基础数据结构**：提供统一的张量抽象与动态数组，作为上层算法的基础数据容器，支持任意维度张量及其逐元素二元运算。
- **矩阵运算**：涵盖矩阵转置、矩阵-向量乘法与矩阵-矩阵乘法，面向 x86 / ARM 等异构架构做针对性优化。
- **神经网络**：提供计算图构建、前向推理与反向传播训练能力，内置线性层与激活函数等基础算子，支持模型保存与加载。

## English

FastIV is dedicated to high-performance algorithms for speech, image and computer vision, machine learning and neural networks, large models, and SPICE simulation, with optimized implementations across a variety of heterogeneous architectures.

### Modules

- **Tensor & data structures**: A unified tensor abstraction and dynamic array that serve as the foundational data containers for higher-level algorithms, supporting N-dimensional tensors and element-wise binary operations.
- **Matrix operations**: Matrix transpose, matrix-vector multiplication and matrix-matrix multiplication, with targeted optimizations for heterogeneous architectures such as x86 and ARM.
- **Neural networks**: Computation-graph construction, forward inference and backpropagation training, with built-in basic operators such as linear layers and activation functions, plus model save/load support.

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
- `test_nn_mnist` — end-to-end MNIST accuracy test (skips if data is absent)

Each prints `PASS=n FAIL=0` on success and exits non-zero on failure.

## License

GPL v3. See [License/LICENSE](License/LICENSE). Copyright (C) 2026 Celery Chen.

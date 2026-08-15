# FastIV

> Fast Image and Vision — 语音、图像视频、计算机视觉与神经网络、机器学习、大模型等算法在各类异构平台下的**最优化实现**。

FastIV 是一套面向性能的 C 算法库：同一套代码在 **x86（AVX2/FMA）**、**ARM（NEON）** 与纯标量平台间自动选择最优内核，追求每个算子的极致吞吐与缓存友好。

## 优化理念

- **SIMD 多平台内核**：AVX2 / NEON 手写 intrinsics，无 SIMD 时自动回退标量实现
- **缓存感知**：矩阵分块大小随 L3 缓存自适应（`-DFIV_L3_CACHE_BYTES` 可调），小矩阵不分块、大矩阵分块
- **零大块临时分配**：alpha/beta 直通内核，不引入额外 M×N 缓冲
- **C23 严格类型**：枚举显式底层类型，结构体内存布局紧凑且可预测

## 当前模块

| 模块 | 说明 |
|------|------|
| `fiv_darray` | 稠密数组（创建/释放/访问/克隆等） |
| `fiv_ctensor` | 1D~5D 张量与图像类型，float32/int32 二元算子（add/sub/mul/div） |
| `fiv_matrix` | 矩阵转置、矩阵×向量、通用 GEMM（小矩阵 + L3 感知分块） |

每个功能配有独立测试（`src/test/`），覆盖正确性、边界与错误路径。

## 目录结构

```
api/            对外头文件
src/ctensor/    张量与算子实现
src/mat/        矩阵运算实现
src/test/       单元测试（按功能拆分）
src/reference/  参考实现（不参与构建、不入库）
build/          构建脚本
License/        GPL v3 许可证
```

## 构建

Windows（需支持 C23 的 GCC 或 MSVC）：

```
build\build.bat              # 默认 GCC
build\build.bat msvc         # MSVC
```

macOS / Linux：

```
make -C build          # 构建全部测试（x86 自动启用 AVX2，ARM 自动启用 NEON）
make -C build run      # 构建并运行
make -C build clean
```

产物为多个独立测试可执行文件（`build/` 下）。

## License

[GPL v3](License/LICENSE) © 2026 Celery Chen

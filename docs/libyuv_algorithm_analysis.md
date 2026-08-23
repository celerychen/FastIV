# libyuv 算法实现分析（缩放 + YUV→RGB）

> 目的：把 `src/reference/libyuv` 下两套核心子系统——**图像缩放（scale）**与 **YUV→RGB 色彩空间转换（convert）**——的算法思路与实现技术细节整理清楚，作为 FastIV 后续参考。
>
> 分析对象：`src/reference/libyuv/libyuv/`（libyuv 源码树）。
> 说明：本轮只做**分析**，不改动、不移植代码。参考目录按项目约定保持原样（git 已忽略 `src/reference/`）。
>
> 价值：libyuv 与 FastIV 卷积优化是**同一类工程哲学**——定点整数、无分支 SIMD 内核、边界单独标量、CPU feature 分派、把格式/对齐当第一约束。理解它能少走弯路。

---

## 0. 总体定位与设计哲学

libyuv 不是通用高质量重采样器（没有 Lanczos / cubic / 抗混叠），而是一个面向**实时视频/相机流**的高性能、低复杂度、零内存越界风险的整型定点库。两条主线共享同一套原则（`docs/filtering.md` 明确）：

1. **最大性能**：SIMD + 特化整数比例路径，热点全部下沉到"一行一个 row 函数"。
2. **绝不内存越界**：所有采样坐标都被 clamp；行缓冲固定大小、每帧不 malloc。
3. **最小代码/复杂度**：刻意追求"与 ffmpeg 在所有比例上下采样逐像素一致、可重复、纯整数、无越界"，而不是理论最优滤波器。

两子系统的结构高度一致：

- **可分离处理**：缩放拆成水平/垂直两个一维操作；YUV→RGB 拆成每像素标量公式（Y 项 + U 项 + V 项）。
- **顶层分发 + row 函数**：`ScalePlane()` / `I420ToARGBMatrix()` 负责参数校验、CPU feature 检测、逐行循环；真正的算术都在 `ScaleXxxRow_*` / `I422ToARGBRow_*` 里，每个架构一套（`_C` / `_SSSE3` / `_AVX2` / `_NEON` / `_SVE2` / `_RVV` / `_LSX` ...）。
- **对齐主路径 + `_Any_` 尾部**：宽度对齐到向量宽度时用完整向量版；否则用 `_Any_` 版处理任意宽度（内部做未对齐加载或标量补齐）。

---

## 1. 图像缩放（scale）算法分析

> 本节为上一轮分析结论的归档。

### 1.1 核心：可分离缩放 + 16.16 定点步进

- 用 **16.16 定点**表示所有坐标。`dx = FixedDiv(src_width, dst_width)` 是"每输出一个像素，在源里走多少"的斜率。
- `ScaleSlope()` 根据滤波模式算出起始 `x/y` 和步长 `dx/dy`，并处理**居中（centering）**：
  - 下采样：`x = dx>>1 - 0.5`，采样点落在盒子正中；
  - 上采样：用 `FIXEDDIV1`（`((src<<16)-0x10001)/((dst<<16)-0x10000)`）把最后像素"挤"到不到最后一像素的位置，避免"读最后一像素 100% + 下一像素 0%"的越界读。
- 负数 `src_width` = 水平镜像（斜率取负 + 起点补偿）。

### 1.2 四种滤波模式

| 模式 | 实现 | 要点 |
|------|------|------|
| `kFilterNone`（最近邻） | `ScalePlaneSimple` | `xi = x>>16` 直接取源像素。下采样取中像素、上采样复制。2× 上采样有 `ScaleColsUp2` 特化。 |
| `kFilterLinear`（仅水平双线性） | `ScaleFilterCols` + `InterpolateRow(fraction=0)` | 只对色度做水平缩放（如 I422→I444），垂直不插值。 |
| `kFilterBilinear`（双线性） | `ScaleFilterCols`（水平）+ `InterpolateRow`（垂直） | 2 行滚动缓冲，换行只做指针交换 `rowptr += rowstride; rowstride = -rowstride`。 |
| `kFilterBox`（面积平均） | `ScalePlaneBox`（`ScaleAddRow` + `ScaleAddCols`） | 最高质量下采样；`uint16`/`uint32` 累加防溢出；按 `dx` 定点区分 `ScaleAddCols0/1/2`。 |

逐像素混合公式（定点）：

- 水平 `ScaleFilterCols_C`：`BLENDER(a,b,f) = a + ((f*(b-a) + 0x8000) >> 16)`，`f = x & 0xffff`（16 位小数，`+0x8000` 四舍五入）。
- 垂直 `InterpolateRow`：`dst = (row0*(256-yf) + row1*yf) >> 8`，`yf = (y>>8)&255`（8.8 定点）。

### 1.3 整数比例的特化路径（性能主来源）

`ScalePlane()` 主分发**优先识别整数比例**并用极致优化的 row 函数（`scale.cc` 的 `ScalePlaneDown2/4/34/38`、`ScalePlaneUp2_*`）：

- 1:1 → `CopyPlane` 直拷；
- 3/4、1/2、3/8、1/4 下采样 → 各自固定 tap/权重（2× 下采样盒式用 `urhadd` 四舍五入半加）；
- 2× 上采样 → `ScalePlaneUp2_Linear/Bilinear`。

这些覆盖视频/相机最频繁的分辨率切换。

### 1.4 SIMD 实现细节（以 NEON 为例）

- `ScaleFilterCols_NEON`：`vdup` 把 `x`、`dx` 广播成 4 路，预计算 `x, x+dx, …, x+7dx` 共 8 个定点位置；`trn1` 拆出 16 位整数坐标 `xi` 与小数 `f`；`ssubl` 算 `b-a`、`mul` 乘 `f`、`rshrn` 移位加、`xtn` 截断回 `uint8`——**一次循环处理 8 个目标像素**。
- `InterpolateRow_NEON`：两行 `umull/umlal` 做 8bit×8bit→16bit 乘积再 `rshrn` 右移 8 四舍五入；`yf==0`（纯拷贝）和 `yf==128`（50/50）走 `urhadd` 快速路径，省掉乘法。
- 同样有 SSSE3 / AVX2 / LSX / RVV / SVE2 / SME 版本；16bit 深度图用 `uint32` 累加缓冲。

### 1.5 边界与内存安全

- 下采样永远有合法邻域；上采样通过 `max_y = (src_height-1)<<16` 把 `y` clamp 到末行，杜绝越界读。
- 行缓冲固定大小（`(dst_width+31)&~31` 对齐），每帧不 malloc。
- `ScalePlane` 入口拒绝 `width>32768`，防止 `FixedDiv` 有符号溢出导致 SIGFPE。

---

## 2. YUV→RGB 色彩空间转换算法分析

### 2.1 色彩空间与矩阵

对外 API 在 `include/libyuv/convert_argb.h`，核心是 `I420ToARGB()` 及其 `*Matrix` 形式。转换矩阵由 `struct YuvConstants` 承载（声明见 `include/libyuv/row.h:967` 与 `:982`，分 ARM/RISC-V 版与 Intel 版两套布局）。

libyuv 提供多套标准矩阵常量（`convert_argb.h:25-36`）：

- `kYuvI601Constants` / `kYvuI601Constants` —— BT.601 limited range（视频标准，Y∈[16,235]）
- `kYuvJPEGConstants` —— BT.601 full range（JPEG，Y∈[0,255]）
- `kYuvH709Constants` / `kYuvF709Constants` —— BT.709（HD）
- `kYuv2020Constants` / `kYuvV2020Constants` —— BT.2020（UHD）
- 带 `VU` 后缀的是 UV 互换版（如 `kYuvI601ConstantsVU = kYvuI601Constants`），配合宏 `NV12ToABGRMatrix` 等做端序/通道互换，无需另写一份代码。

标准 BT.601 limited-range 公式为（浮点参考）：

```
R = 1.164*(Y-16) + 1.596*(V-128)
G = 1.164*(Y-16) - 0.391*(U-128) - 0.813*(V-128)
B = 1.164*(Y-16) + 2.018*(U-128)
```

libyuv 的全部工作，就是**把这套公式用整数定点 + SIMD 重写成零分支、可读、防越界**的形态。

### 2.2 整数定点数学模型（2.14 定点）

两个关键设计点（C 参考见 `source/row_common.cc:1763-1786` 的 `LOAD_YUV_CONSTANTS` / `CALC_RGB16`）：

**(a) Y 的字节复制技巧 —— `y * 0x0101`**
`uint32_t y32 = y * 0x0101;` 把单字节 `y` 复制成 16 位 `0x00YY00YY`（即 `y*257`）。这样一次 16×16→32 乘法 `y32 * yg >> 16` 就能完成 `Y` 的缩放项，且**不必先减 16**——"减 16" 与 Y 的偏置被整体折叠进常数表（`yb`/`bb`/`bg`/`br` 或 ARM 的 `kRGBCoeffBias`）。

**(b) U/V 的显式去偏 —— `ui -= 0x80; vi -= 0x80`**
U/V 是**有符号**中心化的，必须显式减 128 才有正负；而 Y 是通过常数偏置一次性吸收，二者处理方式不同，是这套定点模型的精妙处。

**ARM / RISC-V 版（`CALC_RGB16` 的 `#if` 上半段，`row_common.cc:1763`）**：

```
ub,ug,vg,vr  <- kUVCoeff 派生的有符号系数（每通道）
bb,bg,br     <- kRGBCoeffBias 的偏置项（已含 Y 的 (Y-16) 偏置）
y1 = (y32 * yg) >> 16
b16 = y1 + (u * ub) - bb
g16 = y1 + bg - (u * ug + v * vg)
r16 = y1 + (v * vr) - br
```

中间结果 `b16/g16/r16` 是 **2.14 定点**（16 位，含 14 位小数），最终 `Clamp(... >> 6)` 收窄到 8 位。

**Intel 版（`#else` 分支，`row_common.cc:1769`）**：

```
ub = kUVToB[0]; ug = kUVToG[0]; vg = kUVToG[1]; vr = kUVToR[1]
yg = kYToRgb[0]; yb = kYBiasToRgb[0]
y1 = (y32 * yg >> 16) + yb
ui = (int8_t)u - 0x80; vi = (int8_t)v - 0x80
b16 = y1 + (ui * ub)
g16 = y1 - (ui * ug + vi * vg)
r16 = y1 + (vi * vr)
```

两者数学等价，只是常数表的**存放布局不同**（见 2.6）。

### 2.3 C 参考实现（`source/row_common.cc`）

- `YuvPixel()`（`row_common.cc:1790`）：单像素 8→8，先 `CALC_RGB16` 得 16 位，再 `Clamp(...>>6)` 写出 B/G/R。是所有 SIMD 版本的"正确性基准"。
- `I444ToARGBRow_C()`（`:1954`）：Y/U/V 三者 1:1，逐像素 `YuvPixel` + 写 A=255，步进 4 字节/像素。
- `I422ToARGBRow_C()`（`:1990`）：**每 2 个 Y 共享 1 对 U/V**（4:2:2）。循环 `x += 2`，两个 Y 用同一 `src_u[0]/src_v[0]` 算两次，步进 8 字节（2 像素）；奇数宽度末尾单独补 1 像素。
- I420 走的就是 `I422ToARGBRow` + **垂直方向每两行共享一行 U/V**（见 2.4）。

另有 10bit/12bit 输入版本（`YuvPixel10_16` / `YuvPixel12_16`，`:1823`/`:1842`）：把高比特 Y 用 `<<6 | >>4` 之类做 bit-replication 摊成 16 位，U/V 先 `>>2`/`>>4` 降回 8 位再用同一套 `CALC_RGB16`，保证 HDR 路径与 8bit 路径共用内核。

### 2.4 顶层分发架构（`source/convert_argb.cc`）

以 `I420ToARGB()` 为例：

```
I420ToARGB() -> I420ToARGBMatrix(..., &kYuvI601Constants, ...)   // 选默认 BT.601 矩阵
I420ToARGBMatrix():
  1. 参数校验 (src/dst 非空, width>0, height!=0/INT_MIN)
  2. 负 height => 翻转图像 (dst 指针移到末行, stride 取负)
  3. CPU feature 分派: 选 I422ToARGBRow_C / _SSSE3 / _AVX2 / _AVX512BW / _NEON / _SVE2 / _SME / _LSX / _LASX / _RVV
     - 对齐宽度用完整向量版 (IS_ALIGNED(width, 8/16/32))
     - 否则用 _Any_ 版
  4. 逐行循环:
       I422ToARGBRow(src_y, src_u, src_v, dst, yuvconstants, width)
       src_y += src_stride_y
       if (y & 1) { src_u += src_stride_u; src_v += src_stride_v; }  // 垂直半采样
```

要点：

- **I420 的垂直色度上采样是免费的**：U/V 指针每隔一行才前进一行（`y & 1`），相邻两行 RGB 复用同一对 U/V —— 这就是 4:2:0 的"每行重复色度"语义，零额外计算。
- **Filter 变体**（`I420ToARGBMatrixFilter`，`convert_argb.cc:2067`）：对色度做**双线性**上采样（先 `ScalePlane` 风格的 2× 水平/垂直插值再转 RGB），质量更高但更慢，用于严格正确的高质量路径。
- 所有 YUV 变体（I420/I422/I444/NV12/NV21/YUY2/I010/I210...）都收口到同一个 `I422ToARGBRow`（或 NV 版 `NVTORGB`），靠**行读取宏**（`READYUV422` / `READYUV210` ...）和 **UV 是否交错**区分，矩阵选择靠 `kYuv*` 常量指针。

### 2.5 ARM NEON 实现技术（`source/row_neon64.cc`）

`I422ToARGBRow_NEON`（`:443`）主循环：

```asm
YUVTORGB_SETUP                       // ld4r 把 kUVCoeff 广播成 v28-31, kRGBCoeffBias 广播成 v24-27
movi  v19.8b, #255                  // A 恒为 255
1:
  READYUV422                        // 读 8 个 Y, 4 个 U, 4 个 V; zip1 把每个 U/V 复制成对
  I4XXTORGB                        // YUV -> 2.14 定点 RGB (v16=B, v17=G, v18=R, 均为 .8h)
  RGBTORGB8                        // uqshrn vX.8b, vX.8h, #6  (2.14 -> 8bit, 饱和收窄)
  st4   {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst]], #32   // 一次写 8 个 ARGB 像素
  subs  width, width, #8
  b.gt  1b
```

关键技巧：

- **`READYUV422`**：`zip1 v1.8b, v1.8b, v1.8b` 把 4 个 U 字节复制成 8 个（每对 Y 共享一个 U），一次喂足 8 像素。
- **`I4XXTORGB`**（核心乘加，`row_neon64.cc:175`）：
  - `umull v0.4s, v0.4h, v24.4h` + `umull2 v3.4s, v0.8h, v24.8h` → Y 项 × Y 系数（2.14）；
  - `umull v4.8h, v1.8b, v28.8b` / `umull v5.8h, v2.8b, v29.8b` → U 与 V 的乘项；
  - `umull v6.8h, v1.8b, v30.8b` + `umlal v6.8h, v2.8b, v31.8b` → U×ug + V×vg 的绿色交叉项；
  - `add v16/17/18 = Y项 + 偏置`, `uqsub ... v6` → **饱和减**掉交叉项，即 `G = Y_bias - (U*ug + V*vg)`。
- **`RGBTORGB8`**：`uqshrn`（unsigned saturating narrow right shift）一步完成 `>>6` + 8 位饱和 + 收窄回 `.8b`。这是定点输出的标准收尾。
- **`st4`**：把 B/G/R/A 四通道用单条四结构存储写出，充分利用 NEON 的交织存储。
- `prfm pldl1keep` 预取源数据，掩盖访存延迟。

`NVTORGB`（`row_neon64.cc:158`）与 `I4XXTORGB` 几乎相同，只是 U/V 在单个寄存器的高低半区（NV12/NV21 交错），体现"同一算法、不同取数方式"的复用。

### 2.6 Intel SSSE3/AVX2 实现技术（`source/row_win.cc`）

`I422ToARGBRow_AVX2`（`:1091`）的精髓是 **`vpmaddubsw`（SSSE3 `pmaddubsw`）的"无符号字节 × 有符号字节 → 有符号 16 位"乘加**：

```cpp
ymm_kUVToB = loadu(kUVToB)   // 32 字节系数表
ymm3 = unpacklo_epi8(u,v)    // 把 U/V 交错成一个向量
ymm3 = sub_epi8(ymm3, 0x80)  // U/V 去偏 (中心化到 [-128,127])
ymm4 = mulhi_epu16(y, kYToRgb)            // Y 项: Y*yg 取高 16 位
ymm0 = maddubs_epi16(kUVToB, ymm3)        // U/V 的 B 项 (无符号×有符号→有符号16)
ymm1 = maddubs_epi16(kUVToG, ymm3)        // G 项 (含 ug+vg 同号合并)
ymm2 = maddubs_epi16(kUVToR, ymm3)        // R 项
ymm4 = add_epi16(ymm4, kYBiasToRgb)       // + Y 偏置
ymm0 = adds_epi16(ymm0, ymm4)             // B = Y项 + U*ub
ymm1 = subs_epi16(ymm4, ymm1)             // G = Y项 - (U*ug+V*vg)   <-- 用减实现负号
ymm2 = adds_epi16(ymm2, ymm4)             // R = Y项 + V*vr
ymm0/1/2 = srai_epi16(., 6)               // 2.14 -> 8bit (算术右移, 含符号)
ymm0/1/2 = packus_epi16(., .)             // 饱和打包回 8bit
// unpacklo/permute4x64 重排成 ARGB 交错, storeu 写出 2×8 像素
```

关键技巧：

- **`kUVToB/kUVToG/kUVToR` 是 32 字节（16 对）查表**：利用 `maddubs` 一次性对 16 个 (字节, 字节) 对做"乘+加"，通过系数表的符号与配对，**把正项（ub/vr）和负项（ug/vg）都编码进去**——这是 libyuv 著名的"SMUL"定点技巧，避免分别做有符号乘法。
- **G 通道用 `subs_epi16(ymm4, ymm1)`** 而非另算：`ymm1` 里已经把 `ug` 与 `vg` 预合并成同号项，于是 `Y - (U*ug+V*vg)` 直接变成一次饱和减。
- **`mulhi_epu16`** 取 Y×系数乘积的高 16 位，等价于 `>>16`，配合 `add kYBiasToRgb` 完成 Y 缩放与去 16 偏置。
- 最终 `srai #6` + `packus`（unsigned saturate pack）对应 NEON 的 `uqshrn #6`，两架构收尾完全一致。

### 2.7 安全与边界

- `Clamp()`（`row_common.cc` 的 `STATIC_CAST(uint8_t, Clamp(...))`）和 SIMD 的 `uqshrn`/`packus` 做**饱和**，防止 RGB 溢出到负或 >255。
- 顶层 `I420ToARGBMatrix` 拒绝空指针、`width<=0`、`height==0/INT_MIN`。
- **负 height 翻转**图像：把 `dst_argb` 移到末行、stride 取负，循环逻辑不变即可实现垂直镜像/翻转输出。
- 行函数只处理 `width` 像素，对齐主路径 + `_Any_` 尾部保证**任意宽度无越界**；常数表 `kYuv*` 是静态只读全局，无每帧分配。

---

## 3. 与 FastIV 卷积优化的共性

| 维度 | libyuv | FastIV 卷积（`fiv_conv2d_plane_*`） |
|------|--------|--------------------------------------|
| 数值表示 | 2.14 / 16.16 定点整数 | 整型累加（conv sum），不依赖浮点 |
| 内核结构 | 无分支 SIMD 内核 + 边界单独标量 | `fiv_conv2d_plane_5x5_s2` 无分支内部 + `fiv_conv2d_px_pad_5x5_s2` 标量边界 |
| 分派 | `TestCpuFlag` + `_C/_NEON/_AVX2/_SVE2/...` + 对齐主路径/`_Any_` | `FIV_USE_ARM_NEON/FIV_USE_AVX2` 宏 + NEON/AVX2 分支 |
| 性能来源 | 整数比例特化（`ScalePlaneDown2/4/34/38`）+ row 函数下沉 | 特殊卷积（3×3/5×5、s1/s2）拆独立 plane 函数 + SIMD |
| 格式/对齐 | 行缓冲对齐 `(w+31)&~31`，4 通道 `st4` 交织 | NCHW 布局、连续 `FIV_32F1`、向量宽度对齐 |
| 正确性基准 | `*_C` 参考实现（如 `YuvPixel`） | `fiv_conv2d_generic_std/dw` 通用标量（gold） |

**可借鉴点**：当新写一个高性能算子时，优先把"纯算术内核"与"参数校验/循环/分派"分离；内核内部追求无分支 + SIMD，边界/尾部单独标量；保持一个**纯 C 参考实现**作为 SIMD 正确性对照（FastIV 的 `test_nn_conv2d` 数值梯度校验与 libyuv 的 `*_C` 是同一思路）。

---

## 4. 关键文件与符号速查

| 主题 | 文件 | 关键符号 |
|------|------|----------|
| 缩放 API | `include/libyuv/scale.h` | `ScalePlane`, `kFilterNone/Linear/Bilinear/Box` |
| 缩放分发 | `source/scale.cc`, `source/scale_common.cc` | `ScalePlane`, `ScaleSlope`, `ScalePlaneBox/Simple`, `FixedDiv` |
| 缩放 SIMD | `source/scale_neon64.cc`, `row_neon64.cc` | `ScaleFilterCols_NEON`, `InterpolateRow_NEON`, `ScaleAddRow` |
| YUV→RGB API | `include/libyuv/convert_argb.h` | `I420ToARGB`, `kYuvI601Constants`, `kYvu*`, `*Matrix` |
| YUV→RGB 分发 | `source/convert_argb.cc` | `I420ToARGBMatrix`, `I422ToARGBRow` 分派 |
| YUV→RGB 参考 | `source/row_common.cc` | `YuvPixel`, `I422ToARGBRow_C`, `CALC_RGB16`, `LOAD_YUV_CONSTANTS` |
| YUV→RGB 常量 | `include/libyuv/row.h:967/982` | `struct YuvConstants`（ARM / Intel 两套布局） |
| YUV→RGB NEON | `source/row_neon64.cc` | `I422ToARGBRow_NEON`, `I4XXTORGB`, `READYUV422`, `RGBTORGB8` |
| YUV→RGB AVX2 | `source/row_win.cc` | `I422ToARGBRow_AVX2`, `maddubs_epi16`（SMUL） |

> 注：`src/reference/` 已被 `.gitignore` 排除（项目约定"参考代码默认不动"）。本文件置于项目 `docs/`，属于可跟踪的 FastIV 自身文档，不修改参考源码。

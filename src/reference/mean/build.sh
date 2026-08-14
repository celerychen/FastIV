#!/usr/bin/env bash
# 构建脚本 —— 必须用 GCC 15.2 的绝对路径
# PATH 上默认的 gcc 是 miniconda 自带的 5.3.0，已损坏不可用
set -e

GCC=/c/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64msvcrt-13.0.0-r1/mingw64/bin/gcc
OUT=build
mkdir -p "$OUT"

COMMON="-std=c11 -Wall -Wextra -Wno-unused-parameter -fno-lto -I."

echo "== [1/6] 原始实现，-O2 -mavx2（保持代码原样的基准编译） =="
$GCC $COMMON -O2 -mavx2 -DMEANVAR_NO_MAIN -c mean_var.c -o "$OUT/mean_var_o2.o"

echo "== [2/6] 同一份原始源码，-O3 -march=native（考察自动向量化 / FMA 收缩） =="
$GCC $COMMON -O3 -march=native -DMEANVAR_NO_MAIN \
     -Dmean_variance_serial=mean_variance_serial_autovec \
     -Dmean_variance_avx2_welford=mean_variance_avx2_welford_o3 \
     -Dwelford_merge=welford_merge_o3 \
     -c mean_var.c -o "$OUT/mean_var_o3.o"

echo "== [3/6] 标量 sum/sumsq 基线 =="
$GCC $COMMON -O2 -c baselines.c -o "$OUT/baselines.o"

echo "== [4/6] 优化版实现 =="
$GCC $COMMON -O3 -march=native -c mean_var_opt.c -o "$OUT/mean_var_opt.o"

echo "== [5/6] 正确性测试 =="
$GCC $COMMON -O2 -march=native -c test_correctness.c -o "$OUT/test_correctness.o"
$GCC "$OUT/test_correctness.o" "$OUT/mean_var_o2.o" "$OUT/mean_var_o3.o" \
     "$OUT/baselines.o" "$OUT/mean_var_opt.o" -o "$OUT/test_correctness.exe" -lm

echo "== [6/6] 性能测试 =="
$GCC $COMMON -O2 -march=native -c bench.c -o "$OUT/bench.o"
$GCC "$OUT/bench.o" "$OUT/mean_var_o2.o" "$OUT/mean_var_o3.o" \
     "$OUT/baselines.o" "$OUT/mean_var_opt.o" -o "$OUT/bench.exe" -lm

echo ""
echo "构建完成:"
echo "  $OUT/test_correctness.exe"
echo "  $OUT/bench.exe"

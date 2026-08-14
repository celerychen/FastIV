# 简单构建（gcc / clang）。MSVC 见下方注释。
CC      ?= cc
CFLAGS  ?= -O3 -Wall -Wextra
AVX     ?= -mavx2 -mfma

# MSVC 等价命令（去掉上面的 make，用这一行）：
#   cl /O2 /arch:AVX2 /fp:fast test_softmax.c softmax_basic.c softmax_online.c softmax_avx2.c
#   cl /O2 /arch:AVX2 /fp:fast bench_softmax.c softmax_basic.c softmax_online.c softmax_avx2.c

all: test_softmax bench_softmax

test_softmax: test_softmax.c softmax_basic.c softmax_online.c softmax_avx2.c softmax.h
	$(CC) $(CFLAGS) $(AVX) -o $@ test_softmax.c softmax_basic.c softmax_online.c softmax_avx2.c

bench_softmax: bench_softmax.c softmax_basic.c softmax_online.c softmax_avx2.c softmax.h
	$(CC) $(CFLAGS) $(AVX) -o $@ bench_softmax.c softmax_basic.c softmax_online.c softmax_avx2.c

clean:
	rm -f test_softmax bench_softmax

.PHONY: all clean

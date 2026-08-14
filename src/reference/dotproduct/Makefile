# 简单构建（gcc / clang）。MSVC 见下方注释。
CC      ?= cc
CFLAGS  ?= -O3 -Wall -Wextra
AVX     ?= -mavx2 -mfma

# MSVC 等价命令（去掉上面的 make，用这两行）：
#   cl /O2 /arch:AVX2 /fp:fast test_dotproduct.c dot_basic.c dot_avx.c
#   cl /O2 /arch:AVX2 /fp:fast bench_dotproduct.c dot_basic.c dot_avx.c

all: test_dotproduct bench_dotproduct

test_dotproduct: test_dotproduct.c dot_basic.c dot_avx.c dot_avx_v2.c dotproduct.h
	$(CC) $(CFLAGS) $(AVX) -o $@ test_dotproduct.c dot_basic.c dot_avx.c dot_avx_v2.c

bench_dotproduct: bench_dotproduct.c dot_basic.c dot_avx.c dot_avx_v2.c dotproduct.h
	$(CC) $(CFLAGS) $(AVX) -o $@ bench_dotproduct.c dot_basic.c dot_avx.c dot_avx_v2.c

clean:
	rm -f test_dotproduct bench_dotproduct

.PHONY: all clean

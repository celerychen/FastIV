/*
 * FastIV - Fast image and vision
 * Copyright (C) 2026 Celery Chen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * See LICENSE file in project root for full license text.
 */

/*
 * detect_transcendental.c
 * 检测本机 CPU 是否支持「原生超越函数指令」(exp / sin / log 等硬件指令)，
 * 并说明它们对「批量 fp32 softmax」是否有意义。
 *
 * 编译(Windows / VS2022 cl):
 *   cl /O2 detect_transcendental.c
 * 运行:
 *   detect_transcendental.exe
 *
 * 核心结论(经 CPUID 实测):
 *   - x87 标量 FSIN/F2XM1  : 所有 x86 都有，但标量、80-bit、慢，不适合批量 SIMD。
 *   - AVX-512ER  VEXP2PS   : 仅 Xeon Phi(Knights Landing/Mill)，主流 CPU/AMD 均无；
 *                            精度 2^-23(约 7 位)，比多项式近似(~1e-7)粗。
 *   - AVX-512-FP16 VEXP2PH : 仅 Sapphire Rapids / Zen4，且只算 fp16，不是 fp32。
 *   - SIMD sin/cos         : x86 上不存在；只有 x87 标量慢指令。
 *   故在「主流 CPU + fp32」下，手写 AVX2 多项式近似就是当前最优解。
 */
#include <stdio.h>
#include <string.h>
#include <intrin.h>   /* __cpuidex / _xgetbv (MSVC) */
#ifdef _WIN32
#include <windows.h>  /* SetConsoleOutputCP(CP_UTF8) 避免控制台中文乱码 */
#endif

static void cpuid(unsigned eax, unsigned ecx, unsigned out[4]) {
    __cpuidex((int*)out, (int)eax, (int)ecx);
}

/* 检测 OS 是否已在 XCR0 中启用对应向量状态(否则即便 CPU 支持也会 #UD) */
static unsigned long long get_xcr0(void) {
    /* _xgetbv(0) 需要 /arch 层面支持；这里直接调用，老编译器也接受 */
    return _xgetbv(0);
}

int main(void) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);   /* 让控制台按 UTF-8 解读输出，中文不乱码 */
#endif
    unsigned regs[4];
    char vendor[13];

    cpuid(0, 0, regs);
    memcpy(vendor + 0, &regs[1], 4);   /* EBX */
    memcpy(vendor + 4, &regs[3], 4);   /* ECX */
    memcpy(vendor + 8, &regs[2], 4);   /* EDX */
    vendor[12] = 0;
    printf("Vendor : %s\n", vendor);

    /* 处理器品牌串 leaf 0x80000002..0x80000004 */
    char brand[49];
    for (int i = 0; i < 3; i++) {
        cpuid(0x80000002u + (unsigned)i, 0, regs);
        memcpy(brand + i * 16, regs, 16);
    }
    brand[48] = 0;
    printf("Brand  : %s\n", brand);

    /* ---- leaf 1: 基础向量/浮点 ---- */
    cpuid(1, 0, regs);
    unsigned f1_ecx = regs[2];
    unsigned f1_edx = regs[3];

    /* ---- leaf 7 sub0: AVX2 / AVX-512 系列 ---- */
    unsigned l7[4];
    cpuid(7, 0, l7);
    unsigned ebx7 = l7[1];
    unsigned edx7 = l7[3];

    /* ---- OS 向量状态使能(XCR0) ---- */
    unsigned long long xcr0 = get_xcr0();
    int os_avx  = (xcr0 & (1ULL << 2)) != 0;            /* YMM 状态 */
    int os_avx512 = (xcr0 & ((1ULL<<5)|(1ULL<<6)|(1ULL<<7))) == ((1ULL<<5)|(1ULL<<6)|(1ULL<<7));

    printf("\n========== 向量 / 超越函数相关特性 ==========\n");
    printf("[leaf 1]\n");
    printf("  SSE      : %s  (EDX bit 25)\n", f1_edx & (1u<<25) ? "YES" : "no");
    printf("  FMA      : %s  (ECX bit 12)\n", f1_ecx & (1u<<12) ? "YES" : "no");
    printf("  AVX      : %s  (ECX bit 28)   OS-YMM 使能: %s\n",
           f1_ecx & (1u<<28) ? "YES" : "no", os_avx ? "yes" : "NO");

    printf("[leaf 7]\n");
    printf("  AVX2        : %s  (EBX bit 5)\n",  ebx7 & (1u<<5)  ? "YES" : "no");
    printf("  AVX-512F    : %s  (EBX bit 16)  OS-ZMM 使能: %s\n",
           ebx7 & (1u<<16) ? "YES" : "no", os_avx512 ? "yes" : "NO");
    printf("  AVX-512ER   : %s  (EBX bit 27)  <-- 含 VEXP2PS 原生 SIMD 指数(仅 Xeon Phi)\n",
           ebx7 & (1u<<27) ? "YES" : "no");
    printf("  AVX-512-FP16: %s  (EDX bit 23)  <-- 含 VEXP2PH/VLOG2PH(仅 fp16)\n",
           edx7 & (1u<<23) ? "YES" : "no");

    /* ---- 结论 ---- */
    printf("\n========== 对「批量 fp32 softmax」的意义 ==========\n");
    if (ebx7 & (1u<<27)) {
        printf("  [本机] 支持 AVX-512ER -> 有原生 SIMD 指数 VEXP2PS(2^x, ~23-bit 精度)。\n");
        printf("         这是 Xeon Phi 等加速器专属；若目标机是它，可考虑用 VEXP2PS 替代多项式。\n");
    } else {
        printf("  [本机] 无 AVX-512ER -> 没有原生 SIMD fp32 指数指令。\n");
        printf("         当前手写 AVX2 多项式近似(误差 ~1e-7)就是主流 CPU 上的最优解。\n");
    }
    if (edx7 & (1u<<23)) {
        printf("  [本机] 支持 AVX-512-FP16 -> 有 VEXP2PH(仅 fp16)。\n");
        printf("         若把 softmax 改算 fp16 可受益；fp32 仍需下转/上转，通常不划算。\n");
    }
    printf("  x87 标量 FSIN/F2XM1 在所有 x86 都存在，但标量、80-bit、延迟 50-100+ 周期，\n");
    printf("  不适合批量 SIMD；SIMD 的 sin/cos 在 x86 上根本不存在。\n");
    return 0;
}

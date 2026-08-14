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

#ifndef BENCH_UTIL_H
#define BENCH_UTIL_H

/* Windows / MinGW-w64 微基准测试工具集
 * - QPC 墙钟 + TSC 周期计数（含 TSC 频率标定）
 * - P-core 绑定（Alder Lake 混合架构必需）
 * - 防编译器消除
 * - 尾部保护页分配，用于越界读检测
 */

#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <cpuid.h>

/* ------------------------------------------------------------------ */
/* 防优化                                                              */
/* ------------------------------------------------------------------ */

/* 让编译器认为浮点值被使用了，且内存可能被改动 */
#define BU_CONSUME_F(f) __asm__ __volatile__("" :: "x"(f) : "memory")
/* 打断指针的公共子表达式消除，阻止把循环内调用提出去 */
#define BU_ESCAPE_PTR(p) __asm__ __volatile__("" : "+r"(p) :: "memory")
#define BU_BARRIER() __asm__ __volatile__("" ::: "memory")

/* ------------------------------------------------------------------ */
/* 计时                                                                */
/* ------------------------------------------------------------------ */

static double bu_qpc_freq = 0.0;   /* QPC ticks / 秒，通常为 1e7 */
static double bu_tsc_freq = 0.0;   /* TSC ticks / 秒 = 实际有效频率 */

static inline int64_t bu_qpc(void)
{
    LARGE_INTEGER v;
    QueryPerformanceCounter(&v);
    return (int64_t)v.QuadPart;
}

static inline uint64_t bu_tsc(void)
{
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}

static inline double bu_qpc_to_sec(int64_t ticks)
{
    return (double)ticks / bu_qpc_freq;
}

/* 用忙等标定 TSC 频率。QPC 频率固定 10MHz，不等于 CPU 频率，
 * 必须实测 TSC 才能把耗时换算成 cycles/element。 */
static void bu_calibrate(void)
{
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    bu_qpc_freq = (double)f.QuadPart;

    /* 预热，让 CPU 升频 */
    volatile double warm = 0.0;
    int64_t w0 = bu_qpc();
    while (bu_qpc_to_sec(bu_qpc() - w0) < 0.10) warm += 1.0;
    (void)warm;

    /* 取三轮中位数，每轮 60ms */
    double samples[3];
    for (int r = 0; r < 3; r++) {
        int64_t q0 = bu_qpc();
        uint64_t t0 = bu_tsc();
        volatile double spin = 0.0;
        while (bu_qpc_to_sec(bu_qpc() - q0) < 0.060) spin += 1.0;
        uint64_t t1 = bu_tsc();
        int64_t q1 = bu_qpc();
        (void)spin;
        samples[r] = (double)(t1 - t0) / bu_qpc_to_sec(q1 - q0);
    }
    /* 三值中位数 */
    double a = samples[0], b = samples[1], c = samples[2];
    double med = (a > b) ? ((b > c) ? b : ((a > c) ? c : a))
                         : ((a > c) ? a : ((b > c) ? c : b));
    bu_tsc_freq = med;
}

/* ------------------------------------------------------------------ */
/* CPU 拓扑：找出 P-core                                               */
/* ------------------------------------------------------------------ */

/* CPUID leaf 0x1A: EAX[31:24] 为 core type，0x40 = Intel Core (P-core),
 * 0x20 = Intel Atom (E-core)。leaf 不存在则说明是非混合架构。 */
static int bu_current_core_type(void)
{
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned int maxleaf = __get_cpuid_max(0, NULL);
    if (maxleaf < 0x1A) return -1;
    if (!__get_cpuid_count(0x1A, 0, &eax, &ebx, &ecx, &edx)) return -1;
    if (eax == 0) return -1;
    return (int)(eax >> 24);
}

/* 返回选中的逻辑核编号；失败返回 -1 */
static int bu_pin_to_p_core(void)
{
    DWORD_PTR procMask = 0, sysMask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask))
        return -1;

    int nlog = 0;
    for (int i = 0; i < 64; i++) if (procMask & (1ULL << i)) nlog++;

    int chosen = -1;
    int hybrid = 0;

    /* 逐核探测 core type。避开逻辑核 0（中断负载重）。 */
    for (int i = 0; i < 64; i++) {
        if (!(procMask & (1ULL << i))) continue;
        DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), 1ULL << i);
        if (!prev) continue;
        Sleep(0); /* 强制调度到目标核 */
        int t = bu_current_core_type();
        if (t > 0) hybrid = 1;
        if (t == 0x40 && i >= 2 && chosen < 0) chosen = i;
        SetThreadAffinityMask(GetCurrentThread(), prev);
    }

    if (chosen < 0) chosen = (nlog > 2) ? 2 : 0;   /* 非混合架构的兜底 */

    SetThreadAffinityMask(GetCurrentThread(), 1ULL << chosen);
    Sleep(0);

    fprintf(stderr, "[bench] 逻辑核数=%d 混合架构=%s 绑定到核心 #%d (%s)\n",
            nlog, hybrid ? "是" : "否", chosen,
            hybrid ? (bu_current_core_type() == 0x40 ? "P-core" : "E-core") : "n/a");
    return chosen;
}

static void bu_boost_priority(void)
{
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}

/* ------------------------------------------------------------------ */
/* 常规大块内存分配（32 字节对齐）                                     */
/* ------------------------------------------------------------------ */

static float* bu_alloc_floats(size_t n)
{
    if (n == 0) n = 1;
    void* p = _aligned_malloc(n * sizeof(float), 64);
    if (!p) {
        fprintf(stderr, "[bench] 分配 %zu 个 float (%.1f MB) 失败\n",
                n, (double)(n * sizeof(float)) / (1024.0 * 1024.0));
        exit(2);
    }
    return (float*)p;
}

static void bu_free_floats(float* p) { if (p) _aligned_free(p); }

/* ------------------------------------------------------------------ */
/* 尾部保护页缓冲区：数组末尾紧贴 PAGE_NOACCESS 页                      */
/* 任何越界读立即触发访问违例 -> 进程崩溃 -> 由父进程按退出码判定       */
/* ------------------------------------------------------------------ */

typedef struct {
    void*  base;        /* VirtualAlloc 返回的基址 */
    size_t reserve;     /* 保留的总字节数 */
    float* data;        /* 对外可用的数组首地址 */
    size_t n;
} GuardedBuf;

static int bu_guarded_alloc(GuardedBuf* g, size_t n)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    size_t page = si.dwPageSize;

    size_t bytes = n * sizeof(float);
    if (bytes == 0) bytes = 1;
    size_t datapages = (bytes + page - 1) / page;
    size_t reserve = (datapages + 1) * page;

    void* base = VirtualAlloc(NULL, reserve, MEM_RESERVE, PAGE_NOACCESS);
    if (!base) return 0;

    /* 只提交数据区；最后一页保持未提交 -> 访问即违例 */
    if (!VirtualAlloc(base, datapages * page, MEM_COMMIT, PAGE_READWRITE)) {
        VirtualFree(base, 0, MEM_RELEASE);
        return 0;
    }

    g->base = base;
    g->reserve = reserve;
    g->n = n;
    /* 让数组末尾正好落在保护页边界上 */
    g->data = (float*)((char*)base + datapages * page - bytes);
    return 1;
}

static void bu_guarded_free(GuardedBuf* g)
{
    if (g && g->base) {
        VirtualFree(g->base, 0, MEM_RELEASE);
        g->base = NULL;
        g->data = NULL;
    }
}

/* ------------------------------------------------------------------ */
/* 统计小工具                                                          */
/* ------------------------------------------------------------------ */

static int bu_cmp_double(const void* a, const void* b)
{
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static double bu_median(double* v, int n)
{
    qsort(v, n, sizeof(double), bu_cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

#endif /* BENCH_UTIL_H */

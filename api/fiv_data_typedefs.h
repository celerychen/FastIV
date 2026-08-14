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

#ifndef FIV_DATA_TYPEDEFS_H
#define FIV_DATA_TYPEDEFS_H

// ===================== Basic Integer Types (All Platforms & Architectures) =====================
typedef unsigned char      iv8u;
typedef signed   char      iv8s;
typedef unsigned short     iv16u;
typedef signed   short     iv16s;
typedef unsigned int       iv32u;
typedef signed   int       iv32s;
typedef unsigned long long iv64u;
typedef signed   long long iv64s;

// ===================== Standard Floating-Point Types (Full Platform Compatibility) =====================
typedef float  ivf32;
typedef double ivf64;

// ===================== 16-bit Floating-Point Types =====================

// --- IEEE 754 half-precision (FP16) ---
// relaxed mode: requires the compiler to support a 16-bit float type
#if (defined(_MSC_VER) && defined(_M_ARM64)) || \
    defined(__aarch64__) || \
    (defined(__arm__) && defined(__ARM_NEON__))
// ARM: use __fp16
typedef __fp16  ivf16;
#define FIV_HAS_FP16_TYPE 1

#elif (defined(__GNUC__) && __GNUC__ >= 7) || \
      (defined(__clang__) && __clang_major__ >= 6) || \
      defined(__AVX512FP16__)
// x86/x64: use C11 standard type _Float16
typedef _Float16 ivf16;
#define FIV_HAS_FP16_TYPE 1

#else
#define FIV_HAS_FP16_TYPE 0
typedef iv16u   ivf16;   // Fallback: store as uint16
#endif

// --- Brain Float 16 (BF16) ---
// relaxed mode: requires the compiler to support the __bf16 type
// note: MSVC (any architecture) does not support __bf16
#if defined(__aarch64__) || \
    (defined(__arm__) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
// ARM: use __bf16 (ARMv8.6+ / ARMv9)
typedef __bf16  ivbf16;
#define FIV_HAS_BF16_TYPE 1

#elif (defined(__GNUC__) && __GNUC__ >= 13) || \
      (defined(__clang__) && __clang_major__ >= 15) || \
      defined(__AVX512BF16__)
// x86/x64: use __bf16 (GCC 13+ / Clang 15+ / AVX512-BF16)
typedef __bf16  ivbf16;
#define FIV_HAS_BF16_TYPE 1

#else
#define FIV_HAS_BF16_TYPE 0
typedef iv16u   ivbf16;   // Fallback: store as uint16
#endif

// ===================== SIMD Instruction Set Headers & Detection (Cross-Platform) =====================
// Auto-include headers + Detect SSE/AVX/AVX2/AVX512/NEON/ARMv8/ARMv9/SVE/SME
#undef FIV_USE_X86_SIMD
#undef FIV_USE_ARM_NEON
#undef FIV_USE_AVX
#undef FIV_USE_AVX2
#undef FIV_USE_AVX512

// ARM Advanced Instruction Set Macros
#undef FIV_USE_ARM_NEON_FP16
#undef FIV_USE_ARM_NEON_BF16
#undef FIV_USE_ARM_I8MM
#undef FIV_USE_ARM_SVE
#undef FIV_USE_ARM_SVE2
#undef FIV_USE_ARM_SME
#undef FIV_USE_ARM_SME2

// ===================== x86/x86_64: SSE / AVX / AVX2 / AVX512 =====================
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define FIV_USE_X86_SIMD 1

// Include unified SIMD headers
#if defined(__GNUC__) || defined(__clang__)
#include <x86intrin.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#endif

// Compile-time detection: AVX (256bit)
// Fixed: MSVC requires enabling AVX in project settings to define _M_AVX
#if defined(__AVX__) || defined(_M_AVX)
#define FIV_USE_AVX 1
#endif

// Compile-time detection: AVX2 (256bit Enhanced)
// Fixed: MSVC requires enabling AVX2 in project settings to define _M_AVX2
#if defined(__AVX2__) || defined(_M_AVX2)
#define FIV_USE_AVX2 1
#endif

// Compile-time detection: AVX512 (512bit)
#if defined(__AVX512F__) || defined(_M_AVX512F)
#define FIV_USE_AVX512 1
#endif

// ===================== ARM / ARM64: NEON / ARMv8 / ARMv9 / SVE / SME (Latest Instructions) =====================
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
#define FIV_USE_ARM_NEON 1
#include <arm_neon.h>

// ===================== ARM Architecture Version Detection =====================
// ARMv8 / ARMv8.x Extended Instruction Sets
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define FIV_USE_ARM_NEON_FP16 1   // ARMv8.2+ NEON FP16 Vector Arithmetic
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#define FIV_USE_ARM_NEON_BF16 1   // ARMv8.6+ NEON BF16 Vector Arithmetic
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
#define FIV_USE_ARM_I8MM 1         // ARMv8.5+ 8-bit Integer Matrix Multiply (AI Accelerate)
#endif

// ===================== ARMv9 Latest Extensions =====================
#if defined(__ARM_FEATURE_SVE)
#define FIV_USE_ARM_SVE 1         // ARMv8.2+ SVE Scalable Vector Extension
#include <arm_sve.h>
#endif

#if defined(__ARM_FEATURE_SVE2)
#define FIV_USE_ARM_SVE2 1        // ARMv9+ SVE2 Enhanced Scalable Vector
#endif

#if defined(__ARM_FEATURE_SME)
#define FIV_USE_ARM_SME 1         // ARMv9+ SME Scalable Matrix Extension
#include <arm_sme.h>
#endif

#if defined(__ARM_FEATURE_SME2)
#define FIV_USE_ARM_SME2 1        // ARMv9.2+ SME2 Latest Matrix Extension
#endif

#endif



#if defined(_MSC_VER)
#define FIV_ALIGNED(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
#define FIV_ALIGNED(n) __attribute__((aligned(n)))
#else
#define FIV_ALIGNED(n)
#endif

#include <stdint.h>

/* Align pointer p up to the align boundary (align must be a power of two)
   usage: p = (iv8u*)FIV_PTR_ALIGN(p, FIV_STRIDE_ALIGN);
*/
#define FIV_PTR_ALIGN(p, align) \
    (void*)(((uintptr_t)(p) + ((uintptr_t)(align) - 1)) & ~((uintptr_t)(align) - 1))


#if defined(FIV_USE_X86_SIMD)
    #if defined(_WIN64) || defined(__x86_64__) || defined(_M_X64)
        #define FIV_STRIDE_ALIGN    32
        #define FIV_DALIGNED        FIV_ALIGNED(FIV_STRIDE_ALIGN)
    #elif defined(WIN32) || defined(__i386__) || defined(_M_IX86)
        #define FIV_STRIDE_ALIGN    16
        #define FIV_DALIGNED        FIV_ALIGNED(FIV_STRIDE_ALIGN)
        #define FIV_SSE_OPTED
    #else
        #define FIV_DALIGNED
    #endif
#elif defined(FIV_USE_ARM_NEON)
    #define FIV_STRIDE_ALIGN    16
    #define FIV_DALIGNED        FIV_ALIGNED(FIV_STRIDE_ALIGN)
#else
    #define FIV_DALIGNED            
#endif

/* Fallback: unknown platforms (or the x86 branch that is neither x64 nor i386)
   do not define FIV_STRIDE_ALIGN, which fiv_malloc depends on; without it the
   build fails outright.
*/
#ifndef FIV_STRIDE_ALIGN
    #define FIV_STRIDE_ALIGN    16
#endif




typedef enum {
    FIV_RET_OK,
    FIV_RET_ERR_PARA,
    FIV_RET_ERR_MEM,
    FIV_RET_ERR_NOT_SUPPORT = 0x04,
    FIV_RET_ERR_OPEN_FILE = 0x08,
    FIV_RET_ERR_DATA_UNINITED = 0x10,
    FIV_RET_ERR_UNKNOWN = 0x20,
    FIV_RET_ERR_END_OF_FILE = 0x40,
    FIV_RET_ERR_DATA_WAITING = 0x80,
    FIV_RET_ERR_DATA_NOT_ENOUGH = 0x100,
    FIV_RET_DATA_ALREADY_EXISTS = 0x101,
    FIV_RET_DATA_NOT_FOUND = 0x102,
    FIV_RET_THREAD_LOCK_FAIL = 0x104
}fiv_ret;




typedef enum {
    FIV_8U1 = 0,
    FIV_8S1,
    FIV_16U1,
    FIV_16S1,
    FIV_32U1,
    FIV_32S1,
    FIV_64U1,
    FIV_64S1,
    FIV_32F1,
    FIV_64F1,
    FIV_16F1,
	FIV_16BF1,
	FIV_UNDEF12,
	FIV_UNDEF13,
	FIV_UNDEF14,
	FIV_UNDEF15,
	
	FIV_8U2 = 16,
    FIV_8S2,
    FIV_16U2,
    FIV_16S2,
    FIV_32U2,
    FIV_32S2,
    FIV_64U2,
    FIV_64S2,
    FIV_32F2,
    FIV_64F2,
    FIV_16F2,
	FIV_16BF2,
	FIV_UNDEF28,
	FIV_UNDEF29,
	FIV_UNDEF30,
	FIV_UNDEF31,
	
	FIV_8U3 = 32,
    FIV_8S3,
    FIV_16U3,
    FIV_16S3,
    FIV_32U3,
    FIV_32S3,
    FIV_64U3,
    FIV_64S3,
    FIV_32F3,
    FIV_64F3,
    FIV_16F3,
	FIV_16BF3,
	FIV_UNDEF44,
	FIV_UNDEF45,
	FIV_UNDEF46,
	FIV_UNDEF47,
	
	FIV_8U4 = 48,
    FIV_8S4,
    FIV_16U4,
    FIV_16S4,
    FIV_32U4,
    FIV_32S4,
    FIV_64U4,
    FIV_64S4,
    FIV_32F4,
    FIV_64F4,
    FIV_16F4,
	FIV_16BF4,
	FIV_UNDEF60,
	FIV_UNDEF61,
	FIV_UNDEF62,
	FIV_UNDEF63,	
	
	
	FIV_8U8 = 64,
    FIV_8S8,
    FIV_16U8,
    FIV_16S8,
    FIV_32U8,
    FIV_32S8,
    FIV_64U8,
    FIV_64S8,
    FIV_32F8,
    FIV_64F8,
    FIV_16F8,
	FIV_16BF8,
	FIV_UNDEF76,
	FIV_UNDEF77,
	FIV_UNDEF78,
	FIV_UNDEF79,	
	
	
	FIV_8U16 = 80,
    FIV_8S16,
    FIV_16U16,
    FIV_16S16,
    FIV_32U16,
    FIV_32S16,
    FIV_64U16,
    FIV_64S16,
    FIV_32F16,
    FIV_64F16,
    FIV_16F16,
	FIV_16BF16,


}fiv_data_type;














#endif // FAI_TYPE_DEF_H
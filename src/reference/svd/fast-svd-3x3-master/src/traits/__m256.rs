use std::arch::x86_64::{__m256, _mm256_add_ps, _mm256_and_ps, _mm256_blendv_ps, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_rsqrt_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_sub_ps, _mm256_xor_ps};

use super::SVDCompatible;

impl SVDCompatible for __m256 {
    type Mask = __m256;
    #[inline(always)]
    fn default() -> Self {
        unsafe { _mm256_setzero_ps() }
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        unsafe { _mm256_add_ps(*self, *other) }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        unsafe { _mm256_sub_ps(*self, *other) }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        unsafe { _mm256_mul_ps(*self, *other) }
    }

    #[inline(always)]
    fn max(&self, other: &Self) -> Self {
        unsafe { _mm256_max_ps(*self, *other) }
    }

    #[inline(always)]
    fn and(&self, other: &Self) -> Self {
        unsafe { _mm256_and_ps(*self, *other) }
    }

    #[inline(always)]
    fn xor(&self, other: &Self) -> Self {
        unsafe { _mm256_xor_ps(*self, *other) }
    }

    #[inline(always)]
    fn cmpge(&self, other: &Self) -> Self {
        unsafe { _mm256_cmp_ps::<13>(*self, *other) }
    }

    #[inline(always)]
    fn cmple(&self, other: &Self) -> Self {
        unsafe { _mm256_cmp_ps::<2>(*self, *other) }
    }

    #[inline(always)]
    fn cmplt(&self, other: &Self) -> Self {
        unsafe { _mm256_cmp_ps::<1>(*self, *other) }
    }

    #[inline(always)]
    fn splat(value: f32) -> Self {
        unsafe { _mm256_set1_ps(value) }
    }

    #[inline(always)]
    fn rsqrt(&self) -> Self {
        unsafe { _mm256_rsqrt_ps(*self) }
    }

    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn blend(mask: &Self::Mask, other1: &Self, other2: &Self) -> Self {
        unsafe { _mm256_blendv_ps(*other1, *other2, *mask) }
    }

    #[inline(always)]
    fn maskz(mask: &Self::Mask, other: &Self) -> Self {
        unsafe { _mm256_and_ps(*mask, *other) }
    }

    #[inline(always)]
    fn ones() -> Self {
        unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFFu32 as i32)) }
    }
}
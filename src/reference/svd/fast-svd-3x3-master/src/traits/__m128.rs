use std::arch::x86_64::{__m128, _mm_add_ps, _mm_and_ps, _mm_blendv_ps, _mm_castsi128_ps, _mm_cmp_ps, _mm_max_ps, _mm_mul_ps, _mm_rsqrt_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_sub_ps, _mm_xor_ps};

use super::SVDCompatible;

impl SVDCompatible for __m128 {
    type Mask = __m128;

    #[inline(always)]
    fn default() -> Self {
        unsafe { _mm_setzero_ps() }
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        unsafe { _mm_add_ps(*self, *other) }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        unsafe { _mm_sub_ps(*self, *other) }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        unsafe { _mm_mul_ps(*self, *other) }
    }

    #[inline(always)]
    fn max(&self, other: &Self) -> Self {
        unsafe { _mm_max_ps(*self, *other) }
    }

    #[inline(always)]
    fn and(&self, other: &Self) -> Self {
        unsafe { _mm_and_ps(*self, *other) }
    }

    #[inline(always)]
    fn xor(&self, other: &Self) -> Self {
        unsafe { _mm_xor_ps(*self, *other) }
    }

    #[inline(always)]
    fn cmpge(&self, other: &Self) -> Self {
        unsafe { _mm_cmp_ps::<13>(*self, *other) }
    }

    #[inline(always)]
    fn cmple(&self, other: &Self) -> Self {
        unsafe { _mm_cmp_ps::<2>(*self, *other) }
    }

    #[inline(always)]
    fn cmplt(&self, other: &Self) -> Self {
        unsafe { _mm_cmp_ps::<1>(*self, *other) }
    }

    #[inline(always)]
    fn splat(value: f32) -> Self {
        unsafe { _mm_set1_ps(value) }
    }

    #[inline(always)]
    fn rsqrt(&self) -> Self {
        unsafe { _mm_rsqrt_ps(*self) }
    }

    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn blend(mask: &Self::Mask, other1: &Self, other2: &Self) -> Self {
        unsafe { _mm_blendv_ps(*other1, *other2, *mask) }
    }

    #[inline(always)]
    fn maskz(mask: &Self::Mask, other: &Self) -> Self {
        unsafe { _mm_and_ps(*mask, *other) }
    }

    #[inline(always)]
    fn ones() -> Self {
        unsafe { _mm_castsi128_ps(_mm_set1_epi32(0xFFFFFFFFu32 as i32)) }
    }
}

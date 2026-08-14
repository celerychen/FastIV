
use std::arch::x86_64::{__m512, __mmask16, _mm512_add_ps, _mm512_and_ps, _mm512_castsi512_ps, _mm512_cmp_ps_mask, _mm512_mask_blend_ps, _mm512_maskz_or_ps, _mm512_max_ps, _mm512_mul_ps, _mm512_rsqrt14_ps, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_ps, _mm512_sub_ps, _mm512_xor_ps};

use super::SVDCompatible;

impl SVDCompatible for __m512 {
    type Mask = __mmask16;
    #[inline(always)]
    fn default() -> Self {
        unsafe { _mm512_setzero_ps() }
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        unsafe { _mm512_add_ps(*self, *other) }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        unsafe { _mm512_sub_ps(*self, *other) }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        unsafe { _mm512_mul_ps(*self, *other) }
    }

    #[inline(always)]
    fn max(&self, other: &Self) -> Self {
        unsafe { _mm512_max_ps(*self, *other) }
    }

    #[inline(always)]
    fn and(&self, other: &Self) -> Self {
        unsafe { _mm512_and_ps(*self, *other) }
    }

    #[inline(always)]
    fn xor(&self, other: &Self) -> Self {
        unsafe { _mm512_xor_ps(*self, *other) }
    }

    #[inline(always)]
    fn cmpge(&self, other: &Self) -> Self::Mask {
        unsafe { _mm512_cmp_ps_mask::<13>(*self, *other) }
    }

    #[inline(always)]
    fn cmple(&self, other: &Self) -> Self::Mask {
        unsafe { _mm512_cmp_ps_mask::<2>(*self, *other) }
    }

    #[inline(always)]
    fn cmplt(&self, other: &Self) -> Self::Mask {
        unsafe { _mm512_cmp_ps_mask::<1>(*self, *other) }
    }

    #[inline(always)]
    fn splat(value: f32) -> Self {
        unsafe { _mm512_set1_ps(value) }
    }

    #[inline(always)]
    fn rsqrt(&self) -> Self {
        unsafe { _mm512_rsqrt14_ps(*self) }
    }

    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn blend(mask: &Self::Mask, other1: &Self, other2: &Self) -> Self {
        unsafe { _mm512_mask_blend_ps(*mask, *other1, *other2) }
    }

    #[inline(always)]
    fn maskz(mask: &Self::Mask, other: &Self) -> Self {
        unsafe { _mm512_maskz_or_ps(*mask, *other, *other) }
    }

    #[inline(always)]
    fn ones() -> Self {
        unsafe { _mm512_castsi512_ps(_mm512_set1_epi32(0xFFFFFFFFu32 as i32)) }
    }
}
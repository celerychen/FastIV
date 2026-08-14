use std::simd::{cmp::{SimdPartialEq, SimdPartialOrd}, f32x16, num::SimdFloat, LaneCount, Mask, Simd, StdFloat, SupportedLaneCount};

use super::SVDCompatible;

impl<const N: usize> SVDCompatible for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount {
    type Mask = <Simd<f32, N> as SimdPartialEq>::Mask;
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        *self + *other
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        *self * *other
    }

    #[inline(always)]
    fn max(&self, other: &Self) -> Self {
        self.simd_max(*other)
    }

    #[inline(always)]
    fn and(&self, other: &Self) -> Self {
        Self::from_bits(Self::to_bits(*self) & Self::to_bits(*other))
    }

    #[inline(always)]
    fn xor(&self, other: &Self) -> Self {
        Self::from_bits(Self::to_bits(*self) ^ Self::to_bits(*other))
    }

    #[inline(always)]
    fn cmpge(&self, other: &Self) -> Self::Mask {
        self.simd_ge(*other)
    }

    #[inline(always)]
    fn cmple(&self, other: &Self) -> Self::Mask {
        self.simd_le(*other)
    }

    #[inline(always)]
    fn cmplt(&self, other: &Self) -> Self::Mask {
        self.simd_lt(*other)
    }

    #[inline(always)]
    fn splat(value: f32) -> Self {
        Self::splat(value)
    }

    #[inline(always)]
    fn rsqrt(&self) -> Self {
        self.sqrt().recip()
    }

    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn blend(mask: &Self::Mask, other_false: &Self, other_true: &Self) -> Self {
        mask.select(*other_true, *other_false)
    }

    #[inline(always)]
    fn maskz(mask: &Self::Mask, other: &Self) -> Self {
        mask.select(*other, Self::splat(0.0))
    }

    #[inline(always)]
    fn ones() -> Self {
        Self::splat(f32::from_bits(0xFFFFFFFF))
    }
}
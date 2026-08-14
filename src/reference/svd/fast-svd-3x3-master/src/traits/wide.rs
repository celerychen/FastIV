use super::SVDCompatible;
use std::ops::{BitAnd, BitXor};
use wide::{CmpGe, CmpLe, CmpLt, f32x8};

impl SVDCompatible for f32x8 {
    type Mask = f32x8;
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
        Self::max(*self, *other)
    }

    #[inline(always)]
    fn and(&self, other: &Self) -> Self {
        self.bitand(other)
    }

    #[inline(always)]
    fn xor(&self, other: &Self) -> Self {
        self.bitxor(other)
    }

    #[inline(always)]
    fn cmpge(&self, other: &Self) -> Self::Mask {
        self.cmp_ge(*other)
    }

    #[inline(always)]
    fn cmple(&self, other: &Self) -> Self::Mask {
        self.cmp_le(*other)
    }

    #[inline(always)]
    fn cmplt(&self, other: &Self) -> Self::Mask {
        self.cmp_lt(*other)
    }

    #[inline(always)]
    fn splat(value: f32) -> Self {
        Self::splat(value)
    }

    #[inline(always)]
    fn rsqrt(&self) -> Self {
        self.recip_sqrt()
    }

    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn blend(mask: &Self::Mask, other_false: &Self, other_true: &Self) -> Self {
        mask.blend(*other_true, *other_false)
    }

    #[inline(always)]
    fn maskz(mask: &Self::Mask, other: &Self) -> Self {
        mask.blend(*other, Self::splat(0.0))
    }

    #[inline(always)]
    fn ones() -> Self {
        Self::splat(f32::from_bits(0xFFFFFFFF))
    }
}

use super::SVDCompatible;

impl SVDCompatible for f32 {
    type Mask = f32;
    #[inline(always)]
    fn default() -> Self {
        0.0
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
        f32::max(*self, *other)
    }

    #[inline(always)]
    fn and(&self, other: &Self) -> Self {
        f32::from_bits(f32::to_bits(*self) & f32::to_bits(*other))
    }

    #[inline(always)]
    fn xor(&self, other: &Self) -> Self {
        f32::from_bits(f32::to_bits(*self) ^ f32::to_bits(*other))
    }

    #[inline(always)]
    fn cmpge(&self, other: &Self) -> Self {
        if self >= other { f32::from_bits(0xFFFFFFFF) } else { f32::from_bits(0) }
    }

    #[inline(always)]
    fn cmple(&self, other: &Self) -> Self {
        if self <= other { f32::from_bits(0xFFFFFFFF) } else { f32::from_bits(0) }
    }

    #[inline(always)]
    fn cmplt(&self, other: &Self) -> Self {
        if self < other { f32::from_bits(0xFFFFFFFF) } else { f32::from_bits(0) }
    }

    #[inline(always)]
    fn splat(value: f32) -> Self {
        value
    }

    #[inline(always)]
    fn rsqrt(&self) -> Self {
        1.0 / self.sqrt()
    }

    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn blend(mask: &f32, other_false: &Self, other_true: &Self) -> Self {
        if mask.to_bits() != 0 { *other_true } else { *other_false }
    }

    #[inline(always)]
    fn maskz(mask: &f32, other: &Self) -> Self {
        if mask.to_bits() != 0 { *other } else { 0.0 }
    }

    #[inline(always)]
    fn ones() -> Self {
        f32::from_bits(0xFFFFFFFF)
    }
}
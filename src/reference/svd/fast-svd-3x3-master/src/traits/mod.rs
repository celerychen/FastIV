
mod f32;
#[cfg(feature = "sse")]
mod __m128;
#[cfg(feature = "avx")]
mod __m256;
#[cfg(feature = "avx512")]
mod __m512;
#[cfg(feature = "portable_simd")]
mod portable_simd;
#[cfg(feature = "wide")]
mod wide;

pub trait SVDCompatible {
    type Mask;
    fn default() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn max(&self, other: &Self) -> Self;
    fn and(&self, other: &Self) -> Self;
    fn xor(&self, other: &Self) -> Self;
    fn cmpge(&self, other: &Self) -> Self::Mask;
    fn cmple(&self, other: &Self) -> Self::Mask;
    fn cmplt(&self, other: &Self) -> Self::Mask;
    fn splat(value: f32) -> Self;
    fn rsqrt(&self) -> Self;
    fn clone(&self) -> Self;
    fn blend(mask: &Self::Mask, other_false: &Self, other_true: &Self) -> Self;
    fn maskz(mask: &Self::Mask, other: &Self) -> Self; // Set to zero if mask is 0
    fn ones() -> Self; // Return data that all bits set to 1
}

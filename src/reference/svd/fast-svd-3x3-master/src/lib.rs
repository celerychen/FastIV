#![cfg_attr(feature = "portable_simd", feature(portable_simd))]
#![cfg_attr(feature = "avx512", feature(stdarch_x86_avx512))]

mod jacobi_conjugation;
mod givens_qr_factorization;
mod svd;
mod utils;
pub mod traits;
pub use svd::{svd};
pub use utils::{svd_mat};

#[cfg(feature = "wide")]
pub use wide::f32x8;

// Tests
#[cfg(test)]
mod tests;
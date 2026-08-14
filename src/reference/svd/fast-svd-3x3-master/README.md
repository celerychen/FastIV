# fast-svd-3x3
Quick Singular Value Decomposition for 3x3 matrix. SIMD Supported.
Direct port of [Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations](https://pages.cs.wisc.edu/~sifakis/project_pages/svd.html) by A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis.

## Notes
- This SVD method returns U, V as rotation matrix. (Determinants of these matricies are non-negative.)
- Because of this, singular values **can be negative**.
- Singular values are ordered in descending order.

## Features
- Get SVD (U, V, Singular values)
- Get Singular values only
- Get U/V as Quaternion
- SIMD
- Works for any types implementing `SVDCompatible` trait
- Supports [wide](https://docs.rs/wide/latest/wide/) crate f32x8

## Installation
```bash
cargo add fast-svd-3x3
```
### SIMD
```toml
[dependencies]
fast-svd-3x3 = { features = ["sse", "avx", "avx512", "portable_simd", "wide"] }
```
To get preset trait for SIMD types, you should add features. (Note that avx512 currently does not work for many cpus)

```bash
RUSTFLAGS="-C target-cpu=native"
```
Note that you should set rustc flags to use SIMD.

## Basic Usage (Example)
### Scalar (f32)
```rs
use std::ops::Range;

use fast_svd_3x3::svd_mat;
use rand::Rng;

const RANGE: Range<f32> = 0.0..10.0;

fn main() {
    let mut rng = rand::rng();
    let mut a11 = rng.random_range(RANGE); let mut a12 = rng.random_range(RANGE); let mut a13 = rng.random_range(RANGE);
    let mut a21 = rng.random_range(RANGE); let mut a22 = rng.random_range(RANGE); let mut a23 = rng.random_range(RANGE);
    let mut a31 = rng.random_range(RANGE); let mut a32 = rng.random_range(RANGE); let mut a33 = rng.random_range(RANGE);
    let ori_a11 = a11; let ori_a12 = a12; let ori_a13 = a13;
    let ori_a21 = a21; let ori_a22 = a22; let ori_a23 = a23;
    let ori_a31 = a31; let ori_a32 = a32; let ori_a33 = a33;

    let mut u11 = 0.0; let mut  u12 = 0.0; let mut u13 = 0.0;
    let mut u21 = 0.0; let mut  u22 = 0.0; let mut u23 = 0.0;
    let mut u31 = 0.0; let mut  u32 = 0.0; let mut u33 = 0.0;
    let mut v11 = 0.0; let mut  v12 = 0.0; let mut v13 = 0.0;
    let mut v21 = 0.0; let mut  v22 = 0.0; let mut v23 = 0.0;
    let mut v31 = 0.0; let mut  v32 = 0.0; let mut v33 = 0.0;

    svd_mat(
        &mut a11, &mut a12, &mut a13,
        &mut a21, &mut a22, &mut a23,
        &mut a31, &mut a32, &mut a33,
        &mut u11, &mut u12, &mut u13,
        &mut u21, &mut u22, &mut u23,
        &mut u31, &mut u32, &mut u33,
        &mut v11, &mut v12, &mut v13,
        &mut v21, &mut v22, &mut v23,
        &mut v31, &mut v32, &mut v33,
    );

    println!("Original Matrix:");
    println!("{} {} {}", ori_a11, ori_a12, ori_a13);
    println!("{} {} {}", ori_a21, ori_a22, ori_a23);
    println!("{} {} {}", ori_a31, ori_a32, ori_a33);
    println!("U Matrix:");
    println!("{} {} {}", u11, u12, u13);
    println!("{} {} {}", u21, u22, u23);
    println!("{} {} {}", u31, u32, u33);
    println!("Singular Values:");
    println!("{} {} {}", a11, a22, a33);
    println!("V Matrix:");
    println!("{} {} {}", v11, v12, v13);
    println!("{} {} {}", v21, v22, v23);
    println!("{} {} {}", v31, v32, v33);
}
```

### AVX2 (__m256)
```rs

use std::{arch::x86_64::{__m256, _mm256_set_ps, _mm256_setzero_ps}, ops::Range};

use fast_svd_3x3::svd_mat;
use rand::{rngs::ThreadRng, Rng};

const RANGE: Range<f32> = 0.0..10.0;

fn get_random_mm256(rng: &mut ThreadRng, range: Range<f32>) -> __m256 {
unsafe {
    _mm256_set_ps(
        rng.random_range(range.clone()), rng.random_range(range.clone()), rng.random_range(range.clone()), rng.random_range(range.clone()),
        rng.random_range(range.clone()), rng.random_range(range.clone()), rng.random_range(range.clone()), rng.random_range(range.clone())
    )
}
}

fn main() {
unsafe {
    let mut rng = rand::rng();
    let mut a11 = get_random_mm256(&mut rng, 0.0..1.0);
    let mut a12 = get_random_mm256(&mut rng, RANGE);
    let mut a13 = get_random_mm256(&mut rng, RANGE);
    let mut a21 = get_random_mm256(&mut rng, RANGE);
    let mut a22 = get_random_mm256(&mut rng, RANGE);
    let mut a23 = get_random_mm256(&mut rng, RANGE);
    let mut a31 = get_random_mm256(&mut rng, RANGE);
    let mut a32 = get_random_mm256(&mut rng, RANGE);
    let mut a33 = get_random_mm256(&mut rng, RANGE);

    let ori_a11 = a11; let ori_a12 = a12; let ori_a13 = a13;
    let ori_a21 = a21; let ori_a22 = a22; let ori_a23 = a23;
    let ori_a31 = a31; let ori_a32 = a32; let ori_a33 = a33;

    let zero = _mm256_setzero_ps();
    let mut u11 = zero; let mut u12 = zero; let mut u13 = zero;
    let mut u21 = zero; let mut u22 = zero; let mut u23 = zero;
    let mut u31 = zero; let mut u32 = zero; let mut u33 = zero;
    let mut v11 = zero; let mut v12 = zero; let mut v13 = zero;
    let mut v21 = zero; let mut v22 = zero; let mut v23 = zero;
    let mut v31 = zero; let mut v32 = zero; let mut v33 = zero;

    svd_mat(
        &mut a11, &mut a12, &mut a13,
        &mut a21, &mut a22, &mut a23,
        &mut a31, &mut a32, &mut a33,
        &mut u11, &mut u12, &mut u13,
        &mut u21, &mut u22, &mut u23,
        &mut u31, &mut u32, &mut u33,
        &mut v11, &mut v12, &mut v13,
        &mut v21, &mut v22, &mut v23,
        &mut v31, &mut v32, &mut v33,
    );

    println!("Original Matrix:");
    println!("{:?} {:?} {:?}", ori_a11, ori_a12, ori_a13);
    println!("{:?} {:?} {:?}", ori_a21, ori_a22, ori_a23);
    println!("{:?} {:?} {:?}", ori_a31, ori_a32, ori_a33);
    println!("U Matrix:");
    println!("{:?} {:?} {:?}", u11, u12, u13);
    println!("{:?} {:?} {:?}", u21, u22, u23);
    println!("{:?} {:?} {:?}", u31, u32, u33);
    println!("Singular Values:");
    println!("{:?} {:?} {:?}", a11, a22, a33);
    println!("V Matrix:");
    println!("{:?} {:?} {:?}", v11, v12, v13);
    println!("{:?} {:?} {:?}", v21, v22, v23);
    println!("{:?} {:?} {:?}", v31, v32, v33);
}
}
```

### SVD Function with more features
```rs
pub fn svd<VType: SVDCompatible>(
    compute_u_as_matrix: bool,
    compute_v_as_matrix: bool,
    compute_u_as_quaternion: bool,
    compute_v_as_quaternion: bool,
    use_accurate_rsqrt_in_jacobi_conjugation: bool,
    perform_strict_quaternion_renormalization: bool,
    Va11: &mut VType, Va12: &mut VType, Va13: &mut VType,
    Va21: &mut VType, Va22: &mut VType, Va23: &mut VType,
    Va31: &mut VType, Va32: &mut VType, Va33: &mut VType,
    Vu11: &mut VType, Vu12: &mut VType, Vu13: &mut VType,
    Vu21: &mut VType, Vu22: &mut VType, Vu23: &mut VType,
    Vu31: &mut VType, Vu32: &mut VType, Vu33: &mut VType,
    Vv11: &mut VType, Vv12: &mut VType, Vv13: &mut VType,
    Vv21: &mut VType, Vv22: &mut VType, Vv23: &mut VType,
    Vv31: &mut VType, Vv32: &mut VType, Vv33: &mut VType,
    Vqus: &mut VType,
    Vquvx: &mut VType,
    Vquvy: &mut VType,
    Vquvz: &mut VType,
    Vqvs: &mut VType,
    Vqvvx: &mut VType,
    Vqvvy: &mut VType,
    Vqvvz: &mut VType,
)
```
You can also use `svd` function with more features:
- Compute or not compute U as Matrix
- Compute or not compute V as Matrix
- Compute or not compute U as Quaternion
- Compute or not compute V as Quaternion
- Use accurate Rsqrt in jacobi conjugation
- Perform strict quaternion renormalization

Actually, `svd_mat` is specialization of this function to compute U, V as matrix and turn off accurate rsqrt and strict quaternion renormalization.

## Advanced Usage
```rs
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
```
You can implement SVDCompatible trait for your own type. Then `svd` and `svd_mat` will work for that type.

## About Performance
- I checked that performance of trait-based AVX2(__m256) implementation is same as hand-ported(preprocess original source code, convert it to rust) version for AVX2.
- Portable SIMD on f32x8 is slower than AVX2(__m256). Running time: x1.4.
- f32x8 from wide crate is slightly slower than AVX(__m256). Running time: x1.04.
- The function uses input mutable variables as temporal storage several times, not just "copying result to them" once. So it may be good to locate input variables in contiguous memory location for spatial locality.

## Tests
### Test
```bash
cargo test --release
```
### Test (include AVX)
```bash
cargo test --release --features avx
```
### Test (include portable SIMD)
```bash
cargo test --release --features portable_simd
```
### f32 performance test
```bash
cargo test --release -- --ignored --nocapture f32_performance_test
```
### AVX performance test
```bash
cargo test --release --features avx -- --ignored --nocapture __mm256_performance_test
```
### Portable SIMD performance test
```bash
cargo test --release --features portable_simd -- --ignored --nocapture portable_simd_performance_test
```
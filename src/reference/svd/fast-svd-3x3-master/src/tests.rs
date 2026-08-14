use std::{arch::x86_64::{__m256, _mm256_set_ps, _mm256_setzero_ps, _mm256_storeu_ps}, hint::black_box, ops::Range};

#[cfg(feature = "portable_simd")]
use std::{simd::{f32x8, LaneCount, Simd, SupportedLaneCount}};

use rand::{rngs::ThreadRng, Rng};

use crate::svd;

extern crate nalgebra as na;

const TEST_RANGE: Range<f32> = 0.0..10.0;

fn randoms_f32(length: usize, range: Range<f32>, rng: &mut ThreadRng) -> Vec<f32> {
    (0..length).map(|_| rng.random_range(range.clone())).collect()
}

fn randoms_m256(length: usize, range: Range<f32>, rng: &mut ThreadRng) -> Vec<__m256> {
    unsafe {
    (0..length).map(|_| _mm256_set_ps(
        rng.random_range(range.clone()),
        rng.random_range(range.clone()),
        rng.random_range(range.clone()),
        rng.random_range(range.clone()),
        rng.random_range(range.clone()),
        rng.random_range(range.clone()),
        rng.random_range(range.clone()),
        rng.random_range(range.clone()),
    )).collect()
    }
}

#[cfg(feature = "portable_simd")]
fn randoms_portable_simd<const N: usize>(length: usize, range: Range<f32>, rng: &mut ThreadRng) -> Vec<Simd<f32, N>>
where
    LaneCount<N>: SupportedLaneCount,
{
    (0..length).map(|_| {
        let mut arr = [0.0; N];
        for i in 0..N {
            arr[i] = rng.random_range(range.clone());
        }
        return Simd::from_array(arr);
    }).collect()
}

#[cfg(feature = "wide")]
fn randoms_wide(length: usize, range: Range<f32>, rng: &mut ThreadRng) -> Vec<wide::f32x8>
{
    (0..length).map(|_| {
        let mut arr: [f32; 8] = [0.0; 8];
        for i in 0..arr.len() {
            arr[i] = rng.random_range(range.clone());
        }
        return wide::f32x8::from(arr);
    }).collect()
}

const MAT_THRESHOLD: f32 = 2e-1;
const ORTHO_THRESHOLD: f32 = 1e-5;
const QUAT_THRESHOLD: f32 = 1e-5;

fn test_results(original: &na::Matrix3<f32>, u: &na::Matrix3<f32>, s: &na::Matrix3<f32>, v: &na::Matrix3<f32>, qu: &na::UnitQuaternion<f32>, qv: &na::UnitQuaternion<f32>) {
    let reconstructed = u * s * v.transpose();
    let distance_original_reconstructed = (original - reconstructed).norm();
    let u01 = u.column(0).dot(&u.column(1));
    let u02 = u.column(0).dot(&u.column(2));
    let u12 = u.column(1).dot(&u.column(2));
    let v01 = v.column(0).dot(&v.column(1));
    let v02 = v.column(0).dot(&v.column(2));
    let v12 = v.column(1).dot(&v.column(2));
    let u_orthogonality = u01.abs() + u02.abs() + u12.abs();
    let v_orthogonality = v01.abs() + v02.abs() + v12.abs();
    let distance_u_qu = ((na::UnitQuaternion::from_matrix(&u.transpose()) * qu).norm() - 1.0).abs();
    let distance_v_qv = ((na::UnitQuaternion::from_matrix(&v.transpose()) * qv).norm() - 1.0).abs();

    assert_eq!(true, distance_original_reconstructed < MAT_THRESHOLD, "Distance between original and reconstructed matrix is too large: {}", distance_original_reconstructed);
    assert_eq!(true, u_orthogonality < ORTHO_THRESHOLD, "U orthogonality is too large: {}", u_orthogonality);
    assert_eq!(true, v_orthogonality < ORTHO_THRESHOLD, "V orthogonality is too large: {}", v_orthogonality);
    assert_eq!(true, distance_u_qu < QUAT_THRESHOLD, "Distance between U and qu is too large: {}", distance_u_qu);
    assert_eq!(true, distance_v_qv < QUAT_THRESHOLD, "Distance between V and qv is too large: {}", distance_v_qv);
}

const ERROR_TESTS: usize = 100000;

#[test]
fn _f32_error_test() {
    let mut rng = rand::rng();
    let a11v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a12v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a13v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a21v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a22v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a23v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a31v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a32v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a33v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    for i in 0..ERROR_TESTS {
        let mut a11 = a11v[i]; let mut a12 = a12v[i]; let mut a13 = a13v[i];
        let mut a21 = a21v[i]; let mut a22 = a22v[i]; let mut a23 = a23v[i];
        let mut a31 = a31v[i]; let mut a32 = a32v[i]; let mut a33 = a33v[i];
        let mut u11 = 0.0; let mut u12 = 0.0; let mut u13 = 0.0;
        let mut u21 = 0.0; let mut u22 = 0.0; let mut u23 = 0.0;
        let mut u31 = 0.0; let mut u32 = 0.0; let mut u33 = 0.0;
        let mut v11 = 0.0; let mut v12 = 0.0; let mut v13 = 0.0;
        let mut v21 = 0.0; let mut v22 = 0.0; let mut v23 = 0.0;
        let mut v31 = 0.0; let mut v32 = 0.0; let mut v33 = 0.0;
        let mut qus = 0.0;
        let mut quvx = 0.0;
        let mut quvy = 0.0;
        let mut quvz = 0.0;
        let mut qvs = 0.0;
        let mut qvvx = 0.0;
        let mut qvvy = 0.0;
        let mut qvvz = 0.0;

        let original = na::Matrix3::new(
            a11, a12, a13,
            a21, a22, a23,
            a31, a32, a33
        );

        svd(
            true,
            true,
            true,
            true,
            true,
            true,
            &mut a11, &mut a12, &mut a13,
            &mut a21, &mut a22, &mut a23,
            &mut a31, &mut a32, &mut a33,
            &mut u11, &mut u12, &mut u13,
            &mut u21, &mut u22, &mut u23,
            &mut u31, &mut u32, &mut u33,
            &mut v11, &mut v12, &mut v13,
            &mut v21, &mut v22, &mut v23,
            &mut v31, &mut v32, &mut v33,
            &mut qus, &mut quvx, &mut quvy, &mut quvz,
            &mut qvs, &mut qvvx, &mut qvvy, &mut qvvz
        );

        let u = na::Matrix3::new(
            u11, u12, u13,
            u21, u22, u23,
            u31, u32, u33
        );
        let s = na::Matrix3::from_diagonal(&na::Vector3::new(
            a11, a22, a33
        ));
        let v = na::Matrix3::new(
            v11, v12, v13,
            v21, v22, v23,
            v31, v32, v33
        );

        let qu = na::UnitQuaternion::new_normalize(na::Quaternion::new(
            qus, quvx, quvy, quvz
        ));
        let qv = na::UnitQuaternion::new_normalize(na::Quaternion::new(
            qvs, qvvx, qvvy, qvvz
        ));
        test_results(&original, &u, &s, &v, &qu, &qv);
    }
}

#[cfg(feature = "avx")]
#[test]
fn __m256_error_test() {
unsafe {
    let mut rng = rand::rng();
    let a11v = randoms_m256(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a12v = randoms_m256(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a13v = randoms_m256(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a21v = randoms_m256(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a22v = randoms_m256(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a23v = randoms_m256(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a31v = randoms_m256(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a32v = randoms_m256(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a33v = randoms_m256(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    
    for i in 0..ERROR_TESTS {
        let mut a11 = a11v[i]; let mut a12 = a12v[i]; let mut a13 = a13v[i];
        let mut a21 = a21v[i]; let mut a22 = a22v[i]; let mut a23 = a23v[i];
        let mut a31 = a31v[i]; let mut a32 = a32v[i]; let mut a33 = a33v[i];

        let mut a11_cp: [f32; 8] = [0.0; 8];
        let mut a12_cp: [f32; 8] = [0.0; 8];
        let mut a13_cp: [f32; 8] = [0.0; 8];
        let mut a21_cp: [f32; 8] = [0.0; 8];
        let mut a22_cp: [f32; 8] = [0.0; 8];
        let mut a23_cp: [f32; 8] = [0.0; 8];
        let mut a31_cp: [f32; 8] = [0.0; 8];
        let mut a32_cp: [f32; 8] = [0.0; 8];
        let mut a33_cp: [f32; 8] = [0.0; 8];
        _mm256_storeu_ps(a11_cp.as_mut_ptr(), a11);
        _mm256_storeu_ps(a12_cp.as_mut_ptr(), a12);
        _mm256_storeu_ps(a13_cp.as_mut_ptr(), a13);
        _mm256_storeu_ps(a21_cp.as_mut_ptr(), a21);
        _mm256_storeu_ps(a22_cp.as_mut_ptr(), a22);
        _mm256_storeu_ps(a23_cp.as_mut_ptr(), a23);
        _mm256_storeu_ps(a31_cp.as_mut_ptr(), a31);
        _mm256_storeu_ps(a32_cp.as_mut_ptr(), a32);
        _mm256_storeu_ps(a33_cp.as_mut_ptr(), a33);
        let original: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::new(
                a11_cp[i], a12_cp[i], a13_cp[i],
                a21_cp[i], a22_cp[i], a23_cp[i],
                a31_cp[i], a32_cp[i], a33_cp[i]
            )
        }).collect::<Vec<_>>().try_into().unwrap();

        let mut u11 = _mm256_setzero_ps(); let mut u12 = _mm256_setzero_ps(); let mut u13 = _mm256_setzero_ps();
        let mut u21 = _mm256_setzero_ps(); let mut u22 = _mm256_setzero_ps(); let mut u23 = _mm256_setzero_ps();
        let mut u31 = _mm256_setzero_ps(); let mut u32 = _mm256_setzero_ps(); let mut u33 = _mm256_setzero_ps();
        let mut v11 = _mm256_setzero_ps(); let mut v12 = _mm256_setzero_ps(); let mut v13 = _mm256_setzero_ps();
        let mut v21 = _mm256_setzero_ps(); let mut v22 = _mm256_setzero_ps(); let mut v23 = _mm256_setzero_ps();
        let mut v31 = _mm256_setzero_ps(); let mut v32 = _mm256_setzero_ps(); let mut v33 = _mm256_setzero_ps();
        let mut qus = _mm256_setzero_ps();
        let mut quvx = _mm256_setzero_ps();
        let mut quvy = _mm256_setzero_ps();
        let mut quvz = _mm256_setzero_ps();
        let mut qvs = _mm256_setzero_ps();
        let mut qvvx = _mm256_setzero_ps();
        let mut qvvy = _mm256_setzero_ps();
        let mut qvvz = _mm256_setzero_ps();

        svd::svd(
            true,
            true,
            true,
            true,
            true,
            true,
            &mut a11, &mut a12, &mut a13,
            &mut a21, &mut a22, &mut a23,
            &mut a31, &mut a32, &mut a33,
            &mut u11, &mut u12, &mut u13,
            &mut u21, &mut u22, &mut u23,
            &mut u31, &mut u32, &mut u33,
            &mut v11, &mut v12, &mut v13,
            &mut v21, &mut v22, &mut v23,
            &mut v31, &mut v32, &mut v33,
            &mut qus,
            &mut quvx,
            &mut quvy,
            &mut quvz,
            &mut qvs,
            &mut qvvx,
            &mut qvvy,
            &mut qvvz
        );

        _mm256_storeu_ps(a11_cp.as_mut_ptr(), a11);
        _mm256_storeu_ps(a12_cp.as_mut_ptr(), a12);
        _mm256_storeu_ps(a13_cp.as_mut_ptr(), a13);
        _mm256_storeu_ps(a21_cp.as_mut_ptr(), a21);
        _mm256_storeu_ps(a22_cp.as_mut_ptr(), a22);
        _mm256_storeu_ps(a23_cp.as_mut_ptr(), a23);
        _mm256_storeu_ps(a31_cp.as_mut_ptr(), a31);
        _mm256_storeu_ps(a32_cp.as_mut_ptr(), a32);
        _mm256_storeu_ps(a33_cp.as_mut_ptr(), a33);
        let mut u11_cp: [f32; 8] = [0.0; 8]; let mut u12_cp: [f32; 8] = [0.0; 8]; let mut u13_cp: [f32; 8] = [0.0; 8];
        let mut u21_cp: [f32; 8] = [0.0; 8]; let mut u22_cp: [f32; 8] = [0.0; 8]; let mut u23_cp: [f32; 8] = [0.0; 8];
        let mut u31_cp: [f32; 8] = [0.0; 8]; let mut u32_cp: [f32; 8] = [0.0; 8]; let mut u33_cp: [f32; 8] = [0.0; 8];
        let mut v11_cp: [f32; 8] = [0.0; 8]; let mut v12_cp: [f32; 8] = [0.0; 8]; let mut v13_cp: [f32; 8] = [0.0; 8];
        let mut v21_cp: [f32; 8] = [0.0; 8]; let mut v22_cp: [f32; 8] = [0.0; 8]; let mut v23_cp: [f32; 8] = [0.0; 8];
        let mut v31_cp: [f32; 8] = [0.0; 8]; let mut v32_cp: [f32; 8] = [0.0; 8]; let mut v33_cp: [f32; 8] = [0.0; 8];
        let mut qus_cp: [f32; 8] = [0.0; 8];
        let mut quvx_cp: [f32; 8] = [0.0; 8];
        let mut quvy_cp: [f32; 8] = [0.0; 8];
        let mut quvz_cp: [f32; 8] = [0.0; 8];
        let mut qvs_cp: [f32; 8] = [0.0; 8];
        let mut qvvx_cp: [f32; 8] = [0.0; 8];
        let mut qvvy_cp: [f32; 8] = [0.0; 8];
        let mut qvvz_cp: [f32; 8] = [0.0; 8];
        _mm256_storeu_ps(u11_cp.as_mut_ptr(), u11);
        _mm256_storeu_ps(u12_cp.as_mut_ptr(), u12);
        _mm256_storeu_ps(u13_cp.as_mut_ptr(), u13);
        _mm256_storeu_ps(u21_cp.as_mut_ptr(), u21);
        _mm256_storeu_ps(u22_cp.as_mut_ptr(), u22);
        _mm256_storeu_ps(u23_cp.as_mut_ptr(), u23);
        _mm256_storeu_ps(u31_cp.as_mut_ptr(), u31);
        _mm256_storeu_ps(u32_cp.as_mut_ptr(), u32);
        _mm256_storeu_ps(u33_cp.as_mut_ptr(), u33);
        _mm256_storeu_ps(v11_cp.as_mut_ptr(), v11);
        _mm256_storeu_ps(v12_cp.as_mut_ptr(), v12);
        _mm256_storeu_ps(v13_cp.as_mut_ptr(), v13);
        _mm256_storeu_ps(v21_cp.as_mut_ptr(), v21);
        _mm256_storeu_ps(v22_cp.as_mut_ptr(), v22);
        _mm256_storeu_ps(v23_cp.as_mut_ptr(), v23);
        _mm256_storeu_ps(v31_cp.as_mut_ptr(), v31);
        _mm256_storeu_ps(v32_cp.as_mut_ptr(), v32);
        _mm256_storeu_ps(v33_cp.as_mut_ptr(), v33);
        _mm256_storeu_ps(qus_cp.as_mut_ptr(), qus);
        _mm256_storeu_ps(quvx_cp.as_mut_ptr(), quvx);
        _mm256_storeu_ps(quvy_cp.as_mut_ptr(), quvy);
        _mm256_storeu_ps(quvz_cp.as_mut_ptr(), quvz);
        _mm256_storeu_ps(qvs_cp.as_mut_ptr(), qvs);
        _mm256_storeu_ps(qvvx_cp.as_mut_ptr(), qvvx);
        _mm256_storeu_ps(qvvy_cp.as_mut_ptr(), qvvy);
        _mm256_storeu_ps(qvvz_cp.as_mut_ptr(), qvvz);
        let u: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::new(
                u11_cp[i], u12_cp[i], u13_cp[i],
                u21_cp[i], u22_cp[i], u23_cp[i],
                u31_cp[i], u32_cp[i], u33_cp[i]
            )
        }).collect::<Vec<_>>().try_into().unwrap();
        let s: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::from_diagonal(&na::Vector3::new(
                a11_cp[i], a22_cp[i], a33_cp[i]
            ))
        }).collect::<Vec<_>>().try_into().unwrap();
        let v: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::new(
                v11_cp[i], v12_cp[i], v13_cp[i],
                v21_cp[i], v22_cp[i], v23_cp[i],
                v31_cp[i], v32_cp[i], v33_cp[i]
            )
        }).collect::<Vec<_>>().try_into().unwrap();
        let qu: [na::UnitQuaternion<f32>; 8] = (0..8).map(|i| {
            na::UnitQuaternion::new_normalize(na::Quaternion::new(
                qus_cp[i], quvx_cp[i], quvy_cp[i], quvz_cp[i]
            ))
        }).collect::<Vec<_>>().try_into().unwrap();
        let qv: [na::UnitQuaternion<f32>; 8] = (0..8).map(|i| {
            na::UnitQuaternion::new_normalize(na::Quaternion::new(
                qvs_cp[i], qvvx_cp[i], qvvy_cp[i], qvvz_cp[i]
            ))
        }).collect::<Vec<_>>().try_into().unwrap();
        for j in 0..8 {
            test_results(&original[j], &u[j], &s[j], &v[j], &qu[j], &qv[j]);
        }
    }
}
}

#[cfg(feature = "portable_simd")]
#[test]
fn portable_simd_error_test() {
unsafe {
    let mut rng = rand::rng();
    let a11v = randoms_portable_simd(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a12v = randoms_portable_simd(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a13v = randoms_portable_simd(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a21v = randoms_portable_simd(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a22v = randoms_portable_simd(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a23v = randoms_portable_simd(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a31v = randoms_portable_simd(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a32v = randoms_portable_simd(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a33v = randoms_portable_simd(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    
    for i in 0..ERROR_TESTS {
        let mut a11 = a11v[i]; let mut a12 = a12v[i]; let mut a13 = a13v[i];
        let mut a21 = a21v[i]; let mut a22 = a22v[i]; let mut a23 = a23v[i];
        let mut a31 = a31v[i]; let mut a32 = a32v[i]; let mut a33 = a33v[i];

        let original: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::new(
                a11[i], a12[i], a13[i],
                a21[i], a22[i], a23[i],
                a31[i], a32[i], a33[i]
            )
        }).collect::<Vec<_>>().try_into().unwrap();

        let mut u11 = f32x8::splat(0.0); let mut u12 = f32x8::splat(0.0); let mut u13 = f32x8::splat(0.0);
        let mut u21 = f32x8::splat(0.0); let mut u22 = f32x8::splat(0.0); let mut u23 = f32x8::splat(0.0);
        let mut u31 = f32x8::splat(0.0); let mut u32 = f32x8::splat(0.0); let mut u33 = f32x8::splat(0.0);
        let mut v11 = f32x8::splat(0.0); let mut v12 = f32x8::splat(0.0); let mut v13 = f32x8::splat(0.0);
        let mut v21 = f32x8::splat(0.0); let mut v22 = f32x8::splat(0.0); let mut v23 = f32x8::splat(0.0);
        let mut v31 = f32x8::splat(0.0); let mut v32 = f32x8::splat(0.0); let mut v33 = f32x8::splat(0.0);
        let mut qus = f32x8::splat(0.0);
        let mut quvx = f32x8::splat(0.0);
        let mut quvy = f32x8::splat(0.0);
        let mut quvz = f32x8::splat(0.0);
        let mut qvs = f32x8::splat(0.0);
        let mut qvvx = f32x8::splat(0.0);
        let mut qvvy = f32x8::splat(0.0);
        let mut qvvz = f32x8::splat(0.0);

        svd::svd(
            true,
            true,
            true,
            true,
            true,
            true,
            &mut a11, &mut a12, &mut a13,
            &mut a21, &mut a22, &mut a23,
            &mut a31, &mut a32, &mut a33,
            &mut u11, &mut u12, &mut u13,
            &mut u21, &mut u22, &mut u23,
            &mut u31, &mut u32, &mut u33,
            &mut v11, &mut v12, &mut v13,
            &mut v21, &mut v22, &mut v23,
            &mut v31, &mut v32, &mut v33,
            &mut qus,
            &mut quvx,
            &mut quvy,
            &mut quvz,
            &mut qvs,
            &mut qvvx,
            &mut qvvy,
            &mut qvvz
        );

        let u: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::new(
                u11[i], u12[i], u13[i],
                u21[i], u22[i], u23[i],
                u31[i], u32[i], u33[i]
            )
        }).collect::<Vec<_>>().try_into().unwrap();
        let s: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::from_diagonal(&na::Vector3::new(
                a11[i], a22[i], a33[i]
            ))
        }).collect::<Vec<_>>().try_into().unwrap();
        let v: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::new(
                v11[i], v12[i], v13[i],
                v21[i], v22[i], v23[i],
                v31[i], v32[i], v33[i]
            )
        }).collect::<Vec<_>>().try_into().unwrap();
        let qu: [na::UnitQuaternion<f32>; 8] = (0..8).map(|i| {
            na::UnitQuaternion::new_normalize(na::Quaternion::new(
                qus[i], quvx[i], quvy[i], quvz[i]
            ))
        }).collect::<Vec<_>>().try_into().unwrap();
        let qv: [na::UnitQuaternion<f32>; 8] = (0..8).map(|i| {
            na::UnitQuaternion::new_normalize(na::Quaternion::new(
                qvs[i], qvvx[i], qvvy[i], qvvz[i]
            ))
        }).collect::<Vec<_>>().try_into().unwrap();
        for j in 0..8 {
            test_results(&original[j], &u[j], &s[j], &v[j], &qu[j], &qv[j]);
        }
    }
}
}

#[cfg(feature = "wide")]
#[test]
fn wide_error_test() {
    use wide::f32x8;
    let mut rng = rand::rng();
    let a11v = randoms_wide(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a12v = randoms_wide(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a13v = randoms_wide(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a21v = randoms_wide(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a22v = randoms_wide(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a23v = randoms_wide(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a31v = randoms_wide(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a32v = randoms_wide(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    let a33v = randoms_wide(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
    
    for i in 0..ERROR_TESTS {
        let mut a11 = a11v[i]; let mut a12 = a12v[i]; let mut a13 = a13v[i];
        let mut a21 = a21v[i]; let mut a22 = a22v[i]; let mut a23 = a23v[i];
        let mut a31 = a31v[i]; let mut a32 = a32v[i]; let mut a33 = a33v[i];

        let original: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::new(
                a11.as_array_ref()[i], a12.as_array_ref()[i], a13.as_array_ref()[i],
                a21.as_array_ref()[i], a22.as_array_ref()[i], a23.as_array_ref()[i],
                a31.as_array_ref()[i], a32.as_array_ref()[i], a33.as_array_ref()[i]
            )
        }).collect::<Vec<_>>().try_into().unwrap();

        let mut u11 = f32x8::splat(0.0); let mut u12 = f32x8::splat(0.0); let mut u13 = f32x8::splat(0.0);
        let mut u21 = f32x8::splat(0.0); let mut u22 = f32x8::splat(0.0); let mut u23 = f32x8::splat(0.0);
        let mut u31 = f32x8::splat(0.0); let mut u32 = f32x8::splat(0.0); let mut u33 = f32x8::splat(0.0);
        let mut v11 = f32x8::splat(0.0); let mut v12 = f32x8::splat(0.0); let mut v13 = f32x8::splat(0.0);
        let mut v21 = f32x8::splat(0.0); let mut v22 = f32x8::splat(0.0); let mut v23 = f32x8::splat(0.0);
        let mut v31 = f32x8::splat(0.0); let mut v32 = f32x8::splat(0.0); let mut v33 = f32x8::splat(0.0);
        let mut qus = f32x8::splat(0.0);
        let mut quvx = f32x8::splat(0.0);
        let mut quvy = f32x8::splat(0.0);
        let mut quvz = f32x8::splat(0.0);
        let mut qvs = f32x8::splat(0.0);
        let mut qvvx = f32x8::splat(0.0);
        let mut qvvy = f32x8::splat(0.0);
        let mut qvvz = f32x8::splat(0.0);

        svd::svd(
            true,
            true,
            true,
            true,
            true,
            true,
            &mut a11, &mut a12, &mut a13,
            &mut a21, &mut a22, &mut a23,
            &mut a31, &mut a32, &mut a33,
            &mut u11, &mut u12, &mut u13,
            &mut u21, &mut u22, &mut u23,
            &mut u31, &mut u32, &mut u33,
            &mut v11, &mut v12, &mut v13,
            &mut v21, &mut v22, &mut v23,
            &mut v31, &mut v32, &mut v33,
            &mut qus,
            &mut quvx,
            &mut quvy,
            &mut quvz,
            &mut qvs,
            &mut qvvx,
            &mut qvvy,
            &mut qvvz
        );

        let u: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::new(
                u11.as_array_ref()[i], u12.as_array_ref()[i], u13.as_array_ref()[i],
                u21.as_array_ref()[i], u22.as_array_ref()[i], u23.as_array_ref()[i],
                u31.as_array_ref()[i], u32.as_array_ref()[i], u33.as_array_ref()[i]
            )
        }).collect::<Vec<_>>().try_into().unwrap();
        let s: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::from_diagonal(&na::Vector3::new(
                a11.as_array_ref()[i], a22.as_array_ref()[i], a33.as_array_ref()[i]
            ))
        }).collect::<Vec<_>>().try_into().unwrap();
        let v: [na::Matrix3<f32>; 8] = (0..8).map(|i| {
            na::Matrix3::new(
                v11.as_array_ref()[i], v12.as_array_ref()[i], v13.as_array_ref()[i],
                v21.as_array_ref()[i], v22.as_array_ref()[i], v23.as_array_ref()[i],
                v31.as_array_ref()[i], v32.as_array_ref()[i], v33.as_array_ref()[i]
            )
        }).collect::<Vec<_>>().try_into().unwrap();
        let qu: [na::UnitQuaternion<f32>; 8] = (0..8).map(|i| {
            na::UnitQuaternion::new_normalize(na::Quaternion::new(
                qus.as_array_ref()[i], quvx.as_array_ref()[i], quvy.as_array_ref()[i], quvz.as_array_ref()[i]
            ))
        }).collect::<Vec<_>>().try_into().unwrap();
        let qv: [na::UnitQuaternion<f32>; 8] = (0..8).map(|i| {
            na::UnitQuaternion::new_normalize(na::Quaternion::new(
                qvs.as_array_ref()[i], qvvx.as_array_ref()[i], qvvy.as_array_ref()[i], qvvz.as_array_ref()[i]
            ))
        }).collect::<Vec<_>>().try_into().unwrap();
        for j in 0..8 {
            test_results(&original[j], &u[j], &s[j], &v[j], &qu[j], &qv[j]);
        }
    }
}

const PERFORMANCE_TESTS: usize = 10;
const PERFORMANCE_TEST_LENGTH: usize = 1000000;

#[test]
#[ignore]
fn f32_performance_test() {
    let mut rng = rand::rng();

    for _ in 0..PERFORMANCE_TESTS {
        let a11v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
        let a12v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
        let a13v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
        let a21v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
        let a22v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
        let a23v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
        let a31v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
        let a32v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
        let a33v = randoms_f32(ERROR_TESTS, TEST_RANGE.clone(), &mut rng);
        let start = std::time::Instant::now();
        for i in 0..ERROR_TESTS {
            let mut a11 = a11v[i]; let mut a12 = a12v[i]; let mut a13 = a13v[i];
            let mut a21 = a21v[i]; let mut a22 = a22v[i]; let mut a23 = a23v[i];
            let mut a31 = a31v[i]; let mut a32 = a32v[i]; let mut a33 = a33v[i];
            let mut u11 = 0.0; let mut u12 = 0.0; let mut u13 = 0.0;
            let mut u21 = 0.0; let mut u22 = 0.0; let mut u23 = 0.0;
            let mut u31 = 0.0; let mut u32 = 0.0; let mut u33 = 0.0;
            let mut v11 = 0.0; let mut v12 = 0.0; let mut v13 = 0.0;
            let mut v21 = 0.0; let mut v22 = 0.0; let mut v23 = 0.0;
            let mut v31 = 0.0; let mut v32 = 0.0; let mut v33 = 0.0;
            let mut qus = 0.0;
            let mut quvx = 0.0;
            let mut quvy = 0.0;
            let mut quvz = 0.0;
            let mut qvs = 0.0;
            let mut qvvx = 0.0;
            let mut qvvy = 0.0;
            let mut qvvz = 0.0;

            svd(
                true,
                true,
                true,
                true,
                true,
                true,
                &mut a11, &mut a12, &mut a13,
                &mut a21, &mut a22, &mut a23,
                &mut a31, &mut a32, &mut a33,
                &mut u11, &mut u12, &mut u13,
                &mut u21, &mut u22, &mut u23,
                &mut u31, &mut u32, &mut u33,
                &mut v11, &mut v12, &mut v13,
                &mut v21, &mut v22, &mut v23,
                &mut v31, &mut v32, &mut v33,
                &mut qus, &mut quvx, &mut quvy, &mut quvz,
                &mut qvs, &mut qvvx, &mut qvvy, &mut qvvz
            );

            black_box(a11); black_box(a12); black_box(a13);
            black_box(a21); black_box(a22); black_box(a23);
            black_box(a31); black_box(a32); black_box(a33);
            black_box(u11); black_box(u12); black_box(u13);
            black_box(u21); black_box(u22); black_box(u23);
            black_box(u31); black_box(u32); black_box(u33);
            black_box(v11); black_box(v12); black_box(v13);
            black_box(v21); black_box(v22); black_box(v23);
            black_box(v31); black_box(v32); black_box(v33);
            black_box(qus);
            black_box(quvx);
            black_box(quvy);
            black_box(quvz);
            black_box(qvs);
            black_box(qvvx);
            black_box(qvvy);
            black_box(qvvz);
        }
        println!("Time taken for f32 SVD: {:?}", start.elapsed());
    }
}

#[cfg(feature = "avx")]
#[test]
#[ignore]
fn __mm256_performance_test() {
unsafe {
    let mut rng = rand::rng();

    for _ in 0..PERFORMANCE_TESTS {
        let a11v = randoms_m256(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a12v = randoms_m256(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a13v = randoms_m256(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a21v = randoms_m256(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a22v = randoms_m256(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a23v = randoms_m256(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a31v = randoms_m256(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a32v = randoms_m256(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a33v = randoms_m256(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
    
        let start = std::time::Instant::now();
        for i in 0..PERFORMANCE_TEST_LENGTH {
            let mut a11 = a11v[i]; let mut a12 = a12v[i]; let mut a13 = a13v[i];
            let mut a21 = a21v[i]; let mut a22 = a22v[i]; let mut a23 = a23v[i];
            let mut a31 = a31v[i]; let mut a32 = a32v[i]; let mut a33 = a33v[i];

            let mut u11 = _mm256_setzero_ps();
            let mut u12 = _mm256_setzero_ps();
            let mut u13 = _mm256_setzero_ps();
            let mut u21 = _mm256_setzero_ps();
            let mut u22 = _mm256_setzero_ps();
            let mut u23 = _mm256_setzero_ps();
            let mut u31 = _mm256_setzero_ps();
            let mut u32 = _mm256_setzero_ps();
            let mut u33 = _mm256_setzero_ps();
            let mut v11 = _mm256_setzero_ps();
            let mut v12 = _mm256_setzero_ps();
            let mut v13 = _mm256_setzero_ps();
            let mut v21 = _mm256_setzero_ps();
            let mut v22 = _mm256_setzero_ps();
            let mut v23 = _mm256_setzero_ps();
            let mut v31 = _mm256_setzero_ps();
            let mut v32 = _mm256_setzero_ps();
            let mut v33 = _mm256_setzero_ps();
            let mut qus = _mm256_setzero_ps();
            let mut quvx = _mm256_setzero_ps();
            let mut quvy = _mm256_setzero_ps();
            let mut quvz = _mm256_setzero_ps();
            let mut qvs = _mm256_setzero_ps();
            let mut qvvx = _mm256_setzero_ps();
            let mut qvvy = _mm256_setzero_ps();
            let mut qvvz = _mm256_setzero_ps();

            svd::svd(
                true,
                true,
                true,
                true,
                true,
                true,
                &mut a11, &mut a12, &mut a13,
                &mut a21, &mut a22, &mut a23,
                &mut a31, &mut a32, &mut a33,
                &mut u11, &mut u12, &mut u13,
                &mut u21, &mut u22, &mut u23,
                &mut u31, &mut u32, &mut u33,
                &mut v11, &mut v12, &mut v13,
                &mut v21, &mut v22, &mut v23,
                &mut v31, &mut v32, &mut v33,
                &mut qus,
                &mut quvx,
                &mut quvy,
                &mut quvz,
                &mut qvs,
                &mut qvvx,
                &mut qvvy,
                &mut qvvz
            );

            black_box(a11); black_box(a12); black_box(a13);
            black_box(a21); black_box(a22); black_box(a23);
            black_box(a31); black_box(a32); black_box(a33);
            black_box(u11); black_box(u12); black_box(u13);
            black_box(u21); black_box(u22); black_box(u23);
            black_box(u31); black_box(u32); black_box(u33);
            black_box(v11); black_box(v12); black_box(v13);
            black_box(v21); black_box(v22); black_box(v23);
            black_box(v31); black_box(v32); black_box(v33);
            black_box(qus);
            black_box(quvx);
            black_box(quvy);
            black_box(quvz);
            black_box(qvs);
            black_box(qvvx);
            black_box(qvvy);
            black_box(qvvz);
        }

        println!("Time taken for __m256 SVD: {:?}", start.elapsed());
    }
}
}

#[cfg(feature = "portable_simd")]
#[test]
#[ignore]
fn portable_simd_performance_test() {
unsafe {
    let mut rng = rand::rng();

    for _ in 0..PERFORMANCE_TESTS {
        let a11v = randoms_portable_simd(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a12v = randoms_portable_simd(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a13v = randoms_portable_simd(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a21v = randoms_portable_simd(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a22v = randoms_portable_simd(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a23v = randoms_portable_simd(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a31v = randoms_portable_simd(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a32v = randoms_portable_simd(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a33v = randoms_portable_simd(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
    
        let start = std::time::Instant::now();
        for i in 0..PERFORMANCE_TEST_LENGTH {
            let mut a11 = a11v[i]; let mut a12 = a12v[i]; let mut a13 = a13v[i];
            let mut a21 = a21v[i]; let mut a22 = a22v[i]; let mut a23 = a23v[i];
            let mut a31 = a31v[i]; let mut a32 = a32v[i]; let mut a33 = a33v[i];

            let mut u11 = f32x8::splat(0.0);
            let mut u12 = f32x8::splat(0.0);
            let mut u13 = f32x8::splat(0.0);
            let mut u21 = f32x8::splat(0.0);
            let mut u22 = f32x8::splat(0.0);
            let mut u23 = f32x8::splat(0.0);
            let mut u31 = f32x8::splat(0.0);
            let mut u32 = f32x8::splat(0.0);
            let mut u33 = f32x8::splat(0.0);
            let mut v11 = f32x8::splat(0.0);
            let mut v12 = f32x8::splat(0.0);
            let mut v13 = f32x8::splat(0.0);
            let mut v21 = f32x8::splat(0.0);
            let mut v22 = f32x8::splat(0.0);
            let mut v23 = f32x8::splat(0.0);
            let mut v31 = f32x8::splat(0.0);
            let mut v32 = f32x8::splat(0.0);
            let mut v33 = f32x8::splat(0.0);
            let mut qus = f32x8::splat(0.0);
            let mut quvx = f32x8::splat(0.0);
            let mut quvy = f32x8::splat(0.0);
            let mut quvz = f32x8::splat(0.0);
            let mut qvs = f32x8::splat(0.0);
            let mut qvvx = f32x8::splat(0.0);
            let mut qvvy = f32x8::splat(0.0);
            let mut qvvz = f32x8::splat(0.0);

            svd::svd(
                true,
                true,
                true,
                true,
                true,
                true,
                &mut a11, &mut a12, &mut a13,
                &mut a21, &mut a22, &mut a23,
                &mut a31, &mut a32, &mut a33,
                &mut u11, &mut u12, &mut u13,
                &mut u21, &mut u22, &mut u23,
                &mut u31, &mut u32, &mut u33,
                &mut v11, &mut v12, &mut v13,
                &mut v21, &mut v22, &mut v23,
                &mut v31, &mut v32, &mut v33,
                &mut qus,
                &mut quvx,
                &mut quvy,
                &mut quvz,
                &mut qvs,
                &mut qvvx,
                &mut qvvy,
                &mut qvvz
            );

            black_box(a11); black_box(a12); black_box(a13);
            black_box(a21); black_box(a22); black_box(a23);
            black_box(a31); black_box(a32); black_box(a33);
            black_box(u11); black_box(u12); black_box(u13);
            black_box(u21); black_box(u22); black_box(u23);
            black_box(u31); black_box(u32); black_box(u33);
            black_box(v11); black_box(v12); black_box(v13);
            black_box(v21); black_box(v22); black_box(v23);
            black_box(v31); black_box(v32); black_box(v33);
            black_box(qus);
            black_box(quvx);
            black_box(quvy);
            black_box(quvz);
            black_box(qvs);
            black_box(qvvx);
            black_box(qvvy);
            black_box(qvvz);
        }

        println!("Time taken for portable simd SVD: {:?}", start.elapsed());
    }
}
}

#[cfg(feature = "wide")]
#[test]
#[ignore]
fn wide_performance_test() {
    use wide::f32x8;
    let mut rng = rand::rng();

    for _ in 0..PERFORMANCE_TESTS {
        let a11v = randoms_wide(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a12v = randoms_wide(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a13v = randoms_wide(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a21v = randoms_wide(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a22v = randoms_wide(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a23v = randoms_wide(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a31v = randoms_wide(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a32v = randoms_wide(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
        let a33v = randoms_wide(PERFORMANCE_TEST_LENGTH, TEST_RANGE.clone(), &mut rng);
    
        let start = std::time::Instant::now();
        for i in 0..PERFORMANCE_TEST_LENGTH {
            let mut a11 = a11v[i]; let mut a12 = a12v[i]; let mut a13 = a13v[i];
            let mut a21 = a21v[i]; let mut a22 = a22v[i]; let mut a23 = a23v[i];
            let mut a31 = a31v[i]; let mut a32 = a32v[i]; let mut a33 = a33v[i];

            let mut u11 = f32x8::splat(0.0);
            let mut u12 = f32x8::splat(0.0);
            let mut u13 = f32x8::splat(0.0);
            let mut u21 = f32x8::splat(0.0);
            let mut u22 = f32x8::splat(0.0);
            let mut u23 = f32x8::splat(0.0);
            let mut u31 = f32x8::splat(0.0);
            let mut u32 = f32x8::splat(0.0);
            let mut u33 = f32x8::splat(0.0);
            let mut v11 = f32x8::splat(0.0);
            let mut v12 = f32x8::splat(0.0);
            let mut v13 = f32x8::splat(0.0);
            let mut v21 = f32x8::splat(0.0);
            let mut v22 = f32x8::splat(0.0);
            let mut v23 = f32x8::splat(0.0);
            let mut v31 = f32x8::splat(0.0);
            let mut v32 = f32x8::splat(0.0);
            let mut v33 = f32x8::splat(0.0);
            let mut qus = f32x8::splat(0.0);
            let mut quvx = f32x8::splat(0.0);
            let mut quvy = f32x8::splat(0.0);
            let mut quvz = f32x8::splat(0.0);
            let mut qvs = f32x8::splat(0.0);
            let mut qvvx = f32x8::splat(0.0);
            let mut qvvy = f32x8::splat(0.0);
            let mut qvvz = f32x8::splat(0.0);

            svd::svd(
                true,
                true,
                true,
                true,
                true,
                true,
                &mut a11, &mut a12, &mut a13,
                &mut a21, &mut a22, &mut a23,
                &mut a31, &mut a32, &mut a33,
                &mut u11, &mut u12, &mut u13,
                &mut u21, &mut u22, &mut u23,
                &mut u31, &mut u32, &mut u33,
                &mut v11, &mut v12, &mut v13,
                &mut v21, &mut v22, &mut v23,
                &mut v31, &mut v32, &mut v33,
                &mut qus,
                &mut quvx,
                &mut quvy,
                &mut quvz,
                &mut qvs,
                &mut qvvx,
                &mut qvvy,
                &mut qvvz
            );

            black_box(a11); black_box(a12); black_box(a13);
            black_box(a21); black_box(a22); black_box(a23);
            black_box(a31); black_box(a32); black_box(a33);
            black_box(u11); black_box(u12); black_box(u13);
            black_box(u21); black_box(u22); black_box(u23);
            black_box(u31); black_box(u32); black_box(u33);
            black_box(v11); black_box(v12); black_box(v13);
            black_box(v21); black_box(v22); black_box(v23);
            black_box(v31); black_box(v32); black_box(v33);
            black_box(qus);
            black_box(quvx);
            black_box(quvy);
            black_box(quvz);
            black_box(qvs);
            black_box(qvvx);
            black_box(qvvy);
            black_box(qvvz);
        }

        println!("Time taken for wide SVD: {:?}", start.elapsed());
    }
}
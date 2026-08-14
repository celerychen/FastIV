#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use crate::jacobi_conjugation;
use crate::givens_qr_factorization;
use crate::traits::SVDCompatible;

// #ifdef PRINT_DEBUGGING_OUTPUT

// #ifdef USE_SSE_IMPLEMENTATION
//     float buf[4];
//     float A11,A21,A31,A12,A22,A32,A13,A23,A33;
//     float S11,S21,S31,S22,S32,S33;
// #ifdef COMPUTE_V_AS_QUATERNION
//     float QVS,QVVX,QVVY,QVVZ;
// #endif
// #ifdef COMPUTE_V_AS_MATRIX
//     float V11,V21,V31,V12,V22,V32,V13,V23,V33;
// #endif
// #ifdef COMPUTE_U_AS_QUATERNION
//     float QUS,QUVX,QUVY,QUVZ;
// #endif
// #ifdef COMPUTE_U_AS_MATRIX
//     float U11,U21,U31,U12,U22,U32,U13,U23,U33;
// #endif
// #endif

// #ifdef USE_AVX_IMPLEMENTATION
//     float buf[8];
//     float A11,A21,A31,A12,A22,A32,A13,A23,A33;
//     float S11,S21,S31,S22,S32,S33;
// #ifdef COMPUTE_V_AS_QUATERNION
//     float QVS,QVVX,QVVY,QVVZ;
// #endif
// #ifdef COMPUTE_V_AS_MATRIX
//     float V11,V21,V31,V12,V22,V32,V13,V23,V33;
// #endif
// #ifdef COMPUTE_U_AS_QUATERNION
//     float QUS,QUVX,QUVY,QUVZ;
// #endif
// #ifdef COMPUTE_U_AS_MATRIX
//     float U11,U21,U31,U12,U22,U32,U13,U23,U33;
// #endif
// #endif

// #endif

const Four_Gamma_Squared: f32 = 5.82842712474619; // sqrt(8.)+3.;
const Sine_Pi_Over_Eight: f32 = 0.3826834323650897; // .5*sqrt(2.-sqrt(2.));
const Cosine_Pi_Over_Eight: f32 = 0.9238795325112867; // .5*sqrt(2.+sqrt(2.));

#[inline(always)]
pub fn svd<VType: SVDCompatible>(
compute_u_as_matrix: bool,
compute_v_as_matrix: bool,
compute_u_as_quaternion: bool,
compute_v_as_quaternion: bool,
use_accurate_rsqrt_in_jacobi_conjugation: bool,
perform_strict_quaternion_renormalization: bool,
Va11: &mut VType,
Va12: &mut VType,
Va13: &mut VType,
Va21: &mut VType,
Va22: &mut VType,
Va23: &mut VType,
Va31: &mut VType,
Va32: &mut VType,
Va33: &mut VType,
Vu11: &mut VType,
Vu12: &mut VType,
Vu13: &mut VType,
Vu21: &mut VType,
Vu22: &mut VType,
Vu23: &mut VType,
Vu31: &mut VType,
Vu32: &mut VType,
Vu33: &mut VType,
Vv11: &mut VType,
Vv12: &mut VType,
Vv13: &mut VType,
Vv21: &mut VType,
Vv22: &mut VType,
Vv23: &mut VType,
Vv31: &mut VType,
Vv32: &mut VType,
Vv33: &mut VType,
Vqus: &mut VType,
Vquvx: &mut VType,
Vquvy: &mut VType,
Vquvz: &mut VType,
Vqvs: &mut VType,
Vqvvx: &mut VType,
Vqvvy: &mut VType,
Vqvvz: &mut VType,
) {
// Declarations
let Vfour_gamma_squared = VType::splat(Four_Gamma_Squared);
let Vsine_pi_over_eight = VType::splat(Sine_Pi_Over_Eight);
let Vcosine_pi_over_eight = VType::splat(Cosine_Pi_Over_Eight);
let Vone_half = VType::splat(0.5);
let Vone = VType::splat(1.0);
let Vtiny_number = VType::splat(1.0e-20);
let Vsmall_number = VType::splat(1.0e-12);
let Vone_mask = VType::ones();

let mut Vc;
let mut Vs;
let mut Vch;
let mut Vsh;
let mut Vtmp1;
let mut Vtmp2;
let mut Vtmp3;
let mut Vtmp4;
let mut Vtmp5;
let mut Mtmp;

{ // Begin block : Scope of qV (if not maintained)


{ // Begin block : Symmetric eigenanalysis

let mut Vs11;
let mut Vs21;
let mut Vs31;
let mut Vs22;
let mut Vs32;
let mut Vs33;

*Vqvs = VType::splat(1.0);
*Vqvvx = VType::splat(0.0);
*Vqvvy = VType::splat(0.0);
*Vqvvz = VType::splat(0.0);

//###########################################################
// Compute normal equations matrix
//###########################################################

Vs11=Va11.mul(Va11);
Vtmp1=Va21.mul(Va21);
Vs11=Vtmp1.add(&Vs11);
Vtmp1=Va31.mul(Va31);
Vs11=Vtmp1.add(&Vs11);

Vs21=Va12.mul(Va11);
Vtmp1=Va22.mul(Va21);
Vs21=Vtmp1.add(&Vs21);
Vtmp1=Va32.mul(Va31);
Vs21=Vtmp1.add(&Vs21);

Vs31=Va13.mul(Va11);
Vtmp1=Va23.mul(Va21);
Vs31=Vtmp1.add(&Vs31);
Vtmp1=Va33.mul(Va31);
Vs31=Vtmp1.add(&Vs31);

Vs22=Va12.mul(Va12);
Vtmp1=Va22.mul(Va22);
Vs22=Vtmp1.add(&Vs22);
Vtmp1=Va32.mul(Va32);
Vs22=Vtmp1.add(&Vs22);

Vs32=Va13.mul(Va12);
Vtmp1=Va23.mul(Va22);
Vs32=Vtmp1.add(&Vs32);
Vtmp1=Va33.mul(Va32);
Vs32=Vtmp1.add(&Vs32);

Vs33=Va13.mul(Va13);
Vtmp1=Va23.mul(Va23);
Vs33=Vtmp1.add(&Vs33);
Vtmp1=Va33.mul(Va33);
Vs33=Vtmp1.add(&Vs33);

//###########################################################
// Solve symmetric eigenproblem using Jacobi iteration
//###########################################################

for _ in 0..4 {
// First Jacobi conjugation
jacobi_conjugation!(
    VType,
    use_accurate_rsqrt_in_jacobi_conjugation,
    Vfour_gamma_squared,
    Vsine_pi_over_eight,
    Vcosine_pi_over_eight,
    Vone_half,
    Vone,
    Vtiny_number,
    Vsmall_number,
    Vc,
    Vs,
    Vch,
    Vsh,
    Vtmp1,
    Vtmp2,
    Vtmp3,
    Vtmp4,
    Vtmp5,
    Vqvs,
    Vqvvx,
    Vqvvy,
    Vqvvz,
    Vs11,
    Vs21,
    Vs31,
    Vs22,
    Vs32,
    Vs33,
    Vqvvx,
    Vqvvy,
    Vqvvz,
    Vtmp1,
    Vtmp2,
    Vtmp3,
    Mtmp
);

// Second Jacobi conjugation
jacobi_conjugation!(
    VType,
    use_accurate_rsqrt_in_jacobi_conjugation,
    Vfour_gamma_squared,
    Vsine_pi_over_eight,
    Vcosine_pi_over_eight,
    Vone_half,
    Vone,
    Vtiny_number,
    Vsmall_number,
    Vc,
    Vs,
    Vch,
    Vsh,
    Vtmp1,
    Vtmp2,
    Vtmp3,
    Vtmp4,
    Vtmp5,
    Vqvs,
    Vqvvx,
    Vqvvy,
    Vqvvz,
    Vs22,
    Vs32,
    Vs21,
    Vs33,
    Vs31,
    Vs11,
    Vqvvy,
    Vqvvz,
    Vqvvx,
    Vtmp2,
    Vtmp3,
    Vtmp1,
    Mtmp
);

// Third Jacobi conjugation
jacobi_conjugation!(
    VType,
    use_accurate_rsqrt_in_jacobi_conjugation,
    Vfour_gamma_squared,
    Vsine_pi_over_eight,
    Vcosine_pi_over_eight,
    Vone_half,
    Vone,
    Vtiny_number,
    Vsmall_number,
    Vc,
    Vs,
    Vch,
    Vsh,
    Vtmp1,
    Vtmp2,
    Vtmp3,
    Vtmp4,
    Vtmp5,
    Vqvs,
    Vqvvx,
    Vqvvy,
    Vqvvz,
    Vs33,
    Vs31,
    Vs32,
    Vs11,
    Vs21,
    Vs22,
    Vqvvz,
    Vqvvx,
    Vqvvy,
    Vtmp3,
    Vtmp1,
    Vtmp2,
    Mtmp
);
}

// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar S ="<<std::endl;
//     std::cout<<std::setw(12)<<Ss11.f<<std::endl;
//     std::cout<<std::setw(12)<<Ss21.f<<"  "<<std::setw(12)<<Ss22.f<<std::endl;
//     std::cout<<std::setw(12)<<Ss31.f<<"  "<<std::setw(12)<<Ss32.f<<"  "<<std::setw(12)<<Ss33.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Vs11);S11=buf[0];
//     _mm_storeu_ps(buf,Vs21);S21=buf[0];
//     _mm_storeu_ps(buf,Vs31);S31=buf[0];
//     _mm_storeu_ps(buf,Vs22);S22=buf[0];
//     _mm_storeu_ps(buf,Vs32);S32=buf[0];
//     _mm_storeu_ps(buf,Vs33);S33=buf[0];
//     std::cout<<"Vector S ="<<std::endl;
//     std::cout<<std::setw(12)<<S11<<std::endl;
//     std::cout<<std::setw(12)<<S21<<"  "<<std::setw(12)<<S22<<std::endl;
//     std::cout<<std::setw(12)<<S31<<"  "<<std::setw(12)<<S32<<"  "<<std::setw(12)<<S33<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Vs11);S11=buf[0];
//     _mm256_storeu_ps(buf,Vs21);S21=buf[0];
//     _mm256_storeu_ps(buf,Vs31);S31=buf[0];
//     _mm256_storeu_ps(buf,Vs22);S22=buf[0];
//     _mm256_storeu_ps(buf,Vs32);S32=buf[0];
//     _mm256_storeu_ps(buf,Vs33);S33=buf[0];
//     std::cout<<"Vector S ="<<std::endl;
//     std::cout<<std::setw(12)<<S11<<std::endl;
//     std::cout<<std::setw(12)<<S21<<"  "<<std::setw(12)<<S22<<std::endl;
//     std::cout<<std::setw(12)<<S31<<"  "<<std::setw(12)<<S32<<"  "<<std::setw(12)<<S33<<std::endl;
// #endif
// #endif

} // End block : Symmetric eigenanalysis

//###########################################################
// Normalize quaternion for matrix V
//###########################################################

if !use_accurate_rsqrt_in_jacobi_conjugation || perform_strict_quaternion_renormalization {
Vtmp2=Vqvs.mul(Vqvs);
Vtmp1=Vqvvx.mul(Vqvvx);
Vtmp2=Vtmp1.add(&Vtmp2);
Vtmp1=Vqvvy.mul(Vqvvy);
Vtmp2=Vtmp1.add(&Vtmp2);
Vtmp1=Vqvvz.mul(Vqvvz);
Vtmp2=Vtmp1.add(&Vtmp2);

Vtmp1=Vtmp2.rsqrt();
Vtmp4=Vtmp1.mul(&Vone_half);
Vtmp3=Vtmp1.mul(&Vtmp4);
Vtmp3=Vtmp1.mul(&Vtmp3);
Vtmp3=Vtmp2.mul(&Vtmp3);
Vtmp1=Vtmp1.add(&Vtmp4);
Vtmp1=Vtmp1.sub(&Vtmp3);

*Vqvs=Vqvs.mul(&Vtmp1);
*Vqvvx=Vqvvx.mul(&Vtmp1);
*Vqvvy=Vqvvy.mul(&Vtmp1);
*Vqvvz=Vqvvz.mul(&Vtmp1);

// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar qV ="<<std::endl;
//     std::cout<<std::setw(12)<<Sqvs.f<<"  "<<std::setw(12)<<Sqvvx.f<<"  "<<std::setw(12)<<Sqvvy.f<<"  "<<std::setw(12)<<Sqvvz.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Vqvs);QVS=buf[0];
//     _mm_storeu_ps(buf,Vqvvx);QVVX=buf[0];
//     _mm_storeu_ps(buf,Vqvvy);QVVY=buf[0];
//     _mm_storeu_ps(buf,Vqvvz);QVVZ=buf[0];
//     std::cout<<"Vector qV ="<<std::endl;
//     std::cout<<std::setw(12)<<QVS<<"  "<<std::setw(12)<<QVVX<<"  "<<std::setw(12)<<QVVY<<"  "<<std::setw(12)<<QVVZ<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Vqvs);QVS=buf[0];
//     _mm256_storeu_ps(buf,Vqvvx);QVVX=buf[0];
//     _mm256_storeu_ps(buf,Vqvvy);QVVY=buf[0];
//     _mm256_storeu_ps(buf,Vqvvz);QVVZ=buf[0];
//     std::cout<<"Vector qV ="<<std::endl;
//     std::cout<<std::setw(12)<<QVS<<"  "<<std::setw(12)<<QVVX<<"  "<<std::setw(12)<<QVVY<<"  "<<std::setw(12)<<QVVZ<<std::endl;
// #endif
// #endif

}

{ // Begin block : Conjugation with V

//###########################################################
// Transform quaternion to matrix V
//###########################################################

Vtmp1=Vqvvx.mul(Vqvvx);
Vtmp2=Vqvvy.mul(Vqvvy);
Vtmp3=Vqvvz.mul(Vqvvz);
*Vv11=Vqvs.mul(Vqvs);
*Vv22=Vv11.sub(&Vtmp1);
*Vv33=Vv22.sub(&Vtmp2);
*Vv33=Vv33.add(&Vtmp3);
*Vv22=Vv22.add(&Vtmp2);
*Vv22=Vv22.sub(&Vtmp3);
*Vv11=Vv11.add(&Vtmp1);
*Vv11=Vv11.sub(&Vtmp2);
*Vv11=Vv11.sub(&Vtmp3);
Vtmp1=Vqvvx.add(Vqvvx);
Vtmp2=Vqvvy.add(Vqvvy);
Vtmp3=Vqvvz.add(Vqvvz);
*Vv32=Vqvs.mul(&Vtmp1);
*Vv13=Vqvs.mul(&Vtmp2);
*Vv21=Vqvs.mul(&Vtmp3);
Vtmp1=Vqvvy.mul(&Vtmp1);
Vtmp2=Vqvvz.mul(&Vtmp2);
Vtmp3=Vqvvx.mul(&Vtmp3);
*Vv12=Vtmp1.sub(Vv21);
*Vv23=Vtmp2.sub(Vv32);
*Vv31=Vtmp3.sub(Vv13);
*Vv21=Vtmp1.add(Vv21);
*Vv32=Vtmp2.add(Vv32);
*Vv13=Vtmp3.add(Vv13);

// #ifdef COMPUTE_V_AS_MATRIX
// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar V ="<<std::endl;
//     std::cout<<std::setw(12)<<Sv11.f<<"  "<<std::setw(12)<<Sv12.f<<"  "<<std::setw(12)<<Sv13.f<<std::endl;
//     std::cout<<std::setw(12)<<Sv21.f<<"  "<<std::setw(12)<<Sv22.f<<"  "<<std::setw(12)<<Sv23.f<<std::endl;
//     std::cout<<std::setw(12)<<Sv31.f<<"  "<<std::setw(12)<<Sv32.f<<"  "<<std::setw(12)<<Sv33.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Vv11);V11=buf[0];
//     _mm_storeu_ps(buf,Vv21);V21=buf[0];
//     _mm_storeu_ps(buf,Vv31);V31=buf[0];
//     _mm_storeu_ps(buf,Vv12);V12=buf[0];
//     _mm_storeu_ps(buf,Vv22);V22=buf[0];
//     _mm_storeu_ps(buf,Vv32);V32=buf[0];
//     _mm_storeu_ps(buf,Vv13);V13=buf[0];
//     _mm_storeu_ps(buf,Vv23);V23=buf[0];
//     _mm_storeu_ps(buf,Vv33);V33=buf[0];
//     std::cout<<"Vector V ="<<std::endl;
//     std::cout<<std::setw(12)<<V11<<"  "<<std::setw(12)<<V12<<"  "<<std::setw(12)<<V13<<std::endl;
//     std::cout<<std::setw(12)<<V21<<"  "<<std::setw(12)<<V22<<"  "<<std::setw(12)<<V23<<std::endl;
//     std::cout<<std::setw(12)<<V31<<"  "<<std::setw(12)<<V32<<"  "<<std::setw(12)<<V33<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Vv11);V11=buf[0];
//     _mm256_storeu_ps(buf,Vv21);V21=buf[0];
//     _mm256_storeu_ps(buf,Vv31);V31=buf[0];
//     _mm256_storeu_ps(buf,Vv12);V12=buf[0];
//     _mm256_storeu_ps(buf,Vv22);V22=buf[0];
//     _mm256_storeu_ps(buf,Vv32);V32=buf[0];
//     _mm256_storeu_ps(buf,Vv13);V13=buf[0];
//     _mm256_storeu_ps(buf,Vv23);V23=buf[0];
//     _mm256_storeu_ps(buf,Vv33);V33=buf[0];
//     std::cout<<"Vector V ="<<std::endl;
//     std::cout<<std::setw(12)<<V11<<"  "<<std::setw(12)<<V12<<"  "<<std::setw(12)<<V13<<std::endl;
//     std::cout<<std::setw(12)<<V21<<"  "<<std::setw(12)<<V22<<"  "<<std::setw(12)<<V23<<std::endl;
//     std::cout<<std::setw(12)<<V31<<"  "<<std::setw(12)<<V32<<"  "<<std::setw(12)<<V33<<std::endl;
// #endif
// #endif
// #endif

//###########################################################
// Multiply (from the right) with V
//###########################################################

Vtmp2=Va12.clone();
Vtmp3=Va13.clone();
*Va12=Vv12.mul(Va11);
*Va13=Vv13.mul(Va11);
*Va11=Vv11.mul(Va11);
Vtmp1=Vv21.mul(&Vtmp2);
*Va11=Va11.add(&Vtmp1);
Vtmp1=Vv31.mul(&Vtmp3);
*Va11=Va11.add(&Vtmp1);
Vtmp1=Vv22.mul(&Vtmp2);
*Va12=Va12.add(&Vtmp1);
Vtmp1=Vv32.mul(&Vtmp3);
*Va12=Va12.add(&Vtmp1);
Vtmp1=Vv23.mul(&Vtmp2);
*Va13=Va13.add(&Vtmp1);
Vtmp1=Vv33.mul(&Vtmp3);
*Va13=Va13.add(&Vtmp1);

Vtmp2=Va22.clone();
Vtmp3=Va23.clone();
*Va22=Vv12.mul(Va21);
*Va23=Vv13.mul(Va21);
*Va21=Vv11.mul(Va21);
Vtmp1=Vv21.mul(&Vtmp2);
*Va21=Va21.add(&Vtmp1);
Vtmp1=Vv31.mul(&Vtmp3);
*Va21=Va21.add(&Vtmp1);
Vtmp1=Vv22.mul(&Vtmp2);
*Va22=Va22.add(&Vtmp1);
Vtmp1=Vv32.mul(&Vtmp3);
*Va22=Va22.add(&Vtmp1);
Vtmp1=Vv23.mul(&Vtmp2);
*Va23=Va23.add(&Vtmp1);
Vtmp1=Vv33.mul(&Vtmp3);
*Va23=Va23.add(&Vtmp1);

Vtmp2=Va32.clone();
Vtmp3=Va33.clone();
*Va32=Vv12.mul(Va31);
*Va33=Vv13.mul(Va31);
*Va31=Vv11.mul(Va31);
Vtmp1=Vv21.mul(&Vtmp2);
*Va31=Va31.add(&Vtmp1);
Vtmp1=Vv31.mul(&Vtmp3);
*Va31=Va31.add(&Vtmp1);
Vtmp1=Vv22.mul(&Vtmp2);
*Va32=Va32.add(&Vtmp1);
Vtmp1=Vv32.mul(&Vtmp3);
*Va32=Va32.add(&Vtmp1);
Vtmp1=Vv23.mul(&Vtmp2);
*Va33=Va33.add(&Vtmp1);
Vtmp1=Vv33.mul(&Vtmp3);
*Va33=Va33.add(&Vtmp1);

// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar A (after multiplying with V) ="<<std::endl;
//     std::cout<<std::setw(12)<<Sa11.f<<"  "<<std::setw(12)<<Sa12.f<<"  "<<std::setw(12)<<Sa13.f<<std::endl;
//     std::cout<<std::setw(12)<<Sa21.f<<"  "<<std::setw(12)<<Sa22.f<<"  "<<std::setw(12)<<Sa23.f<<std::endl;
//     std::cout<<std::setw(12)<<Sa31.f<<"  "<<std::setw(12)<<Sa32.f<<"  "<<std::setw(12)<<Sa33.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Va11);A11=buf[0];
//     _mm_storeu_ps(buf,Va21);A21=buf[0];
//     _mm_storeu_ps(buf,Va31);A31=buf[0];
//     _mm_storeu_ps(buf,Va12);A12=buf[0];
//     _mm_storeu_ps(buf,Va22);A22=buf[0];
//     _mm_storeu_ps(buf,Va32);A32=buf[0];
//     _mm_storeu_ps(buf,Va13);A13=buf[0];
//     _mm_storeu_ps(buf,Va23);A23=buf[0];
//     _mm_storeu_ps(buf,Va33);A33=buf[0];
//     std::cout<<"Vector A (after multiplying with V) ="<<std::endl;
//     std::cout<<std::setw(12)<<A11<<"  "<<std::setw(12)<<A12<<"  "<<std::setw(12)<<A13<<std::endl;
//     std::cout<<std::setw(12)<<A21<<"  "<<std::setw(12)<<A22<<"  "<<std::setw(12)<<A23<<std::endl;
//     std::cout<<std::setw(12)<<A31<<"  "<<std::setw(12)<<A32<<"  "<<std::setw(12)<<A33<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Va11);A11=buf[0];
//     _mm256_storeu_ps(buf,Va21);A21=buf[0];
//     _mm256_storeu_ps(buf,Va31);A31=buf[0];
//     _mm256_storeu_ps(buf,Va12);A12=buf[0];
//     _mm256_storeu_ps(buf,Va22);A22=buf[0];
//     _mm256_storeu_ps(buf,Va32);A32=buf[0];
//     _mm256_storeu_ps(buf,Va13);A13=buf[0];
//     _mm256_storeu_ps(buf,Va23);A23=buf[0];
//     _mm256_storeu_ps(buf,Va33);A33=buf[0];
//     std::cout<<"Vector A (after multiplying with V) ="<<std::endl;
//     std::cout<<std::setw(12)<<A11<<"  "<<std::setw(12)<<A12<<"  "<<std::setw(12)<<A13<<std::endl;
//     std::cout<<std::setw(12)<<A21<<"  "<<std::setw(12)<<A22<<"  "<<std::setw(12)<<A23<<std::endl;
//     std::cout<<std::setw(12)<<A31<<"  "<<std::setw(12)<<A32<<"  "<<std::setw(12)<<A33<<std::endl;
// #endif
// #endif

} // End block : Conjugation with V

} // End block : Scope of qV (if not maintained)

//###########################################################
// Permute columns such that the singular values are sorted
//###########################################################

Vtmp1=Va11.mul(Va11);
Vtmp4=Va21.mul(Va21);
Vtmp1=Vtmp1.add(&Vtmp4);
Vtmp4=Va31.mul(Va31);
Vtmp1=Vtmp1.add(&Vtmp4);

Vtmp2=Va12.mul(Va12);
Vtmp4=Va22.mul(Va22);
Vtmp2=Vtmp2.add(&Vtmp4);
Vtmp4=Va32.mul(Va32);
Vtmp2=Vtmp2.add(&Vtmp4);

Vtmp3=Va13.mul(Va13);
Vtmp4=Va23.mul(Va23);
Vtmp3=Vtmp3.add(&Vtmp4);
Vtmp4=Va33.mul(Va33);
Vtmp3=Vtmp3.add(&Vtmp4);

// Swap columns 1-2 if necessary

Mtmp=Vtmp1.cmplt(&Vtmp2);
Vtmp4=VType::maskz(&Mtmp, &Vone_mask);
Vtmp5=Va11.xor(Va12);
Vtmp5=Vtmp5.and(&Vtmp4);
*Va11=Va11.xor(&Vtmp5);
*Va12=Va12.xor(&Vtmp5);

Vtmp5=Va21.xor(Va22);
Vtmp5=Vtmp5.and(&Vtmp4);
*Va21=Va21.xor(&Vtmp5);
*Va22=Va22.xor(&Vtmp5);

Vtmp5=Va31.xor(Va32);
Vtmp5=Vtmp5.and(&Vtmp4);
*Va31=Va31.xor(&Vtmp5);
*Va32=Va32.xor(&Vtmp5);

if compute_v_as_matrix {
Vtmp5=Vv11.xor(Vv12);
Vtmp5=Vtmp5.and(&Vtmp4);
*Vv11=Vv11.xor(&Vtmp5);
*Vv12=Vv12.xor(&Vtmp5);

Vtmp5=Vv21.xor(Vv22);
Vtmp5=Vtmp5.and(&Vtmp4);
*Vv21=Vv21.xor(&Vtmp5);
*Vv22=Vv22.xor(&Vtmp5);

Vtmp5=Vv31.xor(Vv32);
Vtmp5=Vtmp5.and(&Vtmp4);
*Vv31=Vv31.xor(&Vtmp5);
*Vv32=Vv32.xor(&Vtmp5);
}

Vtmp5=Vtmp1.xor(&Vtmp2);
Vtmp5=Vtmp5.and(&Vtmp4);
Vtmp1=Vtmp1.xor(&Vtmp5);
Vtmp2=Vtmp2.xor(&Vtmp5);

// If columns 1-2 have been swapped, negate 2nd column of A and V so that V is still a rotation

Vtmp5=VType::splat(-2.);
Vtmp5=Vtmp5.and(&Vtmp4);
Vtmp4=Vone.clone();
Vtmp4=Vtmp4.add(&Vtmp5);

*Va12=Va12.mul(&Vtmp4);
*Va22=Va22.mul(&Vtmp4);
*Va32=Va32.mul(&Vtmp4);

if compute_v_as_matrix {
*Vv12=Vv12.mul(&Vtmp4);
*Vv22=Vv22.mul(&Vtmp4);
*Vv32=Vv32.mul(&Vtmp4);
}

// If columns 1-2 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)

if compute_v_as_quaternion {
Vtmp4=Vtmp4.mul(&Vone_half);
Vtmp4=Vtmp4.sub(&Vone_half);

Vtmp5=Vtmp4.mul(Vqvvz);
Vtmp5=Vtmp5.add(Vqvs);
*Vqvs=Vqvs.mul(&Vtmp4);
*Vqvvz=Vqvvz.sub(Vqvs);
*Vqvs=Vtmp5;

Vtmp5=Vtmp4.mul(Vqvvx);
Vtmp5=Vtmp5.add(Vqvvy);
*Vqvvy=Vqvvy.mul(&Vtmp4);
*Vqvvx=Vqvvx.sub(Vqvvy);
*Vqvvy=Vtmp5;
}

// Swap columns 1-3 if necessary

Mtmp=Vtmp1.cmplt(&Vtmp3);
Vtmp4=VType::maskz(&Mtmp, &Vone_mask);
Vtmp5=Va11.xor(Va13);
Vtmp5=Vtmp5.and(&Vtmp4);
*Va11=Va11.xor(&Vtmp5);
*Va13=Va13.xor(&Vtmp5);

Vtmp5=Va21.xor(Va23);
Vtmp5=Vtmp5.and(&Vtmp4);
*Va21=Va21.xor(&Vtmp5);
*Va23=Va23.xor(&Vtmp5);

Vtmp5=Va31.xor(Va33);
Vtmp5=Vtmp5.and(&Vtmp4);
*Va31=Va31.xor(&Vtmp5);
*Va33=Va33.xor(&Vtmp5);

if compute_v_as_matrix {
Vtmp5=Vv11.xor(Vv13);
Vtmp5=Vtmp5.and(&Vtmp4);
*Vv11=Vv11.xor(&Vtmp5);
*Vv13=Vv13.xor(&Vtmp5);

Vtmp5=Vv21.xor(Vv23);
Vtmp5=Vtmp5.and(&Vtmp4);
*Vv21=Vv21.xor(&Vtmp5);
*Vv23=Vv23.xor(&Vtmp5);

Vtmp5=Vv31.xor(Vv33);
Vtmp5=Vtmp5.and(&Vtmp4);
*Vv31=Vv31.xor(&Vtmp5);
*Vv33=Vv33.xor(&Vtmp5);
}

Vtmp5=Vtmp1.xor(&Vtmp3);
Vtmp5=Vtmp5.and(&Vtmp4);
// Vtmp1=Vtmp1.xor(&Vtmp5);
Vtmp3=Vtmp3.xor(&Vtmp5);

// If columns 1-3 have been swapped, negate 1st column of A and V so that V is still a rotation

Vtmp5=VType::splat(-2.);
Vtmp5=Vtmp5.and(&Vtmp4);
Vtmp4=Vone.clone();
Vtmp4=Vtmp4.add(&Vtmp5);

*Va11=Va11.mul(&Vtmp4);
*Va21=Va21.mul(&Vtmp4);
*Va31=Va31.mul(&Vtmp4);

if compute_v_as_matrix  {
*Vv11=Vv11.mul(&Vtmp4);
*Vv21=Vv21.mul(&Vtmp4);
*Vv31=Vv31.mul(&Vtmp4);
}

// If columns 1-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)

if compute_v_as_quaternion {
Vtmp4=Vtmp4.mul(&Vone_half);
Vtmp4=Vtmp4.sub(&Vone_half);

Vtmp5=Vtmp4.mul(Vqvvy);
Vtmp5=Vtmp5.add(Vqvs);
*Vqvs=Vqvs.mul(&Vtmp4);
*Vqvvy=Vqvvy.sub(Vqvs);
*Vqvs=Vtmp5;

Vtmp5=Vtmp4.mul(Vqvvz);
Vtmp5=Vtmp5.add(Vqvvx);
*Vqvvx=Vqvvx.mul(&Vtmp4);
*Vqvvz=Vqvvz.sub(Vqvvx);
*Vqvvx=Vtmp5;
}

// Swap columns 2-3 if necessary

Mtmp=Vtmp2.cmplt(&Vtmp3);
Vtmp4=VType::maskz(&Mtmp, &Vone_mask);
Vtmp5=Va12.xor(Va13);
Vtmp5=Vtmp5.and(&Vtmp4);
*Va12=Va12.xor(&Vtmp5);
*Va13=Va13.xor(&Vtmp5);

Vtmp5=Va22.xor(Va23);
Vtmp5=Vtmp5.and(&Vtmp4);
*Va22=Va22.xor(&Vtmp5);
*Va23=Va23.xor(&Vtmp5);

Vtmp5=Va32.xor(Va33);
Vtmp5=Vtmp5.and(&Vtmp4);
*Va32=Va32.xor(&Vtmp5);
*Va33=Va33.xor(&Vtmp5);

if compute_v_as_matrix {
Vtmp5=Vv12.xor(Vv13);
Vtmp5=Vtmp5.and(&Vtmp4);
*Vv12=Vv12.xor(&Vtmp5);
*Vv13=Vv13.xor(&Vtmp5);

Vtmp5=Vv22.xor(Vv23);
Vtmp5=Vtmp5.and(&Vtmp4);
*Vv22=Vv22.xor(&Vtmp5);
*Vv23=Vv23.xor(&Vtmp5);

Vtmp5=Vv32.xor(Vv33);
Vtmp5=Vtmp5.and(&Vtmp4);
*Vv32=Vv32.xor(&Vtmp5);
*Vv33=Vv33.xor(&Vtmp5);
}

// Vtmp5=Vtmp2.xor(&Vtmp3);
// Vtmp5=Vtmp5.and(&Vtmp4);
// Vtmp2=Vtmp2.xor(&Vtmp5);
// Vtmp3=Vtmp3.xor(&Vtmp5);

// If columns 2-3 have been swapped, negate 3rd column of A and V so that V is still a rotation

Vtmp5=VType::splat(-2.);
Vtmp5=Vtmp5.and(&Vtmp4);
Vtmp4=Vone.clone();
Vtmp4=Vtmp4.add(&Vtmp5);

*Va13=Va13.mul(&Vtmp4);
*Va23=Va23.mul(&Vtmp4);
*Va33=Va33.mul(&Vtmp4);

if compute_v_as_matrix {
*Vv13=Vv13.mul(&Vtmp4);
*Vv23=Vv23.mul(&Vtmp4);
*Vv33=Vv33.mul(&Vtmp4);
}

// If columns 2-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)

if compute_v_as_quaternion {
Vtmp4=Vtmp4.mul(&Vone_half);
Vtmp4=Vtmp4.sub(&Vone_half);

Vtmp5=Vtmp4.mul(Vqvvx);
Vtmp5=Vtmp5.add(Vqvs);
*Vqvs=Vqvs.mul(&Vtmp4);
*Vqvvx=Vqvvx.sub(Vqvs);
*Vqvs=Vtmp5.clone();

Vtmp5=Vtmp4.mul(Vqvvy);
Vtmp5=Vtmp5.add(Vqvvz);
*Vqvvz=Vqvvz.mul(&Vtmp4);
*Vqvvy=Vqvvy.sub(Vqvvz);
*Vqvvz=Vtmp5.clone();
}

// #ifdef COMPUTE_V_AS_MATRIX
// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar V ="<<std::endl;
//     std::cout<<std::setw(12)<<Sv11.f<<"  "<<std::setw(12)<<Sv12.f<<"  "<<std::setw(12)<<Sv13.f<<std::endl;
//     std::cout<<std::setw(12)<<Sv21.f<<"  "<<std::setw(12)<<Sv22.f<<"  "<<std::setw(12)<<Sv23.f<<std::endl;
//     std::cout<<std::setw(12)<<Sv31.f<<"  "<<std::setw(12)<<Sv32.f<<"  "<<std::setw(12)<<Sv33.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Vv11);V11=buf[0];
//     _mm_storeu_ps(buf,Vv21);V21=buf[0];
//     _mm_storeu_ps(buf,Vv31);V31=buf[0];
//     _mm_storeu_ps(buf,Vv12);V12=buf[0];
//     _mm_storeu_ps(buf,Vv22);V22=buf[0];
//     _mm_storeu_ps(buf,Vv32);V32=buf[0];
//     _mm_storeu_ps(buf,Vv13);V13=buf[0];
//     _mm_storeu_ps(buf,Vv23);V23=buf[0];
//     _mm_storeu_ps(buf,Vv33);V33=buf[0];
//     std::cout<<"Vector V ="<<std::endl;
//     std::cout<<std::setw(12)<<V11<<"  "<<std::setw(12)<<V12<<"  "<<std::setw(12)<<V13<<std::endl;
//     std::cout<<std::setw(12)<<V21<<"  "<<std::setw(12)<<V22<<"  "<<std::setw(12)<<V23<<std::endl;
//     std::cout<<std::setw(12)<<V31<<"  "<<std::setw(12)<<V32<<"  "<<std::setw(12)<<V33<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Vv11);V11=buf[0];
//     _mm256_storeu_ps(buf,Vv21);V21=buf[0];
//     _mm256_storeu_ps(buf,Vv31);V31=buf[0];
//     _mm256_storeu_ps(buf,Vv12);V12=buf[0];
//     _mm256_storeu_ps(buf,Vv22);V22=buf[0];
//     _mm256_storeu_ps(buf,Vv32);V32=buf[0];
//     _mm256_storeu_ps(buf,Vv13);V13=buf[0];
//     _mm256_storeu_ps(buf,Vv23);V23=buf[0];
//     _mm256_storeu_ps(buf,Vv33);V33=buf[0];
//     std::cout<<"Vector V ="<<std::endl;
//     std::cout<<std::setw(12)<<V11<<"  "<<std::setw(12)<<V12<<"  "<<std::setw(12)<<V13<<std::endl;
//     std::cout<<std::setw(12)<<V21<<"  "<<std::setw(12)<<V22<<"  "<<std::setw(12)<<V23<<std::endl;
//     std::cout<<std::setw(12)<<V31<<"  "<<std::setw(12)<<V32<<"  "<<std::setw(12)<<V33<<std::endl;
// #endif
// #endif
// #endif

// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar A (after multiplying with V) ="<<std::endl;
//     std::cout<<std::setw(12)<<Sa11.f<<"  "<<std::setw(12)<<Sa12.f<<"  "<<std::setw(12)<<Sa13.f<<std::endl;
//     std::cout<<std::setw(12)<<Sa21.f<<"  "<<std::setw(12)<<Sa22.f<<"  "<<std::setw(12)<<Sa23.f<<std::endl;
//     std::cout<<std::setw(12)<<Sa31.f<<"  "<<std::setw(12)<<Sa32.f<<"  "<<std::setw(12)<<Sa33.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Va11);A11=buf[0];
//     _mm_storeu_ps(buf,Va21);A21=buf[0];
//     _mm_storeu_ps(buf,Va31);A31=buf[0];
//     _mm_storeu_ps(buf,Va12);A12=buf[0];
//     _mm_storeu_ps(buf,Va22);A22=buf[0];
//     _mm_storeu_ps(buf,Va32);A32=buf[0];
//     _mm_storeu_ps(buf,Va13);A13=buf[0];
//     _mm_storeu_ps(buf,Va23);A23=buf[0];
//     _mm_storeu_ps(buf,Va33);A33=buf[0];
//     std::cout<<"Vector A (after multiplying with V) ="<<std::endl;
//     std::cout<<std::setw(12)<<A11<<"  "<<std::setw(12)<<A12<<"  "<<std::setw(12)<<A13<<std::endl;
//     std::cout<<std::setw(12)<<A21<<"  "<<std::setw(12)<<A22<<"  "<<std::setw(12)<<A23<<std::endl;
//     std::cout<<std::setw(12)<<A31<<"  "<<std::setw(12)<<A32<<"  "<<std::setw(12)<<A33<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Va11);A11=buf[0];
//     _mm256_storeu_ps(buf,Va21);A21=buf[0];
//     _mm256_storeu_ps(buf,Va31);A31=buf[0];
//     _mm256_storeu_ps(buf,Va12);A12=buf[0];
//     _mm256_storeu_ps(buf,Va22);A22=buf[0];
//     _mm256_storeu_ps(buf,Va32);A32=buf[0];
//     _mm256_storeu_ps(buf,Va13);A13=buf[0];
//     _mm256_storeu_ps(buf,Va23);A23=buf[0];
//     _mm256_storeu_ps(buf,Va33);A33=buf[0];
//     std::cout<<"Vector A (after multiplying with V) ="<<std::endl;
//     std::cout<<std::setw(12)<<A11<<"  "<<std::setw(12)<<A12<<"  "<<std::setw(12)<<A13<<std::endl;
//     std::cout<<std::setw(12)<<A21<<"  "<<std::setw(12)<<A22<<"  "<<std::setw(12)<<A23<<std::endl;
//     std::cout<<std::setw(12)<<A31<<"  "<<std::setw(12)<<A32<<"  "<<std::setw(12)<<A33<<std::endl;
// #endif
// #endif

//###########################################################
// Re-normalize quaternion for matrix V
//###########################################################

if compute_v_as_quaternion  {
Vtmp2=Vqvs.mul(Vqvs);
Vtmp1=Vqvvx.mul(Vqvvx);
Vtmp2=Vtmp1.add(&Vtmp2);
Vtmp1=Vqvvy.mul(Vqvvy);
Vtmp2=Vtmp1.add(&Vtmp2);
Vtmp1=Vqvvz.mul(Vqvvz);
Vtmp2=Vtmp1.add(&Vtmp2);
Vtmp1=Vtmp2.rsqrt();

if perform_strict_quaternion_renormalization {
Vtmp4=Vtmp1.mul(&Vone_half);
Vtmp3=Vtmp1.mul(&Vtmp4);
Vtmp3=Vtmp1.mul(&Vtmp3);
Vtmp3=Vtmp2.mul(&Vtmp3);
Vtmp1=Vtmp1.add(&Vtmp4);
Vtmp1=Vtmp1.sub(&Vtmp3);
}

*Vqvs=Vqvs.mul(&Vtmp1);
*Vqvvx=Vqvvx.mul(&Vtmp1);
*Vqvvy=Vqvvy.mul(&Vtmp1);
*Vqvvz=Vqvvz.mul(&Vtmp1);

// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar qV ="<<std::endl;
//     std::cout<<std::setw(12)<<Sqvs.f<<"  "<<std::setw(12)<<Sqvvx.f<<"  "<<std::setw(12)<<Sqvvy.f<<"  "<<std::setw(12)<<Sqvvz.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Vqvs);QVS=buf[0];
//     _mm_storeu_ps(buf,Vqvvx);QVVX=buf[0];
//     _mm_storeu_ps(buf,Vqvvy);QVVY=buf[0];
//     _mm_storeu_ps(buf,Vqvvz);QVVZ=buf[0];
//     std::cout<<"Vector qV ="<<std::endl;
//     std::cout<<std::setw(12)<<QVS<<"  "<<std::setw(12)<<QVVX<<"  "<<std::setw(12)<<QVVY<<"  "<<std::setw(12)<<QVVZ<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Vqvs);QVS=buf[0];
//     _mm256_storeu_ps(buf,Vqvvx);QVVX=buf[0];
//     _mm256_storeu_ps(buf,Vqvvy);QVVY=buf[0];
//     _mm256_storeu_ps(buf,Vqvvz);QVVZ=buf[0];
//     std::cout<<"Vector qV ="<<std::endl;
//     std::cout<<std::setw(12)<<QVS<<"  "<<std::setw(12)<<QVVX<<"  "<<std::setw(12)<<QVVY<<"  "<<std::setw(12)<<QVVZ<<std::endl;
// #endif
// #endif
}

//###########################################################
// Construct QR factorization of A*V (=U*D) using Givens rotations
//###########################################################

if compute_u_as_matrix {
*Vu11=Vone.clone();
*Vu21=Vu21.xor(Vu21);
*Vu31=Vu31.xor(Vu31);
*Vu12=Vu12.xor(Vu12);
*Vu22=Vone.clone();
*Vu32=Vu32.xor(Vu32);
*Vu13=Vu13.xor(Vu13);
*Vu23=Vu23.xor(Vu23);
*Vu33=Vone.clone();
}

if compute_u_as_quaternion  {
*Vqus=Vone.clone();
*Vquvx=Vquvx.xor(Vquvx);
*Vquvy=Vquvy.xor(Vquvy);
*Vquvz=Vquvz.xor(Vquvz);
}

// First Givens rotation
givens_qr_factorization!(
    VType,
    compute_u_as_matrix,
    Vfour_gamma_squared,
    Vsine_pi_over_eight,
    Vcosine_pi_over_eight, 
    Vone_half,
    Vone,
    Vtiny_number,
    Vsmall_number,
    Vc,
    Vs,
    Vch,
    Vsh,
    Vtmp1,
    Vtmp2,
    Vtmp3,
    Vtmp4,
    Vtmp5,
    Va11,
    Va21,
    Va11,
    Va21,
    Va12,
    Va22,
    Va13,
    Va23,
    Vu11,
    Vu12,
    Vu21,
    Vu22,
    Vu31,
    Vu32,
    Mtmp
);

// Update quaternion representation of U

if compute_u_as_quaternion {
*Vqus=Vch.clone();
*Vquvx=Vquvx.xor(&Vquvx);
*Vquvy=Vquvy.xor(&Vquvy);
*Vquvz=Vsh.clone();
}

// Second Givens rotation
givens_qr_factorization!(
    VType,
    compute_u_as_matrix,
    Vfour_gamma_squared,
    Vsine_pi_over_eight,
    Vcosine_pi_over_eight, 
    Vone_half,
    Vone,
    Vtiny_number,
    Vsmall_number,
    Vc,
    Vs,
    Vch,
    Vsh,
    Vtmp1,
    Vtmp2,
    Vtmp3,
    Vtmp4,
    Vtmp5,
    Va11,
    Va31,
    Va11,
    Va31,
    Va12,
    Va32,
    Va13,
    Va33,
    Vu11,
    Vu13,
    Vu21,
    Vu23,
    Vu31,
    Vu33,
    Mtmp
);

// Update quaternion representation of U

if compute_u_as_quaternion {
*Vquvx=Vsh.mul(Vquvz);
Vsh=Vsh.mul(Vqus);
*Vquvy=Vquvy.sub(&Vsh);
*Vqus=Vch.mul(Vqus);
*Vquvz=Vch.mul(Vquvz);
}

// Third Givens rotation
givens_qr_factorization!(
    VType,
    compute_u_as_matrix,
    Vfour_gamma_squared,
    Vsine_pi_over_eight,
    Vcosine_pi_over_eight, 
    Vone_half,
    Vone,
    Vtiny_number,
    Vsmall_number,
    Vc,
    Vs,
    Vch,
    Vsh,
    Vtmp1,
    Vtmp2,
    Vtmp3,
    Vtmp4,
    Vtmp5,
    Va22,
    Va32,
    Va21,
    Va31,
    Va22,
    Va32,
    Va23,
    Va33,
    Vu12,
    Vu13,
    Vu22,
    Vu23,
    Vu32,
    Vu33,
    Mtmp
);

// Update quaternion representation of U

if compute_u_as_quaternion {
Vtmp1=Vsh.mul(Vquvx);
Vtmp2=Vsh.mul(Vquvy);
Vtmp3=Vsh.mul(Vquvz);
Vsh=Vsh.mul(Vqus);
*Vqus=Vch.mul(Vqus);
*Vquvx=Vch.mul(Vquvx);
*Vquvy=Vch.mul(Vquvy);
*Vquvz=Vch.mul(Vquvz);
*Vquvx=Vquvx.add(&Vsh);
*Vqus=Vqus.sub(&Vtmp1);
*Vquvy=Vquvy.add(&Vtmp3);
*Vquvz=Vquvz.sub(&Vtmp2);
}

// #ifdef COMPUTE_U_AS_MATRIX
// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar U ="<<std::endl;
//     std::cout<<std::setw(12)<<Su11.f<<"  "<<std::setw(12)<<Su12.f<<"  "<<std::setw(12)<<Su13.f<<std::endl;
//     std::cout<<std::setw(12)<<Su21.f<<"  "<<std::setw(12)<<Su22.f<<"  "<<std::setw(12)<<Su23.f<<std::endl;
//     std::cout<<std::setw(12)<<Su31.f<<"  "<<std::setw(12)<<Su32.f<<"  "<<std::setw(12)<<Su33.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Vu11);U11=buf[0];
//     _mm_storeu_ps(buf,Vu21);U21=buf[0];
//     _mm_storeu_ps(buf,Vu31);U31=buf[0];
//     _mm_storeu_ps(buf,Vu12);U12=buf[0];
//     _mm_storeu_ps(buf,Vu22);U22=buf[0];
//     _mm_storeu_ps(buf,Vu32);U32=buf[0];
//     _mm_storeu_ps(buf,Vu13);U13=buf[0];
//     _mm_storeu_ps(buf,Vu23);U23=buf[0];
//     _mm_storeu_ps(buf,Vu33);U33=buf[0];
//     std::cout<<"Vector U ="<<std::endl;
//     std::cout<<std::setw(12)<<U11<<"  "<<std::setw(12)<<U12<<"  "<<std::setw(12)<<U13<<std::endl;
//     std::cout<<std::setw(12)<<U21<<"  "<<std::setw(12)<<U22<<"  "<<std::setw(12)<<U23<<std::endl;
//     std::cout<<std::setw(12)<<U31<<"  "<<std::setw(12)<<U32<<"  "<<std::setw(12)<<U33<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Vu11);U11=buf[0];
//     _mm256_storeu_ps(buf,Vu21);U21=buf[0];
//     _mm256_storeu_ps(buf,Vu31);U31=buf[0];
//     _mm256_storeu_ps(buf,Vu12);U12=buf[0];
//     _mm256_storeu_ps(buf,Vu22);U22=buf[0];
//     _mm256_storeu_ps(buf,Vu32);U32=buf[0];
//     _mm256_storeu_ps(buf,Vu13);U13=buf[0];
//     _mm256_storeu_ps(buf,Vu23);U23=buf[0];
//     _mm256_storeu_ps(buf,Vu33);U33=buf[0];
//     std::cout<<"Vector U ="<<std::endl;
//     std::cout<<std::setw(12)<<U11<<"  "<<std::setw(12)<<U12<<"  "<<std::setw(12)<<U13<<std::endl;
//     std::cout<<std::setw(12)<<U21<<"  "<<std::setw(12)<<U22<<"  "<<std::setw(12)<<U23<<std::endl;
//     std::cout<<std::setw(12)<<U31<<"  "<<std::setw(12)<<U32<<"  "<<std::setw(12)<<U33<<std::endl;
// #endif
// #endif
// #endif

// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar A (after multiplying with U-transpose and V) ="<<std::endl;
//     std::cout<<std::setw(12)<<Sa11.f<<"  "<<std::setw(12)<<Sa12.f<<"  "<<std::setw(12)<<Sa13.f<<std::endl;
//     std::cout<<std::setw(12)<<Sa21.f<<"  "<<std::setw(12)<<Sa22.f<<"  "<<std::setw(12)<<Sa23.f<<std::endl;
//     std::cout<<std::setw(12)<<Sa31.f<<"  "<<std::setw(12)<<Sa32.f<<"  "<<std::setw(12)<<Sa33.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Va11);A11=buf[0];
//     _mm_storeu_ps(buf,Va21);A21=buf[0];
//     _mm_storeu_ps(buf,Va31);A31=buf[0];
//     _mm_storeu_ps(buf,Va12);A12=buf[0];
//     _mm_storeu_ps(buf,Va22);A22=buf[0];
//     _mm_storeu_ps(buf,Va32);A32=buf[0];
//     _mm_storeu_ps(buf,Va13);A13=buf[0];
//     _mm_storeu_ps(buf,Va23);A23=buf[0];
//     _mm_storeu_ps(buf,Va33);A33=buf[0];
//     std::cout<<"Vector A (after multiplying with U-transpose and V) ="<<std::endl;
//     std::cout<<std::setw(12)<<A11<<"  "<<std::setw(12)<<A12<<"  "<<std::setw(12)<<A13<<std::endl;
//     std::cout<<std::setw(12)<<A21<<"  "<<std::setw(12)<<A22<<"  "<<std::setw(12)<<A23<<std::endl;
//     std::cout<<std::setw(12)<<A31<<"  "<<std::setw(12)<<A32<<"  "<<std::setw(12)<<A33<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Va11);A11=buf[0];
//     _mm256_storeu_ps(buf,Va21);A21=buf[0];
//     _mm256_storeu_ps(buf,Va31);A31=buf[0];
//     _mm256_storeu_ps(buf,Va12);A12=buf[0];
//     _mm256_storeu_ps(buf,Va22);A22=buf[0];
//     _mm256_storeu_ps(buf,Va32);A32=buf[0];
//     _mm256_storeu_ps(buf,Va13);A13=buf[0];
//     _mm256_storeu_ps(buf,Va23);A23=buf[0];
//     _mm256_storeu_ps(buf,Va33);A33=buf[0];
//     std::cout<<"Vector A (after multiplying with U-transpose and V) ="<<std::endl;
//     std::cout<<std::setw(12)<<A11<<"  "<<std::setw(12)<<A12<<"  "<<std::setw(12)<<A13<<std::endl;
//     std::cout<<std::setw(12)<<A21<<"  "<<std::setw(12)<<A22<<"  "<<std::setw(12)<<A23<<std::endl;
//     std::cout<<std::setw(12)<<A31<<"  "<<std::setw(12)<<A32<<"  "<<std::setw(12)<<A33<<std::endl;
// #endif
// #endif

// #ifdef COMPUTE_U_AS_QUATERNION
// #ifdef PRINT_DEBUGGING_OUTPUT
// #ifdef USE_SCALAR_IMPLEMENTATION
//     std::cout<<"Scalar qU ="<<std::endl;
//     std::cout<<std::setw(12)<<Squs.f<<"  "<<std::setw(12)<<Squvx.f<<"  "<<std::setw(12)<<Squvy.f<<"  "<<std::setw(12)<<Squvz.f<<std::endl;
// #endif
// #ifdef USE_SSE_IMPLEMENTATION
//     _mm_storeu_ps(buf,Vqus);QUS=buf[0];
//     _mm_storeu_ps(buf,Vquvx);QUVX=buf[0];
//     _mm_storeu_ps(buf,Vquvy);QUVY=buf[0];
//     _mm_storeu_ps(buf,Vquvz);QUVZ=buf[0];
//     std::cout<<"Vector qU ="<<std::endl;
//     std::cout<<std::setw(12)<<QUS<<"  "<<std::setw(12)<<QUVX<<"  "<<std::setw(12)<<QUVY<<"  "<<std::setw(12)<<QUVZ<<std::endl;
// #endif
// #ifdef USE_AVX_IMPLEMENTATION
//     _mm256_storeu_ps(buf,Vqus);QUS=buf[0];
//     _mm256_storeu_ps(buf,Vquvx);QUVX=buf[0];
//     _mm256_storeu_ps(buf,Vquvy);QUVY=buf[0];
//     _mm256_storeu_ps(buf,Vquvz);QUVZ=buf[0];
//     std::cout<<"Vector qU ="<<std::endl;
//     std::cout<<std::setw(12)<<QUS<<"  "<<std::setw(12)<<QUVX<<"  "<<std::setw(12)<<QUVY<<"  "<<std::setw(12)<<QUVZ<<std::endl;
// #endif
// #endif
// #endif

}

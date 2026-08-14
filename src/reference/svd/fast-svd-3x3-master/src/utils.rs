
#![allow(non_snake_case)]
use crate::{svd::svd, traits::SVDCompatible};

#[inline(always)]
pub fn svd_mat<VType: SVDCompatible>(
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
) {
    // Compiler wil get rid of this
    let mut Vqus = VType::default();
    let mut Vquvx = VType::default();
    let mut Vquvy = VType::default();
    let mut Vquvz = VType::default();
    let mut vqvs = VType::default();
    let mut vqvvx = VType::default();
    let mut vqvvy = VType::default();
    let mut vqvvz = VType::default();
    svd(
        true,
        true,
        false,
        false,
        false,
        false,
        Va11,
        Va12,
        Va13,
        Va21,
        Va22,
        Va23,
        Va31,
        Va32,
        Va33,
        Vu11,
        Vu12,
        Vu13,
        Vu21,
        Vu22,
        Vu23,
        Vu31,
        Vu32,
        Vu33,
        Vv11,
        Vv12,
        Vv13,
        Vv21,
        Vv22,
        Vv23,
        Vv31,
        Vv32,
        Vv33,
        &mut Vqus,
        &mut Vquvx,
        &mut Vquvy,
        &mut Vquvz,
        &mut vqvs,
        &mut vqvvx,
        &mut vqvvy,
        &mut vqvvz
    );
}
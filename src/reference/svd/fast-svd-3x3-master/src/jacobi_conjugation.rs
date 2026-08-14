
#[macro_export]
macro_rules! jacobi_conjugation {
(
    $VType: ty,
    $use_accurate_rsqrt_in_jacobi_conjugation: expr,
    $Vfour_gamma_squared: expr,
    $Vsine_pi_over_eight: expr,
    $Vcosine_pi_over_eight: expr,
    $Vone_half: expr,
    $Vone: expr,
    $Vtiny_number: expr,
    $_: expr,
    $Vc: expr,
    $Vs: expr,
    $Vch: expr,
    $Vsh: expr,
    $Vtmp1: expr,
    $Vtmp2: expr,
    $Vtmp3: expr,
    $Vtmp4: expr,
    $Vtmp5: expr,
    $Vqvs: expr,
    $Vqvvx: expr,
    $Vqvvy: expr,
    $Vqvvz: expr,
    $VS11: expr,
    $VS21: expr,
    $VS31: expr,
    $VS22: expr,
    $VS32: expr,
    $VS33: expr,
    $VQVVX: expr,
    $VQVVY: expr,
    $VQVVZ: expr,
    $VTMP1: expr,
    $VTMP2: expr,
    $VTMP3: expr,
    $Mtmp: expr
) => {
$Vsh=$VS21.mul(&$Vone_half);
$Vtmp5=$VS11.sub(&$VS22);

$Vtmp2=$Vsh.mul(&$Vsh);
$Mtmp=$Vtmp2.cmpge(&$Vtiny_number);
$Vsh=<$VType>::maskz(&$Mtmp,&$Vsh);
$Vch=<$VType>::blend(&$Mtmp,&$Vone,&$Vtmp5);



$Vtmp1=$Vsh.mul(&$Vsh);
$Vtmp2=$Vch.mul(&$Vch);
$Vtmp3=$Vtmp1.add(&$Vtmp2);
$Vtmp4=$Vtmp3.rsqrt();

if $use_accurate_rsqrt_in_jacobi_conjugation {
$Vs=$Vtmp4.mul(&$Vone_half);
$Vc=$Vtmp4.mul(&$Vs);
$Vc=$Vtmp4.mul(&$Vc);
$Vc=$Vtmp3.mul(&$Vc);
$Vtmp4=$Vtmp4.add(&$Vs);
$Vtmp4=$Vtmp4.sub(&$Vc);
}

$Vsh=$Vtmp4.mul(&$Vsh);
$Vch=$Vtmp4.mul(&$Vch);

$Vtmp1=$Vfour_gamma_squared.mul(&$Vtmp1);
$Mtmp=$Vtmp2.cmple(&$Vtmp1);

$Vsh=<$VType>::blend(&$Mtmp,&$Vsh,&$Vsine_pi_over_eight);


$Vch=<$VType>::blend(&$Mtmp,&$Vch,&$Vcosine_pi_over_eight);



$Vtmp1=$Vsh.mul(&$Vsh);
$Vtmp2=$Vch.mul(&$Vch);
$Vc=$Vtmp2.sub(&$Vtmp1);
$Vs=$Vch.mul(&$Vsh);
$Vs=$Vs.add(&$Vs);

//###########################################################
// Perform the actual Givens conjugation
//###########################################################

if $use_accurate_rsqrt_in_jacobi_conjugation {
$Vtmp3=$Vtmp1.add(&$Vtmp2);
$VS33=$VS33.mul(&$Vtmp3);
$VS31=$VS31.mul(&$Vtmp3);
$VS32=$VS32.mul(&$Vtmp3);
$VS33=$VS33.mul(&$Vtmp3);
}

$Vtmp1=$Vs.mul(&$VS31);
$Vtmp2=$Vs.mul(&$VS32);
$VS31=$Vc.mul(&$VS31);
$VS32=$Vc.mul(&$VS32);
$VS31=$Vtmp2.add(&$VS31);
$VS32=$VS32.sub(&$Vtmp1);

$Vtmp2=$Vs.mul(&$Vs);         
$Vtmp1=$VS22.mul(&$Vtmp2);
$Vtmp3=$VS11.mul(&$Vtmp2);
$Vtmp4=$Vc.mul(&$Vc);
$VS11=$VS11.mul(&$Vtmp4);
$VS22=$VS22.mul(&$Vtmp4);
$VS11=$VS11.add(&$Vtmp1);
$VS22=$VS22.add(&$Vtmp3);
$Vtmp4=$Vtmp4.sub(&$Vtmp2);
$Vtmp2=$VS21.add(&$VS21);
$VS21=$VS21.mul(&$Vtmp4);
$Vtmp4=$Vc.mul(&$Vs);
$Vtmp2=$Vtmp2.mul(&$Vtmp4);
$Vtmp5=$Vtmp5.mul(&$Vtmp4);
$VS11=$VS11.add(&$Vtmp2);
$VS21=$VS21.sub(&$Vtmp5);
$VS22=$VS22.sub(&$Vtmp2);

//###########################################################
// Compute the cumulative rotation, in quaternion form
//###########################################################

$Vtmp1=$Vsh.mul($Vqvvx);
$Vtmp2=$Vsh.mul($Vqvvy);
$Vtmp3=$Vsh.mul($Vqvvz);
$Vsh=$Vsh.mul($Vqvs);

*$Vqvs=$Vch.mul($Vqvs);
*$Vqvvx=$Vch.mul($Vqvvx);
*$Vqvvy=$Vch.mul($Vqvvy);
*$Vqvvz=$Vch.mul($Vqvvz);

*$VQVVZ=$VQVVZ.add(&$Vsh);
*$Vqvs=$Vqvs.sub(&$VTMP3);
*$VQVVX=$VQVVX.add(&$VTMP2);
*$VQVVY=$VQVVY.sub(&$VTMP1);
}
}
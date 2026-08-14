

#[macro_export]
macro_rules! givens_qr_factorization {
(
    $VType: ty,
    $compute_u_as_matrix: expr,
    $Vfour_gamma_squared: expr,
    $Vsine_pi_over_eight: expr,
    $Vcosine_pi_over_eight: expr,
    $Vone_half: expr,
    $Vone: expr,
    $Vtiny_number: expr,
    $Vsmall_number: expr,
    $Vc: expr,
    $Vs: expr,
    $Vch: expr,
    $Vsh: expr,
    $Vtmp1: expr,
    $Vtmp2: expr,
    $Vtmp3: expr,
    $Vtmp4: expr,
    $Vtmp5: expr,
    $VAPIVOT: expr,
    $VANPIVOT: expr,
    $VA11: expr,
    $VA21: expr,
    $VA12: expr,
    $VA22: expr,
    $VA13: expr,
    $VA23: expr,
    $VU11: expr,
    $VU12: expr,
    $VU21: expr,
    $VU22: expr,
    $VU31: expr,
    $VU32: expr,
    $Mtmp: expr
) => {

$Vsh=$VANPIVOT.mul($VANPIVOT);
$Mtmp=$Vsh.cmpge(&$Vsmall_number);
$Vsh=<$VType>::maskz(&$Mtmp,$VANPIVOT);

$Vtmp5=$Vtmp5.xor(&$Vtmp5);
$Vch=$Vtmp5.sub($VAPIVOT);
$Vch=$Vch.max($VAPIVOT);
$Vch=$Vch.max(&$Vsmall_number);
$Mtmp=$VAPIVOT.cmpge(&$Vtmp5);

$Vtmp1=$Vch.mul(&$Vch);
$Vtmp2=$Vsh.mul(&$Vsh);
$Vtmp2=$Vtmp1.add(&$Vtmp2);
$Vtmp1=$Vtmp2.rsqrt();

$Vtmp4=$Vtmp1.mul(&$Vone_half);
$Vtmp3=$Vtmp1.mul(&$Vtmp4);
$Vtmp3=$Vtmp1.mul(&$Vtmp3);
$Vtmp3=$Vtmp2.mul(&$Vtmp3);
$Vtmp1=$Vtmp1.add(&$Vtmp4);
$Vtmp1=$Vtmp1.sub(&$Vtmp3);
$Vtmp1=$Vtmp1.mul(&$Vtmp2);

$Vch=$Vch.add(&$Vtmp1);

$Vtmp1=$Vch.clone();
$Vch=<$VType>::blend(&$Mtmp,&$Vsh,&$Vch);
$Vsh=<$VType>::blend(&$Mtmp,&$Vtmp1,&$Vsh);




$Vtmp1=$Vch.mul(&$Vch);
$Vtmp2=$Vsh.mul(&$Vsh);
$Vtmp2=$Vtmp1.add(&$Vtmp2);
$Vtmp1=$Vtmp2.rsqrt();

$Vtmp4=$Vtmp1.mul(&$Vone_half);
$Vtmp3=$Vtmp1.mul(&$Vtmp4);
$Vtmp3=$Vtmp1.mul(&$Vtmp3);
$Vtmp3=$Vtmp2.mul(&$Vtmp3);
$Vtmp1=$Vtmp1.add(&$Vtmp4);
$Vtmp1=$Vtmp1.sub(&$Vtmp3);

$Vch=$Vch.mul(&$Vtmp1);
$Vsh=$Vsh.mul(&$Vtmp1);

$Vc=$Vch.mul(&$Vch);
$Vs=$Vsh.mul(&$Vsh);
$Vc=$Vc.sub(&$Vs);
$Vs=$Vsh.mul(&$Vch);
$Vs=$Vs.add(&$Vs);

//###########################################################
// Rotate matrix A
//###########################################################

$Vtmp1=$Vs.mul($VA11);
$Vtmp2=$Vs.mul($VA21);
*$VA11=$Vc.mul($VA11);
*$VA21=$Vc.mul($VA21);
*$VA11=$VA11.add(&$Vtmp2);
*$VA21=$VA21.sub(&$Vtmp1);

$Vtmp1=$Vs.mul($VA12);
$Vtmp2=$Vs.mul($VA22);
*$VA12=$Vc.mul($VA12);
*$VA22=$Vc.mul($VA22);
*$VA12=$VA12.add(&$Vtmp2);
*$VA22=$VA22.sub(&$Vtmp1);

$Vtmp1=$Vs.mul($VA13);
$Vtmp2=$Vs.mul($VA23);
*$VA13=$Vc.mul($VA13);
*$VA23=$Vc.mul($VA23);
*$VA13=$VA13.add(&$Vtmp2);
*$VA23=$VA23.sub(&$Vtmp1);

//###########################################################
// Update matrix U
//###########################################################

if $compute_u_as_matrix {
$Vtmp1=$Vs.mul($VU11);
$Vtmp2=$Vs.mul($VU12);
*$VU11=$Vc.mul($VU11);
*$VU12=$Vc.mul($VU12);
*$VU11=$VU11.add(&$Vtmp2);
*$VU12=$VU12.sub(&$Vtmp1);

$Vtmp1=$Vs.mul($VU21);
$Vtmp2=$Vs.mul($VU22);
*$VU21=$Vc.mul($VU21);
*$VU22=$Vc.mul($VU22);
*$VU21=$VU21.add(&$Vtmp2);
*$VU22=$VU22.sub(&$Vtmp1);

$Vtmp1=$Vs.mul($VU31);
$Vtmp2=$Vs.mul($VU32);
*$VU31=$Vc.mul($VU31);
*$VU32=$Vc.mul($VU32);
*$VU31=$VU31.add(&$Vtmp2);
*$VU32=$VU32.sub(&$Vtmp1);
}
}
}
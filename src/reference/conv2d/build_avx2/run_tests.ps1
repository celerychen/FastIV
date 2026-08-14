Set-Location "D:\aicode\aicode\conv2d\build_avx2"
$out = ""
foreach ($v in 2,3,4,5,6) {
    $r = & ".\test_v$v.exe" 2>&1 | Out-String
    $out += "########## conv2d_v$v ##########`n" + $r + "`n"
}
$out | Out-File -Encoding UTF8 "D:\aicode\aicode\conv2d\build_avx2\test_results.log"

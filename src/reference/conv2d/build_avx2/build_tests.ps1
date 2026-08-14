$ErrorActionPreference = "Stop"
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
Set-Location "D:\aicode\aicode\conv2d"
$flags = @("/nologo","/O2","/arch:AVX2","/W3")
$ok = $true
foreach ($v in 2,3,4,5,6) {
    Write-Host "=== building conv2d_v$v`_test ==="
    & cl @flags "conv2d_ref.c" "conv2d_v$v.c" "conv2d_v${v}_test.c" "/Fe:build_avx2\test_v$v.exe" "/Fo:build_avx2\" | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "BUILD_FAIL v$v"; $ok = $false }
}
Write-Host "=== building conv2d_avx2_bench ==="
& cl @flags "conv2d_avx2_bench.c" "conv2d_ref.c" "conv2d_v1.c" "conv2d_v2.c" "conv2d_v3.c" "conv2d_v4.c" "conv2d_v5.c" "conv2d_v6.c" "/Fe:build_avx2\bench.exe" "/Fo:build_avx2\" | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD_FAIL bench"; $ok = $false }
if ($ok) { Write-Host "ALL_BUILDS_OK" } else { Write-Host "SOME_BUILD_FAILED" }

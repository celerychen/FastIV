& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 2>&1 | Out-Null

$dir = "D:\aicode\aicode\conv2d\3x1_1x3"
Set-Location $dir

& cl /nologo /O2 /arch:AVX2 /utf-8 /W3 ..\conv2d_ref.c conv2d_3x1_1x3_v2.c conv2d_3x1_1x3_v2_test.c /Fe:build_avx2\test_v2.exe /Fo:build_avx2\ > build_avx2\test_build.log 2>&1
echo "TEST_BUILD_EXIT=$LASTEXITCODE" >> build_avx2\test_build.log

& cl /nologo /O2 /arch:AVX2 /utf-8 /W3 conv2d_3x1_1x3_avx2_bench.c ..\conv2d_ref.c conv2d_3x1_1x3_v1.c conv2d_3x1_1x3_v2.c /Fe:build_avx2\bench.exe /Fo:build_avx2\ > build_avx2\bench_build.log 2>&1
echo "BENCH_BUILD_EXIT=$LASTEXITCODE" >> build_avx2\bench_build.log

if (Test-Path build_avx2\test_v2.exe) {
    & build_avx2\test_v2.exe > build_avx2\test_results.log 2>&1
    echo "TEST_RUN_DONE" >> build_avx2\build.log
} else {
    echo "TEST_EXE_MISSING" >> build_avx2\build.log
}

if (Test-Path build_avx2\bench.exe) {
    & build_avx2\bench.exe > build_avx2\bench_console.log 2>&1
    echo "BENCH_RUN_DONE" >> build_avx2\build.log
} else {
    echo "BENCH_EXE_MISSING" >> build_avx2\build.log
}

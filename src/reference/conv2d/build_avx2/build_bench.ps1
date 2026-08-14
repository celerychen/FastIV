& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation *> "D:\aicode\aicode\conv2d\build_avx2\devshell.log"
Set-Location "D:\aicode\aicode\conv2d"
& cl /nologo /O2 /arch:AVX2 /utf-8 /W3 "conv2d_avx2_bench.c" "conv2d_ref.c" "conv2d_v1.c" "conv2d_v2.c" "conv2d_v3.c" "conv2d_v4.c" "conv2d_v5.c" "conv2d_v6.c" "/Fe:build_avx2\bench.exe" "/Fo:build_avx2\" *> "D:\aicode\aicode\conv2d\build_avx2\bench_build.log"
"EXITCODE=$LASTEXITCODE" | Out-File -Append "D:\aicode\aicode\conv2d\build_avx2\bench_build.log"

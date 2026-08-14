@echo off
call "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" >nul
cd /d D:/aicode/aicode/conv2d
set FLAGS=/nologo /O2 /arch:AVX2 /W3
for %%V in (2 3 4 5 6) do (
  echo === building conv2d_v%%V_test ===
  cl %FLAGS% conv2d_ref.c conv2d_v%%V.c conv2d_v%%V_test.c /Fe:build_avx2\test_v%%V.exe /Fo:build_avx2\ >nul
  if errorlevel 1 (echo BUILD_FAIL v%%V & exit /b 1)
)
echo === building conv2d_avx2_bench ===
cl %FLAGS% conv2d_avx2_bench.c conv2d_ref.c conv2d_v1.c conv2d_v2.c conv2d_v3.c conv2d_v4.c conv2d_v5.c conv2d_v6.c /Fe:build_avx2\bench.exe /Fo:build_avx2\ >nul
if errorlevel 1 (echo BUILD_FAIL bench & exit /b 1)
echo ALL_BUILDS_OK

@echo off
rem ============================================================================
rem  matvec CUDA 构建脚本 (CUDA 12.9, RTX 3060 / sm_86)
rem  依赖:
rem    - MSVC 14.44.35207 工具集 (头/库可被 cudafe++ 解析, _MSC_VER 1944)
rem    - Windows SDK 10.0.22621.0
rem    - CUDA 12.9 自带的 vcvars64.bat 垫片:
rem        C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
rem      (仅做环境变量设置, 因为只装了工具集、没装完整 VS C++ 工作负载)
rem  用法: 双击本文件, 或 cmd 中 build_cuda.bat
rem ============================================================================
setlocal
set MSVC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKVER=10.0.22621.0
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe

set PATH=%MSVC%\bin\Hostx64\x64;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKVER%\ucrt;%SDK%\Include\%SDKVER%\um;%SDK%\Include\%SDKVER%\shared
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKVER%\ucrt\x64;%SDK%\Lib\%SDKVER%\um\x64
set TMP=C:\tmp\nvtmp
set TEMP=C:\tmp\nvtmp
if not exist %TMP% mkdir %TMP%

cd /d %~dp0
"%NVCC%" -O2 -arch=sm_86 -ccbin "%MSVC%\bin\Hostx64\x64" -I. ^
  matvet_cuda.cu matvet_cuda_v1.cu matvet_cuda_v2.cu matvet_cuda_v3.cu matvet_cuda_v4.cu matvet_cuda_v5.cu matvet_cuda_cublas.cu test_matvet_cuda.cu matvet_v0.c ^
  -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" -lcublas -o test_cuda.exe
if errorlevel 1 (
  echo [FAILED] 编译出错, 见上方日志
  exit /b 1
)
echo [OK] 生成 test_cuda.exe
echo 运行: test_cuda.exe
endlocal

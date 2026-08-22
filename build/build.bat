@echo off
REM build.bat - build all FastIV correctness tests (Windows)
REM   default: GCC (WinLibs 15.2.0)
REM   usage: build.bat msvc   -> build with MSVC
REM
REM macOS / Linux use build/Makefile instead.
REM Layout (after the directory reshuffle):
REM   public headers : api/*.h
REM   implementations: src/ctensor/*.c
REM   tests          : src/test/*.c
REM   intermediates  : obj/*.obj
REM   executables    : build/*.exe
setlocal
set DDIR=%~dp0
set API=%DDIR%..\api
set SRC=%DDIR%..\src
set CT=%SRC%\ctensor
set MAT=%SRC%\mat
set TEST=%SRC%\test
set OBJ=%DDIR%..\obj
set BIN=%DDIR%

if not exist "%OBJ%" mkdir "%OBJ%"
if not exist "%BIN%" mkdir "%BIN%"

set GCC="C:\winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64msvcrt-13.0.0-r1\mingw64\bin\gcc.exe"

set NN=%SRC%\nn
set REF=%SRC%\reference\c_face_detect_release\c

if "%1"=="msvc" goto msvc

echo [GCC] compiling objects...
%GCC% -std=c23 -O2 -I "%API%" -c "%CT%\fiv_common.c"    -o "%OBJ%\fiv_common.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%CT%\fiv_ctensor.c"   -o "%OBJ%\fiv_ctensor.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%CT%\fiv_binary_op.c" -o "%OBJ%\fiv_binary_op.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%MAT%\fiv_mat_transpose.c" -o "%OBJ%\fiv_mat_transpose.o"
%GCC% -std=c23 -O2 -I "%API%" -I "%MAT%" -c "%MAT%\fiv_mat_vec.c"  -o "%OBJ%\fiv_mat_vec.o"
%GCC% -std=c23 -O2 -I "%API%" -I "%MAT%" -c "%MAT%\fiv_mat_mul.c"  -o "%OBJ%\fiv_mat_mul.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_activate_fn.c"     -o "%OBJ%\fiv_activate_fn.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_linear_node.c"     -o "%OBJ%\fiv_linear_node.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_nn_topo.c"         -o "%OBJ%\fiv_nn_topo.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_nn_infer.c"        -o "%OBJ%\fiv_nn_infer.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_nn_train.c"        -o "%OBJ%\fiv_nn_train.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_nn_conv2d.c"  -o "%OBJ%\fiv_nn_conv2d.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_flatten_node.c"  -o "%OBJ%\fiv_flatten_node.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_max_2d.c"  -o "%OBJ%\fiv_max_2d.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_add_node.c"  -o "%OBJ%\fiv_add_node.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%NN%\fiv_pad_node.c"  -o "%OBJ%\fiv_pad_node.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%TEST%\test_darray.c"  -o "%OBJ%\test_darray.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%TEST%\test_ctensor.c" -o "%OBJ%\test_ctensor.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%TEST%\test_mat_transpose.c" -o "%OBJ%\test_mat_transpose.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%TEST%\test_mat_vec.c"       -o "%OBJ%\test_mat_vec.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%TEST%\test_mat_mul.c"       -o "%OBJ%\test_mat_mul.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%TEST%\test_nn.c"  -o "%OBJ%\test_nn.o"
%GCC% -std=c23 -O2 -I "%API%" -c "%TEST%\test_nn_mnist.c"  -o "%OBJ%\test_nn_mnist.o"
%GCC% -std=c23 -O2 -I "%API%" -I "%NN%" -c "%TEST%\test_nn_conv2d.c"  -o "%OBJ%\test_nn_conv2d.o"
%GCC% -std=c23 -O2 -I "%API%" -I "%NN%" -c "%TEST%\test_nn_mnist_conv.c"  -o "%OBJ%\test_nn_mnist_conv.o"
if errorlevel 1 (echo GCC compile FAILED & exit /b 1)

echo [GCC] linking test_darray...
%GCC% -std=c23 -O2 "%OBJ%\test_darray.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_darray.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_ctensor...
%GCC% -std=c23 -O2 "%OBJ%\test_ctensor.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_ctensor.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_mat_transpose...
%GCC% -std=c23 -O2 "%OBJ%\test_mat_transpose.o" "%OBJ%\fiv_mat_transpose.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_mat_transpose.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_mat_vec...
%GCC% -std=c23 -O2 "%OBJ%\test_mat_vec.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_mat_vec.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_mat_mul...
%GCC% -std=c23 -O2 "%OBJ%\test_mat_mul.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_mat_mul.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_nn...
%GCC% -std=c23 -O2 "%OBJ%\test_nn.o" "%OBJ%\fiv_nn_infer.o" "%OBJ%\fiv_nn_topo.o" "%OBJ%\fiv_nn_train.o" "%OBJ%\fiv_linear_node.o" "%OBJ%\fiv_activate_fn.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_flatten_node.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_add_node.o" "%OBJ%\fiv_pad_node.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_mat_reduce.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_nn.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_nn_mnist...
%GCC% -std=c23 -O2 "%OBJ%\test_nn_mnist.o" "%OBJ%\fiv_nn_infer.o" "%OBJ%\fiv_nn_topo.o" "%OBJ%\fiv_nn_train.o" "%OBJ%\fiv_linear_node.o" "%OBJ%\fiv_activate_fn.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_flatten_node.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_add_node.o" "%OBJ%\fiv_pad_node.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_mat_reduce.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_nn_mnist.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] linking test_nn_mnist_conv...
%CL% /utf-8 /O2 "%OBJ%\test_nn_mnist_conv.obj" "%OBJ%\fiv_nn_infer.obj" "%OBJ%\fiv_nn_topo.obj" "%OBJ%\fiv_nn_train.obj" "%OBJ%\fiv_linear_node.obj" "%OBJ%\fiv_activate_fn.obj" "%OBJ%\fiv_nn_conv2d.obj" "%OBJ%\fiv_flatten_node.obj" "%OBJ%\fiv_max_2d.obj" "%OBJ%\fiv_add_node.obj" "%OBJ%\fiv_pad_node.obj" "%OBJ%\fiv_mat_vec.obj" "%OBJ%\fiv_mat_reduce.obj" "%OBJ%\fiv_mat_mul.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_nn_mnist_conv.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_nn_mnist_conv...
%GCC% -std=c23 -O2 "%OBJ%\test_nn_mnist_conv.o" "%OBJ%\fiv_nn_infer.o" "%OBJ%\fiv_nn_topo.o" "%OBJ%\fiv_nn_train.o" "%OBJ%\fiv_linear_node.o" "%OBJ%\fiv_activate_fn.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_flatten_node.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_add_node.o" "%OBJ%\fiv_pad_node.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_mat_reduce.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_nn_mnist_conv.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_nn_conv2d...
%GCC% -std=c23 -O2 "%OBJ%\test_nn_conv2d.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_nn_conv2d.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] compiling reference (BlazeFace golden) objects...
%GCC% -std=c23 -O2 -I "%REF%" -c "%REF%\weights.c" -o "%OBJ%\ref_weights.o"
%GCC% -std=c23 -O2 -I "%REF%" -c "%REF%\geom.c"    -o "%OBJ%\ref_geom.o"
%GCC% -std=c23 -O2 -I "%REF%" -c "%REF%\cnn_ops.c" -o "%OBJ%\ref_cnn_ops.o"
%GCC% -std=c23 -O2 -I "%REF%" -c "%REF%\detect.c"  -o "%OBJ%\ref_detect.o"
%GCC% -std=c23 -O2 -I "%API%" -I "%NN%" -I "%REF%" -c "%TEST%\test_blazeface.c" -o "%OBJ%\test_blazeface.o"
if errorlevel 1 (echo GCC compile FAILED & exit /b 1)
echo [GCC] linking test_blazeface...
%GCC% -std=c23 -O2 "%OBJ%\test_blazeface.o" "%OBJ%\ref_weights.o" "%OBJ%\ref_geom.o" "%OBJ%\ref_cnn_ops.o" "%OBJ%\ref_detect.o" "%OBJ%\fiv_nn_infer.o" "%OBJ%\fiv_nn_topo.o" "%OBJ%\fiv_nn_train.o" "%OBJ%\fiv_linear_node.o" "%OBJ%\fiv_activate_fn.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_flatten_node.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_add_node.o" "%OBJ%\fiv_pad_node.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_mat_reduce.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_blazeface.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] OK -^> %BIN%\test_darray.exe, %BIN%\test_ctensor.exe, %BIN%\test_mat_transpose.exe, %BIN%\test_mat_vec.exe, %BIN%\test_mat_mul.exe, %BIN%\test_nn.exe, %BIN%\test_nn_mnist.exe, %BIN%\test_nn_conv2d.exe, %BIN%\test_nn_mnist_conv.exe, %BIN%\test_blazeface.exe
goto end

:msvc
set CL="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe"
set INCLUDE=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared
set LIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64

echo [MSVC] compiling objects...
%CL% /utf-8 /O2 /I "%API%" /c "%CT%\fiv_common.c"    /Fo"%OBJ%\fiv_common.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%CT%\fiv_ctensor.c"   /Fo"%OBJ%\fiv_ctensor.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%CT%\fiv_binary_op.c" /Fo"%OBJ%\fiv_binary_op.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%MAT%\fiv_mat_transpose.c" /Fo"%OBJ%\fiv_mat_transpose.obj"
%CL% /utf-8 /O2 /I "%API%" /I "%MAT%" /c "%MAT%\fiv_mat_vec.c"  /Fo"%OBJ%\fiv_mat_vec.obj"
%CL% /utf-8 /O2 /I "%API%" /I "%MAT%" /c "%MAT%\fiv_mat_mul.c"  /Fo"%OBJ%\fiv_mat_mul.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_activate_fn.c"     /Fo"%OBJ%\fiv_activate_fn.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_linear_node.c"     /Fo"%OBJ%\fiv_linear_node.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_nn_topo.c"         /Fo"%OBJ%\fiv_nn_topo.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_nn_infer.c"        /Fo"%OBJ%\fiv_nn_infer.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_nn_train.c"        /Fo"%OBJ%\fiv_nn_train.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_nn_conv2d.c"  /Fo"%OBJ%\fiv_nn_conv2d.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_flatten_node.c"  /Fo"%OBJ%\fiv_flatten_node.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_max_2d.c"  /Fo"%OBJ%\fiv_max_2d.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_add_node.c"  /Fo"%OBJ%\fiv_add_node.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%NN%\fiv_pad_node.c"  /Fo"%OBJ%\fiv_pad_node.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%TEST%\test_darray.c"  /Fo"%OBJ%\test_darray.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%TEST%\test_ctensor.c" /Fo"%OBJ%\test_ctensor.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%TEST%\test_mat_transpose.c" /Fo"%OBJ%\test_mat_transpose.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%TEST%\test_mat_vec.c"       /Fo"%OBJ%\test_mat_vec.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%TEST%\test_mat_mul.c"       /Fo"%OBJ%\test_mat_mul.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%TEST%\test_nn.c"  /Fo"%OBJ%\test_nn.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%TEST%\test_nn_mnist.c"  /Fo"%OBJ%\test_nn_mnist.obj"
%CL% /utf-8 /O2 /I "%API%" /I "%NN%" /c "%TEST%\test_nn_conv2d.c" /Fo"%OBJ%\test_nn_conv2d.obj"
%CL% /utf-8 /O2 /I "%API%" /I "%NN%" /c "%TEST%\test_nn_mnist_conv.c" /Fo"%OBJ%\test_nn_mnist_conv.obj"
if errorlevel 1 (echo MSVC compile FAILED & exit /b 1)

echo [MSVC] linking test_darray...
%CL% /utf-8 /O2 "%OBJ%\test_darray.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_darray.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] linking test_ctensor...
%CL% /utf-8 /O2 "%OBJ%\test_ctensor.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_ctensor.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] linking test_mat_transpose...
%CL% /utf-8 /O2 "%OBJ%\test_mat_transpose.obj" "%OBJ%\fiv_mat_transpose.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_mat_transpose.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] linking test_mat_vec...
%CL% /utf-8 /O2 "%OBJ%\test_mat_vec.obj" "%OBJ%\fiv_mat_vec.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_mat_vec.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] linking test_mat_mul...
%CL% /utf-8 /O2 "%OBJ%\test_mat_mul.obj" "%OBJ%\fiv_mat_mul.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_mat_mul.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] linking test_nn...
%CL% /utf-8 /O2 "%OBJ%\test_nn.obj" "%OBJ%\fiv_nn_infer.obj" "%OBJ%\fiv_nn_topo.obj" "%OBJ%\fiv_nn_train.obj" "%OBJ%\fiv_linear_node.obj" "%OBJ%\fiv_activate_fn.obj" "%OBJ%\fiv_nn_conv2d.obj" "%OBJ%\fiv_flatten_node.obj" "%OBJ%\fiv_max_2d.obj" "%OBJ%\fiv_add_node.obj" "%OBJ%\fiv_pad_node.obj" "%OBJ%\fiv_mat_vec.obj" "%OBJ%\fiv_mat_reduce.obj" "%OBJ%\fiv_mat_mul.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_nn.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] linking test_nn_mnist...
%CL% /utf-8 /O2 "%OBJ%\test_nn_mnist.obj" "%OBJ%\fiv_nn_infer.obj" "%OBJ%\fiv_nn_topo.obj" "%OBJ%\fiv_nn_train.obj" "%OBJ%\fiv_linear_node.obj" "%OBJ%\fiv_activate_fn.obj" "%OBJ%\fiv_nn_conv2d.obj" "%OBJ%\fiv_flatten_node.obj" "%OBJ%\fiv_max_2d.obj" "%OBJ%\fiv_add_node.obj" "%OBJ%\fiv_pad_node.obj" "%OBJ%\fiv_mat_vec.obj" "%OBJ%\fiv_mat_reduce.obj" "%OBJ%\fiv_mat_mul.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_nn_mnist.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] linking test_nn_conv2d...
%CL% /utf-8 /O2 "%OBJ%\test_nn_conv2d.obj" "%OBJ%\fiv_nn_conv2d.obj" "%OBJ%\fiv_max_2d.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_nn_conv2d.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] compiling reference (BlazeFace golden) objects...
%CL% /utf-8 /O2 /I "%REF%" /c "%REF%\weights.c" /Fo"%OBJ%\ref_weights.obj"
%CL% /utf-8 /O2 /I "%REF%" /c "%REF%\geom.c"    /Fo"%OBJ%\ref_geom.obj"
%CL% /utf-8 /O2 /I "%REF%" /c "%REF%\cnn_ops.c" /Fo"%OBJ%\ref_cnn_ops.obj"
%CL% /utf-8 /O2 /I "%REF%" /c "%REF%\detect.c"  /Fo"%OBJ%\ref_detect.obj"
%CL% /utf-8 /O2 /I "%API%" /I "%NN%" /I "%REF%" /c "%TEST%\test_blazeface.c" /Fo"%OBJ%\test_blazeface.obj"
if errorlevel 1 (echo MSVC compile FAILED & exit /b 1)
echo [MSVC] linking test_blazeface...
%CL% /utf-8 /O2 "%OBJ%\test_blazeface.obj" "%OBJ%\ref_weights.obj" "%OBJ%\ref_geom.obj" "%OBJ%\ref_cnn_ops.obj" "%OBJ%\ref_detect.obj" "%OBJ%\fiv_nn_infer.obj" "%OBJ%\fiv_nn_topo.obj" "%OBJ%\fiv_nn_train.obj" "%OBJ%\fiv_linear_node.obj" "%OBJ%\fiv_activate_fn.obj" "%OBJ%\fiv_nn_conv2d.obj" "%OBJ%\fiv_flatten_node.obj" "%OBJ%\fiv_max_2d.obj" "%OBJ%\fiv_add_node.obj" "%OBJ%\fiv_pad_node.obj" "%OBJ%\fiv_mat_vec.obj" "%OBJ%\fiv_mat_reduce.obj" "%OBJ%\fiv_mat_mul.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_blazeface.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] OK -^> %BIN%\test_darray.exe, %BIN%\test_ctensor.exe, %BIN%\test_mat_transpose.exe, %BIN%\test_mat_vec.exe, %BIN%\test_mat_mul.exe, %BIN%\test_nn.exe, %BIN%\test_nn_mnist.exe, %BIN%\test_blazeface.exe

:end
endlocal

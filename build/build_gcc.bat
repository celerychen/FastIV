@echo off
REM build_gcc.bat - build all FastIV correctness tests with GCC (Windows)
REM   GCC only (WinLibs 15.2.0). For MSVC CL build use build_msvc.bat instead.
REM
REM macOS / Linux use build/Makefile instead.
REM Layout:
REM   public headers : api/*.h
REM   implementations: src/ctensor/*.c
REM   tests          : src/test/*.c
REM   intermediates  : obj/*.o
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

echo [GCC] compiling objects...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%CT%\fiv_common.c"    -o "%OBJ%\fiv_common.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%CT%\fiv_ctensor.c"   -o "%OBJ%\fiv_ctensor.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%CT%\fiv_binary_op.c" -o "%OBJ%\fiv_binary_op.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%MAT%\fiv_mat_transpose.c" -o "%OBJ%\fiv_mat_transpose.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%MAT%" -c "%MAT%\fiv_mat_vec.c"  -o "%OBJ%\fiv_mat_vec.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%MAT%" -c "%MAT%\fiv_mat_reduce.c" -o "%OBJ%\fiv_mat_reduce.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%MAT%" -c "%MAT%\fiv_mat_mul.c"  -o "%OBJ%\fiv_mat_mul.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%MAT%" -c "%MAT%\fiv_linalg_kernels.c" -o "%OBJ%\fiv_linalg_kernels.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%MAT%" -c "%MAT%\fiv_mat_cholesky.c" -o "%OBJ%\fiv_mat_cholesky.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%MAT%" -c "%MAT%\fiv_mat_lu.c" -o "%OBJ%\fiv_mat_lu.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_activate_fn.c"     -o "%OBJ%\fiv_activate_fn.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_linear_node.c"     -o "%OBJ%\fiv_linear_node.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_nn_topo.c"         -o "%OBJ%\fiv_nn_topo.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_nn_infer.c"        -o "%OBJ%\fiv_nn_infer.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_nn_train.c"        -o "%OBJ%\fiv_nn_train.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_nn_conv2d.c"  -o "%OBJ%\fiv_nn_conv2d.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_flatten_node.c"  -o "%OBJ%\fiv_flatten_node.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_max_2d.c"  -o "%OBJ%\fiv_max_2d.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_add_node.c"  -o "%OBJ%\fiv_add_node.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%NN%\fiv_pad_node.c"  -o "%OBJ%\fiv_pad_node.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%TEST%\test_darray.c"  -o "%OBJ%\test_darray.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%TEST%\test_ctensor.c" -o "%OBJ%\test_ctensor.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%TEST%\test_mat_transpose.c" -o "%OBJ%\test_mat_transpose.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%TEST%\test_mat_vec.c"       -o "%OBJ%\test_mat_vec.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%MAT%" -c "%TEST%\test_mat_mul.c"       -o "%OBJ%\test_mat_mul.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%MAT%" -c "%TEST%\test_mat_cholesky.c" -o "%OBJ%\test_mat_cholesky.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%MAT%" -c "%TEST%\test_mat_lu.c" -o "%OBJ%\test_mat_lu.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%TEST%\test_nn.c"  -o "%OBJ%\test_nn.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -c "%TEST%\test_nn_mnist.c"  -o "%OBJ%\test_nn_mnist.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%NN%" -c "%TEST%\test_nn_conv2d.c"  -o "%OBJ%\test_nn_conv2d.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%NN%" -c "%TEST%\test_nn_mnist_conv.c"  -o "%OBJ%\test_nn_mnist_conv.o"
if errorlevel 1 (echo GCC compile FAILED & exit /b 1)

echo [GCC] linking test_darray...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_darray.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_darray.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_ctensor...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_ctensor.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_ctensor.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_mat_transpose...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_mat_transpose.o" "%OBJ%\fiv_mat_transpose.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_mat_transpose.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_mat_vec...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_mat_vec.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_mat_reduce.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_mat_vec.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_mat_mul...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_mat_mul.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_mat_mul.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_mat_cholesky...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_mat_cholesky.o" "%OBJ%\fiv_mat_cholesky.o" "%OBJ%\fiv_linalg_kernels.o" "%OBJ%\fiv_mat_transpose.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_mat_cholesky.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_mat_lu...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_mat_lu.o" "%OBJ%\fiv_mat_lu.o" "%OBJ%\fiv_linalg_kernels.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_mat_lu.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_nn...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_nn.o" "%OBJ%\fiv_nn_infer.o" "%OBJ%\fiv_nn_topo.o" "%OBJ%\fiv_nn_train.o" "%OBJ%\fiv_linear_node.o" "%OBJ%\fiv_activate_fn.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_flatten_node.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_add_node.o" "%OBJ%\fiv_pad_node.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_mat_reduce.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_nn.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_nn_mnist...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_nn_mnist.o" "%OBJ%\fiv_nn_infer.o" "%OBJ%\fiv_nn_topo.o" "%OBJ%\fiv_nn_train.o" "%OBJ%\fiv_linear_node.o" "%OBJ%\fiv_activate_fn.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_flatten_node.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_add_node.o" "%OBJ%\fiv_pad_node.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_mat_reduce.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_nn_mnist.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_nn_mnist_conv...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_nn_mnist_conv.o" "%OBJ%\fiv_nn_infer.o" "%OBJ%\fiv_nn_topo.o" "%OBJ%\fiv_nn_train.o" "%OBJ%\fiv_linear_node.o" "%OBJ%\fiv_activate_fn.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_flatten_node.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_add_node.o" "%OBJ%\fiv_pad_node.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_mat_reduce.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_nn_mnist_conv.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_nn_conv2d...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_nn_conv2d.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_nn_conv2d.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] compiling reference (BlazeFace golden) objects...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%REF%" -c "%REF%\weights.c" -o "%OBJ%\ref_weights.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%REF%" -c "%REF%\geom.c"    -o "%OBJ%\ref_geom.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%REF%" -c "%REF%\cnn_ops.c" -o "%OBJ%\ref_cnn_ops.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%REF%" -c "%REF%\detect.c"  -o "%OBJ%\ref_detect.o"
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma -I "%API%" -I "%NN%" -I "%REF%" -c "%TEST%\test_blazeface.c" -o "%OBJ%\test_blazeface.o"
if errorlevel 1 (echo GCC compile FAILED & exit /b 1)
echo [GCC] linking test_blazeface...
%GCC% -std=c23 -O3 -march=native -mavx2 -mfma "%OBJ%\test_blazeface.o" "%OBJ%\ref_weights.o" "%OBJ%\ref_geom.o" "%OBJ%\ref_cnn_ops.o" "%OBJ%\ref_detect.o" "%OBJ%\fiv_nn_infer.o" "%OBJ%\fiv_nn_topo.o" "%OBJ%\fiv_nn_train.o" "%OBJ%\fiv_linear_node.o" "%OBJ%\fiv_activate_fn.o" "%OBJ%\fiv_nn_conv2d.o" "%OBJ%\fiv_flatten_node.o" "%OBJ%\fiv_max_2d.o" "%OBJ%\fiv_add_node.o" "%OBJ%\fiv_pad_node.o" "%OBJ%\fiv_mat_vec.o" "%OBJ%\fiv_mat_reduce.o" "%OBJ%\fiv_mat_mul.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -lm -o "%BIN%\test_blazeface.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] OK -^> %BIN%\test_darray.exe, %BIN%\test_ctensor.exe, %BIN%\test_mat_transpose.exe, %BIN%\test_mat_vec.exe, %BIN%\test_mat_mul.exe, %BIN%\test_mat_cholesky.exe, %BIN%\test_mat_lu.exe, %BIN%\test_nn.exe, %BIN%\test_nn_mnist.exe, %BIN%\test_nn_conv2d.exe, %BIN%\test_nn_mnist_conv.exe, %BIN%\test_blazeface.exe
goto end

:end
endlocal

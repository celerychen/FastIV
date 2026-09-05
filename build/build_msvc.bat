@echo off
REM build_msvc.bat - build FastIV with MSVC CL only (standalone, NOT mixed with GCC build_gcc.bat)
REM   builds: test_face_api.exe  (per request; reference/test_blazeface skipped - repo dropped c_face_detect_release/)
REM   intermediates: obj/cl/*.obj   (isolated from GCC's obj/*.o)
REM   executable : build/test_face_api.exe
setlocal
set DDIR=%~dp0
set API=%DDIR%..\api
set SRC=%DDIR%..\src
set CT=%SRC%\ctensor
set MAT=%SRC%\mat
set NN=%SRC%\nn
set IMG=%SRC%\image
set FACE=%SRC%\..\app\face\blazeFace
set IMP=%SRC%\..\import\image_io
set TEST=%SRC%\test
set OBJ=%DDIR%..\obj\cl
set BIN=%DDIR%

if not exist "%OBJ%" mkdir "%OBJ%"
if not exist "%BIN%" mkdir "%BIN%"

set CLEXE="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe"
set INCLUDE=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared
set LIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64

REM ---- ctensor / mat / nn (CFLAGS: /O2 /arch:AVX2 /fp:fast /GL) ----
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%CT%\fiv_common.c"          /Fo"%OBJ%\fiv_common.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%CT%\fiv_ctensor.c"         /Fo"%OBJ%\fiv_ctensor.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%CT%\fiv_binary_op.c"       /Fo"%OBJ%\fiv_binary_op.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%MAT%\fiv_mat_transpose.c"  /Fo"%OBJ%\fiv_mat_transpose.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%MAT%\fiv_mat_vec.c"   /Fo"%OBJ%\fiv_mat_vec.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%MAT%\fiv_mat_mul.c"    /Fo"%OBJ%\fiv_mat_mul.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%MAT%\fiv_linalg_kernels.c" /Fo"%OBJ%\fiv_linalg_kernels.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%MAT%\fiv_mat_cholesky.c"   /Fo"%OBJ%\fiv_mat_cholesky.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%MAT%\fiv_mat_lu.c"         /Fo"%OBJ%\fiv_mat_lu.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%MAT%\fiv_mat_reduce.c"     /Fo"%OBJ%\fiv_mat_reduce.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%MAT%\fiv_mat_vec_db.c"    /Fo"%OBJ%\fiv_mat_vec_db.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%MAT%\fiv_mat_reduce_db.c" /Fo"%OBJ%\fiv_mat_reduce_db.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%MAT%\fiv_mat_mul_db.c"    /Fo"%OBJ%\fiv_mat_mul_db.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_activate_fn.c"     /Fo"%OBJ%\fiv_activate_fn.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_linear_node.c"     /Fo"%OBJ%\fiv_linear_node.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_nn_topo.c"         /Fo"%OBJ%\fiv_nn_topo.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_nn_infer.c"        /Fo"%OBJ%\fiv_nn_infer.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_nn_train.c"        /Fo"%OBJ%\fiv_nn_train.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_nn_conv2d.c"       /Fo"%OBJ%\fiv_nn_conv2d.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_flatten_node.c"    /Fo"%OBJ%\fiv_flatten_node.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_max_2d.c"          /Fo"%OBJ%\fiv_max_2d.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_add_node.c"        /Fo"%OBJ%\fiv_add_node.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%NN%\fiv_pad_node.c"        /Fo"%OBJ%\fiv_pad_node.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /c "%NN%\fiv_nn_1x1_conv2d.c"      /Fo"%OBJ%\fiv_nn_1x1_conv2d.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /c "%NN%\fiv_nn_2x2_conv2d.c"      /Fo"%OBJ%\fiv_nn_2x2_conv2d.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /c "%NN%\fiv_nn_3x3_conv2d.c"      /Fo"%OBJ%\fiv_nn_3x3_conv2d.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /c "%NN%\fiv_nn_5x5_conv2d.c"      /Fo"%OBJ%\fiv_nn_5x5_conv2d.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /c "%NN%\fiv_concat_node.c"        /Fo"%OBJ%\fiv_concat_node.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /c "%NN%\fiv_prelu_node.c"         /Fo"%OBJ%\fiv_prelu_node.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /c "%NN%\fiv_sigmoid_node.c"       /Fo"%OBJ%\fiv_sigmoid_node.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /c "%NN%\fiv_spatial_pad_node.c"   /Fo"%OBJ%\fiv_spatial_pad_node.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /c "%NN%\fiv_upsample_node.c"      /Fo"%OBJ%\fiv_upsample_node.obj"
if errorlevel 1 (echo [MSVC] compile FAILED & exit /b 1)

REM ---- image / face ----
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%IMG%" /I "%IMP%" /c "%IMG%\fiv_image_io.c"          /Fo"%OBJ%\fiv_image_io.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%IMG%\fiv_image_color_space.c"                       /Fo"%OBJ%\fiv_image_color_space.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%FACE%" /c "%FACE%\fiv_face_weights.c"               /Fo"%OBJ%\fiv_face_weights.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%FACE%" /c "%FACE%\fiv_face_warp.c"                  /Fo"%OBJ%\fiv_face_warp.obj"
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /I "%FACE%" /c "%FACE%\fiv_face.c"             /Fo"%OBJ%\fiv_face.obj"
if errorlevel 1 (echo [MSVC] compile FAILED & exit /b 1)

REM ---- test_face_api ----
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%NN%" /I "%FACE%" /c "%TEST%\test_face_api.c"        /Fo"%OBJ%\test_face_api.obj"
if errorlevel 1 (echo [MSVC] compile FAILED & exit /b 1)

REM ---- test_mat_cholesky ----
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%TEST%\test_mat_cholesky.c" /Fo"%OBJ%\test_mat_cholesky.obj"
if errorlevel 1 (echo [MSVC] compile FAILED & exit /b 1)
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /LTCG "%OBJ%\test_mat_cholesky.obj" "%OBJ%\fiv_mat_cholesky.obj" "%OBJ%\fiv_linalg_kernels.obj" "%OBJ%\fiv_mat_transpose.obj" "%OBJ%\fiv_mat_mul.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_mat_cholesky.exe"
if errorlevel 1 (echo [MSVC] link FAILED & exit /b 1)

REM ---- test_mat_lu ----
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /I "%MAT%" /c "%TEST%\test_mat_lu.c" /Fo"%OBJ%\test_mat_lu.obj"
if errorlevel 1 (echo [MSVC] compile FAILED & exit /b 1)
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /LTCG "%OBJ%\test_mat_lu.obj" "%OBJ%\fiv_mat_lu.obj" "%OBJ%\fiv_linalg_kernels.obj" "%OBJ%\fiv_mat_mul.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_mat_lu.exe"
if errorlevel 1 (echo [MSVC] link FAILED & exit /b 1)

REM ---- test_mat_transpose ----
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /GL /I "%API%" /c "%TEST%\test_mat_transpose.c" /Fo"%OBJ%\test_mat_transpose.obj"
if errorlevel 1 (echo [MSVC] compile FAILED & exit /b 1)
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /LTCG "%OBJ%\test_mat_transpose.obj" "%OBJ%\fiv_mat_transpose.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_mat_transpose.exe"
if errorlevel 1 (echo [MSVC] link FAILED & exit /b 1)

REM ---- link (CFLAGS + /LTCG) ----
%CLEXE% /utf-8 /O2 /arch:AVX2 /fp:fast /LTCG "%OBJ%\test_face_api.obj" "%OBJ%\fiv_face.obj" "%OBJ%\fiv_face_warp.obj" "%OBJ%\fiv_face_weights.obj" "%OBJ%\fiv_image_io.obj" "%OBJ%\fiv_image_color_space.obj" "%OBJ%\fiv_nn_infer.obj" "%OBJ%\fiv_nn_topo.obj" "%OBJ%\fiv_nn_train.obj" "%OBJ%\fiv_linear_node.obj" "%OBJ%\fiv_activate_fn.obj" "%OBJ%\fiv_nn_conv2d.obj" "%OBJ%\fiv_nn_1x1_conv2d.obj" "%OBJ%\fiv_nn_2x2_conv2d.obj" "%OBJ%\fiv_nn_3x3_conv2d.obj" "%OBJ%\fiv_nn_5x5_conv2d.obj" "%OBJ%\fiv_flatten_node.obj" "%OBJ%\fiv_max_2d.obj" "%OBJ%\fiv_add_node.obj" "%OBJ%\fiv_pad_node.obj" "%OBJ%\fiv_concat_node.obj" "%OBJ%\fiv_prelu_node.obj" "%OBJ%\fiv_sigmoid_node.obj" "%OBJ%\fiv_spatial_pad_node.obj" "%OBJ%\fiv_upsample_node.obj" "%OBJ%\fiv_mat_vec.obj" "%OBJ%\fiv_mat_vec_db.obj" "%OBJ%\fiv_mat_reduce.obj" "%OBJ%\fiv_mat_reduce_db.obj" "%OBJ%\fiv_mat_mul.obj" "%OBJ%\fiv_mat_mul_db.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_face_api.exe"
if errorlevel 1 (echo [MSVC] link FAILED & exit /b 1)

echo [MSVC] OK -^> %BIN%\test_face_api.exe, %BIN%\test_mat_transpose.exe, %BIN%\test_mat_cholesky.exe, %BIN%\test_mat_lu.exe
goto end

:end
endlocal

@echo off
REM build_darray.bat - build the FastIV correctness tests
REM   default: GCC (WinLibs 15.2.0)
REM   usage: build_darray.bat msvc   -> build with MSVC
REM
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
set TEST=%SRC%\test
set OBJ=%DDIR%..\obj
set BIN=%DDIR%

if not exist "%OBJ%" mkdir "%OBJ%"
if not exist "%BIN%" mkdir "%BIN%"

set GCC="C:\winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64msvcrt-13.0.0-r1\mingw64\bin\gcc.exe"

if "%1"=="msvc" goto msvc

echo [GCC] compiling objects...
%GCC% -O2 -I "%API%" -c "%CT%\fiv_common.c"    -o "%OBJ%\fiv_common.o"
%GCC% -O2 -I "%API%" -c "%CT%\fiv_ctensor.c"   -o "%OBJ%\fiv_ctensor.o"
%GCC% -O2 -I "%API%" -c "%CT%\fiv_binary_op.c" -o "%OBJ%\fiv_binary_op.o"
%GCC% -O2 -I "%API%" -c "%TEST%\test_darray.c"  -o "%OBJ%\test_darray.o"
%GCC% -O2 -I "%API%" -c "%TEST%\test_ctensor.c" -o "%OBJ%\test_ctensor.o"
if errorlevel 1 (echo GCC compile FAILED & exit /b 1)

echo [GCC] linking test_darray...
%GCC% -O2 "%OBJ%\test_darray.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_darray.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] linking test_ctensor...
%GCC% -O2 "%OBJ%\test_ctensor.o" "%OBJ%\fiv_ctensor.o" "%OBJ%\fiv_binary_op.o" "%OBJ%\fiv_common.o" -o "%BIN%\test_ctensor.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [GCC] OK -^> %BIN%\test_darray.exe, %BIN%\test_ctensor.exe
goto end

:msvc
set CL="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe"
set INCLUDE=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared
set LIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64

echo [MSVC] compiling objects...
%CL% /utf-8 /O2 /I "%API%" /c "%CT%\fiv_common.c"    /Fo"%OBJ%\fiv_common.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%CT%\fiv_ctensor.c"   /Fo"%OBJ%\fiv_ctensor.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%CT%\fiv_binary_op.c" /Fo"%OBJ%\fiv_binary_op.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%TEST%\test_darray.c"  /Fo"%OBJ%\test_darray.obj"
%CL% /utf-8 /O2 /I "%API%" /c "%TEST%\test_ctensor.c" /Fo"%OBJ%\test_ctensor.obj"
if errorlevel 1 (echo MSVC compile FAILED & exit /b 1)

echo [MSVC] linking test_darray...
%CL% /utf-8 /O2 "%OBJ%\test_darray.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_darray.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] linking test_ctensor...
%CL% /utf-8 /O2 "%OBJ%\test_ctensor.obj" "%OBJ%\fiv_ctensor.obj" "%OBJ%\fiv_binary_op.obj" "%OBJ%\fiv_common.obj" /Fe"%BIN%\test_ctensor.exe"
if errorlevel 1 (echo link FAILED & exit /b 1)
echo [MSVC] OK -^> %BIN%\test_darray.exe, %BIN%\test_ctensor.exe

:end
endlocal

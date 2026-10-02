@echo off
call "d:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cl /O2 /EHsc /arch:AVX2 gemm_pattern.cpp /Fe:bin\gemm_pattern.exe
exit /b %errorlevel%
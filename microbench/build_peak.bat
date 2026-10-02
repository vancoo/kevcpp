@echo off
call "d:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /O2 /EHsc /arch:AVX512 peak_int8.cpp /Fe:bin\peak_int8.exe
exit /b %errorlevel%
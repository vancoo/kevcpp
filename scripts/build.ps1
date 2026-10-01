# build.ps1 — 用 MSVC 2022 构建 kevcpp (纯 CPU 版, llama.cpp 静态库 + kev 可执行)
# 用法:  powershell -ExecutionPolicy Bypass -File scripts\build.ps1 [-Clean] [-Config Debug|Release]
param(
    [switch]$Clean,
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build"

# ---- 1) 加载 MSVC 环境 (cl.exe 不在 PATH, 必须先 vcvars64.bat) ----
$vcvars = "d:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found: $vcvars"
}

# Build a batch wrapper that calls vcvars64.bat then dumps the environment.
$envDump = Join-Path $env:TEMP "msvc_env_$PID.cmd"
$envOut  = Join-Path $env:TEMP "msvc_env_$PID.txt"
@"
call "$vcvars"
set
"@ | Set-Content -Path $envDump -Encoding ascii

& cmd.exe /c "`"$envDump`" > `"$envOut`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Remove-Item $envDump, $envOut -Force -ErrorAction SilentlyContinue
    throw "vcvars64.bat failed (exit $LASTEXITCODE)"
}

# Load the environment preserved after vcvars64.bat into this process.
Get-Content -Path $envOut | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        # Only import the variables that describe the dev environment (paths,
        # CL, CMake defaults). Import path-involved vars via SetEnvironmentVariable.
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}
Remove-Item $envDump, $envOut -Force -ErrorAction SilentlyContinue

# ---- 2) 配置 (cmake + VS generator) ----
if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "## Cleaning $buildDir"
    Remove-Item $buildDir -Recurse -Force
}

$cfgLog = Join-Path $buildDir "configure.log"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

# NOTE: we run via cmd.exe with cmd-level file redirection. PowerShell piping the
# stdout/stderr of a CMake child (MSBuild) through a named pipe is blocked by the
# sandbox (EPERM) and kills the build; cmd-level `> file 2>&1` sidesteps that.
Write-Host "## CMake configure (config=$Config)"
$configureCmd = "cmake -S `"$root`" -B `"$buildDir`" -G `"Visual Studio 17 2022`" -A x64 -DCMAKE_BUILD_TYPE=$Config -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=ON -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_COMMON=OFF -DBUILD_SHARED_LIBS=OFF > `"$cfgLog`" 2>&1"
& cmd.exe /d /c $configureCmd
if ($LASTEXITCODE -ne 0) {
    Get-Content $cfgLog -Tail 40
    throw "CMake configure failed (exit $LASTEXITCODE), see $cfgLog"
}
Write-Host "## CMake configure OK"

# ---- 3) 构建 ----
Write-Host "## MSBuild build ($Config) (single-node, see build.log)"
$buildLog = Join-Path $buildDir "build.log"

# IMPORTANT: build WITHOUT MSBuild /m (multi-node). In this sandbox,
# MSBuild worker-node IPC uses named pipes that are blocked, so /m fails with a bare
# "Checking Build System" and no error. Single-node msbuild works. We also build via
# cmd.exe with cmd-level file redirection (PowerShell piping of a CMake/MSBuild child
# is blocked by the sandbox as well).
$projFiles = @("kev.vcxproj", "headbench.vcxproj", "m3_bench.vcxproj", "m5_bench.vcxproj", "m7_bench.vcxproj", "main_server.vcxproj", "m8_bench.vcxproj", "merge_lora.vcxproj", "lora_ab.vcxproj")

foreach ($p in $projFiles) {
    $proj = Join-Path $buildDir $p
    $projCmd = "msbuild `"$proj`" /p:Configuration=$Config /v:m > `"$buildLog`" 2>&1"
    Write-Host "   cmd: $projCmd"
    $buildCmd = "call `"$vcvars`" >nul 2>&1 && $projCmd"
    $batch2 = Join-Path $env:TEMP "kevcpp_build_$PID.cmd"
    Set-Content -Path $batch2 -Value $buildCmd -Encoding ascii
    & cmd.exe /d /c "`"$batch2`""
    $code = $LASTEXITCODE
    Remove-Item $batch2 -Force -ErrorAction SilentlyContinue
    if ($code -ne 0) {
        Get-Content $buildLog -Tail 80
        throw "Build failed for $p (exit $code), see $buildLog"
    }
}
Write-Host "## MSBuild build OK (see $buildLog)"

$exeCfg = if ($Config -eq "Release") { "Release" } else { "Debug" }
Write-Host ""
Write-Host "## Build OK -> $buildDir\bin\$exeCfg\kev.exe, headbench.exe"
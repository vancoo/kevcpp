# build_vulkan.ps1 — configure + build kev-backend-vulkan.dll (GGML_VULKAN) in an
# ISOLATED build dir (build\vulkan-exp\vulkan-build) so the default CPU build dir
# (build\) stays Vulkan-free and green. Pure MSVC cl/link (Visual Studio 17 2022
# generator, x64, single-node msbuild -- /m is blocked by the sandbox named-pipe IPC).
#
# Env requirements:
#   VULKAN_SDK = D:\SDK\Vulkan\1.4.363.0   (or pass through $env:VULKAN_SDK)
#   System loader: C:\Windows\System32\vulkan-1.dll (present; runtime falls back to it)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build_vulkan.ps1 [-Clean] [-Config Release]
param(
    [switch]$Clean,
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

# This script lives at <repo>/scripts/build_vulkan.ps1, so the repo root is one level up.
$root     = Split-Path -Parent $PSScriptRoot
$vkExp    = Join-Path $root "build\vulkan-exp"
$buildDir = Join-Path $vkExp "vulkan-build"

$vcvars  = "d:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars" }

if (-not $env:VULKAN_SDK) { $env:VULKAN_SDK = "D:\SDK\Vulkan\1.4.363.0" }
if (-not (Test-Path (Join-Path $env:VULKAN_SDK "Bin\glslc.exe"))) {
    throw "Vulkan SDK glslc not found under VULKAN_SDK=$env:VULKAN_SDK"
}

# ---- 1) load MSVC env from vcvars64.bat ----
$envDump = Join-Path $env:TEMP "vk_env_$PID.cmd"
$envOut  = Join-Path $env:TEMP "vk_env_$PID.txt"
@"
call "$vcvars"
set
"@ | Set-Content -Path $envDump -Encoding ascii
& cmd.exe /c "`"$envDump`" > `"$envOut`" 2>&1"
if ($LASTEXITCODE -ne 0) { Remove-Item $envDump,$envOut -Force -ErrorAction SilentlyContinue; throw "vcvars64.bat failed" }
Get-Content -Path $envOut | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}
Remove-Item $envDump,$envOut -Force -ErrorAction SilentlyContinue

# ---- 2) clean if requested ----
if ($Clean -and (Test-Path $buildDir)) { Remove-Item $buildDir -Recurse -Force }

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

# ---- 3) configure ----
$cfgLog = Join-Path $buildDir "configure.log"
$configureCmd = "cmake -S `"$root`" -B `"$buildDir`" -G `"Visual Studio 17 2022`" -A x64 -DCMAKE_BUILD_TYPE=$Config -DKEV_BUILD_BACKEND_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=ON -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_COMMON=OFF -DBUILD_SHARED_LIBS=OFF > `"$cfgLog`" 2>&1"
Write-Host "## CMake configure (VULKAN_SDK=$env:VULKAN_SDK)"
& cmd.exe /d /c $configureCmd
if ($LASTEXITCODE -ne 0) { Get-Content $cfgLog -Tail 80; throw "CMake configure failed (exit $LASTEXITCODE), see $cfgLog" }
Write-Host "## CMake configure OK"
# print the vulkan lines from configure
Get-Content $cfgLog | Select-String -Pattern "Vulkan|vulkan|SPIRV" | Select-Object -First 40

# ---- 4) build kev-backend-vulkan only (single-node msbuild) ----
$proj = Join-Path $buildDir "kev-backend-vulkan.vcxproj"
if (-not (Test-Path $proj)) { throw "project not generated: $proj" }
$buildLog = Join-Path $buildDir "build.log"
$projCmd = "msbuild `"$proj`" /p:Configuration=$Config /v:m > `"$buildLog`" 2>&1"
Write-Host "## MSBuild kev-backend-vulkan ($Config, single-node)"
$buildCmd = "call `"$vcvars`" >nul 2>&1 && $projCmd"
$batch = Join-Path $env:TEMP "kevcpp_vkbuild_$PID.cmd"
Set-Content -Path $batch -Value $buildCmd -Encoding ascii
& cmd.exe /d /c "`"$batch`""
$code = $LASTEXITCODE
Remove-Item $batch -Force -ErrorAction SilentlyContinue
if ($code -ne 0) { Get-Content $buildLog -Tail 100; throw "MSBuild kev-backend-vulkan failed (exit $code), see $buildLog" }

Write-Host ""
Write-Host "## Build OK -> $buildDir\bin\Release\kev-backend-vulkan.dll"

# softbf16-v100-toolkit installer
#
#   .\install.ps1                       # full install (downloads CUDA headers, builds the kernel)
#   .\install.ps1 -CudaDir X:\cudalibs  # reuse an existing cudalibs folder (offline install)
#   .\install.ps1 -NoBuild              # stage files only, skip the CUDA build
#
param(
    [string]$Python = "D:\ComfyUI\python\python.exe",
    [string]$Comfy  = "D:\ComfyUI",
    [string]$CudaDir = "",
    [string]$Mirror = "https://mirrors.cloud.tencent.com/pypi/",
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$Pkg  = Split-Path -Parent $MyInvocation.MyCommand.Path
$Libs = Join-Path $Pkg "cudalibs"

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------------ python + torch
Step "python: $Python"
if (-not (Test-Path $Python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { Fail "python not found; pass -Python <path>" }
    $Python = $cmd.Source
}
& $Python -c "import torch; assert torch.cuda.is_available(); m,n=torch.cuda.get_device_capability(); print('torch',torch.__version__,'cuda',torch.version.cuda,'cc',m,n)"
if ($LASTEXITCODE -ne 0) { Fail "torch + CUDA check failed" }

$torchLib = & $Python -c "import os,torch;print(os.path.join(os.path.dirname(torch.__file__),'lib','cublas64_12.dll'))"
if (-not (Test-Path $torchLib)) { Fail "cublas64_12.dll not found next to torch: $torchLib" }

# ------------------------------------------------------------------ MSVC
Step "locating MSVC"
$vcvars = ""
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vs) { $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat" }
}
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    $fb = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $fb) { $vcvars = $fb }
}
if (-not $vcvars) { Fail "MSVC not found; install Visual Studio Build Tools with the C++ workload" }
Write-Host "    $vcvars"

# ------------------------------------------------------------------ CUDA headers
if (Test-Path (Join-Path $Libs "nvidia\cublas\include\cublas_v2.h")) {
    Step "CUDA headers already in $Libs"
} elseif ($CudaDir -and (Test-Path $CudaDir)) {
    Step "copying CUDA headers from $CudaDir"
    New-Item -ItemType Directory -Force -Path $Libs | Out-Null
    Copy-Item (Join-Path $CudaDir "*") $Libs -Recurse -Force
} else {
    Step "downloading CUDA headers (cublas/cusparse/cusolver, ~1.2 GB)"
    $wheels = @(
        "packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
        "packages/73/ef/063500c25670fbd1cbb0cd3eb7c8a061585b53adb4dd8bf3492bb49b0df3/nvidia_cusparse_cu12-12.5.10.65-py3-none-win_amd64.whl",
        "packages/32/5d/feb7f86b809f89b14193beffebe24cf2e4bf7af08372ab8cdd34d19a65a0/nvidia_cusolver_cu12-11.7.5.82-py3-none-win_amd64.whl"
    )
    $dl = Join-Path $Pkg "dl"
    New-Item -ItemType Directory -Force -Path $dl, $Libs | Out-Null
    foreach ($w in $wheels) {
        $name = Split-Path $w -Leaf
        $out  = Join-Path $dl $name
        if (-not (Test-Path $out)) {
            Write-Host "    $name"
            & curl.exe -sL --fail --retry 5 --retry-delay 3 -o $out ($Mirror + $w)
            if (-not (Test-Path $out)) { Fail "download failed: $name (retry, or pass -CudaDir)" }
        }
        & $Python -m zipfile -e $out $Libs
    }
}

if ($NoBuild) { Write-Host "`nFiles staged (build skipped)." -ForegroundColor Green; exit 0 }

# ------------------------------------------------------------------ cublas import library
Step "building cublas.lib"
$gen1 = Join-Path $Pkg "_gen1.bat"
@"
@echo off
call "$vcvars" >nul
dumpbin /nologo /exports "$torchLib" > "$Libs\cublas_exports.txt"
"@ | Set-Content $gen1 -Encoding ASCII
cmd /c "`"$gen1`""
& $Python (Join-Path $Pkg "tools\make_cublas_def.py") $Libs
if ($LASTEXITCODE -ne 0) { Fail "could not build cublas.def" }

$gen2 = Join-Path $Pkg "_gen2.bat"
@"
@echo off
call "$vcvars" >nul
if not exist "$Libs\lib" mkdir "$Libs\lib"
lib /nologo /def:"$Libs\cublas.def" /machine:x64 /out:"$Libs\lib\cublas.lib"
"@ | Set-Content $gen2 -Encoding ASCII
cmd /c "`"$gen2`""
if (-not (Test-Path (Join-Path $Libs "lib\cublas.lib"))) { Fail "cublas.lib generation failed" }

# ------------------------------------------------------------------ build + self test
$pyDir = Split-Path $Python -Parent
Step "compiling the CUDA extension (first build ~1-2 min)"
$build = Join-Path $Pkg "_build.bat"
@"
@echo off
chcp 65001 >nul
call "$vcvars" >nul
set PATH=$pyDir\Scripts;%PATH%
set PYTHONUTF8=1
"$Python" "$Pkg\tools\build_ext.py"
"@ | Set-Content $build -Encoding ASCII
cmd /c "`"$build`""
if ($LASTEXITCODE -ne 0) { Fail "extension build failed" }

Step "self test"
$test = Join-Path $Pkg "_selftest.bat"
@"
@echo off
chcp 65001 >nul
call "$vcvars" >nul
set PATH=$pyDir\Scripts;%PATH%
set PYTHONUTF8=1
"$Python" "$Pkg\tools\selftest.py"
"@ | Set-Content $test -Encoding ASCII
cmd /c "`"$test`""
if ($LASTEXITCODE -ne 0) { Fail "self test failed" }

Remove-Item (Join-Path $Pkg "_*.bat") -Force -ErrorAction SilentlyContinue

Write-Host "`nInstall complete." -ForegroundColor Green
Write-Host @"

ComfyUI with bf16 weights (separate instance, same port):
  `$env:SOFTBF16_COMFY = "$Comfy"
  $Python "$Pkg\shim\run_patched.py" --listen --preview-method auto --disable-cuda-malloc --bf16-unet

Training:
  import sys; sys.path.insert(0, r"$Pkg\shim"); import train_shim; train_shim.enable()

fp8 conversion:
  $Python "$Pkg\tools\fp8_convert.py" <src.safetensors> <dst.safetensors>

Free VRAM:
  $Python "$Pkg\tools\free_vram.py" 8188 --now

See README.md for the measured numbers and the caveats.
"@

# Removes build artifacts created by install.ps1.
# Nothing was ever written into ComfyUI itself, so there is nothing to restore there.
param([switch]$All)

$Pkg = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($d in @("cudalibs", "ext_cache", "dl")) {
    $p = Join-Path $Pkg $d
    if (Test-Path $p) {
        Remove-Item $p -Recurse -Force
        Write-Host "removed $d"
    }
}
Get-ChildItem $Pkg -Filter "_*.bat" -ErrorAction SilentlyContinue | Remove-Item -Force
if ($All) {
    Set-Location ..
    Remove-Item $Pkg -Recurse -Force
    Write-Host "removed the toolkit folder"
} else {
    Write-Host "Build artifacts removed; shim/tools/README left in place (pass -All to delete everything)."
}

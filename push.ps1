# Creates the GitHub repo (if missing) and pushes main.
#
#   $env:GH_TOKEN = "ghp_xxx"      # classic token with 'repo' scope
#   .\push.ps1
#
# Pass -Private if the repo should be private. The token is only used in the
# remote URL for this one push and is removed from .git/config afterwards.
param(
    [string]$User  = "LeoChen0621",
    [string]$Repo  = "softbf16-v100-toolkit",
    [string]$Token = $env:GH_TOKEN,
    [string]$Proxy = "http://127.0.0.1:7890",
    [switch]$Private
)

$ErrorActionPreference = "Stop"
if (-not $Token) { throw "set GH_TOKEN (classic PAT, 'repo' scope) or pass -Token" }

$headers = @{ Authorization = "Bearer $Token"; "User-Agent" = "softbf16-v100-toolkit"; Accept = "application/vnd.github+json" }

$exists = $true
try {
    Invoke-RestMethod "https://api.github.com/repos/$User/$Repo" -Headers $headers | Out-Null
} catch {
    $exists = $false
}

if (-not $exists) {
    $body = @{
        name        = $Repo
        private     = [bool]$Private
        description = "bf16/fp8 on Tesla V100 (sm70) for ComfyUI and training"
        has_issues  = $true
    } | ConvertTo-Json
    Invoke-RestMethod -Method Post "https://api.github.com/user/repos" -Headers $headers -Body $body -ContentType "application/json" | Out-Null
    Write-Host "created https://github.com/$User/$Repo"
} else {
    Write-Host "repo exists: https://github.com/$User/$Repo"
}

Push-Location $PSScriptRoot
try {
    git remote remove origin 2>$null
    git remote add origin "https://${User}:${Token}@github.com/$User/$Repo.git"
    git -c "http.proxy=$Proxy" push -u origin main
    if ($LASTEXITCODE -ne 0) { throw "push failed" }
} finally {
    git remote set-url origin "https://github.com/$User/$Repo.git"
    Pop-Location
}

Write-Host "pushed: https://github.com/$User/$Repo" -ForegroundColor Green
Write-Host "Revoke the token now if it was a one-off: https://github.com/settings/tokens"

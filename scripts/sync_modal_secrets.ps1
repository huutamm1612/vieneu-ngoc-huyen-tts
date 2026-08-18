[CmdletBinding()]
param(
    [string]$Environment = "main",
    [string]$SecretName = "huggingface-secret",
    [string]$EnvFile = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($EnvFile)) {
    $EnvFile = Join-Path $projectRoot ".env.modal"
}
$resolvedEnvFile = (Resolve-Path -LiteralPath $EnvFile -ErrorAction Stop).Path

$hfTokenLine = Get-Content -LiteralPath $resolvedEnvFile | Where-Object {
    $_ -match '^\s*HF_TOKEN\s*=\s*\S+'
} | Select-Object -First 1
if ($null -eq $hfTokenLine -or $hfTokenLine -match 'hf_replace_with_your_read_token') {
    throw "The env file must contain a real, non-empty HF_TOKEN value: $resolvedEnvFile"
}

$modalCommand = Get-Command "modal.exe" -ErrorAction SilentlyContinue
if ($null -eq $modalCommand) {
    $modalCommand = Get-Command "modal" -ErrorAction SilentlyContinue
}
if ($null -eq $modalCommand) {
    throw "Modal CLI was not found. Install modal==1.5.4 in your Python 3.12 environment."
}

Write-Host "Syncing local env file to Modal Secret..."
Write-Host "Environment : $Environment"
Write-Host "Secret      : $SecretName"
Write-Host "Source file : $resolvedEnvFile"

& $modalCommand.Source secret create `
    --env $Environment `
    --from-dotenv $resolvedEnvFile `
    --force `
    $SecretName
if ($LASTEXITCODE -ne 0) {
    throw "Modal exited with code $LASTEXITCODE"
}

Write-Host "Modal Secret synchronized. Token values were not printed."

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[^/\s]+/[^/\s]+$')]
    [string]$RepoId,
    [string]$RunName = "ngoc-a10-v1",
    [string]$Environment = "main",
    [string]$CommitMessage = "Upload Phase 3 final VieNeu-TTS model",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$modalCommand = Get-Command "modal.exe" -ErrorAction SilentlyContinue
if ($null -eq $modalCommand) {
    $modalCommand = Get-Command "modal" -ErrorAction SilentlyContinue
}
if ($null -eq $modalCommand) {
    throw "Modal CLI was not found. Install modal==1.5.4 in the active Python 3.12 environment."
}

$modalApp = Join-Path $projectRoot "cloud\modal_hf_upload.py"
$modalArguments = @(
    "run",
    "--env", $Environment,
    $modalApp,
    "--repo-id", $RepoId,
    "--run-name", $RunName,
    "--commit-message", $CommitMessage
)
if ($Public) {
    $modalArguments += "--public"
}

$visibility = if ($Public) { "public" } else { "private" }
Write-Host "Source      : tts-training-results:/runs/$RunName/phase3_finetune/final"
Write-Host "Destination : https://huggingface.co/$RepoId"
Write-Host "Visibility  : $visibility"
Write-Host "Environment : $Environment"

Push-Location $projectRoot
try {
    & $modalCommand.Source @modalArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Modal Hugging Face upload failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

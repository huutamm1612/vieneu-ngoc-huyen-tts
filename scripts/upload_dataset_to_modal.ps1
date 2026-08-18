[CmdletBinding()]
param(
    [string]$VolumeName = "tts-dataset",
    [string]$LocalDataPath = (Join-Path $PSScriptRoot "..\data"),
    [string]$RemotePath = "/dataset",
    [string]$Environment = "main",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command modal -ErrorAction SilentlyContinue)) {
    throw "Modal CLI is not installed or is not available in PATH."
}

$resolvedDataPath = (Resolve-Path -LiteralPath $LocalDataPath).Path
if (-not (Test-Path -LiteralPath $resolvedDataPath -PathType Container)) {
    throw "Dataset directory does not exist: $resolvedDataPath"
}

$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& modal profile current | Out-Host
$modalExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($modalExitCode -ne 0) {
    throw "Modal is not authenticated. Run 'modal setup' once, then retry."
}

$files = @(Get-ChildItem -LiteralPath $resolvedDataPath -File -Recurse)
$wavFiles = @($files | Where-Object { $_.Extension -ieq ".wav" })
$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
$totalGiB = [math]::Round($totalBytes / 1GB, 3)

Write-Host ""
Write-Host "Local dataset : $resolvedDataPath"
Write-Host "Files         : $($files.Count) total, $($wavFiles.Count) WAV"
Write-Host "Size          : $totalGiB GiB"
Write-Host "Modal target  : $Environment / $VolumeName / $RemotePath"
Write-Host ""

$ErrorActionPreference = "Continue"
& modal volume ls --env $Environment $VolumeName 2>$null | Out-Null
$volumeLookupExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($volumeLookupExitCode -ne 0) {
    Write-Host "Creating Modal Volume '$VolumeName' in environment '$Environment'..."
    $ErrorActionPreference = "Continue"
    & modal volume create --env $Environment $VolumeName
    $modalExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($modalExitCode -ne 0) {
        throw "Could not find or create Modal Volume '$VolumeName' (exit code $modalExitCode). Check the connection, profile, and environment."
    }
}

$uploadArgs = @(
    "volume", "put",
    "--env", $Environment
)
if ($Force) {
    $uploadArgs += "--force"
}
$uploadArgs += @(
    $VolumeName,
    $resolvedDataPath,
    $RemotePath
)

Write-Host "Uploading dataset to Modal..."
$ErrorActionPreference = "Continue"
& modal @uploadArgs
$modalExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($modalExitCode -ne 0) {
    throw "Dataset upload failed with exit code $modalExitCode."
}

Write-Host ""
Write-Host "Upload completed. Verifying remote dataset root..."
$ErrorActionPreference = "Continue"
& modal volume ls --env $Environment $VolumeName $RemotePath
$modalExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($modalExitCode -ne 0) {
    throw "Upload finished, but remote verification failed."
}

Write-Host ""
Write-Host "Dataset is available in Modal Volume '$VolumeName' at '$RemotePath'."

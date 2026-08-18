[CmdletBinding()]
param(
    [string]$RunName = "ngoc-a10-v1",
    [string]$Config = "pipeline_3phase.yaml",
    [string]$Environment = "main",
    [Nullable[double]]$Phase1LearningRate = $null,
    [Nullable[double]]$Phase1Epochs = $null,
    [Nullable[double]]$Phase3LearningRate = $null,
    [Nullable[double]]$Phase3Epochs = $null
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$modalCommand = Get-Command "modal.exe" -ErrorAction SilentlyContinue
if ($null -eq $modalCommand) {
    $modalCommand = Get-Command "modal" -ErrorAction SilentlyContinue
}
if ($null -eq $modalCommand) {
    throw "Modal CLI was not found. Install modal==1.5.4 in your Python 3.12 environment."
}

$modalApp = Join-Path $projectRoot "cloud\modal_app.py"
$modalArguments = @(
    "run",
    "--detach",
    "--env", $Environment,
    $modalApp,
    "--config", $Config,
    "--run-name", $RunName
)

if ($null -ne $Phase1LearningRate) {
    $modalArguments += @("--phase1-lr", $Phase1LearningRate.ToString([Globalization.CultureInfo]::InvariantCulture))
}
if ($null -ne $Phase1Epochs) {
    $modalArguments += @("--phase1-epochs", $Phase1Epochs.ToString([Globalization.CultureInfo]::InvariantCulture))
}
if ($null -ne $Phase3LearningRate) {
    $modalArguments += @("--phase3-lr", $Phase3LearningRate.ToString([Globalization.CultureInfo]::InvariantCulture))
}
if ($null -ne $Phase3Epochs) {
    $modalArguments += @("--phase3-epochs", $Phase3Epochs.ToString([Globalization.CultureInfo]::InvariantCulture))
}

Write-Host "Stage 1: NeuCodec on Modal T4 (4 CPU cores, 6 GiB RAM)"
Write-Host "Stage 2: all three training phases on A10 (4 CPU cores, 6 GiB RAM)"
Write-Host "Run name       : $RunName"
Write-Host "Dataset Volume : tts-dataset:/dataset"
Write-Host "Results Volume : tts-training-results:/runs/$RunName"

Push-Location $projectRoot
try {
    & $modalCommand.Source @modalArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Modal exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$ReferenceAudio,

    [Parameter(Mandatory = $true)]
    [string]$ReferenceText,

    [string]$OutputPath = ".\outputs\inference.wav",
    [string]$Devices = "auto",
    [int]$NumGpus = 1,
    [string]$Config = ".\configs\inference.yaml"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Push-Location $projectRoot
try {
    python -m inference `
        --config $Config `
        --model $Model `
        --input $InputPath `
        --reference-audio $ReferenceAudio `
        --reference-text $ReferenceText `
        --output $OutputPath `
        --devices $Devices `
        --num-gpus $NumGpus
    if ($LASTEXITCODE -ne 0) {
        throw "Inference failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

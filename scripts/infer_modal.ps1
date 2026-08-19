param(
    [string]$Model = "/mnt/tts-results/runs/ngoc-a10-v1/phase3_finetune/final",
    [string]$InputPath = "story.txt",
    [string]$OutputPath = "story_complete.wav",
    [string]$ReferenceAudio = "/mnt/tts-dataset/dataset/audio/ngochuyen_00769.wav",
    [string]$ReferenceText = "",
    [ValidateRange(1, 8)]
    [int]$NumGpus = 1,
    [string]$Gpu = "A10",
    [int]$BatchSize = 128,
    [int]$MaxRuntimeBatchSize = 0,
    [string]$Environment = "main"
)

$ErrorActionPreference = "Stop"
$defaultReferenceTextBase64 = "VsOsIHRo4bq/LCDEkOG7k25nIGNow60gbHXDtG4gxJHGsOG7o2MgY8OhYyDEkOG7k25nIGNow60gbMOjbmggxJHhuqFvIGPhuqVwIGNhbyBj4bunYSDEkOG6o25nIHRpbiB0xrDhu59uZywgxJHDoW5oIGdpw6EgY2FvLg=="
if ([string]::IsNullOrWhiteSpace($ReferenceText)) {
    $ReferenceText = [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String($defaultReferenceTextBase64)
    )
}
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Push-Location $projectRoot
try {
    modal run --detach --env $Environment cloud/modal_inference.py `
        --model $Model `
        --input-path $InputPath `
        --output $OutputPath `
        --reference-audio $ReferenceAudio `
        --reference-text $ReferenceText `
        --gpu $Gpu `
        --num-gpus $NumGpus `
        --batch-size $BatchSize `
        --max-runtime-batch-size $MaxRuntimeBatchSize
    if ($LASTEXITCODE -ne 0) {
        throw "Modal inference failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [string]$RemoteName = "story.txt",
    [string]$VolumeName = "tts-inference-inputs",
    [string]$Environment = "main"
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path -LiteralPath $InputPath).Path
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Input TXT not found: $source"
}

$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
modal volume ls --env $Environment $VolumeName *> $null
$lookupExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($lookupExitCode -ne 0) {
    modal volume create --env $Environment $VolumeName
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create Modal Volume $VolumeName"
    }
}
modal volume put --env $Environment --force $VolumeName $source "/$RemoteName"
if ($LASTEXITCODE -ne 0) {
    throw "Could not upload $source to Modal Volume $VolumeName"
}
modal volume ls --env $Environment $VolumeName "/$RemoteName"

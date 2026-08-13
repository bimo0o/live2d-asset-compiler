param(
    [Parameter(Mandatory=$true)]
    [string]$RunId
)

$ErrorActionPreference = "Stop"
$runDir = Join-Path "output" $RunId
if (-not (Test-Path -LiteralPath $runDir)) {
    throw "Run directory does not exist: $runDir"
}

$archive = Join-Path "output" "$RunId.zip"
if (Test-Path -LiteralPath $archive) {
    Remove-Item -LiteralPath $archive
}

Compress-Archive -LiteralPath $runDir -DestinationPath $archive
Write-Host $archive


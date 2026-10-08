$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$credentialPath = Join-Path (Join-Path $workspacePath 'work') 'kaggle-token.dpapi'
$previousToken = $env:KAGGLE_API_TOKEN
$secureToken = (Get-Content -LiteralPath $credentialPath -Raw).Trim() | ConvertTo-SecureString
$tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
try {
    $env:KAGGLE_API_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
    & (Join-Path $PSScriptRoot 'stage-lightning.ps1') -Mode prepare
    $commandExitCode = $LASTEXITCODE
}
finally {
    $env:KAGGLE_API_TOKEN = $previousToken
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
    $secureToken.Dispose()
}
exit $commandExitCode



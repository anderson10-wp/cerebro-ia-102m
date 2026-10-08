$ErrorActionPreference = 'Stop'
$credentialPath = Join-Path $PSScriptRoot 'kaggle-token.dpapi'
$previousToken = $env:KAGGLE_API_TOKEN
$secureToken = (Get-Content -LiteralPath $credentialPath -Raw).Trim() | ConvertTo-SecureString
$tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
try {
    $env:KAGGLE_API_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
    & kaggle @args
    $commandExitCode = $LASTEXITCODE
}
finally {
    $env:KAGGLE_API_TOKEN = $previousToken
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
    $secureToken.Dispose()
}
exit $commandExitCode

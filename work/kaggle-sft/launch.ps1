$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1')
$previousToken = $env:KAGGLE_API_TOKEN
$secureToken = (Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\kaggle-token.dpapi') -Raw).Trim() | ConvertTo-SecureString
$tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
try {
    $env:KAGGLE_API_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
    & 'C:\Users\Fernando\AppData\Local\Programs\Python\Python314\python.exe' (Join-Path $PSScriptRoot 'launch.py') @args
    $code = $LASTEXITCODE
} finally {
    $env:KAGGLE_API_TOKEN = $previousToken
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
    $secureToken.Dispose()
}
exit $code

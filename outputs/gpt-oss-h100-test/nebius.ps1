param([ValidateSet('inspect','prepare','submit','status','download','stop','billing','complete-transfer','resume','capacity','run','cleanup')][string]$Mode)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$previousKey = $env:LIGHTNING_API_KEY
$previousUser = $env:LIGHTNING_USER_ID
$secureKey = (Get-Content (Join-Path $workspacePath 'work/lightning-api-key.dpapi') -Raw).Trim() | ConvertTo-SecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:LIGHTNING_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $env:LIGHTNING_USER_ID = '8de7254a-490e-40e7-b77c-6f2de340dcb2'
    & python (Join-Path $PSScriptRoot 'nebius_test.py') $Mode
    $cloudExit = $LASTEXITCODE
}
finally {
    $env:LIGHTNING_API_KEY = $previousKey
    $env:LIGHTNING_USER_ID = $previousUser
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secureKey.Dispose()
}
exit $cloudExit




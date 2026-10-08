$ErrorActionPreference = 'Stop'
$previousKey = $env:LIGHTNING_API_KEY
$previousUser = $env:LIGHTNING_USER_ID
$secureKey = (Get-Content (Join-Path $PSScriptRoot 'lightning-api-key.dpapi') -Raw).Trim() | ConvertTo-SecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:LIGHTNING_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $env:LIGHTNING_USER_ID = '8de7254a-490e-40e7-b77c-6f2de340dcb2'
    & python (Join-Path (Split-Path $PSScriptRoot -Parent) 'outputs/lightning-inspection.py')
    $checkExit = $LASTEXITCODE
}
finally {
    $env:LIGHTNING_API_KEY = $previousKey
    $env:LIGHTNING_USER_ID = $previousUser
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secureKey.Dispose()
}
exit $checkExit

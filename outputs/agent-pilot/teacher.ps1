$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$oldKey = $env:GPT_OSS_LOCAL_KEY
$secret = (Get-Content (Join-Path $workspacePath 'work/gpt-oss-api-key.dpapi') -Raw).Trim() | ConvertTo-SecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $env:GPT_OSS_LOCAL_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    & python (Join-Path $PSScriptRoot 'routing.py') teacher
    $teacherExit = $LASTEXITCODE
}
finally {
    $env:GPT_OSS_LOCAL_KEY = $oldKey
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secret.Dispose()
}
exit $teacherExit

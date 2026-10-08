param([int]$Seed=106)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$oldKey=$env:GEMINI_API_KEY
$secret=(Get-Content (Join-Path $workspacePath 'work/gemini-api-key.dpapi') -Raw).Trim() | ConvertTo-SecureString
$pointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $env:GEMINI_API_KEY=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    & 'C:\Users\Fernando\AppData\Local\Programs\Python\Python314\python.exe' (Join-Path $PSScriptRoot 'teacher.py') --seed $Seed
    $teacherExit=$LASTEXITCODE
} finally {
    $env:GEMINI_API_KEY=$oldKey
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secret.Dispose()
}
exit $teacherExit

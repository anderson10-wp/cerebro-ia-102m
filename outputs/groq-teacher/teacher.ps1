param([int]$Seed=203)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
$workspacePath=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$oldKey=$env:GROQ_API_KEY
$secret=(Get-Content (Join-Path $workspacePath 'work/groq-api-key.dpapi') -Raw).Trim() | ConvertTo-SecureString
$pointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $env:GROQ_API_KEY=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    & 'C:\Users\Fernando\AppData\Local\Programs\Python\Python314\python.exe' (Join-Path $workspacePath 'outputs/gemini-teacher/teacher.py') --provider groq --seed $Seed
    $teacherExit=$LASTEXITCODE
} finally {
    $env:GROQ_API_KEY=$oldKey
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secret.Dispose()
}
exit $teacherExit


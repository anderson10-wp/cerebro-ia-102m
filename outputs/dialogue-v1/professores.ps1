param([ValidateSet('prepare','generate','dev','review','collect','spot','judge','bridge','inspect-gemini')][string]$Action='generate',[int]$Limit=100,[ValidateSet('groq','openrouter','gemini')][string]$Provider='groq',[ValidateSet('openai/gpt-oss-120b','openai/gpt-oss-20b')][string]$Model='openai/gpt-oss-120b')
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
$workspacePath=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$savedGroq=$env:GROQ_API_KEY; $savedRouter=$env:OPENROUTER_API_KEY; $savedGemini=$env:GEMINI_API_KEY
$pointers=@(); $secrets=@()
try {
    $providersToLoad=@('groq','openrouter')
    if ($Action -eq 'inspect-gemini' -or $Provider -eq 'gemini') { $providersToLoad+= 'gemini' }
    foreach ($secretProvider in $providersToLoad) {
        $secret=(Get-Content -LiteralPath (Join-Path $workspacePath "work/$secretProvider-api-key.dpapi") -Raw).Trim() | ConvertTo-SecureString
        $secrets+=$secret
        $pointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret); $pointers+=$pointer
        $value=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        if ($secretProvider -eq 'groq') {$env:GROQ_API_KEY=$value} elseif ($secretProvider -eq 'openrouter') {$env:OPENROUTER_API_KEY=$value} else {$env:GEMINI_API_KEY=$value}
    }
    if ($Action -eq 'inspect-gemini') {
        & (Join-Path $workspacePath '.venv-directml/Scripts/python.exe') (Join-Path $PSScriptRoot 'inspect_gemini.py')
    } elseif ($Action -eq 'judge') {
        & (Join-Path $workspacePath '.venv-directml/Scripts/python.exe') (Join-Path $PSScriptRoot 'judge.py')
    } else {
        & (Join-Path $workspacePath '.venv-directml/Scripts/python.exe') (Join-Path $PSScriptRoot 'pipeline.py') $Action --limit $Limit --provider $Provider --groq-model $Model
    }
    $resultCode=$LASTEXITCODE
} finally {
    $env:GROQ_API_KEY=$savedGroq; $env:OPENROUTER_API_KEY=$savedRouter; $env:GEMINI_API_KEY=$savedGemini
    foreach ($pointer in $pointers) {[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)}
    foreach ($secret in $secrets) {$secret.Dispose()}
}
exit $resultCode

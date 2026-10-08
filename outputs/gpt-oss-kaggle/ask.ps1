param([string]$Prompt = 'Responda em português: quanto é 7 vezes 8?')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$endpoint = (Get-Content (Join-Path $PSScriptRoot 'endpoint.json') -Raw | ConvertFrom-Json).base_url
$previousKey = $env:GPT_OSS_LOCAL_KEY
$secureKey = (Get-Content (Join-Path $projectRoot 'work/gpt-oss-api-key.dpapi') -Raw).Trim() | ConvertTo-SecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:GPT_OSS_LOCAL_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    & python (Join-Path $PSScriptRoot 'client.py') --endpoint $endpoint --prompt $Prompt
    $requestExitCode = $LASTEXITCODE
}
finally {
    $env:GPT_OSS_LOCAL_KEY = $previousKey
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    $secureKey.Dispose()
}
exit $requestExitCode

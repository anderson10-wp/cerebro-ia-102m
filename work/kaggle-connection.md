Kaggle access for this workspace

Run Kaggle CLI commands through `powershell -NoProfile -File ./work/kaggle.ps1 <arguments>`.
The provided API token is encrypted in `work/kaggle-token.dpapi` using Windows DPAPI for the current Windows user. The wrapper decrypts it only for the child CLI process and preserves the pre-existing environment setting. Do not print or copy the token into logs or deliverables.

Authenticated account: andersonluan10ad.

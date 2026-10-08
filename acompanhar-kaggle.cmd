@echo off
cd /d "%~dp0"
powershell -NoProfile -File "%~dp0work\kaggle.ps1" kernels status andersonluan10ad/cerebro-102m-sft-treino-oct08
powershell -NoProfile -File "%~dp0work\kaggle.ps1" kernels logs andersonluan10ad/cerebro-102m-sft-treino-oct08
pause

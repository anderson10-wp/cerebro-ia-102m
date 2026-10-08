@echo off
chcp 65001 >nul
"%~dp0.venv-directml\Scripts\python.exe" "%~dp0outputs\h100-pretrain\chat_terminal.py" --device directml
if errorlevel 1 pause

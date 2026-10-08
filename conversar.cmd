@echo off
chcp 65001 >nul
"C:\Users\Fernando\AppData\Local\Programs\Python\Python314\python.exe" "%~dp0outputs\h100-pretrain\chat_terminal.py"
if errorlevel 1 pause

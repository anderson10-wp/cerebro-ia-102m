@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0outputs\groq-teacher\teacher.ps1" %*

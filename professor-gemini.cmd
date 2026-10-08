@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0outputs\gemini-teacher\teacher.ps1" %*

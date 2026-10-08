@echo off
chcp 65001 >nul
"%~dp0.venv-directml\Scripts\python.exe" -u "%~dp0outputs\auto_train.py" %*
set "auto_exit=%errorlevel%"
echo.
pause
exit /b %auto_exit%



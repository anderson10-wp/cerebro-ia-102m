@echo off
chcp 65001 >nul
"%~dp0.venv-directml\Scripts\python.exe" -u "%~dp0outputs\train_local.py" --base-run "%~dp0outputs\local-run-20261007-190808-8c59a4" %*
set "curriculum_exit=%errorlevel%"
echo.
pause
exit /b %curriculum_exit%



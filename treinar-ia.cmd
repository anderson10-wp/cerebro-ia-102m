@echo off
"%~dp0.venv-directml\Scripts\python.exe" -u "%~dp0outputs\train_local.py" %*
set "training_exit=%errorlevel%"
echo.
if not "%training_exit%"=="0" echo A rodada falhou ou foi interrompida. Consulte o registro informado acima.
pause
exit /b %training_exit%



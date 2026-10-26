@echo off
title Cerebro MoE 500M - AMD Radeon RX 580 (DirectML)
chcp 65001 > nul
cd /d "C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m"
cls
echo ======================================================================
echo    INICIANDO CEREBRO MoE 500M NA AMD RADEON RX 580 (DIRECTML)
echo ======================================================================
echo.
"C:\Users\Fernando\AppData\Local\Programs\Python\Python312\python.exe" chat_cerebro_moe.py
pause

@echo off
chcp 65001 >nul
rem Conversa com a versao em testes (candidato local treinado). O modelo oficial continua no conversar.cmd
"C:\Users\Fernando\AppData\Local\Programs\Python\Python314\python.exe" "%~dp0outputs\h100-pretrain\chat_terminal.py" --historico --checkpoint "%~dp0outputs\candidato-atual\best-model.pt"
if errorlevel 1 pause

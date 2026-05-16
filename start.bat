@echo off
title MNQ Divergence Server
cd /d "%~dp0"
echo Iniciando servidor MNQ-CL...
start /b node server.js
timeout /t 2 /nobreak >nul
start "" "http://localhost:3000"
echo Servidor rodando em http://localhost:3000
echo Pressione qualquer tecla para encerrar...
pause >nul
taskkill /f /im node.exe >nul 2>&1

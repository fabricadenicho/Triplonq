@echo off
title Triplonq — Live2 PropFirm
cd /d "%~dp0"
echo Iniciando servidor Triplonq...

set "PATH=%PATH%;%ProgramFiles%\nodejs;C:\Program Files\nodejs;%AppData%\npm"

start /b "" node server.js
if errorlevel 1 (
    echo ERRO: Nao foi possivel iniciar o servidor Node.js
    echo Verifique se o Node.js esta instalado: node --version
    pause
    exit /b 1
)

timeout /t 3 /nobreak >nul

start "" "http://localhost:3000/live2"
echo Servidor rodando em http://localhost:3000/live2
echo Pressione qualquer tecla para encerrar...
pause >nul
taskkill /f /im node.exe >nul 2>&1

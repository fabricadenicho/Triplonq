@echo off
title MNQ Divergence Server
cd /d "%~dp0"
echo Iniciando servidor MNQ-CL...

:: Garante que o Node estara no PATH (tanto sistema quanto usuario)
set "PATH=%PATH%;%ProgramFiles%\nodejs;C:\Program Files\nodejs;%AppData%\npm"

:: Inicia o servidor em segundo plano com /b (corrigido: aspas duplas no titulo)
start /b "" node server.js
if errorlevel 1 (
    echo ERRO: Nao foi possivel iniciar o servidor Node.js
    echo Verifique se o Node.js esta instalado: node --version
    pause
    exit /b 1
)

timeout /t 3 /nobreak >nul

:: Abre o navegador
start "" "http://localhost:3000/mnqcl.html"
echo Servidor rodando em http://localhost:3000/mnqcl.html
echo Pressione qualquer tecla para encerrar...
pause >nul
taskkill /f /im node.exe >nul 2>&1

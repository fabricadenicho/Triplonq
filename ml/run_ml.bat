@echo off
title MNQ — ML Pipeline
cd /d "%~dp0"

echo ════════════════════════════════════
echo  MNQ Divergence — ML Pipeline
echo ════════════════════════════════════
echo.

echo [1/3] Instalando dependencias...
pip install -r requirements.txt -q
if errorlevel 1 ( echo ERRO na instalacao & pause & exit /b 1 )

echo.
echo [2/3] Coletando dados historicos (yfinance)...
python collect_data.py
if errorlevel 1 ( echo ERRO na coleta & pause & exit /b 1 )

echo.
echo [3/3] Treinando modelo XGBoost...
python train.py %*
if errorlevel 1 ( echo ERRO no treino & pause & exit /b 1 )

echo.
echo ════════════════════════════════════
echo  Concluido! Arquivos gerados:
echo    data.db               (dados historicos)
echo    model.pkl             (modelo treinado)
echo    feature_importance.png (grafico)
echo ════════════════════════════════════
pause

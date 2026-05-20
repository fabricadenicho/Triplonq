# Backtest — CL PropFirm ML

**Arquivos:** `strategy.pine` / `overlay.pine`
**Período:** Últimos 30 dias (gerado automaticamente)
**Ativo:** Crude Oil Futures (CL1!)

## strategy.pine

Script `strategy()` para o **Strategy Tester** do TradingView com todos os trades LONG do modelo ML.

**Resultado do período atual:** 5 trades | WR 80% | AvgR +1.30R

**Como usar:** Colar no Pine Editor → aba "Strategy Tester".

## overlay.pine

Indicador overlay com sinais ML LONG no gráfico de preço:
- Setas triangulares verdes nas entradas LONG
- Linhas de stop (vermelho) e target (verde)
- Fundo verde durante posição ativa

## Características do CL

- **LONG only** — modelo treinado apenas para compra
- Volume reduzido (~5 trades/mês) mas WR historicamente alto
- Melhor horário: antes das 14h UTC (London + abertura NY)

## Como regenerar

```bash
cd ml/teste/backtest_ml
python gerar_pine.py
```

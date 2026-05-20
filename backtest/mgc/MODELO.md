# Backtest — MGC PropFirm ML

**Arquivos:** `strategy.pine` / `overlay.pine`
**Período:** Últimos 30 dias (gerado automaticamente)
**Ativo:** Micro Gold Futures (MGC1!)

## strategy.pine

Script `strategy()` para o **Strategy Tester** do TradingView com todos os trades LONG e SHORT do modelo ML.

**Resultado do período atual:** 15 trades | WR 46.7% | AvgR +0.13R

**Como usar:** Colar no Pine Editor → aba "Strategy Tester".

## overlay.pine

Indicador overlay com sinais ML no gráfico de preço:
- Verde=LONG, Vermelho=SHORT nas entradas
- Linhas de stop e target durante posição
- Marcadores de resultado (WIN/LOSS/EXPIRED)

## Características do MGC

- **BOTH** — opera LONG e SHORT
- Feature mais importante: `dist_to_mo < 0.49%` (Monthly Open)
- Maior número de trades após BTC (~15/mês)

## Como regenerar

```bash
cd ml/teste/backtest_ml
python gerar_pine.py
```

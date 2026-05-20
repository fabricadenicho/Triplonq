# Backtest — BTC PropFirm ML

**Arquivos:** `strategy.pine` / `overlay.pine`
**Período:** Últimos 30 dias (gerado automaticamente)
**Ativo:** Bitcoin (BTCUSDT)

## strategy.pine

Script `strategy()` para o **Strategy Tester** do TradingView com todos os trades SHORT do modelo ML.

**Como usar:** Colar no Pine Editor → aba "Strategy Tester".

## overlay.pine

Indicador overlay com sinais ML no gráfico de preço:
- Setas nas entradas (vermelho=SHORT para BTC)
- Linhas de stop e target durante posição
- Marcadores de resultado no exit

## Características do BTC

- **Maior volume de trades** — ~20 por mês
- **Direção:** SHORT only no modelo atual
- Trades gerados pelo XGBoost com features: RSI, ret4, BB position, DOW, ADX, DI

## Como regenerar

```bash
cd ml/teste/backtest_ml
python gerar_pine.py
```

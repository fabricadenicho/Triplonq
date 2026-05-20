# Backtest — MNQ PropFirm ML

**Arquivos:** `strategy.pine` / `overlay.pine`
**Período:** Últimos 30 dias (gerado automaticamente)
**Ativo:** Micro Nasdaq Futures (MNQ1!)

## strategy.pine

Script `strategy()` para o **Strategy Tester** do TradingView. Mostra entrada/saída com setas no gráfico, P&L cumulativo e métricas.

**Como usar:** Colar no Pine Editor do TradingView → aba "Strategy Tester" aparece automaticamente.

## overlay.pine

Indicador `indicator(overlay=true)` que plota os sinais ML no gráfico de preço:
- Setas triangulares nas entradas (verde=LONG, vermelho=SHORT)
- Linhas de stop (vermelho) e target (verde) durante posição ativa
- Fundo colorido durante posição aberta
- Marcadores no exit (diamond=WIN, cross=LOSS, xcross=EXPIRED)

**Como usar:** Adicionar como indicador separado no mesmo gráfico do MNQ.

## Resultados do período atual

Gerados pelo modelo ML PropFirm. Ver painel na tabela do strategy.pine após carregar no TradingView.

## Como regenerar

```bash
cd ml/teste/backtest_ml
python gerar_pine.py
```

Os arquivos são sobrescritos automaticamente com os trades do último mês.

# Histograma — BTC PropFirm

**Arquivo:** `histograma - btc.pine`
**Tipo:** Pane separado (overlay=false)
**Ativo:** Bitcoin (BTCUSDT)

## O que faz

Painel completo de análise do BTC. Mostra divergência BTC-MNQ, features do modelo ML (ret4, BB position, DOW), ML Score SHORT em tempo real e sinais históricos dos últimos 30 dias.

## ML Score (alinhado com a árvore XGBoost)

**Máximo: 9 pontos — direção: SHORT**

| Condição | Peso | Descrição |
|---|---|---|
| `RSI >= 55` | 2 | Sobrecomprado (melhor para SHORT) |
| `ret4 < -0.5%` | 2 | Retorno 4 barras negativo (momentum bear) |
| `DI- > DI+` | 2 | Direcional bearish (DMI) |
| `ADX >= limiar` | 1 | Tendência forte |
| `BB position > 0.7` | 1 | Próximo da banda superior (sobrecomprado) |
| `vol_btc < 0.4%` | 1 | Volume baixo |

**Sinal:** score >= 6 = ML VENDER BTC | >= 4 = parcial (amarelo)

## Key Levels monitorados

- `pdh / pdl` — Previous Day High/Low
- `mo` — Monthly Open
- `wo` — Weekly Open
- `id_h / id_l` — Intraday High/Low

## Features adicionais no painel

- **ret4** — retorno das últimas 4 barras
- **BB position** — onde o preço está dentro das Bandas de Bollinger (0=baixo, 1=topo)
- **DOW** — dia da semana (seg/sex = bias SHORT, qua = bias LONG)
- **dist_to_mo** — distância percentual ao Monthly Open

## Performance ML

| Métrica | Valor |
|---|---|
| AUC | 0.555 |
| Acurácia | 39% |

> BTC tem o maior volume de trades (20 no último mês). RETORNOS representam 19% do peso da árvore.

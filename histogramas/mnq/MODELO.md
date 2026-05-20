# Histograma — MNQ PropFirm

**Arquivo:** `histograma - mnq.pine`
**Tipo:** Pane separado (overlay=false)
**Ativo:** Micro Nasdaq Futures (MNQ1!)

## O que faz

Painel completo de análise do MNQ em pane separado. Mostra divergência CL-MNQ, Key Levels, ML Score em tempo real e sinais históricos do modelo ML (últimos 30 dias injetados como marcadores).

## ML Score (alinhado com a árvore XGBoost)

**Máximo: 6 pontos por direção (LONG ou SHORT)**

| Condição | Peso | Descrição |
|---|---|---|
| `vol_mnq < 0.334%` | 2 | Volume normalizado baixo |
| `sma50 >= 2 AND DI+` (LONG) | 2 | Alinhamento bullish + DMI favorável |
| `sma50 == 0 AND DI-` (SHORT) | 2 | Alinhamento bearish + DMI favorável |
| `div CL-MNQ > 2` | 1 | Divergência ativa |
| `in_us` (13-20h UTC) | 1 | Dentro da sessão US |

**Sinal:** score >= 4 = ML LONG/SHORT | >= 3 = parcial (amarelo)

## Key Levels monitorados

- `pdh / pdl` — Previous Day High/Low
- `pmh / pml` — Previous Month High/Low
- `mo` — Monthly Open (abertura do mês)
- `wo` — Weekly Open (abertura da semana)
- `id_h / id_l` — Intraday High/Low (resetam todo dia)

## Sinais históricos ML injetados

Marcadores azul/vermelho nas barras dos últimos 30 dias mostrando onde o modelo ML realmente entrou. Fundo verde/vermelho durante posição ativa.

## Performance ML

| Métrica | Valor |
|---|---|
| AUC | 0.602 |
| Acurácia | 43% |

> Melhor AUC do projeto. KEY LEVELS representam 34% do peso da árvore.

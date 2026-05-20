# Histograma — CL PropFirm

**Arquivo:** `histograma - cl.pine`
**Tipo:** Pane separado (overlay=false)
**Ativo:** Crude Oil Futures (CL1!)

## O que faz

Painel completo de análise do CL. Mostra divergência CL-MNQ, Key Levels, horário de trading, ML Score LONG em tempo real e sinais históricos dos últimos 30 dias (5 trades, WR 80%).

## ML Score (alinhado com a árvore XGBoost)

**Máximo: 9 pontos — direção: LONG**

| Condição | Peso | Descrição |
|---|---|---|
| `hora < 14h UTC` | 2 | Melhor horário (London + início NY) |
| `vol_cl < 0.402%` | 2 | Volume normalizado baixo |
| `prev_day_range < 0.54%` | 2 | Range do dia anterior comprimido |
| `dist Monthly Open < 3%` | 1 | Próximo ao Monthly Open (KEY LEVEL) |
| `dist PDL < 1.5%` (near PDL) | 1 | Próximo ao Previous Day Low |
| `DI+ > DI-` | 1 | Direcional bullish (DMI) |

**Sinal:** score >= 6 = ML COMPRAR CL | >= 4 = parcial (amarelo)

## Key Levels monitorados

- `pdh / pdl` — Previous Day High/Low
- `pmh / pml` — Previous Month High/Low
- `mo` — Monthly Open
- `wo` — Weekly Open
- `id_h / id_l` — Intraday High/Low
- `dist_to_mo_v` — distância % ao Monthly Open (exibida no painel)

## Características do CL

- **LONG only** — o modelo ML do CL só opera compras
- **KEY LEVELS = 37%** do peso da árvore (maior entre todos os ativos)
- **TEMPORAL = 13%** — hora do dia muito importante
- Evening session CL: 23h-02h UTC (óleo opera 23h/dia)

## Performance ML

| Métrica | Valor |
|---|---|
| AUC | 0.474 |
| Acurácia | 52% |
| WR último mês | 80% (4/5 trades) |

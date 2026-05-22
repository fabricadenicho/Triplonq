# Histograma — MNQ PropFirm

**Arquivo:** `histograma - mnq.pine`
**Tipo:** Pane separado (overlay=false)
**Ativo:** Micro Nasdaq Futures (MNQ1!)
**Atualizado:** 2026-05-22 (modelo sqlite_clean)

## ML Score (modelo sqlite_clean 2026-05-22)

**Máximo: 6 pontos por direção (LONG ou SHORT)**

| Condição | Peso | Feature do modelo |
|---|---|---|
| `ADX MNQ > 17` | 2 | adx_p: 3.8% (top 2) |
| `sma50_align ≥ 2 AND DI+` (LONG) | 2 | sma50_alignment (1º split da árvore) |
| `sma50_align = 0 AND DI-` (SHORT) | 2 | sma50_alignment |
| `vol_mnq < 0.370%` | 1 | vol_p: 3.7% |
| `prev_day_range < 3.5%` | 1 | prev_day_range_pct: 4.3% (top 1) |

**Sinal ativo:** score ≥ 4 → ML COMPRAR / ML VENDER MNQ

## Importância por categoria (modelo atual)

| Categoria | Peso |
|---|---|
| KEY LEVELS | 29.1% |
| VOLATILIDADE | 16.5% |
| ADX/DI | 15.2% |
| RETORNOS | 15.1% |
| RSI | 9.6% |
| TEMPORAL | 7.4% |
| MEDIAS | 6.4% |

## Top 5 features

| Peso | Feature |
|:----:|---------|
| 4.3% | `prev_day_range_pct` |
| 3.8% | `adx_p` (MNQ) |
| 3.8% | `adx_1` (BTC) |
| 3.8% | `adx_2` (CL) |
| 3.7% | `vol_p` |

## Mudança vs modelo anterior (contaminado)

- `hora in_us` removido — era artefato de rollover (rollovers em horários previsíveis)
- `price_div_cl > 2.0` removido — era artefato de retorno contaminado
- ADX ganhou peso central (era 8.8%, agora 15.2%)
- TEMPORAL caiu (era 10.6%, agora 7.4%)
- AUC: OOS 2025 = 0.5736 | Live 2026 = 0.5118

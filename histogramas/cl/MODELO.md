# Histograma — CL PropFirm

**Arquivo:** `histograma - cl.pine`
**Tipo:** Pane separado (overlay=false)
**Ativo:** Crude Oil Futures (CL1!)
**Atualizado:** 2026-05-22 (modelo sqlite_clean)

## ML Score (modelo sqlite_clean 2026-05-22)

**Máximo: 9 pontos — direção: LONG only**

| Condição | Peso | Feature do modelo |
|---|---|---|
| `ADX CL > 17` | 2 | adx_p: 4.8% (top 1) |
| `ADX MNQ > 17` | 2 | adx_1_mnq: 3.7% (top 2) |
| `vol_cl < 0.402%` | 2 | vol_p: 3.1% |
| `prev_day_range < 4.0%` | 1 | prev_day_range_pct |
| `close > monthly low` | 1 | dist_to_pml: 2.9% |
| `DI+ > DI-` | 1 | di_spread (bullish) |

**Sinal ativo:** score ≥ 6 → ML COMPRAR CL

## Importância por categoria (modelo atual)

| Categoria | Peso |
|---|---|
| KEY LEVELS | 26.3% |
| RETORNOS | 19.5% |
| ADX/DI | 17.2% |
| VOLATILIDADE | 16.2% |
| RSI | 7.7% |
| TEMPORAL | 6.4% |
| MEDIAS | 5.3% |

## Top 5 features

| Peso | Feature |
|:----:|---------|
| 4.8% | `adx_p` (CL) |
| 3.7% | `adx_1` (MNQ) |
| 3.2% | `adx_2` (BTC) |
| 3.1% | `vol_p` |
| 2.9% | `dist_to_pml` |

## Mudança vs modelo anterior (contaminado)

- `hora < 14h UTC` removido com peso 2 — era artefato de rollover mensal do CL (expiração em horários fixos)
- `prev_day_range < 0.54%` era threshold irreal (rollover=baixo range artificial) → corrigido para `< 4.0%`
- ADX promovido para top 1 feature (era 8.5%, agora 17.2% ADX/DI total)
- TEMPORAL caiu de 12.9% para 6.4%
- Adicionado `mnq_adx` via request.security para capturar ADX do MNQ
- AUC: OOS 2025 = 0.5419 | Live 2026 = 0.5521 (melhora em 2026)

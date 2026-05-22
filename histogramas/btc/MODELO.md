# Histograma — BTC PropFirm

**Arquivo:** `histograma - btc.pine`
**Tipo:** Pane separado (overlay=false)
**Ativo:** Bitcoin (BTCUSDT)
**Atualizado:** 2026-05-22 (modelo sqlite_clean)

## ML Score (modelo sqlite_clean 2026-05-22)

**Máximo: 8 pontos — direção: SHORT only**

| Condição | Peso | Feature do modelo |
|---|---|---|
| `ADX BTC > 17` | 2 | adx_p: 3.6% (top 2) |
| `prev_day_range > 1.5%` | 2 | prev_day_range_pct: 3.6% (top 1) |
| `SMA50 declining (5 barras)` | 1 | sma50_slope_p: 3.3% |
| `DI- > DI+` | 1 | di_spread (bearish) |
| `vol_btc < 0.4%` | 1 | vol_p |
| `close < weekly low` | 1 | dist_to_pwl: 2.8% |

**Sinal ativo:** score ≥ 6 → ML VENDER BTC

## Importância por categoria (modelo atual)

| Categoria | Peso |
|---|---|
| KEY LEVELS | 28.0% |
| RETORNOS | 18.5% |
| VOLATILIDADE | 15.4% |
| ADX/DI | 14.7% |
| RSI | 8.9% |
| MEDIAS | 7.0% |
| TEMPORAL | 6.4% |

## Top 5 features

| Peso | Feature |
|:----:|---------|
| 3.6% | `prev_day_range_pct` |
| 3.6% | `adx_p` (BTC) |
| 3.3% | `sma50_slope_p` |
| 2.9% | `adx_2` (CL) |
| 2.8% | `dist_to_pwl` |

## Mudança vs modelo anterior (contaminado)

- Labels de dia da semana SHORT/LONG removidos — `dow_sin` dominava com 6.1% por detectar rollovers em datas fixas
- `rsi >= 55` e `ret4 < -0.5%` rebaixados — eram artificialmente importantes por contaminação
- ADX + prev_range passam a liderar (sinal real de tendência e volatilidade)
- TEMPORAL caiu de 14.5% para 6.4% — confirmação que o dia da semana era artefato
- AUC: OOS 2025 = 0.5385 | Live 2026 = 0.5370 (estável)

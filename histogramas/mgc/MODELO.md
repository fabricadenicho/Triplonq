# Histograma — MGC PropFirm

**Arquivo:** `histograma - mgc.pine`
**Tipo:** Pane separado (overlay=false)
**Ativo:** Micro Gold Futures (MGC1!)

## O que faz

Painel completo de análise do MGC (Micro Gold). Monitora o Prime Setup (above PMH + ADX + sessão US PM), Key Levels com ênfase no Monthly Open, ML Score LONG/SHORT e sinais históricos dos últimos 30 dias (15 trades, WR 46.7%).

## ML Score (alinhado com a árvore XGBoost)

**Máximo: 8 pontos — LONG ou SHORT**

| Condição | Peso | Descrição |
|---|---|---|
| `vol_mgc < 0.256%` | 2 | Volume normalizado baixo |
| `dist Monthly Open < 0.49%` | **3** | MUITO próximo ao Monthly Open — feature mais forte! |
| `prev_day_range < 0.977%` | 1 | Range do dia anterior comprimido |
| RSI < 50 (LONG) / >= 50 (SHORT) | 1 | Momentum direcional |
| DI+ > DI- (LONG) / DI- > DI+ (SHORT) | 1 | DMI favorável |

**Sinal:** score >= 5 = ML COMPRAR/VENDER MGC | >= 4 = parcial (amarelo)

**Destaque:** `dist_to_mo < 0.49%` vale 3 pontos — o Monthly Open é o nível mais importante do modelo MGC. Quando o ouro está dentro de 0.49% do MO, a probabilidade de setup aumenta drasticamente.

## Key Levels monitorados

- `pd_high / pd_low` — Previous Day High/Low
- `pm_high / pm_low` — Previous Month High/Low
- `mo` — Monthly Open (mais importante — 3/8 pontos do ML Score)
- `wo` — Weekly Open
- `id_h / id_l` — Intraday High/Low
- **DIST MO (KL)** — exibido no painel substituindo BB% CL

## Prime Setup (lógica original)

Além do ML Score, o histograma mantém a lógica original:
- `above_pmh` — preço acima do Previous Month High
- `strong_adx` — ADX forte
- `in_us_pm` — dentro da sessão US PM (após 16h UTC)

## Performance ML

| Métrica | Valor |
|---|---|
| AUC | 0.529 |
| Acurácia | — |
| KEY LEVELS | 37% do peso da árvore |

> MGC opera BOTH (LONG e SHORT). 15 trades no último mês.

# Indicador — MNQ A++ Setup

**Arquivo:** `indicador - mnq.pine`
**Tipo:** Overlay (plota direto no gráfico de preço)
**Ativo:** Micro Nasdaq Futures (MNQ1!)

## O que faz

Identifica setups de alta probabilidade (A++) no MNQ usando a divergência RSI entre CL e MNQ como gatilho principal. Plota setas ▲ LONG e ▼ SHORT no gráfico de preço com painel informativo.

## Lógica do sinal

| Condição | LONG | SHORT |
|---|---|---|
| ADX >= 17 | obrigatório | obrigatório |
| Kill Zone (London/NY) | obrigatório | obrigatório |
| Div RSI CL-MNQ >= 8 | CL mais forte que MNQ (bull) | MNQ mais forte que CL (bear) |
| Regime EMA20 | acima da EMA20 ou trend bull | abaixo da EMA20 ou trend bear |

## Indicadores no painel

- **ADX** — força da tendência
- **RSI 21** — momentum
- **DIV CL-MNQ** — diferença RSI entre CL e MNQ (positivo = LONG bias)
- **Sessão / Kill Zone** — hora UTC (London 8-12h, NY 13-20h, Overlap 13-15h)
- **Regime EMA20** — posição do preço relativa à EMA20 e SMA50
- **SMA50 Align** — alinhamento 0-3 (3=forte bull, 0=bear)
- **Checks** — div / ADX / US / KZ ativos

## Configurações

- `ADX mínimo`: 17 (padrão)
- `Lookback divergência`: 10 barras
- `Div CL-MNQ mínima`: 8 pontos RSI
- `Só sinais em Kill Zone`: ativado por padrão

## Performance ML

| Métrica | Valor |
|---|---|
| AUC | 0.467 |
| Acurácia | 32% |

> AUC próximo de 0.5 = sinal modesto. Usar junto com o histograma para confirmação.

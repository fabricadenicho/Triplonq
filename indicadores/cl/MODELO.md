# Indicador — CL A++ Setup

**Arquivo:** `indicador - cl.pine`
**Tipo:** Overlay (plota direto no gráfico de preço)
**Ativo:** Crude Oil Futures (CL1!)

## O que faz

Identifica setups A++ LONG no CL usando divergência RSI entre CL e MNQ. Quando o CL está mais fraco que o MNQ (RSI CL < RSI MNQ), o CL tende a reverter para cima. Inclui marcador diamante para situações com 75% de win rate histórico.

## Lógica do sinal

| Condição | LONG |
|---|---|
| ADX >= 14 | obrigatório |
| Kill Zone (London/NY) | obrigatório (configurável) |
| Div RSI CL-MNQ <= -8 | CL mais fraco → tendência a subir |
| Regime EMA20 | acima ou trend bull |
| **Crítico (75% LONG)** | diamante quando todas condições + hora específica |

## Indicadores no painel

- **ADX / RSI 21** — força e momentum do CL
- **DIV CL-MNQ** — diferença RSI (negativo = CL fraco = oportunidade LONG)
- **Sessão / Kill Zone** — London 8-12h, NY 13-20h, Evening Session CL
- **Regime EMA20 / SMA50 Align**
- **Crítico** — marcador especial quando convergência máxima

## Configurações

- `ADX mínimo`: 14 (mais sensível que MNQ/BTC)
- `Div CL-MNQ mínima`: 8 pontos RSI
- `Diamante 75% LONG histórico`: ativado via opção

## Uso recomendado

Usar em conjunto com o **histograma CL** que exibe o ML Score em tempo real (vol + hora + prev range + dist Monthly Open).

# Indicador — BTC A++ Setup

**Arquivo:** `indicador - btc.pine`
**Tipo:** Overlay (plota direto no gráfico de preço)
**Ativo:** Bitcoin (BTCUSDT / BTCUSD)

## O que faz

Identifica setups A++ no BTC combinando divergência RSI com lógica contrarian: bull breakout em London/Overlap tende a reverter (SHORT 45% histórico). Plota setas no gráfico com painel informativo e alertas.

## Lógica do sinal

| Condição | LONG | SHORT |
|---|---|---|
| ADX >= 17 | obrigatório | obrigatório |
| Kill Zone (London/NY) | obrigatório | obrigatório |
| Divergência RSI | bull_div (preço cai, RSI sobe, RSI < 50) | bear_div (preço sobe, RSI cai, RSI > 50) |
| Regime EMA20 | acima da EMA20 ou trend bull | abaixo da EMA20 ou trend bear |
| **Contrarian** | bear breakout em NY = LONG | bull breakout em London/Overlap = SHORT |

### DOW bias histórico
- Segunda e Sexta → SHORT
- Quarta → LONG

## Indicadores no painel

- **ADX / RSI 21** — força e momentum
- **DOW BIAS** — dia da semana (seg/sex = SHORT, qua = LONG)
- **Sessão / Kill Zone** — London 8-12h, NY 13-20h UTC
- **CONTRARIAN** — alerta quando breakout contra-tendência detectado
- **Regime EMA20 / SMA50 Align** — contexto de tendência

## Configurações

- `ADX mínimo`: 17
- `Lookback divergência`: 10 barras
- `Lookback breakout`: 20 barras
- `Contrarian London/Overlap`: ativado por padrão

## Performance ML

| Métrica | Valor |
|---|---|
| AUC | 0.555 |
| Acurácia | 39% |

> Melhor AUC dos ativos. Sinal mais confiável quando DOW bias + Kill Zone + ADX convergem.

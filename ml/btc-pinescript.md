# Indicador Pine Script para o modelo BTC

## Objetivo
Transformar o setup mais assertivo do modelo de ML de **BTC** em um indicador Pine Script. O foco é refletir as features mais importantes usadas pelo modelo.

## Premissa do modelo BTC
O modelo BTC usa dados de **BTC, MNQ e CL** no timeframe de **1 hora** e prevê a direção nas próximas **4 horas**. As features mais importantes são:

- `dow` (dia da semana) — feature #1 do modelo
- `vol_mnq` — volatilidade do MNQ
- `price_div_abs` (BTC × MNQ) — divergência de preço
- `sma50_align_mnq_cl` — alinhamento SMA50 entre MNQ e CL
- `adx_btc` — força da tendência do BTC
- `price_div_cl` — divergência de preço BTC vs CL
- `div_cl` — divergência RSI BTC vs CL
- `dist_ema20_cl` — distância do CL à EMA20

## Setup recomendado para o indicador

1. timeframe: `60` minutos
2. símbolos usados:
   - BTC primário: `BTCUSD`
   - MNQ secundário: `MNQ=F`
   - CL secundário: `CL=F`
3. principais componentes:
   - `rsiB = ta.rsi(close, 21)`
   - `adxB = ta.adx(17)`
   - `sma50B = ta.sma(close, 50)`
   - `ema20B = ta.ema(close, 20)`
   - `priceDivAbs = math.abs(change(close) * change(closeMNQ))`
   - `divRSI_CL = rsiB - rsiCL`
   - `sma50AlignMNQCL = (close > sma50MNQ and closeCL > sma50CL) or (close < sma50MNQ and closeCL < sma50CL)`
   - `volMNQ = ta.stdev(ta.change(closeMNQ), 20)`
   - `dowBias` a partir do dia da semana

## Regras de sinal

### Long
- `priceDivAbs` alto
- `adxB > 17`
- `sma50AlignMNQCL` em regime de tendência positiva
- `divRSI_CL` levemente negativo ou neutro
- `volMNQ` elevado como confirmação de movimento forte
- `dayofweek` preferencialmente no final da sessão (15-17h)

### Short
- `priceDivAbs` alto
- `adxB > 17`
- `divRSI_CL` positivo
- `sma50AlignMNQCL` em regime fraco ou de rolagem de tendência
- dias com bias de `dow` favorável a baixa (11h-13h tem maior chance de short)

## Exemplo de template Pine Script

```pinescript
//@version=5
indicator("BTC ML Proxy", overlay=false)

symMNQ = "MNQ=F"
symCL  = "CL1!"

closeMNQ = request.security(symMNQ, "60", close)
closeCL  = request.security(symCL,  "60", close)

rsiB   = ta.rsi(close, 21)
rsiMNQ = ta.rsi(closeMNQ, 21)
rsiCL  = ta.rsi(closeCL, 21)

adxB = ta.adx(17)

sma50B   = ta.sma(close, 50)
ema20B   = ta.ema(close, 20)
sma50MNQ = ta.sma(closeMNQ, 50)
sma50CL  = ta.sma(closeCL, 50)

priceDivAbs = math.abs(ta.change(close) * ta.change(closeMNQ))
priceDivSign = ta.change(close) * ta.change(closeMNQ)

divRSI_CL = rsiB - rsiCL
volMNQ    = ta.stdev(ta.change(closeMNQ), 20)

sma50AlignMNQCL = (close > sma50MNQ and closeCL > sma50CL) or (close < sma50MNQ and closeCL < sma50CL)

isShortBias = (dayofweek == dayofweek.wednesday or dayofweek == dayofweek.thursday)

longSignal  = priceDivAbs > 0.0005 and adxB > 17 and sma50AlignMNQCL and divRSI_CL <= 0 and volMNQ > ta.sma(volMNQ, 20)
shortSignal = priceDivAbs > 0.0005 and adxB > 17 and divRSI_CL > 0 and not sma50AlignMNQCL and isShortBias

plot(priceDivAbs, title="price_div_abs", color=color.blue)
plot(adxB, title="ADX BTC", color=color.orange)
plotshape(longSignal,  location=location.belowbar, color=color.green, style=shape.triangleup, text="LONG")
plotshape(shortSignal, location=location.abovebar, color=color.red,   style=shape.triangledown, text="SHORT")

hline(17, "ADX 17", color=color.gray)
```

## Ajustes finos

- Ajuste o limiar de `priceDivAbs` para o ativo e timeframe.
- Use `dow` como fator qualitativo: dias de baixa probabilidade de long podem ser evitados.
- `volMNQ` é um proxy importante: alta volatilidade no MNQ tende a indicar moves mais robustos no BTC.
- Considere usar `dist_ema20_cl` e `dist_ema20_mnq` para verificar regimes extremos.

## Observação

Esse indicador tenta capturar a lógica de ML do BTC: forte sazonalidade semanal, divergência com MNQ e CL, e regime de volatilidade do MNQ como confirmador.
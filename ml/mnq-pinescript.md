# Indicador Pine Script para o modelo MNQ

## Objetivo
Criar um indicador de tendência e divergência inspirado no modelo de ML do **MNQ** (Mini Nasdaq) usando os setups mais assertivos do modelo.

## Premissa do modelo MNQ
O modelo MNQ usa dados de **MNQ, BTC e CL** no timeframe de **1 hora** e busca prever a direção nas próximas **4 horas**. Os principais sinais do modelo são:

- RSI 21 no MNQ, BTC e CL
- ADX 17 no MNQ
- SMA50 e EMA20 no MNQ
- Divergências de preço e RSI entre MNQ e CL/BTC
- `price_div_abs` = |ret1(MNQ) × ret1(CL)| como a feature mais importante
- setup `strong_div` quando MNQ e CL divergem de preço e ADX > 17
- viés de regime: MNQ + BTC acima/abaixo de EMA20

## Setup recomendado para o indicador

1. timeframe: `60` minutos
2. símbolos usados:
   - MNQ primário: `MNQ=F`
   - CL secundário: `CL=F`
   - BTC secundário: `BTC-USD`
3. principais componentes:
   - `rsiM = ta.rsi(close, 21)`
   - `adxM = ta.adx(17)`
   - `sma50M = ta.sma(close, 50)`
   - `ema20M = ta.ema(close, 20)`
   - `priceDivCL = math.abs(change(close) * change(closeCL))`
   - `divRSI_CL = rsiM - rsiCL`
   - `divRSI_BTC = rsiM - rsiBTC`
   - `strongDiv = priceDivCL > threshold and adxM > 17`

## Regras de sinal

### Long
- `priceDivCL` alto
- `priceDivCL` é negativo originalmente (MNQ e CL em direções opostas)
- `adxM > 17`
- `divRSI_CL < 0` (MNQ com RSI inferior ao CL)
- `close > ema20M` ou pelo menos MNQ/BTC em tendência positiva sobre EMA20

### Short
- `priceDivCL` alto
- `adxM > 17`
- `divRSI_CL > 0` (MNQ com RSI superior ao CL)
- regime fraco de EMA20 ou MNQ abaixo da EMA20

## Exemplo de template Pine Script

```pinescript
//@version=5
indicator("MNQ ML Proxy", overlay=false)

symCL  = "CL1!"
symBTC = "BTCUSD"

closeCL  = request.security(symCL,  "60", close)
closeBTC = request.security(symBTC, "60", close)

rsiM   = ta.rsi(close, 21)
rsiCL  = ta.rsi(closeCL, 21)
rsiBTC = ta.rsi(closeBTC, 21)

adxM = ta.adx(17)

sma50M = ta.sma(close, 50)
ema20M = ta.ema(close, 20)

ret1M  = ta.change(close)
ret1CL = ta.change(closeCL)
priceDivCL = math.abs(ret1M * ret1CL)
priceDivSign = ret1M * ret1CL

divRSI_CL  = rsiM - rsiCL
sma50Above = close > sma50M
ema20Above = close > ema20M

strongDiv = (priceDivSign < 0) and (adxM > 17)
longSignal  = strongDiv and divRSI_CL < 0 and ema20Above
shortSignal = strongDiv and divRSI_CL > 0 and not ema20Above

plot(priceDivCL, color=color.new(color.blue, 0), title="Price Divergence |MNQ x CL|")
plot(adxM, color=color.new(color.orange, 0), title="ADX MNQ")
plotshape(longSignal,  location=location.belowbar, color=color.green, style=shape.labelup, text="LONG")
plotshape(shortSignal, location=location.abovebar, color=color.red,   style=shape.labeldown, text="SHORT")

hline(17, "ADX 17", color=color.gray)
```

## Ajustes finos

- Use `threshold` para `priceDivCL` com base em volatilidade histórica.
- Para filtrar ruído, só permita sinais se `rsiM` estiver entre 20 e 80.
- Combine `sma50Above` com `ema20Above` para o viés de regime.
- Opcional: adicione `divRSI_BTC` para confirmar direção.

## Observação

Esse arquivo descreve um indicador proxy, não o modelo exato. Ele captura os sinais mais relevantes do ML MNQ: divergência CL, ADX 17, SMA/EMA, e regime de tendência.
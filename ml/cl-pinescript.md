# Indicador Pine Script para o modelo CL

## Objetivo
Construir um indicador inspirado no setup mais assertivo do modelo de ML de **CL** (Crude Oil), com foco no sinal `us_prime_setup` e na relação CL-MNQ-BTC.

## Premissa do modelo CL
O modelo CL analisa **CL, MNQ e BTC** no timeframe de **1 hora** e prevê a direção nas próximas **4 horas**. As principais características do modelo são:

- `us_prime_setup` é a feature mais importante
- `hour` e `is_us_morning` têm grande peso
- `rsi_cl` e `adx_btc` são críticos
- `bb_spread_cl_mnq`, `dist_sma50_cl`, `dist_ema20_mnq`, `vol_spread_cl_mnq` e `div_mnq` ajudam a confirmar o sinal
- strong_div com CL e MNQ é o setup central

## Setup recomendado para o indicador

1. timeframe: `60` minutos
2. símbolos usados:
   - CL primário: `CL=F`
   - MNQ secundário: `MNQ=F`
   - BTC secundário: `BTC-USD`
3. principais componentes:
   - `rsiCL = ta.rsi(close, 21)`
   - `adxB = ta.adx(17)` no BTC
   - `sma50CL = ta.sma(close, 50)`
   - `ema20MNQ = ta.ema(closeMNQ, 20)`
   - `bbWidthCL = ta.bbwidth(close, 20, 2)`
   - `priceDivCL = ta.change(close) * ta.change(closeMNQ)`
   - `divRSI_MNQ = rsiCL - rsiMNQ`
   - `usPrimeSetup = strongDiv and session.isin("0600-1200:1234567")`

## Regras de sinal

### Long
- `usPrimeSetup` ativo
- `strongDiv` verdadeiro: CL e MNQ movem-se em direções opostas com ADX BTC forte
- `rsiCL` não extremo, mas em torno de 40–60 para reversão
- `bb_spread_cl_mnq` elevado (CL mais volátil que MNQ)
- `dist_sma50_cl` alto (CL distante da SMA50)
- `dist_ema20_mnq` alto
- `CL` e `BTC` ambos abaixo da EMA20 são um forte sinal de LONG reversão

### Short
- `strongDiv` ativo
- `divRSI_MNQ > 0`
- regime misto ou CL/BTC não tão oversold
- `hour` no bloco US 11-13h pode ter maior pressão de short

## Exemplo de template Pine Script

```pinescript
//@version=5
indicator("CL ML Proxy", overlay=false)

symMNQ = "MNQ=F"
symBTC = "BTCUSD"

closeMNQ = request.security(symMNQ, "60", close)
closeBTC = request.security(symBTC, "60", close)

rsiCL  = ta.rsi(close, 21)
rsiMNQ = ta.rsi(closeMNQ, 21)

adxBTC = ta.adx(17)

sma50CL  = ta.sma(close, 50)
ema20CL  = ta.ema(close, 20)
ema20MNQ = ta.ema(closeMNQ, 20)

bbWidthCL = ta.bbwidth(close, 20, 2)

priceDivCL = ta.change(close) * ta.change(closeMNQ)
priceDivAbs = math.abs(priceDivCL)
divRSI_MNQ = rsiCL - rsiMNQ

isUSSession = session.isin("0900-1700:12345")
isUSMorning = session.isin("0900-1300:12345")

strongDiv = (priceDivCL < 0) and (adxBTC > 17)
usPrimeSetup = strongDiv and isUSSession

longSignal  = usPrimeSetup and divRSI_MNQ < 0 and bbWidthCL > ta.sma(bbWidthCL, 20)
shortSignal = strongDiv and divRSI_MNQ > 0 and not usPrimeSetup

plot(priceDivAbs, title="price_div_abs CL x MNQ", color=color.blue)
plot(adxBTC, title="ADX BTC", color=color.orange)
plotshape(longSignal,  location=location.belowbar, color=color.green, style=shape.labelup, text="LONG")
plotshape(shortSignal, location=location.abovebar, color=color.red,   style=shape.labeldown, text="SHORT")

hline(17, "ADX 17 BTC", color=color.gray)
```

## Ajustes finos

- Use `usPrimeSetup` como sinal principal para LONG.
- Para LONG, prefira horas US de manhã (9-13h) e mantenha `adxBTC > 17`.
- Se `CL` e `BTC` estiverem ambos abaixo da EMA20, aumente o peso do LONG.
- Use `bbWidthCL` como proxy de volatilidade e `dist_sma50_cl` para identificar extremos.

## Observação

O indicador captura o setup mais forte do modelo CL: `us_prime_setup` com CL-MNQ divergente e confirmação de ADX no BTC. É uma estrutura de proxy para transformar os principais insights do ML em Pine Script.
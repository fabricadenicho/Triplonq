# Modelo Preditivo BTC (Bitcoin) — XGBoost Multiclasse

## Visao Geral

Modelo de Machine Learning para prever a direcao do **BTC** (Bitcoin) nas proximas 4 horas. Classifica em 3 classes:

| Classe | Label | Descricao |
|--------|-------|-----------|
| SHORT | 0 | Preco vai cair > 0.1% |
| NEUTRO | 1 | Preco vai ficar entre -0.1% e +0.1% |
| LONG | 2 | Preco vai subir > 0.1% |

## Arquitetura

```
yfinance (BTC, MNQ, CL)
       |
 collect_data.py
       |
   data.db (SQLite)
       |
  train.py -> build_features() -> 38 features
       |
   XGBoost (multiclass)
       |
   model.pkl + feature_importance.png
       |
  predict.py -> JSON -> server.js -> btc.html (dashboard)
```

## Timeframe

- **Intervalo dos candles**: 1 hora
- **Forward**: 4 horas
- **Periodo de dados**: ~2 anos (1h) + 5 anos (1d)
- **3 ativos**: BTC-USD (primario), MNQ=F (Mini Nasdaq), CL=F (Crude Oil)

## Top Features por Importancia (Gain)

```
dow                          8.3%   <- dia da semana (feature #1 do BTC!)
vol_mnq                      4.4%   <- volatilidade do MNQ
price_div_abs                4.0%   <- magnitude divergencia preco BTC-MNQ
sma50_align_mnq_cl           3.9%   <- alinhamento SMA50 entre MNQ e CL
adx_btc                      3.6%   <- forca tendencia do BTC
price_div_cl                 3.3%   <- divergencia preco BTC vs CL
div_cl                       3.3%   <- divergencia RSI BTC vs CL
dist_ema20_cl                3.2%   <- distancia do CL da EMA20
sma50_slope_mnq              3.1%   <- inclinacao SMA50 do MNQ
ret1_mnq                     3.1%   <- retorno 1h do MNQ
```

Top 5 concentram 24.2% da importancia. **DOW (dia da semana) e a feature #1** — o BTC tem forte sazonalidade semanal.

### Categorias de Features

| Categoria | Importancia |
|-----------|-------------|
| Tempo (hora/sessao) | 14.8% |
| Divergencia RSI | 11.9% |
| Divergencia preco | 7.3% |
| EMA20 alignment | 7.3% |
| MA50 alignment | 6.8% |
| Spreads volatilidade | 5.7% |
| Spreads ADX | 3.0% |
| Spreads Bollinger | 2.7% |
| Co-movement (prod) | 2.7% |
| DI spread | 2.8% |
| (outros) | 35.0% |

## Resultados Obtidos (Walk-Forward, Sessao US)

```
Dataset: 6499 amostras  |  SHORT=43.0%  NEUTRO=13.4%  LONG=43.6%

              precision    recall  f1-score
   SHORT       0.43      0.43      0.43
  NEUTRO       0.24      0.33      0.28
    LONG       0.40      0.37      0.38

accuracy                         0.39
ROC-AUC (macro ovr): 0.5248
```

### Analise dos Gatilhos

| Gatilho | N | LONG | SHORT |
|---------|---|------|-------|
| strong_div | 902 | 42.7% | 46.2% |
| us_prime_setup | 902 | 42.7% | 46.2% |

**No BTC, o strong_div favorece SHORT (46.2% vs 42.7% LONG)** — comportamento oposto ao MNQ. Isso faz sentido: quando BTC e MNQ divergem, o BTC tende a seguir na direcao oposta ao MNQ.

### Vies EMA20 (BTC + MNQ)

| Estado | LONG real | SHORT real |
|--------|-----------|------------|
| Ambos abaixo da EMA20 | 40.2% | **48.4%** <- melhor cenario SHORT |
| Misturado | **45.2%** | 41.5% <- melhor cenario LONG |
| Ambos acima da EMA20 | 41.8% | 44.7% |

Regime misto (um acima, outro abaixo da EMA20) favorece LONG. Ambos abaixo favorece SHORT.

### Acertividade por Hora

| Hora | LONG | SHORT | N |
|------|------|-------|---|
| 09h | 39.8% | 43.5% | 216 |
| 10h | 40.1% | 45.6% | 217 |
| 11h | 41.9% | **49.8%** | 217 |
| 12h | 42.1% | **50.0%** | 216 |
| 13h | 40.7% | 49.1% | 216 |
| 14h | 43.8% | 42.9% | 217 |
| 15h | **44.2%** | 41.0% | 217 |
| 16h | 44.2% | 40.1% | 217 |
| 17h | **45.6%** | 40.6% | 217 |

Manha (11-13h) favorece SHORT, tarde (15-17h) equilibra para LONG.

## Estrategia Recomendada

### Para LONG

1. **Sessao**: US 9-17h (preferir 15-17h)
2. **Regime EMA20**: misturado (BTC e MNQ em lados opostos da EMA20)
3. **ADX BTC**: > 17 (tendencia presente)
4. **Divergencia**: price_div_abs alto + divergencia RSI favoravel
5. **DOW**: verificar dia da semana (feature #1)

### Para SHORT

1. **Sessao**: US 9-17h (preferir 11-13h)
2. **Regime EMA20**: ambos abaixo da EMA20 (-48.4% SHORT)
3. **strong_div**: ativo (BTC tende a seguir na direcao oposta ao MNQ)
4. **ADX**: > 17

## Diferencas para o Modelo MNQ

| Aspecto | MNQ | BTC |
|---------|-----|-----|
| AUC | 0.602 | 0.525 |
| Feature #1 | price_div_abs | dow (dia da semana) |
| strong_div edge | +10.6% LONG | +3.5% SHORT |
| Melhor regime LONG | ambos acima EMA20 | misturado |
| Melhor regime SHORT | ambos abaixo | ambos abaixo |
| Sazonalidade semanal | fraca | **forte (8.3%)** |

O BTC e o ativo com maior influencia de **sazonalidade semanal** (dow = 8.3%). Diferente do MNQ, o BTC reverte quando o strong_div esta ativo (comportamento de contra-tendencia vs MNQ).

## Como Usar

```bash
cd ml/btc
python collect_data.py         # baixa dados historicos
python train.py                 # treina modelo otimizado

# Predicao em tempo real
python predict.py               # JSON com prob_long, prob_short, sinal
```

Acessar dashboard: `http://localhost:3000/btc`

## Dependencias

```
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
ta>=0.11.0
xgboost>=2.0.3
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

## Arquivos do Pipeline

| Arquivo | Descricao |
|---------|-----------|
| `collect_data.py` | Baixa dados historicos para SQLite |
| `train.py` | Constroi features e treina XGBoost |
| `predict.py` | Predicao em tempo real (JSON) |
| `analyze_importance.py` | Analisa importancia das features |
| `data.db` | Dados historicos |
| `model.pkl` | Modelo treinado |
| `feature_importance.png` | Grafico de importancia |
| `MODELO.md` | Este documento |

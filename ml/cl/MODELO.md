# Modelo Preditivo CL (Crude Oil) — XGBoost Multiclasse

## Visao Geral

Modelo de Machine Learning para prever a direcao do **CL** (Crude Oil Futures) nas proximas 4 horas. Classifica em 3 classes:

| Classe | Label | Descricao |
|--------|-------|-----------|
| SHORT | 0 | Preco vai cair > 0.1% |
| NEUTRO | 1 | Preco vai ficar entre -0.1% e +0.1% |
| LONG | 2 | Preco vai subir > 0.1% |

## Arquitetura

```
yfinance (CL, MNQ, BTC)
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
  predict.py -> JSON -> server.js -> cl.html (dashboard)
```

## Timeframe

- **Intervalo dos candles**: 1 hora
- **Forward**: 4 horas (configuravel com `--forward`)
- **Periodo de dados**: 2 anos (1h) + 5 anos (1d)
- **3 ativos**: CL=F (Crude Oil — primario), MNQ=F (Mini Nasdaq), BTC-USD (Bitcoin)

## Pipeline de Treino

### 1. Coleta de Dados (`collect_data.py`)
Baixa dados OHLCV historicos via Yahoo Finance para CL, MNQ e BTC.
- Intervalos: 1h (730d) e 1d (5y)
- Salva em SQLite (`ml/cl/data.db`)

### 2. Construcao de Features

Mesma estrutura do modelo MNQ, mas com CL como ativo **primario**. As features refletem relacoes de CL contra MNQ e BTC.

#### Indicadores Individuais (RSI 21, ADX 17)
Aplicados em CL (primario), MNQ e BTC (secundarios).

#### Spreads entre Ativos
- `div_mnq` = RSI(CL) - RSI(MNQ)
- `div_btc` = RSI(CL) - RSI(BTC)
- `price_div_cl` = ret1(CL) x ret1(MNQ)
- Spreads de ADX, volatilidade, Bollinger, MA50, EMA20 entre CL e os demais

#### Triggers Compostos
- `strong_div`: price_div_cl < 0 AND ADX(CL) > 17
- `us_prime_setup`: strong_div + horario US (9-17h)
- `prime_setup`: strong_div + horario 18-21h

### 3. Treino

```python
XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.75, min_child_weight=8,
    objective='multi:softprob', num_class=3,
    eval_metric='mlogloss', early_stopping_rounds=40,
)
```

- **Walk-Forward**: 70% treino, 30% teste (cronologico)
- **Class weights**: balanceamento por frequencia inversa
- **Sessao US**: treinado apenas em horario 9-17h (maior edge)

## Top Features por Importancia (Gain)

```
us_prime_setup               5.4%   <- trigger composto mais importante
hour                         3.8%   <- hora do dia
is_us_morning                3.3%   <- manha americana
rsi_cl                       3.0%   <- RSI do CL
adx_btc                      2.9%   <- ADX do BTC (volatilidade externa)
bb_spread_cl_mnq             2.9%   <- diferenca Bollinger CL vs MNQ
dist_sma50_cl                2.9%   <- distancia do CL da SMA50
dist_ema20_mnq               2.9%   <- distancia do MNQ da EMA20
vol_spread_cl_mnq            2.9%   <- diferenca de volatilidade CL-MNQ
div_mnq                      2.9%   <- divergencia RSI CL vs MNQ
```

Top 5 features concentram 18.4% da importancia (features bem distribuidas).

### Categorias de Features

| Categoria | Importancia |
|-----------|-------------|
| Tempo (hora/sessao) | 14.1% |
| Divergencia RSI | 8.5% |
| EMA20 alignment | 8.2% |
| Spreads volatilidade | 5.7% |
| Divergencia preco | 5.5% |
| MA50 alignment | 5.1% |
| Spreads Bollinger | 2.9% |
| Spreads ADX | 2.8% |
| DI spread | 2.8% |
| Co-movement (prod) | 2.6% |
| (outros) | 41.9% |

## Resultados Obtidos (Walk-Forward, Sessao US)

```
Dataset: 5392 amostras  |  SHORT=43.1%  NEUTRO=9.9%  LONG=47.0%

              precision    recall  f1-score
   SHORT       0.42      0.45      0.43
  NEUTRO       0.14      0.23      0.17
    LONG       0.46      0.37      0.41

accuracy                         0.39
ROC-AUC (macro ovr): 0.5232
```

### Analise dos Gatilhos

| Gatilho | N | LONG | SHORT |
|---------|---|------|-------|
| strong_div | 787 | 50.1% | 40.8% |
| us_prime_setup | 787 | 50.1% | 40.8% |

**strong_div** da edge de **9.3%** em CL (50.1% LONG vs 40.8% SHORT), consistente com o comportamento visto no MNQ.

### Vies EMA20 (CL + BTC)

| Estado | LONG real | SHORT real |
|--------|-----------|------------|
| Ambos abaixo da EMA20 | **56.3%** | 36.1% <- melhor cenario LONG |
| Misturado | 45.6% | 46.4% |
| Ambos acima da EMA20 | 49.0% | 41.6% |

**Descoberta importante**: quando CL e BTC estao ambos abaixo da EMA20, o CL tem 56.3% de chance de subir nas proximas 4h. Isso faz sentido: petroleo oversold com BTC fraco = oportunidade de reversao.

## Estrategia Recomendada

### Para LONG

1. **Sessao**: US 9-17h (manha tem mais edge)
2. **Trigger**: us_prime_setup ativo (strong_div + horario US)
3. **Regime EMA20**: CL + BTC ambos abaixo da EMA20 (+56.3% LONG)
4. **ADX**: > 17 (tendencia presente)
5. **RSI**: `div_mnq` < 0 (RSI do CL abaixo do MNQ = CL sobrevendido relativo)
6. **Bollinger**: `bb_spread_cl_mnq` alto (CL mais volátil que MNQ)

### Para SHORT

1. **Sessao**: US 9-17h
2. **Regime EMA20**: regime misto ou ambos acima
3. **ADX**: > 17
4. **Confirmacao adicional**: requer mais filtros que LONG (modelo mais fraco pra SHORT)

## Diferencas para o Modelo MNQ

| Aspecto | MNQ | CL |
|---------|-----|-----|
| AUC | 0.602 | 0.523 |
| Melhor hora LONG | 10h/13h | 11h |
| strong_div edge | +10.6% | +9.3% |
| Melhor regime LONG | ambos acima EMA20 | ambos abaixo EMA20 |
| Distribuicao SHORT | 38.1% | 43.1% |

O modelo CL tem performance inferior ao MNQ (AUC 0.52 vs 0.60), mas o **strong_div** ainda oferece edge consistente (+9.3%). O regime EMA20 tem comportamento **oposto** ao MNQ: CL performa melhor quando ambos estao abaixo da EMA20 (reverso).

## Como Usar

```bash
cd ml/cl
python collect_data.py         # baixa dados historicos
python train.py                 # treina modelo otimizado (38 features)

# Predicao em tempo real
python predict.py               # JSON com prob_long, prob_short, sinal
```

Acessar dashboard: `http://localhost:3000/cl`

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

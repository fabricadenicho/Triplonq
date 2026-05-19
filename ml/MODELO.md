# Modelo Preditivo MNQ — XGBoost Multiclasse

## Visao Geral

Modelo de Machine Learning para prever a direcao do **MNQ** (Mini Nasdaq Futures) nas proximas 4 horas. Classifica em 3 classes:

| Classe | Label | Descricao |
|--------|-------|-----------|
| SHORT | 0 | Preco vai cair > 0.1% |
| NEUTRO | 1 | Preco vai ficar entre -0.1% e +0.1% |
| LONG | 2 | Preco vai subir > 0.1% |

## Arquitetura

```
yfinance (MNQ, BTC, CL)
       |
 collect_data.py
       |
   data.db (SQLite)
       |
  train.py → build_features() → ~48 features tecnicas
       |
   XGBoost (multiclass)
       |
   model.pkl + feature_importance.png
       |
  predict.py → JSON → server.js (dashboard)
```

## Timeframe

- **Intervalo dos candles**: 1 hora
- **Forward** (horas a frente): 4 horas (configuravel com `--forward`)
- **Periodo de dados**: 2 anos (1h) + 5 anos (1d)
- **3 ativos**: MNQ=F (Mini Nasdaq), BTC-USD (Bitcoin), CL=F (Crude Oil)

## Pipeline de Treino

### 1. Coleta de Dados (`collect_data.py`)
Baixa dados OHLCV historicos via Yahoo Finance para MNQ, BTC e CL.
- Intervalos: 1h (730d) e 1d (5y)
- Salva em SQLite (`data.db`)

### 2. Construcao de Features (`build_features` em `train.py`)

#### Indicadores Individuais (calculados para cada ativo)
| Indicador | Janela | Descricao |
|-----------|--------|-----------|
| RSI | 21 | Forca do movimento |
| ADX | 17 | Forca da tendencia |
| DI+ / DI- | 17 | Direcao da tendencia |
| Retorno % | 1h, 4h, 8h | Retorno passado |
| Volatilidade | 20h | Desvio padrao do retorno |
| Bollinger Width | 20h | Largura das bandas |
| SMA 50 | 50h | Media movel simples (melhor single MA) |
| EMA 20 | 20h | Media movel exponencial curta (complementar) |
| Distancia % da SMA/EMA | — | % acima/abaixo |
| Acima/abaixo da SMA/EMA | — | Binario |

#### Spreads entre Ativos (relacoes MNQ vs BTC vs CL)

**Divergencias RSI**
- `div_cl` = RSI(MNQ) − RSI(CL)
- `div_btc` = RSI(MNQ) − RSI(BTC)
- `rsi_spread_btc_cl` = RSI(BTC) − RSI(CL)
- Magnitudes absolutas de cada divergencia

**Divergencias de Preco**
- `price_div_cl` = ret1(MNQ) × ret1(CL) (negativo = direcoes opostas)
- `price_div_abs` = |price_div_cl| — **feature mais importante do modelo**

**ADX Spreads**
- Diferenca de ADX entre todos os pares (MNQ-BTC, MNQ-CL, BTC-CL)
- Magnitudes absolutas

**Co-movement (Produtos de Retorno)**
- `ret1_prod_mnq_btc`, `ret1_prod_btc_cl`, `ret1_prod_mnq_cl`
- `ret4_prod_mnq_btc`, `ret4_prod_mnq_cl`, `ret4_prod_btc_cl`
- Positivo = ativos andando juntos, Negativo = divergindo

**Volatilidade Spreads**
- `vol_spread_mnq_btc`, `vol_spread_mnq_cl`, `vol_spread_btc_cl`

**Bollinger Spreads**
- `bb_spread_mnq_btc`, `bb_spread_mnq_cl`, `bb_spread_btc_cl`

**Alinhamentos (Binarios)**
- Alinhamento SMA50: pares + todos os 3
- Alinhamento EMA20: pares + todos os 3 + bias MNQ+BTC

**DI Spread**
- `di_spread_mnq`, `di_spread_btc`, `di_spread_cl` = DI+ − DI-

#### Triggers Compostos
- `strong_div`: price_div_cl < 0 AND ADX > 17
- `us_prime_setup`: strong_div + horario US (9-17h)
- `prime_setup`: strong_div + horario 18-21h
- `triple_signal`: RSI(BTC) < 45 E divergencias RSI negativas
- `cl_down_mnq_up`: CL caindo E MNQ subindo

#### Features Temporais
- `hour`, `dow` (hora e dia da semana)
- `is_us_session` (9-17h), `is_us_morning` (9-13h), `is_us_afternoon` (14-17h)
- `is_evening` (18-21h)

### 3. Treino (`train_model`)

```python
XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.75,
    min_child_weight=8,
    objective='multi:softprob',  # 3 classes
    num_class=3,
    eval_metric='mlogloss',
    early_stopping_rounds=40,
)
```

- **Walk-Forward**: 70% treino, 30% teste (cronologico)
- **Class weights**: balanceamento por frequencia inversa
- **Early stopping**: para se nao melhorar por 40 rodadas

## Feature Sets Disponiveis

| Modo | Flag | Features | AUC |
|------|------|----------|-----|
| **Otimizado (default)** | (nenhuma) | **38** | **0.6017** |
| Completo | `--full` | 52 | — |
| Spreads-only | `--spreads-only` | 63 | — |

## Top Features por Importancia (SMA50 + EMA20, RSI 21)

Atualizar rodando `train.py` e abrindo `feature_importance.png`.

## Melhor Spread: `price_div_abs`

`price_div_abs` = |ret1(MNQ) × ret1(CL)|

Mede a **magnitude da divergencia de preco** entre MNQ e CL no candle atual. Quando os dois ativos andam em direcoes opostas (um sobe e outro desce), o produto e negativo, e o valor absoluto e alto. Isso indica:

- **Divergencia forte**: oportunidade de reversao ou continuidade
- **Combinado com ADX > 17**: gatilho `strong_div` com 47.1% de acertar LONG

## Resultados Obtidos (Walk-Forward, SMA50 + EMA20, RSI 21, ADX 17)

```
Dataset: 5400 amostras  |  SHORT=38.1%  NEUTRO=16.7%  LONG=45.1%

              precision    recall  f1-score
   SHORT    0.39      0.32      0.35
  NEUTRO    0.37      0.57      0.45
    LONG    0.48      0.38      0.43

ROC-AUC (macro ovr): 0.6017
```

### Analise dos Gatilhos

| Gatilho | N | LONG | SHORT |
|---------|---|------|-------|
| strong_div | 2459 | 47.1% | 36.5% |
| us_prime_setup | 2459 | 47.1% | 36.5% |

### Vies EMA20 (MNQ + BTC)

| Estado | LONG real | SHORT real |
|--------|-----------|------------|
| Ambos abaixo da EMA20 | 43.6% | **37.0%** |
| Misturado | 40.7% | 31.7% |
| Ambos acima da EMA20 | **40.6%** | 30.2% |

## Estrategia Recomendada

### Para LONG

1. **Sessao**: US 9-17h (preferencialmente 10h ou 13h)
2. **Divergencia**: `price_div_abs` alto + `price_div_cl` < 0 (MNQ e CL opostos)
3. **Regime**: EMA20 bias >= 1 (pelo menos 1 ativo acima da EMA20)
4. **Tendencia**: ADX > 17 (ADX 17 no modelo)
5. **RSI**: `div_cl` < 0 (RSI do MNQ abaixo do CL = MNQ sobrevendido relativo)
6. **Volatilidade**: `vol_mnq` elevado

### Para SHORT

1. **Sessao**: US 9-17h
2. **Divergencia**: `price_div_abs` alto
3. **Regime**: EMA20 bias <= 1 (pelo menos 1 abaixo)
4. **Tendencia**: ADX > 17 (ADX 17 no modelo)
5. **RSI**: `div_cl` > 0 (RSI do MNQ acima do CL = MNQ sobrecomprado relativo)

## Como Usar

```bash
# Pipeline completo
cd ml
run_ml.bat                     # instala deps, coleta dados, treina
# Ou passo a passo:
python collect_data.py         # baixa dados historicos
python train.py --all-hours    # treina modelo otimizado (38 features)
python train.py --all-hours --full  # treina modelo completo (52 features)

# Opcoes avancadas
python train.py --forward 8 --all-hours          # prever 8h a frente
python train.py --forward 4 --interval 1d        # timeframe diario
python train.py --all-hours --spreads-only       # so features spread

# Predicao em tempo real (chamado pelo server.js)
python predict.py              # JSON com prob_long, prob_short, sinal
```

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
| `run_ml.bat` | Script automatizado do pipeline |
| `analyze_importance.py` | Analisa importancia das features |
| `data.db` | Dados historicos (ignorado pelo git) |
| `model.pkl` | Modelo treinado (ignorado pelo git) |
| `feature_importance.png` | Grafico de importancia |
| `requirements.txt` | Dependencias Python |

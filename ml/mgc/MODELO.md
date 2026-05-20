# Modelo Preditivo MGC (Micro Gold) — XGBoost Multiclasse

## Visao Geral

Modelo de Machine Learning para prever a direcao do **MGC** (Micro Gold Futures) nas proximas 4 horas. Classifica em 3 classes:

| Classe | Label | Descricao |
|--------|-------|-----------|
| SHORT | 0 | Preco vai cair > 0.1% |
| NEUTRO | 1 | Preco vai ficar entre -0.1% e +0.1% |
| LONG | 2 | Preco vai subir > 0.1% |

## Arquitetura

```
yfinance (MGC, MNQ, BTC)
       |
 collect_data.py
       |
   data.db (SQLite)
       |
  train.py -> build_features() -> 70 features
       |
   XGBoost (multiclass, all-hours)
       |
   model.pkl + model_kz_{asia,london,ny}.pkl + feature_importance.png
       |
  predict.py -> JSON -> server.js -> live.html (dashboard)
```

## Timeframe

- **Intervalo dos candles**: 1 hora
- **Forward**: 4 horas (configuravel com `--forward`)
- **Periodo de dados**: ~2 anos (1h) + 5 anos (1d)
- **3 ativos**: MGC=F (Micro Gold — primario), MNQ=F (Mini Nasdaq), BTC-USD (Bitcoin)
- **Sessao**: All-hours (inclui Asia, Londres, NY)

## Diferencas Estruturais vs MNQ/CL

O MGC e o modelo mais distinto dos 4. Enquanto MNQ e CL sao modelos de divergencia entre ativos, o MGC e um modelo de **sessao e niveis**.

| Aspecto | MGC | MNQ | CL |
|---------|:---:|:---:|:---:|
| Feature #1 | is_us_afternoon (5.3%) | price_div_abs (4.3%) | us_prime_setup (5.4%) |
| Kill Zone total | **17.8%** | 41.8% | 35.8% |
| Divergencia preco | 7.2% | 11.9% | 12.4% |
| AUC | **0.535** | 0.602 | 0.523 |
| Sessoes | All-hours | US session | US session |
| Niveis mensais | **Top 10** | Irrelevante | Irrelevante |

## Top Features por Importancia (Gain)

```
Rank Feature                     Gain%
-----------------------------------------
  1  is_us_afternoon              5.30
  2  is_london                    4.42
  3  hour                         3.87
  4  is_us_morning                3.87
  5  above_pmh                    2.29
  6  kz_overlap                   1.90
  7  prev_day_range_pct           1.90
  8  dist_to_pml                  1.90
  9  dow                          1.89
 10  bb_cl                        1.88
 11  dist_to_mday_h               1.84
 12  bb_spread_cl_mnq             1.82
 13  dist_ema20_cl                1.80
 14  rsi_abs_cl_mnq               1.80
 15  vol_cl                       1.77
 16  sma50_alignment              1.76
 17  kz_range                     1.76
 18  dist_to_do                   1.76
 19  adx_mnq                      1.75
 20  above_wo                     1.75
 21  ema20_alignment              1.74
 22  dist_to_pdl                  1.72
 23  dist_to_pwh                  1.71
 24  dist_to_pdh                  1.70
 25  dist_to_pwl                  1.70
```

### Categorias

| Categoria | Peso | Interpretacao |
|-----------|:----:|---------------|
| **Kill Zone / Sessao** | **17.8%** | Modelo dominantemente temporal |
| SMA/EMA | 11.4% | Alinhamento de medias moveis |
| **Divergencias** | **7.2%** | Pouca relevancia vs MNQ/CL |
| ADX | 6.6% | Forca de tendencia |
| Volatilidade | 3.7% | Bollinger/vol do CL e MGC |
| Retornos | 2.9% | Retornos recentes |
| RSI | 1.5% | Quase irrelevante |

## Interpretacao das Features Principais

### 1. is_us_afternoon (5.3% — Feature #1)
O periodo da **tarde americana (14-17h)** e o sinal mais forte para o MGC. O ouro tem comportamento distinto neste horario vs manha US (9-13h).

### 2. is_london (4.4%)
A **sessao de Londres (8-14h)** e a segunda feature mais importante — mais relevante que a propria sessao NY. Isto faz sentido porque Londres e o maior centro de negociacao de ouro do mundo (LBMA).

### 3. hour (3.9%) + is_us_morning (3.9%)
O **horario bruto** importa. O modelo aprendeu padroes intraday especificos do MGC que nao sao capturados apenas pelas sessoes.

### 4. above_pmh (2.3%) — Acima do maximo mensal
Nivel de **resistencia mensal**. Quando o MGC esta acima do maximo do mes anterior, o modelo ajusta a probabilidade. Isto e unico do MGC — nem MNQ nem CL tem niveis mensais no top 15.

### 5. Nao ha divergencia forte
`price_div_abs` (a feature #1 do MNQ) nao aparece no top 25. O MGC nao aprendeu a usar divergencia MGC-MNQ como sinal relevante. O modelo prefere:
- Saber **que horas sao** (sessao)
- Saber **onde esta** (niveis mensais/semanais)
- Saber **o que o CL esta a fazer** (bb_cl, bb_spread_cl_mnq, vol_cl)

## Triggers

Os mesmos gatilhos dos outros modelos, mas com relevancia muito menor:

| Gatilho | Condicao | Relevancia |
|---------|----------|:----------:|
| strong_div | price_div_cl < 0 AND ADX > 14 | Baixa (divergencia nao e relevante) |
| us_prime_setup | strong_div + US session | Baixa |
| prime_setup | strong_div + evening | Baixa |

## Performance

| Metrica | Valor |
|---------|:-----:|
| AUC | **0.535** |
| Forward | 4 horas |
| Total features | 70 |
| Sessoes | All-hours (Asia, Londres, NY) |
| Ensemble KZ | Sim (modelos especialistas por sessao) |

## Resumo Pratico

1. **MGC segue o relogio**: a feature mais importante e saber se e tarde US ou sessao Londres
2. **Niveis mensais importam**: acima/abaixo do maximo/minimo do mes
3. **MGC segue o CL**: Bollinger do CL, spread Bollinger CL-MGC, volatilidade do CL estao no top 15
4. **Divergencias sao irrelevantes**: strong_div nao tem o mesmo edge que no CL (+21.8%)
5. **AUC moderado (0.535)**: melhor que o CL (0.523) mas pior que o MNQ (0.602)

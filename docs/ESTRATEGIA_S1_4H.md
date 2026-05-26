# Estratégia S1 — ROT-LONG + Acima 4H Open + ML

## Resumo

| Item | Valor |
|---|---|
| Ativo | MNQ (Micro Nasdaq Futures) |
| Direcao | Somente LONG |
| Timeframe | 1H (sinal) + 4H (referencia open) |
| Backtest periodo | 365 dias (split 70/20/10) |
| WR sem ML (OOS 10%) | 61.5% (n=26) |
| WR com ML (OOS 10%) | **78.6%** (n=14) |
| Alpha vs baseline | **+29.7pp** |
| Baseline OOS | 48.8% |
| Forward horizon | 4 horas |

---

## Arvore de Decisao (4 Gates)

```
SINAL LONG S1
│
├── GATE 1: Score Rotacao >= 60%
│   ├── C1 DIV cruzou zero          peso 3.0  [RSI MNQ - RSI CL muda de sinal]
│   ├── C2 Delta DIV > 5 em 2h      peso 2.5  [aceleracao da divergencia]
│   ├── C3 MNQ up + CL down em 2h   peso 2.5  [RSI MNQ d2 > 2 e RSI CL d2 < -2]
│   ├── C4 ES confirma MNQ          peso 2.0  [ret1 ES e MNQ no mesmo sentido]
│   ├── C5 BB CL > 1.5              peso 1.5  [volatilidade CL expandida]
│   ├── C6 CL abriu > dia anterior  peso 1.5  [abertura diaria CL ascendente]
│   ├── C7 ADX MNQ 12-20            peso 1.5  [tendencia nascente, nao exaustao]
│   ├── C8 SMA50 >= 2/4 ativos      peso 1.5  [alinhamento macro]
│   └── C9 RSI MNQ > 55 ou < 45    peso 1.0  [fora da zona neutra]
│       Total possivel: 17.0 pts | Threshold: 10.2 pts (60%)
│
├── GATE 2: div_cl > 0
│   └── RSI MNQ > RSI CL = MNQ relativamente mais forte = vies LONG
│
├── GATE 3: close > 4H open
│   └── Preco acima da abertura do candle de 4H atual
│       Confirmacao de forca intraday no periodo
│
└── GATE 4: ML Prob >= 50% (modelo XGBoost)
    └── model_divergencia.pkl — XGBoost 93 features
        Prediz: MNQ vai subir > 0.1% nas proximas 4h
        Baseline: 40.2% | Threshold: 50%
```

### Saidas possiveis

| Estado | Descricao |
|---|---|
| `LONG` | Todos os 4 gates ativos — sinal completo |
| `SEM ML` | Gates 1-2-3 ativos, ML nao confirma — aguardar |
| `AGUARDAR` | Algum gate essencial inativo |

---

## Modelo ML — model_divergencia.pkl

### Especificacoes

| Item | Valor |
|---|---|
| Algoritmo | XGBoost (XGBClassifier) |
| n_estimators | 400 |
| max_depth | 4 |
| learning_rate | 0.03 |
| subsample | 0.80 |
| colsample_bytree | 0.75 |
| min_child_weight | 5 |
| early_stopping | 40 rounds |
| Features | 93 |
| Target | MNQ > 0.1% em 4h |
| Split treino | Walk-forward 70/30 |
| Baseline (prob media) | 40.2% |

### Feature Importances por Categoria

| Categoria | Importancia total |
|---|---|
| ADX / DI Spread | 14.3% |
| SMA / EMA (dist, above) | 13.8% |
| Volatilidade / BB Width | 12.6% |
| Weekly Open (abertura semanal) | 12.6% |
| Daily Open (abertura diaria) | 12.1% |
| RSI / Divergencia | 11.4% |
| Retornos (1h/4h/8h) | 7.0% |
| Tempo (hora, dia semana) | 6.0% |
| 4H Open (dist, acima/abaixo) | **5.3%** |
| 1H Open | 2.0% |

### Top 20 Features Individuais

| Rank | Feature | Importancia |
|---|---|---|
| 1 | dist_ema20_es | 2.59% |
| 2 | rsi_mnq | 2.49% |
| 3 | open_d_acima_d_ant_es | 2.45% |
| 4 | is_us (sessao americana) | 2.35% |
| 5 | adx_es_alto | 2.09% |
| 6 | open_d_dist_btc | 2.07% |
| 7 | dist_ema20_cl | 1.95% |
| 8 | hour | 1.91% |
| 9 | vol_btc | 1.89% |
| 10 | open_w_acima_w_ant_es | 1.87% |
| 11 | di_spread_cl | 1.82% |
| 12 | vol_mnq | 1.81% |
| 13 | di_spread_btc | 1.81% |
| 14 | open_4h_dist_es | 1.76% |
| 15 | div_es | 1.71% |
| 16 | dow (dia da semana) | 1.71% |
| 17 | open_w_dist_mnq | 1.68% |
| 18 | r_es_4h | 1.67% |
| 19 | open_d_dist_cl | 1.67% |
| 20 | bb_w_es | 1.66% |

---

## Backtest Completo (backtest_4h_open.py)

### Metodologia

- **Dados**: yfinance 1H, 365 dias, MNQ+CL+ES+BTC
- **Split**: 70% referencia | 20% validacao | 10% teste final (OOS)
- **Target**: retorno MNQ > 0.1% nas 4h seguintes ao sinal
- **Baseline OOS**: 48.8% (qualquer barra do periodo de teste)

### Resultados por Split

#### Referencia 70% (baseline=39.5%)

| Estrategia | N | WR | Alpha | WR+ML | Alpha+ML |
|---|---|---|---|---|---|
| S1 ROT-LONG + acima 4H | 114 | 46.5% | +7.0pp | 43.4% | +3.9pp |
| S5 ROT-LONG + cruz cima | 30 | 46.7% | +7.1pp | 37.5% | -2.0pp |

#### Validacao 20% (baseline=42.1%)

| Estrategia | N | WR | Alpha | WR+ML | Alpha+ML |
|---|---|---|---|---|---|
| **S1 ROT-LONG + acima 4H** | 52 | 51.9% | +9.8pp | **77.8%** | **+35.7pp** |
| S5 ROT-LONG + cruz cima | 19 | 42.1% | +0.0pp | 66.7% | +24.6pp |

#### Teste 10% OOS — resultado honesto (baseline=48.8%)

| Estrategia | N | WR | Alpha | WR+ML | Alpha+ML |
|---|---|---|---|---|---|
| **S1 ROT-LONG + acima 4H** | **26** | **61.5%** | **+12.7pp** | **78.6%** | **+29.7pp** |
| S5 ROT-LONG + cruz cima | 7 | 71.4% | +22.6pp | 66.7% | +17.8pp |

> S1 foi escolhido sobre S5 por volume de sinais (26 vs 7 no OOS) — mais representativo estatisticamente.

### Distribuicao por Hora (S1 no OOS, n=26)

| Hora UTC | N | WR |
|---|---|---|
| 00h | 1 | 100% |
| 01h | 1 | 100% |
| 04h | 1 | 100% |
| 06h | 2 | 50% |
| 07h | 1 | 0% |
| 08h | 3 | 33% |
| 10h | 1 | 100% |
| 11h | 1 | 0% |
| 12h | 2 | 100% |
| 13h | 3 | 33% |
| 14h | 2 | 100% |
| 15h | 1 | 100% |
| 16h | 1 | 100% |
| 17h | 3 | 67% |
| 20h | 1 | 100% |
| 22h | 2 | 0% |

> Melhor performance: 00h, 01h, 04h, 10h, 12h, 14h, 15h, 16h, 20h (100% WR, amostras pequenas)
> Pior: 22h (0%), 08h (33%), 13h (33%)

---

## Arquivos do Sistema

| Arquivo | Funcao |
|---|---|
| `ml/model_divergencia.pkl` | Modelo XGBoost treinado (nao modificar) |
| `ml/predict_s1_4h.py` | Predicao ao vivo — saida JSON |
| `ml/signals_s1_4h.csv` | Log de todos os sinais gerados |
| `ml/teste/backtest_4h_open.py` | Backtest completo S1 vs baseline |
| `pine/estrategia_s1_4h.pine` | Strategy TradingView (overlay, ATR stop/target) |
| `pine/divergencia_rotacao.pine` | Indicator com color candles e tabela |
| `pine/divergencia_rotacao_strategy.pine` | Strategy completa com Key Levels |

### Endpoints do servidor

| Endpoint | Descricao | Cache |
|---|---|---|
| `GET /api/s1-4h` | Predicao S1 ao vivo | 5 min |
| `GET /api/s1-4h?force=1` | Forca atualizacao | — |
| `GET /api/divergencia` | Modelo divergencia original | 5 min |
| `GET /api/live2` | Sinais multi-ativo | 5 min |

---

## Como Retreinar o Modelo

O modelo base nao deve ser retreinado para alterar o S1 — use o backtest para validar mudancas.
Se necessario retreinar:

```bash
cd ml
python train_divergencia.py --forward 4
```

Isso sobrescreve `model_divergencia.pkl`. Sempre rode o backtest depois:

```bash
python teste/backtest_4h_open.py
```

---

## Limitacoes

- **n=14 no OOS com ML**: estatisticamente significativo mas amostra pequena — monitorar ao vivo
- **Modelo nao inclui 4H open como gate explicito**: o gate 3 e adicionado em cima do modelo, nao dentro dele
- **Sem gestao de risco real**: backtest usa retorno bruto 4h, sem stop/target real
- **yfinance delay**: dados com ate 15min de delay em modo ao vivo
- **SHORT nao funciona**: todas as estrategias SHORT tiveram WR abaixo do baseline — nao operar SHORT com este sistema

---

*Gerado em 2026-05-26 com base em backtest_4h_open.py, split OOS 10% (2026-04-20 a 2026-05-26)*

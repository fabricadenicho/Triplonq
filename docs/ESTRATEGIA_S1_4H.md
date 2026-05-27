# Estrategia S1/S2 — ROT-LONG/SHORT + 4H Open + Regime + ML

## Visao Geral

| Item | S1 (LONG) | S2 (SHORT) |
|---|---|---|
| Ativo | MNQ (Micro Nasdaq) | MNQ (Micro Nasdaq) |
| Direcao | Somente LONG | Somente SHORT |
| Regime exigido | Acima SMA200 1h | Abaixo SMA200 1h |
| Backtest periodo | 720 dias, 80/10/10 split | 720 dias, 80/10/10 split |
| Baseline OOS | 46.6% | 32.9% |
| WR sem ML (OOS) | 52.4% | 54.3% |
| WR com ML (OOS) | **62.0%** | **54.3%** |
| Edge vs baseline | **+15.4pp** | **+21.4pp** |
| Frequencia | ~3-5 sinais/semana | ~1-2 sinais/semana |
| Forward horizon | 4 horas | 4 horas |

---

## Arvore de Decisao S1 — LONG (5 Gates)

```
SINAL LONG S1
|
|-- GATE 1: Score Rotacao >= 60%
|   |-- C1 DIV cruzou zero          peso 3.0  [RSI MNQ - RSI CL muda de sinal]
|   |-- C2 Delta DIV > 5 em 2h      peso 2.5  [aceleracao da divergencia]
|   |-- C3 MNQ up + CL down em 2h   peso 2.5  [RSI MNQ d2 > 2 e RSI CL d2 < -2]
|   |-- C4 ES confirma MNQ          peso 2.0  [ret1 ES e MNQ no mesmo sentido]
|   |-- C5 BB CL > 1.5              peso 1.5  [volatilidade CL expandida]
|   |-- C6 CL abriu > dia anterior  peso 1.5  [abertura diaria CL ascendente]
|   |-- C7 ADX MNQ 12-20            peso 1.5  [tendencia nascente, nao exaustao]
|   |-- C8 SMA50 >= 2/4 ativos      peso 1.5  [alinhamento macro]
|   +-- C9 RSI MNQ > 55 ou < 45    peso 1.0  [fora da zona neutra]
|       Total possivel: 17.0 pts | Threshold: 10.2 pts (60%)
|
|-- GATE 2: div_cl > 0
|   +-- RSI MNQ > RSI CL = MNQ relativamente mais forte = vies LONG
|
|-- GATE 3: close > 4H open
|   +-- Preco acima da abertura do candle de 4H atual
|
|-- GATE 4: ML Prob >= 50%
|   +-- model_s1_4h.pkl — XGBoost 93 features, AUC=0.5613
|
+-- GATE 5: Regime BULL
    +-- MNQ close > SMA200 (1h, 200 periodos)
        Filtra: so operar LONG em tendencia de alta macro
```

## Arvore de Decisao S2 — SHORT (5 Gates, simetrico)

```
SINAL SHORT S2
|
|-- GATE 1: Score Rotacao SHORT >= 60%
|   |-- C1 DIV cruzou zero          peso 3.0  [igual ao S1]
|   |-- C2 Delta DIV > 5 em 2h      peso 2.5  [igual ao S1]
|   |-- C3 MNQ down + CL up em 2h   peso 2.5  [INVERTIDO: RSI MNQ d2 < -2 e RSI CL d2 > 2]
|   |-- C4 ES confirma MNQ          peso 2.0  [igual ao S1]
|   |-- C5 BB CL > 1.5              peso 1.5  [igual ao S1]
|   |-- C6 CL abriu < dia anterior  peso 1.5  [INVERTIDO: abertura CL descendente]
|   |-- C7 ADX MNQ 12-20            peso 1.5  [igual ao S1]
|   |-- C8 SMA50 <= 2/4 ativos      peso 1.5  [INVERTIDO: maioria abaixo da SMA50]
|   +-- C9 RSI MNQ > 55 ou < 45    peso 1.0  [igual ao S1]
|
|-- GATE 2: div_cl < 0
|   +-- RSI CL > RSI MNQ = CL relativamente mais forte = vies SHORT
|
|-- GATE 3: close < 4H open
|   +-- Preco abaixo da abertura do candle de 4H atual
|
|-- GATE 4: ML Prob >= 50%
|   +-- model_s2_4h.pkl — XGBoost 93 features, AUC=0.5506
|
+-- GATE 5: Regime BEAR
    +-- MNQ close < SMA200 (1h, 200 periodos)
        Filtra: so operar SHORT em tendencia de baixa macro
```

---

## Modelos ML

### model_s1_4h.pkl (LONG)

| Item | Valor |
|---|---|
| Algoritmo | XGBoost (XGBClassifier) |
| n_estimators | 500 |
| max_depth | 4 |
| learning_rate | 0.025 |
| Features | 93 |
| Target | MNQ > +0.1% em 4h |
| Dados treino | 720 dias, split 80/10/10 temporal |
| Baseline (barras S1) | 46.6% |
| AUC OOS | 0.5613 |
| WR S1 + ML OOS | 62.0% |

### model_s2_4h.pkl (SHORT)

| Item | Valor |
|---|---|
| Algoritmo | XGBoost (XGBClassifier) |
| n_estimators | 500 |
| max_depth | 4 |
| learning_rate | 0.025 |
| Features | 93 |
| Target | MNQ < -0.1% em 4h |
| Dados treino | 720 dias, split 80/10/10 temporal |
| Baseline (barras S2) | 32.9% |
| AUC OOS | 0.5506 |
| WR S2 + ML OOS | 54.3% (N=35) |

### Comparacao com modelo anterior

| | model_divergencia.pkl (antigo) | model_s1_4h.pkl (atual) |
|---|---|---|
| Treinado em | Todos os candles MNQ | Barras que passam no filtro S1 |
| Baseline | 40.2% (qualquer barra) | 46.6% (barras S1 filtradas) |
| Split | ~80/20 simples | 80/10/10 temporal |
| Dados | ~360 dias | 720 dias |
| AUC OOS | 0.5425 | **0.5613** |
| WR no contexto S1 | ~52% | **62%** |

---

## Backtest Metodologia

- **Dados**: yfinance 1H, 720 dias, MNQ+CL+ES+BTC
- **Split temporal**: 80% treino | 10% validacao interna | 10% OOS cego (nunca visto)
- **Rollover filter**: remove barras com |ret1h_mnq| > 3% (rollovers)
- **Regime filter**: S1 so conta barras BULL (acima SMA200); S2 so conta barras BEAR
- **Early stopping**: 50 rounds no XGBoost com validacao interna 80/20 do treino
- **Target LONG**: MNQ close[+4] / close - 1 > +0.001
- **Target SHORT**: MNQ close[+4] / close - 1 < -0.001

### Resultados S1 por periodo

| Periodo | N | WR sem ML | WR com ML | Baseline | Edge ML |
|---|---|---|---|---|---|
| Treino 80% | — | — | — | ~46% | — |
| Val 10% | — | — | — | ~46% | — |
| **OOS 10% (cego)** | **~50** | **52.4%** | **62.0%** | **46.6%** | **+15.4pp** |

### Impacto do Regime Filter no S1 OOS

| Regime | WR sem filtro | WR com filtro |
|---|---|---|
| BULL (acima SMA200) | 62% | 62% |
| BEAR (abaixo SMA200) | 35% | bloqueado |
| **Geral** | — | **+11pp** |

---

## Arquivos do Sistema

| Arquivo | Funcao | Status |
|---|---|---|
| `ml/model_s1_4h.pkl` | Modelo XGBoost LONG | Ativo |
| `ml/model_s2_4h.pkl` | Modelo XGBoost SHORT | Ativo |
| `ml/model_divergencia.pkl` | Modelo original divergencia | Mantido (usado por /api/divergencia) |
| `ml/predict_s1_4h.py` | Predicao S1 ao vivo — JSON stdout | Ativo |
| `ml/predict_s2_4h.py` | Predicao S2 ao vivo — JSON stdout | Ativo |
| `ml/backtest_s1_4h.py` | Backtest S1 com 80/10/10 split | Referencia |
| `ml/backtest_s2_4h.py` | Backtest S2 com 80/10/10 split | Referencia |
| `ml/signals_s1_4h.csv` | Log de sinais S1 gerados | Ativo |
| `ml/signals_s2_4h.csv` | Log de sinais S2 gerados | Ativo |
| `ml/validate_live.py` | Resolve outcomes de sinais reais | Ativo |
| `ml/compare_rot_s1.py` | Comparacao Rotacao vs S1 (analise) | Referencia |
| `pine/estrategia_s1s2_4h.pine` | Strategy TradingView S1+S2+Regime | Ativo |
| `pine/divergencia_rotacao.pine` | Histograma com RSI/DIV/Score | Ativo |
| `docs/trigger-rotacao.md` | Documentacao Trigger de Rotacao | Referencia |

### Endpoints do servidor

| Endpoint | Descricao | Cache | Auto-loop |
|---|---|---|---|
| `GET /api/s1-4h` | Predicao S1 LONG ao vivo | 5 min | Sim, 5 min |
| `GET /api/s1-4h?force=1` | Forca atualizacao S1 | — | — |
| `GET /api/s2-4h` | Predicao S2 SHORT ao vivo | 5 min | Sim, 5 min |
| `GET /api/s2-4h?force=1` | Forca atualizacao S2 | — | — |
| `GET /api/divergencia` | Modelo divergencia original | 5 min | Nao |
| `GET /api/live2` | Sinais multi-ativo propfirm | 5 min | Sim, 1 min |

### Auto-loop Telegram

| Loop | Frequencia | Condicao de envio |
|---|---|---|
| `checkSignals()` | 1 min | Sinal live2 muda de estado |
| `checkS1S2()` | 5 min | S1 muda para LONG ou S2 muda para SHORT |

---

## Como Retreinar

Para retreinar S1 (LONG):
```bash
cd ml
python backtest_s1_4h.py
```
Isso treina e salva `model_s1_4h.pkl` + gera `backtest_s1_trades.csv`.

Para retreinar S2 (SHORT):
```bash
cd ml
python backtest_s2_4h.py
```
Isso treina e salva `model_s2_4h.pkl` + gera `backtest_s2_trades.csv`.

Apos retreinar, reiniciar o servidor para carregar o novo modelo.

---

## Limitacoes

- **S2 frequencia baixa**: N=35 no OOS — amostra menor que o S1, monitorar mais tempo ao vivo antes de confiar plenamente
- **Regime BEAR raro em bull market**: S2 so dispara quando MNQ esta abaixo da SMA200 — pode ficar semanas sem sinal em mercados de alta
- **S1 e S2 sao mutuamente exclusivos**: por construccao, nunca disparam ao mesmo tempo (um exige BULL, outro BEAR)
- **yfinance delay**: dados com ate 15min de delay em modo ao vivo
- **Forward 4h fixo**: o modelo nao tem gestao de risco interna — stop/target devem ser definidos pelo operador

---

*Atualizado em 2026-05-27 — modelo_s1_4h.pkl (720d, 80/10/10, AUC=0.5613) + S2 SHORT + Regime SMA200*

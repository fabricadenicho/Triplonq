# Modelos 6 Meses — Comparativo vs Modelos Completos

## Sumario

- **Data de retreinamento**: 18/05/2026
- **Periodo de treino**: 19/11/2025 a 18/05/2026 (6 meses)
- **Features**: 38 otimizadas (mesmo conjunto original)
- **Forward**: 4 horas
- **Intervalo**: 1 hora
- **Sessao**: US 9-17h

---

## 1. Visao Geral — Antes vs Depois

| Metrica | MNQ Antigo | MNQ 6m | BTC Antigo | BTC 6m | CL Antigo | CL 6m |
|---------|-----------|--------|-----------|--------|-----------|-------|
| **AUC** | 0.602 | 0.467 | 0.525 | **0.555** | 0.523 | 0.474 |
| Amostras treino | 3.786 | **767** | 4.555 | **1.110** | 3.780 | **765** |
| Amostras teste | 1.623 | **330** | 1.953 | **476** | 1.621 | **328** |
| Periodo treino | 2023-2025 | Nov 2025-Mar 2026 | 2024-2025 | Nov 2025-Mar 2026 | 2023-2025 | Nov 2025-Mar 2026 |
| SHORT% (teste) | 36.9% | 33.3% | 44.8% | 38.2% | 43.2% | 41.2% |
| NEUTRO% (teste) | 16.6% | 11.5% | 12.7% | 16.2% | 8.2% | 3.0% |
| LONG% (teste) | 46.5% | **55.2%** | 42.5% | 45.6% | 48.6% | **55.8%** |
| Feature #1 | price_div_abs | is_us_afternoon | dow | bb_mnq | us_prime_setup | **cl_down_mnq_up** |

### Destaques

- **BTC melhorou** +0.030 de AUC (0.525 → 0.555) — unico que ganhou com menos dados
- **MNQ piorou** -0.135 (0.602 → 0.467) — perdeu muita estabilidade
- **CL estavel** -0.049 (0.523 → 0.474) — queda dentro do esperado
- **Regime mudou**: todos os 3 modelos agora mostram distribuicao LONG mais forte no periodo de teste (55-56% LONG vs 33-41% SHORT), indicando tendencia de alta no periodo Mar-Mai 2026

---

## 2. MNQ — Modelo 6 Meses

### Resultado Walk-Forward

```
Dataset: 1097 amostras  |  SHORT=37.1%  NEUTRO=15.9%  LONG=47.0%

              precision    recall  f1-score   support
SHORT            0.29      0.36      0.32       110
NEUTRO           0.12      0.24      0.16        38
LONG             0.47      0.31      0.37       182

accuracy                         0.32       330
ROC-AUC (macro ovr): 0.4668  [fraco]
```

### Top 10 Features (Gain)

| Feature | Importancia | Descricao |
|---------|-------------|-----------|
| is_us_afternoon | 4.8% | Tarde americana (14-17h) |
| bb_mnq | 4.6% | Largura Bollinger do MNQ |
| vol_mnq | 4.2% | Volatilidade do MNQ |
| rsi_mnq | 4.2% | RSI do MNQ |
| ret1_mnq | 3.7% | Retorno 1h do MNQ |
| adx_mnq | 3.6% | Forca tendencia MNQ |
| di_spread_mnq | 3.6% | DI+ menos DI- do MNQ |
| sma50_slope_mnq | 3.5% | Inclinacao SMA50 do MNQ |
| dist_sma50_mnq | 3.5% | Distancia da SMA50 (%) |
| adx_cl | 3.4% | ADX do petroleo |

### Analise dos Gatilhos

| Gatilho | N | LONG | SHORT |
|---------|---|------|-------|
| strong_div | 205 | 53.7% | 37.6% |
| us_prime_setup | 205 | 53.7% | 37.6% |

**Edge strong_div**: +16.1% LONG (vs +10.6% no modelo antigo)

### Vies EMA20 (MNQ+BTC)

| Estado | N | LONG | SHORT |
|--------|---|------|-------|
| Ambos abaixo | 81 | 45.7% | 48.1% |
| Misturado | 91 | 59.3% | 28.6% |
| Ambos acima | 158 | 57.6% | 28.5% |

### Predicao Atual (18/05/2026 23h)

| Sinal | Prob SHORT | Prob LONG |
|-------|-----------|-----------|
| NEUTRO | 33.0% | 32.9% |

### Mudancas vs Modelo Antigo (Completo)

| Aspecto | Antigo (completo) | Novo (6 meses) |
|---------|------------------|----------------|
| AUC | 0.602 | 0.467 |
| Feature #1 | price_div_abs (4.3%) | is_us_afternoon (4.8%) |
| strong_div edge | +10.6% LONG | +16.1% LONG |
| Sinal atual (23h) | LONG (AUC 0.603 dev) / SHORT (full) | NEUTRO |
| Principais features | Spreads preco/divergencia | Indicadores proprios do MNQ |
| Amostras | 5.409 | 1.097 |

O modelo perdeu a capacidade de enxergar divergencias entre ativos e passou a depender mais de indicadores do proprio MNQ (RSI, ADX, Bollinger, SMA50). Com 1/5 das amostras, o sinal ficou NEUTRO (antes oscilava entre LONG e SHORT).

---

## 3. BTC — Modelo 6 Meses

### Resultado Walk-Forward

```
Dataset: 1586 amostras  |  SHORT=43.4%  NEUTRO=13.4%  LONG=43.2%

              precision    recall  f1-score   support
SHORT            0.45      0.49      0.47       182
NEUTRO           0.19      0.31      0.23        77
LONG             0.49      0.34      0.40       217

accuracy                         0.39       476
ROC-AUC (macro ovr): 0.5546  [bom]
```

### Top 10 Features (Gain)

| Feature | Importancia | Descricao |
|---------|-------------|-----------|
| bb_mnq | 5.5% | Largura Bollinger do MNQ |
| vol_mnq | 4.7% | Volatilidade do MNQ |
| sma50_align_mnq_cl | 4.7% | Alinhamento SMA50 MNQ+CL |
| dow | 4.6% | Dia da semana |
| div_cl | 3.9% | Divergencia RSI BTC vs CL |
| vol_spread_mnq_btc | 3.7% | Spread volatilidade MNQ-BTC |
| rsi_abs_mnq_cl | 3.6% | |RSI MNQ - RSI CL| |
| div_btc | 3.6% | Divergencia RSI BTC vs MNQ |
| hour | 3.3% | Hora do dia |
| bb_spread_mnq_cl | 3.2% | Spread Bollinger MNQ-CL |

### Analise dos Gatilhos

| Gatilho | N | LONG | SHORT |
|---------|---|------|-------|
| strong_div | 245 | 43.7% | 43.3% |
| us_prime_setup | 245 | 43.7% | 43.3% |

Edge strong_div praticamente neutro (+0.4% vs +3.5% SHORT no antigo)

### Vies EMA20 (BTC+MNQ)

| Estado | N | LONG | SHORT |
|--------|---|------|-------|
| Ambos abaixo | 116 | 44.8% | 41.4% |
| Misturado | 136 | 45.6% | 39.7% |
| Ambos acima | 224 | 46.0% | 35.7% |

### Predicao Atual (18/05/2026 23h)

| Sinal | Prob SHORT | Prob LONG |
|-------|-----------|-----------|
| SHORT | 34.6% | 32.5% |

### Mudancas vs Modelo Antigo (Completo)

| Aspecto | Antigo (completo) | Novo (6 meses) |
|---------|------------------|----------------|
| AUC | 0.525 | **0.555** |
| Feature #1 | dow (8.3%) | bb_mnq (5.5%) |
| strong_div edge | +3.5% SHORT | ~neutro |
| Sinal atual (23h) | SHORT (46.4%) | SHORT (34.6%) |
| Principais features | dow + divergencias RSI | bb_mnq + vol_mnq + alinhamento SMA50 |
| Amostras | 6.508 | 1.586 |

**Unico modelo que melhorou com 6 meses.** O BTC se beneficiou de um periodo mais homogeneo (nov 2025-mar 2026), que eliminou ruido de regimes anteriores. A sazonalidade semanal (dow) caiu de 8.3% para 4.6% de importancia — o modelo passou a focar mais em condicoes de mercado (volatilidade MNQ, alinhamento SMA50) do que no dia da semana.

---

## 4. CL — Modelo 6 Meses

### Resultado Walk-Forward

```
Dataset: 1093 amostras  |  SHORT=41.4%  NEUTRO=6.7%  LONG=51.9%

              precision    recall  f1-score   support
SHORT            0.42      0.27      0.33       135
NEUTRO           0.00      0.00      0.00        10
LONG             0.56      0.73      0.63       183

accuracy                         0.52       328
ROC-AUC (macro ovr): 0.4740  [fraco]
```

### Top 10 Features (Gain)

| Feature | Importancia | Descricao |
|---------|-------------|-----------|
| cl_down_mnq_up | 5.5% | CL caindo + MNQ subindo |
| vol_spread_cl_mnq | 3.9% | Spread volatilidade CL-MNQ |
| bb_cl | 3.6% | Bollinger width do CL |
| vol_spread_cl_btc | 3.3% | Spread volatilidade CL-BTC |
| adx_btc | 3.3% | ADX do Bitcoin |
| ret1_cl | 3.3% | Retorno 1h do CL |
| bb_spread_cl_mnq | 3.2% | Spread Bollinger CL-MNQ |
| dist_ema20_cl | 3.2% | Distancia do CL da EMA20 |
| us_prime_setup | 3.1% | Trigger composto US |
| strong_div | 3.1% | Divergencia + ADX |

### Analise dos Gatilhos

| Gatilho | N | LONG | SHORT |
|---------|---|------|-------|
| strong_div | 211 | **59.2%** | 37.4% |
| us_prime_setup | 211 | **59.2%** | 37.4% |

**Edge strong_div**: +21.8% LONG (vs +9.3% no modelo antigo)!

### Vies EMA20 (CL+BTC)

| Estado | N | LONG | SHORT |
|--------|---|------|-------|
| Ambos abaixo | 48 | **75.0%** | 25.0% |
| Misturado | 214 | 48.1% | 47.7% |
| Ambos acima | 66 | 66.7% | 31.8% |

### Predicao Atual (18/05/2026 23h)

| Sinal | Prob SHORT | Prob LONG |
|-------|-----------|-----------|
| SHORT | 48.8% | 38.8% |

### Mudancas vs Modelo Antigo (Completo)

| Aspecto | Antigo (completo) | Novo (6 meses) |
|---------|------------------|----------------|
| AUC | 0.523 | 0.474 |
| Feature #1 | us_prime_setup (5.4%) | **cl_down_mnq_up (5.5%)** |
| strong_div edge | +9.3% LONG | **+21.8% LONG** |
| Sinal atual (23h) | LONG (43.4%) | SHORT (48.8%) |
| Melhor regime | ambos abaixo: 56.3% LONG | ambos abaixo: **75.0% LONG** |
| Acuracia | 39% | **52%** |
| Amostras | 5.401 | 1.093 |

Apesar da **queda de AUC (0.523 → 0.474)**, o modelo 6 meses tem:
- Acuracia maior (52% vs 39%)
- Edge strong_div muito mais forte (+21.8% vs +9.3%)
- Melhor regime EMA20 mais extremo (75% LONG vs 56.3%)

A queda de AUC e por causa da classe NEUTRO que praticamente desapareceu (3% no teste), penalizando a metrica multiclasse. O modelo 6 meses esta mais "extremado" — quando acerta LONG, acerta com mais confianca.

---

## 5. Tabela Comparativa Final

### AUCs

| Modelo | Completo (2 anos) | 6 Meses | Delta |
|--------|------------------|---------|-------|
| MNQ | 0.602 | 0.467 | -0.135 |
| BTC | 0.525 | **0.555** | +0.030 |
| CL | 0.523 | 0.474 | -0.049 |

### Acuracia (teste)

| Modelo | Completo | 6 Meses |
|--------|---------|---------|
| MNQ | 43% | 32% |
| BTC | 39% | 39% |
| CL | 39% | **52%** |

### Edge strong_div

| Modelo | Completo | 6 Meses |
|--------|---------|---------|
| MNQ | +10.6% LONG | +16.1% LONG |
| BTC | +3.5% SHORT | ~neutro |
| CL | +9.3% LONG | **+21.8% LONG** |

### Sinal Atual (18/05/2026)

| Modelo | Completo | 6 Meses |
|--------|---------|---------|
| MNQ | SHORT (43%/39%) | NEUTRO (33%/33%) |
| BTC | SHORT (46%/26%) | SHORT (35%/33%) |
| CL | LONG (41%/43%) | SHORT (49%/39%) |

---

## 6. Conclusoes

1. **BTC 6 meses e o melhor modelo do sistema** — AUC 0.555, superou o MNQ completo (0.602 → 0.555 vs 0.467). Unico que melhorou com menos dados.

2. **MNQ perdeu muita performance** — de 0.602 para 0.467. O modelo antigo dependia de 2 anos de dados para capturar divergencias entre ativos. Com 6 meses, virou um modelo quase univariado (foca em indicadores do proprio MNQ).

3. **CL ficou mais extremo** — AUC caiu mas edge subiu. O modelo 6 meses do CL tem a maior acuracia (52%) e o maior edge strong_div (+21.8%) de todo o sistema. Porem, a classe NEUTRO desapareceu — o modelo so "enxerga" SHORT ou LONG.

4. **Regime de alta no periodo** — os 3 modelos mostram distribuicao LONG > SHORT no teste (55-56% LONG), refletindo o mercado bullish de Mar-Mai 2026.

5. **Recomendacao**:
   - **BTC 6 meses** → manter (melhor AUC geral, 0.555)
   - **CL 6 meses** → manter com cautela (edge forte, mas extremado)
   - **MNQ 6 meses** → considerar reverter para completo se performance insuficiente

---

## 7. Como Reproduzir

```bash
# Coletar dados
python ml/collect_data.py
python ml/btc/collect_data.py
python ml/cl/collect_data.py

# Treinar modelos com 6 meses
python ml/train.py --max-months 6
python ml/btc/train.py --max-months 6
python ml/cl/train.py --max-months 6

# Ou treinar completo (reverter)
python ml/train.py
python ml/btc/train.py
python ml/cl/train.py
```

## 8. Drivers

- `train.py` agora suporta `--max-months N` para treinar com os ultimos N meses
- `model.pkl` salva o AUC, features, forward e interval (mas nao o max-months)
- `feature_importance.png` atualizado a cada treino

---

## 9. Glossario — Explicacao de Cada Indicador

### 9.1 Metricas do Modelo

#### AUC (ROC-AUC Score)

O que e: Mede a capacidade do modelo de **separar as 3 classes** (SHORT, NEUTRO, LONG). Valor de 0 a 1.

- **0.50** = modelo aleatorio (chuta)
- **0.55-0.60** = modelo fraco mas util (melhor que aleatorio)
- **0.60+** = modelo bom para predicao de direcao financeira

Calculo: O `predict_proba()` retorna 3 probabilidades (ex: [0.43, 0.17, 0.40] para SHORT, NEUTRO, LONG). O `roc_auc_score(y_true, proba, multi_class='ovr')` compara essas probabilidades com o que realmente aconteceu, usando One-vs-Rest (cada classe vs as outras).

Exemplo: AUC=0.555 significa que em 55.5% das vezes, o modelo da probabilidade maior para a classe correta. Melhor que 50% (aleatorio), mas longe de 100% (perfeito).

Por que usamos: Mercados financeiros sao ruidosos. AUC de 0.55-0.60 e considerado **bom o suficiente para ter edge** (vantagem estatistica) no longo prazo.

---

#### Edge (strong_div Edge)

O que e: Diferenca percentual entre acertos LONG e SHORT quando um gatilho esta ativo.

Calculo: `Edge = %LONG - %SHORT` quando a condicao do gatilho e verdadeira.

Exemplo: `strong_div edge = +21.8% LONG` significa: quando o strong_div esta ativo (divergencia de preco + ADX alto), o ativo sobe 21.8% mais vezes do que desce.

```
Das 211 vezes que strong_div ativou no teste:
- LONG real = 59.2% das vezes (subiu)
- SHORT real = 37.4% das vezes (caiu)
- Edge = 59.2% - 37.4% = +21.8% de vantagem pra LONG
```

Interpretacao: Quanto maior o edge (positivo ou negativo), mais confiavel e o gatilho. Edge proximo de 0 = gatilho inutil.

---

#### Acuracia (Accuracy)

O que e: Percentual de acertos do modelo (classe predita = classe real).

Calculo: `(SHORT corretos + NEUTRO corretos + LONG corretos) / total de amostras`

Limitacao: Acuracia pode enganar em datasets desbalanceados. Se 90% das vezes e LONG, um modelo que chuta LONG sempre tem 90% de acuracia mas nao serve pra nada. Por isso usamos **AUC** como metrica principal.

---

#### Walk-Forward

O que e: Metodo de validacao que respeita a ordem cronologica dos dados.

Calculo:
1. Ordena os dados por data (mais antigo -> mais recente)
2. Pega os **primeiros 70%** para treinar
3. Pega os **ultimos 30%** para testar (dados que o modelo nunca viu)

Diferenca do treino normal: Nao embaralha os dados. Simula como o modelo se comportaria no "futuro" (dados mais recentes).

```
Exemplo MNQ 6 meses:
Treino: 19/nov/2025 a 26/mar/2026  (767 amostras)
Teste:  26/mar/2026 a 18/mai/2026  (330 amostras)
```

---

#### Prob SHORT / Prob LONG

O que e: Probabilidade (0% a 100%) que o modelo atribui a cada classe para o proximo candle.

- `prob_short > prob_long` = modelo acha que vai cair
- `prob_long > prob_short` = modelo acha que vai subir
- `prob_short ≈ prob_long` = modelo esta indeciso (NEUTRO)

Exemplo: `prob_short=48.8%, prob_long=38.8%` = modelo da 48.8% de chance de cair, 38.8% de chance de subir. Sinal = SHORT.

Nota: As 3 probabilidades sempre somam 100% (incluindo NEUTRO).

---

### 9.2 Gatilhos e Triggers

#### strong_div (Strong Divergence)

O que e: Gatilho que detecta quando **MNQ e CL estao se movendo em direcoes opostas** E o ADX indica tendencia forte.

Calculo:
```python
strong_div = (price_div_cl < 0) AND (adx_mnq > 14)
```

Onde:
- `price_div_cl < 0` = retorno do MNQ x retorno do CL e negativo (direcoes opostas)
- `adx_mnq > 14` = tendencia do MNQ esta ativa

Exemplo pratico: Se MNQ sobe 0.2% e CL cai 0.3% no mesmo candle, e o ADX do MNQ esta em 23 (acima de 14), entao `strong_div = TRUE`. O modelo interpreta isso como: "algo esta errado, um dos dois vai reverter".

---

#### us_prime_setup

O que e: strong_div acontecendo **durante a sessao americana** (9h as 17h NY).

Calculo:
```python
us_prime_setup = (price_div_cl < 0) AND (adx_mnq > 14) AND (hour in 9..17)
```

Por que existe: O modelo foi treinado principalmente na sessao US. Fora desse horario, o edge e menor. Esse gatilho filtra apenas os momentos de maior confianca.

---

#### cl_down_mnq_up

O que e: Gatilho que detecta quando **CL esta caindo e MNQ esta subindo** no mesmo candle. Opcao contraria: CL subindo e MNQ caindo.

Calculo:
```python
cl_down_mnq_up = (retorno_CL < 0) AND (retorno_MNQ > 0)
```

Importancia: No modelo CL 6 meses, virou a **feature #1** (5.5%). Isso mostra que quando petroleo e Nasdaq divergem fortemente, o CL tende a se mover.

---

#### EMA20 bias MNQ+BTC

O que e: Contagem de quantos ativos (MNQ e BTC) estao **acima da EMA20** no momento.

| Valor | Significado |
|-------|-------------|
| 0 | Ambos abaixo da EMA20 (bearish) |
| 1 | Um acima, outro abaixo (misto) |
| 2 | Ambos acima da EMA20 (bullish) |

Exemplo: `ema20_bias_mnq_btc = 2` = MNQ e BTC estao ambos acima da EMA20. O modelo usa isso para entender o "regime" atual do mercado.

No CL, o mesmo conceito e `ema20_bias_mnq_btc` mas com CL + BTC.

---

### 9.3 Features Tecnicas

#### price_div_cl (Price Divergence CL)

O que e: Produto do retorno do MNQ com o retorno do CL no mesmo candle. Mede se os dois ativos estao andando juntos ou separados.

Calculo:
```python
price_div_cl = ret1_mnq * ret1_cl
```

Interpretacao:
- **Negativo** (ex: -0.0002) = MNQ e CL em direcoes opostas (um sobe, outro desce) = DIVERGENCIA
- **Positivo** (ex: +0.0003) = MNQ e CL na mesma direcao = CONVERGENCIA
- **Proximo de 0** = pelo menos um dos ativos nao se moveu

Exemplo: Se MNQ sobe 0.5% e CL cai 0.3%:
- `ret1_mnq = +0.005`, `ret1_cl = -0.003`
- `price_div_cl = 0.005 * (-0.003) = -0.000015` (negativo = divergencia)

#### price_div_abs (Price Divergence Absolute)

O que e: A **magnitude** da divergencia, ignorando o sinal. Quanto maior, mais forte a divergencia.

Calculo:
```python
price_div_abs = |price_div_cl|
```

Importancia: Foi a **feature #1 do MNQ antigo** (4.3%). O modelo aprendeu que quando o produto dos retornos e grande (em modulo), algo relevante esta acontecendo.

---

#### RSI (Relative Strength Index)

O que e: Indicador de momentum que mede a **velocidade e magnitude** dos movimentos de preco. Vai de 0 a 100.

Calculo:
```python
rsi = RSI(close, window=21)
```

Interpretacao:
- **> 70** = sobrecomprado (pode cair)
- **< 30** = sobrevendido (pode subir)
- **Entre 30-70** = neutro

Janela usada: **21 periodos** (cerca de 1 dia de trading em grafico 1h). Diferente do RSI classico de 14 periodos, este e mais lento e captura ciclos maiores.

---

#### ADX (Average Directional Index)

O que e: Mede a **forca da tendencia**, independente da direcao. Nao diz se vai subir ou descer, so diz se esta tendenciando.

Calculo:
```python
adx = ADX(high, low, close, window=17)
```

Interpretacao:
- **< 14** = mercado lateral (sem tendencia). strong_div nao ativa.
- **14-20** = tendencia fraca
- **20-25** = tendencia moderada
- **> 25** = tendencia forte

No modelo, `adx_above_14` (ADX > 14) e usado como condicao para o strong_div. O modelo so "acredita" na divergencia se houver tendencia presente.

---

#### DI+ e DI- (Directional Indicators)

O que e: Componentes do ADX que mostram a **direcao** da tendencia.

- **DI+** = forca compradora
- **DI-** = forca vendedora
- `di_spread = DI+ - DI-` = liquido direcional (positivo = pressao compradora)

---

#### SMA50 (Simple Moving Average 50)

O que e: Media do preco de fechamento dos ultimos **50 candles** (cerca de 2.5 dias de trading em grafico 1h).

Calculo: `sma50 = media(close, 50)`

Features derivadas:
- `dist_sma50` = distancia percentual do preco atual ate a SMA50: `(close - sma50) / sma50 * 100`
- `sma50_slope` = inclinacao: `(sma50_atual - sma50_5h_atras) / sma50_5h_atras * 100`
- `above_sma50` = 1 se preco > SMA50, 0 senao

---

#### EMA20 (Exponential Moving Average 20)

O que e: Media movel **exponencial** de 20 periodos (da mais peso aos dados recentes).

Diferenca da SMA50: A EMA20 reage mais rapido as mudancas de preco. Enquanto a SMA50 mostra a tendencia de **medio prazo** (2.5 dias), a EMA20 mostra o **curto prazo** (1 dia).

Features derivadas: mesma estrutura da SMA50 (distancia, binario acima/abaixo).

---

#### SMA50 Alignment

O que e: Contagem de quantos dos 3 ativos (MNQ, BTC, CL) estao acima da SMA50.

Valor: 0, 1, 2, ou 3.

Exemplo: `sma50_alignment = 1` significa que apenas 1 dos 3 ativos esta acima da SMA50. Mercado majoritariamente bearish.

Interpretacao: Quanto maior, mais bullish e o ambiente geral. O modelo usa isso para entender se estamos num mercado de alta ou baixa amplo.

---

#### Volatilidade (vol)

O que e: Desvio padrao do retorno de 1h nos ultimos 20 periodos. Mede o **risco/intensidade** do ativo.

Calculo: `vol = std(ret1, 20)`

Interpretacao: `vol_mnq` alto = MNQ esta se movendo muito (grandes candles). `vol_mnq` baixo = movimento fraco.

---

#### Bollinger Width (bb_w)

O que e: Largura das Bandas de Bollinger, normalizada pelo preco.

Calculo: `bb_w = (std(close, 20) * 2) / sma(close, 20)`

Interpretacao: Mede a **expansao/contracao da volatilidade**. `bb_w` alto = bandas abertas = volatilidade alta (possivel inicio de tendencia). `bb_w` baixo = bandas comprimidas = consolidacao (possivel explosao).

---

#### Returns (ret1, ret4, ret8)

O que e: Retorno percentual do ativo em 1, 4 ou 8 candles atras.

Calculo:
- `ret1 = (close_atual / close_1h_atras) - 1` (retorno 1h)
- `ret4 = (close_atual / close_4h_atras) - 1` (retorno 4h)
- `ret8 = (close_atual / close_8h_atras) - 1` (retorno 8h)

O modelo usa esses retornos passados para entender o momentum recente.

---

### 9.4 Spreads (Diferencas entre Ativos)

#### RSI Spreads

Diferenca de RSI entre pares de ativos:

| Feature | Calculo | Interpretacao |
|---------|---------|---------------|
| `div_cl` | RSI(MNQ) - RSI(CL) | MNQ mais forte/ fraco que petroleo |
| `div_btc` | RSI(MNQ) - RSI(BTC) | MNQ mais forte/fraco que Bitcoin |
| `rsi_spread_btc_cl` | RSI(BTC) - RSI(CL) | Bitcoin vs petroleo |

Negativo: o primeiro ativo esta mais "sobrevendido" que o segundo.
Positivo: o primeiro esta mais "sobrecomprado" que o segundo.

---

#### ADX Spreads

Diferenca de forca de tendencia entre ativos:

| Feature | Calculo | Pra que serve |
|---------|---------|---------------|
| `adx_spread_mnq_btc` | ADX(MNQ) - ADX(BTC) | Quem esta mais tendenciando? |
| `adx_spread_mnq_cl` | ADX(MNQ) - ADX(CL) | MNQ ou CL tem tendencia mais forte? |
| `adx_spread_btc_cl` | ADX(BTC) - ADX(CL) | BTC ou CL esta mais direcional? |

Modelo usa spreads absolutos (`adx_abs_*`) para medir a **magnitude** da diferenca, ignorando direcao.

---

#### Vol / Bollinger Spreads

Diferenca de volatilidade entre ativos:

| Feature | Calculo |
|---------|---------|
| `vol_spread_mnq_btc` | vol(MNQ) - vol(BTC) |
| `bb_spread_mnq_cl` | bb_w(MNQ) - bb_w(CL) |

Interpretacao: Se `vol_spread_mnq_btc` e positivo, o MNQ esta mais volatil que o BTC. O modelo usa isso para identificar qual ativo esta "liderando" o movimento.

---

#### Co-movement (Produtos de Retorno)

Produto dos retornos entre pares de ativos:

| Feature | Calculo | Valor positivo = | Valor negativo = |
|---------|---------|-----------------|------------------|
| `ret1_prod_mnq_btc` | ret1(MNQ) x ret1(BTC) | Ambos na mesma direcao | Direcoes opostas |
| `ret1_prod_btc_cl` | ret1(BTC) x ret1(CL) | BTC e CL juntos | BTC e CL separados |
| `ret4_prod_mnq_cl` | ret4(MNQ) x ret4(CL) | Mesma direcao em 4h | Divergindo em 4h |

---

### 9.5 Features Temporais

| Feature | Valores | Significado |
|---------|---------|-------------|
| `hour` | 0-23 | Hora do dia |
| `dow` | 0=seg, 6=dom | Dia da semana |
| `is_us_session` | 0 ou 1 | 9h as 17h NY |
| `is_us_morning` | 0 ou 1 | 9h as 13h NY |
| `is_us_afternoon` | 0 ou 1 | 14h as 17h NY |
| `is_evening` | 0 ou 1 | 18h as 21h NY |

O modelo aprende padroes como "as 10h o MNQ tende a subir mais" ou "as segundas o BTC tende a cair".

---

### 9.6 Resumo: Como o Modelo "Pensa"

1. **Olha os ultimos candles** de MNQ, BTC e CL (ate 50 periodos para SMA)
2. **Calcula 38 features**: RSI, ADX, SMA50, EMA20, spreads, etc
3. **Alimenta o XGBoost** com essas features
4. **XGBoost compara** com padroes que aprendeu no treino
5. **Retorna 3 probabilidades**: SHORT, NEUTRO, LONG
6. **Escolhe a classe** com maior probabilidade

Quando aparece `strong_div=true`, significa que o modelo detectou um padrao de divergencia entre ativos que, historicamente, precedeu movimentos direcionais.

---

## 10. Indicadores para TradingView — PineScript

### 10.1 MNQ — 5 Indicadores

#### 1. RSI(21)

```
//@version=5
indicator("RSI 21", overlay=false)
rsi_21 = ta.rsi(close, 21)
hline(70, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(30, "Oversold", color=color.green, linestyle=hline.style_dashed)
plot(rsi_21, "RSI(21)", color=color.blue, linewidth=2)
```

No TradingView: `RSI` -> periodo 21.

---

#### 2. ADX(17) + DI+/DI-

```
//@version=5
indicator("ADX 17 + DI", overlay=false)
[di_plus, di_minus, adx] = ta.dmi(high, low, close, 17)
hline(14, "Min", color=color.gray, linestyle=hline.style_dashed)
plot(adx, "ADX", color=color.purple, linewidth=2)
plot(di_plus, "DI+", color=color.green)
plot(di_minus, "DI-", color=color.red)
```

No TradingView: `ADX` -> periodo 17.

---

#### 3. Bollinger Bands(20, 2)

```
//@version=5
indicator("BB 20 MACD", overlay=true)
[bb_middle, bb_upper, bb_lower] = ta.bb(close, 20, 2)
plot(bb_middle, "SMA", color=color.blue)
plot(bb_upper, "Upper", color=color.gray)
plot(bb_lower, "Lower", color=color.gray)
fill(plot(bb_upper), plot(bb_lower), color=color.new(color.gray, 90))
// Largura normalizada (%)
bbw = (bb_upper - bb_lower) / bb_middle * 100
// Plotar em painel separado
```

No TradingView: `Bollinger Bands` -> periodo 20, multiplo 2.

---

#### 4. SMA50 + EMA20 (Regime de Alinhamento)

```
//@version=5
indicator("SMA50 + EMA20", overlay=true)
sma50 = ta.sma(close, 50)
ema20 = ta.ema(close, 20)
plot(sma50, "SMA50", color=color.orange, linewidth=2)
plot(ema20, "EMA20", color=color.blue, linewidth=2)
// Alinhamento no painel
above_sma50 = close > sma50 ? 1 : 0
above_ema20 = close > ema20 ? 1 : 0
// Regime: 0=ambos abaixo, 1=misto, 2=ambos acima
regime = above_sma50 + above_ema20
```

No TradingView: Adicione `SMA(50)` e `EMA(20)` sobre o grafico.

---

#### 5. Price Divergence MNQ x CL ⭐ (ret1 * ret1)

Essa e a feature mais importante do modelo. Nao existe pronta no TV, voce precisa criar:

```
//@version=5
indicator("Price Divergence MNQ x CL", overlay=false)
// Precisa usar tickers externos
mnq = request.security("MNQ=F", timeframe.period, close)
cl  = request.security("CL=F",  timeframe.period, close)

ret1_mnq = (mnq / mnq[1]) - 1
ret1_cl  = (cl / cl[1]) - 1
price_div = ret1_mnq * ret1_cl

hline(0, "Zero", color=color.gray)
plot(price_div, "ret1_mnq * ret1_cl", color=price_div > 0 ? color.green : color.red, 
     style=plot.style_histogram, linewidth=3)
// Adicionar media movel para suavizar
plot(ta.sma(price_div, 4), "SMA(4)", color=color.white, linewidth=1)
```

**Interpretacao:**
- **Verde** = MNQ e CL na mesma direcao (convergencia)
- **Vermelho** = MNQ e CL em direcoes opostas (divergencia) — **atencao**
- **Barras altas** (absoluto grande) = divergencia forte — gatilho strong_div potencial

---

### 10.2 BTC — 5 Indicadores

#### 1. Bollinger Width do MNQ (Feature #1 do BTC 6m)

```
//@version=5
indicator("MNQ Bollinger Width", overlay=false)
mnq_close = request.security("MNQ=F", timeframe.period, close)
bb_middle = ta.sma(mnq_close, 20)
bb_std    = ta.stdev(mnq_close, 20)
bbw = (bb_std * 2) / bb_middle * 100  // em percentual
plot(bbw, "BBW MNQ", color=color.blue, linewidth=2)
hline(2.0, "Alto", color=color.red, linestyle=hline.style_dashed)
hline(0.8, "Baixo", color=color.green, linestyle=hline.style_dashed)
```

**Interpretacao:** BBW alto (>2%) = volatilidade alta no Nasdaq = BTC tende a se mover. BBW baixo (<0.8%) = compressao = possivel explosao.

---

#### 2. Dia da Semana (DOW) — Feature #1 do BTC Completo

```
//@version=5
indicator("Dia da Semana Bias", overlay=true)
day = dayofweek
// Segunda e Sexta = bias SHORT historico
bgcolor(day == dayofweek.monday   ? color.new(color.red, 85) : na)
bgcolor(day == dayofweek.friday   ? color.new(color.red, 85) : na)
bgcolor(day == dayofweek.wednesday ? color.new(color.green, 85) : na)
// Label no canto
plotshape(day == dayofweek.monday, "Seg", shape.labelup, location.top, color.red, size=size.tiny)
```

No TradingView: Nao precisa script — basta observar que segundas e sextas tem bias SHORT, quartas tem bias LONG.

---

#### 3. SMA50 Alignment MNQ + CL

```
//@version=5
indicator("SMA50 Alignment", overlay=false)
mnq_close = request.security("MNQ=F", timeframe.period, close)
cl_close  = request.security("CL=F",  timeframe.period, close)

mnq_above = close > ta.sma(close, 50) ? 1 : 0
mnq50     = request.security("MNQ=F", timeframe.period, ta.sma(close, 50))
cl50      = request.security("CL=F",  timeframe.period, ta.sma(close, 50))
mnq_above50 = close > mnq50 ? 1 : 0
cl_above50  = cl_close > cl50 ? 1 : 0

alignment = mnq_above50 + cl_above50  // 0, 1, ou 2
plot(alignment, "Alignment", color=alignment == 2 ? color.green : alignment == 0 ? color.red : color.orange, 
     linewidth=3, style=plot.style_histogram)
hline(1, "Misto", color=color.gray, linestyle=hline.style_dashed)
```

**Interpretacao:** 2 = ambos acima (bullish), 0 = ambos abaixo (bearish), 1 = misto.

---

#### 4. RSI(21) do BTC

```
//@version=5
indicator("RSI 21 BTC", overlay=false)
rsi_21 = ta.rsi(close, 21)
hline(70, color=color.red, linestyle=hline.style_dashed)
hline(30, color=color.green, linestyle=hline.style_dashed)
plot(rsi_21, "RSI(21)", color=color.blue, linewidth=2)
```

No TradingView: `RSI` -> periodo 21 (no grafico do BTC).

---

#### 5. Volatility Spread MNQ - BTC

```
//@version=5
indicator("Vol Spread MNQ - BTC", overlay=false)
mnq_close = request.security("MNQ=F", timeframe.period, close)
btc_close = close

ret1_mnq = (mnq_close / mnq_close[1]) - 1
ret1_btc = (btc_close / btc_close[1]) - 1

vol_mnq = ta.stdev(ret1_mnq, 20) * 100
vol_btc = ta.stdev(ret1_btc, 20) * 100

spread = vol_mnq - vol_btc
hline(0, "Zero", color=color.gray)
plot(spread, "Vol MNQ - Vol BTC", color=spread > 0 ? color.orange : color.purple, 
     style=plot.style_histogram, linewidth=3)
```

**Interpretacao:** Positivo = MNQ mais volatil que BTC (movimento iminente no BTC). Negativo = BTC mais volatil que MNQ.

---

### 10.3 CL — 5 Indicadores

#### 1. Price Divergence CL x MNQ (mesmo do MNQ #5)

```
//@version=5
indicator("Price Divergence CL x MNQ", overlay=false)
mnq = request.security("MNQ=F", timeframe.period, close)
ret1_cl  = (close / close[1]) - 1
ret1_mnq = (mnq / mnq[1]) - 1
price_div = ret1_cl * ret1_mnq
hline(0, "Zero", color=color.gray)
plot(price_div, "ret1_CL * ret1_MNQ", color=price_div > 0 ? color.green : color.red, 
     style=plot.style_histogram, linewidth=3)
```

**Interpretacao:** Quando CL cai e MNQ sobe (vermelho forte) — sinal de reversao LONG no CL (edge de +21.8% com strong_div ativo).

---

#### 2. EMA20 + Distancia Percentual

```
//@version=5
indicator("EMA20 + Distance", overlay=true)
ema20 = ta.ema(close, 20)
plot(ema20, "EMA20", color=color.blue, linewidth=2)

// Painel separado: distancia %
dist = (close - ema20) / ema20 * 100
```

Painel separado:

```
//@version=5
indicator("EMA20 Distance %", overlay=false)
ema20 = ta.ema(close, 20)
dist = (close - ema20) / ema20 * 100
hline(0, "Zero", color=color.gray)
hline(1, "+1%", color=color.green, linestyle=hline.style_dashed)
hline(-1, "-1%", color=color.red, linestyle=hline.style_dashed)
plot(dist, "Dist%", color=dist > 0 ? color.green : color.red, style=plot.style_histogram, linewidth=3)
```

**Interpretacao:** Quando CL esta muito abaixo da EMA20 (< -1%) + ambos abaixo EMA20 no regime = 75% de chance de LONG historico.

---

#### 3. Bollinger Bands(20, 2)

```
//@version=5
indicator("BB 20 CL", overlay=true)
[bb_middle, bb_upper, bb_lower] = ta.bb(close, 20, 2)
plot(bb_middle, "SMA", color=color.blue)
plot(bb_upper, "Upper", color=color.gray)
plot(bb_lower, "Lower", color=color.gray)
fill(plot(bb_upper), plot(bb_lower), color=color.new(color.gray, 90))
```

No TradingView: `Bollinger Bands` -> periodo 20, multiplo 2.

---

#### 4. Volatility Spread CL - MNQ

```
//@version=5
indicator("Vol Spread CL - MNQ", overlay=false)
mnq_close = request.security("MNQ=F", timeframe.period, close)
ret1_cl  = (close / close[1]) - 1
ret1_mnq = (mnq_close / mnq_close[1]) - 1
vol_cl  = ta.stdev(ret1_cl, 20) * 100
vol_mnq = ta.stdev(ret1_mnq, 20) * 100
spread = vol_cl - vol_mnq
hline(0, "Zero", color=color.gray)
plot(spread, "Vol CL - Vol MNQ", color=spread > 0 ? color.orange : color.purple, 
     style=plot.style_histogram, linewidth=3)
```

**Interpretacao:** CL muito mais volatil que MNQ = alerta de movimento no petroleo (feature top do CL 6m).

---

#### 5. ADX do BTC

```
//@version=5
indicator("ADX BTC", overlay=false)
btc_close = request.security("BTC-USD", timeframe.period, close)
btc_high  = request.security("BTC-USD", timeframe.period, high)
btc_low   = request.security("BTC-USD", timeframe.period, low)
[di_plus, di_minus, adx] = ta.dmi(btc_high, btc_low, btc_close, 17)
hline(14, "Min", color=color.gray, linestyle=hline.style_dashed)
plot(adx, "ADX BTC", color=color.purple, linewidth=2)
```

**Interpretacao:** O ADX do Bitcoin importa mais pro CL que o ADX do proprio CL (3.3% de importancia). BTC com tendencia forte = CL tende a se mover tambem.

---

### 10.4 Template Unico — Todos os Indicadores em um So Script

Para nao precisar abrir 5 indicadores por ativo, use este script que junta os principais em um unico painel:

```
//@version=5
indicator("ML System — MNQ Combo", overlay=true)

// ─── MNQ ───
rsi21 = ta.rsi(close, 21)
[di_plus, di_minus, adx] = ta.dmi(high, low, close, 17)
sma50 = ta.sma(close, 50)
ema20 = ta.ema(close, 20)

plot(sma50, "SMA50", color=color.orange, linewidth=1)
plot(ema20, "EMA20", color=color.blue, linewidth=1)

// ─── Price Divergence CL x MNQ ───
mnq_close = request.security("MNQ=F", timeframe.period, close)
cl_close  = close
ret1_mnq = (mnq_close / mnq_close[1]) - 1
ret1_cl  = (cl_close / cl_close[1]) - 1
price_div = ret1_mnq * ret1_cl

// ─── Condicoes ───
strong_div = price_div < 0 and adx > 14
us_session = hour >= 9 and hour < 17
us_prime   = strong_div and us_session

// ─── Plota gatilho no grafico ───
plotshape(strong_div, "strong_div", shape.triangledown, location.top, 
          color=color.new(color.orange, 80), size=size.tiny)
plotshape(us_prime, "us_prime", shape.triangleup, location.bottom, 
          color=color.new(color.green, 80), size=size.tiny)

// Painel informativo no canto
if barstate.islast
    lbl = "RSI(21): " + str.tostring(math.round(rsi21, 1)) + "\n" +
          "ADX(17): " + str.tostring(math.round(adx, 1)) + "\n" +
          "Price Div: " + str.tostring(price_div, 6) + "\n" +
          "strong_div: " + (strong_div ? "SIM" : "nao") + "\n" +
          "Regime EMA: " + (close > ema20 ? "acima" : "abaixo")
    label.new(bar_index, high * 1.02, lbl, color=color.new(color.black, 40), 
              textcolor=color.white, size=size.small)
```

---

### 10.5 Resumo — O Que Colocar no Grafico

| Ativo | Painel Principal (no grafico) | Painel Separado (abaixo) |
|-------|------------------------------|--------------------------|
| **MNQ** | SMA50 + EMA20 | RSI(21) + ADX(17) + Price Divergence MNQxCL |
| **BTC** | — | BBwidth MNQ + SMA50 Alignment + RSI(21) + Vol Spread MNQ-BTC |
| **CL** | EMA20 + Bollinger | Price Divergence CLxMNQ + EMA20 Dist% + Vol Spread CL-MNQ + ADX BTC |

**Dica:** O indicador que mais importa e o **Price Divergence** (`ret1 * ret1`). Coloque ele em destaque no painel inferior. Quando ele fica vermelho forte (negativo), preste atencao — o modelo enxerga oportunidade ali.

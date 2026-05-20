# ML Prop Firm — Estratégia de Trading ao Vivo

> Sistema de sinais ML multi-ativo para contas prop firm (FTMO, MFF, etc.)
> Atualizado em: 20/05/2026

---

## 1. Resumo Executivo

Estratégia de trading sistemática usando **XGBoost multiclasse** para prever direção do preço nas próximas 8 horas em 4 ativos futuros: **MNQ, BTC, CL, MGC**.

| Métrica | Valor |
|---------|-------|
| Capital | $50.000 (prop firm) |
| Risco por trade | 0,5% ($250) |
| Trades/semana | ~5 (3-4 ativos) |
| Taxa de acerto | 68,7% |
| Drawdown máx. | -2,6% |
| Expectativa média | +0,52% por trade |
| Período do backtest | 4,2 anos |

---

## 2. Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      Yahoo Finance                          │
│              (dados OHLCV 1h / 60d historico)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    predict_live.py                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ MODELO   │  │ MODELO   │  │ MODELO   │  │ MODELO   │   │
│  │ MNQ      │  │ BTC      │  │ CL       │  │ MGC      │   │
│  │.pkl      │  │.pkl      │  │.pkl      │  │.pkl      │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│         │            │            │            │           │
│         ▼            ▼            ▼            ▼           │
│  67 features    67 features  67 features  67 features     │
│  (MNQ+BTC+CL)  (BTC+MNQ+CL) (CL+MNQ+BTC) (MGC+MNQ+BTC)  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      server.js                              │
│              GET /api/live2 → JSON                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    live2.html (Dashboard)                    │
│              Sinais, stop, target, risco, contratos         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Modelos ML

### 3.1 Algoritmo

**XGBoost Classifier** — multiclasse com 3 saídas:

| Classe | Índice | Significado |
|--------|--------|-------------|
| SHORT  | 0      | Preço vai cair nas próximas 8h |
| NEUTRO | 1      | Preço vai ficar estável |
| LONG   | 2      | Preço vai subir nas próximas 8h |

### 3.2 Hiperparâmetros

| Parâmetro | Valor |
|-----------|-------|
| Árvores (n_estimators) | 500 |
| Profundidade máxima | 4 |
| Learning rate | 0,03 |
| Subamostragem | 80% |
| Colunas por árvore | 75% |
| Min child weight | 8 |
| Early stopping | 40 rodadas |
| Objective | multi:softprob |

### 3.3 Treinamento

| Ativo | Amostras treino | Período treino | Amostras teste | Período teste | Acurácia | AUC |
|-------|:---------------:|:--------------:|:--------------:|:-------------:|:--------:|:---:|
| MNQ   | 9.559 | 2021-08 a 2025-08 | 4.097 | 2025-08 a 2026-05 | 37,0% | 0,56 |
| BTC   | 12.889 | 2021-07 a 2025-09 | 5.525 | 2025-09 a 2026-05 | 43,7% | 0,56 |
| CL    | 9.338 | 2021-08 a 2025-08 | 4.003 | 2025-08 a 2026-05 | 40,0% | 0,54 |
| MGC   | 9.528 | 2021-08 a 2025-08 | 4.084 | 2025-08 a 2026-05 | 42,7% | 0,54 |

> AUC ~0,55 indica poder preditivo marginal, mas o filtro de confiança (≥50%) + stop/target assimétrico gera EV positivo consistente.

---

## 4. Features — O que o ML Analisa

São **67 features** por ativo, divididas em 7 categorias:

### 4.1 RSI (9 features)

| Feature | Descrição |
|---------|-----------|
| `rsi_p` | RSI(21) do ativo principal |
| `rsi_1` | RSI(21) do ativo secundário 1 |
| `rsi_2` | RSI(21) do ativo secundário 2 |
| `div_1` | Diferença RSI principal - secundário 1 |
| `div_2` | Diferença RSI principal - secundário 2 |
| `rsi_spread_1_2` | Diferença RSI entre secundários |
| `rsi_abs_*` | Valor absoluto das diferenças |

### 4.2 ADX / Directional Indicators (11 features)

| Feature | Descrição |
|---------|-----------|
| `adx_p`, `adx_1`, `adx_2` | Força da tendência (0-100) |
| `pdi_p`, `mdi_p` | Positive/Negative Directional Index |
| `di_spread_p`, `di_spread_1`, `di_spread_2` | PDI - MDI (compradores vs vendedores) |
| `dadx_p` | Variação do ADX em 2 barras |

### 4.3 Retornos (16 features)

| Feature | Descrição |
|---------|-----------|
| `ret1_p`, `ret4_p`, `ret8_p` | Retorno 1, 4 e 8 barras do principal |
| `ret1_1`, `ret4_1` | Retorno do secundário 1 |
| `ret1_2`, `ret4_2` | Retorno do secundário 2 |
| `ret*_spread_*` | Diferenças de retorno entre ativos |
| `ret*_prod_*` | Produto dos retornos (correlação) |
| `price_div_p_2` | Produto retorno principal × secundário 2 |
| `price_div_abs` | Valor absoluto da divergência |

### 4.4 Volatilidade (6 features)

| Feature | Descrição |
|---------|-----------|
| `vol_p` | Desvio padrão do retorno (20 barras) |
| `bb_p` | Largura das Bandas de Bollinger (2 desvios / média) |
| `vol_spread_*` | Diferença de volatilidade entre ativos |
| `bb_spread_*` | Diferença de largura das BB entre ativos |

### 4.5 Médias Móveis (18 features)

| Feature | Descrição |
|---------|-----------|
| `dist_sma50_p`, `dist_sma50_1`, `dist_sma50_2` | Distância % do preço à SMA50 |
| `sma50_slope_p` | Inclinação da SMA50 (5 barras) |
| `above_sma50_p`, `above_sma50_1`, `above_sma50_2` | Preço > SMA50? (0/1) |
| `sma50_alignment` | Soma de above_sma50 dos 3 ativos (0-3) |
| `dist_ema20_p`, `dist_ema20_1`, `dist_ema20_2` | Distância % do preço à EMA20 |
| `above_ema20_p`, `above_ema20_1`, `above_ema20_2` | Preço > EMA20? (0/1) |
| `ema20_bias_p_1` | Viés EMA20: principal + secundário 1 |
| `ema20_alignment` | Soma de above_ema20 dos 3 ativos (0-3) |

### 4.6 Temporais (6 features)

| Feature | Descrição |
|---------|-----------|
| `hour` | Hora do dia (0-23) |
| `dow` | Dia da semana (0-6) |
| `hour_sin`, `hour_cos` | Hora codificada em seno/cosseno |
| `dow_sin`, `dow_cos` | Dia da semana codificado em seno/cosseno |

### 4.7 Key Levels (22 features)

| Feature | Descrição |
|---------|-----------|
| `dist_to_pdh` | Distância % ao topo do dia anterior |
| `dist_to_pdl` | Distância % ao fundo do dia anterior |
| `dist_to_do` | Distância % à abertura do dia anterior |
| `above_pdh`, `above_pdl`, `above_do` | Preço acima desses níveis? |
| `prev_day_range_pct` | Range % do dia anterior |
| `dist_to_pwh`, `dist_to_pwl`, `dist_to_wo` | Níveis da **semana** anterior |
| `above_pwh`, `above_pwl`, `above_wo` | Preço acima dos níveis semanais? |
| `dist_to_pmh`, `dist_to_pml`, `dist_to_mo` | Níveis do **mês** anterior |
| `above_pmh`, `above_pml`, `above_mo` | Preço acima dos níveis mensais? |
| `dist_to_mday_h`, `dist_to_mday_l` | Níveis da **segunda-feira** |
| `above_mday_h`, `above_mday_l` | Preço acima dos níveis de segunda? |

---

## 5. Setup por Ativo

| Ativo | Opera | ML ≥ | Stop | Target | Por quê? |
|-------|:----:|:----:|:----:|:------:|----------|
| **MNQ** | LONG e SHORT | 50% | 1,5R | 3,0R | Mercado mais eficiente, ambos os lados funcionam |
| **BTC** | SHORT only | 50% | 1,5R | 3,0R | Otimização mostrou que SHORT tem EV muito maior que LONG |
| **CL** | LONG only | 50% | 1,5R | 2,0R | Petróleo tem viés de alta no período, LONG só |
| **MGC** | LONG e SHORT | 50% | 1,5R | 2,0R | Ouro opera bem nos 2 lados com target menor |

Onde **R = ATR(14)** médio das últimas 14 horas.

---

## 6. Gestão de Risco

### 6.1 Por Trade

```
Risco = Capital × 0,5%
      = $50.000 × 0,005
      = $250 por trade
```

### 6.2 Cálculo de Contratos

```
distância_stop = stop_r × ATR / preço
contratos = round(risco_dolar / (distância_stop × preço))
```

Exemplo (BTC):
- Preço: $77.000, ATR: $325, stop_r: 1,5
- distância_stop = 1,5 × 325 / 77.000 = 0,63%
- contratos = 250 / (0,0063 × 77.000) = ~0,5 → 1 contrato

### 6.3 Regras de Operação

- Máximo 1 posição por ativo por vez
- Stop loss fixo (1,5R) + target fixo (2-3R)
- Posição encerrada automaticamente ao atingir stop ou target
- Sem reentrada na mesma barra

### 6.4 Drawdown Máximo Esperado

| Risco por trade | Drawdown máx. (backtest) | Trades/semana |
|:---------------:|:------------------------:|:-------------:|
| 0,5% | -2,6% | 4,9 |
| 1,0% | -19,0% | 5,0 |

> Para prop firm (limite 10% de DD), usar **0,5%** por trade é seguro.

---

## 7. Resultados do Backtest

### 7.1 Geral (0,5% risco)

| Métrica | Valor |
|---------|:-----:|
| Período | 2022-01 a 2026-05 |
| Total trades | 1.081 |
| Trades/semana | 4,9 |
| Taxa de acerto | 68,7% |
| PnL médio por trade | +0,52% |
| Drawdown máximo | -2,6% |
| Sharpe Ratio | ~1,8 |

### 7.2 Por Ativo

| Ativo | Trades | Acerto | PnL médio |
|-------|:-----:|:-----:|:---------:|
| MNQ | 74/ano | 60% | +1,20R |
| BTC | 143/ano | 69,6% (SHORT) | +1,63R |
| CL | 83/ano | 70,2% (LONG) | +0,96R |
| MGC | 221/ano | 61,1% | +0,64R |

---

## 8. Top Features por Ativo

### MNQ — Top 5

| Peso | Feature | O que significa |
|:----:|---------|----------------|
| 4,0% | `hour_cos` | Horário do dia é crucial |
| 3,1% | `above_pmh` | Preço acima do topo do mês passado |
| 3,0% | `vol_p` | Volatilidade atual |
| 3,0% | `above_mday_l` | Preço acima da mínima da segunda |
| 2,8% | `price_div_abs` | Divergência de preço entre ativos |

### BTC — Top 5

| Peso | Feature | O que significa |
|:----:|---------|----------------|
| 6,1% | `dow_sin` | **Dia da semana** domina as decisões |
| 2,7% | `vol_p` | Volatilidade atual |
| 2,5% | `dow` | Dia da semana (numérico) |
| 2,3% | `bb_p` | Largura das Bandas de Bollinger |
| 2,3% | `above_pdl` | Preço acima do fundo de ontem |

### CL — Top 5

| Peso | Feature | O que significa |
|:----:|---------|----------------|
| 4,6% | `hour` | **Horário define** o petróleo |
| 2,5% | `hour_sin` | Sazonalidade horária |
| 2,0% | `above_pwl` | Acima do fundo da semana |
| 2,0% | `above_do` | Acima da abertura de hoje |
| 2,0% | `above_pml` | Acima do fundo do mês |

### MGC — Top 5

| Peso | Feature | O que significa |
|:----:|---------|----------------|
| 2,6% | `vol_p` | Volatilidade |
| 2,3% | `above_pdh` | Acima do topo de ontem |
| 2,1% | `hour` | Horário |
| 2,0% | `above_mo` | Acima da abertura do mês |
| 2,0% | `prev_day_range_pct` | Range do dia anterior |

---

## 9. Exemplo Prático — Sinal BTC

**Cenário:** ML detecta probabilidade SHORT alta.

```
Preço BTC:    $77.000
ATR(14):      $325
Confiança ML: 55,3% SHORT (acima de 50%)

Cálculos:
  Stop Loss = 77.000 + 1,5 × 325 = $77.487
  Target    = 77.000 - 3,0 × 325 = $76.025
  Relação risco:retorno = 1:2

Gestão de risco:
  Risco por contrato = 1,5 × 325 = $487,50
  Risco desejado     = $250,00
  Contratos = 250 / 487,50 = 0,51 → 1 contrato

Resultado no dashboard:
  SINAL: SHORT
  Entrada: $77.000
  Stop: $77.487
  Target: $76.025
  Risco: $250
  Contratos: 1
```

---

## 10. Como Usar

### 10.1 Iniciar o servidor

```bash
# Na raiz do projeto
npm start
```

### 10.2 Acessar o dashboard

```
http://localhost:3000/live2
```

### 10.3 API JSON

```
GET /api/live2
```

Retorna:

```json
{
  "ts": "2026-05-20 10:54",
  "assets": {
    "btc": {
      "sinal": "SHORT",
      "direcao": "SHORT",
      "preco": 77000.0,
      "stop": 77487.0,
      "target": 76025.0,
      "conf_long": 20.0,
      "conf_short": 55.3,
      "conf": 55.3,
      "atr": 325.0,
      "risco_dolar": 250.0,
      "contratos": 1,
      "hora": "2026-05-20 13:00",
      "forward_h": 8,
      "setup": {
        "direcao": "short",
        "stop_r": 1.5,
        "target_r": 3.0,
        "ml_min": 0.5
      }
    }
  }
}
```

### 10.4 Sinais "NO_TRADE"

Quando `"sinal": "NO_TRADE"`, significa que a confiança do ML está abaixo de 50%, ou a direção não coincide com o setup permitido para o ativo. Nenhuma ação necessária.

---

## 11. Arquivos do Projeto

| Arquivo | Função |
|---------|--------|
| `ml/teste/predict_live.py` | Script principal de predição ao vivo |
| `ml/teste/train.py` | Script de treino dos modelos |
| `ml/teste/propfirm_model_mnq.pkl` | Modelo treinado MNQ |
| `ml/teste/propfirm_model_btc.pkl` | Modelo treinado BTC |
| `ml/teste/propfirm_model_cl.pkl` | Modelo treinado CL |
| `ml/teste/propfirm_model_mgc.pkl` | Modelo treinado MGC |
| `ml/teste/backtest_multiativo.py` | Backtest multi-ativo |
| `ml/teste/otimizar_propfirm.py` | Otimizador de parâmetros prop firm |
| `server.js` | Servidor Express (rota /api/live2) |
| `live2.html` | Dashboard de sinais |
| `live.html` | Dashboard original (com link para Live 2.0) |

---

## 12. Avisos

- **Performance passada não garante resultados futuros.**
- O modelo tem AUC ~0,55 — poder preditivo marginal, mas consistente.
- Recomenda-se começar com **conta demo** para validar a estratégia ao vivo.
- Re-treinar os modelos a cada **3-6 meses** para manter a relevância.
- Em caso de mudança brusca de regime de mercado (ex: novo all-time high do BTC), o modelo pode perder performance temporariamente.

---

> Documento gerado automaticamente — parte do sistema Live 2.0

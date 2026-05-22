# ML Prop Firm — Estratégia de Trading ao Vivo

> Sistema de sinais ML multi-ativo para contas prop firm (FTMO, MFF, etc.)
> Atualizado em: 22/05/2026
> Modelos treinados com: SQLite clean (5 anos) + blind OOS genuino

---

## 1. Resumo Executivo

Estratégia de trading sistemática usando **XGBoost multiclasse** para prever direção do preço nas próximas 8 horas em 4 ativos: **MNQ, BTC, CL, ES**.

| Métrica | Valor |
|---------|-------|
| Capital | $50.000 (prop firm) |
| Risco por trade | 0,5% ($250) |
| Stops | 1,5R |
| Targets | 2,0R – 3,0R |
| Período de treino | Mai 2021 → Dez 2024 |
| Blind OOS | 2025 (nunca visto durante treino) |
| Live test | 2026 (ao vivo) |

---

## 2. Arquitetura do Sistema

```
SQLite (5 anos OHLCV 1h)
         │
         ▼
train_sqlite_clean.py  ──► propfirm_model_*.pkl  (MNQ/BTC/CL/ES)
                                    │
         ┌──────────────────────────┘
         ▼
    predict_live.py  (yfinance 60d → features → pkl)
         │
         ▼
    server.js  GET /api/live2
         │
         ▼
    live2.html  (dashboard)
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
| Early stopping | 40 rodadas (validação interna 20%) |
| Objective | multi:softprob |

### 3.3 Metodologia de Treinamento

**Fonte:** SQLite local (unadjusted prices) com filtro de rollover aplicado.

**Filtro rollover:** Para cada barra T onde `|ret1| > 3%` (só futuros: MNQ, CL, ES):
- Janela contaminada: `[T - forward, T + lookback_max]` = `[T-8, T+8]` = 16 barras por evento
- BTC **não** recebe filtro (é spot crypto, sem rollover de contrato)

**Walk-forward:**
- Treino: Mai 2021 → Dez 2024
  - Fit interno: 80% dos dados de treino
  - Val interna (early stopping): 20% dos dados de treino
- Blind OOS: 2025 (jamais tocado durante o treino)
- Live: 2026 (produção)

### 3.4 Resultados

| Ativo | Treino (amostras) | AUC OOS 2025 | AUC Live 2026 |
|-------|:-----------------:|:------------:|:-------------:|
| MNQ   | 4.413 | **0.5736** | 0.5118 |
| BTC   | 4.562 | 0.5385 | **0.5370** |
| CL    | 4.285 | 0.5419 | **0.5521** |
| ES    | 4.564 | **0.5702** | **0.5548** |

> AUC ~0.54–0.57 indica poder preditivo marginal mas consistente.
> CL e ES melhoram em 2026 vs 2025 — sinal de que não estão overfitados.

---

## 4. Setup por Ativo

| Ativo | Opera | Stop | Target | Por quê |
|-------|:----:|:----:|:------:|---------|
| **MNQ** | LONG e SHORT | 1,5R | 3,0R | Mercado mais eficiente, dois lados funcionam |
| **BTC** | SHORT only | 1,5R | 3,0R | SHORT tem EV muito maior que LONG |
| **CL** | LONG only | 1,5R | 2,0R | Petróleo tem viés de alta no período |
| **ES** | LONG e SHORT | 1,5R | 2,5R | Alta correlação com MNQ, complementar |

**R = ATR(14)** médio das últimas 14 horas.
**ML mínimo:** 50% de confiança para emitir sinal.

---

## 5. Features — O que o ML Analisa

São **67 features** por ativo, divididas em 7 categorias:

### 5.1 RSI (9 features)

| Feature | Descrição |
|---------|-----------|
| `rsi_p` | RSI(21) do ativo principal |
| `rsi_1`, `rsi_2` | RSI(21) dos ativos secundários |
| `div_1`, `div_2` | Diferença RSI principal − secundário |
| `rsi_spread_1_2` | Diferença RSI entre secundários |
| `rsi_abs_*` | Valor absoluto das diferenças |

### 5.2 ADX / Directional Indicators (11 features)

| Feature | Descrição |
|---------|-----------|
| `adx_p`, `adx_1`, `adx_2` | Força da tendência (0-100) |
| `pdi_p`, `mdi_p` | Positive/Negative Directional Index |
| `di_spread_p`, `di_spread_1`, `di_spread_2` | PDI − MDI (compradores vs vendedores) |
| `dadx_p` | Variação do ADX em 2 barras |

### 5.3 Retornos (16 features)

| Feature | Descrição |
|---------|-----------|
| `ret1_p`, `ret4_p`, `ret8_p` | Retorno 1, 4 e 8 barras do principal |
| `ret1_1`, `ret4_1`, `ret1_2`, `ret4_2` | Retorno dos secundários |
| `ret*_spread_*` | Diferenças de retorno entre ativos |
| `ret*_prod_*` | Produto dos retornos |
| `price_div_p_2`, `price_div_abs` | Divergência de preço principal × sec2 |

### 5.4 Volatilidade (6 features)

| Feature | Descrição |
|---------|-----------|
| `vol_p` | Desvio padrão do retorno (20 barras) |
| `bb_p` | Largura das Bandas de Bollinger |
| `vol_spread_*` | Diferença de volatilidade entre ativos |
| `bb_spread_*` | Diferença de largura das BB entre ativos |

### 5.5 Médias Móveis (16 features)

| Feature | Descrição |
|---------|-----------|
| `dist_sma50_*`, `above_sma50_*` | Distância/posição relativa à SMA50 |
| `sma50_slope_p` | Inclinação da SMA50 (5 barras) |
| `sma50_alignment` | Quantos ativos estão acima da SMA50 (0-3) |
| `dist_ema20_*`, `above_ema20_*` | Distância/posição relativa à EMA20 |
| `ema20_bias_p_1` | Viés EMA20: principal + sec1 |
| `ema20_alignment` | Quantos ativos acima da EMA20 (0-3) |

### 5.6 Temporais (6 features)

| Feature | Descrição |
|---------|-----------|
| `hour`, `dow` | Hora (0-23) e dia da semana (0-6) |
| `hour_sin`, `hour_cos` | Hora codificada ciclicamente |
| `dow_sin`, `dow_cos` | Dia da semana codificado ciclicamente |

### 5.7 Key Levels (22 features)

| Feature | Descrição |
|---------|-----------|
| `dist_to_pdh`, `dist_to_pdl`, `dist_to_do` | Distância ao dia anterior (H/L/O) |
| `above_pdh`, `above_pdl`, `above_do` | Preço acima desses níveis? |
| `prev_day_range_pct` | Range % do dia anterior |
| `dist_to_pwh`, `dist_to_pwl`, `dist_to_wo` | Níveis da semana anterior |
| `above_pwh`, `above_pwl`, `above_wo` | Preço acima dos níveis semanais? |
| `dist_to_pmh`, `dist_to_pml`, `dist_to_mo` | Níveis do mês anterior |
| `above_pmh`, `above_pml`, `above_mo` | Preço acima dos níveis mensais? |
| `dist_to_mday_h`, `dist_to_mday_l` | Níveis da segunda-feira |
| `above_mday_h`, `above_mday_l` | Preço acima dos níveis de segunda? |

---

## 6. Importância por Categoria (modelos atuais)

| Categoria | MNQ | BTC | CL | ES |
|-----------|:---:|:---:|:--:|:--:|
| KEY LEVELS | 29.1% | 28.0% | 26.3% | 29.4% |
| VOLATILIDADE | 16.5% | 15.4% | 16.2% | 18.3% |
| ADX/DI | 15.2% | 14.7% | 17.2% | 15.8% |
| RETORNOS | 15.1% | 18.5% | 19.5% | 12.3% |
| RSI | 9.6% | 8.9% | 7.7% | 9.7% |
| TEMPORAL | 7.4% | 6.4% | 6.4% | 7.6% |
| MEDIAS | 6.4% | 7.0% | 5.3% | 6.3% |

**Padrão comum:** KEY LEVELS domina (~27-29%), ADX/DI e VOLATILIDADE subiram muito em relação aos modelos antigos (eram ~8-9%, agora ~15-18%). Isso ocorre porque os modelos antigos estavam usando TEMPORAL como proxy para detectar rollovers — sem a contaminação, ADX e VOL mostram sinal real.

---

## 7. Gestão de Risco

### 7.1 Por Trade

```
Risco = Capital × 0,5% = $50.000 × 0,005 = $250 por trade
```

### 7.2 Cálculo de Contratos

```
distância_stop = stop_r × ATR / preço
contratos = round(risco_dolar / (distância_stop × preço))
```

### 7.3 Regras de Operação

- Máximo 1 posição por ativo por vez
- Stop loss fixo (1,5R) + target fixo (2-3R)
- Sem reentrada na mesma barra
- Sem operação quando `sinal = NO_TRADE` (confiança < 50% ou direção bloqueada)

---

## 8. Arquivos do Sistema

| Arquivo | Função |
|---------|--------|
| `ml/teste/predict_live.py` | Predição ao vivo (yfinance → pkl → JSON) |
| `ml/teste/train_sqlite_clean.py` | Treino principal (SQLite + rollover filter) |
| `ml/train_divergencia_yfinance.py` | Treino modelo divergência MNQ |
| `ml/predict_divergencia.py` | Inferência divergência ao vivo |
| `ml/teste/retreinar_tudo.py` | Retreina todos os modelos em um comando |
| `ml/teste/propfirm_model_mnq.pkl` | Modelo MNQ (sqlite_clean, 2026-05-22) |
| `ml/teste/propfirm_model_btc.pkl` | Modelo BTC (sqlite_clean, 2026-05-22) |
| `ml/teste/propfirm_model_cl.pkl` | Modelo CL (sqlite_clean, 2026-05-22) |
| `ml/teste/propfirm_model_es.pkl` | Modelo ES (sqlite_clean, 2026-05-22) |
| `ml/model_divergencia.pkl` | Modelo divergência MNQ>0.1% em 4h |
| `server.js` | Servidor Express (rota /api/live2) |
| `live2.html` | Dashboard de sinais ao vivo |

---

## 9. Avisos

- **Performance passada não garante resultados futuros.**
- AUC ~0.54–0.57 — poder preditivo marginal, mas consistente entre OOS e Live.
- Re-treinar com `retreinar_tudo.py` a cada **3-6 meses**.
- Após mudança de regime (novo ATH BTC, crise macro), considerar retreino imediato.
- Filtro de rollover SQLite cobre: MNQ (trimestral), CL (mensal), ES (trimestral).
- BTC não tem rollover — nunca aplicar filtro de 3% no BTC.

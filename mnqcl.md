# MNQ Divergence — Dashboard MNQ ↔ BTC ↔ CL

## O que faz

A página `mnqcl.html` exibe em tempo real os indicadores técnicos de **MNQ**, **BTC** e **CL** via Yahoo Finance, mais a predição do modelo ML XGBoost para as próximas 4h do MNQ.

---

## Fluxo de dados

### 1. scan() — chamada principal

```
Usuário clica SCAN (ou muda RSI/ADX)
  ↓
fetch GET /api/mnq-cl/scan?rsi=21&adx=17
  ↓
[server.js:150] app.get('/api/mnq-cl/scan')
```

### 2. Backend (server.js:150-170)

| Passo | O que acontece |
|-------|----------------|
| 1 | `getInstrument('mnq')` — baixa 5d de candles de 5m do MNQ=F via Yahoo |
| 2 | `getInstrument('btc')` — mesmo processo para BTC-USD |
| 3 | `getInstrument('cl')` — mesmo processo para CL=F |
| 4 | Para cada ativo, calcula **RSI** (Wilder smoothing), **ADX** (14 períodos), **MA50**, distância % da MA50 e flag acima/abaixo |
| 5 | Calcula divergências: `mnq_cl_divergence = RSI(MNQ) − RSI(CL)`, `mnq_btc_spread = RSI(MNQ) − RSI(BTC)` |
| 6 | Retorna JSON com `{ mnq, btc, cl, divergences, ts }` |

**Cache**: resultados cacheados por 60s (CACHE_TTL).

### 3. Frontend — render(d) (linha 504)

Recebe o JSON e:

1. **Banner** (linhas 111-114): exibe RSI, preço, ADX, MA50% e viés de cada ativo (MNQ, BTC, CL)
2. **Divergências** (linhas 117-119): `mnq_cl_divergence` e `mnq_btc_spread` com codificação por cor (verde = negativo = favorável LONG)
3. **Sinal** (linhas 552-573): tripla confirmação LONG quando BTC RSI < 45 + ambas divergências negativas
4. **Condições** (linhas 575-578): 3 checkboxes visuais da tripla confirmação
5. **Tabela** (linhas 580-605): linhas para MNQ, BTC, CL com preço, RSI, ADX, condição, divergência

---

## Seção ML (Machine Learning)

Chamada via `scanML()` → `GET /api/ml/predict` (server.js:252).

O backend executa `predict.py` que carrega o modelo XGBoost treinado (`model.pkl`) e retorna:

| Campo | Descrição |
|-------|-----------|
| `prob_long` | Probabilidade de MNQ subir > 0.1% nas próximas 4h |
| `prob_short` | Probabilidade de MNQ cair > 0.1% nas próximas 4h |
| `signal_label` | LONG / SHORT / NEUTRO |
| `conf_diff` | prob_long − prob_short |
| `adx_mnq` | ADX atual do MNQ (do pipeline Python) |
| `strong_div` | Flag: price_div_cl < 0 AND ADX > 17 |
| `us_prime_setup` | Flag: strong_div + sessão US (9-17h) |
| `prime_setup` | Flag: strong_div + 18-21h |
| `cl_down_mnq_up` | Flag: CL caindo e MNQ subindo |
| `adx_active` | Flag: ADX > 14 (threshold original) |
| `is_us_session` / `is_us_morning` | Sessão atual |
| `hour` | Hora atual (UTC-5 NY) |
| `ema50_bias_mnq_btc` | 0=abaixo, 1=misto, 2=acima |
| `moving_against` | CL e MNQ em direções opostas |

**Decisão recomendada** (frontend, linhas 694-722):

| Confiança (L−S) | Decisão | Critério |
|-----------------|---------|----------|
| > 0.10 | LONG forte | edge +33.7% no teste |
| > 0.05 e prob_long > 0.50 | LONG moderado | edge +14.7% |
| < −0.10 | SHORT | edge contra, cuidado |
| < −0.05 e prob_short > 0.45 | SHORT moderado | usar filtro extra |
| outros | AGUARDAR | sem edge suficiente |

---

## Seção BTC Derivatives (Binance Futures)

Chamada via `scanBtcDer()` → `GET /api/btc/derivatives` (server.js:223).

| Métrica | Fonte | Interpretação |
|---------|-------|---------------|
| **LSR** | `topLongShortPositionRatio` | Long/Short ratio posicional |
| **OI Delta** | `openInterestHist` (2 períodos) | Variação % do open interest |
| **Taker B/S** | `takerlongshortRatio` | Razão compra/venda agressiva |
| **CVD Delta** | Klines (taker buy vol) | Cumulative Volume Delta % |

**Expansion Score** (frontend, computeExpansionScore, linha 306): score 0-100 que combina OI, taker, CVD, RSI, LSR e distância do nível para medir força do movimento.

---

## Filtros (topo da página)

| Controle | Opções | Padrão |
|----------|--------|--------|
| RSI | 21 / 14 | **21** |
| ADX | 17 / 14 / 7 | **17** |
| Auto refresh | 5 minutos | Desligado |

O frontend envia `?rsi=21&adx=17` na query string do scan.

---

## Sinal de Tripla Confirmação (seção superior)

| Condição | Significado |
|----------|-------------|
| BTC RSI < 45 | BTC oversold, condição necessária |
| `mnq_cl_divergence` < 0 | MNQ mais oversold que CL |
| `mnq_btc_spread` < 0 | MNQ mais oversold que BTC |

Quando as 3 condições são verdade → `LONG ✓` (verde) no card de sinal.

---

## Auto-refresh (▶ AUTO 5M)

- Botão amarelo alterna "▶ AUTO 5M" / "⏸ PARAR"
- A cada 5 minutos executa `scan()` (não o ML)
- Countdown regressivo visível

---

## Endpoints

| Endpoint | Descrição |
|----------|-----------|
| `GET /api/mnq-cl/scan?rsi=&adx=` | Scan Yahoo MNQ/BTC/CL |
| `GET /api/ml/predict` | Predição ML (cache 5min) |
| `GET /api/btc/derivatives` | BTC derivatives Binance (cache 5min) |

---

## Dependências

- **Server**: `rsi()`, `adx()`, `ma50()`, `fetchYahoo()` — todas em `server.js`
- **ML**: `ml/predict.py` + `ml/model.pkl` (XGBoost multiclasse)
- **Cache**: NodeCache in-memory (60s scan, 5min ML/BTC)
- **Frontend**: Vanilla JS, tema escuro CSS custom properties

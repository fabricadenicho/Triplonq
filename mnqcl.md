mnq# CL Divergence — Live Scan Page

## O que faz

A página `public/cl-divergence.html` escaneia ao vivo até 500 altcoins da Binance Futures (API pública) e calcula três métricas principais para identificar setups de **tripla confirmação LONG**:

| Métrica | Definição | Interpretação |
|---------|-----------|---------------|
| `cl_divergence` | RSI(altcoin) − RSI(CLUSDT) | Negativo = altcoin mais oversold que CL |
| `rsi_spread` | RSI(altcoin) − RSI(BTC) | Negativo = altcoin mais oversold que BTC |
| BTC RSI | RSI do Bitcoin | < 45 = BTC oversold (condição necessária) |

As 3 condições juntas formam a **tripla confirmação** com WR 56.2% (backtest 5m → 60m).

---

## Fluxo de dados

### 1. scan() — chamada principal

```
Usuário clica SCAN (ou muda RSI/Moedas/Força)
  ↓
fetch GET /api/cl-divergence/live-scan?rsi=14&limit=500
  ↓
[server.js, linha 3210] app.get('/api/cl-divergence/live-scan')
  ↓
```

### 2. Backend (server.js:3209-3284)

| Passo | O que acontece |
|-------|----------------|
| 1 | `getTopTickers(market, top)` — busca N tickers da Binance via `/ticker/24hr`, filtra por quoteVolume ≥ 200k USDT |
| 2 | `getKlines(BTCUSDT, '5m', 72)` — 72 candles de 5m (6h de histórico) para calcular RSI do BTC |
| 3 | `getKlines(CLUSDT, '5m', 72)` — mesmos candles para CLUSDT, calcula RSI como referência |
| 4 | Consulta `ensemble_decisions` — busca a decisão mais recente do ensemble para cada símbolo |
| 5 | Consulta `asset_classifications` — carrega classificação (oversold_bounce/momentum_follower/zombie) de cada símbolo |
| 6 | Consulta `market_snapshots` — carrega `oi_delta_pct` mais recente de cada símbolo |
| 7 | Batches de 10 símbolos — para cada altcoin, busca 72 klines, calcula RSI (14 ou 21 períodos), `cl_divergence = rsi - clRsi`, `rsi_spread = rsi - btcRsi` |
| 8 | Retorna `{ btc: { rsi, price }, cl: { rsi }, assets: [{ symbol, price, rsi, cl_divergence, rsi_spread, classification, oi_delta_pct, ens_pred, ens_conf }], total, rsiCol, ts }` |

**Cache**: klines são cacheadas em SQLite + memória com TTL de 120s para intervalo 5m (KLINE_TTL). Requisições subsequentes dentro de 2 minutos reusam o cache.

### 3. Frontend — render(d) (linha 216)

Recebe o JSON e:

1. **Banner** (linhas 224-237): exibe BTC RSI (verde < 45, amarelo 45-55, vermelho > 55), CL RSI, contagem de tripla confirmação, total de ativos
2. **Filtro de força** (linhas 243-254): ordena `cl_divergence` negativos ascendente, pega o p-ésimo percentil mais negativo, filtra
3. **SÓ SETUP** (linhas 256-258): filtra apenas os que satisfazem tripla confirmação
4. **Agrupamento** (linhas 260-268): classifica em Oversold Bounce → Momentum Follower → Zombie → Sem classificação
5. **Renderização** (linhas 270-286): para cada grupo, gera linhas HTML chamando `row()`

---

## Filtro de Força (percentil)

```
negVals = cl_divergence < 0, ordenado ascendente (mais negativo primeiro)
keep = max(1, floor(N * pct / 100))
threshold = negVals[keep - 1]
resultado = filtrar cl_divergence <= threshold
```

| Select | Comportamento |
|--------|---------------|
| Tudo | Sem filtro |
| Bottom 50% | Mantém os 50% mais negativos |
| Bottom 30% | Mantém os 30% mais negativos |
| Bottom 20% | Mantém os 20% mais negativos |
| Bottom 10% | Mantém os 10% mais negativos |
| Bottom 5% | Mantém os 5% mais negativos |

---

## Sinal CL (coluna "Sinal CL")

| Condição | Sinal |
|----------|-------|
| BTC RSI < 45 AND cl_divergence < 0 AND rsi_spread < 0 | `LONG ✓` (verde) |
| BTC RSI > 55 AND cl_divergence > 0 | `NADA` (vermelho) |
| BTC RSI < 45 AND cl_divergence < 0 AND rsi_spread ≥ 0 | `parcial` (amarelo) |
| Outros | `—` |

---

## Coluna ENS

Busca da tabela `ensemble_decisions` a predição mais recente do stacking ensemble:
- `LONG` → exibe `L {conf}%` em roxo (#c77dff)
- `SHORT` → exibe `S {conf}%` em roxo
- `NEUTRAL` → exibe `N {conf}%` em roxo
- Sem decisão → `—` (cinza)

---

## Auto-refresh (▶ AUTO 5M)

- Botão amarelo "▶ AUTO 5M" que alterna para "⏸ PARAR"
- A cada 5 minutos (300.000ms) executa `scan()` novamente
- Countdown regressivo visível ao lado: "próximo scan em 4:32"
- Para com "⏸ PARAR" ou recarregando a página

---

## Classificação dos ativos

As classificações vêm da tabela `asset_classifications` no banco SQLite:

| Classificação | Badge | Prioridade |
|---------------|-------|------------|
| `oversold_bounce` | badge-yel | 1 (topo) |
| `momentum_follower` | badge-blu | 2 |
| `zombie` | badge-red | 3 |
| `unknown` | badge-gray | 4 (final) |

A ordenação da tabela segue esta prioridade.

---

## RSI 14 vs RSI 21

O seletor `#rsiSelect` altera o parâmetro `rsi` na query string:

```
/api/cl-divergence/live-scan?rsi=14&limit=500
/api/cl-divergence/live-scan?rsi=21&limit=500
```

O backend usa `calcRsi(closes, rsiPeriod)` que implementa o RSI Wilder smoothing para qualquer período. A mudança afeta todas as 3 métricas simultaneamente (BTC RSI, cl_divergence, rsi_spread).

---

## Endpoints relacionados

| Endpoint | Descrição |
|----------|-----------|
| `GET /api/cl-divergence/live-scan?rsi=&limit=` | Scan via Binance (padrão) |
| `GET /api/cl-divergence/scan?rsi=&limit=` | Fallback via DB (apenas RSI 14 tem cl_divergence) |

---

## Dependências

- **Server**: `getKlines()`, `getTopTickers()`, `calcRsi()` — todas em `server.js`
- **Cache**: `KLINE_TTL['5m'] = 120` segundos (cache de klines em SQLite + NodeCache)
- **DB**: tabelas `ensemble_decisions` e `asset_classifications`
- **Frontend**: Vanilla JS, sem frameworks. CSS custom properties com tema escuro.

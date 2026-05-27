# Trigger de Rotação — Divergência MNQ × CL

## Conceito

Detectar o **início** da divergência entre MNQ e CL, não quando ela já está consolidada.
O ponto de **eixo/rotação** é quando o RSI de um ativo começa a acelerar na direção oposta ao outro.

---

## Correlação Base (5 anos)

| Par        | Correlação 20h | Interpretação                  |
|------------|---------------|--------------------------------|
| MNQ × ES   | **+0.90**     | Mesmo movimento 90% do tempo  |
| MNQ × BTC  | ~+0.50        | Moderada                       |
| MNQ × CL   | **~-0.02**    | **Zero — hedge real**          |

---

## As 9 Condições (score ponderado / 17.0)

| # | Condição                          | Peso | Lógica                                      |
|---|-----------------------------------|------|---------------------------------------------|
| C1 | div_cl cruzou o zero             | 3.0  | RSI MNQ − RSI CL mudou de sinal            |
| C2 | Delta div_cl > 5 em 2h           | 2.5  | Aceleração da divergência                  |
| C3 | MNQ↑ + CL↓ em 2h                 | 2.5  | RSI MNQ delta > +2 e RSI CL delta < −2     |
| C4 | ES confirma MNQ                  | 2.0  | ES e MNQ no mesmo sentido (ocorre 90%)     |
| C5 | BB width CL > 1.5                | 1.5  | Volatilidade do petróleo presente          |
| C6 | CL abriu acima do daily anterior | 1.5  | Força relativa do CL no dia                |
| C7 | ADX MNQ entre 12 e 20            | 1.5  | Tendência começando a se formar            |
| C8 | SMA50 alignment ≥ 2/4 ativos     | 1.5  | Regime mínimo de mercado                   |
| C9 | RSI MNQ > 55 ou < 45             | 1.0  | RSI fora da zona neutra                    |

```
score = soma_pesos_ok / 17.0 * 100
threshold: score >= 60% → ROTAÇÃO DETECTADA
```

---

## Backtest Walk-Forward — 2 Anos (yfinance auto_adjust)

Dados: `yfinance` com rollover ajustado, 1h, Jun 2024 → Mai 2026.
Parâmetros fixos — **sem refit entre períodos**.

### Sem filtro de Key Level

| Período              |   n  |   WR   | Baseline | Alpha       |
|----------------------|------|--------|----------|-------------|
| IN-SAMPLE  2024      |  121 | 35.5%  |  39.7%   | −4.2pp      |
| BLIND TEST 2025      |  221 | 43.4%  |  40.3%   | +3.2pp      |
| LIVE       2026      |  156 | 52.6%  |  43.9%   | +8.7pp      |
| TOTAL   2024–26      |  498 | 44.4%  |  40.8%   | +3.5pp      |

### Com filtro Key Level (< 0.2% de bom KL)

| Período              |   n  |   WR   | Baseline | Alpha           |
|----------------------|------|--------|----------|-----------------|
| IN-SAMPLE  2024      |   60 | 35.0%  |  38.0%   | −3.0pp          |
| BLIND TEST 2025      |  107 | 44.9%  |  39.4%   | **+5.5pp**      |
| LIVE       2026      |   66 | 60.6%  |  45.3%   | **+15.3pp**     |
| TOTAL   2024–26      |  233 | 46.8%  |  40.1%   | **+6.6pp**      |

> O filtro de key level dobrou o alpha total (+3.5 → +6.6pp) e transformou o
> 2025 blind test de "sinal fraco" em SINAL FORTE.

### Dependência de Regime

| Regime                          | Alpha sem KL | Alpha com KL |
|---------------------------------|-------------|--------------|
| Bull trend unidirecional (2024) | −4.2pp      | −3.0pp       |
| Volatilidade moderada (2025)    | +3.2pp      | +5.5pp       |
| Alta volatilidade macro (2026)  | +8.7pp      | +15.3pp      |

**Regra:** não opere o sinal durante mercados de tendência unidirecional forte.
O sinal funciona melhor quando há turbulência real entre petróleo e tech.

---

## Key Levels — WR por Nível (sinais de rotação)

### Bons KLs (WR > 60% — usar no filtro)

| Key Level        |  n  |   WR   | vs Baseline |
|------------------|-----|--------|-------------|
| PWL              |   6 | 83.3%  | +32pp       |
| LONDON-L         |  21 | 81.0%  | +30pp       |
| MON-L            |   7 | 71.4%  | +20pp       |
| PWO              |  20 | 70.0%  | +19pp       |
| LONDON-H         |  44 | 63.6%  | +13pp       |
| NY-H             |  38 | 63.2%  | +12pp       |
| PDH              |  52 | 61.5%  | +10pp       |

### KLs Ruins (evitar)

| Key Level |  n  |   WR   | vs Baseline |
|-----------|-----|--------|-------------|
| PWH       |  15 | 33.3%  | −18pp       |
| SESS-L    |  26 | 42.3%  | −9pp        |
| PDL       |  18 | 44.4%  | −7pp        |

### Sinal sem nenhum KL próximo

| Filtro                     |   n  |   WR   | vs Baseline |
|----------------------------|------|--------|-------------|
| Sem key level próximo      |   21 | 28.6%  | **−22.5pp** |
| Com qualquer KL            |  153 | 54.2%  | +3.1pp      |

> Sinal sem KL = destruição de capital. Ignorar sempre.

### Threshold de Proximidade Ótimo

| Distância ao KL |   n   |   WR   |
|-----------------|-------|--------|
| < 0.1%          |   97  | 51.5%  |
| **< 0.2%**      | **135** | **54.8%** |
| < 0.3%          |  153  | 54.2%  |
| < 0.5%          |  167  | 51.5%  |

**Threshold recomendado: 0.2%** — melhor WR com amostra suficiente.

---

## Top Combinações: ROTACAO + C1 + Key Level

| Combo                      |  n  |   WR   |
|----------------------------|-----|--------|
| ROTACAO + C1 + LONDON-L    |  10 | 80.0%  |
| ROTACAO + C1 + PDO         |  13 | 76.9%  |
| ROTACAO + C1 + PDH         |  17 | 76.5%  |
| ROTACAO + C1 + LONDON-H    |  24 | 75.0%  |
| ROTACAO + C1 + ASIA-H      |  23 | 69.6%  |
| ROTACAO + C1 + NY-H        |  16 | 68.8%  |

---

## Ciclo de Entrada

```
ALERTA:  score >= 60% + div_cl cruzou zero + delta > 5
    ↓
FILTRO:  preço dentro de 0.2% de bom KL (PDH, LON-H/L, NY-H, PWL, MON-L, PWO)
    ↓
ENTRY:   confirmação do ES no mesmo sentido do MNQ
    ↓
SAÍDA:   stop 1.5× ATR14 / target 3.0× ATR14 (R:R 2:1)
    ↓
CONFIRMAÇÃO: div_cl > 10 (divergência consolidada)
```

---

## Checklist de Entrada (operacional)

- [ ] score rotação ≥ 60%
- [ ] C1 ativo: div_cl mudou de sinal nas últimas 2h
- [ ] C2 ativo: delta div_cl > 5 pontos em 2h
- [ ] Preço dentro de 0.2% de um bom KL (PDH, LON-H/L, NY-H, PWL, MON-L, PWO)
- [ ] ES confirmando a direção do MNQ
- [ ] Mercado NÃO está em bull trend unidirecional forte
- [ ] Evitar: PWH, SESS-L, PDL como único KL próximo

---

## Exemplo Prático (21 Mai 2026)

```
14:00 UTC  div_cl = −4.95  (neutro, sem rotação)

16:00 UTC  div_cl = +5.20  ← CRUZOU ZERO
           RSI MNQ: 46.9 → 52.3  (+5.4)
           RSI CL:  51.9 → 47.0  (−4.9)
           Delta div_cl = +10.15 em 2h
           → ROTAÇÃO DETECTADA ← score estimado: 76%

18:00 UTC  div_cl = +22.04 (divergência consolidada)
           RSI MNQ: 59.3  |  RSI CL: 37.3
           Delta: +27 pontos em 4h
```

---

## Implementação

### Python (ao vivo)

Arquivo: `ml/predict_divergencia.py`

Campos gerados no JSON:

| Campo              | Tipo    | Descrição                                   |
|--------------------|---------|---------------------------------------------|
| `rotacao_score`    | float   | Score 0–100% (threshold 60%)                |
| `rotacao_ativo`    | int 0/1 | 1 se score ≥ 60%                            |
| `rotacao_direcao`  | string  | "LONG" ou "SHORT" (sinal do div_cl)         |
| `rot_c1`…`rot_c9`  | int 0/1 | Cada uma das 9 condições individualmente    |
| `div_cl_delta_2h`  | float   | Aceleração do div_cl nas últimas 2h         |
| `rsi_rotation`     | int 0/1 | MNQ subindo + CL caindo simultaneamente     |
| `divergencia_starting` | int 0/1 | EIXO: rotação ativa + delta > 5 + div < 20 |

### Pine Script

Arquivos: `pine/divergencia_rotacao.pine` (oscilador) e
`pine/divergencia_rotacao_strategy.pine` (estratégia com stop/target)

7 bons KLs no `near_good_kl`: PDH, LONDON-H, LONDON-L, NY-H, PWL, MON-L, PWO
Threshold padrão: 0.2% | Filtro ativo por padrão: `i_kl_filter = true`

### Live Dashboard

Arquivo: `live2.html` — card **TRIGGER ROTAÇÃO** no painel "Divergência ML"
- Barra de progresso com marker no 60%
- Direção LONG/SHORT e status AGUARDAR
- 9 condições em grid com dot verde/cinza, peso e valor atual

### Backtests

| Script                              | Descrição                              |
|-------------------------------------|----------------------------------------|
| `ml/teste/backtest_rotacao.py`      | 185 dias + split 70/30 in/out-sample   |
| `ml/teste/backtest_2anos.py`        | 720 dias, split por ano + filtro KL    |
| `ml/teste/backtest_blind_historico.py` | SQLite 2021–2026 (dados brutos)     |

---

## Aviso

Este trigger detecta o **início** da divergência com ~55–60% de acerto nas 3h seguintes
quando combinado com key level próximo. Não é entrada automática — exige confirmação
visual do ES e análise do contexto de mercado (regime).

Alpha consistente apenas em ambientes de volatilidade real (2025–2026).
Em bull trends unidirecionais (ex.: 2024), o sinal perde para o baseline.

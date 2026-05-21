# Trigger de Aceleração/Rotação — Divergência MNQ × CL

## Conceito

Detectar o **início** da divergência entre MNQ e CL, não quando ela já está consolidada. O ponto de **eixo/rotação** é quando o RSI de um ativo começa a acelerar na direção oposta ao outro.

## Estatísticas (5 anos de dados)

### Correlação Base

| Par | Correlação 20h | Interpretação |
|-----|---------------|---------------|
| MNQ × ES | **+0.90** | Mesmo movimento 90% do tempo |
| MNQ × BTC | ~+0.50 | Moderada |
| MNQ × CL | **~-0.02** | **Zero — hedge real** |

### Quando MNQ sobe + CL cai, o que os outros fazem?

| Horário (BRT) | ES segue MNQ | BTC segue MNQ | WR MNQ 3h |
|---------------|-------------|---------------|-----------|
| 06h | 90% | 68% | 37% |
| **07h** | **92%** | 60% | 40% |
| 08h | 85% | 68% | 42% |
| 09h | 90% | 63% | 41% |
| **10h** | **84%** | 65% | **53%** |
| **11h** | **88%** | 64% | **52%** |
| 12h | 83% | 65% | 40% |

## Gatilho de Rotação (Eixo)

O timing importa, não a hora fixa. A rotação é detectada quando as condições abaixo ocorrem SIMULTANEAMENTE, independente do horário.

### Condições para Detectar o Início da Divergência

1. **div_cl mudando de direção** — RSI MNQ - RSI CL cruza de negativo para positivo (ou vice-versa)
2. **Aceleração do div_cl** — delta do div_cl nas últimas 2h > 5 pontos (a velocidade importa mais que a magnitude absoluta)
3. **MNQ RSI subindo + CL RSI caindo** no mesmo candle (rotação ativa)
4. **ES confirmando** — ES no mesmo sentido do MNQ (ocorre 90% das vezes quando há divergência)
5. **BB width do CL expandindo** — volatilidade aumentando

### Timing da Rotação (não a hora)

O ciclo típico após a rotação:

```
ALERTA: div_cl cruza o zero + delta > 5 (ROTAÇÃO DETECTADA)
    ↓
ENTRY: próxima confirmação do ES no mesmo sentido (WR 53% nas 3h seguintes)
    ↓
CONFIRMAÇÃO: divergência consolidada com div_cl > 10
```

Estatisticamente, a rotação ocorre com mais frequência entre 06-08h BRT, mas o trigger é **puramente técnico** — quando as condições batem, o timing é válido em qualquer horário.

### Checklist de Entrada (sem hora fixa)

- [ ] div_cl mudou de sinal nas últimas 2h (rotação)
- [ ] delta div_cl > 5 pontos (aceleração)
- [ ] RSI MNQ subindo + RSI CL caindo (rotação ativa)
- [ ] RSI MNQ > RSI CL (divergência positiva para LONG, ou inversa para SHORT)
- [ ] MNQ e CL andando opostos no candle atual
- [ ] ES confirmando (mesma direção do MNQ)
- [ ] CL BB width > 1.5 (volatilidade presente)
- [ ] Score checklist ponderado > 40%

## Score de Rotação (0-100)

O peso reflete a importância de cada condição, sem depender de horário fixo:

```python
pesos = {
    'div_cl_mudou_sinal':       3.0,  # cruzou de negativo pra positivo
    'div_cl_delta_2h > 5':      2.5,  # acelerou > 5 pontos em 2h
    'rsi_mnq_subindo + cl_cain': 2.5,  # rotacao ativa no mesmo candle
    'es_mesmo_sentido':          2.0,  # ES confirma MNQ (90% das vezes)
    'bb_w_cl_alto':              1.5,  # > 1.5 (volatilidade)
    'cl_daily_acima_ant':        1.5,  # CL abriu acima do daily anterior
    'adx_mnq_entre_12_20':       1.5,  # tendencia comecando a formar
    'sma50_alignment >= 2':      1.5,  # regime minimo
    'rsi_mnq_fora_neutro':       1.0,  # > 55 ou < 45
}
threshold: >= 60% → ROTAÇÃO DETECTADA
```

## Exemplo Prático (21 Maio 2026)

```
14:00 UTC (11h BRT): div_cl = -4.95 (neutro, sem rotacao)
16:00 UTC (13h BRT): div_cl = +5.20  (cruzou zero! delta = +10)
                      RSI MNQ: 46.9 → 52.3 (+5.4)
                      RSI CL:  51.9 → 47.0 (-4.9)
                      → ROTAÇÃO DETECTADA ←
18:00 UTC (15h BRT): div_cl = +22.04 (divergencia consolidada)
                      RSI MNQ: 59.3  RSI CL: 37.3
                      Delta div_cl: 27 pontos em 4h ⚡
```

A rotação foi detectada **quando o div_cl cruzou o zero** acompanhado de aceleração > 5 pontos. Não importa se eram 11h ou 13h BRT — o trigger é o mesmo.

## Implementação

No `predict_divergencia.py`:
- `div_cl_delta_2h` — diferença do div_cl atual vs 2h atrás
- `rsi_mnq_delta` — RSI MNQ atual vs anterior
- `rsi_cl_delta` — RSI CL atual vs anterior
- `rotacao_detectada` — booleano: div_cl mudou de sinal E delta > 5

## Aviso

Este trigger detecta o **início** da divergência com ~55-60% de acerto nas 3h seguintes. Não é entrada automática — exige confirmação do ES e análise visual dos níveis.

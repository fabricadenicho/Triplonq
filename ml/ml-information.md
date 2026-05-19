# ML Information — Comparativo Completo MNQ / BTC / CL

## Visao Geral dos 3 Modelos

| Metrica | MNQ | BTC | CL |
|---------|-----|-----|-----|
| AUC | **0.602** | 0.525 | 0.523 |
| Amostras de treino | 5400 | 6500 | 5392 |
| Features | 38 | 38 | 38 |
| Forward | 4h | 4h | 4h |
| Intervalo | 1h | 1h | 1h |
| Sessao de treino | US 9-17h | US 9-17h | US 9-17h |
| Feature #1 | price_div_abs (4.3%) | **dow (8.3%)** | us_prime_setup (5.4%) |
| strong_div edge | **+10.6% LONG** | +3.5% SHORT | +9.3% LONG |
| Acuracia direcional | ~47% | ~43% | ~47% |

---

## 1. MNQ (Mini Nasdaq-100 Futures) — O Melhor Modelo

### Pontos Fortes
- **AUC 0.602** — unico modelo acima de 0.60, considerado "bom" para predicao de direcao
- **strong_div com +10.6% edge** — quando ativo, LONG acerta 47.1% vs SHORT 36.5%
- **Melhor horario**: 10h (50.3% LONG) e 13h (49.2% LONG)
- **Feature #1: price_div_abs** — a magnitude da divergencia de preco entre MNQ e CL e a feature mais importante

### Curiosidades e Insights
- O MNQ e o unico ativo onde `price_div_abs` (divergencia de preco) e a feature #1. Isso significa que o modelo aprendeu que **quando Nasdaq e Petroleo andam em direcoes opostas, o MNQ tende a reverter na direcao do CL**
- O triple_signal (BTC RSI < 45 + divergencias RSI negativas) foi criado manualmente mas o XGBoost **nao deu importancia pra ele** — o modelo prefere usar as divergencias de preco bruto
- O MNQ tem **NEUTRO = 16.7%**, o maior entre os 3 ativos. Isso significa que o MNQ passa mais tempo em consolidacao que BTC e CL
- **Sazonalidade semanal fraca** — diferente do BTC, `dow` nao aparece no top 20. MNQ se importa mais com a **hora do dia** do que com o dia da semana

### Melhores Divergencias

| Divergencia | Descricao | Como usar |
|-------------|-----------|-----------|
| **price_div_cl < 0** | MNQ e CL em direcoes opostas | LONG quando CL cai e MNQ sobre (ou vice-versa) |
| **price_div_abs alto** | Magnitude da divergencia | Quanto maior, maior a chance de reversao |
| **div_cl < 0** | RSI(MNQ) < RSI(CL) | MNQ mais oversold que petroleo — favorece LONG |
| **div_btc < 0** | RSI(MNQ) < RSI(BTC) | MNQ mais oversold que Bitcoin — confirma |

### Quando usar na pratica
- **Preferencia**: Sessao US, 10h ou 13h
- **Entrada LONG**: strong_div ativo + price_div_cl < 0 + ADX > 17 + EMA20 bias >= 1
- **Entrada SHORT**: price_div_abs alto + price_div_cl > 0 + EMA20 bias <= 1
- **Evitar**: Fora do horario US (9-17h) — modelo nao foi treinado pra isso
- **Stop sugerido**: 12-35pts MNQ (baseado no ADX * 1.2)
- **Contratos (50k prop firm)**: ate 3 contratos em confianca alta, 2 em moderada

---

## 2. BTC (Bitcoin) — A Sazonalidade Semanal

### Pontos Fortes
- **Feature #1: dow (8.3%)** — o BTC tem a sazonalidade semanal mais forte entre os 3 ativos. O XGBoost aprendeu padroes como "BTC cai segundas, sobe quartas"
- **Divergencia RSI (11.9%)** — a categoria mais importante, mostrando que o BTC reage fortemente a divergencias de RSI com MNQ e CL
- **Melhor horario**: 17h (45.6% LONG) e 11-13h (~50% SHORT)
- **Confiabilidade nas primeiras horas**: o modelo tem mais amostras que CL (6500 vs 5392)

### Curiosidades e Insights
- **Comportamento contra-tendencia**: diferente do MNQ, quando `strong_div` esta ativo, o BTC favorece **SHORT** (46.2% vs 42.7%). Enquanto o MNQ reverte na direcao oposta ao CL, o BTC segue na mesma direcao
- **Volatilidade do MNQ influencia BTC**: `vol_mnq` e a 2a feature mais importante (4.4%). A volatilidade do Nasdaq influencia mais o Bitcoin que a propria volatilidade do BTC
- **Melhor regime**: EMA20 misturado (BTC acima, MNQ abaixo ou vice-versa) favorece LONG com 45.2%
- **PIOR regime**: ambos abaixo da EMA20 — SHORT domina com 48.4%
- **Diferente do MNQ**: o BTC **nao** tem um horario claramente favoravel pra LONG como o MNQ tem (10h). Em vez disso, os horarios sao mais equilibrados

### Melhores Divergencias

| Divergencia | Descricao | Como usar |
|-------------|-----------|-----------|
| **price_div_abs** (4.0%) | Magnitude divergencia BTC-MNQ | Quando BTC e MNQ divergem forte, atencao |
| **price_div_cl** (3.3%) | BTC vs CL divergencia de preco | BTC tende a seguir a direcao do CL |
| **div_cl** (3.3%) | RSI(BTC) < RSI(CL) | BTC oversold vs petroleo |
| **sma50_align_mnq_cl** (3.9%) | Alinhamento SMA50 MNQ+CL | Quando ambos estao no mesmo lado, BTC segue |

### Quando usar na pratica
- **LONG**: EMA20 misturado + 15-17h + ADX BTC > 17
- **SHORT**: EMA20 ambos abaixo + 11-13h + strong_div ativo
- **Filtro extra**: verificar o dia da semana (se for segunda ou sexta, SHORT tem mais chance)
- **Stop sugerido**: 100-500pts (BTC e muito mais volátil)
- **Contratos (50k prop firm)**: max 2 contratos confianca alta, 1 moderada

---

## 3. CL (Crude Oil) — O Reverso do MNQ

### Pontos Fortes
- **strong_div com +9.3% edge LONG** — o unico ativo que chega perto do MNQ em consistencia de edge
- **Melhor regime LONG: ambos abaixo da EMA20 = 56.3%** — o numero mais alto de qualquer regime em qualquer dos 3 ativos
- **us_prime_setup e a feature #1 (5.4%)** — mostra que o trigger composto funciona bem pra petroleo
- **Melhor horario**: 11h (52.0% LONG) — o horario mais bullish entre todos os modelos
- **Distribuicao LONG mais alta**: 47.0% LONG vs 43.1% SHORT — o CL e naturalmente mais comprador que os outros

### Curiosidades e Insights
- **Comportamento REVERSO ao MNQ**: quando CL e BTC estao abaixo da EMA20, o CL tem 56.3% de subir (reversao de oversold). No MNQ, ambos abaixo e ruim. No CL, ambos abaixo e otimo.
- **ADX do BTC importa mais que o ADX do CL**: `adx_btc` (2.9%) aparece no top 10, enquanto o ADX do proprio CL nao aparece. A volatilidade do Bitcoin influencia o petroleo.
- **Bollinger spread CL-MNQ e relevante**: `bb_spread_cl_mnq` (2.9%) mostra que quando as bandas de Bollinger entre CL e MNQ divergem, o CL tende a se mover
- **Divergencia de RSI com MNQ**: `div_mnq` = RSI(CL) - RSI(MNQ) com 2.9% de importancia. O modelo compara o RSI do CL com o MNQ constantemente
- **NEUTRO mais baixo (9.9%)**: o CL passa menos tempo consolidando que MNQ e BTC. E mais direcional.

### Melhores Divergencias

| Divergencia | Descricao | Como usar |
|-------------|-----------|-----------|
| **price_div_cl** (2.8%) | CL vs MNQ direcoes opostas | CL reverte na direcao oposta ao MNQ |
| **bb_spread_cl_mnq** (2.9%) | Diferenca Bollinger CL-MNQ | CL volatil vs MNQ calmo = movimento iminente |
| **vol_spread_cl_mnq** (2.9%) | Diferenca volatilidade CL-MNQ | CL muito mais volatil que MNQ = oportunidade |
| **div_mnq** (2.9%) | RSI(CL) < RSI(MNQ) | CL oversold relativo ao MNQ |

### Quando usar na pratica
- **LONG (preferencia)**: strong_div ativo + CL+BTC abaixo da EMA20 + 11h + ADX > 17 = **56.3% de acertar**
- **SHORT**: EMA20 misturado + requer mais confirmacao que LONG
- **Melhor horario**: 11h (52% LONG)
- **Edge mais forte**: us_prime_setup ativo + ambos abaixo EMA20 (56.3% LONG)
- **Stop sugerido**: 30-100pts (baseado no ADX * 3)
- **Contratos (50k prop firm)**: max 3 contratos confianca alta (CL e $10/pt)

---

## Tabela Comparativa: Qual ativo usar em cada situacao

| Situacao | Melhor ativo | Por que |
|----------|-------------|---------|
| Maior confianca estatistica | **MNQ** | AUC 0.602, edge +10.6% strong_div |
| Reversao de oversold | **CL** | 56.3% LONG quando ambos abaixo EMA20 |
| Manha (11-13h) | **CL** (52% LONG) ou **BTC** (50% SHORT) | CL pra LONG, BTC pra SHORT |
| Tarde (15-17h) | **MNQ** (49-50% LONG) | Horario com melhor win rate do MNQ |
| Segunda-feira | **BTC SHORT** | Sazonalidade semanal (dow) |
| Sexta-feira | **BTC SHORT** | Sazonalidade semanal (dow) |
| ADX > 25 | **MNQ** | Modelo mais robusto em tendencia forte |
| Mercado calmo (ADX < 17) | **NENHUM** | Nenhum modelo funciona sem tendencia |
| CL e BTC caindo junto | **CL LONG** | 56.3% de subir (reversao) |
| BTC RSI < 45 | **MNQ LONG** | Triple signal do MNQ |

---

## Matriz de Correlacao dos Modelos

```
              MNQ          BTC          CL
MNQ    -                   
BTC    OPOSTO        -     
CL     MESMO         OPOSTO   -
```

- **MNQ e CL**: andam juntos (strong_div favorece LONG em ambos)
- **MNQ e BTC**: andam opostos (strong_div favorece LONG no MNQ, SHORT no BTC)
- **BTC e CL**: andam opostos (BTC reverte quando CL forte)

**Implicacao pratica**: se o ML SCAN do MNQ da LONG com confianca alta e o BTC SCAN da SHORT com confianca alta, e um **sinal mais forte** — os modelos estao concordando indiretamente.

---

## Regras de Ouro

1. **NUNCA** operar contra o sinal do `strong_div` no MNQ — e o trigger mais consistente (+10.6%)
2. **Sempre** verificar o regime EMA20 antes de entrar — muda completamente a probabilidade em todos os ativos
3. **Sem ADX > 17, sem trade** — nenhum modelo tem edge sem tendencia presente
4. **Horario US 9-17h ou nada** — os modelos foram treinados exclusivamente na sessao americana
5. **BTC: dia da semana importa** — se for segunda ou sexta, o bias natural e SHORT
6. **CL: quando ambos abaixo EMA20, comprar** — 56.3% de acertar e o numero mais alto do sistema
7. **MNQ: divergencia de preco e rei** — price_div_abs e a feature #1, preste atencao quando MNQ e CL divergem
8. **Se MNQ e BTC concordam (LONG + SHORT), e sinal forte** — os modelos tem comportamentos opostos, entao quando se alinham e significativo

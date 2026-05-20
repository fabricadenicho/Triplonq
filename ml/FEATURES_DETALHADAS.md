# Top 10 Features por Ativo — Explicacoes Detalhadas

## Sumario

- [MNQ (Mini Nasdaq)](#1-mnq-mini-nasdaq)
- [BTC (Bitcoin)](#2-btc-bitcoin)
- [CL (Crude Oil)](#3-cl-petroleo)
- [MGC (Micro Gold)](#4-mgc-ouro)

---

## 1. MNQ (Mini Nasdaq)

### #1 — kz_range (9.20%)

Ganho percentual entre o topo e o fundo da kill zone atual (Asia, Londres ou NY). Quanto maior o range, mais volatil a sessao.

**Exemplo grafico:** Imagine a sessao de Londres (8-14h UTC). As 8h o MNQ abre em 19.000, faz uma minima de 18.950 e as 13h esta em 19.080. `kz_range = (19080-18950)/18950 = 0.69%`. Um range grande como este indica que a sessao tem movimentos amplos — o modelo aprendeu que rangess grandes precedem continuacao ou reversao.

### #2 — is_us_session (5.60%)

Flag binaria: 1 quando 9-17h NY, 0 fora.

O modelo prefere saber **se a sessao US esta aberta** do que qualquer indicador tecnico. Historicamente, 70% da volatilidade do MNQ ocorre neste horario.

### #3 — is_london (4.68%)

Flag binaria: 1 quando 8-14h UTC (sessao de Londres).

Londres sobrepoe-se com a manha US (9-14h), criando o periodo de maior liquidez do dia. O modelo detetou que o MNQ tem comportamento diferente quando Londres esta ativa — maior probabilidade de gaps e reversoes.

### #4 — is_us_morning (4.56%)

Flag binaria: 1 quando 9-13h NY.

**Exemplo grafico:** Num dia tipico, 9h30 NY saem os dados economicos (empregos, PIB, CPI). O MNQ pode saltar 50-100 pontos em minutos. O modelo sabe que a manha US tem mais noticias e reacoes abruptas do que a tarde.

### #5 — kz_dist_high (3.55%)

Distancia percentual do preco atual ate o topo da kill zone atual.

**Exemplo grafico:** Sessao NY comeca, MNQ faz topo em 19.120 as 14h. Agora sao 16h e o preco esta em 19.080. `kz_dist_high = (19080-19120)/19120 = -0.21%`. O modelo ve que o preco esta 0.21% abaixo do topo da sessao. Se o preco romper os 19.120, pode ser breakout. Se continuar a afastar-se, pode ser reversao.

### #6 — is_asia (3.50%)

Flag binaria: 1 quando 0-7h UTC (sessao asiatica).

**Exemplo grafico:** A sessao asiatica e de baixa liquidez. O MNQ move-se menos, mas as vezes faz "falsos breakout" que sao corrigidos na abertura de Londres. O modelo sabe distingui-los.

### #7 — price_div_abs (2.92%)

Menor que no modelo US-only (4.3%). `|ret1(MNQ) x ret1(CL)|`.

**Veja explicacao detalhada na seccao de divergencia.**

### #8 — vol_mnq (2.62%)

Desvio padrao do retorno de 1h nos ultimos 20 periodos.

**Exemplo grafico:** O VIX (volatilidade implicita) esta alto, o MNQ mexe-se 0.5% por hora. `vol_mnq` calcula a media desses movimentos. Se `vol_mnq` esta alto (>0.8%), o modelo sabe que esta num regime de alta volatilidade e ajusta as probabilidades.

### #9 — hour (2.59%)

Hora do dia (0-23).

**Exemplo grafico:** `hour=10` (10h NY). O modelo sabe que o periodo pos-abertura (9h30-11h) tem padroes diferentes do lunch period (12-13h) ou do fecho (15-16h).

### #10 — rsi_mnq (2.40%)

RSI periodo 21 do MNQ.

Valores < 30 indicam sobrevenda, > 70 sobrecompra. No entanto, o modelo pesa-o menos que as sessoes e divergencias.

---

## 2. BTC (Bitcoin)

### #1 — kz_range (7.15%)

O BTC e o unico ativo que negocia 24/7 — mas as kill zones ainda importam porque a volatilidade concentra-se nas sessoes de NY e Asia.

### #2 — dow (4.76%)

Dia da semana (0=segunda a 6=domingo).

**Exemplo grafico:** Historicamente, segundas-feiras tendem a ter gaps (herdados do fim-de-semana). Quartas-feiras tem bias LONG, sextas-feiras bias SHORT. No modelo US-only, `dow` era a feature #1 com 8.3% — mostra que o BTC e o ativo mais sazonal da semana.

### #3 — vol_mnq (3.61%)

Volatilidade do Nasdaq, nao do proprio BTC.

**Exemplo grafico:** O VIX salta, o MNQ cai 2%. O BTC frequentemente segue o movimento do Nasdaq com 15-30 minutos de atraso. O modelo aprendeu que quando o MNQ esta volatil, o BTC provavelmente vai mexer-se tambem, mesmo que o RSI do BTC esteja neutro.

### #4 — is_asia (3.30%)

A sessao asiatica e particularmente importante para BTC porque grande parte do volume de criptomoedas vem da Asia (Coreia, Japao, China).

### #5 — bb_mnq (2.69%)

Largura das Bandas de Bollinger do MNQ.

**Exemplo grafico:** Bollinger do MNQ contrai-se (squeeze) — `bb_mnq` baixo. O modelo sabe que quando o Nasdaq esta comprimido, o BTC tambem tende a ficar num range estreito. Quando o Bollinger expande, o BTC costuma acompanhar o movimento.

### #6 — is_ny (2.42%)

Flag da sessao de NY (13-19h UTC). O BTC tem picos de volatilidade na abertura dos mercados americanos.

### #7 — is_evening (2.33%)

Periodo 18-21h NY.

**Exemplo grafico:** As 18h NY, apos o fecho dos mercados tradicionais, o BTC frequentemente faz movimentos "surpresa" — o modelo aprendeu que este periodo tem comportamento diferente do resto do dia.

### #8 — is_us_afternoon (2.28%)

Tarde americana (14-17h NY). Periodo de menor volatilidade, frequentemente de consolidacao.

### #9 — dist_ema20_cl (2.27%)

Distancia do petroleo (CL) a sua EMA20.

**Exemplo grafico:** O petroleo esta 3% abaixo da EMA20 (`dist_ema20_cl = -3%`). O modelo interpreta que se o petroleo esta "esticado" para baixo, pode reverter, e o BTC pode ser afetado indiretamente.

### #10 — price_div_abs (2.22%)

`|ret1(BTC) x ret1(MNQ)|`. Magnitude da divergencia BTC vs MNQ.

---

## 3. CL (Petroleo)

### #1 — kz_range (5.70%)

O CL e a commodity com kill zones mais definidas porque os contratos futuros tem horarios de negociacao bem marcados.

### #2 — hour (4.41%)

Hora do dia. O CL tem a sazonalidade intraday mais forte dos 4 ativos.

**Exemplo grafico:** As 10h NY saem os relatorios de inventario da EIA (Energy Information Administration). O CL pode mover-se 1-2% em minutos. As 14h30 NY, o fecho do mercado de opcoes de petroleo tambem causa movimentos. O modelo aprendeu estes padroes horarios.

### #3 — is_london (4.10%)

Londres e um centro global de negociacao de commodities, especialmente petroleo (ICE Futures Europe). A sessao de Londres e crucial para o CL.

### #4 — is_evening (4.04%)

**Exemplo grafico:** As 18h NY, o CL frequentemente estende ou reverte o movimento do dia. Periodo de baixa liquidez onde stops sao "cacados".

### #5 — is_us_session (2.99%)

Sessao americana. O CL segue os horarios do NYMEX (9-17h NY).

### #6 — is_asia (2.91%)

**Exemplo grafico:** Durante a Asia, o CL pode mover-se com noticias da China (maior importador de petroleo). Se a China anuncia estimulos economicos as 4h UTC, o CL salta antes da abertura europeia.

### #7 — kz_dist_high (2.83%)

Distancia ao topo da kill zone atual. Similar ao MNQ.

### #8 — vol_cl (2.52%)

Volatilidade do proprio CL.

**Exemplo grafico:** Se o petroleo esta a mover-se 0.8% por hora (`vol_cl = 0.008`), o modelo sabe que esta num regime de alta volatilidade e ajusta as probabilidades. Em periodos de calma (`vol_cl < 0.003`), o modelo favorece NEUTRO.

### #9 — is_us_morning (2.52%)

Manha americana. Periodo de maior liquidez e noticias.

### #10 — cl_down_mnq_up (2.41%)

Gatilho binario: CL caindo (`ret1_cl < 0`) E MNQ subindo (`ret1_mnq > 0`) no mesmo candle.

**Exemplo grafico:**

```
Hora   | MNQ (var%) | CL (var%) | cl_down_mnq_up
-------|------------|-----------|---------------
10:00  | +0.15%     | -0.20%    | 1
```

Neste cenario, o Nasdaq sobe mas o petroleo cai — divergencia pura. O modelo CL aprendeu que quando o petroleo cai enquanto o Nasdaq sobe, ha 59.2% de probabilidade do CL reverter para LONG (edge de +21.8%). E a divergencia mais rentavel de todo o sistema.

---

## 4. MGC (Ouro)

### #1 — is_us_afternoon (5.30%)

**Exemplo grafico:** Saem dados de inflacao (CPI) as 9h30 NY. O ouro reage fortemente. Mas o modelo descobriu que a **tarde US (14-17h)** e ainda mais relevante que a manha para o MGC — periodo em que os grandes players ajustam posicoes e o ouro faz os movimentos mais significativos.

### #2 — is_london (4.42%)

Londres e o maior centro de negociacao de ouro do mundo (LBMA - London Bullion Market Association). O fixing do ouro ocorre duas vezes ao dia em Londres (10h30 e 15h Londres = 9h30 e 14h UTC).

**Exemplo grafico:** As 10h30 Londres (9h30 UTC), o preco do ouro e "fixado" pelos 5 maiores bancos do mundo (HSBC, JP Morgan, etc.). Este fixing frequentemente define a direcao do resto do dia. O modelo detetou que a sessao de Londres e a segunda feature mais importante para o MGC.

### #3 — hour (3.87%)

**Exemplo grafico:**

| Hora (NY) | O que acontece |
|-----------|----------------|
| 9h30 | Dados economicos US (CPI, empregos) |
| 10h30 | Fixing do ouro em Londres |
| 14h30 | Fecho do mercado de opcoes |
| 15h00 | Fixing da tarde em Londres |
| 17h00 | Fecho do mercado a vista |

O modelo aprendeu que cada hora tem um padrao de probabilidade diferente para o ouro.

### #4 — is_us_morning (3.87%)

Manha americana — noticias, dados economicos, abertura dos mercados.

### #5 — above_pmh (2.29%)

Preco atual **acima do maximo do mes anterior**. Unico modelo onde niveis mensais aparecem no top 10.

**Exemplo grafico:** Imagine que em Marco o MGC fez um topo em $2.150. Agora estamos em Abril e o preco esta em $2.175 — `above_pmh = 1` (esta acima do maximo de Marco).

Interpretacao do modelo: se o MGC esta **acima** do maximo do mes anterior:
- Pode continuar a subir (rompeu resistencia mensal = sinal de forca)
- Pode reverter (falso breakout, volta para dentro do range mensal)

O modelo pesa este nivel como a 5a feature mais importante — algo que nao acontece com MNQ nem BTC.

### #6 — kz_overlap (1.90%)

Sobreposicao London + NY (13-14h UTC = 8-9h NY).

**Exemplo grafico:** As 8h NY (13h UTC), Londres ainda esta ativa e NY acabou de abrir. E o periodo de maxima liquidez global — o volume de negociacao de ouro dispara. O modelo sabe que este overlap frequentemente precede movimentos direcionais.

### #7 — prev_day_range_pct (1.90%)

Amplitude percentual do dia anterior: `(high - low) / close`.

**Exemplo grafico:** Ontem o MGC fez um range de $2.100 a $2.150 (range de 2.4%). Hoje, se o range esta a ser menor, o modelo pode esperar expansao (continuacao do range de ontem). Se o range de ontem foi muito grande (>3%), o modelo pode esperar consolidacao.

### #8 — dist_to_pml (1.90%)

Distancia percentual do preco ate o **minimo do mes anterior**. Par complementar de `above_pmh`.

**Exemplo grafico:** O minimo do mes passado foi $2.080. O MGC esta agora em $2.100. `dist_to_pml = (2100-2080)/2080 = 0.96%` (o preco esta 0.96% acima do minimo mensal).

Se o preco estiver muito proximo do minimo mensal (`dist_to_pml < 0.5%`), o modelo sabe que esta num nivel de suporte importante.

### #9 — dow (1.89%)

Dia da semana.

**Exemplo grafico:** O ouro tem bias de compra as quartas-feiras (fecho do Comex) e bias de venda as sextas-feiras (realizacao de lucros antes do fim-de-semana). O modelo BTC tem um `dow` muito mais forte (4.76%), mas para o MGC tambem e relevante.

### #10 — bb_cl (1.88%)

Bollinger Band do petroleo (CL).

**Exemplo grafico:** As Bandas de Bollinger do CL estao a expandir-se — `bb_cl` esta alto (>0.05). O ouro frequentemente move-se em correlacao com o petroleo (ambos sao commodities). O modelo MGC aprendeu que a volatilidade do CL e um领先 indicador para o MGC.

---

## Divergencias — Tema Transversal

### price_div_abs (presente em MNQ #7, BTC #10)

`|ret1(AtivoA) x ret1(AtivoB)|`.

Mede a **magnitude** da divergencia de preco entre dois ativos no mesmo candle.

**Exemplo grafico MNQ-CL:**

```
Hora   | MNQ (preco) | MNQ (var%) | CL (preco) | CL (var%) | price_div_cl | price_div_abs
-------|-------------|------------|------------|-----------|--------------|---------------
10:00  | 19.000      | —          | 78,50      | —         | —            | —
11:00  | 19.050      | +0.26%     | 78,20      | -0.38%    | -0.0010      | 0.0010
```

`price_div_cl = 0.0026 x (-0.0038) = -0.00000988` (negativo = divergencia)
`price_div_abs = | -0.00000988 | = 0.00000988`

O valor absoluto (0.0010 = 0.1%) e pequeno, mas se o MNQ tivesse subido 0.5% e CL descido 0.6%, `price_div_abs = 0.0030 = 0.3%` — divergencia forte.

O modelo MNQ usa `price_div_abs` como feature #1 no modelo US-only porque aprendeu que **quando o Nasdaq e o petroleo divergem fortemente, um dos dois vai reverter na direcao do outro**.

### cl_down_mnq_up (CL #10)

Versao binaria da divergencia. 1 = CL cai E MNQ sobe.

**Edge mais forte do sistema: +21.8% LONG no CL.** Quando este gatilho ativa, o CL tem 59.2% de probabilidade de subir vs 37.4% de cair.

---

## Resumo: O que cada modelo "pergunta"

| Modelo | Pergunta principal |
|--------|--------------------|
| **MNQ** | "Estou ativo em que zona e qual a divergencia com o petroleo?" |
| **BTC** | "Que dia e que horas sao, e o Nasdaq esta volatil?" |
| **CL** | "Que horas sao, e o CL esta a divergir do MNQ?" |
| **MGC** | "E tarde US ou Londres, e estou acima/abaixo do mes passado?" |

---

*Gerado em 19/05/2026 a partir dos modelos ML treinados (all-hours, forward 4h).*

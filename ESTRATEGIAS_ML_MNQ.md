# Triplonq — Estratégias MNQ com Suporte de ML

> Documento gerado com base na análise estatística de ~14.000 candles horários
> de MNQ=F, BTC-USD e CL=F. Modelo: XGBoost treinado no período 2022–2025.
> Forward: 4 barras (≈ 4h). ROC-AUC out-of-sample: ~0.5477.

---

## 1. Janela Horária com Melhor Edge

O modelo foi treinado **exclusivamente na sessão americana** após análise estatística
provar que fora dela o win rate cai abaixo de 40%.

| Horário (BRT) | Dados (UTC/ET) | Win Rate LONG | Classificação |
|---------------|----------------|---------------|---------------|
| 12h – 14h     | 09h – 11h      | ~47%          | **PICO**      |
| 14h – 17h     | 11h – 14h      | ~44–46%       | ATIVA         |
| 21h – 22h     | 18h – 19h      | ~40%          | Sem edge      |
| Madrugada     | fora de mercado| ~33–38%       | Evitar        |

**Regra #1:** Só operar entre 12h e 17h BRT. Fora dessa janela o modelo não foi
treinado e os dados mostram desvantagem estatística.

---

## 2. Gatilho Principal — `us_prime_setup`

Condições simultâneas para o setup de maior qualidade:

```
CL e MNQ andando contra (price_div_cl < 0)
  +
ADX do MNQ acima de 14 (tendência ativa)
  +
Dentro da sessão americana (12h–17h BRT)
```

**Como identificar no dashboard:**
- Card "CL vs MNQ": mostra "CONTRA" quando `price_div_cl < 0`
- Card "ADX": mostra "ATIVO" quando ADX > 14
- Card "Sessão": mostra "PICO" ou "ATIVA"

Quando os três estão ativos ao mesmo tempo → `us_prime_setup = true` no ML.

---

## 3. Filtro de Tendência — MA50 Alignment

A posição dos 3 ativos em relação à MA50 define o regime de mercado:

| Score MA50 (0–3) | Regime          | Ação sugerida              |
|------------------|-----------------|----------------------------|
| 3 (todos acima)  | Bullish forte   | Priorizar LONGs             |
| 2                | Bullish parcial | LONGs com cautela          |
| 1                | Bearish parcial | Reduzir tamanho / pular    |
| 0 (todos abaixo) | Bearish forte   | Evitar LONGs               |

**Regra #2:** Com `ma50_alignment` = 0 ou 1, não entrar em LONG mesmo com
`us_prime_setup` ativo. O modelo pesa essa feature como uma das mais importantes.

O dashboard exibe o badge **MA50** em cada ativo (verde = acima, vermelho = abaixo).

---

## 4. Sinal ML — `prob_long`

O modelo retorna uma probabilidade de LONG nas próximas 4 horas.

| prob_long | Interpretação           | Ação                        |
|-----------|-------------------------|-----------------------------|
| > 0.65    | Alta convicção          | Entrada com tamanho normal  |
| 0.55–0.65 | Convicção moderada      | Entrada reduzida (50%)      |
| 0.45–0.55 | Neutro / incerto        | Aguardar confirmação        |
| < 0.45    | Viés BAIXO              | Não operar LONG             |

**Importante:** O modelo tem AUC ~0.55, o que é útil como filtro mas não é
infalível. Nunca usar `prob_long` sozinho — sempre combinado com os gatilhos
abaixo.

---

## 5. Setup Completo (Alta Qualidade)

Para considerar uma entrada de **melhor qualidade**, os seguintes critérios
devem estar presentes:

```
[ ] Horário: 12h–14h BRT (pico)
[ ] CL e MNQ andando CONTRA (price_div_cl < 0)
[ ] ADX MNQ > 14
[ ] MA50 Alignment >= 2 (2 ou 3 ativos acima da MA50)
[ ] prob_long >= 0.55
[ ] BTC RSI < 65 (sem sobrecompra severa no BTC)
```

Setup com 5 ou 6 checks → entrada de alta prioridade.
Setup com 3–4 checks → entrada possível com tamanho reduzido.
Menos de 3 → aguardar.

---

## 6. Filtros de Qualidade (BTC Derivatives)

Os dados da Binance Futures adicionam contexto sobre o posicionamento
do mercado em BTC, que tem alta correlação com MNQ:

| Indicador        | Bullish para LONG MNQ | Bearish / Evitar    |
|------------------|-----------------------|---------------------|
| LSR (Long/Short) | > 1.1 (longs dominam) | < 0.9               |
| OI Delta%        | > 0% (OI crescendo)   | < -1% (liquidação)  |
| Taker Buy/Sell   | > 1.0 (buy pressure)  | < 0.9               |
| CVD Delta%       | > 0% (buyers ativos)  | < -5%               |

O **Expansion Score** no dashboard combina esses 4 fatores em um score único
de 0 a 100. Score > 60 favorece LONGs em MNQ.

---

## 7. O que o Modelo NÃO Aprendeu / Evitar

Com base na análise estatística:

- **Sessão 21h BRT**: win rate ~40%, abaixo da base. O `prime_setup` (21h + div + ADX)
  mostrou 45% vs base de 47% — sem edge real. Evitar trades nesse horário.

- **Triple Signal sozinho** (RSI BTC < 45 + div CL + div BTC): era a estratégia original
  mas mostrou edge fraco sem os demais filtros.

- **ADX muito alto (> 25)**: mercado em tendência forte pode reverter menos.
  O modelo trata isso como feature separada — não necessariamente melhor.

- **Fora da sessão americana**: o modelo não foi treinado nesses horários.
  Qualquer sinal gerado à madrugada ou tarde da noite não tem suporte estatístico.

---

## 8. Workflow Diário Sugerido

```
08h45 BRT — Verificar MA50 alignment dos 3 ativos
           — Ver BTC Derivatives (OI, LSR, taker)
           — Verificar nível de suporte/resistência mais próximo

12h00 BRT — Sessão começa. Observar se us_prime_setup está ativo
           — Rodar /api/ml/predict para pegar prob_long atualizado

12h–14h   — Janela principal. Entrar apenas com setup completo (≥5 checks)

14h–17h   — Segunda janela. Tamanho reduzido, exige prob_long ≥ 0.60

17h00     — Encerrar posições abertas. Fora da janela treinada.
```

---

## 9. Retreinar o Modelo

O modelo deve ser retreinado periodicamente para capturar regime atual:

```bash
# Na pasta ml/:
python collect_data.py       # atualiza o banco de dados
python train.py --forward 4  # retreina focado na sessão US
```

Retreinar quando:
- Passaram mais de 30 dias desde o último treino
- O mercado passou por evento macro relevante (Fed, inflação, etc.)
- O win rate percebido nas entradas caiu abaixo de 42% por 2 semanas seguidas

---

*Gerado em 2026-05-16. Baseado em análise de 14.365 candles horários.*
*Este documento é para referência pessoal. Não é recomendação financeira.*

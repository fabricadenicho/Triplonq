# ML Key Zones

Key Zones são níveis de suporte/resistência baseados em _swing points_ de timeframes maiores (diário, semanal, mensal e range de segunda-feira). No modelo ML sem viés, estas features representaram **~30% da importância total (Gain)** — a categoria mais relevante entre todos os 4 ativos (MNQ, BTC, CL, MGC).

## 📐 Como são calculadas

```python
# Para cada barra, calcula-se a distância % do preço atual até o nível
# e um flag binário indicando se o preço está acima ou abaixo do nível.
```

### Timeframes usados

| Timeframe | Níveis | Features geradas |
|-----------|--------|-----------------|
| **Diário** | High, Low, Open do dia anterior | `dist_to_pdh`, `dist_to_pdl`, `dist_to_do`, `above_pdh`, `above_pdl`, `above_do`, `prev_day_range_pct` |
| **Semanal** | High, Low, Open da semana anterior | `dist_to_pwh`, `dist_to_pwl`, `dist_to_wo`, `above_pwh`, `above_pwl`, `above_wo` |
| **Mensal** | High, Low, Open do mês anterior | `dist_to_pmh`, `dist_to_pml`, `dist_to_mo`, `above_pmh`, `above_pml`, `above_mo` |
| **Monday Range** | High, Low dos candles de segunda (≥4 semanas) | `dist_to_mday_h`, `dist_to_mday_l`, `above_mday_h`, `above_mday_l` |

### Prefixos das features

| Prefixo | Significado |
|---------|-------------|
| `pd` | Previous Day |
| `pw` | Previous Week |
| `pm` | Previous Month |
| `mday` | Monday Day range |
| `dist_to_` | Distância percentual do close até o nível |
| `above_` | Binário: 1 se close > nível |

## 🏆 Performance nos modelos

### Importância por categoria nos 4 ativos

| Categoria | MNQ | BTC | CL | MGC | Média |
|-----------|:---:|:---:|:--:|:---:|:----:|
| **Key Zones** | **32.2%** | **33.5%** | **32.0%** | **29.3%** | **31.8%** |
| Retornos/Vol | 30.7% | 30.0% | 28.7% | 30.1% | 29.9% |
| Tempo | 12.9% | 13.5% | 14.9% | 14.7% | 14.0% |
| Médias | 8.2% | 10.7% | 9.1% | 9.1% | 9.3% |

### Top Key Zone features (média entre ativos)

| Feature | Descrição | Gain médio |
|---------|-----------|:----------:|
| `above_pmh` | Acima do high do mês anterior | 2.0% |
| `dist_to_pmh` | Distância do high mensal | 1.9% |
| `dist_to_pwh` | Distância do high semanal | 1.8% |
| `above_mday_h/l` | Acima do range de Monday | 1.8% |
| `dist_to_mday_h` | Distância do high Monday range | 1.8% |
| `dist_to_mo` | Distância da abertura mensal | 1.7% |
| `above_pml` | Acima do low do mês anterior | 1.7% |

## 🧠 Por que Key Zones funcionam no ML

1. **Auto-correção de preço**: O mercado tende a reagir em níveis psicológicos (swing highs/lows de timeframes maiores).
2. **Quebras (breakouts)**: Quando o preço rompe um nível, a volatilidade e direção tendem a se acelerar.
3. **Naturais do preço**: Diferente de indicadores como RSI ou ADX (que são derivados), Key Zones são baseadas em ações concretas de preço — o modelo confia mais nelas.
4. **Funcionam em qualquer ativo**: A importância é consistente entre MNQ, BTC, CL e MGC (~30% em todos).

## 🔗 Uso no código

### Treino (`ml/teste/train.py`)

```python
KEY_LEVEL_FEATURES = [
    'dist_to_pdh', 'dist_to_pdl', 'dist_to_do',
    'above_do', 'above_pdh', 'above_pdl', 'prev_day_range_pct',
    'dist_to_pwh', 'dist_to_pwl', 'dist_to_wo',
    'above_wo', 'above_pwh', 'above_pwl',
    'dist_to_pmh', 'dist_to_pml', 'dist_to_mo',
    'above_mo', 'above_pmh', 'above_pml',
    'dist_to_mday_h', 'dist_to_mday_l',
    'above_mday_h', 'above_mday_l',
]
```

### Função `add_key_levels()`

A função reamostra o OHLC do ativo principal para 1D, 1W (W-SUN) e 1M (MS), obtém o valor anterior do high/low/open, e calcula:

```python
dist = (close - nivel) / nivel * 100     # dist_to_*
above = int(close > nivel)                # above_*
```

Para Monday range, usa apenas candles de segunda-feira das últimas ≥4 semanas.

## ⚠️ Observações

- Key Zones mensais (`dist_to_pmh`, `above_pmh`) têm ligeiramente mais peso que diários/semanais.
- `prev_day_range_pct` (range do dia anterior em %) captura expansão/contração — útil para prever volatilidade futura.
- O modelo novo (sem viés) deu **mais peso a Key Zones** que o modelo antigo (com viés), sugerindo que os vieses de sessão/kill zone estavam atrapalhando o modelo de enxergar esses níveis.

# Arvores de Decisao — ML Prop Firm

> Arvore 0 de cada modelo XGBoost (500 arvores cada)
> Leaf > 0 = favorece LONG | Leaf < 0 = favorece SHORT
> Fonte: SQLite clean (5 anos) | Metodologia: walk-forward + blind OOS genuino
> Treinado em: 2026-05-22

---

## MNQ

- **Secundarios:** BTC (sec1), CL (sec2)
- **AUC Blind OOS 2025:** 0.5736 | amostras: 5409
- **AUC Live 2026:** 0.5118 | amostras: 1748
- **Features ativas:** 60

```
0:[sma50_alignment<1] yes=1,no=2,missing=2
	1:[vol_spread_p_2<-0.0011347062] yes=3,no=4,missing=4
		3:[vol_spread_p_2<-0.0017894489] yes=7,no=8,missing=8
			7:[div_1<0.691191137] yes=15,no=16,missing=16
				15:leaf=0.0226255246
				16:leaf=-0.00974853989
			8:[adx_1<19.2446384] yes=17,no=18,missing=18
				17:leaf=0.0151304211
				18:leaf=0.0372162983
		4:[bb_spread_p_1<-0.0143644176] yes=9,no=10,missing=10
			9:[vol_spread_p_1<-0.0266405176] yes=19,no=20,missing=20
				19:leaf=0.0133740567
				20:leaf=-0.0120190633
			10:[prev_day_range_pct<3.50350881] yes=21,no=22,missing=22
				21:leaf=0.0086185718
				22:leaf=0.0406658351
	2:[adx_1<25.3683586] yes=5,no=6,missing=6
		5:[ret1_spread_p_1<0.0383349322] yes=11,no=12,missing=12
			11:[dist_to_mo<7.49921656] yes=23,no=24,missing=24
				23:leaf=-0.00629906449
				24:leaf=0.0115222894
			12:[dist_to_pdl<0.535706758] yes=25,no=26,missing=26
				25:leaf=0.00495879492
				26:leaf=0.0400566533
		6:[prev_day_range_pct<3.50350881] yes=13,no=14,missing=14
			13:[bb_spread_p_1<-0.15010342] yes=27,no=28,missing=28
				27:leaf=0.00958091579
				28:leaf=-0.000488349004
			14:[di_spread_1<2.16412997] yes=29,no=30,missing=30
				29:leaf=0.0335519388
				30:leaf=0.00789047405
```

### Importancia por Categoria

| Categoria | Peso |
|-----------|:----:|
| KEY LEVELS | 29.1% |
| VOLATILIDADE | 16.5% |
| ADX/DI | 15.2% |
| RETORNOS | 15.1% |
| RSI | 9.6% |
| TEMPORAL | 7.4% |
| MEDIAS | 6.4% |

### Top 10 Features

| Peso | Feature | Significado |
|:----:|---------|-------------|
| 4.3% | `prev_day_range_pct` | Range % do dia anterior |
| 3.8% | `adx_p` | Forca tendencia MNQ |
| 3.8% | `adx_1` | Forca tendencia BTC |
| 3.8% | `adx_2` | Forca tendencia CL |
| 3.7% | `vol_p` | Volatilidade MNQ |
| 3.3% | `dist_to_pmh` | Distancia ao topo do mes passado |
| 3.2% | `dist_to_mo` | Distancia ao open mensal |
| 3.1% | `hour_cos` | Hora do dia (ciclico) |
| 3.0% | `bb_p` | BB Width MNQ |
| 3.0% | `ret4_1` | Retorno 4h BTC |

---

## BTC

- **Secundarios:** MNQ (sec1), CL (sec2)
- **AUC Blind OOS 2025:** 0.5385 | amostras: 8448
- **AUC Live 2026:** 0.5370 | amostras: 2834
- **Features ativas:** 64

```
0:[dow_sin<-0.781831503] yes=1,no=2,missing=2
	1:[bb_spread_p_2<-0.0199918821] yes=3,no=4,missing=4
		3:leaf=0.0192591082
		4:[dist_to_do<0.98186022] yes=7,no=8,missing=8
			7:[rsi_p<54.8954887] yes=13,no=14,missing=14
				13:leaf=-0.0158361308
				14:leaf=-0.00560825039
			8:[vol_spread_p_2<0.00040306116] yes=15,no=16,missing=16
				15:leaf=0.0224515218
				16:leaf=-0.00681905169
	2:[ret1_2<0.00134771632] yes=5,no=6,missing=6
		5:[hour<13] yes=9,no=10,missing=10
			9:[dist_to_wo<15.2295713] yes=17,no=18,missing=18
				17:leaf=0.0022518679
				18:leaf=0.0202752668
			10:[price_div_abs<1.45577758e-07] yes=19,no=20,missing=20
				19:leaf=0.0137353502
				20:leaf=-0.00424647238
		6:[bb_spread_p_2<-0.0109631466] yes=11,no=12,missing=12
			11:[div_1<-11.3626394] yes=21,no=22,missing=22
				21:leaf=0.00497123506
				22:leaf=0.0268560145
			12:[di_spread_1<26.8002625] yes=23,no=24,missing=24
				23:leaf=0.00360113918
				24:leaf=0.0212114621
```

### Importancia por Categoria

| Categoria | Peso |
|-----------|:----:|
| KEY LEVELS | 28.0% |
| RETORNOS | 18.5% |
| VOLATILIDADE | 15.4% |
| ADX/DI | 14.7% |
| RSI | 8.9% |
| MEDIAS | 7.0% |
| TEMPORAL | 6.4% |

### Top 10 Features

| Peso | Feature | Significado |
|:----:|---------|-------------|
| 3.6% | `prev_day_range_pct` | Range % do dia anterior |
| 3.6% | `adx_p` | Forca tendencia BTC |
| 3.3% | `sma50_slope_p` | Inclinacao SMA50 BTC |
| 2.9% | `adx_2` | Forca tendencia CL |
| 2.8% | `dist_to_pwl` | Distancia ao fundo semanal |
| 2.8% | `vol_spread_p_2` | Diferenca vol BTC-CL |
| 2.8% | `dist_to_pdl` | Distancia ao fundo de ontem |
| 2.8% | `adx_1` | Forca tendencia MNQ |
| 2.7% | `dist_to_pmh` | Distancia ao topo do mes passado |
| 2.7% | `bb_spread_p_2` | Diferenca BB Width BTC-CL |

---

## CL

- **Secundarios:** MNQ (sec1), BTC (sec2)
- **AUC Blind OOS 2025:** 0.5419 | amostras: 5265
- **AUC Live 2026:** 0.5521 | amostras: 1752
- **Features ativas:** 63

```
0:[dist_to_pwh<2.97529078] yes=1,no=2,missing=2
	1:[adx_1<33.7999649] yes=3,no=4,missing=4
		3:[dist_to_pdl<0.461420953] yes=7,no=8,missing=8
			7:[vol_spread_p_1<0.00201741769] yes=15,no=16,missing=16
				15:leaf=-0.00240795501
				16:leaf=0.0101746302
			8:[hour<15] yes=17,no=18,missing=18
				17:leaf=-0.00163230905
				18:leaf=-0.0110637788
		4:[ret4_2<0.0344365686] yes=9,no=10,missing=10
			9:[ret8_p<-0.0125358524] yes=19,no=20,missing=20
				19:leaf=-0.00875034649
				20:leaf=0.00974166766
			10:[ema20_bias_p_1<2] yes=21,no=22,missing=22
				21:leaf=-0.0169734117
				22:leaf=0.00141050667
	2:[vol_spread_p_1<-0.00045457852] yes=5,no=6,missing=6
		5:[dist_to_do<-0.284120739] yes=11,no=12,missing=12
			11:leaf=-0.02017444
			12:[bb_spread_p_1<-0.00170970801] yes=23,no=24,missing=24
				23:leaf=-0.00274423114
				24:leaf=0.0247065313
		6:[vol_p<0.00614120625] yes=13,no=14,missing=14
			13:[dist_to_do<0.217593506] yes=25,no=26,missing=26
				25:leaf=0.0370231718
				26:leaf=0.0150996903
			14:leaf=-0.0055800858
```

### Importancia por Categoria

| Categoria | Peso |
|-----------|:----:|
| KEY LEVELS | 26.3% |
| RETORNOS | 19.5% |
| ADX/DI | 17.2% |
| VOLATILIDADE | 16.2% |
| RSI | 7.7% |
| TEMPORAL | 6.4% |
| MEDIAS | 5.3% |

### Top 10 Features

| Peso | Feature | Significado |
|:----:|---------|-------------|
| 4.8% | `adx_p` | Forca tendencia CL |
| 3.7% | `adx_1` | Forca tendencia MNQ |
| 3.2% | `adx_2` | Forca tendencia BTC |
| 3.1% | `vol_p` | Volatilidade CL |
| 2.9% | `dist_to_pml` | Distancia ao fundo do mes passado |
| 2.8% | `bb_spread_p_1` | Diferenca BB Width CL-MNQ |
| 2.8% | `bb_spread_p_2` | Diferenca BB Width CL-BTC |
| 2.7% | `bb_p` | BB Width CL |
| 2.6% | `ret4_2` | Retorno 4h BTC |
| 2.6% | `vol_spread_p_1` | Diferenca vol CL-MNQ |

---

## ES

- **Secundarios:** MNQ (sec1), BTC (sec2)
- **AUC Blind OOS 2025:** 0.5702 | amostras: 5482
- **AUC Live 2026:** 0.5548 | amostras: 2142
- **Features ativas:** 59

```
0:[dist_to_mday_h<-1.13404512] yes=1,no=2,missing=2
	1:[di_spread_2<-4.44157696] yes=3,no=4,missing=4
		3:[bb_p<0.00562281488] yes=7,no=8,missing=8
			7:[adx_1<30.1767521] yes=15,no=16,missing=16
				15:leaf=0.0289659835
				16:leaf=0.00749496697
			8:[bb_spread_p_1<-0.00531661278] yes=17,no=18,missing=18
				17:leaf=0.0212075207
				18:leaf=0.00056606991
		4:[bb_spread_p_2<-0.00618956983] yes=9,no=10,missing=10
			9:[dist_to_wo<-1.26119363] yes=19,no=20,missing=20
				19:leaf=-0.00788844191
				20:leaf=0.011436563
			10:[div_1<-0.835395455] yes=21,no=22,missing=22
				21:leaf=-0.00636804942
				22:leaf=0.0312473271
	2:[ret1_2<0.00788710546] yes=5,no=6,missing=6
		5:[hour<16] yes=11,no=12,missing=12
			11:[prev_day_range_pct<0.875640869] yes=23,no=24,missing=24
				23:leaf=0.00611825753
				24:leaf=-0.00215529255
			12:[di_spread_1<-3.88246679] yes=25,no=26,missing=26
				25:leaf=0.00183224818
				26:leaf=-0.0110135134
		6:[di_spread_2<7.96054506] yes=13,no=14,missing=14
			13:[prev_day_range_pct<0.8604123] yes=27,no=28,missing=28
				27:leaf=-0.00856744871
				28:leaf=-0.0196932238
			14:[di_spread_p<-5.80791903] yes=29,no=30,missing=30
				29:leaf=-0.012815618
				30:leaf=0.00293066306
```

### Importancia por Categoria

| Categoria | Peso |
|-----------|:----:|
| KEY LEVELS | 29.4% |
| VOLATILIDADE | 18.3% |
| ADX/DI | 15.8% |
| RETORNOS | 12.3% |
| RSI | 9.7% |
| TEMPORAL | 7.6% |
| MEDIAS | 6.3% |

### Top 10 Features

| Peso | Feature | Significado |
|:----:|---------|-------------|
| 4.6% | `adx_1` | Forca tendencia MNQ |
| 4.3% | `vol_p` | Volatilidade ES |
| 4.2% | `prev_day_range_pct` | Range % do dia anterior |
| 3.8% | `ret4_2` | Retorno 4h BTC |
| 3.6% | `vol_spread_p_1` | Diferenca vol ES-MNQ |
| 3.4% | `dist_to_mo` | Distancia ao open mensal |
| 3.3% | `div_1` | DIV RSI ES-MNQ |
| 3.1% | `hour_cos` | Hora do dia (ciclico) |
| 3.0% | `dist_to_pmh` | Distancia ao topo do mes passado |
| 3.0% | `adx_p` | Forca tendencia ES |

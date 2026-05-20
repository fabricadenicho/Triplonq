# Arvores de Decisao — ML Prop Firm

> Arvore 0 de cada modelo XGBoost (500 arvores cada)
> Leaf > 0 = favorece LONG | Leaf < 0 = favorece SHORT

## MNQ- **Arvore:** 0 de 500 arvores- **Features:** 67 (conjunto otimizado)```0:[vol_p<0.00334275188] yes=1,no=2,missing=2,gain=50.3000183,cover=3387.51758
	1:[sma50_alignment<1] yes=3,no=4,missing=4,gain=29.2431374,cover=2735.79443
		3:[div_2<-5.68693924] yes=7,no=8,missing=8,gain=14.6849403,cover=258.749603
			7:[div_1<-4.3718791] yes=15,no=16,missing=16,gain=9.57003403,cover=86.1638641
				15:leaf=0.0103913611,cover=58.2701683
				16:leaf=0.0318203829,cover=27.8936939
			8:[vol_spread_p_1<-0.0264883246] yes=17,no=18,missing=18,gain=15.7657576,cover=172.585754
				17:leaf=0.0256033018,cover=22.0363541
				18:leaf=-0.00104032503,cover=150.549393
		4:[hour<5] yes=9,no=10,missing=10,gain=14.9054508,cover=2477.04492
			9:[div_1<-27.3629742] yes=19,no=20,missing=20,gain=9.04796219,cover=598.843811
				19:leaf=0.0108984895,cover=23.239872
				20:leaf=-0.00786961894,cover=575.603943
			10:[hour<19] yes=21,no=22,missing=22,gain=17.8894997,cover=1878.20105
				21:leaf=-6.79174263e-05,cover=1437.35303
				22:leaf=-0.00697018486,cover=440.848114
	2:[di_spread_2<-24.4214706] yes=5,no=6,missing=6,gain=21.764534,cover=651.723083
		5:[ret1_prod_p_1<-2.03484751e-06] yes=11,no=12,missing=12,gain=1.10151672,cover=30.5147457
			11:leaf=0.0182808582,cover=8.20352459
			12:leaf=0.0355146378,cover=22.3112221
		6:[vol_p<0.0139131229] yes=13,no=14,missing=14,gain=18.320467,cover=621.208313
			13:[dist_to_mo<7.74228573] yes=23,no=24,missing=24,gain=11.6689091,cover=553.794495
				23:leaf=0.00332928286,cover=533.658142
				24:leaf=0.0260752626,cover=20.1363335
			14:[vol_spread_p_1<-0.0142395133] yes=25,no=26,missing=26,gain=14.5529938,cover=67.4138565
				25:leaf=0.00155633304,cover=23.6275406
				26:leaf=0.0307246353,cover=43.7863159```### Importancia por Categoria| Categoria | Peso ||-----------|:----:|| KEY LEVELS | 33.9% || RETORNOS | 18.0% || VOLATILIDADE | 11.3% || TEMPORAL | 10.6% || ADX/DI | 8.8% || MEDIAS | 6.8% || OUTROS | 5.8% || RSI | 4.8% |## BTC- **Arvore:** 0 de 500 arvores- **Features:** 67 (conjunto otimizado)```0:[dow_sin<-0.781831503] yes=1,no=2,missing=2,gain=76.6027527,cover=4537.01611
	1:[ret4_1<-0.00907359272] yes=3,no=4,missing=4,gain=18.4377556,cover=853.173462
		3:[vol_spread_p_1<0.00282892445] yes=7,no=8,missing=8,gain=9.35080719,cover=34.9480019
			7:leaf=0.0297008362,cover=15.7243624
			8:[rsi_p<40.164917] yes=15,no=16,missing=16,gain=4.14765596,cover=19.2236404
				15:leaf=-0.0139859756,cover=9.74209213
				16:leaf=0.0125431623,cover=9.48154736
		4:[rsi_p<60.6416664] yes=9,no=10,missing=10,gain=18.5318756,cover=818.225464
			9:[ret4_2<-0.00934578013] yes=17,no=18,missing=18,gain=11.5354462,cover=758.218628
				17:leaf=0.00690944446,cover=34.1564026
				18:leaf=-0.010770577,cover=724.062256
			10:[vol_p<0.00260156463] yes=19,no=20,missing=20,gain=8.84916401,cover=60.0068474
				19:leaf=0.0287486166,cover=12.4467669
				20:leaf=0.00115857075,cover=47.5600777
	2:[bb_p<0.0481697433] yes=5,no=6,missing=6,gain=23.6378765,cover=3683.84253
		5:[adx_1<28.5282383] yes=11,no=12,missing=12,gain=19.6162109,cover=3424.79028
			11:[dist_to_mday_h<10.402647] yes=21,no=22,missing=22,gain=17.583744,cover=2314.53101
				21:leaf=-0.000386653846,cover=2276.53564
				22:leaf=0.019930182,cover=37.9955177
			12:[vol_spread_p_1<-0.000774841988] yes=23,no=24,missing=24,gain=16.818182,cover=1110.25916
				23:leaf=0.0146631496,cover=135.685516
				24:leaf=0.00341831474,cover=974.573669
		6:[di_spread_1<-14.2664375] yes=13,no=14,missing=14,gain=14.277874,cover=259.052307
			13:[bb_p<0.0951605737] yes=25,no=26,missing=26,gain=6.98229456,cover=45.5860748
				25:leaf=0.0085285306,cover=20.4685612
				26:leaf=-0.0145976599,cover=25.1175137
			14:[div_1<-18.7578754] yes=27,no=28,missing=28,gain=12.0431633,cover=213.466217
				27:leaf=-0.0111037428,cover=15.0935125
				28:leaf=0.0161221251,cover=198.372711```### Importancia por Categoria| Categoria | Peso ||-----------|:----:|| KEY LEVELS | 32.5% || RETORNOS | 18.8% || TEMPORAL | 14.5% || VOLATILIDADE | 10.8% || ADX/DI | 8.2% || OUTROS | 6.0% || MEDIAS | 5.1% || RSI | 4.1% |## CL- **Arvore:** 0 de 500 arvores- **Features:** 67 (conjunto otimizado)```0:[hour<14] yes=1,no=2,missing=2,gain=44.0446968,cover=3302.97485
	1:[vol_p<0.0040207603] yes=3,no=4,missing=4,gain=26.99333,cover=1918.40552
		3:[prev_day_range_pct<0.543941081] yes=7,no=8,missing=8,gain=16.7744236,cover=1196.03796
			7:[di_spread_1<7.22549963] yes=15,no=16,missing=16,gain=5.11898708,cover=94.7191544
				15:leaf=-0.0162701812,cover=68.5438156
				16:leaf=-0.000678863376,cover=26.1753426
			8:[hour<8] yes=17,no=18,missing=18,gain=13.4710817,cover=1101.31885
				17:leaf=-0.00187513675,cover=614.156677
				18:leaf=0.004799366,cover=487.16217
		4:[dist_to_pwh<-9.22645283] yes=9,no=10,missing=10,gain=20.0502892,cover=722.367493
			9:[prev_day_range_pct<7.91681004] yes=19,no=20,missing=20,gain=13.7191496,cover=157.637451
				19:leaf=-0.00515525788,cover=141.072983
				20:leaf=0.0229543205,cover=16.5644569
			10:[bb_spread_p_2<-0.0696618035] yes=21,no=22,missing=22,gain=13.560936,cover=564.730042
				21:leaf=-0.00375250238,cover=57.4095078
				22:leaf=0.0115730623,cover=507.320526
	2:[dist_to_mo<8.0661726] yes=5,no=6,missing=6,gain=17.8157043,cover=1384.56934
		5:[dist_to_pdl<0.123097166] yes=11,no=12,missing=12,gain=9.86322784,cover=1329.20605
			11:[bb_spread_p_2<-0.0827257633] yes=23,no=24,missing=24,gain=14.7355766,cover=368.091064
				23:leaf=0.0268552843,cover=15.6385603
				24:leaf=-0.00203452073,cover=352.452515
			12:[dist_to_mday_l<3.97282887] yes=25,no=26,missing=26,gain=15.1471519,cover=961.11499
				25:leaf=-0.00893971976,cover=678.841797
				26:leaf=-0.000669952424,cover=282.273163
		6:[dist_to_pwh<1.89107418] yes=13,no=14,missing=14,gain=12.0473633,cover=55.363308
			13:leaf=0.0408340879,cover=9.80431652
			14:[adx_1<16.1848354] yes=27,no=28,missing=28,gain=9.13191223,cover=45.5589943
				27:leaf=-0.0184448808,cover=10.1754036
				28:leaf=0.0126146954,cover=35.3835907```### Importancia por Categoria| Categoria | Peso ||-----------|:----:|| KEY LEVELS | 37.2% || RETORNOS | 17.8% || TEMPORAL | 12.9% || VOLATILIDADE | 9.0% || ADX/DI | 8.5% || OUTROS | 5.3% || MEDIAS | 5.2% || RSI | 4.1% |## MGC- **Arvore:** 0 de 500 arvores- **Features:** 67 (conjunto otimizado)```0:[vol_p<0.00255768211] yes=1,no=2,missing=2,gain=29.1372433,cover=3364.78296
	1:[dist_to_mo<0.49292165] yes=3,no=4,missing=4,gain=19.8471947,cover=2656.27026
		3:[dist_to_mday_l<-2.24054527] yes=7,no=8,missing=8,gain=15.3730965,cover=925.909912
			7:[di_spread_1<-5.48780489] yes=15,no=16,missing=16,gain=12.6243553,cover=52.8197861
				15:leaf=-0.00486007193,cover=24.8549995
				16:leaf=0.0241202656,cover=27.9647884
			8:[dist_to_pdl<0.488302141] yes=17,no=18,missing=18,gain=13.991375,cover=873.090149
				17:leaf=-0.0100589218,cover=395.055908
				18:leaf=-0.00242737518,cover=478.03421
		4:[di_spread_2<-28.1122017] yes=9,no=10,missing=10,gain=16.4730949,cover=1730.36023
			9:[dist_to_mo<2.39691472] yes=19,no=20,missing=20,gain=7.25707817,cover=39.659668
				19:leaf=0.00750410696,cover=22.3291874
				20:leaf=0.0334265046,cover=17.3304825
			10:[hour<13] yes=21,no=22,missing=22,gain=13.5028067,cover=1690.70056
				21:leaf=0.00246184506,cover=934.611694
				22:leaf=-0.0029270798,cover=756.088928
	2:[dist_to_mday_l<3.70069385] yes=5,no=6,missing=6,gain=24.2476387,cover=708.512695
		5:[bb_p<0.00661103614] yes=11,no=12,missing=12,gain=14.1232738,cover=555.411682
			11:[prev_day_range_pct<0.977408409] yes=23,no=24,missing=24,gain=3.42702293,cover=68.6638489
				23:leaf=0.00159282051,cover=16.8054047
				24:leaf=-0.0138717331,cover=51.8584442
			12:[rsi_p<59.8768272] yes=25,no=26,missing=26,gain=12.4622278,cover=486.747864
				25:leaf=0.00645037368,cover=406.956757
				26:leaf=-0.0064568813,cover=79.7911072
		6:[prev_day_range_pct<1.15384614] yes=13,no=14,missing=14,gain=14.9358406,cover=153.100998
			13:[div_2<0.82593447] yes=27,no=28,missing=28,gain=10.1805058,cover=44.4039307
				27:leaf=-0.0138867609,cover=20.4891453
				28:leaf=0.0142959449,cover=23.9147835
			14:[dist_to_wo<3.62233067] yes=29,no=30,missing=30,gain=6.98442078,cover=108.697075
				29:leaf=0.0143671306,cover=57.6785965
				30:leaf=0.0300273113,cover=51.0184784```### Importancia por Categoria| Categoria | Peso ||-----------|:----:|| KEY LEVELS | 36.6% || RETORNOS | 16.5% || VOLATILIDADE | 11.0% || TEMPORAL | 10.2% || ADX/DI | 9.6% || MEDIAS | 5.8% || OUTROS | 5.6% || RSI | 4.7% |
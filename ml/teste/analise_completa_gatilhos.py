"""
ANALISE COMPLETA DE GATILHOS — MNQ vs CL, ES, BTC
==================================================
Objetivo: entender TODAS as variaveis que influenciam quando
MNQ sobe e CL cai simultaneamente, incluindo:
- ES (S&P) e BTC como ativos de correlacao
- Key levels: abertura 1h, 4h, daily, weekly (qual ativo acima/abaixo)
- Spreads, divergencias RSI, alinhamentos MA
- Pesos e scoring para gerar edge operacional

Uso: python analise_completa_gatilhos.py
"""

import sqlite3, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import ta
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent

def load_sym(conn, sym):
    df = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts",
        conn, params=(sym,), parse_dates=['ts'], index_col='ts')
    return df

def compute_full(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
    adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=17)
    df['adx'] = adx_i.adx()
    df['pdi'] = adx_i.adx_pos()
    df['mdi'] = adx_i.adx_neg()
    df['ret1'] = df['close'].pct_change(1)
    df['ret4'] = df['close'].pct_change(4)
    df['ret8'] = df['close'].pct_change(8)
    df['vol'] = df['ret1'].rolling(20).std()
    df['bb_w'] = df['close'].rolling(20).std() * 2 / df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['dist_sma50'] = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['above_sma50'] = (df['close'] > df['sma50']).astype(int)
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['dist_ema20'] = (df['close'] - df['ema20']) / df['ema20'] * 100
    df['above_ema20'] = (df['close'] > df['ema20']).astype(int)
    return df

def resample_ohlc(series, rule):
    """Retorna DataFrame com open, high, low da serie em timeframe especifico."""
    s = series.copy()
    o = s.resample(rule).first()
    h = s.resample(rule).max()
    l = s.resample(rule).min()
    return o, h, l

def main():
    print("=" * 90)
    print("  ANALISE COMPLETA DE GATILHOS — 'MNQ SOBE + CL CAI'")
    print("=" * 90)

    # ── 1. CARREGAR DADOS ──
    print("\n[1] Carregando dados...")

    conn_main = sqlite3.connect(BASE / 'data.db')
    mnq_raw = load_sym(conn_main, 'mnq')
    btc_raw = load_sym(conn_main, 'btc')
    cl_raw  = load_sym(conn_main, 'cl')
    conn_main.close()

    conn_es = sqlite3.connect(BASE / 'es' / 'data.db')
    es_raw = load_sym(conn_es, 'es')
    conn_es.close()

    print(f"  MNQ: {mnq_raw.index[0].date()} -> {mnq_raw.index[-1].date()} ({len(mnq_raw)} candles)")
    print(f"  ES:  {es_raw.index[0].date()} -> {es_raw.index[-1].date()} ({len(es_raw)} candles)")
    print(f"  BTC: {btc_raw.index[0].date()} -> {btc_raw.index[-1].date()} ({len(btc_raw)} candles)")
    print(f"  CL:  {cl_raw.index[0].date()} -> {cl_raw.index[-1].date()} ({len(cl_raw)} candles)")

    # ── 2. COMPUTAR INDICADORES ──
    print("\n[2] Computando indicadores...")
    mnq = compute_full(mnq_raw)
    es  = compute_full(es_raw)
    btc = compute_full(btc_raw)
    cl  = compute_full(cl_raw)

    # Sincronizar indices
    idx = mnq.index.intersection(es.index).intersection(btc.index).intersection(cl.index)
    mnq = mnq.loc[idx]
    es   = es.loc[idx]
    btc = btc.loc[idx]
    cl  = cl.loc[idx]

    print(f"  Candles sincronizados: {len(idx)}")
    print(f"  Periodo: {idx[0].date()} -> {idx[-1].date()}")

    # ── 3. MONTAR FEATURES ──
    print("\n[3] Montando matriz de features...")
    f = pd.DataFrame(index=idx)

    # Precos
    f['mnq'] = mnq['close']
    f['es']  = es['close']
    f['btc'] = btc['close']
    f['cl']  = cl['close']

    # ==========================================
    # A. RETORNOS INDIVIDUAIS
    # ==========================================
    f['r_mnq_1h'] = mnq['ret1'] * 100
    f['r_es_1h']  = es['ret1']  * 100
    f['r_btc_1h'] = btc['ret1'] * 100
    f['r_cl_1h']  = cl['ret1']  * 100

    f['r_mnq_4h'] = mnq['ret4'] * 100
    f['r_es_4h']  = es['ret4']  * 100
    f['r_btc_4h'] = btc['ret4'] * 100
    f['r_cl_4h']  = cl['ret4']  * 100

    f['r_mnq_8h'] = mnq['ret8'] * 100

    # ==========================================
    # B. PADRAO PRINCIPAL: MNQ vs CL
    # ==========================================
    f['cl_down_mnq_up'] = ((f['r_cl_1h'] < 0) & (f['r_mnq_1h'] > 0)).astype(int)
    f['cl_up_mnq_down'] = ((f['r_cl_1h'] > 0) & (f['r_mnq_1h'] < 0)).astype(int)
    f['ambos_sobem']    = ((f['r_cl_1h'] > 0) & (f['r_mnq_1h'] > 0)).astype(int)
    f['ambos_caem']     = ((f['r_cl_1h'] < 0) & (f['r_mnq_1h'] < 0)).astype(int)

    # ==========================================
    # C. CORRELACOES COM ES E BTC
    # ==========================================
    f['es_mnq_mesmo_sentido'] = ((f['r_es_1h'] > 0) & (f['r_mnq_1h'] > 0) |
                                  (f['r_es_1h'] < 0) & (f['r_mnq_1h'] < 0)).astype(int)
    f['es_mnq_oposto'] = ((f['r_es_1h'] > 0) & (f['r_mnq_1h'] < 0) |
                          (f['r_es_1h'] < 0) & (f['r_mnq_1h'] > 0)).astype(int)

    f['btc_mnq_mesmo_sentido'] = ((f['r_btc_1h'] > 0) & (f['r_mnq_1h'] > 0) |
                                   (f['r_btc_1h'] < 0) & (f['r_mnq_1h'] < 0)).astype(int)
    f['btc_mnq_oposto'] = ((f['r_btc_1h'] > 0) & (f['r_mnq_1h'] < 0) |
                           (f['r_btc_1h'] < 0) & (f['r_mnq_1h'] > 0)).astype(int)

    # ES + BTC ambos concordam com MNQ
    f['es_btc_concordam_mnq'] = (f['es_mnq_mesmo_sentido'] & f['btc_mnq_mesmo_sentido']).astype(int)
    f['es_btc_discordam_mnq'] = (f['es_mnq_oposto'] & f['btc_mnq_oposto']).astype(int)

    # ==========================================
    # D. DIVERGENCIAS RSI
    # ==========================================
    f['rsi_mnq'] = mnq['rsi']
    f['rsi_es']  = es['rsi']
    f['rsi_btc'] = btc['rsi']
    f['rsi_cl']  = cl['rsi']

    f['div_cl']  = f['rsi_mnq'] - f['rsi_cl']
    f['div_es']  = f['rsi_mnq'] - f['rsi_es']
    f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']

    # Divergencias RSI ampliadas
    f['rsi_mnq_abaixo_40'] = (f['rsi_mnq'] < 40).astype(int)
    f['rsi_mnq_acima_60']  = (f['rsi_mnq'] > 60).astype(int)
    f['rsi_cl_abaixo_40']  = (f['rsi_cl'] < 40).astype(int)
    f['rsi_cl_acima_60']   = (f['rsi_cl'] > 60).astype(int)

    # ==========================================
    # E. KEY LEVELS — ABERTURAS 1h, 4h, Daily, Weekly
    # ==========================================
    print("  -> Key levels (aberturas 1h, 4h, daily, weekly)...")

    # Abertura 1h: o open da hora atual vs close anterior
    # Para cada ativo, calcular distancia % da abertura da hora
    for nome, serie in [('mnq', mnq_raw['open']), ('es', es_raw['open']),
                         ('btc', btc_raw['open']), ('cl', cl_raw['open'])]:
        s = serie.reindex(idx, method='ffill')
        # Open atual vs close anterior
        f[f'open_1h_{nome}'] = s
    f['open_1h_acima_close_ant_mnq'] = (f['open_1h_mnq'] > f['mnq'].shift(1)).astype(int)
    f['open_1h_acima_close_ant_cl']  = (f['open_1h_cl'] > f['cl'].shift(1)).astype(int)
    f['open_1h_acima_close_ant_es']  = (f['open_1h_es'] > f['es'].shift(1)).astype(int)
    f['open_1h_acima_close_ant_btc'] = (f['open_1h_btc'] > f['btc'].shift(1)).astype(int)

    # Abertura 4h: grupo de 4 horas
    for nome, serie in [('mnq', mnq_raw['open']), ('es', es_raw['open']),
                         ('btc', btc_raw['open']), ('cl', cl_raw['open'])]:
        s = serie.reindex(idx, method='ffill')
        open_4h = s.groupby(idx.floor('4h')).transform('first')
        close_4h_ant = s.shift(1).groupby(idx.floor('4h')).transform('first')
        f[f'open_4h_acima_4h_ant_{nome}'] = (open_4h > close_4h_ant).astype(int)
        f[f'open_4h_dist_{nome}'] = (s - open_4h) / open_4h * 100

    # Abertura Daily
    for nome, serie in [('mnq', mnq_raw['open']), ('es', es_raw['open']),
                         ('btc', btc_raw['open']), ('cl', cl_raw['open'])]:
        s = serie.reindex(idx, method='ffill')
        daily_open = s.groupby(idx.date).transform('first')
        daily_close_ant = s.shift(1).groupby(idx.date).transform('first')
        f[f'open_d_acima_d_ant_{nome}'] = (daily_open > daily_close_ant).astype(int)
        f[f'open_d_dist_{nome}'] = (s - daily_open) / daily_open * 100

    # Abertura Weekly
    for nome, serie in [('mnq', mnq_raw['open']), ('es', es_raw['open']),
                         ('btc', btc_raw['open']), ('cl', cl_raw['open'])]:
        s = serie.reindex(idx, method='ffill')
        weekly_open = s.groupby(pd.Grouper(freq='W-MON')).transform('first')
        weekly_close_ant = s.shift(1).groupby(pd.Grouper(freq='W-MON')).transform('first')
        f[f'open_w_acima_w_ant_{nome}'] = (weekly_open > weekly_close_ant).astype(int)
        f[f'open_w_dist_{nome}'] = (s - weekly_open) / weekly_open * 100

    # Comparacoes: qual ativo abriu acima/abaixo do outro
    # MNQ vs CL: open da hora
    f['open_mnq_acima_cl'] = (f['open_1h_mnq'] > f['open_1h_cl']).astype(int)
    f['open_mnq_abaixo_cl'] = (f['open_1h_mnq'] < f['open_1h_cl']).astype(int)
    f['open_mnq_acima_es']  = (f['open_1h_mnq'] > f['open_1h_es']).astype(int)
    f['open_mnq_abaixo_es'] = (f['open_1h_mnq'] < f['open_1h_es']).astype(int)
    f['open_mnq_acima_btc'] = (f['open_1h_mnq'] > f['open_1h_btc']).astype(int)
    f['open_mnq_abaixo_btc'] = (f['open_1h_mnq'] < f['open_1h_btc']).astype(int)

    # Distancia da abertura do dia (percentual)
    for nome in ['mnq', 'es', 'btc', 'cl']:
        f[f'dist_open_d_{nome}'] = f[f'open_d_dist_{nome}']

    # ==========================================
    # F. ADX E TENDENCIA
    # ==========================================
    f['adx_mnq'] = mnq['adx']
    f['adx_es']  = es['adx']
    f['adx_btc'] = btc['adx']
    f['adx_cl']  = cl['adx']

    f['adx_mnq_alto'] = (f['adx_mnq'] > 14).astype(int)
    f['adx_mnq_muito_alto'] = (f['adx_mnq'] > 25).astype(int)
    f['adx_cl_alto'] = (f['adx_cl'] > 14).astype(int)
    f['adx_es_alto'] = (f['adx_es'] > 14).astype(int)

    # DI spread
    f['di_spread_mnq'] = mnq['pdi'] - mnq['mdi']
    f['di_spread_es']  = es['pdi']  - es['mdi']
    f['di_spread_btc'] = btc['pdi'] - btc['mdi']
    f['di_spread_cl']  = cl['pdi']  - cl['mdi']

    # ==========================================
    # G. ALINHAMENTOS SMA50 / EMA20
    # ==========================================
    f['above_sma50_mnq'] = mnq['above_sma50']
    f['above_sma50_es']  = es['above_sma50']
    f['above_sma50_btc'] = btc['above_sma50']
    f['above_sma50_cl']  = cl['above_sma50']
    f['sma50_alignment'] = (f['above_sma50_mnq'] + f['above_sma50_es'] +
                            f['above_sma50_btc'] + f['above_sma50_cl'])

    f['above_ema20_mnq'] = mnq['above_ema20']
    f['above_ema20_es']  = es['above_ema20']
    f['above_ema20_btc'] = btc['above_ema20']
    f['above_ema20_cl']  = cl['above_ema20']
    f['ema20_alignment'] = (f['above_ema20_mnq'] + f['above_ema20_es'] +
                            f['above_ema20_btc'] + f['above_ema20_cl'])

    # Distancias das medias
    f['dist_sma50_mnq'] = mnq['dist_sma50']
    f['dist_sma50_es']  = es['dist_sma50']
    f['dist_sma50_cl']  = cl['dist_sma50']

    f['dist_ema20_mnq'] = mnq['dist_ema20']
    f['dist_ema20_cl']  = cl['dist_ema20']

    # ==========================================
    # H. VOLATILIDADE
    # ==========================================
    f['vol_mnq'] = mnq['vol'] * 100
    f['vol_es']  = es['vol'] * 100
    f['vol_btc'] = btc['vol'] * 100
    f['vol_cl']  = cl['vol'] * 100

    f['bb_w_mnq'] = mnq['bb_w'] * 100
    f['bb_w_cl']  = cl['bb_w'] * 100

    # ==========================================
    # I. HORARIO
    # ==========================================
    f['hour'] = idx.hour
    f['dow']  = idx.dayofweek
    f['is_us'] = f['hour'].between(9, 17).astype(int)
    f['is_asia'] = f['hour'].between(0, 8).astype(int)
    f['is_evening'] = f['hour'].between(18, 23).astype(int)

    # Sessao por grupo de 4h
    f['sessao_4h'] = idx.floor('4h').hour

    # Remover NaN das primeiras linhas
    f = f.dropna()
    print(f"  Features geradas: {len(f.columns)} colunas, {len(f)} linhas")

    # ==========================================
    # 4. VARIAVEL ALVO
    # ==========================================
    # Queremos prever quando MNQ SOBE + CL CAI
    target_col = 'cl_down_mnq_up'
    # Forward: o padrao ocorre NA PROXIMA HORA?
    f['target_1h'] = f[target_col].shift(-1).fillna(0).astype(int)
    # Forward 4h: ocorre em alguma das proximas 4 horas?
    f['target_4h'] = f[target_col].rolling(4, min_periods=1).max().shift(-4).fillna(0).astype(int)

    f = f.dropna(subset=['target_1h', 'target_4h'])
    print(f"  Amostras com target: {len(f)}")
    print(f"  Target 1h (prox hora): {f['target_1h'].mean():.1%}")
    print(f"  Target 4h (prox 4h): {f['target_4h'].mean():.1%}")

    # ==========================================
    # 5. ANALISE UNIVARIADA — Peso de cada feature
    # ==========================================
    print(f"\n{'='*90}")
    print("  5. PESO DE CADA VARIAVEL (Score de Edge)")
    print(f"{'='*90}")

    # Para features binarias: comparar win rate quando =1 vs quando =0
    # Para features continuas: dividir em quartis e comparar

    feature_cols_bin = [c for c in f.columns if c not in
                        ['mnq','es','btc','cl','hour','dow','sessao_4h','target_1h','target_4h',
                         'cl_down_mnq_up','cl_up_mnq_down','ambos_sobem','ambos_caem'] +
                        [c for c in f.columns if 'dist_' in c or 'vol_' in c or 'r_' in c or
                         'bb_w' in c or 'rsi' in c or 'adx' in c or 'di_spread' in c or
                         'div_' in c or 'dist_sma50' in c or 'dist_ema20' in c or
                         'open_d_dist' in c or 'open_w_dist' in c or 'open_4h_dist' in c]]

    results = []
    for col in f.columns:
        if col in ['mnq','es','btc','cl','hour','dow','sessao_4h','target_1h','target_4h']:
            continue
        if col.startswith('open_1h_') and col not in ['open_1h_acima_close_ant_mnq','open_1h_acima_close_ant_cl','open_1h_acima_close_ant_es','open_1h_acima_close_ant_btc']:
            continue
        if col in feature_cols_bin or col in ['cl_down_mnq_up','cl_up_mnq_down','ambos_sobem','ambos_caem']:
            continue

        # Determinar se binaria ou continua
        n_unique = f[col].nunique()
        is_bin = n_unique <= 2

        if is_bin:
            # Feature binaria
            for val in [1]:
                sub = f[f[col] == val]
                if len(sub) < 5: continue
                wr_1h = sub['target_1h'].mean()
                wr_4h = sub['target_4h'].mean()
                total = len(sub)
                results.append({
                    'feature': col, 'tipo': 'binaria', 'condicao': f'{col}=1',
                    'n': total,
                    'wr_1h': wr_1h, 'wr_4h': wr_4h,
                    'edge_1h': wr_1h - f['target_1h'].mean(),
                    'edge_4h': wr_4h - f['target_4h'].mean(),
                })
        else:
            # Feature continua: dividir em quartis
            qs = f[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
            for i in range(4):
                lo = qs.iloc[i]
                hi = qs.iloc[i+1]
                if i == 3:
                    sub = f[f[col] >= lo]
                elif i == 0:
                    sub = f[f[col] <= hi]
                else:
                    sub = f[(f[col] >= lo) & (f[col] < hi)]
                if len(sub) < 5: continue
                wr_1h = sub['target_1h'].mean()
                wr_4h = sub['target_4h'].mean()
                label = f'Q{i+1}[{lo:.3f}-{hi:.3f}]'
                results.append({
                    'feature': col, 'tipo': 'continua', 'condicao': label,
                    'n': len(sub),
                    'wr_1h': wr_1h, 'wr_4h': wr_4h,
                    'edge_1h': wr_1h - f['target_1h'].mean(),
                    'edge_4h': wr_4h - f['target_4h'].mean(),
                })

    df_r = pd.DataFrame(results)
    if len(df_r) == 0:
        print("  Nenhum resultado!")
        return

    # Top edges para target_1h
    print(f"\n  TOP 30 MAIORES EDGES (target = 'cl_down_mnq_up' na prox 1h):")
    print(f"  {'Feature':<55} {'Condicao':<25} {'N':>5} {'WR_1h':>7} {'Edge':>7}")
    print(f"  {'-'*55} {'-'*25} {'-'*5} {'-'*7} {'-'*7}")
    top1 = df_r.sort_values('edge_1h', ascending=False).head(30)
    for _, r in top1.iterrows():
        print(f"  {r['feature']:<55} {r['condicao']:<25} {r['n']:>5} {r['wr_1h']:.1%} {r['edge_1h']:+.1%}")

    print(f"\n  TOP 30 MAIORES EDGES NEGATIVOS (target = 'cl_down_mnq_up' na prox 1h):")
    bot1 = df_r.sort_values('edge_1h', ascending=True).head(30)
    for _, r in bot1.iterrows():
        print(f"  {r['feature']:<55} {r['condicao']:<25} {r['n']:>5} {r['wr_1h']:.1%} {r['edge_1h']:+.1%}")

    # Top edges para target_4h
    print(f"\n  TOP 30 MAIORES EDGES (target = 'cl_down_mnq_up' nas prox 4h):")
    print(f"  {'Feature':<55} {'Condicao':<25} {'N':>5} {'WR_4h':>7} {'Edge':>7}")
    print(f"  {'-'*55} {'-'*25} {'-'*5} {'-'*7} {'-'*7}")
    top4 = df_r.sort_values('edge_4h', ascending=False).head(30)
    for _, r in top4.iterrows():
        print(f"  {r['feature']:<55} {r['condicao']:<25} {r['n']:>5} {r['wr_4h']:.1%} {r['edge_4h']:+.1%}")

    # ==========================================
    # 6. ANALISE MULTIVARIADA — Combinacoes de gatilhos
    # ==========================================
    print(f"\n{'='*90}")
    print("  6. COMBINACOES DE GATILHOS (Score Composto)")
    print(f"{'='*90}")

    # Lista de gatilhos candidatos (os melhores individuais)
    best_features = top1[top1['n'] > 50]['feature'].unique()[:15]

    # Combinacoes de 2, 3 gatilhos
    from itertools import combinations

    combo_results = []
    bin_cols = [c for c in f.columns if f[c].nunique() <= 2 and c not in
                ['target_1h','target_4h','cl_down_mnq_up','cl_up_mnq_down','ambos_sobem','ambos_caem']]
    # So usar binarias que tem pelo menos 50 amostras TRUE
    bin_cols = [c for c in bin_cols if f[c].sum() >= 30]

    print(f"  Testando combinacoes de gatilhos binarios ({len(bin_cols)} gatilhos)...")

    # Combinacoes de 2
    n_combo = 0
    for a, b in combinations(bin_cols, 2):
        sub = f[(f[a] == 1) & (f[b] == 1)]
        if len(sub) < 20: continue
        wr_1h = sub['target_1h'].mean()
        edge_1h = wr_1h - f['target_1h'].mean()
        wr_4h = sub['target_4h'].mean()
        edge_4h = wr_4h - f['target_4h'].mean()
        combo_results.append({
            'combo': f'{a} + {b}', 'n': len(sub),
            'wr_1h': wr_1h, 'edge_1h': edge_1h,
            'wr_4h': wr_4h, 'edge_4h': edge_4h,
        })
        n_combo += 1

    # Combinacoes de 3
    for a, b, c_ in combinations(bin_cols, 3):
        sub = f[(f[a] == 1) & (f[b] == 1) & (f[c_] == 1)]
        if len(sub) < 10: continue
        wr_1h = sub['target_1h'].mean()
        edge_1h = wr_1h - f['target_1h'].mean()
        wr_4h = sub['target_4h'].mean()
        edge_4h = wr_4h - f['target_4h'].mean()
        combo_results.append({
            'combo': f'{a} + {b} + {c_}', 'n': len(sub),
            'wr_1h': wr_1h, 'edge_1h': edge_1h,
            'wr_4h': wr_4h, 'edge_4h': edge_4h,
        })
        n_combo += 1

    df_combo = pd.DataFrame(combo_results)
    if len(df_combo) > 0:
        print(f"  {n_combo} combinacoes testadas")

        print(f"\n  TOP 20 COMBINACOES DE 2 GATILHOS (maior edge 1h):")
        c2 = df_combo[df_combo['combo'].str.count('\\+') == 1].sort_values('edge_1h', ascending=False).head(20)
        for _, r in c2.iterrows():
            print(f"  {r['combo']:<85} N={r['n']:>4} WR_1h={r['wr_1h']:.1%} Edge={r['edge_1h']:+.1%}")

        print(f"\n  TOP 20 COMBINACOES DE 3 GATILHOS (maior edge 1h):")
        c3 = df_combo[df_combo['combo'].str.count('\\+') == 2].sort_values('edge_1h', ascending=False).head(20)
        for _, r in c3.iterrows():
            print(f"  {r['combo']:<85} N={r['n']:>4} WR_1h={r['wr_1h']:.1%} Edge={r['edge_1h']:+.1%}")

    # ==========================================
    # 7. SCORING — Modelo heuristico ponderado
    # ==========================================
    print(f"\n{'='*90}")
    print("  7. SCORE COMPOSTO — Modelo heuristico")
    print(f"{'='*90}")

    # Score COMPOSTO: inclui TANTO binarias nativas QUANTO condicoes de quartil
    # Para cada feature continua, criar variavel binaria "Q4" (top 25%)
    quartil_bins = {}
    for col in f.columns:
        if col in ['mnq','es','btc','cl','hour','dow','sessao_4h','target_1h','target_4h',
                   'score_gatilhos','score_decile']:
            continue
        if f[col].nunique() <= 2:
            continue
        # Quartil superior
        q3 = f[col].quantile(0.75)
        q1 = f[col].quantile(0.25)
        name_hi = f'q_hi_{col}'
        name_lo = f'q_lo_{col}'
        f[name_hi] = (f[col] >= q3).astype(int)
        f[name_lo] = (f[col] <= q1).astype(int)
        quartil_bins[name_hi] = col
        quartil_bins[name_lo] = col

    # Coletar TODOS os candidatos binarios (nativos + quartis)
    bin_candidates = {}
    bin_cols_all = [c for c in f.columns if f[c].nunique() <= 2 and c not in
                    ['target_1h','target_4h','cl_down_mnq_up','cl_up_mnq_down',
                     'ambos_sobem','ambos_caem','score_gatilhos','score_decile']]

    for col in bin_cols_all:
        if f[col].sum() < 30:
            continue
        wr = f[f[col] == 1]['target_1h'].mean()
        edge = wr - f['target_1h'].mean()
        if edge > 0.005:  # so edges positivos minimos
            bin_candidates[col] = edge

    # Selecionar top N por edge
    top_n = min(30, len(bin_candidates))
    best_bins = sorted(bin_candidates.items(), key=lambda x: -x[1])[:top_n]

    print(f"  Top {len(best_bins)} gatilhos (binarios + quartis) para o score:")
    pesos = {}
    for col, edge in best_bins:
        pesos[col] = max(edge, 0)
        origem = quartil_bins.get(col, 'nativo')
        print(f"    {col:<55} edge={edge:+.4f}  ({origem})")

    if sum(pesos.values()) > 0:
        # Normalizar pesos
        total = sum(pesos.values())
        pesos = {k: v/total for k, v in pesos.items()}

        # Calcular score composto
        f['score_gatilhos'] = sum(f[col] * peso for col, peso in pesos.items())

        # Analisar score por decil
        print(f"\n  Score por decil vs Win Rate:")
        f['score_decile'] = pd.qcut(f['score_gatilhos'].rank(method='first'), 10, labels=False, duplicates='drop')
        for dec in sorted(f['score_decile'].unique()):
            sub = f[f['score_decile'] == dec]
            wr = sub['target_1h'].mean()
            wr4 = sub['target_4h'].mean()
            med = sub['score_gatilhos'].median()
            print(f"    Decil {dec}: score_med={med:.4f}  N={len(sub):>5}  WR_1h={wr:.1%}  WR_4h={wr4:.1%}")

        # Tabela operacional por faixa
        print(f"\n  Tabela operacional - Score vs WR:")
        thresholds = sorted(set([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] +
                                [round(v, 2) for v in f['score_gatilhos'].quantile([0.2, 0.4, 0.6, 0.8])]))
        thresholds = [t for t in thresholds if t <= 1.0]
        for i in range(len(thresholds)):
            lo = thresholds[i]
            hi = thresholds[i+1] if i+1 < len(thresholds) else 1.0
            sub = f[(f['score_gatilhos'] >= lo) & (f['score_gatilhos'] < hi)]
            if len(sub) < 5: continue
            wr1 = sub['target_1h'].mean()
            wr4 = sub['target_4h'].mean()
            print(f"    Score [{lo:.2f}-{hi:.2f}): N={len(sub):>5}  WR_1h={wr1:.1%}  WR_4h={wr4:.1%}")

    # ==========================================
    # 8. ANALISE POR SESSAO / HORARIO
    # ==========================================
    print(f"\n{'='*90}")
    print("  8. MELHORES HORARIOS PARA O PADRAO")
    print(f"{'='*90}")

    for h in range(24):
        sub = f[f['hour'] == h]
        if len(sub) < 10: continue
        n_padrao = sub[target_col].sum()
        wr1 = sub['target_1h'].mean()
        wr4 = sub['target_4h'].mean()
        print(f"    Hora {h:02d}: N={len(sub):>5}  padrao_agora={n_padrao/len(sub):.1%}  WR_1h={wr1:.1%}  WR_4h={wr4:.1%}")

    # ==========================================
    # 9. EXPORTAR RESULTADOS
    # ==========================================
    out_dir = Path(__file__).parent
    df_r.sort_values('edge_1h', ascending=False).to_csv(out_dir / 'pesos_gatilhos.csv', index=False)
    if len(combo_results) > 0:
        df_combo.sort_values('edge_1h', ascending=False).to_csv(out_dir / 'combinacoes_gatilhos.csv', index=False)
    print(f"\n  Resultados salvos em:")
    print(f"    - {out_dir}/pesos_gatilhos.csv")
    print(f"    - {out_dir}/combinacoes_gatilhos.csv")
    print(f"\n{'='*90}")
    print("  ANALISE CONCLUIDA")
    print(f"{'='*90}")

if __name__ == '__main__':
    main()

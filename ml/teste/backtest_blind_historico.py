"""
Backtest Cego Histórico — Trigger Rotacao MNQ×CL
Dados: SQLite local 2021-2026 (1h)
Split:  In-Sample 2021-2023 | Blind 2024 | Blind 2025 | Live 2026
"""
import warnings; warnings.filterwarnings('ignore')
import sqlite3
import pandas as pd
import numpy as np
import ta

FWD    = 4      # horas forward
TARGET = 0.001  # MNQ >0.1% = win

# -- Carregar do SQLite -------------------------------------------------------
def load_db(path, symbol):
    conn = sqlite3.connect(path)
    df = pd.read_sql("SELECT ts, open, high, low, close, volume FROM candles ORDER BY ts", conn)
    conn.close()
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df.columns = df.columns.str.lower()
    return df[['open','high','low','close','volume']].dropna()

print("Carregando dados do banco local...")
raw = {
    'mnq': load_db('ml/data.db',     'mnq'),
    'cl':  load_db('ml/cl/data.db',  'cl'),
    'es':  load_db('ml/es/data.db',  'es'),
    'btc': load_db('ml/btc/data.db', 'btc'),
}
for k, v in raw.items():
    print(f"  {k}: {len(v)} candles  |  {v.index[0].date()} -> {v.index[-1].date()}")

# -- Indicadores --------------------------------------------------------------
def compute(df):
    d = df.copy()
    d['rsi']    = ta.momentum.RSIIndicator(d['close'], window=21).rsi()
    adx_i       = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=17)
    d['adx']    = adx_i.adx()
    d['sma50']  = d['close'].rolling(50).mean()
    d['above50']= (d['close'] > d['sma50']).astype(int)
    d['sma_bb'] = d['close'].rolling(20).mean()
    d['std_bb'] = d['close'].rolling(20).std()
    d['bb_w']   = d['std_bb'] * 2 / d['sma_bb'] * 100
    d['ret1']   = d['close'].pct_change(1)
    return d

c = {k: compute(raw[k]) for k in raw}

# -- Alinhar índices ----------------------------------------------------------
idx = c['mnq'].index
for k in ['cl', 'es', 'btc']:
    idx = idx.intersection(c[k].index)
for k in c:
    c[k] = c[k].loc[idx]

# -- Features -----------------------------------------------------------------
f = pd.DataFrame(index=idx)
f['rsi_mnq']    = c['mnq']['rsi']
f['rsi_cl']     = c['cl']['rsi']
f['adx_mnq']    = c['mnq']['adx']
f['bb_w_cl']    = c['cl']['bb_w']
f['ret1_mnq']   = c['mnq']['ret1']
f['ret1_es']    = c['es']['ret1']
f['above50_mnq']= c['mnq']['above50']
f['above50_cl'] = c['cl']['above50']
f['above50_es'] = c['es']['above50']
f['above50_btc']= c['btc']['above50']
f['mnq_close']  = c['mnq']['close']

cl_dopen_daily   = c['cl']['open'].resample('D').first()
f['cl_dopen']    = cl_dopen_daily.reindex(idx, method='ffill')
f['cl_dopen_ant']= cl_dopen_daily.shift(1).reindex(idx, method='ffill')

# -- Divergência e rotação ----------------------------------------------------
f['div_cl']          = f['rsi_mnq'] - f['rsi_cl']
f['div_delta_2h']    = f['div_cl'] - f['div_cl'].shift(2)
f['div_sinal_mudou'] = (((f['div_cl']>0)&(f['div_cl'].shift(1)<=0)) |
                        ((f['div_cl']<0)&(f['div_cl'].shift(1)>=0))).astype(int)
f['rsi_mnq_d2']      = f['rsi_mnq'] - f['rsi_mnq'].shift(2)
f['rsi_cl_d2']       = f['rsi_cl']  - f['rsi_cl'].shift(2)
f['rot_ativa']       = ((f['rsi_mnq_d2']>2.0)&(f['rsi_cl_d2']<-2.0)).astype(int)

# -- 9 Condições (mesmos pesos do trigger-rotacao.md) -------------------------
f['c1'] = f['div_sinal_mudou']
f['c2'] = (f['div_delta_2h'] > 5.0).astype(int)
f['c3'] = f['rot_ativa']
f['c4'] = (((f['ret1_mnq']>0)&(f['ret1_es']>0))|((f['ret1_mnq']<0)&(f['ret1_es']<0))).astype(int)
f['c5'] = (f['bb_w_cl'] > 1.5).astype(int)
f['c6'] = (f['cl_dopen'] > f['cl_dopen_ant']).astype(int)
f['c7'] = ((f['adx_mnq']>=12)&(f['adx_mnq']<=20)).astype(int)
f['c8'] = ((f['above50_mnq']+f['above50_cl']+f['above50_es']+f['above50_btc'])>=2).astype(int)
f['c9'] = ((f['rsi_mnq']>55)|(f['rsi_mnq']<45)).astype(int)

f['sc']        = (f['c1']*3.0 + f['c2']*2.5 + f['c3']*2.5 + f['c4']*2.0 +
                  f['c5']*1.5 + f['c6']*1.5 + f['c7']*1.5 + f['c8']*1.5 + f['c9']*1.0)
f['score_pct'] = f['sc'] / 17.0 * 100
f['rotacao']   = (f['score_pct'] >= 60).astype(int)

f = f.dropna()

# -- Retornos forward ---------------------------------------------------------
# Dados nao ajustados: rollovers de contrato criam retornos absurdos (>3%)
# Filtramos para nao contaminar o WR com artefatos de rollover
f['fwd_ret']    = f['mnq_close'].pct_change(FWD).shift(-FWD)
f.loc[f['fwd_ret'].abs() > 0.03, 'fwd_ret'] = np.nan  # remove artefatos rollover
f['rotacao_new']= ((f['rotacao']==1)&(f['rotacao'].shift(1)==0)).astype(int)

# -- Stats por período --------------------------------------------------------
def stats_periodo(subset, label, baseline_wr):
    e = subset[subset['rotacao_new']==1].dropna(subset=['fwd_ret'])
    bl = (subset['fwd_ret'] > TARGET).mean()*100
    n = len(e)
    if n == 0:
        print(f"\n  {label}: 0 sinais")
        return None
    wins = e['fwd_ret'] > TARGET
    wr   = wins.mean()*100
    exp  = (e['fwd_ret']*100).mean()
    alpha = wr - bl
    print(f"\n  {'-'*48}")
    print(f"  {label}")
    print(f"  {'-'*48}")
    print(f"  Período:    {e.index[0].date()} -> {e.index[-1].date()}")
    print(f"  Sinais:     {n}")
    print(f"  Win Rate:   {wr:.1f}%")
    print(f"  Baseline:   {bl:.1f}%")
    print(f"  Alpha:      {alpha:+.1f}pp")
    print(f"  Expectativa:{exp:+.4f}% por trade")

    # Por mês
    e2 = e.copy(); e2['win'] = wins; e2['mes'] = e2.index.to_period('M')
    for m, g in e2.groupby('mes'):
        wr_m = g['win'].mean()*100
        bar  = '#' * int(wr_m/10)
        print(f"    {m}  n={len(g):3d}  WR={wr_m:4.0f}%  {bar}")
    return {'label': label, 'n': n, 'wr': wr, 'bl': bl, 'alpha': alpha}

# -- Splits por ano -----------------------------------------------------------
periodos = {
    'IN-SAMPLE   2021-2023': (f['2021-01-01':'2023-12-31']),
    'BLIND TEST  2024     ': (f['2024-01-01':'2024-12-31']),
    'BLIND TEST  2025     ': (f['2025-01-01':'2025-12-31']),
    'LIVE        2026     ': (f['2026-01-01':]),
}

bl_global = (f['fwd_ret'] > TARGET).mean()*100
print(f"\n\nBaseline global (qualquer barra): {bl_global:.1f}%")

resultados = []
for label, subset in periodos.items():
    r = stats_periodo(subset, label, bl_global)
    if r:
        resultados.append(r)

# -- Resumo final -------------------------------------------------------------
print(f"\n\n{'='*52}")
print(f"  RESUMO WALK-FORWARD (mesmo threshold, sem refit)")
print(f"{'='*52}")
print(f"  {'Período':<22}  {'Sinais':>6}  {'WR':>6}  {'Baseline':>8}  {'Alpha':>6}")
print(f"  {'-'*50}")
for r in resultados:
    flag = ' OK' if r['alpha'] > 3 else (' XX' if r['alpha'] < 0 else ' --')
    print(f"  {r['label']:<22}  {r['n']:>6}  {r['wr']:>5.1f}%  {r['bl']:>7.1f}%  {r['alpha']:>+5.1f}pp{flag}")
print(f"{'='*52}")
print()

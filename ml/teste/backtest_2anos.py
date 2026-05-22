"""
Backtest 2 Anos — Trigger Rotacao MNQ x CL
Fonte: yfinance auto_adjust=True (rollover ajustado)
Split: In-Sample 2024 | Blind 2025 | Live 2026
"""
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

PERIOD = 720
FWD    = 4
TARGET = 0.001

end   = datetime.now()
start = end - timedelta(days=PERIOD)

def download(ticker):
    df = yf.download(ticker, start=start, end=end, interval='1h',
                     auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()

print("Baixando 2 anos de dados (yfinance ajustado)...")
raw = {k: download(t) for k, t in {'mnq':'MNQ=F','cl':'CL=F','es':'ES=F','btc':'BTC-USD'}.items()}
if any(v is None for v in raw.values()):
    print("ERRO: falha ao baixar algum ativo"); exit()
for k, v in raw.items():
    print(f"  {k}: {len(v)} candles  |  {v.index[0].date()} -> {v.index[-1].date()}")

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

idx = c['mnq'].index
for k in ['cl','es','btc']:
    idx = idx.intersection(c[k].index)
for k in c:
    c[k] = c[k].loc[idx]

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

f['div_cl']          = f['rsi_mnq'] - f['rsi_cl']
f['div_delta_2h']    = f['div_cl'] - f['div_cl'].shift(2)
f['div_sinal_mudou'] = (((f['div_cl']>0)&(f['div_cl'].shift(1)<=0)) |
                        ((f['div_cl']<0)&(f['div_cl'].shift(1)>=0))).astype(int)
f['rsi_mnq_d2']      = f['rsi_mnq'] - f['rsi_mnq'].shift(2)
f['rsi_cl_d2']       = f['rsi_cl']  - f['rsi_cl'].shift(2)
f['rot_ativa']       = ((f['rsi_mnq_d2']>2.0)&(f['rsi_cl_d2']<-2.0)).astype(int)

f['c1'] = f['div_sinal_mudou']
f['c2'] = (f['div_delta_2h'] > 5.0).astype(int)
f['c3'] = f['rot_ativa']
f['c4'] = (((f['ret1_mnq']>0)&(f['ret1_es']>0))|((f['ret1_mnq']<0)&(f['ret1_es']<0))).astype(int)
f['c5'] = (f['bb_w_cl'] > 1.5).astype(int)
f['c6'] = (f['cl_dopen'] > f['cl_dopen_ant']).astype(int)
f['c7'] = ((f['adx_mnq']>=12)&(f['adx_mnq']<=20)).astype(int)
f['c8'] = ((f['above50_mnq']+f['above50_cl']+f['above50_es']+f['above50_btc'])>=2).astype(int)
f['c9'] = ((f['rsi_mnq']>55)|(f['rsi_mnq']<45)).astype(int)

f['sc']        = (f['c1']*3.0+f['c2']*2.5+f['c3']*2.5+f['c4']*2.0+
                  f['c5']*1.5+f['c6']*1.5+f['c7']*1.5+f['c8']*1.5+f['c9']*1.0)
f['score_pct'] = f['sc'] / 17.0 * 100
f['rotacao']   = (f['score_pct'] >= 60).astype(int)
f = f.dropna()

f['fwd_ret']    = f['mnq_close'].pct_change(FWD).shift(-FWD)
f['rotacao_new']= ((f['rotacao']==1)&(f['rotacao'].shift(1)==0)).astype(int)

# -- Key Levels (UTC) ---------------------------------------------------------
# Bons KLs do backtest-rotacao.md: PDH, LONDON-H/L, NY-H, PWL, MON-L, PWO
mnq = c['mnq'].copy()

# PDH / PDL
daily       = mnq['high'].resample('D').max()
daily_l     = mnq['low'].resample('D').min()
f['pdh']    = daily.shift(1).reindex(f.index, method='ffill')
f['pdl']    = daily_l.shift(1).reindex(f.index, method='ffill')

# PWH / PWL / PWO (semana começa segunda)
weekly_h    = mnq['high'].resample('W-MON').max()
weekly_l    = mnq['low'].resample('W-MON').min()
weekly_o    = mnq['open'].resample('W-MON').first()
f['pwh']    = weekly_h.shift(1).reindex(f.index, method='ffill')
f['pwl']    = weekly_l.shift(1).reindex(f.index, method='ffill')
f['pwo']    = weekly_o.shift(1).reindex(f.index, method='ffill')

# MON-L (low da segunda-feira da semana atual, forward fill)
mondays     = mnq['low'][mnq.index.dayofweek == 0].resample('D').min()
mon_l_daily = mondays.reindex(pd.date_range(mnq.index[0], mnq.index[-1], freq='D'), method='ffill')
f['mon_l']  = mon_l_daily.reindex(f.index, method='ffill')

# LONDON session: 08-16 UTC
lon         = mnq[mnq.index.hour.isin(range(8, 16))]
lon_h_d     = lon['high'].resample('D').max()
lon_l_d     = lon['low'].resample('D').min()
f['lon_h']  = lon_h_d.shift(1).reindex(f.index, method='ffill')
f['lon_l']  = lon_l_d.shift(1).reindex(f.index, method='ffill')

# NY session: 14-20 UTC
ny          = mnq[mnq.index.hour.isin(range(14, 21))]
ny_h_d      = ny['high'].resample('D').max()
ny_l_d      = ny['low'].resample('D').min()
f['ny_h']   = ny_h_d.shift(1).reindex(f.index, method='ffill')
f['ny_l']   = ny_l_d.shift(1).reindex(f.index, method='ffill')

# near_good_kl: preco dentro de 0.2% de qualquer bom KL
KL_PCT = 0.002
price = f['mnq_close']
good_kls = ['pdh', 'lon_h', 'lon_l', 'ny_h', 'pwl', 'mon_l', 'pwo']

near = pd.Series(False, index=f.index)
for kl in good_kls:
    dist = (price - f[kl]).abs() / f[kl]
    near = near | (dist <= KL_PCT)

f['near_good_kl'] = near.astype(int)

def stats(subset, label):
    e = subset[subset['rotacao_new']==1].dropna(subset=['fwd_ret'])
    bl = (subset['fwd_ret'] > TARGET).mean()*100
    n = len(e)
    if n == 0:
        print(f"\n  {label}: sem sinais"); return None
    wins  = e['fwd_ret'] > TARGET
    wr    = wins.mean()*100
    wn    = e.loc[wins,  'fwd_ret']*100
    ls    = e.loc[~wins, 'fwd_ret']*100
    exp   = (e['fwd_ret']*100).mean()
    alpha = wr - bl
    print(f"\n  {'-'*50}")
    print(f"  {label}")
    print(f"  {'-'*50}")
    print(f"  Periodo:     {e.index[0].date()} -> {e.index[-1].date()}")
    print(f"  Sinais:      {n}")
    print(f"  Win Rate:    {wr:.1f}%")
    print(f"  Baseline:    {bl:.1f}%")
    print(f"  Alpha:       {alpha:+.1f}pp")
    print(f"  Expectativa: {exp:+.4f}% por trade")
    if len(wn): print(f"  Ganho medio: +{wn.mean():.3f}%  |  Perda media: {ls.mean():.3f}%")
    e2 = e.copy(); e2['win'] = wins; e2['mes'] = e2.index.to_period('M')
    print(f"\n  Por mes:")
    for m, g in e2.groupby('mes'):
        wr_m = g['win'].mean()*100
        ret_m= (g['fwd_ret']*100).sum()
        bar  = '#' * int(wr_m/10)
        s = '+' if ret_m >= 0 else ''
        print(f"    {m}  n={len(g):3d}  WR={wr_m:4.0f}%  {bar}  ret={s}{ret_m:.2f}%")
    return {'label': label, 'n': n, 'wr': wr, 'bl': bl, 'alpha': alpha}

# -- Splits por ano -----------------------------------------------------------
anos = {
    'IN-SAMPLE  2024': f['2024-01-01':'2024-12-31'],
    'BLIND      2025': f['2025-01-01':'2025-12-31'],
    'LIVE       2026': f['2026-01-01':],
    'TOTAL  2024-26 ': f,
}

print(f"\nBaseline global: {(f['fwd_ret']>TARGET).mean()*100:.1f}%")
pct_near = f.loc[f['rotacao_new']==1, 'near_good_kl'].mean()*100
print(f"Sinais com KL proximo: {pct_near:.0f}%\n")

res_sem = []
res_com = []
for label, subset in anos.items():
    print(f"\n{'#'*56}")
    print(f"  {label}  — SEM filtro KL")
    r = stats(subset, label)
    if r: res_sem.append(r)

    subset_kl = subset[subset['near_good_kl']==1]
    print(f"\n  {label}  — COM filtro KL (< 0.2%)")
    r2 = stats(subset_kl, label + ' + KL')
    if r2: res_com.append(r2)

# -- Resumo comparativo -------------------------------------------------------
def veredicto(alpha):
    if   alpha >  5: return 'SINAL FORTE'
    elif alpha >  2: return 'sinal fraco'
    elif alpha >= 0: return 'neutro'
    else:            return '!!! negativo'

print(f"\n\n{'='*62}")
print(f"  SEM FILTRO KEY LEVEL")
print(f"{'='*62}")
print(f"  {'Periodo':<20}  {'n':>5}  {'WR':>6}  {'Base':>6}  {'Alpha':>7}  {'Veredicto'}")
print(f"  {'-'*60}")
for r in res_sem:
    print(f"  {r['label']:<20}  {r['n']:>5}  {r['wr']:>5.1f}%  {r['bl']:>5.1f}%  {r['alpha']:>+6.1f}pp  {veredicto(r['alpha'])}")

print(f"\n\n{'='*62}")
print(f"  COM FILTRO KEY LEVEL (< 0.2% de PDH/LON-H/LON-L/NY-H/PWL/MON-L/PWO)")
print(f"{'='*62}")
print(f"  {'Periodo':<20}  {'n':>5}  {'WR':>6}  {'Base':>6}  {'Alpha':>7}  {'Veredicto'}")
print(f"  {'-'*60}")
for r in res_com:
    print(f"  {r['label']:<24}  {r['n']:>5}  {r['wr']:>5.1f}%  {r['bl']:>5.1f}%  {r['alpha']:>+6.1f}pp  {veredicto(r['alpha'])}")
print(f"{'='*62}\n")

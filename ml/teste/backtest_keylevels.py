# -*- coding: utf-8 -*-
"""
Backtest: Key Levels + Trigger Rotacao MNQ x CL
Hipotese: rotacao perto de key level tem WR maior?
Key levels: PDH/PDL/PDO, PWH/PWL, Monday H/L, Sessoes NY/London/Asia
"""
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

PERIOD    = 185    # dias
FWD       = 4      # horas forward
TARGET    = 0.001  # >0.1% = win
NEAR_PCT  = 0.003  # dentro de 0.3% = "perto" de key level

# ── Download ──────────────────────────────────────────────────────────────────
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

print("Baixando dados...")
raw = {k: download(t) for k, t in
       {'mnq':'MNQ=F','cl':'CL=F','es':'ES=F','btc':'BTC-USD'}.items()}

# ── Indicadores base ──────────────────────────────────────────────────────────
def compute(df):
    d = df.copy()
    d['rsi']     = ta.momentum.RSIIndicator(d['close'], window=21).rsi()
    adx_i        = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=17)
    d['adx']     = adx_i.adx()
    d['sma50']   = d['close'].rolling(50).mean()
    d['above50'] = (d['close'] > d['sma50']).astype(int)
    d['sma_bb']  = d['close'].rolling(20).mean()
    d['std_bb']  = d['close'].rolling(20).std()
    d['bb_w']    = d['std_bb'] * 2 / d['sma_bb'] * 100
    d['ret1']    = d['close'].pct_change(1)
    return d

c = {k: compute(raw[k]) for k in raw}
idx = c['mnq'].index
for k in ['cl','es','btc']:
    idx = idx.intersection(c[k].index)
for k in c:
    c[k] = c[k].loc[idx]

# ── Features rotacao ──────────────────────────────────────────────────────────
f = pd.DataFrame(index=idx)
f['price']       = c['mnq']['close']
f['rsi_mnq']     = c['mnq']['rsi']
f['rsi_cl']      = c['cl']['rsi']
f['adx_mnq']     = c['mnq']['adx']
f['bb_w_cl']     = c['cl']['bb_w']
f['ret1_mnq']    = c['mnq']['ret1']
f['ret1_es']     = c['es']['ret1']
f['above50_mnq'] = c['mnq']['above50']
f['above50_cl']  = c['cl']['above50']
f['above50_es']  = c['es']['above50']
f['above50_btc'] = c['btc']['above50']

cl_dopen_daily    = c['cl']['open'].resample('D').first()
f['cl_dopen']     = cl_dopen_daily.reindex(idx, method='ffill')
f['cl_dopen_ant'] = cl_dopen_daily.shift(1).reindex(idx, method='ffill')

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
f['rotacao_new'] = ((f['rotacao']==1)&(f['rotacao'].shift(1)==0)).astype(int)
f['fwd_ret']   = f['price'].pct_change(FWD).shift(-FWD)

# ── KEY LEVELS em MNQ ─────────────────────────────────────────────────────────
mnq_h = raw['mnq'].copy()

# --- Diario ---
mnq_day = mnq_h.resample('D').agg(dh=('high','max'), dl=('low','min'),
                                   do=('open','first'), dc=('close','last'))
f['pdh'] = mnq_day['dh'].shift(1).reindex(idx, method='ffill')   # prev day high
f['pdl'] = mnq_day['dl'].shift(1).reindex(idx, method='ffill')   # prev day low
f['pdo'] = mnq_day['do'].shift(1).reindex(idx, method='ffill')   # prev day open

# --- Semanal (semana anterior) ---
mnq_week = mnq_h.resample('W-MON').agg(wh=('high','max'), wl=('low','min'),
                                        wo=('open','first'))
f['pwh'] = mnq_week['wh'].shift(1).reindex(idx, method='ffill')
f['pwl'] = mnq_week['wl'].shift(1).reindex(idx, method='ffill')
f['pwo'] = mnq_week['wo'].shift(1).reindex(idx, method='ffill')

# --- Segunda-feira da semana atual ---
mon_bars = mnq_h[mnq_h.index.dayofweek == 0]
mon_week = mon_bars.resample('W-MON').agg(mh=('high','max'), ml=('low','min'),
                                           mo=('open','first'))
f['mon_h'] = mon_week['mh'].reindex(idx, method='ffill')
f['mon_l'] = mon_week['ml'].reindex(idx, method='ffill')
f['mon_o'] = mon_week['mo'].reindex(idx, method='ffill')

# --- Sessoes (UTC): Asia 0-7h, London 7-14h, NY 14-22h ---
def session_name(h):
    if 0 <= h < 7:   return 'asia'
    elif 7 <= h < 14: return 'london'
    else:             return 'ny'

mnq_h2 = mnq_h.copy()
mnq_h2['sess']     = mnq_h2.index.hour.map(session_name)
mnq_h2['date_str'] = mnq_h2.index.strftime('%Y-%m-%d')
mnq_h2['sess_key'] = mnq_h2['date_str'] + '_' + mnq_h2['sess']

sess_agg = mnq_h2.groupby('sess_key').agg(
    sh=('high','max'), sl=('low','min')
).reset_index()

# Para cada barra, encontrar H/L da sessao anterior completa
sess_agg = sess_agg.sort_values('sess_key')
sess_agg['prev_sh'] = sess_agg['sh'].shift(1)
sess_agg['prev_sl'] = sess_agg['sl'].shift(1)
sess_map_h = dict(zip(sess_agg['sess_key'], sess_agg['prev_sh']))
sess_map_l = dict(zip(sess_agg['sess_key'], sess_agg['prev_sl']))

f['sess_key'] = pd.Series(
    mnq_h2['date_str'].values + '_' + mnq_h2['sess'].values, index=mnq_h2.index
).reindex(idx)
f['prev_sess_h'] = f['sess_key'].map(sess_map_h)
f['prev_sess_l'] = f['sess_key'].map(sess_map_l)

# Sessao especifica: prev NY, prev London, prev Asia
for sess in ['ny','london','asia']:
    sess_sub = sess_agg[sess_agg['sess_key'].str.endswith('_'+sess)].copy()
    sess_sub['prev_sh2'] = sess_sub['sh'].shift(1)
    sess_sub['prev_sl2'] = sess_sub['sl'].shift(1)
    map_h2 = dict(zip(sess_sub['sess_key'], sess_sub['prev_sh2']))
    map_l2 = dict(zip(sess_sub['sess_key'], sess_sub['prev_sl2']))
    f[f'p{sess[:2]}_h'] = f['sess_key'].map(map_h2).ffill()
    f[f'p{sess[:2]}_l'] = f['sess_key'].map(map_l2).ffill()

f = f.dropna()

# ── Funcao: distancia % para um nivel ────────────────────────────────────────
def near(price, level):
    return (abs(price - level) / price) < NEAR_PCT

# ── Matriz de key levels ──────────────────────────────────────────────────────
KEY_LEVELS = {
    'PDH':        'pdh',
    'PDL':        'pdl',
    'PDO':        'pdo',
    'PWH':        'pwh',
    'PWL':        'pwl',
    'PWO':        'pwo',
    'MON-H':      'mon_h',
    'MON-L':      'mon_l',
    'MON-O':      'mon_o',
    'SESS-H':     'prev_sess_h',
    'SESS-L':     'prev_sess_l',
    'NY-H':       'pny_h',
    'NY-L':       'pny_l',
    'LONDON-H':   'plo_h',
    'LONDON-L':   'plo_l',
    'ASIA-H':     'pas_h',
    'ASIA-L':     'pas_l',
}

for lbl, col in KEY_LEVELS.items():
    if col in f.columns:
        f[f'near_{lbl}'] = near(f['price'], f[col]).astype(int)

near_cols = [c for c in f.columns if c.startswith('near_')]
f['any_key_level'] = (f[near_cols].sum(axis=1) > 0).astype(int)

# ── Resultados ────────────────────────────────────────────────────────────────
entries = f[f['rotacao_new'] == 1].dropna(subset=['fwd_ret'])

print(f"\nPeriodo: {f.index[0].date()} -> {f.index[-1].date()}")
print(f"Total sinais rotacao: {len(entries)}")
baseline_wr = (entries['fwd_ret'] > TARGET).mean() * 100
print(f"WR baseline (todos): {baseline_wr:.1f}%")

print("\n" + "="*52)
print("  RESULTADO POR PROXIMIDADE DE KEY LEVEL")
print("="*52)

at_kl  = entries[entries['any_key_level'] == 1]
no_kl  = entries[entries['any_key_level'] == 0]

def mini_stats(subset, label):
    if len(subset) == 0:
        print(f"  {label}: 0 sinais"); return
    wr  = (subset['fwd_ret'] > TARGET).mean() * 100
    exp = subset['fwd_ret'].mean() * 100
    print(f"  {label:<30} n={len(subset):3d}  WR={wr:5.1f}%  exp={exp:+.3f}%")

mini_stats(at_kl, "Perto de key level")
mini_stats(no_kl, "Sem key level proximo")

print("\n" + "-"*52)
print("  WR POR KEY LEVEL ESPECIFICO (sinais rotacao)")
print("-"*52)

results = []
for lbl, col in KEY_LEVELS.items():
    near_col = f'near_{lbl}'
    if near_col not in entries.columns: continue
    sub = entries[entries[near_col] == 1]
    if len(sub) < 5: continue
    wr  = (sub['fwd_ret'] > TARGET).mean() * 100
    exp = sub['fwd_ret'].mean() * 100
    results.append((lbl, len(sub), wr, exp))

results.sort(key=lambda x: x[2], reverse=True)
for lbl, n, wr, exp in results:
    bar = '#' * int(wr / 10)
    diff = wr - baseline_wr
    sinal = '+' if diff >= 0 else ''
    print(f"  {lbl:<12}  n={n:3d}  WR={wr:5.1f}%  {bar:<10}  exp={exp:+.3f}%  ({sinal}{diff:.1f}pp vs baseline)")

# ── Combinacoes: rotacao + key level especifico ────────────────────────────────
print("\n" + "="*52)
print("  TOP COMBINACOES: ROTACAO + KEY LEVEL")
print("="*52)

combos = []
for lbl, col in KEY_LEVELS.items():
    near_col = f'near_{lbl}'
    if near_col not in entries.columns: continue
    sub = entries[(entries[near_col] == 1) & (entries['c1'] == 1)]
    if len(sub) < 3: continue
    wr  = (sub['fwd_ret'] > TARGET).mean() * 100
    exp = sub['fwd_ret'].mean() * 100
    combos.append((f"ROTACAO+C1+{lbl}", len(sub), wr, exp))

combos.sort(key=lambda x: x[2], reverse=True)
for lbl, n, wr, exp in combos[:8]:
    print(f"  {lbl:<28}  n={n:2d}  WR={wr:5.1f}%  exp={exp:+.3f}%")

# ── Distancia media ao key level mais proximo nos trades vencedores vs perdedores
print("\n" + "-"*52)
print("  DISTANCIA MEDIA AO KEY LEVEL MAIS PROXIMO")
print("-"*52)

def min_dist(row):
    dists = []
    for lbl, col in KEY_LEVELS.items():
        if col in f.columns and not pd.isna(row.get(col, np.nan)):
            d = abs(row['price'] - row[col]) / row['price'] * 100
            dists.append(d)
    return min(dists) if dists else np.nan

entries2 = entries.copy()
for col in [c for _, c in KEY_LEVELS.items() if c in f.columns]:
    entries2[col] = f[col].reindex(entries2.index)
entries2['min_dist_kl'] = entries2.apply(min_dist, axis=1)

wins  = entries2[entries2['fwd_ret'] > TARGET]
losss = entries2[entries2['fwd_ret'] <= TARGET]
print(f"  Vencedores: dist media ao KL mais proximo = {wins['min_dist_kl'].mean():.3f}%")
print(f"  Perdedores: dist media ao KL mais proximo = {losss['min_dist_kl'].mean():.3f}%")
print(f"  Threshold testado: {NEAR_PCT*100:.1f}%")

# Testar varios thresholds
print("\n  WR por threshold de proximidade:")
for thr in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]:
    sub = entries2[entries2['min_dist_kl'] <= thr*100]
    if len(sub) < 5: continue
    wr = (sub['fwd_ret'] > TARGET).mean() * 100
    print(f"    dist < {thr*100:.1f}%:  n={len(sub):3d}  WR={wr:.1f}%")

print()

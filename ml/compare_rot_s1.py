import warnings; warnings.filterwarnings('ignore')
import yfinance as yf, pandas as pd, numpy as np, ta
from pathlib import Path
from datetime import datetime, timedelta

def download(ticker):
    end = datetime.now()
    start = end - timedelta(days=720)
    df = yf.download(ticker, start=start, end=end, interval='1h', auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()

def compute(df):
    d = df.copy()
    d['rsi']  = ta.momentum.RSIIndicator(d['close'], window=21).rsi()
    d['ret1'] = d['close'].pct_change(1)
    d['vol']  = d['ret1'].rolling(20).std()
    d['bb_w'] = d['close'].rolling(20).std() * 2 / d['close'].rolling(20).mean()
    d['sma50']= d['close'].rolling(50).mean()
    d['above_sma50'] = (d['close'] > d['sma50']).astype(int)
    adx_i = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=17)
    d['adx'] = adx_i.adx()
    return d

print('Baixando...')
mnq_raw = download('MNQ=F')
mnq = compute(mnq_raw)
es  = compute(download('ES=F'))
cl  = compute(download('CL=F'))
cl_raw = download('CL=F')
btc = compute(download('BTC-USD'))

idx = mnq.index.intersection(es.index).intersection(cl.index).intersection(btc.index)
mnq=mnq.loc[idx]; es=es.loc[idx]; cl=cl.loc[idx]; btc=btc.loc[idx]
mnq_raw = mnq_raw.reindex(idx, method='ffill')
cl_raw  = cl_raw.reindex(idx, method='ffill')

f = pd.DataFrame(index=idx)
f['close']       = mnq['close']
f['rsi_mnq']     = mnq['rsi']
f['rsi_cl']      = cl['rsi']
f['adx_mnq']     = mnq['adx']
f['bb_w_cl']     = cl['bb_w'] * 100
f['above50_mnq'] = mnq['above_sma50']
f['above50_es']  = es['above_sma50']
f['above50_cl']  = cl['above_sma50']
f['above50_btc'] = btc['above_sma50']
f['ret1_mnq']    = mnq['ret1'] * 100
f['ret1_es']     = es['ret1'] * 100
f['sma200']      = mnq['close'].rolling(200).mean()

f['div_cl']       = f['rsi_mnq'] - f['rsi_cl']
f['div_delta_2h'] = f['div_cl'] - f['div_cl'].shift(2)
f['rsi_mnq_d2']   = f['rsi_mnq'] - f['rsi_mnq'].shift(2)
f['rsi_cl_d2']    = f['rsi_cl']  - f['rsi_cl'].shift(2)

cl_dopen     = cl_raw['open'].resample('D').first().reindex(idx, method='ffill')
cl_dopen_ant = cl_dopen.shift(1)
f['cl_d_acima_ant'] = (cl_dopen > cl_dopen_ant).astype(int)

open_4h = mnq_raw['open'].resample('4h').first().reindex(idx, method='ffill')
f['open_4h']     = open_4h
f['above_4h']    = (f['close'] > f['open_4h']).astype(int)
f['regime_bull'] = (f['close'] > f['sma200']).astype(int)
f['sma50_align'] = f['above50_mnq']+f['above50_es']+f['above50_cl']+f['above50_btc']

f['fwd4']  = mnq['close'].shift(-4) / mnq['close'] - 1
f['target']= (f['fwd4'] > 0.001).astype(int)
f['hour']  = idx.hour
f = f.dropna()
f = f[f['ret1_mnq'].abs() <= 3.0].copy()

print(f'Dataset: {len(f)} barras  {f.index[0].date()} -> {f.index[-1].date()}')
baseline = float(f['target'].mean())
print(f'Baseline: {baseline:.1%}\n')

# Rot score
prev_div = f['div_cl'].shift(1)
c1 = (((f['div_cl']>0)&(prev_div<=0))|((f['div_cl']<0)&(prev_div>=0))).astype(int)
c2 = (f['div_delta_2h'].abs()>5.0).astype(int)
c3 = ((f['rsi_mnq_d2']>2.0)&(f['rsi_cl_d2']<-2.0)).astype(int)
c4 = (((f['ret1_mnq']>0)&(f['ret1_es']>0))|((f['ret1_mnq']<0)&(f['ret1_es']<0))).astype(int)
c5 = (f['bb_w_cl']>1.5).astype(int)
c6 = f['cl_d_acima_ant']
c7 = ((f['adx_mnq']>=12)&(f['adx_mnq']<=20)).astype(int)
c8 = (f['sma50_align']>=2).astype(int)
c9 = ((f['rsi_mnq']>55)|(f['rsi_mnq']<45)).astype(int)

rot_raw = c1*3.0+c2*2.5+c3*2.5+c4*2.0+c5*1.5+c6*1.5+c7*1.5+c8*1.5+c9*1.0
f['rot_score'] = (rot_raw/17.0*100).round(1)
f['c1'] = c1

# Mascaras
mask_rot = (f['rot_score']>=60) & (f['div_cl']>0) & (f['c1']==1)
mask_s1  = (f['rot_score']>=60) & (f['div_cl']>0) & (f['above_4h']==1) & (f['regime_bull']==1)

n_rot = int(mask_rot.sum()); wr_rot = float(f.loc[mask_rot,'target'].mean())
n_s1  = int(mask_s1.sum());  wr_s1  = float(f.loc[mask_s1,'target'].mean())

print('='*60)
print('COMPARACAO: 720 dias, sem ML, mesmos dados')
print('='*60)
print(f'{"":30} {"N":>5}  {"WR":>6}  {"Edge":>7}  {"freq/semana"}')
print('-'*60)
weeks = (f.index[-1]-f.index[0]).days / 7
print(f'Baseline (qualquer barra){"":6} {len(f):>5}  {baseline:.1%}  {"--":>7}  --')
print(f'Rotacao puro (score+C1+dir){"":4} {n_rot:>5}  {wr_rot:.1%}  {wr_rot-baseline:>+.1%}  {n_rot/weeks:.1f}/sem')
print(f'S1 (score+4h+regime){"":10} {n_s1:>5}  {wr_s1:.1%}  {wr_s1-baseline:>+.1%}  {n_s1/weeks:.1f}/sem')

mask_both = mask_rot & mask_s1
mask_rot_only = mask_rot & ~mask_s1
mask_s1_only  = mask_s1 & ~mask_rot
n_b  = int(mask_both.sum())
n_ro = int(mask_rot_only.sum())
n_so = int(mask_s1_only.sum())
wr_b  = float(f.loc[mask_both,'target'].mean()) if n_b>0 else 0
wr_ro = float(f.loc[mask_rot_only,'target'].mean()) if n_ro>0 else 0
wr_so = float(f.loc[mask_s1_only,'target'].mean()) if n_so>0 else 0

print()
print('SOBREPOSICAO:')
print(f'  Ambos concordam       N={n_b:>4}  WR={wr_b:.1%}  Edge={wr_b-baseline:+.1%}')
print(f'  So Rotacao (S1 perde) N={n_ro:>4}  WR={wr_ro:.1%}  Edge={wr_ro-baseline:+.1%}')
print(f'  So S1 (Rot perde)     N={n_so:>4}  WR={wr_so:.1%}  Edge={wr_so-baseline:+.1%}')

print()
print('ROTACAO: div_cl no momento do trigger (entrada recente vs consolidada)')
for dmin, dmax, label in [(0,5,'0-5 CRUZAMENTO RECENTE'),(5,15,'5-15 divergindo'),(15,30,'15-30 consolidado'),(30,999,'>=30 divergencia forte')]:
    m = mask_rot & (f['div_cl']>=dmin) & (f['div_cl']<dmax)
    n = int(m.sum())
    if n < 3: continue
    wr = float(f.loc[m,'target'].mean())
    print(f'  div_cl {label:<28} N={n:>4}  WR={wr:.1%}  Edge={wr-baseline:+.1%}')

print()
print('S1: div_cl no momento do trigger')
for dmin, dmax, label in [(0,5,'0-5 CRUZAMENTO RECENTE'),(5,15,'5-15 divergindo'),(15,30,'15-30 consolidado'),(30,999,'>=30 divergencia forte')]:
    m = mask_s1 & (f['div_cl']>=dmin) & (f['div_cl']<dmax)
    n = int(m.sum())
    if n < 3: continue
    wr = float(f.loc[m,'target'].mean())
    print(f'  div_cl {label:<28} N={n:>4}  WR={wr:.1%}  Edge={wr-baseline:+.1%}')

print()
print('PERFIL TEMPORAL: quando cada trigger dispara mais')
print(f'{"Hora":>5}  {"Rot N":>6}  {"Rot WR":>7}  {"S1 N":>5}  {"S1 WR":>6}')
for h in [9,10,11,12,13,14,15,16,17]:
    mr = mask_rot & (f['hour']==h)
    ms = mask_s1  & (f['hour']==h)
    nr=int(mr.sum()); ns=int(ms.sum())
    if nr==0 and ns==0: continue
    wr_r2 = float(f.loc[mr,'target'].mean()) if nr>0 else 0
    wr_s2 = float(f.loc[ms,'target'].mean()) if ns>0 else 0
    print(f'  h={h:02d}   {nr:>5}   {wr_r2:.0%}      {ns:>5}   {wr_s2:.0%}')

print()
print('CHAVE: o que C1 (cruzamento do zero) faz ao S1?')
m_s1_c1  = mask_s1 &  (f['c1']==1)
m_s1_nc1 = mask_s1 & ~(f['c1']==1)
wr_c1  = float(f.loc[m_s1_c1,  'target'].mean())
wr_nc1 = float(f.loc[m_s1_nc1, 'target'].mean())
print(f'  S1 com C1 (cruzamento recente):  N={m_s1_c1.sum():>4}  WR={wr_c1:.1%}')
print(f'  S1 sem C1 (divergencia ativa):   N={m_s1_nc1.sum():>4}  WR={wr_nc1:.1%}')

print()
print('QUAL COMBINA OS DOIS? S1 + C1 (cruzamento + 4h + regime):')
mask_comb = mask_s1 & (f['c1']==1)
n_c = int(mask_comb.sum())
wr_c = float(f.loc[mask_comb,'target'].mean()) if n_c>0 else 0
print(f'  S1 + C1  N={n_c}  WR={wr_c:.1%}  Edge={wr_c-baseline:+.1%}  freq={n_c/weeks:.1f}/sem')

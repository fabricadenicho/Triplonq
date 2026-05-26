"""
Backtest -- Estrategias com Abertura 4H + filtro ML
Testa 6 estrategias combinando score de rotacao com posicao relativa
ao 4H open, com e sem confirmacao do modelo ML (model_divergencia.pkl).

Split: 70% referencia | 20% validacao | 10% teste final (out-of-sample)

IMPORTANTE: modelo carregado em modo leitura -- nada e modificado.
"""
import warnings; warnings.filterwarnings('ignore')
import pickle, sys
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from pathlib import Path
from datetime import datetime, timedelta

PERIOD  = 365   # dias de historico (precisa de volume para split 70/20/10)
FWD     = 4     # horas forward para medir resultado
TARGET  = 0.001 # >0.1% = win
ML_THR  = 0.50  # threshold de probabilidade do modelo

BASE       = Path(__file__).parent.parent          # ml/
MODEL_PATH = BASE / 'model_divergencia.pkl'

# -- Carregar modelo (somente leitura) -----------------------------------------
model      = None
feat_cols  = []
ml_baseline= 0.246

if MODEL_PATH.exists():
    with open(MODEL_PATH, 'rb') as fh:
        md = pickle.load(fh)
    model     = md['model']
    feat_cols = md['features']
    ml_baseline = md.get('baseline', 0.246)
    print(f"Modelo carregado: {len(feat_cols)} features  |  baseline={ml_baseline:.3f}")
else:
    print(f"AVISO: {MODEL_PATH} nao encontrado -- rodando SEM ML")

# -- Download ------------------------------------------------------------------
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
raw = {k: download(t) for k, t in {
    'mnq': 'MNQ=F', 'cl': 'CL=F', 'es': 'ES=F', 'btc': 'BTC-USD'
}.items()}
if any(v is None for v in raw.values()):
    print("ERRO: falha ao baixar algum ativo"); sys.exit(1)

# -- Indicadores base ----------------------------------------------------------
def compute(df):
    d = df.copy()
    d['rsi']       = ta.momentum.RSIIndicator(d['close'], window=21).rsi()
    adx_i          = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=17)
    d['adx']       = adx_i.adx()
    d['pdi']       = adx_i.adx_pos()
    d['mdi']       = adx_i.adx_neg()
    d['ret1']      = d['close'].pct_change(1)
    d['ret4']      = d['close'].pct_change(4)
    d['ret8']      = d['close'].pct_change(8)
    d['vol']       = d['ret1'].rolling(20).std()
    d['bb_w']      = d['close'].rolling(20).std() * 2 / d['close'].rolling(20).mean()
    d['sma50']     = d['close'].rolling(50).mean()
    d['dist_sma50']= (d['close'] - d['sma50']) / d['sma50'] * 100
    d['above_sma50']= (d['close'] > d['sma50']).astype(int)
    d['ema20']     = d['close'].ewm(span=20, adjust=False).mean()
    d['dist_ema20']= (d['close'] - d['ema20']) / d['ema20'] * 100
    d['above_ema20']= (d['close'] > d['ema20']).astype(int)
    return d

c = {k: compute(raw[k]) for k in raw}

idx = c['mnq'].index
for k in ['cl','es','btc']:
    idx = idx.intersection(c[k].index)
for k in c:
    c[k] = c[k].loc[idx]

# -- Features completas (espelho do predict_divergencia.py) --------------------
f = pd.DataFrame(index=idx)

f['mnq'] = c['mnq']['close']
f['es']  = c['es']['close']
f['btc'] = c['btc']['close']
f['cl']  = c['cl']['close']

for nome, s in [('mnq',c['mnq']),('es',c['es']),('btc',c['btc']),('cl',c['cl'])]:
    f[f'r_{nome}_1h'] = s['ret1'] * 100
    f[f'r_{nome}_4h'] = s['ret4'] * 100
f['r_mnq_8h'] = c['mnq']['ret8'] * 100

f['es_mnq_mesmo']  = (((f['r_es_1h']>0)&(f['r_mnq_1h']>0))|((f['r_es_1h']<0)&(f['r_mnq_1h']<0))).astype(int)
f['es_mnq_oposto'] = (((f['r_es_1h']>0)&(f['r_mnq_1h']<0))|((f['r_es_1h']<0)&(f['r_mnq_1h']>0))).astype(int)
f['btc_mnq_mesmo'] = (((f['r_btc_1h']>0)&(f['r_mnq_1h']>0))|((f['r_btc_1h']<0)&(f['r_mnq_1h']<0))).astype(int)
f['btc_mnq_oposto']= (((f['r_btc_1h']>0)&(f['r_mnq_1h']<0))|((f['r_btc_1h']<0)&(f['r_mnq_1h']>0))).astype(int)
f['es_btc_discordam_mnq'] = (f['es_mnq_oposto'] & f['btc_mnq_oposto']).astype(int)
f['es_btc_concordam_mnq'] = (f['es_mnq_mesmo']  & f['btc_mnq_mesmo']).astype(int)

for nome in ['mnq','es','btc','cl']:
    f[f'rsi_{nome}'] = c[nome]['rsi']
f['div_cl']  = f['rsi_mnq'] - f['rsi_cl']
f['div_es']  = f['rsi_mnq'] - f['rsi_es']
f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']
f['rsi_mnq_acima_60']  = (f['rsi_mnq'] > 60).astype(int)
f['rsi_mnq_abaixo_40'] = (f['rsi_mnq'] < 40).astype(int)

for nome in ['mnq','es','btc','cl']:
    s = raw[nome]['open'].reindex(idx, method='ffill')
    f[f'open_1h_{nome}'] = s
    f[f'open_1h_acima_close_ant_{nome}'] = (s > f[nome].shift(1)).astype(int)

for nome in ['mnq','es','btc','cl']:
    s  = f[f'open_1h_{nome}']
    o4 = s.groupby(idx.floor('4h')).transform('first')
    ca4= s.shift(1).groupby(idx.floor('4h')).transform('first')
    f[f'open_4h_acima_4h_ant_{nome}'] = (o4 > ca4).astype(int)
    f[f'open_4h_dist_{nome}']         = (s - o4) / o4 * 100

for nome in ['mnq','es','btc','cl']:
    s = f[f'open_1h_{nome}']
    d = s.groupby(idx.date).transform('first')
    ca= s.shift(1).groupby(idx.date).transform('first')
    f[f'open_d_acima_d_ant_{nome}'] = (d > ca).astype(int)
    f[f'open_d_dist_{nome}']        = (s - d) / d * 100

for nome in ['mnq','es','btc','cl']:
    s  = f[f'open_1h_{nome}']
    w  = s.groupby(pd.Grouper(freq='W-MON')).transform('first')
    wc = s.shift(1).groupby(pd.Grouper(freq='W-MON')).transform('first')
    f[f'open_w_acima_w_ant_{nome}'] = (w > wc).astype(int)
    f[f'open_w_dist_{nome}']        = (s - w) / w * 100

f['open_mnq_acima_cl']  = (f['open_1h_mnq'] > f['open_1h_cl']).astype(int)
f['open_mnq_acima_es']  = (f['open_1h_mnq'] > f['open_1h_es']).astype(int)
f['open_mnq_acima_btc'] = (f['open_1h_mnq'] > f['open_1h_btc']).astype(int)

for nome in ['mnq','es','btc','cl']:
    f[f'adx_{nome}']      = c[nome]['adx']
    f[f'di_spread_{nome}']= c[nome]['pdi'] - c[nome]['mdi']
    f[f'above_sma50_{nome}']= c[nome]['above_sma50']
    f[f'above_ema20_{nome}']= c[nome]['above_ema20']
    f[f'dist_sma50_{nome}'] = c[nome]['dist_sma50']
    f[f'dist_ema20_{nome}'] = c[nome]['dist_ema20']
    f[f'vol_{nome}']        = c[nome]['vol'] * 100
    f[f'bb_w_{nome}']       = c[nome]['bb_w'] * 100

f['adx_mnq_alto'] = (f['adx_mnq'] > 14).astype(int)
f['adx_cl_alto']  = (f['adx_cl']  > 14).astype(int)
f['adx_es_alto']  = (f['adx_es']  > 14).astype(int)
f['sma50_alignment'] = sum(f[f'above_sma50_{n}'] for n in ['mnq','es','btc','cl'])
f['ema20_alignment'] = sum(f[f'above_ema20_{n}'] for n in ['mnq','es','btc','cl'])

f['hour']       = idx.hour
f['dow']        = idx.dayofweek
f['is_us']      = f['hour'].between(9, 17).astype(int)
f['is_asia']    = f['hour'].between(0,  8).astype(int)
f['is_evening'] = f['hour'].between(18, 23).astype(int)

# -- Predicao ML para todas as barras -----------------------------------------
f_clean = f.dropna().copy()
if model is not None and len(feat_cols) > 0:
    X = f_clean[[c2 for c2 in feat_cols if c2 in f_clean.columns]].fillna(0)
    f_clean['ml_prob'] = model.predict_proba(X)[:, 1]
else:
    f_clean['ml_prob'] = 0.5  # sem modelo: neutro

# -- Abertura 4H e score rotacao -----------------------------------------------
mnq_4h_open         = raw['mnq']['open'].resample('4h').first()
f_clean['open_4h']  = mnq_4h_open.reindex(f_clean.index, method='ffill')
f_clean['above_4h'] = (f_clean['mnq'] > f_clean['open_4h']).astype(int)
f_clean['cross_4h_up']  = ((f_clean['above_4h']==1)&(f_clean['above_4h'].shift(1)==0)).astype(int)
f_clean['cross_4h_down']= ((f_clean['above_4h']==0)&(f_clean['above_4h'].shift(1)==1)).astype(int)
f_clean['dist_4h_pct']  = (f_clean['mnq'] - f_clean['open_4h']) / f_clean['open_4h'] * 100

cl_dopen_daily           = c['cl']['open'].resample('D').first()
f_clean['cl_dopen']      = cl_dopen_daily.reindex(f_clean.index, method='ffill')
f_clean['cl_dopen_ant']  = cl_dopen_daily.shift(1).reindex(f_clean.index, method='ffill')

f_clean['div_delta_2h']    = f_clean['div_cl'] - f_clean['div_cl'].shift(2)
f_clean['div_sinal_mudou'] = (((f_clean['div_cl']>0)&(f_clean['div_cl'].shift(1)<=0))|
                               ((f_clean['div_cl']<0)&(f_clean['div_cl'].shift(1)>=0))).astype(int)
f_clean['rsi_mnq_d2'] = f_clean['rsi_mnq'] - f_clean['rsi_mnq'].shift(2)
f_clean['rsi_cl_d2']  = f_clean['rsi_cl']  - f_clean['rsi_cl'].shift(2)
f_clean['rot_ativa']  = ((f_clean['rsi_mnq_d2']>2.0)&(f_clean['rsi_cl_d2']<-2.0)).astype(int)
f_clean['above50_mnq']= c['mnq']['above_sma50'].reindex(f_clean.index)
f_clean['above50_cl'] = c['cl']['above_sma50'].reindex(f_clean.index)
f_clean['above50_es'] = c['es']['above_sma50'].reindex(f_clean.index)
f_clean['above50_btc']= c['btc']['above_sma50'].reindex(f_clean.index)

f_clean['c1'] = f_clean['div_sinal_mudou']
f_clean['c2'] = (f_clean['div_delta_2h'] > 5.0).astype(int)
f_clean['c3'] = f_clean['rot_ativa']
f_clean['c4'] = (((f_clean['r_mnq_1h']>0)&(f_clean['r_es_1h']>0))|
                  ((f_clean['r_mnq_1h']<0)&(f_clean['r_es_1h']<0))).astype(int)
f_clean['c5'] = (f_clean['bb_w_cl'] > 1.5).astype(int)
f_clean['c6'] = (f_clean['cl_dopen'] > f_clean['cl_dopen_ant']).astype(int)
f_clean['c7'] = ((f_clean['adx_mnq']>=12)&(f_clean['adx_mnq']<=20)).astype(int)
f_clean['c8'] = ((f_clean['above50_mnq']+f_clean['above50_cl']+
                  f_clean['above50_es']+f_clean['above50_btc'])>=2).astype(int)
f_clean['c9'] = ((f_clean['rsi_mnq']>55)|(f_clean['rsi_mnq']<45)).astype(int)

f_clean['sc']        = (f_clean['c1']*3.0+f_clean['c2']*2.5+f_clean['c3']*2.5+
                        f_clean['c4']*2.0+f_clean['c5']*1.5+f_clean['c6']*1.5+
                        f_clean['c7']*1.5+f_clean['c8']*1.5+f_clean['c9']*1.0)
f_clean['score_pct'] = f_clean['sc'] / 17.0 * 100
f_clean['rotacao']   = (f_clean['score_pct'] >= 60).astype(int)
f_clean['rot_new']   = ((f_clean['rotacao']==1)&(f_clean['rotacao'].shift(1)==0)).astype(int)
f_clean['dir_long']  = (f_clean['div_cl'] > 0).astype(int)

f_clean['fwd_ret'] = f_clean['mnq'].pct_change(FWD).shift(-FWD)
f_clean = f_clean.dropna(subset=['fwd_ret'])

# -- Split 70 / 20 / 10 -------------------------------------------------------
n      = len(f_clean)
n70    = int(n * 0.70)
n90    = int(n * 0.90)
splits = {
    'Referencia 70%': f_clean.iloc[:n70],
    'Validacao  20%': f_clean.iloc[n70:n90],
    'Teste      10%': f_clean.iloc[n90:],
}

# -- Definicao das 6 estrategias -----------------------------------------------
def get_entries(df):
    return {
        'S1 ROT-LONG  + acima 4H':   df[(df['rot_new']==1)&(df['dir_long']==1)&(df['above_4h']==1)],
        'S2 ROT-LONG  + abaixo 4H':  df[(df['rot_new']==1)&(df['dir_long']==1)&(df['above_4h']==0)],
        'S3 ROT-SHORT + abaixo 4H':  df[(df['rot_new']==1)&(df['dir_long']==0)&(df['above_4h']==0)],
        'S4 ROT-SHORT + acima 4H':   df[(df['rot_new']==1)&(df['dir_long']==0)&(df['above_4h']==1)],
        'S5 ROT-LONG  + cruz cima':  df[(df['rot_new']==1)&(df['dir_long']==1)&(df['cross_4h_up']==1)],
        'S6 ROT-SHORT + cruz baixo': df[(df['rot_new']==1)&(df['dir_long']==0)&(df['cross_4h_down']==1)],
    }

def get_entries_ml(df):
    return {
        'S1 ROT-LONG  + acima 4H':   df[(df['rot_new']==1)&(df['dir_long']==1)&(df['above_4h']==1)&(df['ml_prob']>=ML_THR)],
        'S2 ROT-LONG  + abaixo 4H':  df[(df['rot_new']==1)&(df['dir_long']==1)&(df['above_4h']==0)&(df['ml_prob']>=ML_THR)],
        'S3 ROT-SHORT + abaixo 4H':  df[(df['rot_new']==1)&(df['dir_long']==0)&(df['above_4h']==0)&(df['ml_prob']< ML_THR)],
        'S4 ROT-SHORT + acima 4H':   df[(df['rot_new']==1)&(df['dir_long']==0)&(df['above_4h']==1)&(df['ml_prob']< ML_THR)],
        'S5 ROT-LONG  + cruz cima':  df[(df['rot_new']==1)&(df['dir_long']==1)&(df['cross_4h_up']==1)&(df['ml_prob']>=ML_THR)],
        'S6 ROT-SHORT + cruz baixo': df[(df['rot_new']==1)&(df['dir_long']==0)&(df['cross_4h_down']==1)&(df['ml_prob']< ML_THR)],
    }

# -- Funcao de stats -----------------------------------------------------------
def wr_stats(entries, direction='long', bl=None):
    e = entries.dropna(subset=['fwd_ret'])
    if len(e) == 0:
        return {'n': 0, 'wr': float('nan'), 'exp': float('nan'),
                'alpha': float('nan'), 'ret': float('nan')}
    ret  = e['fwd_ret'] if direction == 'long' else -e['fwd_ret']
    wins = ret > TARGET
    wr   = wins.mean() * 100
    exp  = ret.mean() * 100
    rt   = ret.sum() * 100
    alpha= wr - bl if bl is not None else float('nan')
    return {'n': len(e), 'wr': wr, 'exp': exp, 'alpha': alpha, 'ret': rt}

# -- Output --------------------------------------------------------------------
print(f"\n{'='*80}")
print(f"  BACKTEST -- ROTACAO + ABERTURA 4H  |  {PERIOD} dias  |  fwd={FWD}h  |  ML_THR={ML_THR}")
print(f"  Total barras: {n}  |  Referencia ate: {f_clean.index[n70].date()}  |  Teste de: {f_clean.index[n90].date()}")
print(f"{'='*80}")

for split_nome, df_split in splits.items():
    bl = (df_split['fwd_ret'] > TARGET).mean() * 100
    print(f"\n  -- {split_nome}  ({len(df_split)} barras  |  baseline={bl:.1f}%)  ----------------------")
    print(f"  {'Estrategia':<32s}  {'N':>4s}  {'WR':>6s}  {'Alpha':>7s}  {'Exp/tr':>7s}  |  {'N+ML':>4s}  {'WR+ML':>6s}  {'Alpha':>7s}")
    print(f"  {'-'*32}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*7}  |  {'-'*4}  {'-'*6}  {'-'*7}")

    entries    = get_entries(df_split)
    entries_ml = get_entries_ml(df_split) if model else {}

    for nome in entries:
        d      = 'short' if 'SHORT' in nome else 'long'
        r      = wr_stats(entries[nome],    d, bl)
        r_ml   = wr_stats(entries_ml.get(nome, entries[nome].iloc[0:0]), d, bl) if model else {'n':0,'wr':float('nan'),'alpha':float('nan')}

        def fmt(v, sfx=''):
            return f"{v:.1f}{sfx}" if not np.isnan(v) else '--'

        flag    = ' <<' if (not np.isnan(r['alpha'])    and r['alpha']    > 5 and r['n']    >= 5) else ''
        flag_ml = ' <<' if (not np.isnan(r_ml['alpha']) and r_ml['alpha'] > 5 and r_ml['n'] >= 5) else ''

        print(f"  {nome:<32s}  {r['n']:>4d}  {fmt(r['wr'],'%'):>6s}  {fmt(r['alpha'],'+' if not np.isnan(r['alpha']) and r['alpha']>=0 else ''):>6s}pp"
              f"  {fmt(r['exp'],'%'):>7s}  |  {r_ml['n']:>4d}  {fmt(r_ml['wr'],'%'):>6s}  {fmt(r_ml['alpha'],'+' if not np.isnan(r_ml['alpha']) and r_ml['alpha']>=0 else ''):>6s}pp{flag_ml or flag}")

print(f"\n{'='*80}")

# -- Detalhamento do Teste 10% por hora ---------------------------------------
df_test = splits['Teste      10%']
bl_test = (df_test['fwd_ret'] > TARGET).mean() * 100
entries_test    = get_entries(df_test)
entries_test_ml = get_entries_ml(df_test) if model else {}

print(f"\n  DETALHAMENTO TESTE 10% -- por hora de entrada (UTC)")
print(f"  Periodo: {df_test.index[0].date()} -> {df_test.index[-1].date()}  |  baseline={bl_test:.1f}%")

for nome in entries_test:
    d  = 'short' if 'SHORT' in nome else 'long'
    e  = entries_test[nome].dropna(subset=['fwd_ret'])
    if len(e) == 0: continue
    e2 = e.copy()
    e2['ret_dir'] = e2['fwd_ret'] if d == 'long' else -e2['fwd_ret']
    e2['win']     = e2['ret_dir'] > TARGET
    overall_wr    = e2['win'].mean() * 100
    print(f"\n  {nome}  (n={len(e)}  WR={overall_wr:.1f}%  alpha={overall_wr-bl_test:+.1f}pp)")
    e2['hora'] = e2.index.hour
    for h, g in e2.groupby('hora'):
        wr_h = g['win'].mean() * 100
        bar  = '#' * int(wr_h / 10)
        print(f"    {int(h):02d}h  n={len(g):2d}  WR={wr_h:4.0f}%  {bar:<10s}  exp={g['ret_dir'].mean()*100:+.4f}%")

print(f"\n{'='*80}\n")

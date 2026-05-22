"""
Backtest — Trigger Rotacao MNQ×CL (últimos 6 meses, 1h)
Entry: score >= 60%  |  Exit: 4h fixo (forward window do modelo)
"""
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

PERIOD = 185  # dias
FWD    = 4    # horas forward (modelo treinado neste horizonte)
TARGET = 0.001  # MNQ >0.1% = win

# -- Download -----------------------------------------------------------------
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
raw = {k: download(t) for k, t in {'mnq':'MNQ=F','cl':'CL=F','es':'ES=F','btc':'BTC-USD'}.items()}
if any(v is None for v in raw.values()):
    print("ERRO: falha ao baixar algum ativo"); exit()

# -- Indicadores ---------------------------------------------------------------
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

# -- Alinhar índices -----------------------------------------------------------
idx = c['mnq'].index
for k in ['cl','es','btc']:
    idx = idx.intersection(c[k].index)
for k in c:
    c[k] = c[k].loc[idx]

# -- Features ------------------------------------------------------------------
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

# CL daily open vs anterior
cl_dopen_daily   = c['cl']['open'].resample('D').first()
f['cl_dopen']    = cl_dopen_daily.reindex(idx, method='ffill')
f['cl_dopen_ant']= cl_dopen_daily.shift(1).reindex(idx, method='ffill')

# -- Divergência e rotacao -----------------------------------------------------
f['div_cl']          = f['rsi_mnq'] - f['rsi_cl']
f['div_delta_2h']    = f['div_cl'] - f['div_cl'].shift(2)
f['div_sinal_mudou'] = (((f['div_cl']>0)&(f['div_cl'].shift(1)<=0)) |
                        ((f['div_cl']<0)&(f['div_cl'].shift(1)>=0))).astype(int)
f['rsi_mnq_d2']      = f['rsi_mnq'] - f['rsi_mnq'].shift(2)
f['rsi_cl_d2']       = f['rsi_cl']  - f['rsi_cl'].shift(2)
f['rot_ativa']       = ((f['rsi_mnq_d2']>2.0)&(f['rsi_cl_d2']<-2.0)).astype(int)

# -- 9 Condicões ---------------------------------------------------------------
f['c1'] = f['div_sinal_mudou']
f['c2'] = (f['div_delta_2h'] > 5.0).astype(int)
f['c3'] = f['rot_ativa']
f['c4'] = (((f['ret1_mnq']>0)&(f['ret1_es']>0))|((f['ret1_mnq']<0)&(f['ret1_es']<0))).astype(int)
f['c5'] = (f['bb_w_cl'] > 1.5).astype(int)
f['c6'] = (f['cl_dopen'] > f['cl_dopen_ant']).astype(int)
f['c7'] = ((f['adx_mnq']>=12)&(f['adx_mnq']<=20)).astype(int)
f['c8'] = ((f['above50_mnq']+f['above50_cl']+f['above50_es']+f['above50_btc'])>=2).astype(int)
f['c9'] = ((f['rsi_mnq']>55)|(f['rsi_mnq']<45)).astype(int)

f['sc']        = f['c1']*3.0+f['c2']*2.5+f['c3']*2.5+f['c4']*2.0+f['c5']*1.5+f['c6']*1.5+f['c7']*1.5+f['c8']*1.5+f['c9']*1.0
f['score_pct'] = f['sc'] / 17.0 * 100
f['eixo']      = ((f['div_delta_2h'].abs()>5.0)&(f['rot_ativa']==1)&(f['div_cl'].abs()<20.0)).astype(int)
f['rotacao']   = (f['score_pct'] >= 60).astype(int)

f = f.dropna()

# -- Retornos forward ----------------------------------------------------------
f['fwd_ret']   = f['mnq_close'].pct_change(FWD).shift(-FWD)
f['rotacao_new']= ((f['rotacao']==1)&(f['rotacao'].shift(1)==0)).astype(int)
f['eixo_new']   = ((f['eixo']==1)  &(f['eixo'].shift(1)==0)).astype(int)

# -- Resultado ROTACAO ---------------------------------------------------------
def stats(entries, label):
    e = entries.dropna(subset=['fwd_ret'])
    if len(e) == 0:
        print(f"\n{label}: 0 sinais"); return
    wins  = e['fwd_ret'] > TARGET
    wn    = e.loc[wins, 'fwd_ret']*100
    ls    = e.loc[~wins,'fwd_ret']*100
    print(f"\n{'-'*46}")
    print(f"  {label}")
    print(f"{'-'*46}")
    print(f"  Periodo:       {e.index[0].date()} -> {e.index[-1].date()}")
    print(f"  Sinais:        {len(e)}")
    print(f"  Win Rate:      {wins.mean()*100:.1f}%  (>{TARGET*100:.1f}% em {FWD}h)")
    print(f"  Ganho medio:   +{wn.mean():.3f}%  |  Perda media: {ls.mean():.3f}%")
    print(f"  Expectativa:   {(e['fwd_ret']*100).mean():.4f}% por trade")
    print(f"  Retorno total: {(e['fwd_ret']*100).sum():.2f}%")
    print(f"  Máx. ganho:    +{wn.max():.3f}%  |  Máx. perda: {ls.min():.3f}%")

    # Por mês
    print(f"\n  Por mês:")
    e2 = e.copy(); e2['win'] = e['fwd_ret'] > TARGET
    e2['mes'] = e2.index.to_period('M')
    for m, g in e2.groupby('mes'):
        wr_m = g['win'].mean()*100
        ret_m= (g['fwd_ret']*100).sum()
        bar  = '#' * int(wr_m/10)
        sinal= '+' if ret_m >= 0 else ''
        print(f"    {m}  sinais={len(g):3d}  WR={wr_m:4.0f}%  {bar}  ret={sinal}{ret_m:.2f}%")

    # Por condicao ativa
    print(f"\n  Condicões mais ativas nos sinais:")
    for col in ['c1','c2','c3','c4','c5','c6','c7','c8','c9']:
        pct = e[col].mean()*100
        if pct > 0:
            print(f"    {col}: {pct:.0f}% dos sinais")

rot_entries  = f[f['rotacao_new']==1]
eixo_entries = f[f['eixo_new']==1]

stats(rot_entries,  f"ROTACAO  (score >= 60%,  forward {FWD}h)")
stats(eixo_entries, f"EIXO     (delta>5 + rot_ativa + div<20)")

# -- Baseline (aleatório) ------------------------------------------------------
baseline_wr = (f['fwd_ret'] > TARGET).mean()*100
print(f"\n{'-'*46}")
print(f"  Baseline (qualquer barra):  WR = {baseline_wr:.1f}%")
print(f"{'-'*46}\n")

# -- In-Sample vs Out-of-Sample (70/30) ----------------------------------------
split_idx = int(len(f) * 0.70)
f_in  = f.iloc[:split_idx]
f_out = f.iloc[split_idx:]

rot_in  = f_in [f_in ['rotacao_new']==1]
rot_out = f_out[f_out['rotacao_new']==1]

stats(rot_in,  f"IN-SAMPLE   70%  ROTACAO (score >= 60%)")
stats(rot_out, f"OUT-OF-SAMPLE 30% ROTACAO (score >= 60%)")

# -- Resumo comparativo --------------------------------------------------------
def wr(entries):
    e = entries.dropna(subset=['fwd_ret'])
    if len(e) == 0: return float('nan'), 0
    return (e['fwd_ret'] > TARGET).mean()*100, len(e)

wr_in,  n_in  = wr(rot_in)
wr_out, n_out = wr(rot_out)
bl_in  = (f_in ['fwd_ret'] > TARGET).mean()*100
bl_out = (f_out['fwd_ret'] > TARGET).mean()*100

print(f"\n{'='*46}")
print(f"  RESUMO OUT-OF-SAMPLE")
print(f"{'='*46}")
print(f"  {'':20s}  {'In-Sample':>10s}  {'Out-of-Sample':>13s}")
print(f"  {'Período':<20s}  {str(f_in.index[0].date()):>10s}  {str(f_out.index[0].date()):>13s}")
print(f"  {'Sinais':<20s}  {n_in:>10d}  {n_out:>13d}")
print(f"  {'Win Rate':<20s}  {wr_in:>9.1f}%  {wr_out:>12.1f}%")
print(f"  {'Baseline':<20s}  {bl_in:>9.1f}%  {bl_out:>12.1f}%")
print(f"  {'Alpha vs baseline':<20s}  {wr_in-bl_in:>+9.1f}%  {wr_out-bl_out:>+12.1f}%")

if not (wr_in != wr_in or wr_out != wr_out):  # nan check
    decay = wr_in - wr_out
    verdict = "OVERFIT PROVAVEL" if decay > 5 else ("SINAL ROBUSTO" if wr_out > bl_out else "SEM EDGE")
    print(f"\n  Decaimento WR:   {decay:+.1f}pp")
    print(f"  Veredicto:       {verdict}")
print(f"{'='*46}\n")

"""
Quick comparison: 0.5% vs 1% risk for prop firm.
"""
import sys, pickle, sqlite3
import pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from train import build_features, ASSET_CONFIG

BASE = Path(__file__).parent.parent
MDIR = Path(__file__).parent

CONFIGS = {
    'mnq': {'db': 'data.db',     'model': 'propfirm_model_mnq.pkl', 'dir': 'both', 'stop_r': 1.5, 'targ_r': 3.0, 'bars': 24, 'horas': [8,9,10,11,12,13,14,15,16,17,18,19,20]},
    'btc': {'db': 'btc/data.db', 'model': 'propfirm_model_btc.pkl', 'dir': 'short','stop_r': 1.5, 'targ_r': 3.0, 'bars': 24, 'horas': [8,9,10,11,12,13,14,15,16,17,18,19,20]},
    'cl':  {'db': 'cl/data.db',  'model': 'propfirm_model_cl.pkl',  'dir': 'long', 'stop_r': 1.5, 'targ_r': 2.0, 'bars': 24, 'horas': [8,9,10,11,12,13,14,15,16,17,18,19,20]},
    'mgc': {'db': 'mgc/data.db', 'model': 'propfirm_model_mgc.pkl', 'dir': 'both', 'stop_r': 1.5, 'targ_r': 2.0, 'bars': 24, 'horas': [0,4,8,12,16,20]},
}
ML_THRESHOLD = 0.5

for risco in [0.005, 0.01]:
    all_trades = []
    for asset, cfg in CONFIGS.items():
        md = pickle.load(open(MDIR / cfg['model'], 'rb'))
        model, feats, fwd = md['model'], md['features'], md.get('forward', 8)
        conn = sqlite3.connect(BASE / cfg['db'])
        df = build_features(conn, '1h', fwd, syms=ASSET_CONFIG[asset]['syms'])
        conn.close()
        common = [c for c in feats if c in df.columns]
        X = df[common].fillna(0)
        proba = model.predict_proba(X)
        ml_l = proba[:, 2]; ml_s = proba[:, 0]
        conn = sqlite3.connect(BASE / cfg['db'])
        ohlc = pd.read_sql('SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts', conn, params=(asset,), parse_dates=['ts'], index_col='ts')
        conn.close()
        idx = df.index.intersection(ohlc.index); df, ohlc = df.loc[idx], ohlc.loc[idx]; N=len(df)
        h_arr=df['hour'].values.astype(int); close=ohlc['close'].values; high=ohlc['high'].values; low=ohlc['low'].values
        tr=pd.concat([ohlc['high']-ohlc['low'],(ohlc['high']-ohlc['close'].shift()).abs(),(ohlc['low']-ohlc['close'].shift()).abs()],axis=1).max(1)
        atr=tr.rolling(14).mean().bfill().values; sr,tr_,mb=cfg['stop_r'],cfg['targ_r'],cfg['bars']
        horas_set=set(cfg['horas']); dir_f=cfg['dir']
        positions={}
        for i in range(100,N-5):
            closed=[]
            for ts_e,pos in list(positions.items()):
                pos['bh']+=1; exit_px=close[i]; res=None
                if pos['bh']>=mb: res='EXP'
                elif pos['dir']=='LONG':
                    if low[i]<=pos['st']: res='LOSS'
                    elif high[i]>=pos['tg']: res='WIN'
                else:
                    if high[i]>=pos['st']: res='LOSS'
                    elif low[i]<=pos['tg']: res='WIN'
                if res:
                    if res=='WIN': ret_r=tr_
                    elif res=='LOSS': ret_r=-sr
                    else:
                        ret_pct=((exit_px-pos['en'])/pos['en']) if pos['dir']=='LONG' else ((pos['en']-exit_px)/pos['en'])
                        risk_pct=sr*pos['atr']/pos['en']
                        ret_r=ret_pct/risk_pct*sr if risk_pct>0 else 0
                    all_trades.append({'asset':asset,'ret_r':ret_r,'pnl':ret_r*risco*100})
                    closed.append(ts_e)
            for k in closed: del positions[k]
            if positions: continue
            h=h_arr[i]; lp,sp=ml_l[i],ml_s[i]
            is_long=lp>sp; conf=lp if is_long else sp
            if h not in horas_set or conf<ML_THRESHOLD: continue
            if dir_f=='long' and not is_long: continue
            if dir_f=='short' and is_long: continue
            a=atr[i]
            if np.isnan(a) or a<=0: continue
            e=close[i]
            positions[df.index[i]]={'en':e,'bh':0,'dir':'LONG' if is_long else 'SHORT','atr':a,
                'st': e-sr*a if is_long else e+sr*a, 'tg': e+tr_*a if is_long else e-tr_*a}

    df_t=pd.DataFrame(all_trades)
    total=len(df_t); wins=(df_t['ret_r']>0).sum(); wr=wins/total*100
    avg_port=df_t['pnl'].mean(); total_port=df_t['pnl'].sum()
    eq=1.0; peak=1.0; max_dd=0
    cum_pnl=df_t['pnl'].cumsum()
    for p in cum_pnl:
        eq_val=1+p/100; peak=max(peak,eq_val); dd=(eq_val-peak)/peak*100; max_dd=min(max_dd,dd)
    print('Risco=%.1f%%: %d trades  WR=%.1f%%  PnLMed=%+.3f%%  Total=%+.1f%%  DD=%.1f%%  T/sem=%.1f'%(risco*100,total,wr,avg_port,total_port,max_dd,total/4.2/52))

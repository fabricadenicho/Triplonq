"""
ML Hedge Optimizer v2 — usa o build_features do train.py diretamente.
Pré-computa resultados para (stop, target, max_bars) em 1 passada,
depois filtra por ML confidence, horas, direcao.
"""

import warnings; warnings.filterwarnings('ignore')
import sys, argparse
import pandas as pd, numpy as np, ta
from pathlib import Path
from itertools import product
import pickle

sys.path.insert(0, str(Path(__file__).parent))
from train import build_features, ASSET_CONFIG

DB = Path(__file__).parent.parent / 'data.db'
MODEL_PATH = Path(__file__).parent / 'propfirm_model_mnq.pkl'

# Grid
HOURS = [8, 12, 16, 20]
ML_CONFS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
STOPS = [1.0, 1.5, 2.0, 2.5]
TGTS  = [2.0, 2.5, 3.0, 4.0]
MAX_BARSS = [24, 48]
DIRS = ['both', 'long', 'short']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', default='8,12,16,20')
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()
    hours = [int(h) for h in args.hours.split(',')]

    # Carregar modelo
    print('Carregando modelo...', flush=True)
    md = pickle.load(open(MODEL_PATH, 'rb'))
    model = md['model']
    feats = md['features']
    model_forward = md.get('forward', 4)

    # Features via train.py
    print('Gerando features (train.build_features)...', flush=True)
    import sqlite3
    conn = sqlite3.connect(DB)
    cfg = ASSET_CONFIG['mnq']
    df_feat = build_features(conn, interval='1h', forward=model_forward, syms=cfg['syms'])
    conn.close()
    print(f'Features: {len(df_feat)} barras', flush=True)

    common = [c for c in feats if c in df_feat.columns]
    if len(common) < len(feats):
        print(f'AVISO: {len(feats)-len(common)} features ausentes', flush=True)
    X = df_feat[common].fillna(0)

    print('Predicoes ML...', flush=True)
    proba = model.predict_proba(X)  # [SHORT, NEUTRO, LONG]
    ml_long  = proba[:, 2]
    ml_short = proba[:, 0]

    # OHLC para simulacao
    conn = sqlite3.connect(DB)
    mnq = pd.read_sql('SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
                      conn, params=('mnq',), parse_dates=['ts'], index_col='ts')
    cl  = pd.read_sql('SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
                      conn, params=('cl',), parse_dates=['ts'], index_col='ts')
    conn.close()

    idx = df_feat.index.intersection(mnq.index).intersection(cl.index)
    df_feat = df_feat.loc[idx]
    mnq = mnq.loc[idx]
    cl  = cl.loc[idx]
    N = len(df_feat)
    print(f'Alinhados: {N} barras', flush=True)

    # Arrays
    hour_arr  = df_feat['hour'].values
    m_close   = mnq['close'].values
    m_high    = mnq['high'].values
    m_low     = mnq['low'].values
    c_close   = cl['close'].values
    c_high    = cl['high'].values
    c_low     = cl['low'].values
    ml_l_arr  = ml_long[:N] if len(ml_long) >= N else np.append(ml_long, [0]*(N-len(ml_long)))
    ml_s_arr  = ml_short[:N] if len(ml_short) >= N else np.append(ml_short, [0]*(N-len(ml_short)))
    price_div = df_feat['price_div_p_2'].values if 'price_div_p_2' in df_feat.columns else \
                df_feat['price_div_abs'].values if 'price_div_abs' in df_feat.columns else \
                np.zeros(N)

    # ATR
    tr = pd.concat([mnq['high']-mnq['low'], (mnq['high']-mnq['close'].shift()).abs(),
                    (mnq['low']-mnq['close'].shift()).abs()], axis=1).max(1)
    atr_arr = tr.rolling(14).mean().bfill().values

    # Pre-computar entradas (todas as barras validas)
    print('Indexando entradas...', flush=True)
    entry_idxs = []
    for i in range(100, N-5):
        if hour_arr[i] not in hours: continue
        entry_idxs.append(i)
    print(f'Entradas: {len(entry_idxs)}', flush=True)

    # Pre-computar cache de resultados para (stop, target, max_bars)
    # Armazenar: wins_count[ei] pra cada S/T/M
    print('Pre-computando resultados...', flush=True)
    cache = {}
    total_st = sum(1 for sr in STOPS for tr_ in TGTS if sr < tr_)
    done = 0
    for sr in STOPS:
        for tr_ in TGTS:
            if sr >= tr_: continue
            for mb in MAX_BARSS:
                done += 1
                if done % 10 == 0:
                    print(f'  sim {done}/{total_st*len(MAX_BARSS)}...', end='\r', flush=True)
                key = (sr, tr_, mb)
                res = np.zeros(len(entry_idxs), dtype=np.int8)  # -1=invalid, 0=loss, 1=win, 2=exp
                for k, ei in enumerate(entry_idxs):
                    atr = atr_arr[ei]
                    if np.isnan(atr) or atr <= 0:
                        res[k] = -1; continue
                    R = atr
                    em, ec = m_close[ei], c_close[ei]

                    # Simular LONG (usamos como base e depois filtramos direcao)
                    long_stop = (em - sr*R, ec + sr*R)
                    long_tgt  = (em + tr_*R, ec - tr_*R)
                    short_stop = (em + sr*R, ec - sr*R)
                    short_tgt  = (em - tr_*R, ec + tr_*R)

                    r = 2  # expired
                    for j in range(ei+1, min(ei+1+mb, N)):
                        # LONG stops/targets
                        if m_low[j] <= long_stop[0] or c_high[j] >= long_stop[1]:
                            r = 0; break
                        if m_high[j] >= long_tgt[0] or c_low[j] <= long_tgt[1]:
                            r = 1; break
                        # SHORT stops/targets
                        if m_high[j] >= short_stop[0] or c_low[j] <= short_stop[1]:
                            r = 0; break
                        if m_low[j] <= short_tgt[0] or c_high[j] >= short_tgt[1]:
                            r = 1; break
                    res[k] = r
                cache[key] = res
    print(f'\nCaches: {len(cache)}', flush=True)

    # Grid search
    print('Grid search...', flush=True)
    results = []
    total_combo = len(cache) * len(DIRS) * len(ML_CONFS)
    done = 0

    for (sr, tr_, mb), res_arr in cache.items():
        for dir_ in DIRS:
            for ml_c in ML_CONFS:
                done += 1
                if done % 500 == 0:
                    print(f'  grid {done}/{total_combo}...', end='\r', flush=True)

                wins = losses = expired = 0
                for k, ei in enumerate(entry_idxs):
                    if res_arr[k] < 0: continue

                    lp, sp = ml_l_arr[ei], ml_s_arr[ei]
                    is_long = lp > sp
                    conf = lp if is_long else sp
                    if conf < ml_c: continue
                    if dir_ == 'long' and not is_long: continue
                    if dir_ == 'short' and is_long: continue

                    r = res_arr[k]
                    if r == 1: wins += 1
                    elif r == 0: losses += 1
                    else: expired += 1

                total_t = wins + losses + expired
                if total_t < 30: continue
                wr = wins/total_t * 100
                eff_losses = losses + expired
                ev = (wins/total_t)*tr_ - (eff_losses/total_t)*sr
                score = ev * np.sqrt(total_t)
                avg_pnl = (wins*tr_ - eff_losses*sr) / total_t

                results.append({
                    'total': total_t, 'w': wins, 'l': losses, 'e': expired,
                    'wr': round(wr,1), 'avg': round(avg_pnl,3),
                    'ev': round(ev,3), 'score': round(score,2),
                    'ml_c': ml_c, 'stop': sr, 'targ': tr_,
                    'bars': mb, 'dir': dir_,
                })

    print(f'\nValidas: {len(results)}', flush=True)
    if not results:
        print('Nenhuma config valida.')
        return

    df_res = pd.DataFrame(results).sort_values('ev', ascending=False).head(args.top)
    print(f'\n  TOP {args.top} (horas={args.hours})')
    print(f'  {"Trades":>6} {"WR":>5} {"PnlMed":>6} {"EV(R)":>7} {"Score":>7}  '
          f'MLconf Stop Targ Bars Dir')
    for _, r in df_res.iterrows():
        print(f'  {r["total"]:>6} {r["wr"]:>5.1f}% {r["avg"]:>+6.3f}R {r["ev"]:>+7.3f} {r["score"]:>7.2f}  '
              f'{r["ml_c"]:>5.1f} {r["stop"]:>3.1f} {r["targ"]:>3.1f} {r["bars"]:>4} {r["dir"]:<5}')

    Path(__file__).parent / 'otimizacao_ml_hedge.csv'
    df_res.to_csv(Path(__file__).parent / 'otimizacao_ml_hedge.csv', index=False)
    print(f'\nSalvo.')

    best = df_res.iloc[0]
    print(f'\n>>> MELHOR: EV={best["ev"]:+.3f}R | WR={best["wr"]}% | {best["total"]} trades |'
          f' ML>{best["ml_c"]} | S={best["stop"]} T={best["targ"]} B={best["bars"]} D={best["dir"]}')


if __name__ == '__main__':
    main()

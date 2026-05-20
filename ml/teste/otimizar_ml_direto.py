"""
ML Strategy Optimizer — deixa o ML encontrar a melhor estrategia.
Sem hedge fixo, sem regras rigidas. O ML decide direcao, e testamos
diferentes stops/targets/horarios/thresholds.

O ML tem ~55% acuracia previsando 4h a frente. Isso deve dar EV positivo.
"""

import warnings; warnings.filterwarnings('ignore')
import sys, argparse
import pandas as pd, numpy as np
from pathlib import Path
import pickle, sqlite3

sys.path.insert(0, str(Path(__file__).parent))
from train import build_features, ASSET_CONFIG

DB = Path(__file__).parent.parent / 'data.db'
MODEL_PATH = Path(__file__).parent / 'propfirm_model_mnq.pkl'

HOURS   = [[8,12,16,20], [0,4,8,12,16,20], list(range(24))]  # + all
STOPS   = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
TGTS    = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
ML_CS   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MAX_B   = [12, 24, 48, 72]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()

    print('Carregando modelo...', flush=True)
    md = pickle.load(open(MODEL_PATH, 'rb'))
    model, feats = md['model'], md['features']
    fwd = md.get('forward', 4)

    conn = sqlite3.connect(DB)
    print('Features (train.build_features)...', flush=True)
    df = build_features(conn, '1h', fwd, syms=ASSET_CONFIG['mnq']['syms'])
    conn.close()

    common = [c for c in feats if c in df.columns]
    print(f'Features: {len(common)}/{len(feats)}', flush=True)
    X = df[common].fillna(0)
    proba = model.predict_proba(X)
    ml_long  = proba[:, 2]
    ml_short = proba[:, 0]

    # OHLC
    conn = sqlite3.connect(DB)
    mnq = pd.read_sql('SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
                      conn, params=('mnq',), parse_dates=['ts'], index_col='ts')
    conn.close()
    idx = df.index.intersection(mnq.index)
    df, mnq = df.loc[idx], mnq.loc[idx]
    N = len(df)
    print(f'Dados: {N} barras', flush=True)

    h_arr  = df['hour'].values.astype(int)
    close  = mnq['close'].values
    high   = mnq['high'].values
    low    = mnq['low'].values
    ml_l   = ml_long[:N]
    ml_s   = ml_short[:N]

    # ATR
    tr = pd.concat([mnq['high']-mnq['low'], (mnq['high']-mnq['close'].shift()).abs(),
                    (mnq['low']-mnq['close'].shift()).abs()], axis=1).max(1)
    atr = tr.rolling(14).mean().bfill().values

    # Pre-computar resultados
    print('Pre-computando...', flush=True)
    entry_idxs = [i for i in range(100, N-5)]
    print(f'Entradas: {len(entry_idxs)}', flush=True)

    # Nao vale a pena pre-computar cache pq temos muitos S/T
    # Vamos fazer grid com simulacao lazy (mas com numpy, rapido)

    results = []
    total_combos = 0
    for _ in HOURS:
        for _ in STOPS:
            for t in TGTS:
                for _ in MAX_B:
                    for _ in ML_CS:
                        if 0.5 >= t: continue  # filtrar S < T
                        total_combos += 1
    total_combos *= 2  # both + long/short
    print(f'Total combos: {total_combos}', flush=True)
    done = 0

    for hours in HOURS:
        for sr in STOPS:
            for tr_ in TGTS:
                if sr >= tr_: continue
                for mb in MAX_B:
                    for ml_c in ML_CS:
                        # (sr, tr_, mb, ml_c) -> testar 3 dirs
                        for dir_ in ['both', 'long', 'short']:
                            done += 1
                            if done % 500 == 0:
                                print(f'  {done}/{total_combos}...', end='\r', flush=True)

                            wins = losses = expired = 0
                            for ei in entry_idxs:
                                h = h_arr[ei]
                                if h not in hours: continue

                                lp, sp = ml_l[ei], ml_s[ei]
                                is_long = lp > sp
                                conf = lp if is_long else sp
                                if conf < ml_c: continue
                                if dir_ == 'long' and not is_long: continue
                                if dir_ == 'short' and is_long: continue

                                a = atr[ei]
                                if np.isnan(a) or a <= 0: continue
                                R = a
                                entry = close[ei]
                                if is_long:
                                    stop_px = entry - sr*R
                                    tgt_px  = entry + tr_*R
                                else:
                                    stop_px = entry + sr*R
                                    tgt_px  = entry - tr_*R

                                r = 2  # expired
                                for j in range(ei+1, min(ei+1+mb, N)):
                                    if is_long:
                                        if low[j] <= stop_px: r=0; break
                                        if high[j] >= tgt_px: r=1; break
                                    else:
                                        if high[j] >= stop_px: r=0; break
                                        if low[j] <= tgt_px: r=1; break

                                if r == 1: wins += 1
                                elif r == 0: losses += 1
                                else: expired += 1

                            tot = wins + losses + expired
                            if tot < 30: continue
                            wr = wins/tot*100
                            ev = (wins/tot)*tr_ - ((losses+expired)/tot)*sr
                            sc = ev * np.sqrt(tot)
                            avg_pnl = (wins*tr_ - (losses+expired)*sr) / tot
                            results.append({
                                'trades': tot, 'w': wins, 'l': losses, 'e': expired,
                                'wr': round(wr,1), 'avg': round(avg_pnl,3),
                                'ev': round(ev,3), 'score': round(sc,2),
                                'ml_c': ml_c, 'stop': sr, 'targ': tr_,
                                'bars': mb, 'dir': dir_,
                                'hours': ','.join(str(h) for h in sorted(hours)),
                            })

    print(f'\nValidas: {len(results)}', flush=True)
    if not results:
        print('Nenhuma config valida.')
        return

    df_r = pd.DataFrame(results).sort_values('ev', ascending=False).head(args.top)

    print(f'\n  TOP {args.top}')
    print(f'  {"#":>3} {"Trades":>6} {"WR":>5} {"PnlMed":>6} {"EV(R)":>7} {"Score":>7}  '
          f'MLc Stop Targ Bars Dir Horas')
    for i, (_, r) in enumerate(df_r.iterrows(), 1):
        print(f'  {i:>3} {r["trades"]:>6} {r["wr"]:>5.1f}% {r["avg"]:>+6.3f}R {r["ev"]:>+7.3f} {r["score"]:>7.2f}  '
              f'{r["ml_c"]:>4.1f} {r["stop"]:>4.1f} {r["targ"]:>4.1f} {r["bars"]:>4} {r["dir"]:<5} {r["hours"]:<20}')

    Path(__file__).parent / 'otimizacao_ml_direto.csv'
    df_r.to_csv(Path(__file__).parent / 'otimizacao_ml_direto.csv', index=False)
    print(f'\nSalvo.')

    b = df_r.iloc[0]
    print(f'\n>>> MELHOR: EV={b["ev"]:+.3f}R | WR={b["wr"]}% | {b["trades"]} trades |'
          f' MLc>{b["ml_c"]} | S={b["stop"]} T={b["targ"]} B={b["bars"]} D={b["dir"]} H={b["hours"]}')

    # Analisar: quantas configs tem EV > 0?
    pos = sum(1 for r in results if r['ev'] > 0)
    print(f'\n  Configs com EV>0: {pos}/{len(results)} ({pos/len(results)*100:.1f}%)')


if __name__ == '__main__':
    main()

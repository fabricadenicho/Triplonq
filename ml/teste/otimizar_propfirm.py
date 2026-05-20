"""
ML Strategy Optimizer — multi-ativo, focado em prop firm.
Trades curtos (6-24h), stops apertados (0.3-1.0R), targets pequenos (0.75-2.0R).

Uso: python otimizar_ml_propfirm.py --asset mnq
"""
import warnings; warnings.filterwarnings('ignore')
import sys, argparse
import pandas as pd, numpy as np
from pathlib import Path
import pickle, sqlite3

sys.path.insert(0, str(Path(__file__).parent))
from train import build_features, ASSET_CONFIG

# Grid compacto focado em prop firm (trades curtos, stops pequenos)
HOURS   = [[8,12,16,20], [0,4,8,12,16,20], [8,9,10,11,12,13,14,15,16,17,18,19,20]]
STOPS   = [0.3, 0.5, 0.75, 1.0, 1.5]
TGTS    = [0.75, 1.0, 1.5, 2.0, 3.0]
ML_CS   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MAX_B   = [6, 8, 12, 16, 24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='mnq', choices=['mnq','btc','cl','mgc','es'])
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()

    asset = args.asset
    cfg = ASSET_CONFIG[asset]
    db_path = Path(__file__).parent.parent / cfg['db']
    model_path = Path(__file__).parent / f'propfirm_model_{asset}.pkl'
    out_csv = Path(__file__).parent / f'otimizacao_{asset}_propfirm.csv'

    print(f'=== OTIMIZACAO PROPFIRM: {asset.upper()} ===')
    print(f'DB: {db_path}')
    print(f'Modelo: {model_path}')

    if not db_path.exists():
        print(f'ERRO: DB nao encontrado: {db_path}')
        return
    if not model_path.exists():
        print(f'ERRO: Modelo nao encontrado: {model_path}')
        return

    # Carregar modelo
    md = pickle.load(open(model_path, 'rb'))
    model, feats = md['model'], md['features']
    fwd = md.get('forward', 8)
    print(f'Forward: {fwd}h | Features: {len(feats)} | AUC: {md.get("auc","N/A")}')

    # Gerar features
    conn = sqlite3.connect(db_path)
    df = build_features(conn, '1h', fwd, syms=cfg['syms'])
    conn.close()

    common = [c for c in feats if c in df.columns]
    X = df[common].fillna(0)
    proba = model.predict_proba(X)
    ml_long  = proba[:, 2]
    ml_short = proba[:, 0]

    # OHLC do ativo principal
    conn = sqlite3.connect(db_path)
    ohlc = pd.read_sql(
        'SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
        conn, params=(asset,), parse_dates=['ts'], index_col='ts')
    conn.close()

    idx = df.index.intersection(ohlc.index)
    df, ohlc = df.loc[idx], ohlc.loc[idx]
    N = len(df)
    period_years = (idx.max() - idx.min()).days / 365.25
    print(f'Barras: {N} ({period_years:.1f} anos)', flush=True)

    h_arr  = df['hour'].values.astype(int)
    close  = ohlc['close'].values
    high   = ohlc['high'].values
    low    = ohlc['low'].values
    ml_l   = ml_long[:N]
    ml_s   = ml_short[:N]

    # ATR
    tr = pd.concat([ohlc['high']-ohlc['low'],
                    (ohlc['high']-ohlc['close'].shift()).abs(),
                    (ohlc['low']-ohlc['close'].shift()).abs()], axis=1).max(1)
    atr = tr.rolling(14).mean().bfill().values

    # Pre-computar passada unica otimizada
    print('Otimizando...', flush=True)
    results = []
    total_combos = len(HOURS) * len(STOPS) * len(TGTS) * len(MAX_B) * len(ML_CS) * 3
    done = 0

    for hours in HOURS:
        for sr in STOPS:
            for tr_ in TGTS:
                if sr >= tr_: continue
                for mb in MAX_B:
                    for ml_c in ML_CS:
                        for dir_ in ['both', 'long', 'short']:
                            done += 1
                            if done % 500 == 0:
                                print(f'  {done}/{total_combos} ({done*100//total_combos}%)', end='\r', flush=True)

                            wins = losses = expired = 0
                            for ei in range(100, N-5):
                                if h_arr[ei] not in hours: continue
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
                                    stop_px = entry - sr * R
                                    tgt_px  = entry + tr_ * R
                                else:
                                    stop_px = entry + sr * R
                                    tgt_px  = entry - tr_ * R

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
                            if tot < 20: continue
                            wr = wins / tot * 100
                            ev = (wins/tot)*tr_ - ((losses+expired)/tot)*sr
                            sc = ev * np.sqrt(tot)
                            avg_pnl = (wins*tr_ - (losses+expired)*sr) / tot
                            trades_por_ano = tot / period_years

                            results.append({
                                'asset': asset,
                                'trades': tot, 'w': wins, 'l': losses, 'e': expired,
                                'wr': round(wr,1), 'avg': round(avg_pnl,3),
                                'ev': round(ev,3), 'score': round(sc,2),
                                'trades_ano': round(trades_por_ano, 1),
                                'ml_c': ml_c, 'stop': sr, 'targ': tr_,
                                'bars': mb, 'dir': dir_,
                                'hours': ','.join(str(int(h)) for h in sorted(hours)),
                            })

    print(f'\nValidas: {len(results)}', flush=True)
    if not results:
        print('Nenhuma config valida.')
        return

    df_r = pd.DataFrame(results)

    # Mostrar top por EV e por trades/ano
    print(f'\n  === TOP {args.top} POR EV ===')
    top_ev = df_r.sort_values('ev', ascending=False).head(args.top)
    print(f'  {"#":>3} {"Trades":>6} {"WR":>5} {"PnlMed":>6} {"EV(R)":>7} {"T/yr":>6}  '
          f'MLc Stop Targ Bars Dir Horas')
    for i, (_, r) in enumerate(top_ev.iterrows(), 1):
        print(f'  {i:>3} {r["trades"]:>6} {r["wr"]:>5.1f}% {r["avg"]:>+6.3f}R {r["ev"]:>+7.3f} {r["trades_ano"]:>6.1f}  '
              f'{r["ml_c"]:>4.1f} {r["stop"]:>4.1f} {r["targ"]:>4.1f} {r["bars"]:>4} {r["dir"]:<5} {r["hours"]:<20}')

    # Top por trades/ano (mais frequencia) com EV positivo
    print(f'\n  === TOP {args.top} POR FREQUENCIA (EV>0) ===')
    top_freq = df_r[df_r['ev'] > 0].sort_values('trades_ano', ascending=False).head(args.top)
    print(f'  {"#":>3} {"Trades":>6} {"WR":>5} {"PnlMed":>6} {"EV(R)":>7} {"T/yr":>6}  '
          f'MLc Stop Targ Bars Dir Horas')
    for i, (_, r) in enumerate(top_freq.iterrows(), 1):
        print(f'  {i:>3} {r["trades"]:>6} {r["wr"]:>5.1f}% {r["avg"]:>+6.3f}R {r["ev"]:>+7.3f} {r["trades_ano"]:>6.1f}  '
              f'{r["ml_c"]:>4.1f} {r["stop"]:>4.1f} {r["targ"]:>4.1f} {r["bars"]:>4} {r["dir"]:<5} {r["hours"]:<20}')

    # Salvar tudo (nao so top)
    df_r.to_csv(out_csv, index=False)
    print(f'\nResultados completos salvos: {out_csv}')

    # Estatisticas gerais
    pos = (df_r['ev'] > 0).sum()
    print(f'\n  EV>0: {pos}/{len(results)} ({pos*100//len(results)}%)')
    print(f'  Media EV: {df_r["ev"].mean():+.3f}')
    print(f'  Media trades/ano: {df_r["trades_ano"].mean():.1f}')

    # Melhor equilibrio EV x frequencia
    df_r['balance'] = df_r['ev'] * np.log10(df_r['trades'] + 1)
    best_bal = df_r.sort_values('balance', ascending=False).head(5)
    print(f'\n  === MELHOR EQUILIBRIO EV x FREQUENCIA ===')
    for _, r in best_bal.iterrows():
        print(f'  EV={r["ev"]:+.3f}  T/yr={r["trades_ano"]:.1f}  WR={r["wr"]}%  '
              f'Stop={r["stop"]}  Targ={r["targ"]}  Bars={r["bars"]}  MLc={r["ml_c"]}  Dir={r["dir"]:>5}  H={r["hours"]}')


if __name__ == '__main__':
    main()

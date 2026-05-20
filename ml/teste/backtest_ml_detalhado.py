"""
Backtest detalhado do setup ML vencedor.
Compara SEM hedge vs COM hedge (como gerenciamento de risco).
Mostra: equity curve, drawdown, trades, estatisticas mensais.
"""

import warnings; warnings.filterwarnings('ignore')
import sys, pickle, sqlite3
import pandas as pd, numpy as np, ta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from train import build_features, ASSET_CONFIG

DB = Path(__file__).parent.parent / 'data.db'
MODEL_PATH = Path(__file__).parent / 'propfirm_model_mnq.pkl'

# ── Melhor config da otimizacao ──
CFG = {
    'hours':   [0, 4, 8, 12, 16, 20],
    'ml_conf': 0.6,
    'stop_r':  1.0,
    'target_r': 6.0,
    'max_bars': 72,
    'direction': 'long',
}

HEDGE_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    print('='*70)
    print('  BACKTEST DETALHADO - ML Strategy')
    print('  Config: ML>{:.1f} | Stop={}R | Target={}R | MaxBars={} | Direcao={}'.format(
        CFG['ml_conf'], CFG['stop_r'], CFG['target_r'], CFG['max_bars'], CFG['direction']))
    print('  Horarios: {}'.format(','.join(str(h) for h in CFG['hours'])))
    print('='*70)

    # Carregar modelo
    md = pickle.load(open(MODEL_PATH, 'rb'))
    model, feats = md['model'], md['features']
    fwd = md.get('forward', 4)

    conn = sqlite3.connect(DB)
    df = build_features(conn, '1h', fwd, syms=ASSET_CONFIG['mnq']['syms'])
    conn.close()
    common = [c for c in feats if c in df.columns]
    X = df[common].fillna(0)
    proba = model.predict_proba(X)
    ml_l = proba[:, 2]
    ml_s = proba[:, 0]

    # OHLC MNQ + CL
    conn = sqlite3.connect(DB)
    mnq = pd.read_sql(
        'SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
        conn, params=('mnq',), parse_dates=['ts'], index_col='ts')
    cl  = pd.read_sql(
        'SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
        conn, params=('cl',), parse_dates=['ts'], index_col='ts')
    conn.close()

    idx = df.index.intersection(mnq.index).intersection(cl.index)
    df, mnq, cl = df.loc[idx], mnq.loc[idx], cl.loc[idx]
    N = len(df)
    print(f'Periodo: {df.index[0].date()} a {df.index[-1].date()} ({N} barras)')

    h_arr  = df['hour'].values.astype(int)
    m_c, m_h, m_l = mnq['close'].values, mnq['high'].values, mnq['low'].values
    c_c, c_h, c_l = cl['close'].values, cl['high'].values, cl['low'].values
    ml_l_arr = ml_l[:N]
    ml_s_arr = ml_s[:N]

    # ATR
    tr_m = pd.concat([mnq['high']-mnq['low'], (mnq['high']-mnq['close'].shift()).abs(),
                      (mnq['low']-mnq['close'].shift()).abs()], axis=1).max(1)
    atr_m = tr_m.rolling(14).mean().bfill().values
    tr_c = pd.concat([cl['high']-cl['low'], (cl['high']-cl['close'].shift()).abs(),
                      (cl['low']-cl['close'].shift()).abs()], axis=1).max(1)
    atr_c = tr_c.rolling(14).mean().bfill().values

    # Para cada hedge ratio, simular
    all_results = {}
    for hr in HEDGE_RATIOS:
        label = f"SEM HEDGE" if hr == 0 else f"HEDGE {hr:.0%} CL"
        print(f'\n--- Simulando: {label} ---')

        trades = []
        equity = [1.0]  # equity curve
        eq_ts  = [df.index[0]]
        daily_pnl = {}

        for i in range(100, N-5):
            h = h_arr[i]
            if h not in CFG['hours']: continue
            lp, sp = ml_l_arr[i], ml_s_arr[i]
            is_long = lp > sp
            conf = lp if is_long else sp
            if conf < CFG['ml_conf']: continue
            if CFG['direction'] == 'long' and not is_long: continue
            if CFG['direction'] == 'short' and is_long: continue

            atr = atr_m[i]
            if np.isnan(atr) or atr <= 0: continue
            R = atr
            entry_m = m_c[i]
            entry_c = c_c[i]

            # Stop / Target para MNQ
            stop_m = entry_m - CFG['stop_r'] * R
            tgt_m  = entry_m + CFG['target_r'] * R

            result = None
            exit_m = exit_c = None
            bars_held = 0
            max_bars = CFG['max_bars']

            for j in range(i+1, min(i+1+max_bars, N)):
                bars_held += 1
                if m_l[j] <= stop_m:
                    result = 'LOSS'; exit_m = stop_m; break
                if m_h[j] >= tgt_m:
                    result = 'WIN'; exit_m = tgt_m; break

            if result is None:
                result = 'EXPIRED'
                exit_m = m_c[min(i+max_bars, N-1)]

            # Preco de saida CL (sempre no fechamento da barra de saida)
            exit_c = c_c[min(i+bars_held, N-1)]

            # PnL
            pnl_m = ((exit_m - entry_m) / entry_m) * 100
            pnl_c = ((entry_c - exit_c) / entry_c) * 100  # short CL = entry - exit
            pnl_total = pnl_m + hr * pnl_c

            trade = {
                'entry_time': df.index[i], 'hour': h,
                'exit_time': df.index[min(i+bars_held, N-1)],
                'bars_held': bars_held,
                'result': result,
                'entry_mnq': entry_m, 'exit_mnq': exit_m,
                'entry_cl': entry_c, 'exit_cl': exit_c,
                'pnl_mnq_pct': round(pnl_m, 2),
                'pnl_cl_pct': round(-pnl_c * hr, 2) if hr > 0 else 0,
                'pnl_total_pct': round(pnl_total, 2),
                'ml_conf': round(conf, 3),
            }
            trades.append(trade)

            # Equity curve
            for bar in range(i, min(i+bars_held, N)):
                ts = df.index[bar]
                eq_ts.append(ts)
                daily_key = ts.date()
                if daily_key not in daily_pnl:
                    daily_pnl[daily_key] = 0

            # Log PnL no dia de saida
            exit_day = trade['exit_time'].date()
            if exit_day not in daily_pnl:
                daily_pnl[exit_day] = 0
            daily_pnl[exit_day] += pnl_total

        # Calcular equity curve a partir de daily_pnl
        dates_sorted = sorted(daily_pnl.keys())
        eq = 1.0
        eq_curve = [(dates_sorted[0], 1.0)] if dates_sorted else []
        for d in dates_sorted:
            eq *= (1 + daily_pnl[d] / 100)
            eq_curve.append((d, eq))

        # Estatisticas
        df_t = pd.DataFrame(trades)
        all_results[hr] = {'trades': df_t, 'equity': eq_curve, 'daily_pnl': daily_pnl}

        total = len(df_t)
        if total == 0:
            print('  Nenhum trade.')
            continue

        wins  = (df_t['result'] == 'WIN').sum()
        losses = (df_t['result'] == 'LOSS').sum()
        exp   = (df_t['result'] == 'EXPIRED').sum()
        wr    = wins / total * 100
        avg   = df_t['pnl_total_pct'].mean()
        total_pnl = df_t['pnl_total_pct'].sum()
        sd    = df_t['pnl_total_pct'].std()
        sharpe = avg / sd * np.sqrt(365*24) if sd > 0 else 0
        max_dd = 0; peak = 1.0
        for _, v in eq_curve:
            peak = max(peak, v)
            dd = (v - peak) / peak * 100
            max_dd = min(max_dd, dd)
        best_trade = df_t.loc[df_t['pnl_total_pct'].idxmax()]
        worst_trade = df_t.loc[df_t['pnl_total_pct'].idxmin()]

        print(f'  Trades: {total}  Wins: {wins} ({wr:.1f}%)  Losses: {losses}  Exp: {exp}')
        print(f'  PnL medio: {avg:+.2f}%  PnL total: {total_pnl:+.2f}%')
        print(f'  Sharpe (anual): {sharpe:.2f}')
        print(f'  Max Drawdown: {max_dd:.1f}%')
        print(f'  Best trade: {best_trade["pnl_total_pct"]:+.2f}% em {best_trade["entry_time"].date()}')
        print(f'  Worst trade: {worst_trade["pnl_total_pct"]:+.2f}% em {worst_trade["entry_time"].date()}')

    # ── Tabela comparativa ──
    print('\n' + '='*70)
    print('  COMPARACAO: SEM HEDGE vs COM HEDGE')
    print('='*70)
    print(f'  {"Hedge":>10} {"Trades":>7} {"WR":>6} {"PnLTotal":>10} {"PnLMed":>8} {"Sharpe":>8} {"MaxDD":>8} {"Best":>8} {"Worst":>8}')
    print(f'  {"-"*10} {"-"*7} {"-"*6} {"-"*10} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    for hr in HEDGE_RATIOS:
        if hr not in all_results: continue
        r = all_results[hr]
        df_t = r['trades']
        if len(df_t) == 0: continue
        label = f"SEM" if hr == 0 else f"{hr:.0%}"
        total = len(df_t)
        wins = (df_t['result'] == 'WIN').sum()
        wr = wins/total*100
        avg = df_t['pnl_total_pct'].mean()
        total_pnl = df_t['pnl_total_pct'].sum()
        sd = df_t['pnl_total_pct'].std()
        sharpe = avg / sd * np.sqrt(365*24) if sd > 0 else 0
        peak = 1.0; max_dd = 0
        for _, v in r['equity']:
            peak = max(peak, v); dd = (v-peak)/peak*100; max_dd = min(max_dd, dd)
        best = df_t['pnl_total_pct'].max()
        worst = df_t['pnl_total_pct'].min()
        print(f'  {label:>10} {total:>7} {wr:>5.1f}% {total_pnl:>+9.2f}% {avg:>+7.2f}% {sharpe:>7.2f} {max_dd:>7.1f}% {best:>+7.2f}% {worst:>+7.2f}%')

    # ── Salvar trades + equity curves ──
    base = Path(__file__).parent / 'backtest_ml'
    for hr in HEDGE_RATIOS:
        if hr not in all_results: continue
        r = all_results[hr]
        label = 'sem_hedge' if hr == 0 else f'hedge_{int(hr*100)}pct'
        r['trades'].to_csv(base / f'trades_{label}.csv', index=False)
        eq_df = pd.DataFrame(r['equity'], columns=['date', 'equity'])
        eq_df.to_csv(base / f'equity_{label}.csv', index=False)
    print(f'\nResultados salvos em: {base}/')

    # ── Trade log das melhores/piores ──
    if 0.0 in all_results:
        df_no = all_results[0.0]['trades']
        print(f'\n  --- TOP 10 TRADES (sem hedge) ---')
        for _, t in df_no.nlargest(10, 'pnl_total_pct').iterrows():
            print(f'    {t["entry_time"].date():>12} {int(t["hour"]):02d}h  {t["result"]:<8}  '
                  f'MNQ={t["pnl_mnq_pct"]:+.2f}%  Total={t["pnl_total_pct"]:+.2f}%  '
                  f'bars={t["bars_held"]}  conf={t["ml_conf"]:.2f}')
        print(f'\n  --- WORST 10 TRADES (sem hedge) ---')
        for _, t in df_no.nsmallest(10, 'pnl_total_pct').iterrows():
            print(f'    {t["entry_time"].date():>12} {int(t["hour"]):02d}h  {t["result"]:<8}  '
                  f'MNQ={t["pnl_mnq_pct"]:+.2f}%  Total={t["pnl_total_pct"]:+.2f}%  '
                  f'bars={t["bars_held"]}  conf={t["ml_conf"]:.2f}')

    return all_results


if __name__ == '__main__':
    main()

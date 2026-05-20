"""
Backtest Multi-Ativo — estrategia final para prop firm.
Usa os melhores setups de cada ativo encontrados pelo otimizador.
Combina trades de MNQ, BTC, CL, MGC.
"""

import warnings; warnings.filterwarnings('ignore')
import sys, pickle, sqlite3
import pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from train import build_features, ASSET_CONFIG

BASE = Path(__file__).parent.parent

# ── Melhores configs por ativo (da otimizacao propfirm) ──
# PnL em R (1R = risco definido pelo stop). Stop_r e Targ_r sao em unidades de R.
CONFIGS = {
    'mnq': {'db': 'data.db',     'model': 'propfirm_model_mnq.pkl', 'dir': 'both', 'stop_r': 1.5, 'targ_r': 3.0, 'bars': 24, 'horas': [8,9,10,11,12,13,14,15,16,17,18,19,20]},
    'btc': {'db': 'btc/data.db', 'model': 'propfirm_model_btc.pkl', 'dir': 'short','stop_r': 1.5, 'targ_r': 3.0, 'bars': 24, 'horas': [8,9,10,11,12,13,14,15,16,17,18,19,20]},
    'cl':  {'db': 'cl/data.db',  'model': 'propfirm_model_cl.pkl',  'dir': 'long', 'stop_r': 1.5, 'targ_r': 2.0, 'bars': 24, 'horas': [8,9,10,11,12,13,14,15,16,17,18,19,20]},
    'mgc': {'db': 'mgc/data.db', 'model': 'propfirm_model_mgc.pkl', 'dir': 'both', 'stop_r': 1.5, 'targ_r': 2.0, 'bars': 24, 'horas': [0,4,8,12,16,20]},
    'es':  {'db': 'es/data.db',  'model': 'propfirm_model_es.pkl',  'dir': 'both', 'stop_r': 1.5, 'targ_r': 3.0, 'bars': 24, 'horas': [8,9,10,11,12,13,14,15,16,17,18,19,20]},
}

ML_THRESHOLD = 0.5
RISCO_PCT = 0.01  # 1% do capital por trade


def load_model(asset):
    path = Path(__file__).parent / CONFIGS[asset]['model']
    md = pickle.load(open(path, 'rb'))
    return md['model'], md['features'], md.get('forward', 8)


def carregar_ohlc(asset):
    db_path = BASE / CONFIGS[asset]['db']
    sym = asset
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        'SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
        conn, params=(sym,), parse_dates=['ts'], index_col='ts')
    conn.close()
    return df


def simular_ativo(asset, cfg):
    print(f'  {asset.upper()}...', end=' ', flush=True)
    model, feats, fwd = load_model(asset)
    db_path = BASE / cfg['db']

    conn = sqlite3.connect(db_path)
    df_feat = build_features(conn, '1h', fwd, syms=ASSET_CONFIG[asset]['syms'])
    conn.close()

    common = [c for c in feats if c in df_feat.columns]
    X = df_feat[common].fillna(0)
    proba = model.predict_proba(X)
    ml_l = proba[:, 2]
    ml_s = proba[:, 0]

    # OHLC
    ohlc = carregar_ohlc(asset)
    idx = df_feat.index.intersection(ohlc.index)
    df_feat, ohlc = df_feat.loc[idx], ohlc.loc[idx]
    N = len(df_feat)

    h_arr = df_feat['hour'].values.astype(int)
    close = ohlc['close'].values
    high  = ohlc['high'].values
    low   = ohlc['low'].values
    ml_l_arr = ml_l[:N]
    ml_s_arr = ml_s[:N]

    tr = pd.concat([ohlc['high']-ohlc['low'],
                    (ohlc['high']-ohlc['close'].shift()).abs(),
                    (ohlc['low']-ohlc['close'].shift()).abs()], axis=1).max(1)
    atr = tr.rolling(14).mean().bfill().values

    stop_r = cfg['stop_r']
    targ_r = cfg['targ_r']
    max_bars = cfg['bars']
    horas_set = set(cfg['horas'])
    dir_filter = cfg['dir']
    ml_th = ML_THRESHOLD

    trades = []
    positions = {}

    for i in range(100, N-5):
        # Exits for open positions
        closed = []
        for ts_entry, pos in list(positions.items()):
            pos['bars_held'] += 1
            exit_px = close[i]
            result = None

            if pos['bars_held'] >= max_bars:
                result = 'EXPIRED'
            elif pos['dir'] == 'LONG':
                if low[i] <= pos['stop']: result = 'LOSS'
                elif high[i] >= pos['target']: result = 'WIN'
            else:  # SHORT
                if high[i] >= pos['stop']: result = 'LOSS'
                elif low[i] <= pos['target']: result = 'WIN'

            if result:
                if result == 'WIN':
                    ret_r = targ_r
                elif result == 'LOSS':
                    ret_r = -stop_r
                else:
                    # Expired: PnL real em % do ativo, dividido pelo risco % pra dar em R
                    if pos['dir'] == 'LONG':
                        ret_pct = (exit_px - pos['entry']) / pos['entry']
                    else:
                        ret_pct = (pos['entry'] - exit_px) / pos['entry']
                    risk_pct = stop_r * pos['atr'] / pos['entry']
                    ret_r = ret_pct / risk_pct * stop_r if risk_pct > 0 else 0

                pnl_portfolio_pct = ret_r * RISCO_PCT
                trades.append({'asset': asset, 'entry_time': ts_entry, 'exit_time': df_feat.index[i],
                               'direction': pos['dir'], 'result': result,
                               'ret_r': round(ret_r, 2),
                               'pnl_pct': round(pnl_portfolio_pct * 100, 3),
                               'entry_price': round(pos['entry'], 2),
                               'stop_price': round(pos['stop'], 2),
                               'target_price': round(pos['target'], 2),
                               'atr': round(pos['atr'], 2),
                               'conf': round(pos.get('conf', 0), 3)})
                closed.append(ts_entry)

        for k in closed:
            del positions[k]

        # Ja tem posicao aberta nesse ativo?
        if len(positions) > 0:
            continue

        # Check entry
        h = h_arr[i]
        if h not in horas_set: continue
        lp, sp = ml_l_arr[i], ml_s_arr[i]
        is_long = lp > sp
        conf = lp if is_long else sp
        if conf < ml_th: continue

        if dir_filter == 'long' and not is_long: continue
        if dir_filter == 'short' and is_long: continue

        a = atr[i]
        if np.isnan(a) or a <= 0: continue
        R = a
        entry = close[i]
        ts_entry = df_feat.index[i]

        if is_long:
            stop_px = entry - stop_r * R
            tgt_px  = entry + targ_r * R
        else:
            stop_px = entry + stop_r * R
            tgt_px  = entry - targ_r * R

        positions[ts_entry] = {
            'entry': entry, 'stop': stop_px, 'target': tgt_px,
            'dir': 'LONG' if is_long else 'SHORT', 'bars_held': 0,
            'atr': R, 'conf': round(conf, 3),
        }

    return trades


def main():
    print('='*70)
    print('  BACKTEST MULTI-ATIVO PROPFIRM')
    print('  Threshold ML: {:.1f}'.format(ML_THRESHOLD))
    print('='*70)

    all_trades = []
    for asset, cfg in CONFIGS.items():
        trades = simular_ativo(asset, cfg)
        all_trades.extend(trades)
        print(f'{len(trades)} trades', flush=True)

    df_t = pd.DataFrame(all_trades)
    if len(df_t) == 0:
        print('Nenhum trade.')
        return

    # Estatisticas
    total = len(df_t)
    wins = (df_t['result'] == 'WIN').sum()
    losses = (df_t['result'] == 'LOSS').sum()
    exp = (df_t['result'] == 'EXPIRED').sum()
    wr = wins / total * 100
    avg_pnl = df_t['pnl_pct'].mean()
    total_pnl = df_t['pnl_pct'].sum()
    sd = df_t['pnl_pct'].std()
    sharpe = avg_pnl / sd * np.sqrt(52) if sd > 0 else 0  # semanal (52 semanas/ano)

    # Periodo
    start_date = df_t['entry_time'].min()
    end_date = df_t['exit_time'].max()
    period_years = (end_date - start_date).days / 365.25
    trades_per_year = total / period_years
    trades_per_week = trades_per_year / 52

    # Equity curve com compounding
    eq = 1.0
    eq_curve = [(start_date.date(), 1.0)]
    peak = 1.0
    max_dd = 0
    daily_pnl = df_t.groupby(df_t['exit_time'].dt.date)['pnl_pct'].sum()
    for d in sorted(daily_pnl.index):
        eq *= (1 + daily_pnl[d] / 100)
        eq_curve.append((d, eq))
        peak = max(peak, eq)
        dd = (eq - peak) / peak * 100
        max_dd = min(max_dd, dd)

    print(f'\n  === RESULTADOS GLOBAIS ===')
    print(f'  Periodo: {start_date.date()} a {end_date.date()} ({period_years:.1f} anos)')
    print(f'  Total trades: {total}')
    print(f'  Wins: {wins} ({wr:.1f}%)  Losses: {losses}  Expired: {exp}')
    print(f'  Trades/semana: {trades_per_week:.1f}')
    print(f'  PnL medio (do portfolio): {avg_pnl:+.3f}%')
    print(f'  PnL total acumulado: {total_pnl:+.2f}%')
    print(f'  Crescimento total (compounded): {(eq-1)*100:+.2f}%')
    print(f'  Sharpe (semanal): {sharpe:.2f}')
    print(f'  Max Drawdown: {max_dd:.1f}%')

    # Por ativo
    print(f'\n  === POR ATIVO ===')
    for asset in CONFIGS:
        sub = df_t[df_t['asset']==asset]
        if len(sub) == 0: continue
        wr_a = (sub['result']=='WIN').sum() / len(sub) * 100
        avg_ret_r = sub['ret_r'].mean()
        avg_port = sub['pnl_pct'].mean()
        print(f'  {asset.upper()}: {len(sub)} trades  WR={wr_a:.1f}%  PnLMed={avg_ret_r:+.2f}R ({avg_port:+.3f}% port)')

    # Melhores e piores
    print(f'\n  === TOP 5 TRADES (R) ===')
    for _, t in df_t.nlargest(5, 'ret_r').iterrows():
        print(f'  {t["entry_time"].date()}  {t["asset"]:<4} {t["direction"]:<6}  {t["result"]:<8}  {t["ret_r"]:+.1f}R ({t["pnl_pct"]:+.2f}%)')

    print(f'\n  === WORST 5 TRADES (R) ===')
    for _, t in df_t.nsmallest(5, 'ret_r').iterrows():
        print(f'  {t["entry_time"].date()}  {t["asset"]:<4} {t["direction"]:<6}  {t["result"]:<8}  {t["ret_r"]:+.1f}R ({t["pnl_pct"]:+.2f}%)')

    # Save
    out_dir = Path(__file__).parent / 'backtest_ml'
    out_dir.mkdir(exist_ok=True)
    df_t.to_csv(out_dir / 'trades_multi_asset.csv', index=False)
    eq_df = pd.DataFrame(eq_curve, columns=['date', 'equity'])
    eq_df.to_csv(out_dir / 'equity_multi_asset.csv', index=False)
    print(f'\nSalvo em: {out_dir}/')

    print(f'\n  === PROJECAO PROPFIRM ===')
    print(f'  Risco: {RISCO_PCT*100:.0f}% do capital por trade')
    print(f'  Retorno R medio: {df_t["ret_r"].mean():+.2f}R')
    print(f'  Retorno port medio: {avg_pnl:.3f}%')
    print(f'  Trades/semana: {trades_per_week:.1f}')
    print(f'  Retorno semanal esperado: {trades_per_week * avg_pnl:.3f}%')
    print(f'  Retorno mensal esperado: {trades_per_week * avg_pnl * 4.33:.2f}%')
    print(f'  Max Drawdown: {max_dd:.1f}%')
    if max_dd > -10:
        print(f'  [OK] Drawdown dentro do limite prop firm (10%)')
    else:
        print(f'  [REVISAO] Drawdown de {max_dd:.1f}% excede limite prop firm (10%)')


if __name__ == '__main__':
    main()

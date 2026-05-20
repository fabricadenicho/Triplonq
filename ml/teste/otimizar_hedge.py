"""
Otimizador ultra rapido da estrategia Hedge MNQ x CL.
1) Pre-computa entradas potenciais e resultados Win/Loss/Expired
   para cada par (stop_r, target_r, max_bars).
2) Depois filtra por outros parametros sem re-simular.
"""

import warnings; warnings.filterwarnings('ignore')
import sqlite3, argparse
import pandas as pd, numpy as np, ta
from pathlib import Path
from itertools import product

DB = Path(__file__).parent.parent / 'data.db'
STOP_RS = [1.0, 1.5, 2.0, 2.5, 3.0]
TGT_RS  = [2.0, 2.5, 3.0, 4.0, 5.0]
MAX_BARSS = [24, 48]
MIN_ADXS = [20, 25, 30]
MA_MODES = ['both', 'sma20', 'ema20']
RSI_DIVS = [0, 5]
DIRS     = ['both', 'long', 'short']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', default='8,12,16,20')
    args = ap.parse_args()
    hours = [int(h) for h in args.hours.split(',')]

    # ── Dados ──
    conn = sqlite3.connect(DB)
    mnq = pd.read_sql('SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
                      conn, params=('mnq',), parse_dates=['ts'], index_col='ts')
    cl  = pd.read_sql('SELECT ts,open,high,low,close FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
                      conn, params=('cl',), parse_dates=['ts'], index_col='ts')
    conn.close()
    idx = mnq.index.intersection(cl.index)
    mnq, cl = mnq.loc[idx], cl.loc[idx]
    print(f'Barras: {len(mnq)} ({mnq.index[0].date()} a {mnq.index[-1].date()})', flush=True)

    # ── Indicadores ──
    for df in (mnq, cl):
        df['ema20'] = df['close'].ewm(span=20,adjust=False).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['rsi21'] = ta.momentum.RSIIndicator(df['close'],21).rsi()
        df['adx']   = ta.trend.ADXIndicator(df['high'],df['low'],df['close'],14).adx()
        tr = pd.concat([df['high']-df['low'],(df['high']-df['close'].shift()).abs(),
                        (df['low']-df['close'].shift()).abs()], axis=1).max(1)
        df['atr14'] = tr.rolling(14).mean()
    mnq['ret1'] = mnq['close'].pct_change(1)
    cl['ret1']  = cl['close'].pct_change(1)
    N = len(mnq)

    # ── Features (vetorizadas) ──
    h     = mnq.index.hour.values
    pdv   = mnq['ret1'].values * cl['ret1'].values
    rdiv  = mnq['rsi21'].values - cl['rsi21'].values
    m_adx = mnq['adx'].values; c_adx = cl['adx'].values
    m_abv_sma = (mnq['close'].values > mnq['sma20'].values).astype(int)
    c_abv_sma = (cl['close'].values > cl['sma20'].values).astype(int)
    m_abv_ema = (mnq['close'].values > mnq['ema20'].values).astype(int)
    c_abv_ema = (cl['close'].values > cl['ema20'].values).astype(int)
    m_close = mnq['close'].values; c_close = cl['close'].values
    m_atr   = mnq['atr14'].values; c_atr = cl['atr14'].values
    m_high  = mnq['high'].values;  m_low  = mnq['low'].values
    c_high  = cl['high'].values;   c_low  = cl['low'].values

    # ── Pre-computar: entradas validas ──
    print('Pre-computando entradas...', flush=True)
    entry_idxs = []
    for i in range(60, N-5):
        if h[i] not in hours: continue
        if np.isnan(pdv[i]) or pdv[i] >= 0: continue
        entry_idxs.append(i)
    print(f'Entradas potenciais: {len(entry_idxs)}', flush=True)

    # ── Pre-computar resultados para cada (stop, target, max_bars) ──
    #   result_cache[(stop_r, tgt_r, max_b)] = array do mesmo tamanho de entry_idxs
    #   0 = LOSS, 1 = WIN, 2 = EXPIRED, -1 = invalid (atr=0)
    print('Simulando resultados para cada stop/target...', flush=True)
    result_cache = {}
    for sr in STOP_RS:
        for tr_ in TGT_RS:
            if sr >= tr_: continue
            for mb in MAX_BARSS:
                key = (sr, tr_, mb)
                res = np.zeros(len(entry_idxs), dtype=np.int8)
                for k, ei in enumerate(entry_idxs):
                    atr = max(m_atr[ei], c_atr[ei])
                    if np.isnan(atr) or atr <= 0:
                        res[k] = -1; continue
                    R = atr
                    entry_m, entry_c = m_close[ei], c_close[ei]
                    # Precisamos testar LONG e SHORT -- fazer direcao na hora do filtro,
                    # mas aqui ja calculamos os niveis para os 2 casos
                    long_stop = (entry_m - sr*R, entry_c + sr*R)
                    long_tgt  = (entry_m + tr_*R, entry_c - tr_*R)
                    short_stop = (entry_m + sr*R, entry_c - sr*R)
                    short_tgt  = (entry_m - tr_*R, entry_c + tr_*R)

                    result = 2  # expired
                    for j in range(ei+1, min(ei+1+mb, N)):
                        # LONG
                        if m_low[j] <= long_stop[0] or c_high[j] >= long_stop[1]:
                            result = 0; break
                        if m_high[j] >= long_tgt[0] or c_low[j] <= long_tgt[1]:
                            result = 1; break
                        # SHORT (pode nao ser usado, mas ja calculamos)
                        if m_high[j] >= short_stop[0] or c_low[j] <= short_stop[1]:
                            result = 0; break
                        if m_low[j] <= short_tgt[0] or c_high[j] >= short_tgt[1]:
                            result = 1; break
                    res[k] = result
                result_cache[key] = res
    print(f'Caches criados: {len(result_cache)}', flush=True)

    # ── Para cada entrada, pre-computar direcao por (ma_mode, rsi_div_min, direction) ──
    #   Vamos fazer em tempo real (barato)
    # ── Grid search sobre filtros (barato, sem loops de barras) ──
    results = []
    total_combo = len(MA_MODES)*len(RSI_DIVS)*len(DIRS)*len(MIN_ADXS)*len(result_cache)
    done = 0

    for sr, tr_, mb in result_cache:
        res_arr = result_cache[(sr, tr_, mb)]
        for ma in MA_MODES:
            for rd in RSI_DIVS:
                for dir_ in DIRS:
                    for min_adx in MIN_ADXS:
                        done += 1
                        if done % 200 == 0:
                            print(f'  {done}/{total_combo}...', end='\r', flush=True)

                        wins = losses = expired = 0
                        for k, ei in enumerate(entry_idxs):
                            if res_arr[k] < 0: continue
                            if m_adx[ei] < min_adx and c_adx[ei] < min_adx: continue
                            if abs(rdiv[ei]) < rd: continue

                            # MA filter
                            if ma == 'both':
                                long_ok = m_abv_sma[ei] and m_abv_ema[ei] and not c_abv_sma[ei] and not c_abv_ema[ei]
                                short_ok = not m_abv_sma[ei] and not m_abv_ema[ei] and c_abv_sma[ei] and c_abv_ema[ei]
                            elif ma == 'sma20':
                                long_ok = m_abv_sma[ei] and not c_abv_sma[ei]
                                short_ok = not m_abv_sma[ei] and c_abv_sma[ei]
                            elif ma == 'ema20':
                                long_ok = m_abv_ema[ei] and not c_abv_ema[ei]
                                short_ok = not m_abv_ema[ei] and c_abv_ema[ei]

                            if not (long_ok or short_ok): continue

                            if long_ok and short_ok:
                                is_long = rdiv[ei] > 0
                            elif long_ok:
                                is_long = True
                            else:
                                is_long = False

                            if dir_ == 'long' and not is_long: continue
                            if dir_ == 'short' and is_long: continue
                            if is_long and rdiv[ei] <= 0: continue
                            if not is_long and rdiv[ei] >= 0: continue

                            r = res_arr[k]
                            if r == 1: wins += 1
                            elif r == 0: losses += 1
                            else: expired += 1

                        total_t = wins + losses + expired
                        if total_t < 20: continue

                        wr = wins/total_t*100
                        # EV em R: assumindo que trades vencedoras ganham target_r,
                        # perdedoras perdem stop_r, expiradas perdem stop_r tb (worst case)
                        effective_losses = losses + expired
                        ev = (wins/total_t)*tr_ - (effective_losses/total_t)*sr
                        score = ev * np.sqrt(total_t)
                        avg_pnl = (wins*tr_ - effective_losses*sr) / total_t

                        results.append({
                            'total': total_t, 'wins': wins, 'losses': losses, 'exp': expired,
                            'wr': round(wr,1), 'avg_pnl': round(avg_pnl,3),
                            'ev_r': round(ev,3), 'score': round(score,2),
                            'min_adx': min_adx, 'ma_mode': ma, 'rsi_div_min': rd,
                            'stop_r': sr, 'target_r': tr_, 'max_bars': mb, 'direction': dir_,
                        })

    print(f'\nConfigs validas: {len(results)}', flush=True)
    df_res = pd.DataFrame(results).sort_values('ev_r', ascending=False).head(20)
    if len(df_res) == 0:
        print('Nenhuma config valida.')
        return

    print(f'\n  TOP 20 (horas={args.hours})')
    print(f'  {"Trades":>6} {"WR":>5} {"Exp":>3} {"PnlMed":>6} {"EV(R)":>7} {"Score":>7}  '
          f'ADX MA-mode RSI Stp Targ Bars Dir')
    for _, r in df_res.iterrows():
        print(f'  {r["total"]:>6} {r["wr"]:>5.1f}% {r["exp"]:>3} {r["avg_pnl"]:>+6.3f}R '
              f'{r["ev_r"]:>+7.3f} {r["score"]:>7.2f}  '
              f'{r["min_adx"]:>3} {r["ma_mode"]:<7} {r["rsi_div_min"]:>3} '
              f'{r["stop_r"]:>3.1f} {r["target_r"]:>3.1f} {r["max_bars"]:>4} {r["direction"]:<5}')

    csv_path = Path(__file__).parent / 'otimizacao_hedge.csv'
    df_res.to_csv(csv_path, index=False)
    print(f'\nSalvo: {csv_path}')

    best = df_res.iloc[0]
    print(f'\n>>> MELHOR: EV={best["ev_r"]:+.3f}R | WR={best["wr"]}% | {best["total"]} trades |'
          f' ADX>{best["min_adx"]} | {best["ma_mode"]} | div>{best["rsi_div_min"]} |'
          f' Stop={best["stop_r"]}R | Targ={best["target_r"]}R | bars={best["max_bars"]} | {best["direction"]}')

if __name__ == '__main__':
    main()

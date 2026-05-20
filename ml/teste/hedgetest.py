"""
Backtest: Estrategia Hedge MNQ x CL
  - Faixa de abertura 21H + a cada 4h (0,4,8,12,16,20 UTC)
  - Preco um acima, outro abaixo da EMA20/SMA20 (opostos)
  - ADX > 30 em pelo menos um
  - RSI 21 periodos divergente
  - Andando em direcoes opostas (price_div < 0)
  - Se LONG MNQ, CL deve mostrar oportunidade SHORT
  - Entrada com 1.5R de stop, alvo 3R

Uso: python hedgetest.py
"""
import warnings
warnings.filterwarnings('ignore')

import sqlite3
import pandas as pd
import numpy as np
import ta
from pathlib import Path
from datetime import datetime

DB = Path(__file__).parent.parent / 'data.db'
RISK_R = 1.5     # stop loss em R
TARGET_R = 3.0   # take profit em R


def load_symbol(conn, symbol):
    return pd.read_sql(
        'SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
        conn, params=(symbol,), parse_dates=['ts'], index_col='ts'
    )


def main():
    conn = sqlite3.connect(DB)
    mnq = load_symbol(conn, 'mnq')
    cl  = load_symbol(conn, 'cl')
    conn.close()

    # Alinhar indices
    idx = mnq.index.intersection(cl.index)
    mnq = mnq.loc[idx].copy()
    cl  = cl.loc[idx].copy()

    # ── Indicadores ──
    for df in [mnq, cl]:
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['rsi']   = pd.DataFrame(
            ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        )
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['di+'] = adx.adx_pos()
        df['di-'] = adx.adx_neg()
        df['atr'] = 0.0

    # ATR 14
    for df in [mnq, cl]:
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

    # Retornos 1 barra
    mnq['ret1'] = mnq['close'].pct_change(1)
    cl['ret1']  = cl['close'].pct_change(1)

    # ── Features compostas ──
    df = pd.DataFrame(index=idx)
    df['hour'] = idx.hour

    df['mnq_close']  = mnq['close']
    df['cl_close']   = cl['close']
    df['mnq_ema20']  = mnq['ema20']
    df['cl_ema20']   = cl['ema20']
    df['mnq_sma20']  = mnq['sma20']
    df['cl_sma20']   = cl['sma20']
    df['mnq_rsi']    = mnq['rsi']
    df['cl_rsi']     = cl['rsi']
    df['mnq_adx']    = mnq['adx']
    df['cl_adx']     = cl['adx']
    df['mnq_atr']    = mnq['atr']
    df['cl_atr']     = cl['atr']
    df['mnq_ret1']   = mnq['ret1']
    df['cl_ret1']    = cl['ret1']

    # Direcoes
    df['mnq_acima_sma20'] = (df['mnq_close'] > df['mnq_sma20']).astype(int)
    df['cl_acima_sma20']  = (df['cl_close']  > df['cl_sma20']).astype(int)
    df['mnq_acima_ema20'] = (df['mnq_close'] > df['mnq_ema20']).astype(int)
    df['cl_acima_ema20']  = (df['cl_close']  > df['cl_ema20']).astype(int)

    # Price divergence (andando em direcoes opostas)
    df['price_div'] = df['mnq_ret1'] * df['cl_ret1']

    # RSI divergence
    df['rsi_div'] = df['mnq_rsi'] - df['cl_rsi']

    # ── CONDICOES DE ENTRADA ──
    # Horarios de avaliacao: 21H + a cada 4h
    ENTRY_HOURS = [21, 0, 4, 8, 12, 16, 20]

    trades = []

    for i in range(50, len(df) - 5):
        row = df.iloc[i]
        h = row['hour']
        if h not in ENTRY_HOURS:
            continue

        price_div = row['price_div']

        # So opera se estao andando em direcoes opostas
        if pd.isna(price_div) or price_div >= 0:
            continue

        # ADX > 30 em pelo menos um
        adx_ok = (row['mnq_adx'] > 30) or (row['cl_adx'] > 30)
        if not adx_ok:
            continue

        # MAs opostas: verificar SMA20 e EMA20
        # LONG MNQ: MNQ acima de ambas MAs, CL abaixo de ambas
        long_mnq_ma = (row['mnq_acima_sma20'] and row['mnq_acima_ema20']
                       and not row['cl_acima_sma20'] and not row['cl_acima_ema20'])
        # SHORT MNQ: MNQ abaixo de ambas MAs, CL acima de ambas
        short_mnq_ma = (not row['mnq_acima_sma20'] and not row['mnq_acima_ema20']
                        and row['cl_acima_sma20'] and row['cl_acima_ema20'])

        if not (long_mnq_ma or short_mnq_ma):
            continue

        # RSI divergente: LONG MNQ = RSI_MNQ > RSI_CL, SHORT MNQ = RSI_MNQ < RSI_CL
        rsi_div_val = row['rsi_div']
        if long_mnq_ma and rsi_div_val <= 0:
            continue
        if short_mnq_ma and rsi_div_val >= 0:
            continue

        # ── ENTRADA ──
        direction = 'LONG_MNQ' if long_mnq_ma else 'SHORT_MNQ'
        entry_time = df.index[i]
        entry_mnq = row['mnq_close']
        entry_cl  = row['cl_close']
        atr_val = max(row['mnq_atr'], row['cl_atr'])
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # R = 1 * atr (unidade de risco baseada no ATR medio)
        R = atr_val

        if direction == 'LONG_MNQ':
            stop_mnq = entry_mnq - RISK_R * R
            target_mnq = entry_mnq + TARGET_R * R
            stop_cl  = entry_cl + RISK_R * R   # CL esta SHORT
            target_cl = entry_cl - TARGET_R * R
        else:
            stop_mnq = entry_mnq + RISK_R * R
            target_mnq = entry_mnq - TARGET_R * R
            stop_cl  = entry_cl - RISK_R * R
            target_cl = entry_cl + TARGET_R * R

        # Simular saida nas proximas 48 barras (max 48h)
        result = None
        exit_time = None
        exit_mnq = None
        exit_cl = None
        bars_held = 0

        for j in range(i + 1, min(i + 49, len(df))):
            bars_held += 1
            bar = df.iloc[j]
            mnq_ohlc = mnq.iloc[j]
            cl_ohlc  = cl.iloc[j]
            bt = df.index[j]

            hit_stop = False
            hit_target = False

            if direction == 'LONG_MNQ':
                # MNQ long: stop abaixo, target acima
                if mnq_ohlc['low'] <= stop_mnq:
                    hit_stop = True
                    exit_mnq = stop_mnq
                elif mnq_ohlc['high'] >= target_mnq:
                    hit_target = True
                    exit_mnq = target_mnq
                # CL short: stop acima, target abaixo
                if cl_ohlc['high'] >= stop_cl:
                    hit_stop = True
                    exit_cl = stop_cl
                elif cl_ohlc['low'] <= target_cl:
                    hit_target = True
                    exit_cl = target_cl
            else:
                if mnq_ohlc['high'] >= stop_mnq:
                    hit_stop = True
                    exit_mnq = stop_mnq
                elif mnq_ohlc['low'] <= target_mnq:
                    hit_target = True
                    exit_mnq = target_mnq
                if cl_ohlc['low'] <= stop_cl:
                    hit_stop = True
                    exit_cl = stop_cl
                elif cl_ohlc['high'] >= target_cl:
                    hit_target = True
                    exit_cl = target_cl

            if hit_stop or hit_target:
                if exit_mnq is None:
                    exit_mnq = bar['mnq_close']
                if exit_cl is None:
                    exit_cl = bar['cl_close']
                result = 'WIN' if hit_target else 'LOSS'
                exit_time = bt
                break

        if result is None:
            exit_time = df.index[min(i + 48, len(df) - 1)]
            exit_mnq = df.iloc[min(i + 48, len(df) - 1)]['mnq_close']
            exit_cl  = df.iloc[min(i + 48, len(df) - 1)]['cl_close']
            result = 'EXPIRED'

        # Calcular PnL em %
        if direction == 'LONG_MNQ':
            pnl_mnq = (exit_mnq - entry_mnq) / entry_mnq * 100
            pnl_cl  = (entry_cl - exit_cl) / entry_cl * 100
        else:
            pnl_mnq = (entry_mnq - exit_mnq) / entry_mnq * 100
            pnl_cl  = (entry_cl - exit_cl) / entry_cl * 100

        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'result': result,
            'bars_held': bars_held,
            'entry_mnq': round(entry_mnq, 2),
            'exit_mnq': round(exit_mnq, 2),
            'entry_cl': round(entry_cl, 2),
            'exit_cl': round(exit_cl, 2),
            'pnl_mnq_pct': round(pnl_mnq, 2),
            'pnl_cl_pct': round(pnl_cl, 2),
            'pnl_total_pct': round(pnl_mnq + pnl_cl, 2),
            'mnq_adx': round(row['mnq_adx'], 1),
            'cl_adx': round(row['cl_adx'], 1),
            'mnq_rsi': round(row['mnq_rsi'], 1),
            'cl_rsi': round(row['cl_rsi'], 1),
            'rsi_div': round(rsi_div_val, 1),
            'price_div': round(price_div, 6),
            'hour': h,
        })

    # ── RESULTADOS ──
    print('=' * 70)
    print('  BACKTEST - HEDGE MNQ x CL')
    print(f'  Periodo: {df.index[50].date()} ate {df.index[-1].date()}')
    print(f'  Stop: {RISK_R}R  |  Target: {TARGET_R}R')
    print('=' * 70)

    if not trades:
        print('\n  Nenhum trade encontrado com as condicoes atuais.')
        return

    df_trades = pd.DataFrame(trades)

    total = len(df_trades)
    wins = len(df_trades[df_trades['result'] == 'WIN'])
    losses = len(df_trades[df_trades['result'] == 'LOSS'])
    expired = len(df_trades[df_trades['result'] == 'EXPIRED'])
    win_rate = wins / total * 100

    print(f'\n  Total de trades: {total}')
    print(f'  Vitorias: {wins}  ({win_rate:.1f}%)')
    print(f'  Derrotas: {losses}  ({(losses/total*100):.1f}%)')
    print(f'  Expirados: {expired}  ({(expired/total*100):.1f}%)')

    print(f'\n  PnL medio por trade:')
    print(f'    MNQ: {df_trades["pnl_mnq_pct"].mean():.2f}%')
    print(f'    CL:  {df_trades["pnl_cl_pct"].mean():.2f}%')
    print(f'    Total: {df_trades["pnl_total_pct"].mean():.2f}%')

    print(f'\n  PnL acumulado: {df_trades["pnl_total_pct"].sum():.2f}%')
    print(f'  Maior win: {df_trades["pnl_total_pct"].max():.2f}%')
    print(f'  Maior loss: {df_trades["pnl_total_pct"].min():.2f}%')

    # Por hora
    print(f'\n  Trades por hora de entrada:')
    for h in sorted(df_trades['hour'].unique()):
        sub = df_trades[df_trades['hour'] == h]
        wr = (sub['result'] == 'WIN').mean() * 100
        avg_pnl = sub['pnl_total_pct'].mean()
        print(f'    {int(h):02d}h: {len(sub):>3} trades  WR={wr:.0f}%  PnL medio={avg_pnl:+.2f}%')

    # Por direcao
    print(f'\n  Trades por direcao:')
    for direc in ['LONG_MNQ', 'SHORT_MNQ']:
        sub = df_trades[df_trades['direction'] == direc]
        if len(sub) == 0:
            continue
        wr = (sub['result'] == 'WIN').mean() * 100
        avg_pnl = sub['pnl_total_pct'].mean()
        print(f'    {direc:<10}: {len(sub):>3} trades  WR={wr:.0f}%  PnL medio={avg_pnl:+.2f}%')

    # Top 5 e worst 5
    print(f'\n  Top 5 melhores trades:')
    for _, t in df_trades.nlargest(5, 'pnl_total_pct').iterrows():
        hh = int(t['hour'])
        print(f'    {str(t["entry_time"].date()):>12} {hh:02d}h  {t["direction"]:<10}  PnL={t["pnl_total_pct"]:+.2f}%')

    print(f'\n  Top 5 piores trades:')
    for _, t in df_trades.nsmallest(5, 'pnl_total_pct').iterrows():
        hh = int(t['hour'])
        print(f'    {str(t["entry_time"].date()):>12} {hh:02d}h  {t["direction"]:<10}  PnL={t["pnl_total_pct"]:+.2f}%')

    # Salvar CSV
    csv_path = Path(__file__).parent / 'hedge_mnq_cl_resultados.csv'
    df_trades.to_csv(csv_path, index=False)
    print(f'\n  Resultados salvos em: {csv_path}')


if __name__ == '__main__':
    main()

"""
stats_zscore.py — Histogramas de Z-Score para NQ e CL.
Z = (X_hoje - media_20d) / std_20d

Saida: JSON com distribuicao historica + Z-Score atual de:
  - retorno diario, range diario (ADR), body 4H, volume
"""
import warnings; warnings.filterwarnings('ignore')
import json, sys
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = {'NQ': 'MNQ=F', 'CL': 'CL=F'}
ROLL    = 20   # janela para media/std rolling
BINS    = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]


def download(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index   = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()


def rollover_filter(df, max_pct):
    return df[df['close'].pct_change().abs() < max_pct].copy()


def zscore_series(series, roll=ROLL):
    """Z-Score rolling: (X - mean_N) / std_N"""
    mu  = series.rolling(roll, min_periods=5).mean().shift(1)
    sig = series.rolling(roll, min_periods=5).std().shift(1)
    return (series - mu) / sig.replace(0, np.nan)


def make_histogram(z_series):
    """Distribui Z-Scores historicos em bins e retorna contagens."""
    z = z_series.dropna()
    total = len(z)
    if total == 0:
        return []
    hist = []
    edges = BINS + [np.inf]  # ultimo bin captura > 4
    for i in range(len(BINS)):
        lo  = BINS[i]
        hi  = edges[i + 1]
        cnt = int(((z >= lo) & (z < hi)).sum())
        hist.append({
            'bin':  lo,
            'bin_label': f'{lo:+.1f}',
            'count': cnt,
            'pct':   round(cnt / total * 100, 2),
        })
    return hist


def percentile_of(z_series, current_z):
    """Que percentil o Z atual ocupa na distribuicao historica."""
    z = z_series.dropna()
    if len(z) == 0 or np.isnan(current_z):
        return None
    return round(float((z < current_z).mean() * 100), 1)


def r(v, dec=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), dec)


def compute_asset(name, ticker):
    max_pct = 0.015 if name == 'NQ' else 0.04

    df_d  = download(ticker, '5y',   '1d')
    df_1h = download(ticker, '730d', '1h')
    if df_d is None or df_1h is None:
        return {'error': f'falha ao baixar {ticker}'}

    df_d  = rollover_filter(df_d,  max_pct * 3)
    df_1h = rollover_filter(df_1h, max_pct)

    # 1H → 4H
    df_4h = df_1h.resample('4h').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'),     close=('close','last'),
        volume=('volume','sum')
    ).dropna()
    df_4h = df_4h[df_4h['volume'] > 0].copy()

    out = {}

    # ── 1. Retorno diario ─────────────────────────────────────────────────────
    df_d['ret']    = df_d['close'].pct_change() * 100
    df_d['range']  = (df_d['high'] - df_d['low']) / df_d['open'] * 100
    df_d['volume_z_raw'] = df_d['volume']

    df_d['ret_z']   = zscore_series(df_d['ret'])
    df_d['range_z'] = zscore_series(df_d['range'])
    df_d['vol_z']   = zscore_series(df_d['volume_z_raw'])

    # ── 2. Body 4H ────────────────────────────────────────────────────────────
    df_4h['body']   = (df_4h['close'] - df_4h['open']).abs() / df_4h['open'] * 100
    df_4h['body_z'] = zscore_series(df_4h['body'])

    # ── Valores atuais (ultima barra fechada) ─────────────────────────────────
    cur_ret_z   = float(df_d['ret_z'].iloc[-1])   if not df_d['ret_z'].empty else float('nan')
    cur_range_z = float(df_d['range_z'].iloc[-1]) if not df_d['range_z'].empty else float('nan')
    cur_vol_z   = float(df_d['vol_z'].iloc[-1])   if not df_d['vol_z'].empty else float('nan')
    cur_body_z  = float(df_4h['body_z'].iloc[-2]) if len(df_4h) >= 2 else float('nan')

    cur_ret   = r(df_d['ret'].iloc[-1],   3)
    cur_range = r(df_d['range'].iloc[-1], 3)
    cur_vol   = r(df_d['volume_z_raw'].iloc[-1], 0)

    def metric(series_z, cur_z, cur_raw=None, unit='%'):
        z = series_z.dropna()
        return {
            'histogram':   make_histogram(z),
            'n':           len(z),
            'mean':        r(z.mean(), 4),
            'std':         r(z.std(),  4),
            'current_z':   r(cur_z,    3),
            'current_raw': cur_raw,
            'unit':        unit,
            'percentile':  percentile_of(z, cur_z),
            'extremo':     abs(cur_z) >= 2 if not np.isnan(cur_z) else False,
        }

    out['ret_diario'] = metric(df_d['ret_z'],   cur_ret_z,   cur_ret,   '%')
    out['range_diario'] = metric(df_d['range_z'], cur_range_z, cur_range, '%')
    out['body_4h']    = metric(df_4h['body_z'], cur_body_z,  None,      '%')
    out['volume']     = metric(df_d['vol_z'],   cur_vol_z,   cur_vol,   'vol')

    # Estatisticas da distribuicao historica completa de Z
    all_z = pd.concat([df_d['ret_z'], df_d['range_z'], df_d['vol_z'], df_4h['body_z']]).dropna()
    out['meta'] = {
        'ticker':   ticker,
        'nome':     name,
        'roll':     ROLL,
        'n_diario': len(df_d),
        'n_4h':     len(df_4h),
        'ts_inicio': str(df_d.index[0].date()),
        'ts_fim':    str(df_d.index[-1].date()),
    }
    return out


def compute():
    result = {}
    for name, ticker in TICKERS.items():
        result[name] = compute_asset(name, ticker)
    result['ts'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return result


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    data = compute()
    print(json.dumps(data, default=str, ensure_ascii=False))

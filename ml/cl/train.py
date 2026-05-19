"""
Treina XGBoost para prever direcao do CL (LONG/NEUTRO/SHORT).
Feature set adaptado: CL como ativo primario, MNQ/BTC como secundarios.
Uso: python train.py [--forward 4] [--interval 1h] [--all-hours] [--full] [--spreads-only]
"""
import argparse
import sqlite3
import pickle
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DB        = Path(__file__).parent / 'data.db'
MODEL_OUT = Path(__file__).parent / 'model.pkl'
FIG_OUT   = Path(__file__).parent / 'feature_importance.png'


def load_symbol(conn, symbol, interval='1h'):
    ts_len = 19 if interval == '1h' else 10
    return pd.read_sql(
        'SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND LENGTH(ts)=? ORDER BY ts',
        conn, params=(symbol, ts_len), parse_dates=['ts'], index_col='ts',
    )


def compute(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
    adx_i     = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=17)
    df['adx']  = adx_i.adx()
    df['pdi']  = adx_i.adx_pos()
    df['mdi']  = adx_i.adx_neg()
    df['ret1'] = df['close'].pct_change(1)
    df['ret4'] = df['close'].pct_change(4)
    df['ret8'] = df['close'].pct_change(8)
    df['vol']  = df['ret1'].rolling(20).std()
    df['bb_w'] = df['close'].rolling(20).std() * 2 / df['close'].rolling(20).mean()
    df['sma50']       = df['close'].rolling(50).mean()
    df['dist_sma50']  = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['sma50_slope'] = df['sma50'].pct_change(5) * 100
    df['above_sma50'] = (df['close'] > df['sma50']).astype(int)
    df['ema20']       = df['close'].ewm(span=20, adjust=False).mean()
    df['dist_ema20']  = (df['close'] - df['ema20']) / df['ema20'] * 100
    df['above_ema20'] = (df['close'] > df['ema20']).astype(int)
    return df


KEY_LEVEL_FEATURES = [
    'dist_to_pdh', 'dist_to_pdl', 'dist_to_do',
    'above_do', 'above_pdh', 'above_pdl', 'prev_day_range_pct',
    'dist_to_pwh', 'dist_to_pwl', 'dist_to_wo',
    'above_wo', 'above_pwh', 'above_pwl',
    'dist_to_pmh', 'dist_to_pml', 'dist_to_mo',
    'above_mo', 'above_pmh', 'above_pml',
    'dist_to_mday_h', 'dist_to_mday_l',
    'above_mday_h', 'above_mday_l',
]


def add_key_levels(primary_df, f):
    """Adiciona key levels (Daily, Weekly, Monthly, Monday range) como features."""
    idx = f.index
    c   = primary_df['close'].reindex(idx, method='ffill')

    def pct(series, ref):
        safe_ref = ref.replace(0, float('nan'))
        return (series - safe_ref) / safe_ref * 100

    daily = primary_df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pdh = daily['high'].shift(1).reindex(idx, method='ffill')
    pdl = daily['low'].shift(1).reindex(idx, method='ffill')
    do_ = daily['open'].reindex(idx, method='ffill')
    f['dist_to_pdh'] = pct(c, pdh)
    f['dist_to_pdl'] = pct(c, pdl)
    f['dist_to_do']  = pct(c, do_)
    f['above_do']    = (c > do_).astype(int)
    f['above_pdh']   = (c > pdh).astype(int)
    f['above_pdl']   = (c > pdl).astype(int)
    f['prev_day_range_pct'] = pct(
        daily['high'].shift(1), daily['low'].shift(1)
    ).reindex(idx, method='ffill')

    weekly = primary_df.resample('W-SUN').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pwh = weekly['high'].shift(1).reindex(idx, method='ffill')
    pwl = weekly['low'].shift(1).reindex(idx, method='ffill')
    wo  = weekly['open'].reindex(idx, method='ffill')
    f['dist_to_pwh'] = pct(c, pwh)
    f['dist_to_pwl'] = pct(c, pwl)
    f['dist_to_wo']  = pct(c, wo)
    f['above_wo']    = (c > wo).astype(int)
    f['above_pwh']   = (c > pwh).astype(int)
    f['above_pwl']   = (c > pwl).astype(int)

    monthly = primary_df.resample('MS').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pmh = monthly['high'].shift(1).reindex(idx, method='ffill')
    pml = monthly['low'].shift(1).reindex(idx, method='ffill')
    mo  = monthly['open'].reindex(idx, method='ffill')
    f['dist_to_pmh'] = pct(c, pmh)
    f['dist_to_pml'] = pct(c, pml)
    f['dist_to_mo']  = pct(c, mo)
    f['above_mo']    = (c > mo).astype(int)
    f['above_pmh']   = (c > pmh).astype(int)
    f['above_pml']   = (c > pml).astype(int)

    monday_bars = primary_df[primary_df.index.dayofweek == 0]
    if len(monday_bars) >= 4:
        mday_h = monday_bars['high'].resample('W-SUN').max().reindex(idx, method='ffill')
        mday_l = monday_bars['low'].resample('W-SUN').min().reindex(idx, method='ffill')
        f['dist_to_mday_h'] = pct(c, mday_h)
        f['dist_to_mday_l'] = pct(c, mday_l)
        f['above_mday_h']   = (c > mday_h).astype(int)
        f['above_mday_l']   = (c > mday_l).astype(int)
    else:
        for col in ['dist_to_mday_h', 'dist_to_mday_l', 'above_mday_h', 'above_mday_l']:
            f[col] = 0.0


def build_features(conn, interval='1h', forward=8, min_ret=0.001):
    cl  = compute(load_symbol(conn, 'cl',  interval))
    mnq = compute(load_symbol(conn, 'mnq', interval))
    btc = compute(load_symbol(conn, 'btc', interval))

    idx = cl.index
    f = pd.DataFrame(index=idx)

    # RSI dos 3 ativos
    f['rsi_cl']  = cl['rsi']
    f['rsi_mnq'] = mnq['rsi'].reindex(idx, method='ffill')
    f['rsi_btc'] = btc['rsi'].reindex(idx, method='ffill')

    # ADX + DI
    f['adx_cl']  = cl['adx']
    f['adx_mnq'] = mnq['adx'].reindex(idx, method='ffill')
    f['adx_btc'] = btc['adx'].reindex(idx, method='ffill')
    f['pdi_cl']  = cl['pdi']
    f['mdi_cl']  = cl['mdi']

    # Divergencia RSI (CL vs outros)
    f['div_mnq'] = f['rsi_cl'] - f['rsi_mnq']
    f['div_btc'] = f['rsi_cl'] - f['rsi_btc']

    # Retornos e vol CL (primario)
    f['ret1_cl'] = cl['ret1']
    f['ret4_cl'] = cl['ret4']
    f['ret8_cl'] = cl['ret8']
    f['vol_cl']  = cl['vol']
    f['bb_cl']   = cl['bb_w']

    # Retornos MNQ e BTC (secundarios)
    f['ret1_mnq'] = mnq['ret1'].reindex(idx, method='ffill')
    f['ret4_mnq'] = mnq['ret4'].reindex(idx, method='ffill')
    f['ret1_btc'] = btc['ret1'].reindex(idx, method='ffill')
    f['ret4_btc'] = btc['ret4'].reindex(idx, method='ffill')

    # Tempo
    f['hour'] = idx.hour
    f['dow']  = idx.dayofweek

    # Aceleracao do ADX
    f['dadx_cl'] = f['adx_cl'].diff(2)

    # Sinal triplo adaptado (CL como referencia)
    f['triple_signal'] = (
        (f['rsi_btc'] < 45) & (f['div_mnq'] < 0) & (f['div_btc'] < 0)
    ).astype(int)

    # MA50 dos 3 ativos
    f['dist_sma50_cl']   = cl['dist_sma50']
    f['dist_sma50_mnq']  = mnq['dist_sma50'].reindex(idx, method='ffill')
    f['dist_sma50_btc']  = btc['dist_sma50'].reindex(idx, method='ffill')
    f['sma50_slope_cl']  = cl['sma50_slope']
    f['above_sma50_cl']  = cl['above_sma50']
    f['above_sma50_mnq'] = mnq['above_sma50'].reindex(idx, method='ffill')
    f['above_sma50_btc'] = btc['above_sma50'].reindex(idx, method='ffill')
    f['sma50_alignment'] = f['above_sma50_cl'] + f['above_sma50_mnq'] + f['above_sma50_btc']

    # EMA20 dos 3 ativos
    f['dist_ema20_cl']   = cl['dist_ema20']
    f['dist_ema20_mnq']  = mnq['dist_ema20'].reindex(idx, method='ffill')
    f['dist_ema20_btc']  = btc['dist_ema20'].reindex(idx, method='ffill')
    f['above_ema20_cl']  = cl['above_ema20']
    f['above_ema20_mnq'] = mnq['above_ema20'].reindex(idx, method='ffill')
    f['above_ema20_btc'] = btc['above_ema20'].reindex(idx, method='ffill')
    f['ema20_bias_mnq_btc'] = f['above_ema20_cl'] + f['above_ema20_btc']
    f['ema20_alignment']    = f['above_ema20_cl'] + f['above_ema20_mnq'] + f['above_ema20_btc']

    # ── SPREADS COMPLETOS (todas combinacoes entre CL, MNQ, BTC) ──
    ret_mnq = mnq['ret1'].reindex(idx, method='ffill')
    ret_btc = btc['ret1'].reindex(idx, method='ffill')
    vol_mnq = mnq['vol'].reindex(idx, method='ffill')
    vol_btc = btc['vol'].reindex(idx, method='ffill')
    bb_mnq  = mnq['bb_w'].reindex(idx, method='ffill')
    bb_btc  = btc['bb_w'].reindex(idx, method='ffill')
    pdi_mnq = mnq['pdi'].reindex(idx, method='ffill')
    mdi_mnq = mnq['mdi'].reindex(idx, method='ffill')
    pdi_btc = btc['pdi'].reindex(idx, method='ffill')
    mdi_btc = btc['mdi'].reindex(idx, method='ffill')

    # RSI spreads
    f['rsi_spread_mnq_btc'] = f['rsi_mnq'] - f['rsi_btc']
    f['rsi_abs_cl_mnq']     = (f['rsi_cl'] - f['rsi_mnq']).abs()
    f['rsi_abs_cl_btc']     = (f['rsi_cl'] - f['rsi_btc']).abs()
    f['rsi_abs_mnq_btc']    = (f['rsi_mnq'] - f['rsi_btc']).abs()

    # ADX spreads
    f['adx_spread_cl_mnq']  = f['adx_cl'] - f['adx_mnq']
    f['adx_spread_cl_btc']  = f['adx_cl'] - f['adx_btc']
    f['adx_spread_mnq_btc'] = f['adx_mnq'] - f['adx_btc']
    f['adx_abs_cl_mnq']     = (f['adx_cl'] - f['adx_mnq']).abs()
    f['adx_abs_cl_btc']     = (f['adx_cl'] - f['adx_btc']).abs()
    f['adx_abs_mnq_btc']    = (f['adx_mnq'] - f['adx_btc']).abs()

    # Retorno differences
    f['ret1_spread_cl_mnq']  = f['ret1_cl'] - ret_mnq
    f['ret1_spread_cl_btc']  = f['ret1_cl'] - ret_btc
    f['ret1_spread_mnq_btc'] = ret_mnq - ret_btc
    ret4_mnq_v = mnq['ret4'].reindex(idx, method='ffill')
    ret4_btc_v = btc['ret4'].reindex(idx, method='ffill')
    f['ret4_spread_cl_mnq']  = f['ret4_cl'] - ret4_mnq_v
    f['ret4_spread_cl_btc']  = f['ret4_cl'] - ret4_btc_v
    f['ret4_spread_mnq_btc'] = ret4_mnq_v - ret4_btc_v

    # Co-movement
    f['ret1_prod_cl_mnq']  = f['ret1_cl'] * ret_mnq
    f['ret1_prod_cl_btc']  = f['ret1_cl'] * ret_btc
    f['price_div_cl']      = f['ret1_cl'] * ret_mnq
    f['price_div_abs']     = f['price_div_cl'].abs()
    f['ret4_prod_cl_mnq']  = f['ret4_cl'] * ret4_mnq_v
    f['ret4_prod_cl_btc']  = f['ret4_cl'] * ret4_btc_v
    f['ret4_prod_mnq_btc'] = ret4_mnq_v * ret4_btc_v

    # Volatilidade spreads
    f['vol_spread_cl_mnq']  = f['vol_cl'] - vol_mnq
    f['vol_spread_cl_btc']  = f['vol_cl'] - vol_btc
    f['vol_spread_mnq_btc'] = vol_mnq - vol_btc

    # Bollinger spreads
    f['bb_spread_cl_mnq']  = f['bb_cl'] - bb_mnq
    f['bb_spread_cl_btc']  = f['bb_cl'] - bb_btc
    f['bb_spread_mnq_btc'] = bb_mnq - bb_btc

    # MA50 distance spreads
    m50_mnq = mnq['dist_sma50'].reindex(idx, method='ffill')
    m50_btc = btc['dist_sma50'].reindex(idx, method='ffill')
    f['sma50_dist_spread_cl_mnq']  = f['dist_sma50_cl'] - m50_mnq
    f['sma50_dist_spread_cl_btc']  = f['dist_sma50_cl'] - m50_btc
    f['sma50_dist_spread_mnq_btc'] = m50_mnq - m50_btc

    # EMA20 distance spreads
    e20_mnq = mnq['dist_ema20'].reindex(idx, method='ffill')
    e20_btc = btc['dist_ema20'].reindex(idx, method='ffill')
    f['ema20_dist_spread_cl_mnq']  = f['dist_ema20_cl'] - e20_mnq
    f['ema20_dist_spread_cl_btc']  = f['dist_ema20_cl'] - e20_btc
    f['ema20_dist_spread_mnq_btc'] = e20_mnq - e20_btc

    # Alinhamentos MA50 (pares)
    ab50_mnq = mnq['above_sma50'].reindex(idx, method='ffill')
    ab50_btc = btc['above_sma50'].reindex(idx, method='ffill')
    f['sma50_align_cl_mnq']  = f['above_sma50_cl'] + ab50_mnq
    f['sma50_align_cl_btc']  = f['above_sma50_cl'] + ab50_btc
    f['sma50_align_mnq_btc'] = ab50_mnq + ab50_btc

    # Alinhamentos EMA20 (pares)
    eb20_mnq = mnq['above_ema20'].reindex(idx, method='ffill')
    eb20_btc = btc['above_ema20'].reindex(idx, method='ffill')
    f['ema20_align_cl_mnq']  = f['above_ema20_cl'] + eb20_mnq
    f['ema20_align_cl_btc']  = f['above_ema20_cl'] + eb20_btc
    f['ema20_align_mnq_btc'] = eb20_mnq + eb20_btc

    # DI spread (forca direcional intra-ativo)
    f['di_spread_cl']  = f['pdi_cl'] - f['mdi_cl']
    f['di_spread_mnq'] = pdi_mnq - mdi_mnq
    f['di_spread_btc'] = pdi_btc - mdi_btc

    # ── COMPOSITE TRIGGERS ────────────────────────────────────────
    f['adx_above_14'] = (f['adx_cl'] > 14).astype(int)
    f['adx_above_20'] = (f['adx_cl'] > 20).astype(int)
    f['adx_above_25'] = (f['adx_cl'] > 25).astype(int)

    f['triple_signal'] = (
        (f['rsi_btc'] < 45) & (f['div_mnq'] < 0) & (f['div_btc'] < 0)
    ).astype(int)

    f['is_evening'] = f['hour'].isin([18, 19, 20, 21]).astype(int)
    US_HOURS = list(range(9, 18))
    f['is_us_session']   = f['hour'].isin(US_HOURS).astype(int)
    f['is_us_morning']   = f['hour'].isin([9, 10, 11, 12, 13]).astype(int)
    f['is_us_afternoon'] = f['hour'].isin([14, 15, 16, 17]).astype(int)

    f['strong_div'] = (
        (f['price_div_cl'] < 0) & (f['adx_cl'] > 14)
    ).astype(int)
    f['prime_setup'] = (
        (f['price_div_cl'] < 0) & (f['adx_cl'] > 14) & f['hour'].isin([18, 19, 20, 21])
    ).astype(int)
    f['us_prime_setup'] = (
        (f['price_div_cl'] < 0) & (f['adx_cl'] > 14) & f['is_us_session'] == 1
    ).astype(int)
    f['cl_down_mnq_up'] = (
        (f['ret1_cl'] < 0) & (f['ret1_mnq'] > 0)
    ).astype(int)
    # ─────────────────────────────────────────────────────────────

    # ── KILL ZONE FEATURES (UTC) ──
    ASIA_HOURS   = [0, 1, 2, 3, 4, 5, 6, 7]
    LONDON_HOURS = [8, 9, 10, 11, 12, 13, 14]
    NY_HOURS     = [13, 14, 15, 16, 17, 18, 19]
    OVERLAP_HOURS = [13, 14]

    f['is_asia']   = f['hour'].isin(ASIA_HOURS).astype(int)
    f['is_london'] = f['hour'].isin(LONDON_HOURS).astype(int)
    f['is_ny']     = f['hour'].isin(NY_HOURS).astype(int)
    f['kz_overlap'] = f['hour'].isin(OVERLAP_HOURS).astype(int)
    in_kz = (f['is_asia'] == 1) | (f['is_london'] == 1) | (f['is_ny'] == 1)

    prev_in_kz = in_kz.shift(1).fillna(False)
    kz_session_id = (in_kz & ~prev_in_kz).cumsum()
    kz_session_id[~in_kz] = -1

    primary_high = cl['high']
    primary_low  = cl['low']
    primary_close = cl['close']

    session_high = primary_high.groupby(kz_session_id).cummax()
    session_low  = primary_low.groupby(kz_session_id).cummin()

    f['kz_dist_high'] = ((primary_close - session_high) / session_high * 100).where(in_kz == 1, 0)
    f['kz_dist_low']  = ((primary_close - session_low) / session_low * 100).where(in_kz == 1, 0)
    f['kz_range']     = ((session_high - session_low) / session_low * 100).where(in_kz == 1, 0)
    f['kz_progress']  = 0.0
    for kz_col in ['is_asia', 'is_london', 'is_ny']:
        grp = f[kz_col].ne(f[kz_col].shift()).cumsum()
        grp[f[kz_col] == 0] = -1
        progress = grp.groupby(grp).cumcount() / grp.groupby(grp).transform('count')
        f['kz_progress'] = f['kz_progress'].where(f[kz_col] == 0, progress)

    prev_sh = session_high.shift(1)
    prev_sl = session_low.shift(1)
    f['kz_breakout_up'] = ((primary_close > prev_sh) & (primary_close.shift(1) <= prev_sh) & in_kz).astype(int)
    f['kz_breakout_dn'] = ((primary_close < prev_sl) & (primary_close.shift(1) >= prev_sl) & in_kz).astype(int)

    add_key_levels(cl, f)

    # Label ternario baseado no CL
    future_price   = cl['close'].shift(-forward)
    f['future_ret'] = future_price / cl['close'] - 1
    f['label']      = 1
    f.loc[f['future_ret'] >  min_ret, 'label'] = 2
    f.loc[f['future_ret'] < -min_ret, 'label'] = 0

    return f.dropna()


FEATURE_COLS = [
    'rsi_cl', 'rsi_mnq', 'rsi_btc',
    'adx_cl', 'adx_mnq', 'adx_btc',
    'pdi_cl', 'mdi_cl',
    'div_mnq', 'div_btc',
    'ret1_cl', 'ret4_cl', 'ret8_cl', 'vol_cl', 'bb_cl',
    'ret1_mnq', 'ret4_mnq',
    'ret1_btc', 'ret4_btc',
    'hour', 'dow',
    'dadx_cl', 'triple_signal',
    'price_div_cl', 'price_div_abs',
    'adx_above_14', 'adx_above_20', 'adx_above_25',
    'is_evening', 'strong_div', 'prime_setup', 'cl_down_mnq_up',
    'is_us_session', 'is_us_morning', 'is_us_afternoon', 'us_prime_setup',
    # Kill zones
    'is_asia', 'is_london', 'is_ny', 'kz_overlap',
    'kz_dist_high', 'kz_dist_low', 'kz_range', 'kz_progress',
    'kz_breakout_up', 'kz_breakout_dn',
    'dist_sma50_cl', 'dist_sma50_mnq', 'dist_sma50_btc',
    'sma50_slope_cl', 'above_sma50_cl', 'above_sma50_mnq', 'above_sma50_btc',
    'sma50_alignment',
    'dist_ema20_cl', 'dist_ema20_mnq', 'dist_ema20_btc',
    'above_ema20_cl', 'above_ema20_mnq', 'above_ema20_btc',
    'ema20_bias_mnq_btc', 'ema20_alignment',
]

FEATURE_COLS_SPREADS = [
    'div_mnq', 'div_btc',
    'price_div_cl', 'price_div_abs',
    'triple_signal', 'strong_div', 'prime_setup', 'us_prime_setup',
    'cl_down_mnq_up',
    'rsi_spread_mnq_btc',
    'rsi_abs_cl_mnq', 'rsi_abs_cl_btc', 'rsi_abs_mnq_btc',
    'adx_spread_cl_mnq', 'adx_spread_cl_btc', 'adx_spread_mnq_btc',
    'adx_abs_cl_mnq', 'adx_abs_cl_btc', 'adx_abs_mnq_btc',
    'ret1_spread_cl_mnq', 'ret1_spread_cl_btc', 'ret1_spread_mnq_btc',
    'ret4_spread_cl_mnq', 'ret4_spread_cl_btc', 'ret4_spread_mnq_btc',
    'ret1_prod_cl_mnq', 'ret1_prod_cl_btc',
    'ret4_prod_cl_mnq', 'ret4_prod_cl_btc', 'ret4_prod_mnq_btc',
    'vol_spread_cl_mnq', 'vol_spread_cl_btc', 'vol_spread_mnq_btc',
    'bb_spread_cl_mnq', 'bb_spread_cl_btc', 'bb_spread_mnq_btc',
    'sma50_dist_spread_cl_mnq', 'sma50_dist_spread_cl_btc', 'sma50_dist_spread_mnq_btc',
    'ema20_dist_spread_cl_mnq', 'ema20_dist_spread_cl_btc', 'ema20_dist_spread_mnq_btc',
    'sma50_align_cl_mnq', 'sma50_align_cl_btc', 'sma50_align_mnq_btc',
    'sma50_alignment',
    'ema20_align_cl_mnq', 'ema20_align_cl_btc', 'ema20_align_mnq_btc',
    'ema20_alignment', 'ema20_bias_mnq_btc',
    'di_spread_cl', 'di_spread_mnq', 'di_spread_btc',
    'adx_above_14', 'adx_above_20', 'adx_above_25',
    'hour', 'dow',
    'is_evening',
    'is_us_session', 'is_us_morning', 'is_us_afternoon',
    # Kill zones
    'is_asia', 'is_london', 'is_ny', 'kz_overlap',
    'kz_dist_high', 'kz_dist_low', 'kz_range', 'kz_progress',
    'kz_breakout_up', 'kz_breakout_dn',
]

FEATURE_COLS_OPTIMIZED = [
    'hour', 'dow',
    'is_us_session', 'is_us_morning', 'is_us_afternoon', 'is_evening',
    'price_div_abs', 'price_div_cl',
    'div_mnq', 'div_btc',
    'rsi_cl',
    'vol_cl',
    'dist_ema20_cl',
    'bb_cl',
    'sma50_slope_cl',
    'dist_sma50_cl',
    'ret1_cl',
    'adx_mnq',
    'adx_btc',
    'dist_ema20_mnq',
    'sma50_alignment',
    'ema20_bias_mnq_btc',
    'ema20_alignment',
    'adx_above_14',
    'adx_cl',
    'di_spread_cl',
    'vol_spread_cl_mnq',
    'vol_spread_cl_btc',
    'ret4_prod_cl_mnq',
    'adx_spread_mnq_btc',
    'sma50_align_cl_mnq',
    'ema20_align_cl_mnq',
    'rsi_abs_cl_mnq',
    'bb_spread_cl_mnq',
    'strong_div',
    'us_prime_setup',
    'prime_setup',
    'cl_down_mnq_up',
    # Kill Zones & Breakouts
    'is_asia', 'is_london', 'is_ny', 'kz_overlap',
    'kz_dist_high', 'kz_dist_low', 'kz_range',
    'kz_breakout_up', 'kz_breakout_dn',
] + KEY_LEVEL_FEATURES

US_SESSION_HOURS = list(range(9, 18))


def train_model(X, y):
    split = int(len(X) * 0.70)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    counts = y_tr.value_counts().sort_index()
    weights = {c: len(y_tr) / (3 * count) for c, count in counts.items()}
    sample_weight = y_tr.map(weights)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=8,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        early_stopping_rounds=40,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              sample_weight=sample_weight.values, verbose=False)

    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)

    print('\n========================================')
    print('  Walk-Forward - resultado fora da amostra')
    print('========================================')
    print(f'  Treino : {len(X_tr)} amostras  ({X_tr.index[0].date()} ate {X_tr.index[-1].date()})')
    print(f'  Teste  : {len(X_te)} amostras  ({X_te.index[0].date()} ate {X_te.index[-1].date()})')
    dist = y_te.value_counts(normalize=True).sort_index()
    print(f'  Distribuicao real: SHORT={dist.get(0,0):.1%}  NEUTRO={dist.get(1,0):.1%}  LONG={dist.get(2,0):.1%}\n')
    print(classification_report(y_te, preds, target_names=['SHORT', 'NEUTRO', 'LONG']))

    auc = roc_auc_score(y_te, proba, multi_class='ovr')
    rating = 'bom' if auc > 0.55 else 'fraco - mais dados ajudam'
    print(f'  ROC-AUC (macro ovr): {auc:.4f}  [{rating}]')

    print('\n--- Analise dos gatilhos no periodo de teste ---')
    df_te = X_te.copy()
    df_te['label']      = y_te.values
    df_te['prob_short'] = proba[:, 0]
    df_te['prob_long']  = proba[:, 2]
    df_te['pred']       = preds

    for nome, col in [('strong_div', 'strong_div'), ('us_prime_setup', 'us_prime_setup')]:
        sub = df_te[df_te[col] == 1] if col in df_te.columns else pd.DataFrame()
        if len(sub) >= 10:
            long_pct  = (sub['label'] == 2).mean()
            short_pct = (sub['label'] == 0).mean()
            print(f'  {nome}: N={len(sub)}  LONG={long_pct:.1%}  SHORT={short_pct:.1%}  '
                  f'prob_long_media={sub["prob_long"].mean():.1%}  prob_short_media={sub["prob_short"].mean():.1%}')

    if 'ema20_bias_mnq_btc' in df_te.columns:
        print(f'\n  Vies EMA20 (CL+BTC):')
        for v, lbl in [(0, 'ambos abaixo'), (1, 'misturado'), (2, 'ambos acima')]:
            sub = df_te[df_te['ema20_bias_mnq_btc'] == v]
            if len(sub) < 10: continue
            long_pct  = (sub['label'] == 2).mean()
            short_pct = (sub['label'] == 0).mean()
            print(f'    {lbl}: N={len(sub)}  LONG={long_pct:.1%}  SHORT={short_pct:.1%}')

    print('\n  Direcao predominante por hora (teste):')
    for h in sorted(df_te['hour'].unique()):
        sub = df_te[df_te['hour'] == h]
        if len(sub) < 20: continue
        long_pct  = (sub['label'] == 2).mean()
        short_pct = (sub['label'] == 0).mean()
        neutro_pct = (sub['label'] == 1).mean()
        print(f'    {h:02d}h: LONG={long_pct:.1%}  NEUTRO={neutro_pct:.1%}  SHORT={short_pct:.1%}  (N={len(sub)})')

    # Kill zone analysis
    print('\n  Kill Zone analysis (test set):')
    for kz_col, kz_name in [('is_asia', 'Asia'), ('is_london', 'London'), ('is_ny', 'NY'),
                             ('kz_overlap', 'London+NY Overlap')]:
        if kz_col not in df_te.columns: continue
        sub = df_te[df_te[kz_col] == 1]
        if len(sub) < 10: continue
        long_pct  = (sub['label'] == 2).mean()
        short_pct = (sub['label'] == 0).mean()
        prob_l = sub['prob_long'].mean()
        prob_s = sub['prob_short'].mean()
        print(f'    {kz_name}: N={len(sub)}  LONG={long_pct:.1%}  SHORT={short_pct:.1%}  '
              f'prob_long_avg={prob_l:.1%}  prob_short_avg={prob_s:.1%}')

    for bo_col, bo_name in [('kz_breakout_up', 'KZ Bull Breakout'), ('kz_breakout_dn', 'KZ Bear Breakout')]:
        if bo_col not in df_te.columns: continue
        sub = df_te[df_te[bo_col] == 1]
        if len(sub) < 5: continue
        long_pct  = (sub['label'] == 2).mean()
        short_pct = (sub['label'] == 0).mean()
        print(f'    {bo_name}: N={len(sub)}  LONG={long_pct:.1%}  SHORT={short_pct:.1%}')

    return model, auc


def plot_importance(model):
    fig, ax = plt.subplots(figsize=(9, 7))
    xgb.plot_importance(model, ax=ax, max_num_features=20, importance_type='gain')
    ax.set_title('Feature Importance (Gain) - CL XGBoost')
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=130)
    plt.close()
    print(f'\n  Grafico salvo em: {FIG_OUT}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--forward',   type=int,   default=4,     help='Barras a frente (default 4)')
    p.add_argument('--interval',  type=str,   default='1h',  help='1h ou 1d (default 1h)')
    p.add_argument('--min-ret',   type=float, default=0.001, help='Retorno minimo LONG (default 0.1%%)')
    p.add_argument('--all-hours',   action='store_true', help='Treinar em todas as horas (nao so sessao US)')
    p.add_argument('--full',        action='store_true', help='Usar feature set completo (52 features)')
    p.add_argument('--spreads-only', action='store_true', help='Usar SO features spread (63 features)')
    p.add_argument('--max-months', type=int, default=0, help='Usar apenas ultimos N meses de dados (0=todos)')
    p.add_argument('--kill-zone', type=str, default=None, choices=['asia', 'london', 'ny'],
                   help='Treinar apenas para uma kill zone especifica (asia/london/ny)')
    args = p.parse_args()

    if not DB.exists():
        print('DB nao encontrado. Rode collect_data.py primeiro.')
        return

    print(f'Config: interval={args.interval}  forward={args.forward}h  min_ret={args.min_ret:.2%}')
    conn = sqlite3.connect(DB)
    print('Calculando features...')
    df = build_features(conn, args.interval, args.forward, args.min_ret)
    conn.close()

    if args.max_months > 0:
        cutoff = df.index.max() - pd.DateOffset(months=args.max_months)
        total_antes = len(df)
        df = df[df.index >= cutoff].copy()
        print(f'Filtro ultimos {args.max_months} meses: {total_antes} -> {len(df)} amostras')

    if not args.all_hours:
        total_antes = len(df)
        df = df[df['hour'].isin(US_SESSION_HOURS)].copy()
        print(f'Filtro sessao US (horas 9-17): {total_antes} -> {len(df)} amostras')
    else:
        print(f'Modo todas as horas ativo')

    # Filtrar por kill zone especifica
    if args.kill_zone:
        KZ_HOURS = {'asia': [0,1,2,3,4,5,6,7], 'london': [8,9,10,11,12,13,14], 'ny': [13,14,15,16,17,18,19]}
        total_antes = len(df)
        df = df[df['hour'].isin(KZ_HOURS[args.kill_zone])].copy()
        print(f'Filtro kill zone {args.kill_zone}: {total_antes} -> {len(df)} amostras')

    dist = df['label'].value_counts(normalize=True).sort_index()

    print(f'Dataset: {len(df)} amostras  |  SHORT={dist.get(0,0):.1%}  NEUTRO={dist.get(1,0):.1%}  LONG={dist.get(2,0):.1%}')
    print(f'Amostras strong_div: {df["strong_div"].sum()}  ({df["strong_div"].mean():.1%})')
    print(f'Amostras us_prime_setup: {df["us_prime_setup"].sum()}')
    for kz, name in [('is_asia', 'Asia'), ('is_london', 'London'), ('is_ny', 'NY')]:
        n = df[kz].sum()
        print(f'  {name} KZ: {n} amostras ({df[kz].mean():.1%})')
    n_bu = df['kz_breakout_up'].sum()
    n_bd = df['kz_breakout_dn'].sum()
    print(f'  KZ Breakouts: {n_bu} UP / {n_bd} DN')

    if len(df) < 300:
        print('Poucos dados. Rode collect_data.py.')
        return

    if args.spreads_only:
        feat_cols = FEATURE_COLS_SPREADS
    elif args.full:
        feat_cols = FEATURE_COLS
    else:
        feat_cols = FEATURE_COLS_OPTIMIZED
    X, y = df[feat_cols], df['label']
    print(f'Features: {len(feat_cols)}  (modo: {"spreads-only" if args.spreads_only else "completo" if args.full else "otimizado (default)"})')
    model, auc = train_model(X, y)
    plot_importance(model)

    model_path = Path(__file__).parent / f'model_kz_{args.kill_zone}.pkl' if args.kill_zone else MODEL_OUT
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model':    model,
            'features': feat_cols,
            'forward':  args.forward,
            'interval': args.interval,
            'auc':      auc,
            'kill_zone': args.kill_zone,
        }, f)
    print(f'  Modelo salvo em: {model_path}')


if __name__ == '__main__':
    main()

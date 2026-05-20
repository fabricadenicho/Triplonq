"""
Backtest: carrega modelo treinado e testa em dados historicos do DB.
Nao baixa nada do Yahoo Finance.

Uso: python predict.py --asset mnq [--out resultados.csv]
"""
import sys, json, warnings
warnings.filterwarnings('ignore')

import argparse
import sqlite3
import pickle
import pandas as pd
import numpy as np
import ta
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score

ASSET_CONFIG = {
    'mnq': {'db': 'data.db',         'syms': ['mnq', 'btc', 'cl']},
    'btc': {'db': 'btc/data.db',     'syms': ['btc', 'mnq', 'cl']},
    'cl':  {'db': 'cl/data.db',      'syms': ['cl',  'mnq', 'btc']},
    'mgc': {'db': 'mgc/data.db',     'syms': ['mgc', 'mnq', 'btc']},
}
MODEL_DIR = Path(__file__).parent


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
    f['prev_day_range_pct'] = pct(daily['high'].shift(1), daily['low'].shift(1)).reindex(idx, method='ffill')

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


def build_features(conn, interval='1h', forward=4, min_ret=0.001, syms=None):
    p = syms[0]
    s1 = syms[1]
    s2 = syms[2]

    pri = compute(load_symbol(conn, p, interval))
    sec1 = compute(load_symbol(conn, s1, interval))
    sec2 = compute(load_symbol(conn, s2, interval))

    idx = pri.index
    f = pd.DataFrame(index=idx)

    f['rsi_p'] = pri['rsi']
    f['rsi_1'] = sec1['rsi'].reindex(idx, method='ffill')
    f['rsi_2'] = sec2['rsi'].reindex(idx, method='ffill')

    f['div_1']  = f['rsi_p'] - f['rsi_1']
    f['div_2']  = f['rsi_p'] - f['rsi_2']
    f['rsi_spread_1_2'] = f['rsi_1'] - f['rsi_2']
    f['rsi_abs_p_1']    = (f['rsi_p'] - f['rsi_1']).abs()
    f['rsi_abs_p_2']    = (f['rsi_p'] - f['rsi_2']).abs()
    f['rsi_abs_1_2']    = (f['rsi_1'] - f['rsi_2']).abs()

    f['adx_p'] = pri['adx']
    f['adx_1'] = sec1['adx'].reindex(idx, method='ffill')
    f['adx_2'] = sec2['adx'].reindex(idx, method='ffill')
    f['pdi_p'] = pri['pdi']
    f['mdi_p'] = pri['mdi']

    f['adx_spread_p_1'] = f['adx_p'] - f['adx_1']
    f['adx_spread_p_2'] = f['adx_p'] - f['adx_2']
    f['adx_spread_1_2'] = f['adx_1'] - f['adx_2']
    f['adx_abs_p_1']    = (f['adx_p'] - f['adx_1']).abs()
    f['adx_abs_p_2']    = (f['adx_p'] - f['adx_2']).abs()
    f['adx_abs_1_2']    = (f['adx_1'] - f['adx_2']).abs()

    f['di_spread_p'] = f['pdi_p'] - f['mdi_p']
    pdi_1 = sec1['pdi'].reindex(idx, method='ffill')
    mdi_1 = sec1['mdi'].reindex(idx, method='ffill')
    pdi_2 = sec2['pdi'].reindex(idx, method='ffill')
    mdi_2 = sec2['mdi'].reindex(idx, method='ffill')
    f['di_spread_1'] = pdi_1 - mdi_1
    f['di_spread_2'] = pdi_2 - mdi_2

    f['dadx_p'] = f['adx_p'].diff(2)

    f['ret1_p'] = pri['ret1']
    f['ret4_p'] = pri['ret4']
    f['ret8_p'] = pri['ret8']
    f['vol_p']  = pri['vol']
    f['bb_p']   = pri['bb_w']

    f['ret1_1'] = sec1['ret1'].reindex(idx, method='ffill')
    f['ret4_1'] = sec1['ret4'].reindex(idx, method='ffill')
    f['ret1_2'] = sec2['ret1'].reindex(idx, method='ffill')
    f['ret4_2'] = sec2['ret4'].reindex(idx, method='ffill')

    ret_1 = sec1['ret1'].reindex(idx, method='ffill')
    ret_2 = sec2['ret1'].reindex(idx, method='ffill')
    ret4_1 = sec1['ret4'].reindex(idx, method='ffill')
    ret4_2 = sec2['ret4'].reindex(idx, method='ffill')

    f['ret1_spread_p_1'] = f['ret1_p'] - ret_1
    f['ret1_spread_p_2'] = f['ret1_p'] - ret_2
    f['ret1_spread_1_2'] = ret_1 - ret_2
    f['ret4_spread_p_1'] = f['ret4_p'] - ret4_1
    f['ret4_spread_p_2'] = f['ret4_p'] - ret4_2
    f['ret4_spread_1_2'] = ret4_1 - ret4_2

    f['ret1_prod_p_1'] = f['ret1_p'] * ret_1
    f['ret1_prod_1_2'] = ret_1 * ret_2
    f['price_div_p_2'] = f['ret1_p'] * ret_2
    f['price_div_abs'] = f['price_div_p_2'].abs()
    f['ret4_prod_p_1'] = f['ret4_p'] * ret4_1
    f['ret4_prod_p_2'] = f['ret4_p'] * ret4_2
    f['ret4_prod_1_2'] = ret4_1 * ret4_2

    vol_1 = sec1['vol'].reindex(idx, method='ffill')
    vol_2 = sec2['vol'].reindex(idx, method='ffill')
    f['vol_spread_p_1'] = f['vol_p'] - vol_1
    f['vol_spread_p_2'] = f['vol_p'] - vol_2
    f['vol_spread_1_2'] = vol_1 - vol_2

    bb_1 = sec1['bb_w'].reindex(idx, method='ffill')
    bb_2 = sec2['bb_w'].reindex(idx, method='ffill')
    f['bb_spread_p_1'] = f['bb_p'] - bb_1
    f['bb_spread_p_2'] = f['bb_p'] - bb_2
    f['bb_spread_1_2'] = bb_1 - bb_2

    f['dist_sma50_p'] = pri['dist_sma50']
    f['dist_sma50_1'] = sec1['dist_sma50'].reindex(idx, method='ffill')
    f['dist_sma50_2'] = sec2['dist_sma50'].reindex(idx, method='ffill')
    f['sma50_slope_p']= pri['sma50_slope']
    f['above_sma50_p']= pri['above_sma50']
    f['above_sma50_1']= sec1['above_sma50'].reindex(idx, method='ffill')
    f['above_sma50_2']= sec2['above_sma50'].reindex(idx, method='ffill')
    f['sma50_alignment']= f['above_sma50_p'] + f['above_sma50_1'] + f['above_sma50_2']

    f['dist_ema20_p'] = pri['dist_ema20']
    f['dist_ema20_1'] = sec1['dist_ema20'].reindex(idx, method='ffill')
    f['dist_ema20_2'] = sec2['dist_ema20'].reindex(idx, method='ffill')
    f['above_ema20_p']= pri['above_ema20']
    f['above_ema20_1']= sec1['above_ema20'].reindex(idx, method='ffill')
    f['above_ema20_2']= sec2['above_ema20'].reindex(idx, method='ffill')
    f['ema20_bias_p_1'] = f['above_ema20_p'] + f['above_ema20_1']
    f['ema20_alignment'] = f['above_ema20_p'] + f['above_ema20_1'] + f['above_ema20_2']

    m50_1 = sec1['dist_sma50'].reindex(idx, method='ffill')
    m50_2 = sec2['dist_sma50'].reindex(idx, method='ffill')
    f['sma50_dist_spread_p_1'] = f['dist_sma50_p'] - m50_1
    f['sma50_dist_spread_p_2'] = f['dist_sma50_p'] - m50_2
    f['sma50_dist_spread_1_2'] = m50_1 - m50_2

    e50_1 = sec1['dist_ema20'].reindex(idx, method='ffill')
    e50_2 = sec2['dist_ema20'].reindex(idx, method='ffill')
    f['ema20_dist_spread_p_1'] = f['dist_ema20_p'] - e50_1
    f['ema20_dist_spread_p_2'] = f['dist_ema20_p'] - e50_2
    f['ema20_dist_spread_1_2'] = e50_1 - e50_2

    ab50_1 = sec1['above_sma50'].reindex(idx, method='ffill')
    ab50_2 = sec2['above_sma50'].reindex(idx, method='ffill')
    f['sma50_align_p_1'] = f['above_sma50_p'] + ab50_1
    f['sma50_align_p_2'] = f['above_sma50_p'] + ab50_2
    f['sma50_align_1_2'] = ab50_1 + ab50_2

    eb50_1 = sec1['above_ema20'].reindex(idx, method='ffill')
    eb50_2 = sec2['above_ema20'].reindex(idx, method='ffill')
    f['ema20_align_p_1'] = f['above_ema20_p'] + eb50_1
    f['ema20_align_p_2'] = f['above_ema20_p'] + eb50_2
    f['ema20_align_1_2'] = eb50_1 + eb50_2

    f['hour'] = idx.hour
    f['dow']  = idx.dayofweek
    f['hour_sin'] = np.sin(2 * np.pi * f['hour'] / 24)
    f['hour_cos'] = np.cos(2 * np.pi * f['hour'] / 24)
    f['dow_sin']  = np.sin(2 * np.pi * f['dow'] / 7)
    f['dow_cos']  = np.cos(2 * np.pi * f['dow'] / 7)

    add_key_levels(pri, f)

    future_price = pri['close'].shift(-forward)
    f['future_ret'] = future_price / pri['close'] - 1
    f['label']      = 1
    f.loc[f['future_ret'] >  min_ret, 'label'] = 2
    f.loc[f['future_ret'] < -min_ret, 'label'] = 0

    return f.dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', type=str, default='mnq', choices=list(ASSET_CONFIG.keys()))
    ap.add_argument('--model', type=str, default='', help='Caminho do modelo .pkl (opcional)')
    ap.add_argument('--out',   type=str, default='',   help='CSV de saida (opcional)')
    args = ap.parse_args()

    cfg = ASSET_CONFIG[args.asset]
    model_path = Path(args.model) if args.model else MODEL_DIR / f'propfirm_model_{args.asset}.pkl'
    db_path = MODEL_DIR.parent / cfg['db']

    if not model_path.exists():
        print(f'Modelo nao encontrado: {model_path}')
        return

    model_data = pickle.load(open(model_path, 'rb'))
    model      = model_data['model']
    feat_cols  = model_data['features']
    fwd        = model_data.get('forward', 4)

    print(f'Ativo: {args.asset.upper()}')
    print(f'Modelo: {model_path.name}  |  Forward: {fwd}h  |  Features: {len(feat_cols)}')

    conn = sqlite3.connect(db_path)
    df = build_features(conn, forward=fwd, syms=cfg['syms'])
    conn.close()

    X = df[feat_cols]
    proba = model.predict_proba(X)
    preds = model.predict(X)
    labels = df['label'].values

    print(f'\nAmostras: {len(df)}  |  Periodo: {df.index[0].date()} ate {df.index[-1].date()}')
    print(classification_report(labels, preds, target_names=['SHORT', 'NEUTRO', 'LONG']))

    auc_ = roc_auc_score(labels, proba, multi_class='ovr')
    acc_ = (preds == labels).mean()
    print(f'ROC-AUC: {auc_:.4f}  |  Acurácia: {acc_:.2%}')

    if args.out:
        df_out = df[['hour', 'dow', 'label']].copy()
        df_out['pred'] = preds
        df_out['prob_short'] = proba[:, 0]
        df_out['prob_neutro'] = proba[:, 1]
        df_out['prob_long'] = proba[:, 2]
        df_out['correct'] = (preds == labels).astype(int)
        df_out.to_csv(Path(args.out))
        print(f'Resultados salvos em: {args.out}')

    return {'auc': auc_, 'acc': acc_, 'samples': len(df)}


if __name__ == '__main__':
    main()

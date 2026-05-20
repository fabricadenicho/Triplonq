"""
Treina XGBoost sem vies de trade, usando dados ja coletados do Yahoo Finance.
Carrega dados do DB existente — nao baixa nada novo.

Uso: python train.py [--asset mnq] [--forward 4] [--full]
"""
import argparse
import sqlite3
import pickle
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path

ASSET_CONFIG = {
    'mnq': {'db': 'data.db',         'syms': ['mnq', 'btc', 'cl']},
    'btc': {'db': 'btc/data.db',     'syms': ['btc', 'mnq', 'cl']},
    'cl':  {'db': 'cl/data.db',      'syms': ['cl',  'mnq', 'btc']},
    'mgc': {'db': 'mgc/data.db',     'syms': ['mgc', 'mnq', 'btc']},
    'es':  {'db': 'es/data.db',      'syms': ['es',  'mnq', 'btc']},
}


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


def build_features(conn, interval='1h', forward=8, min_ret=0.001, syms=None):
    p = syms[0]    # primary
    s1 = syms[1]   # secondary 1
    s2 = syms[2]   # secondary 2

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


FEATURE_COLS = [
    'rsi_p', 'rsi_1', 'rsi_2',
    'div_1', 'div_2',
    'rsi_spread_1_2',
    'rsi_abs_p_1', 'rsi_abs_p_2', 'rsi_abs_1_2',
    'adx_p', 'adx_1', 'adx_2',
    'pdi_p', 'mdi_p',
    'adx_spread_p_1', 'adx_spread_p_2', 'adx_spread_1_2',
    'adx_abs_p_1', 'adx_abs_p_2', 'adx_abs_1_2',
    'di_spread_p', 'di_spread_1', 'di_spread_2',
    'dadx_p',
    'ret1_p', 'ret4_p', 'ret8_p', 'vol_p', 'bb_p',
    'ret1_1', 'ret4_1',
    'ret1_2', 'ret4_2',
    'ret1_spread_p_1', 'ret1_spread_p_2', 'ret1_spread_1_2',
    'ret4_spread_p_1', 'ret4_spread_p_2', 'ret4_spread_1_2',
    'ret1_prod_p_1', 'ret1_prod_1_2',
    'price_div_p_2', 'price_div_abs',
    'ret4_prod_p_1', 'ret4_prod_p_2', 'ret4_prod_1_2',
    'vol_spread_p_1', 'vol_spread_p_2', 'vol_spread_1_2',
    'bb_spread_p_1', 'bb_spread_p_2', 'bb_spread_1_2',
    'dist_sma50_p', 'dist_sma50_1', 'dist_sma50_2',
    'sma50_slope_p',
    'above_sma50_p', 'above_sma50_1', 'above_sma50_2',
    'sma50_alignment',
    'sma50_dist_spread_p_1', 'sma50_dist_spread_p_2', 'sma50_dist_spread_1_2',
    'sma50_align_p_1', 'sma50_align_p_2', 'sma50_align_1_2',
    'dist_ema20_p', 'dist_ema20_1', 'dist_ema20_2',
    'above_ema20_p', 'above_ema20_1', 'above_ema20_2',
    'ema20_bias_p_1', 'ema20_alignment',
    'ema20_dist_spread_p_1', 'ema20_dist_spread_p_2', 'ema20_dist_spread_1_2',
    'ema20_align_p_1', 'ema20_align_p_2', 'ema20_align_1_2',
    'hour', 'dow', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
] + KEY_LEVEL_FEATURES


OPTIMIZED_FEATURES = [
    'rsi_p', 'rsi_1', 'rsi_2',
    'div_1', 'div_2',
    'adx_p', 'adx_1', 'adx_2',
    'di_spread_p', 'di_spread_1', 'di_spread_2',
    'ret1_p', 'ret4_p', 'ret8_p', 'vol_p', 'bb_p',
    'ret1_1', 'ret4_1', 'ret1_2', 'ret4_2',
    'ret1_spread_p_1', 'ret1_spread_p_2',
    'ret1_prod_p_1', 'ret1_prod_1_2',
    'price_div_p_2', 'price_div_abs',
    'vol_spread_p_1', 'vol_spread_p_2',
    'bb_spread_p_1', 'bb_spread_p_2',
    'dist_sma50_p', 'sma50_slope_p',
    'above_sma50_p', 'sma50_alignment',
    'dist_ema20_p', 'above_ema20_p', 'ema20_bias_p_1', 'ema20_alignment',
    'hour', 'dow', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
] + KEY_LEVEL_FEATURES


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
    tr_start = str(X_tr.index[0].date())
    tr_end   = str(X_tr.index[-1].date())
    te_start = str(X_te.index[0].date())
    te_end   = str(X_te.index[-1].date())
    print(f'  Treino : {len(X_tr)} amostras  ({tr_start} ate {tr_end})')
    print(f'  Teste  : {len(X_te)} amostras  ({te_start} ate {te_end})')
    dist = y_te.value_counts(normalize=True).sort_index()
    acc = (preds == y_te).mean()
    print(f'  Acurácia: {acc:.2%}')
    print(f'  Distribuicao real: SHORT={dist.get(0,0):.1%}  NEUTRO={dist.get(1,0):.1%}  LONG={dist.get(2,0):.1%}\n')
    print(classification_report(y_te, preds, target_names=['SHORT', 'NEUTRO', 'LONG']))

    auc = roc_auc_score(y_te, proba, multi_class='ovr')
    print(f'  ROC-AUC (macro ovr): {auc:.4f}')

    metadata = {
        'train_start': tr_start, 'train_end': tr_end,
        'test_start': te_start, 'test_end': te_end,
        'train_samples': len(X_tr), 'test_samples': len(X_te),
    }
    return model, auc, acc, metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset',     type=str,   default='mnq', choices=list(ASSET_CONFIG.keys()),
                   help='Ativo principal (mnq|btc|cl|mgc)')
    p.add_argument('--forward',   type=int,   default=4,     help='Barras a frente (default 4)')
    p.add_argument('--interval',  type=str,   default='1h',  help='1h ou 1d (default 1h)')
    p.add_argument('--full',      action='store_true',  help='Usar feature set completo')
    p.add_argument('--max-months', type=int, default=0,  help='Usar apenas ultimos N meses (0=todos)')
    args = p.parse_args()

    cfg = ASSET_CONFIG[args.asset]
    db_path = Path(__file__).parent.parent / cfg['db']
    model_out = Path(__file__).parent / f'propfirm_model_{args.asset}.pkl'

    if not db_path.exists():
        print(f'DB nao encontrado: {db_path}')
        return

    print(f'Ativo: {args.asset.upper()}')
    print(f'DB: {db_path}')
    print(f'Config: interval={args.interval}  forward={args.forward}h')

    conn = sqlite3.connect(db_path)
    print('Calculando features (sem vies de trade)...')
    df = build_features(conn, args.interval, args.forward, syms=cfg['syms'])
    conn.close()

    if args.max_months > 0:
        cutoff = df.index.max() - pd.DateOffset(months=args.max_months)
        total_antes = len(df)
        df = df[df.index >= cutoff].copy()
        print(f'Filtro ultimos {args.max_months} meses: {total_antes} -> {len(df)} amostras')

    dist = df['label'].value_counts(normalize=True).sort_index()
    print(f'Dataset: {len(df)} amostras  |  SHORT={dist.get(0,0):.1%}  NEUTRO={dist.get(1,0):.1%}  LONG={dist.get(2,0):.1%}')

    if len(df) < 300:
        print('Poucos dados. Rode collect_data.py primeiro.')
        return

    feat_cols = FEATURE_COLS if args.full else OPTIMIZED_FEATURES
    X, y = df[feat_cols], df['label']
    print(f'Features: {len(feat_cols)}  ({"completo" if args.full else "otimizado (default)"})')
    model, auc, acc, metadata = train_model(X, y)

    with open(model_out, 'wb') as f:
        pickle.dump({
            'model':    model,
            'features': feat_cols,
            'forward':  args.forward,
            'interval': args.interval,
            'auc':      auc,
            'acc':      acc,
            'metadata': metadata,
        }, f)
    print(f'  Modelo salvo em: {model_out}')


if __name__ == '__main__':
    main()

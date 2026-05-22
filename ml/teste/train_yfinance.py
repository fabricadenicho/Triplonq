"""
Treina PropFirm models com yfinance (auto_adjust=True — rollover ajustado).

Problemas do SQLite corrigidos:
  1. Labels contaminadas por rollovers (ret_fwd de +200x)
  2. Features ret1/ret4/ret8 com spikes de rollover
  3. BTC no SQLite so a partir de 2024-05-19 — modelo mnq tinha apenas ~12 meses

Metodologia:
  - 720 dias de dados yfinance ajustados
  - Filtro extra: remove barras onde |ret1_primary| > 3% (seguranca residual)
  - Walk-forward por ano: 2024 (treino) -> 2025 (OOS cego) -> 2026 (live)
  - Mesmas features do predict_live.py (compatibilidade garantida)

Uso:
  python train_yfinance.py --asset mnq
  python train_yfinance.py --asset cl
  python train_yfinance.py --all       (todos os 5 ativos)
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
import pickle
import time
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score

BASE     = Path(__file__).parent
PERIOD   = 720
ROLLOVER = 0.03   # |ret1| > 3% = barra de rollover residual, remover

TICKERS = {
    'mnq': 'MNQ=F',
    'btc': 'BTC-USD',
    'cl':  'CL=F',
    'es':  'ES=F',
}

# Secundarios por ativo (mesma logica do predict_live.py)
SEC = {
    'mnq': ('btc', 'cl'),
    'btc': ('mnq', 'cl'),
    'cl':  ('mnq', 'btc'),
    'es':  ('mnq', 'btc'),
}

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


def download_asset(ticker, retries=3):
    from datetime import datetime, timedelta
    end   = datetime.now()
    start = end - timedelta(days=PERIOD)
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval='1h',
                             auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = df.columns.str.lower()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])
                return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    return None


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


def build_features(pri_raw, sec1_raw, sec2_raw, forward=8, min_ret=0.001):
    pri  = compute(pri_raw)
    sec1 = compute(sec1_raw)
    sec2 = compute(sec2_raw)

    idx  = pri.index
    sec1 = sec1.reindex(idx, method='ffill')
    sec2 = sec2.reindex(idx, method='ffill')

    f = pd.DataFrame(index=idx)

    f['rsi_p'] = pri['rsi']
    f['rsi_1'] = sec1['rsi']
    f['rsi_2'] = sec2['rsi']
    f['div_1'] = f['rsi_p'] - f['rsi_1']
    f['div_2'] = f['rsi_p'] - f['rsi_2']
    f['rsi_spread_1_2'] = f['rsi_1'] - f['rsi_2']
    f['rsi_abs_p_1'] = (f['rsi_p'] - f['rsi_1']).abs()
    f['rsi_abs_p_2'] = (f['rsi_p'] - f['rsi_2']).abs()
    f['rsi_abs_1_2'] = (f['rsi_1'] - f['rsi_2']).abs()

    f['adx_p'] = pri['adx']
    f['adx_1'] = sec1['adx']
    f['adx_2'] = sec2['adx']
    f['pdi_p'] = pri['pdi']
    f['mdi_p'] = pri['mdi']
    f['adx_spread_p_1'] = f['adx_p'] - f['adx_1']
    f['adx_spread_p_2'] = f['adx_p'] - f['adx_2']
    f['adx_spread_1_2'] = f['adx_1'] - f['adx_2']
    f['adx_abs_p_1'] = (f['adx_p'] - f['adx_1']).abs()
    f['adx_abs_p_2'] = (f['adx_p'] - f['adx_2']).abs()
    f['adx_abs_1_2'] = (f['adx_1'] - f['adx_2']).abs()

    f['di_spread_p'] = f['pdi_p'] - f['mdi_p']
    f['di_spread_1'] = sec1['pdi'] - sec1['mdi']
    f['di_spread_2'] = sec2['pdi'] - sec2['mdi']
    f['dadx_p'] = f['adx_p'].diff(2)

    f['ret1_p'] = pri['ret1']
    f['ret4_p'] = pri['ret4']
    f['ret8_p'] = pri['ret8']
    f['vol_p']  = pri['vol']
    f['bb_p']   = pri['bb_w']
    f['ret1_1'] = sec1['ret1']
    f['ret4_1'] = sec1['ret4']
    f['ret1_2'] = sec2['ret1']
    f['ret4_2'] = sec2['ret4']

    f['ret1_spread_p_1'] = f['ret1_p'] - sec1['ret1']
    f['ret1_spread_p_2'] = f['ret1_p'] - sec2['ret1']
    f['ret1_spread_1_2'] = sec1['ret1'] - sec2['ret1']
    f['ret4_spread_p_1'] = f['ret4_p'] - sec1['ret4']
    f['ret4_spread_p_2'] = f['ret4_p'] - sec2['ret4']
    f['ret4_spread_1_2'] = sec1['ret4'] - sec2['ret4']

    f['ret1_prod_p_1'] = f['ret1_p'] * sec1['ret1']
    f['ret1_prod_1_2'] = sec1['ret1'] * sec2['ret1']
    f['price_div_p_2'] = f['ret1_p'] * sec2['ret1']
    f['price_div_abs'] = f['price_div_p_2'].abs()
    f['ret4_prod_p_1'] = f['ret4_p'] * sec1['ret4']
    f['ret4_prod_p_2'] = f['ret4_p'] * sec2['ret4']
    f['ret4_prod_1_2'] = sec1['ret4'] * sec2['ret4']

    f['vol_spread_p_1'] = f['vol_p'] - sec1['vol']
    f['vol_spread_p_2'] = f['vol_p'] - sec2['vol']
    f['vol_spread_1_2'] = sec1['vol'] - sec2['vol']
    f['bb_spread_p_1']  = f['bb_p'] - sec1['bb_w']
    f['bb_spread_p_2']  = f['bb_p'] - sec2['bb_w']
    f['bb_spread_1_2']  = sec1['bb_w'] - sec2['bb_w']

    f['dist_sma50_p'] = pri['dist_sma50']
    f['dist_sma50_1'] = sec1['dist_sma50']
    f['dist_sma50_2'] = sec2['dist_sma50']
    f['sma50_slope_p']= pri['sma50_slope']
    f['above_sma50_p']= pri['above_sma50']
    f['above_sma50_1']= sec1['above_sma50']
    f['above_sma50_2']= sec2['above_sma50']
    f['sma50_alignment'] = f['above_sma50_p'] + f['above_sma50_1'] + f['above_sma50_2']
    f['sma50_dist_spread_p_1'] = f['dist_sma50_p'] - sec1['dist_sma50']
    f['sma50_dist_spread_p_2'] = f['dist_sma50_p'] - sec2['dist_sma50']
    f['sma50_dist_spread_1_2'] = sec1['dist_sma50'] - sec2['dist_sma50']
    f['sma50_align_p_1'] = f['above_sma50_p'] + f['above_sma50_1']
    f['sma50_align_p_2'] = f['above_sma50_p'] + f['above_sma50_2']
    f['sma50_align_1_2'] = f['above_sma50_1'] + f['above_sma50_2']

    f['dist_ema20_p'] = pri['dist_ema20']
    f['dist_ema20_1'] = sec1['dist_ema20']
    f['dist_ema20_2'] = sec2['dist_ema20']
    f['above_ema20_p']= pri['above_ema20']
    f['above_ema20_1']= sec1['above_ema20']
    f['above_ema20_2']= sec2['above_ema20']
    f['ema20_bias_p_1']  = f['above_ema20_p'] + f['above_ema20_1']
    f['ema20_alignment'] = f['above_ema20_p'] + f['above_ema20_1'] + f['above_ema20_2']
    f['ema20_dist_spread_p_1'] = f['dist_ema20_p'] - sec1['dist_ema20']
    f['ema20_dist_spread_p_2'] = f['dist_ema20_p'] - sec2['dist_ema20']
    f['ema20_dist_spread_1_2'] = sec1['dist_ema20'] - sec2['dist_ema20']
    f['ema20_align_p_1'] = f['above_ema20_p'] + f['above_ema20_1']
    f['ema20_align_p_2'] = f['above_ema20_p'] + f['above_ema20_2']
    f['ema20_align_1_2'] = f['above_ema20_1'] + f['above_ema20_2']

    f['hour'] = idx.hour
    f['dow']  = idx.dayofweek
    f['hour_sin'] = np.sin(2 * np.pi * f['hour'] / 24)
    f['hour_cos'] = np.cos(2 * np.pi * f['hour'] / 24)
    f['dow_sin']  = np.sin(2 * np.pi * f['dow'] / 7)
    f['dow_cos']  = np.cos(2 * np.pi * f['dow'] / 7)

    add_key_levels(pri, f)

    # Label (forward return)
    future_price   = pri['close'].shift(-forward)
    f['future_ret'] = future_price / pri['close'] - 1
    f['label']      = 1
    f.loc[f['future_ret'] >  min_ret, 'label'] = 2  # LONG
    f.loc[f['future_ret'] < -min_ret, 'label'] = 0  # SHORT

    f = f.dropna()

    # Filtro de rollover residual: remove barras onde |ret1_primary| > 3%
    n_antes = len(f)
    f = f[f['ret1_p'].abs() <= ROLLOVER].copy()
    n_filtradas = n_antes - len(f)
    if n_filtradas > 0:
        print(f'  Filtro rollover: {n_filtradas} barras removidas (|ret1| > {ROLLOVER:.0%})')

    return f


def train_walk_forward(X, y, label=''):
    """
    Walk-forward por ano com blind test genuino:
      - 2024 = treino  (80% treino real + 20% validacao interna para early stopping)
      - 2025 = BLIND OOS — zero contato com o modelo durante o treino
      - 2026 = live test — também cego

    Early stopping usa validacao INTERNA do treino, nunca o OOS.
    """
    years = X.index.year.unique()

    # Tenta split por ano
    if 2024 in years and 2025 in years:
        mask_tr = X.index.year == 2024
        mask_os = X.index.year == 2025
        mask_lv = X.index.year == 2026 if 2026 in years else pd.Series(False, index=X.index)

        X_tr = X[mask_tr]; y_tr = y[mask_tr]
        X_os = X[mask_os]; y_os = y[mask_os]
        X_lv = X[mask_lv]; y_lv = y[mask_lv]

        if len(X_tr) >= 200 and len(X_os) >= 50:
            mode = 'year'
        else:
            mode = 'split70'
    else:
        mode = 'split70'

    if mode == 'split70':
        split = int(len(X) * 0.70)
        X_tr, X_os = X.iloc[:split], X.iloc[split:]
        y_tr, y_os = y.iloc[:split], y.iloc[split:]
        X_lv = pd.DataFrame(); y_lv = pd.Series()
        print(f'  Modo: split 70/30 (dados insuficientes para split por ano)')
    else:
        print(f'  Modo: walk-forward por ano (blind OOS genuino)')

    # Validacao interna: 80% treino real + 20% early stopping
    # O OOS (2025/2026) nunca e visto durante o treino
    val_split = int(len(X_tr) * 0.80)
    X_fit, X_val = X_tr.iloc[:val_split], X_tr.iloc[val_split:]
    y_fit, y_val = y_tr.iloc[:val_split], y_tr.iloc[val_split:]

    counts = y_fit.value_counts().sort_index()
    weights = {c: len(y_fit) / (3 * count) for c, count in counts.items()}
    sample_weight = y_fit.map(weights)

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
    # early stopping usa validacao INTERNA — OOS nunca e tocado aqui
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
              sample_weight=sample_weight.values, verbose=False)

    results = {}

    print(f'\n  --- {label} ---')
    print(f'  Treino real  : {len(X_fit)} amostras  '
          f'{X_fit.index[0].date()} -> {X_fit.index[-1].date()}')
    print(f'  Validacao ES : {len(X_val)} amostras  '
          f'{X_val.index[0].date()} -> {X_val.index[-1].date()}  (early stopping interno)')
    print(f'  OOS cego     : {len(X_os)} amostras  '
          f'{X_os.index[0].date()} -> {X_os.index[-1].date()}  (nunca visto pelo modelo)')

    for split_name, X_t, y_t in [('BLIND OOS', X_os, y_os), ('LIVE 2026', X_lv, y_lv)]:
        if len(X_t) < 10:
            continue
        preds = model.predict(X_t)
        proba = model.predict_proba(X_t)
        auc   = roc_auc_score(y_t, proba, multi_class='ovr')
        acc   = (preds == y_t).mean()
        dist  = y_t.value_counts(normalize=True).sort_index()
        baseline_acc = dist.max()  # naive majority

        print(f'\n  [{split_name}] {X_t.index[0].date()} -> {X_t.index[-1].date()}')
        print(f'  Amostras: {len(X_t)} | AUC: {auc:.4f} | Acuracia: {acc:.2%} | Baseline: {baseline_acc:.2%}')
        print(f'  Distrib: SHORT={dist.get(0,0):.1%} NEUTRO={dist.get(1,0):.1%} LONG={dist.get(2,0):.1%}')
        print(classification_report(y_t, preds, target_names=['SHORT','NEUTRO','LONG'],
                                    zero_division=0))

        results[split_name] = {'auc': auc, 'acc': acc, 'baseline': baseline_acc,
                               'n': len(X_t)}

    # AUC de referencia = OOS
    main_auc = results.get('BLIND OOS', results.get('LIVE 2026', {})).get('auc', 0.0)
    return model, main_auc, results


def train_asset(asset, forward=8, min_ret=0.001, full=False):
    print(f'\n{"="*60}')
    print(f'  ATIVO: {asset.upper()}  |  forward={forward}h  |  min_ret={min_ret:.2%}')
    print(f'{"="*60}')

    sec1_name, sec2_name = SEC[asset]
    needed = {asset, sec1_name, sec2_name}

    print('  Baixando dados yfinance...')
    raw = {}
    for sym in needed:
        ticker = TICKERS[sym]
        print(f'    {ticker}...', end=' ', flush=True)
        df = download_asset(ticker)
        if df is None or len(df) < 200:
            print(f'FALHOU')
            return None
        print(f'{len(df)} candles  {df.index[0].date()} -> {df.index[-1].date()}')
        raw[sym] = df

    # Alinhar indices
    idx = raw[asset].index
    for sym in [sec1_name, sec2_name]:
        idx = idx.intersection(raw[sym].index)
    for sym in needed:
        raw[sym] = raw[sym].loc[idx]

    print(f'  Indices alinhados: {len(idx)} barras comuns')

    df = build_features(raw[asset], raw[sec1_name], raw[sec2_name],
                        forward=forward, min_ret=min_ret)

    if len(df) < 300:
        print(f'  ERRO: poucos dados apos dropna ({len(df)} amostras). Abortando.')
        return None

    dist = df['label'].value_counts(normalize=True).sort_index()
    print(f'  Dataset limpo: {len(df)} amostras')
    print(f'  Labels: SHORT={dist.get(0,0):.1%} NEUTRO={dist.get(1,0):.1%} LONG={dist.get(2,0):.1%}')

    feat_cols = OPTIMIZED_FEATURES
    available = [c for c in feat_cols if c in df.columns]
    if len(available) < len(feat_cols):
        missing = set(feat_cols) - set(available)
        print(f'  AVISO: {len(missing)} features ausentes: {missing}')

    X, y = df[available], df['label']

    model, auc, results = train_walk_forward(X, y, label=asset.upper())

    model_path = BASE / f'propfirm_model_{asset}.pkl'
    with open(model_path, 'wb') as fp:
        pickle.dump({
            'model':    model,
            'features': available,
            'forward':  forward,
            'interval': '1h',
            'auc':      auc,
            'source':   'yfinance',
            'tickers':  {k: TICKERS[k] for k in needed},
            'results':  results,
        }, fp)
    print(f'\n  Modelo salvo: {model_path}  (AUC OOS={auc:.4f})')
    return auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset',   type=str, default='mnq',
                   choices=['mnq','btc','cl','es'])
    p.add_argument('--all',     action='store_true', help='Treinar todos os 5 ativos')
    p.add_argument('--forward', type=int, default=8, help='Horas a frente (default 8)')
    p.add_argument('--min-ret', type=float, default=0.001)
    args = p.parse_args()

    assets = ['mnq', 'btc', 'cl', 'es'] if args.all else [args.asset]

    resultados = {}
    for asset in assets:
        auc = train_asset(asset, forward=args.forward, min_ret=args.min_ret)
        resultados[asset] = auc

    if len(assets) > 1:
        print(f'\n{"="*60}')
        print('  RESUMO FINAL')
        print(f'{"="*60}')
        for a, auc in resultados.items():
            auc_str = f'{auc:.4f}' if auc else 'N/A'
            status  = 'OK' if auc and auc > 0.52 else 'FRACO'
            print(f'  {a.upper():<6} AUC OOS = {auc_str:<8} [{status}]')


if __name__ == '__main__':
    main()

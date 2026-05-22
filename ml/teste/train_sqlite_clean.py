"""
Treina PropFirm models usando SQLite 2021-2026 com filtro de rollover correto.

Por que SQLite ao inves de yfinance:
  - yfinance limita 720 dias de dados 1h -> so 5 meses de treino (jul-nov/2024)
  - SQLite tem 5 anos (maio/2021 -> maio/2026) -> ~3 anos de treino real

Filtro de rollover correto (mais rigoroso que o anterior):
  - Detecta barras com |ret1_primary| > 3%  (spike de rollover)
  - Para cada rollover na barra T: anula TODAS as barras em [T - forward, T + lookback]
    * T-forward .. T-1 : future_ret dessas barras "ve" o spike no futuro
    * T+1 .. T+lookback: features ret4/ret8 dessas barras "veem" o spike no passado
  - Isso garante que nem labels nem features estao contaminados

Walk-forward (genuino):
  - 2021-2024 = treino (80% treino real + 20% early stopping interno)
  - 2025       = BLIND OOS — nunca visto pelo modelo
  - 2026       = live test — tambem cego

Uso:
  python train_sqlite_clean.py --asset mnq
  python train_sqlite_clean.py --all
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

import sqlite3
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score

BASE = Path(__file__).parent

ASSET_CONFIG = {
    'mnq': {'db': 'data.db',     'syms': ['mnq', 'btc', 'cl']},
    'btc': {'db': 'btc/data.db', 'syms': ['btc', 'mnq', 'cl']},
    'cl':  {'db': 'cl/data.db',  'syms': ['cl',  'mnq', 'btc']},
    'es':  {'db': 'es/data.db',  'syms': ['es',  'mnq', 'btc']},
}

ROLLOVER_THRESH = 0.03   # |ret1| > 3% = barra de rollover (apenas futuros)
LOOKBACK_MAX    = 8      # maior janela de feature (ret8)

# Somente futuros tem rollover de contrato. BTC e crypto (spot), spikes sao reais.
FUTURES_SYMS = {'mnq', 'cl', 'es'}

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


def load_symbol(conn, symbol):
    df = pd.read_sql(
        'SELECT ts,open,high,low,close,volume FROM candles '
        'WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts',
        conn, params=(symbol,), parse_dates=['ts'], index_col='ts',
    )
    df = df[~df.index.duplicated(keep='last')]
    return df


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
    f['dist_to_pdh'] = pct(c, pdh); f['dist_to_pdl'] = pct(c, pdl)
    f['dist_to_do']  = pct(c, do_); f['above_do'] = (c > do_).astype(int)
    f['above_pdh']   = (c > pdh).astype(int); f['above_pdl'] = (c > pdl).astype(int)
    f['prev_day_range_pct'] = pct(
        daily['high'].shift(1), daily['low'].shift(1)
    ).reindex(idx, method='ffill')

    weekly = primary_df.resample('W-SUN').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pwh = weekly['high'].shift(1).reindex(idx, method='ffill')
    pwl = weekly['low'].shift(1).reindex(idx, method='ffill')
    wo  = weekly['open'].reindex(idx, method='ffill')
    f['dist_to_pwh'] = pct(c, pwh); f['dist_to_pwl'] = pct(c, pwl)
    f['dist_to_wo']  = pct(c, wo); f['above_wo'] = (c > wo).astype(int)
    f['above_pwh']   = (c > pwh).astype(int); f['above_pwl'] = (c > pwl).astype(int)

    monthly = primary_df.resample('MS').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pmh = monthly['high'].shift(1).reindex(idx, method='ffill')
    pml = monthly['low'].shift(1).reindex(idx, method='ffill')
    mo  = monthly['open'].reindex(idx, method='ffill')
    f['dist_to_pmh'] = pct(c, pmh); f['dist_to_pml'] = pct(c, pml)
    f['dist_to_mo']  = pct(c, mo); f['above_mo'] = (c > mo).astype(int)
    f['above_pmh']   = (c > pmh).astype(int); f['above_pml'] = (c > pml).astype(int)

    monday_bars = primary_df[primary_df.index.dayofweek == 0]
    if len(monday_bars) >= 4:
        mday_h = monday_bars['high'].resample('W-SUN').max().reindex(idx, method='ffill')
        mday_l = monday_bars['low'].resample('W-SUN').min().reindex(idx, method='ffill')
        f['dist_to_mday_h'] = pct(c, mday_h); f['dist_to_mday_l'] = pct(c, mday_l)
        f['above_mday_h']   = (c > mday_h).astype(int)
        f['above_mday_l']   = (c > mday_l).astype(int)
    else:
        for col in ['dist_to_mday_h', 'dist_to_mday_l', 'above_mday_h', 'above_mday_l']:
            f[col] = 0.0


def apply_rollover_filter(df, forward, ret1_col='ret1_p'):
    """
    Marca como NaN todas as linhas dentro da janela de contaminacao
    de cada barra de rollover detectada.

    Janela por rollover em T:
      [T - forward, T + LOOKBACK_MAX]
      - T-forward .. T-1 : future_ret ve o spike
      - T           : a propria barra de rollover
      - T+1 .. T+8  : features ret4/ret8 ainda veem o spike
    """
    ret1 = df[ret1_col]
    rollover_mask = ret1.abs() > ROLLOVER_THRESH
    rollover_positions = np.where(rollover_mask)[0]

    contaminated = set()
    for pos in rollover_positions:
        start = max(0, pos - forward)
        end   = min(len(df) - 1, pos + LOOKBACK_MAX)
        for i in range(start, end + 1):
            contaminated.add(i)

    if contaminated:
        df = df.copy()
        df.iloc[list(contaminated)] = np.nan

    return df, len(contaminated), len(rollover_positions)


def build_features(conn, syms, forward=8, min_ret=0.001):
    p, s1_name, s2_name = syms[0], syms[1], syms[2]

    pri  = compute(load_symbol(conn, p))
    sec1 = compute(load_symbol(conn, s1_name))
    sec2 = compute(load_symbol(conn, s2_name))

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

    f['adx_p'] = pri['adx']; f['adx_1'] = sec1['adx']; f['adx_2'] = sec2['adx']
    f['pdi_p'] = pri['pdi']; f['mdi_p'] = pri['mdi']
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

    f['ret1_p'] = pri['ret1']; f['ret4_p'] = pri['ret4']; f['ret8_p'] = pri['ret8']
    f['vol_p']  = pri['vol'];  f['bb_p']   = pri['bb_w']
    f['ret1_1'] = sec1['ret1']; f['ret4_1'] = sec1['ret4']
    f['ret1_2'] = sec2['ret1']; f['ret4_2'] = sec2['ret4']

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

    f['hour'] = idx.hour; f['dow'] = idx.dayofweek
    f['hour_sin'] = np.sin(2 * np.pi * f['hour'] / 24)
    f['hour_cos'] = np.cos(2 * np.pi * f['hour'] / 24)
    f['dow_sin']  = np.sin(2 * np.pi * f['dow'] / 7)
    f['dow_cos']  = np.cos(2 * np.pi * f['dow'] / 7)

    add_key_levels(pri, f)

    future_price   = pri['close'].shift(-forward)
    f['future_ret'] = future_price / pri['close'] - 1
    f['label']      = 1
    f.loc[f['future_ret'] >  min_ret, 'label'] = 2
    f.loc[f['future_ret'] < -min_ret, 'label'] = 0

    f = f.dropna(subset=['future_ret'])  # ainda nao dropa tudo — queremos aplicar rollover filter

    # Filtro de rollover — somente para futuros (BTC = crypto, spikes sao reais)
    def _apply_if_futures(sym, ret_col):
        if sym not in FUTURES_SYMS:
            print(f'  Rollover {sym}: IGNORADO (crypto/spot — spikes sao sinais reais)')
            return
        f_ref, n_c, n_r = apply_rollover_filter(f, forward, ret1_col=ret_col)
        # modifica f in-place via loc
        f.loc[f_ref.index, :] = f_ref
        print(f'  Rollover {sym}: {n_r} eventos -> {n_c} barras anuladas')

    if 'ret1_p' in f.columns:
        if p in FUTURES_SYMS:
            f, n_cont, n_rolls = apply_rollover_filter(f, forward, ret1_col='ret1_p')
            print(f'  Rollover primary  ({p}): {n_rolls} eventos -> {n_cont} barras anuladas')
        else:
            print(f'  Rollover primary  ({p}): IGNORADO (crypto/spot)')

    if 'ret1_1' in f.columns:
        if s1_name in FUTURES_SYMS:
            roll_s1 = f['ret1_1'].abs() > ROLLOVER_THRESH
            roll_pos_s1 = np.where(roll_s1.fillna(False).values)[0]
            cont_s1 = set()
            for pos in roll_pos_s1:
                for i in range(max(0, pos - forward), min(len(f), pos + LOOKBACK_MAX + 1)):
                    cont_s1.add(i)
            if cont_s1:
                f.iloc[list(cont_s1)] = np.nan
            print(f'  Rollover secondary1 ({s1_name}): {len(roll_pos_s1)} eventos -> {len(cont_s1)} barras anuladas')
        else:
            print(f'  Rollover secondary1 ({s1_name}): IGNORADO (crypto/spot)')

    if 'ret1_2' in f.columns:
        if s2_name in FUTURES_SYMS:
            roll_s2 = f['ret1_2'].abs() > ROLLOVER_THRESH
            roll_pos_s2 = np.where(roll_s2.fillna(False).values)[0]
            cont_s2 = set()
            for pos in roll_pos_s2:
                for i in range(max(0, pos - forward), min(len(f), pos + LOOKBACK_MAX + 1)):
                    cont_s2.add(i)
            if cont_s2:
                f.iloc[list(cont_s2)] = np.nan
            print(f'  Rollover secondary2 ({s2_name}): {len(roll_pos_s2)} eventos -> {len(cont_s2)} barras anuladas')
        else:
            print(f'  Rollover secondary2 ({s2_name}): IGNORADO (crypto/spot)')

    f = f.dropna()
    return f


def train_walk_forward(X, y, label=''):
    years = X.index.year.unique()

    # Split: tudo ate 2024 = treino, 2025 = OOS, 2026 = live
    mask_tr = X.index.year <= 2024
    mask_os = X.index.year == 2025
    mask_lv = X.index.year == 2026

    X_tr = X[mask_tr]; y_tr = y[mask_tr]
    X_os = X[mask_os]; y_os = y[mask_os]
    X_lv = X[mask_lv]; y_lv = y[mask_lv]

    if len(X_tr) < 300 or len(X_os) < 50:
        split = int(len(X) * 0.70)
        X_tr, X_os = X.iloc[:split], X.iloc[split:]
        y_tr, y_os = y.iloc[:split], y.iloc[split:]
        X_lv = pd.DataFrame(); y_lv = pd.Series()
        print('  Modo: split 70/30 (fallback)')
    else:
        print(f'  Modo: walk-forward por ano (blind OOS genuino)')

    # Validacao interna: 80% treino real + 20% early stopping
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
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
              sample_weight=sample_weight.values, verbose=False)

    print(f'\n  --- {label.upper()} ---')
    print(f'  Treino real  : {len(X_fit)} amostras  '
          f'{X_fit.index[0].date()} -> {X_fit.index[-1].date()}')
    print(f'  Validacao ES : {len(X_val)} amostras  '
          f'{X_val.index[0].date()} -> {X_val.index[-1].date()}  (early stopping interno)')
    print(f'  OOS cego     : {len(X_os)} amostras  '
          f'{X_os.index[0].date()} -> {X_os.index[-1].date()}  (nunca visto pelo modelo)')

    results = {}
    for split_name, X_t, y_t in [('BLIND OOS 2025', X_os, y_os), ('LIVE 2026', X_lv, y_lv)]:
        if len(X_t) < 10:
            continue
        preds = model.predict(X_t)
        proba = model.predict_proba(X_t)
        auc   = roc_auc_score(y_t, proba, multi_class='ovr')
        acc   = (preds == y_t).mean()
        dist  = y_t.value_counts(normalize=True).sort_index()
        baseline = dist.max()

        print(f'\n  [{split_name}] {X_t.index[0].date()} -> {X_t.index[-1].date()}')
        print(f'  N={len(X_t)} | AUC={auc:.4f} | Acuracia={acc:.2%} | Baseline={baseline:.2%}')
        print(f'  Labels: SHORT={dist.get(0,0):.1%} NEUTRO={dist.get(1,0):.1%} LONG={dist.get(2,0):.1%}')
        print(classification_report(y_t, preds, target_names=['SHORT','NEUTRO','LONG'],
                                    zero_division=0))
        results[split_name] = {'auc': auc, 'acc': acc, 'n': len(X_t)}

    main_auc = results.get('BLIND OOS 2025', {}).get('auc', 0.0)
    return model, main_auc, results


def train_asset(asset, forward=8, min_ret=0.001):
    print(f'\n{"="*60}')
    print(f'  ATIVO: {asset.upper()}  |  forward={forward}h  |  SQLite limpo')
    print(f'{"="*60}')

    cfg    = ASSET_CONFIG[asset]
    db_path = BASE / '..' / cfg['db']

    if not db_path.exists():
        print(f'  DB nao encontrado: {db_path}'); return None

    conn = sqlite3.connect(db_path)
    print('  Carregando e calculando features...')
    df = build_features(conn, cfg['syms'], forward=forward, min_ret=min_ret)
    conn.close()

    n_total = len(df)
    years = sorted(df.index.year.unique())
    print(f'\n  Dataset limpo: {n_total} amostras  |  Anos: {years[0]}-{years[-1]}')

    dist = df['label'].value_counts(normalize=True).sort_index()
    print(f'  Labels: SHORT={dist.get(0,0):.1%} NEUTRO={dist.get(1,0):.1%} LONG={dist.get(2,0):.1%}')

    if n_total < 300:
        print('  Poucos dados. Abortando.'); return None

    feat_cols = [c for c in OPTIMIZED_FEATURES if c in df.columns]
    X, y = df[feat_cols], df['label']
    print(f'  Features: {len(feat_cols)}')

    model, auc, results = train_walk_forward(X, y, label=asset)

    model_path = BASE / f'propfirm_model_{asset}.pkl'
    with open(model_path, 'wb') as fp:
        pickle.dump({
            'model':    model,
            'features': feat_cols,
            'forward':  forward,
            'interval': '1h',
            'auc':      auc,
            'source':   'sqlite_clean',
            'results':  results,
        }, fp)
    print(f'\n  Modelo salvo: {model_path}  (AUC OOS={auc:.4f})')
    return auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset',    type=str, default='mnq',
                   choices=['mnq','btc','cl','es'])
    p.add_argument('--all',      action='store_true')
    p.add_argument('--forward',  type=int, default=8)
    p.add_argument('--min-ret',  type=float, default=0.001)
    args = p.parse_args()

    assets = ['mnq','btc','cl','es'] if args.all else [args.asset]

    resultados = {}
    for asset in assets:
        auc = train_asset(asset, forward=args.forward, min_ret=args.min_ret)
        resultados[asset] = auc

    if len(assets) > 1:
        print(f'\n{"="*60}')
        print('  RESUMO FINAL — SQLite limpo vs yfinance (anterior)')
        print(f'{"="*60}')
        prev = {'mnq': 0.559, 'btc': 0.529, 'cl': 0.557, 'es': 0.575}
        for a, auc in resultados.items():
            auc_str  = f'{auc:.4f}' if auc else 'N/A'
            prev_str = f'{prev.get(a,0):.4f}'
            delta    = f'{(auc - prev[a]):+.4f}' if auc else 'N/A'
            status   = 'OK' if auc and auc > 0.54 else 'FRACO'
            print(f'  {a.upper():<6} AUC={auc_str}  (anterior={prev_str}  delta={delta})  [{status}]')


if __name__ == '__main__':
    main()

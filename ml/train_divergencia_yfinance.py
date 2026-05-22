"""
Retreina model_divergencia.pkl com yfinance (auto_adjust=True).
Target: MNQ sobe > 0.1% nas proximas N horas.

Corrige problemas do SQLite:
  - Labels contaminadas por rollovers de futuros
  - BTC escasso (dados so a partir de 2024-05-19 no SQLite)
  - ES em DB separado com issues de sincronizacao

Metodologia walk-forward:
  - 2024 = treino (80% treino real + 20% validacao interna para early stopping)
  - 2025 = BLIND OOS genuino (nunca visto pelo modelo durante treino)
  - 2026 = live test (tambem cego)

Uso: python train_divergencia_yfinance.py [--forward 4]
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

BASE    = Path(__file__).parent
PERIOD  = 720
ROLLOVER_THRESHOLD = 0.03

TICKERS = {
    'mnq': 'MNQ=F',
    'es':  'ES=F',
    'btc': 'BTC-USD',
    'cl':  'CL=F',
}


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
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))
    return None


def compute(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
    adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=17)
    df['adx'] = adx_i.adx(); df['pdi'] = adx_i.adx_pos(); df['mdi'] = adx_i.adx_neg()
    df['ret1'] = df['close'].pct_change(1)
    df['ret4'] = df['close'].pct_change(4)
    df['ret8'] = df['close'].pct_change(8)
    df['vol'] = df['ret1'].rolling(20).std()
    df['bb_w'] = df['close'].rolling(20).std() * 2 / df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['dist_sma50'] = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['above_sma50'] = (df['close'] > df['sma50']).astype(int)
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['dist_ema20'] = (df['close'] - df['ema20']) / df['ema20'] * 100
    df['above_ema20'] = (df['close'] > df['ema20']).astype(int)
    return df


def build_features(raw, forward=4):
    """Constroi features identicas ao predict_divergencia.py."""
    mnq_raw = raw['mnq']; es_raw = raw['es']
    btc_raw = raw['btc']; cl_raw = raw['cl']

    mnq = compute(mnq_raw); es = compute(es_raw)
    btc = compute(btc_raw); cl = compute(cl_raw)

    # Alinhar indices
    idx = mnq.index.intersection(es.index).intersection(btc.index).intersection(cl.index)
    mnq = mnq.loc[idx]; es = es.loc[idx]
    btc = btc.loc[idx]; cl = cl.loc[idx]
    mnq_raw = mnq_raw.reindex(idx, method='ffill')
    es_raw  = es_raw.reindex(idx, method='ffill')
    btc_raw = btc_raw.reindex(idx, method='ffill')
    cl_raw  = cl_raw.reindex(idx, method='ffill')

    f = pd.DataFrame(index=idx)

    # Retornos
    for nome, s in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'r_{nome}_1h'] = s['ret1'] * 100
        f[f'r_{nome}_4h'] = s['ret4'] * 100
    f['r_mnq_8h'] = mnq['ret8'] * 100

    # Correlacoes ES/BTC com MNQ
    f['es_mnq_mesmo']  = (((f['r_es_1h']>0)&(f['r_mnq_1h']>0))|((f['r_es_1h']<0)&(f['r_mnq_1h']<0))).astype(int)
    f['es_mnq_oposto'] = (((f['r_es_1h']>0)&(f['r_mnq_1h']<0))|((f['r_es_1h']<0)&(f['r_mnq_1h']>0))).astype(int)
    f['btc_mnq_mesmo'] = (((f['r_btc_1h']>0)&(f['r_mnq_1h']>0))|((f['r_btc_1h']<0)&(f['r_mnq_1h']<0))).astype(int)
    f['btc_mnq_oposto']= (((f['r_btc_1h']>0)&(f['r_mnq_1h']<0))|((f['r_btc_1h']<0)&(f['r_mnq_1h']>0))).astype(int)
    f['es_btc_discordam_mnq'] = (f['es_mnq_oposto'] & f['btc_mnq_oposto']).astype(int)
    f['es_btc_concordam_mnq'] = (f['es_mnq_mesmo']  & f['btc_mnq_mesmo']).astype(int)

    # RSI e divergencias
    for nome, d in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'rsi_{nome}'] = d['rsi']
    f['div_cl']  = f['rsi_mnq'] - f['rsi_cl']
    f['div_es']  = f['rsi_mnq'] - f['rsi_es']
    f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']
    f['rsi_mnq_acima_60']  = (f['rsi_mnq'] > 60).astype(int)
    f['rsi_mnq_abaixo_40'] = (f['rsi_mnq'] < 40).astype(int)

    # Key levels - abertura 1h
    for nome, raw_d in [('mnq',mnq_raw),('es',es_raw),('btc',btc_raw),('cl',cl_raw)]:
        s = raw_d['open']
        f[f'open_1h_{nome}'] = s
        f[f'open_1h_acima_close_ant_{nome}'] = (s > f[f'rsi_{nome}'].shift(1).mul(0) + mnq['close'].shift(1)).astype(int)

    # Corrige comparacao de abertura com close anterior da propria serie
    for nome, d in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        s = f[f'open_1h_{nome}']
        f[f'open_1h_acima_close_ant_{nome}'] = (s > d['close'].shift(1)).astype(int)

    # Abertura 4h
    for nome in ['mnq','es','btc','cl']:
        s = f[f'open_1h_{nome}']
        o4 = s.groupby(idx.floor('4h')).transform('first')
        ca4 = s.shift(1).groupby(idx.floor('4h')).transform('first')
        f[f'open_4h_acima_4h_ant_{nome}'] = (o4 > ca4).astype(int)
        f[f'open_4h_dist_{nome}'] = (s - o4) / o4.replace(0, np.nan) * 100

    # Abertura daily
    for nome in ['mnq','es','btc','cl']:
        s = f[f'open_1h_{nome}']
        d = s.groupby(idx.date).transform('first')
        ca = s.shift(1).groupby(idx.date).transform('first')
        f[f'open_d_acima_d_ant_{nome}'] = (d > ca).astype(int)
        f[f'open_d_dist_{nome}'] = (s - d) / d.replace(0, np.nan) * 100

    # Abertura weekly
    for nome in ['mnq','es','btc','cl']:
        s = f[f'open_1h_{nome}']
        w = s.groupby(pd.Grouper(freq='W-MON')).transform('first')
        wc = s.shift(1).groupby(pd.Grouper(freq='W-MON')).transform('first')
        f[f'open_w_acima_w_ant_{nome}'] = (w > wc).astype(int)
        f[f'open_w_dist_{nome}'] = (s - w) / w.replace(0, np.nan) * 100

    f['open_mnq_acima_cl']  = (f['open_1h_mnq'] > f['open_1h_cl']).astype(int)
    f['open_mnq_acima_es']  = (f['open_1h_mnq'] > f['open_1h_es']).astype(int)
    f['open_mnq_acima_btc'] = (f['open_1h_mnq'] > f['open_1h_btc']).astype(int)

    # ADX
    for nome, d in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'adx_{nome}'] = d['adx']
    f['adx_mnq_alto'] = (f['adx_mnq'] > 14).astype(int)
    f['adx_cl_alto']  = (f['adx_cl']  > 14).astype(int)
    f['adx_es_alto']  = (f['adx_es']  > 14).astype(int)

    # DI spread
    for nome, d in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'di_spread_{nome}'] = d['pdi'] - d['mdi']

    # SMA50 / EMA20
    for nome, d in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'above_sma50_{nome}'] = d['above_sma50']
        f[f'above_ema20_{nome}'] = d['above_ema20']
        f[f'dist_sma50_{nome}']  = d['dist_sma50']
        f[f'dist_ema20_{nome}']  = d['dist_ema20']
    f['sma50_alignment'] = sum(f[f'above_sma50_{n}'] for n in ['mnq','es','btc','cl'])
    f['ema20_alignment'] = sum(f[f'above_ema20_{n}'] for n in ['mnq','es','btc','cl'])

    # Volatilidade
    for nome, d in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'vol_{nome}']  = d['vol'] * 100
        f[f'bb_w_{nome}'] = d['bb_w'] * 100

    # Tempo
    f['hour'] = idx.hour; f['dow'] = idx.dayofweek
    f['is_us']      = f['hour'].between(9, 17).astype(int)
    f['is_asia']    = f['hour'].between(0, 8).astype(int)
    f['is_evening'] = f['hour'].between(18, 23).astype(int)

    # Target: MNQ > 0.1% nas proximas N horas
    f['mnq_fwd'] = mnq['close'].shift(-forward) / mnq['close'] - 1
    f['target']  = (f['mnq_fwd'] > 0.001).astype(int)

    f = f.dropna()

    # Filtro rollover: remove barras com retorno primario > 3%
    n_antes = len(f)
    f = f[f['r_mnq_1h'].abs() <= (ROLLOVER_THRESHOLD * 100)].copy()
    n_filt = n_antes - len(f)
    if n_filt > 0:
        print(f'  Filtro rollover: {n_filt} barras removidas (|r_mnq_1h| > 3%)')

    # Colunas a excluir do feature set (precos, alvos, colunas de abertura absolutas)
    skip = {
        'mnq_fwd', 'target',
        'open_1h_mnq', 'open_1h_es', 'open_1h_btc', 'open_1h_cl',
        'r_mnq_1h', 'r_es_1h', 'r_btc_1h', 'r_cl_1h',
    }
    feat_cols = [c for c in f.columns if c not in skip]
    return f, feat_cols


def train_walk_forward(X, y, label='modelo_divergencia'):
    years = X.index.year.unique()

    if 2024 in years and 2025 in years:
        mask_tr = X.index.year == 2024
        mask_os = X.index.year == 2025
        mask_lv = X.index.year == 2026 if 2026 in years else pd.Series(False, index=X.index)
        X_tr = X[mask_tr]; y_tr = y[mask_tr]
        X_os = X[mask_os]; y_os = y[mask_os]
        X_lv = X[mask_lv]; y_lv = y[mask_lv]
        if len(X_tr) >= 100 and len(X_os) >= 50:
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
        print('  Modo: split 70/30')
    else:
        print('  Modo: walk-forward por ano (blind OOS genuino)')

    # Validacao interna: 80% treino + 20% early stopping
    val_split = int(len(X_tr) * 0.80)
    X_fit, X_val = X_tr.iloc[:val_split], X_tr.iloc[val_split:]
    y_fit, y_val = y_tr.iloc[:val_split], y_tr.iloc[val_split:]

    ratio = y_fit.sum() / len(y_fit)
    sw = y_fit.map({0: ratio, 1: 1 - ratio})

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
        eval_metric='logloss', early_stopping_rounds=40, random_state=42,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
              sample_weight=sw.values, verbose=False)

    print(f'\n  --- {label} ---')
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
        proba = model.predict_proba(X_t)[:, 1]
        auc   = roc_auc_score(y_t, proba)
        baseline = float(y_t.mean())
        acc = (preds == y_t).mean()

        print(f'\n  [{split_name}] {X_t.index[0].date()} -> {X_t.index[-1].date()}')
        print(f'  N={len(X_t)} | AUC={auc:.4f} | Baseline={baseline:.2%} | Acuracia={acc:.2%}')
        print(classification_report(y_t, preds, target_names=['NAO','DIVERGENCIA'],
                                    zero_division=0))

        # Analise por threshold de probabilidade
        print('  Prob threshold analysis:')
        for th in [0.4, 0.5, 0.6, 0.7]:
            pbin = (proba >= th).astype(int)
            n = pbin.sum()
            if n < 5: continue
            precision = (pbin & y_t.values).sum() / max(n, 1)
            print(f'    th={th:.1f}: N={n:>4}  Precision={precision:.2%}  '
                  f'Edge={precision - baseline:+.2%}')

        results[split_name] = {'auc': auc, 'baseline': baseline, 'n': len(X_t)}

    main_auc = results.get('BLIND OOS 2025', {}).get('auc', 0.0)
    baseline_oos = results.get('BLIND OOS 2025', {}).get('baseline', 0.0)
    return model, main_auc, baseline_oos, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--forward', type=int, default=4, help='Horas a frente (default 4)')
    args = p.parse_args()

    print(f'Treinando model_divergencia (target=MNQ>0.1% em {args.forward}h)')
    print(f'Fonte: yfinance auto_adjust=True  |  Periodo: {PERIOD} dias')
    print(f'Metodologia: walk-forward por ano + blind OOS genuino\n')

    print('Baixando dados...')
    raw = {}
    for sym, ticker in TICKERS.items():
        print(f'  {ticker}...', end=' ', flush=True)
        df = download_asset(ticker)
        if df is None or len(df) < 200:
            print('FALHOU'); return
        print(f'{len(df)} candles  {df.index[0].date()} -> {df.index[-1].date()}')
        raw[sym] = df

    print('\nConstruindo features...')
    df, feat_cols = build_features(raw, forward=args.forward)

    print(f'Dataset limpo: {len(df)} amostras')
    print(f'Features: {len(feat_cols)}')
    print(f'Target=1 (LONG): {df["target"].mean():.2%}  |  Baseline: {df["target"].mean():.2%}')

    if len(df) < 300:
        print('Poucos dados. Abortando.'); return

    X = df[feat_cols].fillna(0)
    y = df['target']

    model, auc, baseline, results = train_walk_forward(X, y)

    out_path = BASE / 'model_divergencia.pkl'
    with open(out_path, 'wb') as fp:
        pickle.dump({
            'model':    model,
            'features': feat_cols,
            'forward':  args.forward,
            'auc':      auc,
            'baseline': baseline,
            'source':   'yfinance',
            'results':  results,
        }, fp)

    print(f'\nModelo salvo: {out_path}')
    print(f'AUC OOS 2025 = {auc:.4f}  |  Baseline = {baseline:.2%}')
    status = 'OK' if auc > 0.52 else 'FRACO'
    print(f'Status: {status}')


if __name__ == '__main__':
    main()

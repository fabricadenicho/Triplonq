"""
Backtest S2 SHORT — simetrico ao S1 LONG
Sinal SHORT quando:
  1. rot_score_short >= 60  (9 condicoes espelhadas)
  2. div_cl < 0             (CL RSI > MNQ RSI = vies SHORT)
  3. close < 4H open        (preco abaixo da abertura do candle 4H)
  4. ml_prob >= threshold   (XGBoost treinado para queda)

Split: 80% treino / 10% validacao OOS / 10% OOS cego
Uso: python backtest_s2_4h.py [--forward 4] [--period 720] [--thr 0.50] [--save-trades]
"""
import argparse, warnings, pickle
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score

BASE               = Path(__file__).parent
ROLLOVER_THRESHOLD = 0.03
FWD_TARGET_PCT     = 0.001   # queda > 0.1% = win

TICKERS = {
    'mnq': 'MNQ=F',
    'es':  'ES=F',
    'btc': 'BTC-USD',
    'cl':  'CL=F',
}


# ─────────────────────────────────────────────────────────────────────────────
def download_asset(ticker, period_days):
    import time
    end   = datetime.now()
    start = end - timedelta(days=period_days)
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, interval='1h',
                             auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = df.columns.str.lower()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df[['open','high','low','close','volume']].dropna(subset=['close'])
        except Exception:
            pass
        if attempt < 2:
            time.sleep(5 * (attempt + 1))
    return None


def compute(df):
    d = df.copy()
    d['rsi']         = ta.momentum.RSIIndicator(d['close'], window=21).rsi()
    adx_i            = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=17)
    d['adx']         = adx_i.adx()
    d['pdi']         = adx_i.adx_pos()
    d['mdi']         = adx_i.adx_neg()
    d['ret1']        = d['close'].pct_change(1)
    d['ret4']        = d['close'].pct_change(4)
    d['ret8']        = d['close'].pct_change(8)
    d['vol']         = d['ret1'].rolling(20).std()
    d['bb_w']        = d['close'].rolling(20).std() * 2 / d['close'].rolling(20).mean()
    d['sma50']       = d['close'].rolling(50).mean()
    d['dist_sma50']  = (d['close'] - d['sma50']) / d['sma50'] * 100
    d['above_sma50'] = (d['close'] > d['sma50']).astype(int)
    d['ema20']       = d['close'].ewm(span=20, adjust=False).mean()
    d['dist_ema20']  = (d['close'] - d['ema20']) / d['ema20'] * 100
    d['above_ema20'] = (d['close'] > d['ema20']).astype(int)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Features — identicas ao S1 (mesmo feature set, target diferente)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(raw, forward=4):
    mnq_raw = raw['mnq']; es_raw = raw['es']
    btc_raw = raw['btc']; cl_raw = raw['cl']

    mnq = compute(mnq_raw); es  = compute(es_raw)
    btc = compute(btc_raw); cl  = compute(cl_raw)

    idx = (mnq.index.intersection(es.index)
                    .intersection(btc.index)
                    .intersection(cl.index))
    mnq = mnq.loc[idx]; es  = es.loc[idx]
    btc = btc.loc[idx]; cl  = cl.loc[idx]
    mnq_raw = mnq_raw.reindex(idx, method='ffill')
    es_raw  = es_raw.reindex(idx,  method='ffill')
    btc_raw = btc_raw.reindex(idx, method='ffill')
    cl_raw  = cl_raw.reindex(idx,  method='ffill')

    f = pd.DataFrame(index=idx)
    f['mnq'] = mnq['close']; f['es'] = es['close']
    f['btc'] = btc['close']; f['cl'] = cl['close']

    for nome, s in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'r_{nome}_1h'] = s['ret1'] * 100
        f[f'r_{nome}_4h'] = s['ret4'] * 100
    f['r_mnq_8h'] = mnq['ret8'] * 100

    f['es_mnq_mesmo']   = (((f['r_es_1h']>0)&(f['r_mnq_1h']>0))|((f['r_es_1h']<0)&(f['r_mnq_1h']<0))).astype(int)
    f['es_mnq_oposto']  = (((f['r_es_1h']>0)&(f['r_mnq_1h']<0))|((f['r_es_1h']<0)&(f['r_mnq_1h']>0))).astype(int)
    f['btc_mnq_mesmo']  = (((f['r_btc_1h']>0)&(f['r_mnq_1h']>0))|((f['r_btc_1h']<0)&(f['r_mnq_1h']<0))).astype(int)
    f['btc_mnq_oposto'] = (((f['r_btc_1h']>0)&(f['r_mnq_1h']<0))|((f['r_btc_1h']<0)&(f['r_mnq_1h']>0))).astype(int)
    f['es_btc_discordam_mnq'] = (f['es_mnq_oposto'] & f['btc_mnq_oposto']).astype(int)
    f['es_btc_concordam_mnq'] = (f['es_mnq_mesmo']  & f['btc_mnq_mesmo']).astype(int)

    for nome, d in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'rsi_{nome}'] = d['rsi']
    f['div_cl']  = f['rsi_mnq'] - f['rsi_cl']
    f['div_es']  = f['rsi_mnq'] - f['rsi_es']
    f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']
    f['rsi_mnq_acima_60']  = (f['rsi_mnq'] > 60).astype(int)
    f['rsi_mnq_abaixo_40'] = (f['rsi_mnq'] < 40).astype(int)

    for nome, raw_d in [('mnq',mnq_raw),('es',es_raw),('btc',btc_raw),('cl',cl_raw)]:
        f[f'open_1h_{nome}'] = raw_d['open']
    for nome, d in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'open_1h_acima_close_ant_{nome}'] = (f[f'open_1h_{nome}'] > d['close'].shift(1)).astype(int)

    for nome in ['mnq','es','btc','cl']:
        s   = f[f'open_1h_{nome}']
        o4  = s.groupby(idx.floor('4h')).transform('first')
        ca4 = s.shift(1).groupby(idx.floor('4h')).transform('first')
        f[f'open_4h_acima_4h_ant_{nome}'] = (o4 > ca4).astype(int)
        f[f'open_4h_dist_{nome}']         = (s - o4) / o4.replace(0, np.nan) * 100

    for nome in ['mnq','es','btc','cl']:
        s  = f[f'open_1h_{nome}']
        d  = s.groupby(idx.date).transform('first')
        ca = s.shift(1).groupby(idx.date).transform('first')
        f[f'open_d_acima_d_ant_{nome}'] = (d > ca).astype(int)
        f[f'open_d_dist_{nome}']        = (s - d) / d.replace(0, np.nan) * 100

    for nome in ['mnq','es','btc','cl']:
        s  = f[f'open_1h_{nome}']
        w  = s.groupby(pd.Grouper(freq='W-MON')).transform('first')
        wc = s.shift(1).groupby(pd.Grouper(freq='W-MON')).transform('first')
        f[f'open_w_acima_w_ant_{nome}'] = (w > wc).astype(int)
        f[f'open_w_dist_{nome}']        = (s - w) / w.replace(0, np.nan) * 100

    f['open_mnq_acima_cl']  = (f['open_1h_mnq'] > f['open_1h_cl']).astype(int)
    f['open_mnq_acima_es']  = (f['open_1h_mnq'] > f['open_1h_es']).astype(int)
    f['open_mnq_acima_btc'] = (f['open_1h_mnq'] > f['open_1h_btc']).astype(int)

    for nome, d in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
        f[f'adx_{nome}']        = d['adx']
        f[f'di_spread_{nome}']  = d['pdi'] - d['mdi']
        f[f'above_sma50_{nome}']= d['above_sma50']
        f[f'above_ema20_{nome}']= d['above_ema20']
        f[f'dist_sma50_{nome}'] = d['dist_sma50']
        f[f'dist_ema20_{nome}'] = d['dist_ema20']
        f[f'vol_{nome}']        = d['vol'] * 100
        f[f'bb_w_{nome}']       = d['bb_w'] * 100

    f['adx_mnq_alto'] = (f['adx_mnq'] > 14).astype(int)
    f['adx_cl_alto']  = (f['adx_cl']  > 14).astype(int)
    f['adx_es_alto']  = (f['adx_es']  > 14).astype(int)
    f['sma50_alignment'] = sum(f[f'above_sma50_{n}'] for n in ['mnq','es','btc','cl'])
    f['ema20_alignment'] = sum(f[f'above_ema20_{n}'] for n in ['mnq','es','btc','cl'])

    f['hour']       = idx.hour
    f['dow']        = idx.dayofweek
    f['is_us']      = f['hour'].between(9, 17).astype(int)
    f['is_asia']    = f['hour'].between(0,  8).astype(int)
    f['is_evening'] = f['hour'].between(18, 23).astype(int)

    # TARGET SHORT: MNQ CAI > 0.1% nas proximas N horas
    f['mnq_fwd']     = mnq['close'].shift(-forward) / mnq['close'] - 1
    f['target_long'] = (f['mnq_fwd'] >  FWD_TARGET_PCT).astype(int)
    f['target']      = (f['mnq_fwd'] < -FWD_TARGET_PCT).astype(int)

    # 4H open do MNQ
    f['open_4h_mnq'] = (
        mnq_raw['open'].resample('4h').first()
                       .reindex(idx, method='ffill')
    )

    f = f.dropna()

    n_antes = len(f)
    f = f[f['r_mnq_1h'].abs() <= (ROLLOVER_THRESHOLD * 100)].copy()
    removed = n_antes - len(f)
    if removed > 0:
        print(f'  Rollover filter: {removed} barras removidas')

    return f


# ─────────────────────────────────────────────────────────────────────────────
# Rot score SHORT — espelho do S1 com condicoes invertidas onde faz sentido
#
# S1 LONG rot_c3: rsi_mnq subiu +2 E rsi_cl caiu -2  (MNQ fortalece vs CL)
# S2 SHORT rot_c3: rsi_mnq caiu -2 E rsi_cl subiu +2 (MNQ fraqueja vs CL)
#
# S1 LONG rot_c6: open_d_cl acima do anterior          (CL diario subindo)
# S2 SHORT rot_c6: open_d_cl ABAIXO do anterior         (CL diario caindo)
#
# S1 LONG rot_c8: sma50_alignment >= 2  (maioria acima da SMA50)
# S2 SHORT rot_c8: sma50_alignment <= 2 (maioria abaixo da SMA50)
#
# Demais condicoes sao neutras (deteccao de rotacao/volatilidade) e ficam iguais.
# ─────────────────────────────────────────────────────────────────────────────
def compute_rot_score_short(f):
    prev_div  = f['div_cl'].shift(1)
    rot_c1    = (((f['div_cl'] > 0) & (prev_div <= 0)) |
                 ((f['div_cl'] < 0) & (prev_div >= 0))).astype(float)

    div_d2    = f['div_cl'] - f['div_cl'].shift(2)
    rot_c2    = (div_d2.abs() > 5.0).astype(float)

    # SHORT: MNQ RSI caiu >2 E CL RSI subiu >2
    rsi_mnq_d2 = f['rsi_mnq'] - f['rsi_mnq'].shift(2)
    rsi_cl_d2  = f['rsi_cl']  - f['rsi_cl'].shift(2)
    rot_c3     = ((rsi_mnq_d2 < -2.0) & (rsi_cl_d2 > 2.0)).astype(float)

    rot_c4 = f['es_mnq_mesmo'].astype(float)   # ambos caindo juntos
    rot_c5 = (f['bb_w_cl'] > 1.5).astype(float)

    # SHORT: CL diario abrindo ABAIXO do anterior (CL fraco = MNQ bearish)
    rot_c6 = (1 - f['open_d_acima_d_ant_cl']).astype(float)

    rot_c7 = ((f['adx_mnq'] >= 12) & (f['adx_mnq'] <= 20)).astype(float)

    # SHORT: maioria dos ativos ABAIXO da SMA50
    rot_c8 = (f['sma50_alignment'] <= 2).astype(float)

    rot_c9 = ((f['rsi_mnq'] > 55) | (f['rsi_mnq'] < 45)).astype(float)

    rot_raw = (rot_c1*3.0 + rot_c2*2.5 + rot_c3*2.5 + rot_c4*2.0 +
               rot_c5*1.5 + rot_c6*1.5 + rot_c7*1.5 + rot_c8*1.5 + rot_c9*1.0)
    return (rot_raw / 17.0 * 100).round(1)


# ─────────────────────────────────────────────────────────────────────────────
def _pct(x): return f'{x:.1%}'

def eval_split(label, fslice, probs=None, ml_thr=0.50,
               save_trades=False, trades_path=None):
    n        = len(fslice)
    baseline = float(fslice['target'].mean())   # % das barras que cairam

    mask_sem = (
        (fslice['rot_score_short'] >= 60) &
        (fslice['div_cl']          <  0 ) &
        (fslice['below_4h_open']   == 1 )
    )
    n_sem  = int(mask_sem.sum())
    wr_sem = float(fslice.loc[mask_sem, 'target'].mean()) if n_sem > 0 else 0.0

    print(f'\n  +- {label}')
    print(f'  |  Periodo : {fslice.index[0].date()} -> {fslice.index[-1].date()}  ({n} barras)')
    print(f'  |  Baseline: {_pct(baseline)}')
    print(f'  |  S2 sem ML  N={n_sem:>4}  WR={_pct(wr_sem)}  Edge={wr_sem-baseline:+.1%}')

    result = {
        'label': label, 'n': n, 'baseline': baseline,
        'n_s2_sem': n_sem, 'wr_s2_sem': wr_sem,
    }

    if probs is not None:
        mask_com = mask_sem & (probs >= ml_thr)
        n_com    = int(mask_com.sum())
        wr_com   = float(fslice.loc[mask_com, 'target'].mean()) if n_com > 0 else 0.0
        alpha    = wr_com - wr_sem
        result.update({'n_s2_com': n_com, 'wr_s2_com': wr_com, 'alpha': alpha})

        print(f'  |  S2 com ML  N={n_com:>4}  WR={_pct(wr_com)}  '
              f'Edge={wr_com-baseline:+.1%}  Alpha vs semML={alpha:+.1%}')

        print(f'  |  ML threshold sweep:')
        for th in [0.45, 0.50, 0.55, 0.60, 0.65]:
            m    = mask_sem & (probs >= th)
            n_th = int(m.sum())
            if n_th < 3:
                continue
            wr_th = float(fslice.loc[m, 'target'].mean())
            print(f'  |    >= {th:.2f}: N={n_th:>4}  WR={_pct(wr_th)}  Edge={wr_th-baseline:+.1%}')

        if n_com >= 10:
            by_hour = (fslice.loc[mask_com]
                             .groupby('hour')['target']
                             .agg(['sum','count','mean'])
                             .rename(columns={'sum':'wins','count':'n','mean':'wr'}))
            if len(by_hour) > 0:
                top = by_hour.nlargest(5, 'wr')
                print(f'  |  Top horas (S2+ML):')
                for h, row in top.iterrows():
                    print(f'  |    h={h:02d}  N={int(row["n"]):>3}  WR={_pct(row["wr"])}')

        if save_trades and trades_path is not None:
            trades = fslice.loc[mask_com, [
                'mnq','div_cl','rot_score_short','below_4h_open','target','hour','dow'
            ]].copy()
            trades['ml_prob'] = probs[mask_com]
            trades['result']  = trades['target'].map({1:'WIN',0:'LOSS'})
            trades['sinal']   = 'SHORT'
            mode = 'a' if trades_path.exists() else 'w'
            trades.to_csv(trades_path, mode=mode, header=(mode == 'w'))

    print(f'  +-')
    return result


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='Backtest S2 SHORT')
    p.add_argument('--forward',     type=int,   default=4)
    p.add_argument('--period',      type=int,   default=720)
    p.add_argument('--thr',         type=float, default=0.50)
    p.add_argument('--save-trades', action='store_true')
    args = p.parse_args()

    trades_path = BASE / 'backtest_s2_trades.csv' if args.save_trades else None
    if trades_path and trades_path.exists():
        trades_path.unlink()

    print('=' * 62)
    print(' BACKTEST S2 SHORT -- MNQ cai > 0.1% em 4h')
    print('=' * 62)
    print(f' Periodo   : {args.period} dias (1h bars)')
    print(f' Forward   : {args.forward}h  |  Target: MNQ < -{FWD_TARGET_PCT*100:.1f}%')
    print(f' ML thr    : {args.thr}')
    print(f' Split     : 80% treino / 10% validacao / 10% OOS cego')
    print('=' * 62)

    print('\nBaixando dados...')
    raw = {}
    for sym, ticker in TICKERS.items():
        print(f'  {ticker}...', end=' ', flush=True)
        df = download_asset(ticker, args.period)
        if df is None or len(df) < 200:
            print('FALHOU'); return
        print(f'{len(df)} candles  {df.index[0].date()} -> {df.index[-1].date()}')
        raw[sym] = df

    print('\nConstruindo features...')
    f = build_features(raw, forward=args.forward)

    target_long_rate = float(f['target_long'].mean())
    target_short_rate = float(f['target'].mean())
    print(f'Dataset: {len(f)} barras  {f.index[0].date()} -> {f.index[-1].date()}')
    print(f'Baseline LONG  (sobe >0.1%): {target_long_rate:.1%}')
    print(f'Baseline SHORT (cai >0.1%):  {target_short_rate:.1%}')

    print('\nCalculando rot_score_short...')
    f['rot_score_short'] = compute_rot_score_short(f)
    f['below_4h_open']   = (f['mnq'] < f['open_4h_mnq']).astype(int)
    f = f.dropna(subset=['rot_score_short']).copy()

    rot_fires = int((f['rot_score_short'] >= 60).sum())
    s2_sem_n  = int(((f['rot_score_short'] >= 60) &
                     (f['div_cl'] < 0) &
                     (f['below_4h_open'] == 1)).sum())
    print(f'rot_score_short >= 60: {rot_fires} barras ({rot_fires/len(f):.1%})')
    print(f'S2 sem ML total: {s2_sem_n} sinais')

    if len(f) < 300:
        print('Poucos dados. Abortando.'); return

    # Splits
    n     = len(f)
    n_tr  = int(n * 0.80)
    n_val = int(n * 0.10)

    f_tr  = f.iloc[:n_tr].copy()
    f_val = f.iloc[n_tr : n_tr + n_val].copy()
    f_oos = f.iloc[n_tr + n_val :].copy()

    print(f'\nSplits:')
    print(f'  Treino (80%)    : {len(f_tr)} barras  {f_tr.index[0].date()} -> {f_tr.index[-1].date()}')
    print(f'  Validacao (10%) : {len(f_val)} barras  {f_val.index[0].date()} -> {f_val.index[-1].date()}')
    print(f'  OOS cego (10%)  : {len(f_oos)} barras  {f_oos.index[0].date()} -> {f_oos.index[-1].date()}')

    # Feature set ML (identico ao S1 — so o target muda)
    SKIP = {
        'target', 'target_long', 'mnq_fwd',
        'mnq', 'es', 'btc', 'cl',
        'open_1h_mnq', 'open_1h_es', 'open_1h_btc', 'open_1h_cl',
        'r_mnq_1h', 'r_es_1h', 'r_btc_1h', 'r_cl_1h',
        'open_4h_mnq', 'rot_score_short', 'below_4h_open',
    }
    feat_cols = [c for c in f.columns if c not in SKIP]

    X_tr  = f_tr [feat_cols].fillna(0)
    y_tr  = f_tr ['target']
    X_val = f_val[feat_cols].fillna(0)
    y_val = f_val['target']
    X_oos = f_oos[feat_cols].fillna(0)
    y_oos = f_oos['target']

    cut   = int(len(X_tr) * 0.80)
    X_fit = X_tr.iloc[:cut];  y_fit = y_tr.iloc[:cut]
    X_es  = X_tr.iloc[cut:];  y_es  = y_tr.iloc[cut:]

    ratio = float(y_fit.mean())
    sw    = y_fit.map({0: ratio, 1: 1 - ratio})

    print(f'\nTreinando XGBoost SHORT ({len(feat_cols)} features)...')
    print(f'  Fit: {len(X_fit)} barras  |  Early-stop val: {len(X_es)} barras')

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
        eval_metric='logloss', early_stopping_rounds=50,
        random_state=42, verbosity=0,
    )
    model.fit(X_fit, y_fit,
              eval_set=[(X_es, y_es)],
              sample_weight=sw.values,
              verbose=False)
    print(f'  Melhor iteracao: {model.best_iteration}')

    prob_tr  = pd.Series(model.predict_proba(X_tr )[:, 1], index=f_tr.index)
    prob_val = pd.Series(model.predict_proba(X_val)[:, 1], index=f_val.index)
    prob_oos = pd.Series(model.predict_proba(X_oos)[:, 1], index=f_oos.index)

    auc_tr  = roc_auc_score(y_tr,  prob_tr)  if y_tr.sum()  > 10 else 0.0
    auc_val = roc_auc_score(y_val, prob_val) if y_val.sum() > 10 else 0.0
    auc_oos = roc_auc_score(y_oos, prob_oos) if y_oos.sum() > 10 else 0.0
    print(f'  AUC treino={auc_tr:.4f}  |  AUC val={auc_val:.4f}  |  AUC OOS={auc_oos:.4f}')

    print('\n' + '=' * 62)
    print(' RESULTADOS POR SPLIT')
    print('=' * 62)

    r_tr  = eval_split('TREINO (80% in-sample)',         f_tr,  prob_tr,
                       ml_thr=args.thr, save_trades=args.save_trades, trades_path=trades_path)
    r_val = eval_split('VALIDACAO OOS (10%)',             f_val, prob_val,
                       ml_thr=args.thr, save_trades=args.save_trades, trades_path=trades_path)
    r_oos = eval_split('OOS CEGO (10% -- nunca visto)',  f_oos, prob_oos,
                       ml_thr=args.thr, save_trades=args.save_trades, trades_path=trades_path)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feat_cols).nlargest(20)
    print('\n' + '=' * 62)
    print(' TOP 20 FEATURES SHORT (gain)')
    print('=' * 62)
    for fname, gain in imp.items():
        bar = '#' * int(gain / imp.max() * 30)
        print(f'  {fname:<40} {gain:.4f}  {bar}')

    # Comparar com S1 LONG se existir
    s1_model_path = BASE / 'model_s1_4h.pkl'
    if s1_model_path.exists():
        s1 = pickle.load(open(s1_model_path, 'rb'))
        s1_imp = pd.Series(s1['model'].feature_importances_, index=s1['features']).nlargest(10)
        s2_imp = imp.head(10)
        top_s1 = set(s1_imp.index)
        top_s2 = set(s2_imp.index)
        so_s2  = top_s2 - top_s1
        so_s1  = top_s1 - top_s2
        print(f'\n  Exclusivo S2 SHORT: {sorted(so_s2)}')
        print(f'  Exclusivo S1 LONG : {sorted(so_s1)}')

    # Resumo
    print('\n' + '=' * 62)
    print(' RESUMO FINAL (OOS CEGO)')
    print('=' * 62)
    base_oos   = r_oos['baseline']
    wr_sem_oos = r_oos['wr_s2_sem']
    wr_com_oos = r_oos.get('wr_s2_com', 0.0)
    n_sem_oos  = r_oos['n_s2_sem']
    n_com_oos  = r_oos.get('n_s2_com', 0)

    print(f'  Baseline SHORT     : {_pct(base_oos)}  (barras que cairam >0.1%)')
    print(f'  S2 sem ML  N={n_sem_oos:>4}  WR={_pct(wr_sem_oos)}  Edge={wr_sem_oos-base_oos:+.1%}')
    print(f'  S2 com ML  N={n_com_oos:>4}  WR={_pct(wr_com_oos)}  Edge={wr_com_oos-base_oos:+.1%}  Alpha={wr_com_oos-wr_sem_oos:+.1%}')
    print(f'  AUC OOS            : {auc_oos:.4f}')

    if 'wr_s2_com' in r_tr and 'wr_s2_com' in r_oos:
        drift  = r_tr['wr_s2_com'] - r_oos['wr_s2_com']
        status = 'OK' if abs(drift) < 0.15 else 'ATENCAO: possivel overfitting'
        print(f'  Drift treino->OOS  : {drift:+.1%}  [{status}]')

    # Salvar modelo S2
    model_out = BASE / 'model_s2_4h.pkl'
    with open(model_out, 'wb') as fh:
        pickle.dump({
            'model':          model,
            'features':       feat_cols,
            'forward':        args.forward,
            'auc':            auc_oos,
            'baseline':       base_oos,
            'ml_thr':         args.thr,
            'source':         'yfinance',
            'split':          '80/10/10',
            'direction':      'SHORT',
            'wr_s2_sem_oos':  wr_sem_oos,
            'wr_s2_com_oos':  wr_com_oos,
        }, fh)
    print(f'\n  Modelo salvo : {model_out}')
    if args.save_trades and trades_path:
        print(f'  Trades salvos: {trades_path}')
    print('=' * 62)


if __name__ == '__main__':
    main()

"""
Backtest ML Scalper — 15m e 5m
Testa se ha edge para MNQ em timeframes curtos.
Forward fixo de 1H (4 barras em 15m, 12 em 5m).
Target: MNQ > +0.1% (LONG) ou < -0.1% (SHORT) no horizonte.
Split temporal 70/15/15.
"""
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd, numpy as np, ta, pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent


def download(ticker, interval, period='59d'):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()


def compute(df, rsi_w=14, adx_w=14, bb_w=20):
    d = df.copy()
    d['rsi']  = ta.momentum.RSIIndicator(d['close'], window=rsi_w).rsi()
    adx_i     = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=adx_w)
    d['adx']  = adx_i.adx()
    d['ret1'] = d['close'].pct_change(1)
    d['ret4'] = d['close'].pct_change(4)
    d['ret8'] = d['close'].pct_change(8)
    d['vol']  = d['ret1'].rolling(20).std()
    d['bb_w'] = d['close'].rolling(bb_w).std() * 2 / d['close'].rolling(bb_w).mean()
    d['sma50'] = d['close'].rolling(50).mean()
    d['above_sma50'] = (d['close'] > d['sma50']).astype(int)
    return d


def run(interval):
    fwd_bars   = 4  if interval == '15m' else 12   # ambos = 1H a frente
    target_pct = 0.001                              # mesmo threshold do S1
    delta_bars = 8  if interval == '15m' else 24   # equivalente a 2H

    print('\n' + '='*62)
    print(f'TIMEFRAME: {interval}  |  Forward: {fwd_bars} barras (1H)  |  Target: >+0.1%')
    print('='*62)

    print('Baixando...')
    raws = {}
    for ticker in ['MNQ=F', 'ES=F', 'CL=F', 'BTC-USD']:
        df = download(ticker, interval)
        if df is None:
            print(f'  ERRO: {ticker}'); return
        raws[ticker] = df
        print(f'  {ticker}: {len(df)} barras  {df.index[0].date()} -> {df.index[-1].date()}')

    mnq = compute(raws['MNQ=F'])
    es  = compute(raws['ES=F'])
    cl  = compute(raws['CL=F'])
    btc = compute(raws['BTC-USD'])
    mnq_raw = raws['MNQ=F']
    cl_raw  = raws['CL=F']

    idx = mnq.index.intersection(es.index).intersection(cl.index).intersection(btc.index)
    mnq=mnq.loc[idx]; es=es.loc[idx]; cl=cl.loc[idx]; btc=btc.loc[idx]
    mnq_raw = mnq_raw.reindex(idx, method='ffill')
    cl_raw  = cl_raw.reindex(idx, method='ffill')

    # ── Features ─────────────────────────────────────────────────────────────
    f = pd.DataFrame(index=idx)

    for nome, s in [('mnq',mnq),('es',es),('cl',cl),('btc',btc)]:
        f[f'ret1_{nome}'] = s['ret1'] * 100
        f[f'ret4_{nome}'] = s['ret4'] * 100
        f[f'ret8_{nome}'] = s['ret8'] * 100
        f[f'rsi_{nome}']  = s['rsi']
        f[f'adx_{nome}']  = s['adx']
        f[f'vol_{nome}']  = s['vol'] * 100
        f[f'bb_w_{nome}'] = s['bb_w'] * 100
        f[f'above50_{nome}'] = s['above_sma50']

    # Divergencias RSI
    f['div_cl']  = f['rsi_mnq'] - f['rsi_cl']
    f['div_es']  = f['rsi_mnq'] - f['rsi_es']
    f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']

    # Deltas no equivalente de 2H
    f['div_cl_delta']  = f['div_cl']  - f['div_cl'].shift(delta_bars)
    f['rsi_mnq_delta'] = f['rsi_mnq'] - f['rsi_mnq'].shift(delta_bars)
    f['rsi_cl_delta']  = f['rsi_cl']  - f['rsi_cl'].shift(delta_bars)
    f['div_cl_cruzou'] = (((f['div_cl'] > 0) & (f['div_cl'].shift(1) <= 0)) |
                          ((f['div_cl'] < 0) & (f['div_cl'].shift(1) >= 0))).astype(int)

    # Alinhamento SMA50
    f['sma50_align'] = (f['above50_mnq'] + f['above50_es'] +
                        f['above50_cl']  + f['above50_btc'])

    # 1H open (referencia de curto prazo — equivalente ao 4H open do S1)
    open_1h = mnq_raw['open'].resample('1h').first().reindex(idx, method='ffill')
    f['above_1h']  = (mnq['close'] > open_1h).astype(int)
    f['dist_1h']   = (mnq['close'] - open_1h) / open_1h * 100

    # Daily open
    open_d = mnq_raw['open'].resample('D').first().reindex(idx, method='ffill')
    f['above_d'] = (mnq['close'] > open_d).astype(int)
    f['dist_d']  = (mnq['close'] - open_d) / open_d * 100

    # CL daily open direction
    cl_dopen     = cl_raw['open'].resample('D').first().reindex(idx, method='ffill')
    cl_dopen_ant = cl_dopen.shift(1)
    f['cl_d_acima_ant'] = (cl_dopen > cl_dopen_ant).astype(int)

    # Regime: SMA50 nativo do TF (evita gaps do resample 1H)
    sma50_native     = mnq['close'].rolling(50).mean()
    f['regime_bull'] = (mnq['close'] > sma50_native).astype(int)
    f['dist_sma50n'] = (mnq['close'] - sma50_native) / sma50_native * 100

    # Tempo
    f['hour'] = idx.hour
    f['dow']  = idx.dayofweek
    f['is_us']  = f['hour'].between(13, 21).astype(int)
    f['is_asia'] = f['hour'].between(0, 8).astype(int)

    # ADX conditions
    f['adx_mnq_range'] = ((f['adx_mnq'] >= 12) & (f['adx_mnq'] <= 25)).astype(int)

    # Targets LONG e SHORT
    fwd_ret = mnq['close'].shift(-fwd_bars) / mnq['close'] - 1
    f['target_long']  = (fwd_ret >  target_pct).astype(int)
    f['target_short'] = (fwd_ret < -target_pct).astype(int)

    # Rollover filter (mais estrito em TF curto)
    ret_abs = mnq['ret1'].abs() * 100
    f = f.dropna()
    f = f[ret_abs.reindex(f.index) <= 0.8].copy()

    print(f'Dataset final: {len(f)} barras  {f.index[0].date()} -> {f.index[-1].date()}')

    # ── Split 70/15/15 temporal ───────────────────────────────────────────────
    n       = len(f)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train   = f.iloc[:n_train]
    val     = f.iloc[n_train:n_train+n_val]
    oos     = f.iloc[n_train+n_val:]
    print(f'Split: treino={len(train)} | val={len(val)} | OOS={len(oos)}')
    print(f'  OOS periodo: {oos.index[0].date()} -> {oos.index[-1].date()}')

    feat_cols = [c for c in f.columns
                 if c not in ['target_long','target_short']]

    results = {}
    for direction in ['long', 'short']:
        target_col = f'target_{direction}'
        bl_train   = float(train[target_col].mean())
        bl_oos     = float(oos[target_col].mean())
        print(f'\n--- {direction.upper()} | baseline treino={bl_train:.1%}  OOS={bl_oos:.1%} ---')

        X_tr = train[feat_cols].fillna(0)
        y_tr = train[target_col]
        X_vl = val[feat_cols].fillna(0)
        y_vl = val[target_col]
        X_os = oos[feat_cols].fillna(0)
        y_os = oos[target_col]

        model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.025,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
            eval_metric='auc', early_stopping_rounds=40,
            use_label_encoder=False, random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        prob_os = model.predict_proba(X_os)[:, 1]
        auc_os  = roc_auc_score(y_os, prob_os)
        print(f'  AUC OOS: {auc_os:.4f}')

        # Thresholds sem filtro
        print(f'  {"thr":>6}  {"N":>5}  {"WR":>6}  {"Edge":>7}')
        for thr in [0.45, 0.50, 0.55, 0.60, 0.65]:
            m = prob_os >= thr
            n_t = int(m.sum())
            if n_t < 5: continue
            wr   = float(y_os[m].mean())
            edge = wr - bl_oos
            print(f'  >={thr:.2f}  {n_t:>5}  {wr:.1%}  {edge:>+.1%}')

        # Com filtro S1/S2-like
        if direction == 'long':
            fmask = (oos['div_cl'] > 0) & (oos['above_1h'] == 1) & (oos['regime_bull'] == 1)
        else:
            fmask = (oos['div_cl'] < 0) & (oos['above_1h'] == 0) & (oos['regime_bull'] == 0)

        n_f = int(fmask.sum())
        if n_f >= 5:
            p_f = prob_os[fmask.values]
            y_f = y_os[fmask]
            bl_f = float(y_f.mean())
            print(f'\n  Filtro S1-like: N={n_f}  baseline={bl_f:.1%}')
            print(f'  {"thr":>6}  {"N":>5}  {"WR":>6}  {"Edge":>7}')
            for thr in [0.45, 0.50, 0.55, 0.60]:
                m2 = p_f >= thr
                n2 = int(m2.sum())
                if n2 < 3: continue
                wr2 = float(y_f[m2].mean())
                print(f'  >={thr:.2f}  {n2:>5}  {wr2:.1%}  {wr2-bl_f:>+.1%}')
        else:
            print(f'  Filtro S1-like: N={n_f} (poucos sinais no OOS)')

        # Salvar modelo
        path = BASE / f'model_scalper_{interval}_{direction}.pkl'
        pickle.dump({
            'model': model, 'features': feat_cols,
            'baseline': bl_train, 'interval': interval,
            'fwd_bars': fwd_bars, 'direction': direction,
            'auc_oos': round(auc_os, 4),
        }, open(path, 'wb'))
        print(f'  -> Salvo: {path.name}')
        results[(interval, direction)] = {'auc': auc_os, 'bl': bl_oos}

    return results


if __name__ == '__main__':
    all_results = {}
    for tf in ['15m', '5m']:
        r = run(tf)
        if r:
            all_results.update(r)

    print('\n' + '='*62)
    print('RESUMO FINAL')
    print('='*62)
    print(f'  {"TF":>4}  {"Dir":>6}  {"AUC OOS":>8}  {"Baseline":>9}')
    print('-'*42)
    for (tf, direction), v in all_results.items():
        print(f'  {tf:>4}  {direction.upper():>6}  {v["auc"]:.4f}    {v["bl"]:.1%}')
    print()
    print('Interpretacao:')
    print('  AUC > 0.53  = algum sinal, vale aprofundar')
    print('  AUC > 0.56  = edge real comparavel ao S1 (1H)')
    print('  AUC < 0.52  = ruido, nao prosseguir')

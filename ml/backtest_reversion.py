"""
Backtest ML — Reversao a Media (15m e 5m)
Logica oposta ao S1: entra quando preco esta LONGE da media, espera voltar.

Filtro de entrada (sem ML):
  LONG:  RSI(14) < 35  E  close < BB inferior
  SHORT: RSI(14) > 65  E  close > BB superior
  Opcional: ADX < 25 (mercado lateral, reversao mais confiavel)

Forward: 4 barras em 15m = 1H | 12 barras em 5m = 1H
Target:  MNQ > +0.1% (LONG) ou < -0.1% (SHORT)
Split:   70/15/15 temporal
"""
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd, numpy as np, ta, pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

BASE     = Path(__file__).parent
DATA_DIR = BASE / 'data_hist'


def load_hist(nome, interval):
    path = DATA_DIR / f'{nome}_{interval}.csv.gz'
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True, compression='gzip')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def download(ticker, interval, period='59d'):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()


def get_data(nome, ticker, interval):
    hist = load_hist(nome, interval)
    if not hist.empty:
        return hist
    return download(ticker, interval)


def compute(df, rsi_w=14, adx_w=14, bb_w=20):
    d = df.copy()
    d['rsi']  = ta.momentum.RSIIndicator(d['close'], window=rsi_w).rsi()
    adx_i     = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=adx_w)
    d['adx']  = adx_i.adx()
    d['ret1'] = d['close'].pct_change(1)
    d['ret4'] = d['close'].pct_change(4)
    d['ret8'] = d['close'].pct_change(8)
    d['vol']  = d['ret1'].rolling(20).std()

    bb         = ta.volatility.BollingerBands(d['close'], window=bb_w, window_dev=2)
    d['bb_mid'] = bb.bollinger_mavg()
    d['bb_upp'] = bb.bollinger_hband()
    d['bb_low'] = bb.bollinger_lband()
    d['bb_w']   = (d['bb_upp'] - d['bb_low']) / d['bb_mid'] * 100
    d['bb_pctb']= bb.bollinger_pband()   # 0=lower, 1=upper, <0=abaixo, >1=acima

    d['ema9']   = d['close'].ewm(span=9,  adjust=False).mean()
    d['ema20']  = d['close'].ewm(span=20, adjust=False).mean()
    d['sma50']  = d['close'].rolling(50).mean()
    d['dist_ema9']  = (d['close'] - d['ema9'])  / d['ema9']  * 100
    d['dist_ema20'] = (d['close'] - d['ema20']) / d['ema20'] * 100
    d['dist_sma50'] = (d['close'] - d['sma50']) / d['sma50'] * 100
    d['above_sma50'] = (d['close'] > d['sma50']).astype(int)

    d['vol_ratio'] = d['volume'] / d['volume'].rolling(20).mean()
    return d


def run(interval):
    fwd_bars   = 4  if interval == '15m' else 12
    target_pct = 0.001

    print('\n' + '='*62)
    print(f'REVERSAO A MEDIA — {interval}  |  Forward: {fwd_bars} barras (1H)')
    print('='*62)

    tickers = {'mnq': 'MNQ=F', 'es': 'ES=F', 'cl': 'CL=F', 'btc': 'BTC-USD'}
    raws = {}
    for nome, ticker in tickers.items():
        df = get_data(nome, ticker, interval)
        if df is None or (hasattr(df, 'empty') and df.empty):
            print(f'  ERRO: {ticker}'); return
        raws[nome] = df
        src = 'hist' if not load_hist(nome, interval).empty else 'yfinance'
        print(f'  {nome:>4}: {len(df):>6} barras  {df.index[0].date()} -> {df.index[-1].date()}  [{src}]')

    computed = {n: compute(raws[n]) for n in tickers}
    mnq = computed['mnq']; es = computed['es']
    cl  = computed['cl'];  btc = computed['btc']
    mnq_raw = raws['mnq']

    idx = mnq.index.intersection(es.index).intersection(cl.index).intersection(btc.index)
    mnq=mnq.loc[idx]; es=es.loc[idx]; cl=cl.loc[idx]; btc=btc.loc[idx]
    mnq_raw = mnq_raw.reindex(idx, method='ffill')

    # ── Features especificas de reversao ─────────────────────────────────────
    f = pd.DataFrame(index=idx)

    # Niveis RSI (o quanto esta sobrecomprado/sobrevendido)
    f['rsi_mnq']    = mnq['rsi']
    f['rsi_cl']     = cl['rsi']
    f['rsi_es']     = es['rsi']
    f['rsi_btc']    = btc['rsi']
    f['rsi_oversold']    = (f['rsi_mnq'] < 35).astype(int)
    f['rsi_overbought']  = (f['rsi_mnq'] > 65).astype(int)
    f['rsi_extreme']     = ((f['rsi_mnq'] < 30) | (f['rsi_mnq'] > 70)).astype(int)

    # Divergencia RSI multi-ativo (MNQ isolado vs outros)
    f['div_cl']  = f['rsi_mnq'] - f['rsi_cl']
    f['div_es']  = f['rsi_mnq'] - f['rsi_es']
    f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']
    # Se so MNQ esta sobrevendido e ES/CL nao = oportunidade de reversao mais forte
    f['mnq_sozinho_oversold']  = ((f['rsi_mnq'] < 35) & (f['rsi_es'] > 40) & (f['rsi_cl'] > 40)).astype(int)
    f['mnq_sozinho_overbought']= ((f['rsi_mnq'] > 65) & (f['rsi_es'] < 60) & (f['rsi_cl'] < 60)).astype(int)

    # Bollinger Bands — o quanto esta fora das bandas
    f['bb_pctb_mnq']  = mnq['bb_pctb']
    f['bb_pctb_cl']   = cl['bb_pctb']
    f['bb_w_mnq']     = mnq['bb_w']
    f['bb_w_cl']      = cl['bb_w']
    f['abaixo_bb']    = (f['bb_pctb_mnq'] < 0).astype(int)
    f['acima_bb']     = (f['bb_pctb_mnq'] > 1).astype(int)
    f['dist_bb_low']  = mnq['bb_pctb'].clip(upper=0).abs()   # quanto abaixo da BB inferior
    f['dist_bb_upp']  = (mnq['bb_pctb'] - 1).clip(lower=0)   # quanto acima da BB superior

    # Distancia das medias (quanto esta esticado)
    f['dist_ema9']   = mnq['dist_ema9']
    f['dist_ema20']  = mnq['dist_ema20']
    f['dist_sma50']  = mnq['dist_sma50']
    f['dist_ema9_cl']  = cl['dist_ema9']
    f['dist_ema20_cl'] = cl['dist_ema20']

    # Extensao do movimento recente (o quanto caiu/subiu antes da entrada)
    f['ret1_mnq']  = mnq['ret1'] * 100
    f['ret4_mnq']  = mnq['ret4'] * 100
    f['ret8_mnq']  = mnq['ret8'] * 100
    f['ret1_es']   = es['ret1']  * 100
    f['ret4_es']   = es['ret4']  * 100
    f['ret1_cl']   = cl['ret1']  * 100

    # ADX — baixo = lateral = melhor para reversao
    f['adx_mnq']    = mnq['adx']
    f['adx_cl']     = cl['adx']
    f['adx_baixo']  = (mnq['adx'] < 20).astype(int)
    f['adx_medio']  = ((mnq['adx'] >= 20) & (mnq['adx'] < 30)).astype(int)

    # Volume — spike de volume em extremo = possivel reversao
    f['vol_ratio_mnq']  = mnq['vol_ratio']
    f['vol_spike']      = (mnq['vol_ratio'] > 1.5).astype(int)

    # Alinhamento SMA50
    f['sma50_align'] = (mnq['above_sma50'] + cl['above_sma50'] +
                        es['above_sma50']  + btc['above_sma50'])

    # 1H open (contexto de direcao do periodo)
    open_1h = mnq_raw['open'].resample('1h').first().reindex(idx, method='ffill')
    f['dist_1h_open'] = (mnq['close'] - open_1h) / open_1h * 100

    # Tempo
    f['hour']    = idx.hour
    f['dow']     = idx.dayofweek
    f['is_us']   = f['hour'].between(13, 21).astype(int)
    f['is_asia'] = f['hour'].between(0, 8).astype(int)

    # ── Filtro de entrada (sem ML) — condicao de reversao ────────────────────
    # LONG: RSI oversold E abaixo da BB inferior
    f['filtro_long']  = ((f['rsi_mnq'] < 35) & (f['bb_pctb_mnq'] < 0)).astype(int)
    # SHORT: RSI overbought E acima da BB superior
    f['filtro_short'] = ((f['rsi_mnq'] > 65) & (f['bb_pctb_mnq'] > 1)).astype(int)

    # ── Targets ───────────────────────────────────────────────────────────────
    fwd_ret = mnq['close'].shift(-fwd_bars) / mnq['close'] - 1
    f['target_long']  = (fwd_ret >  target_pct).astype(int)
    f['target_short'] = (fwd_ret < -target_pct).astype(int)

    # Rollover filter
    ret_abs = mnq['ret1'].abs() * 100
    f = f.dropna()
    f = f[ret_abs.reindex(f.index) <= 0.8].copy()

    print(f'Dataset: {len(f)} barras  {f.index[0].date()} -> {f.index[-1].date()}')
    baseline_l = float(f['target_long'].mean())
    baseline_s = float(f['target_short'].mean())
    print(f'Baseline LONG={baseline_l:.1%}  SHORT={baseline_s:.1%}')

    # Filtro de entrada sem ML
    n_fl = int(f['filtro_long'].sum())
    n_fs = int(f['filtro_short'].sum())
    if n_fl > 0:
        wr_fl = float(f.loc[f['filtro_long']==1,'target_long'].mean())
        print(f'Filtro RSI+BB sem ML  LONG:  N={n_fl:>5}  WR={wr_fl:.1%}  edge={wr_fl-baseline_l:+.1%}')
    if n_fs > 0:
        wr_fs = float(f.loc[f['filtro_short']==1,'target_short'].mean())
        print(f'Filtro RSI+BB sem ML  SHORT: N={n_fs:>5}  WR={wr_fs:.1%}  edge={wr_fs-baseline_s:+.1%}')

    # ── Split temporal 70/15/15 ───────────────────────────────────────────────
    n       = len(f)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train   = f.iloc[:n_train]
    val     = f.iloc[n_train:n_train+n_val]
    oos     = f.iloc[n_train+n_val:]
    print(f'Split: treino={len(train)} | val={len(val)} | OOS={len(oos)}')
    print(f'OOS: {oos.index[0].date()} -> {oos.index[-1].date()}')

    feat_cols = [c for c in f.columns
                 if c not in ['target_long','target_short','filtro_long','filtro_short']]

    results = {}
    for direction in ['long', 'short']:
        target_col  = f'target_{direction}'
        filtro_col  = f'filtro_{direction}'
        bl_oos      = float(oos[target_col].mean())

        print(f'\n--- {direction.upper()} | baseline OOS={bl_oos:.1%} ---')

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
        print(f'  AUC OOS (todas as barras): {auc_os:.4f}')

        # ML em todas as barras
        print(f'  {"thr":>6}  {"N":>5}  {"WR":>6}  {"Edge":>7}')
        for thr in [0.45, 0.50, 0.55, 0.60, 0.65]:
            m = prob_os >= thr
            nt = int(m.sum())
            if nt < 5: continue
            wr   = float(y_os[m].mean())
            edge = wr - bl_oos
            print(f'  >={thr:.2f}  {nt:>5}  {wr:.1%}  {edge:>+.1%}')

        # ML COM filtro RSI+BB (subset oversold/overbought)
        fmask = oos[filtro_col] == 1
        n_f   = int(fmask.sum())
        if n_f >= 5:
            p_f  = prob_os[fmask.values]
            y_f  = y_os[fmask]
            bl_f = float(y_f.mean())
            print(f'\n  Filtro RSI+BB: N={n_f}  baseline={bl_f:.1%}')
            print(f'  {"thr":>6}  {"N":>5}  {"WR":>6}  {"Edge":>7}')
            for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
                m2 = p_f >= thr
                n2 = int(m2.sum())
                if n2 < 3: continue
                wr2 = float(y_f[m2].mean())
                print(f'  >={thr:.2f}  {n2:>5}  {wr2:.1%}  {wr2-bl_f:>+.1%}')
        else:
            print(f'  Filtro RSI+BB: N={n_f} no OOS (poucos sinais)')

        # Top features por importancia
        imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print(f'\n  Top 8 features:')
        for feat, val2 in imp.head(8).items():
            print(f'    {feat:<30} {val2:.3f}')

        # Salvar modelo
        path = BASE / f'model_reversion_{interval}_{direction}.pkl'
        pickle.dump({
            'model': model, 'features': feat_cols,
            'baseline': float(train[target_col].mean()),
            'interval': interval, 'fwd_bars': fwd_bars,
            'auc_oos': round(auc_os, 4), 'direction': direction,
        }, open(path, 'wb'))
        print(f'  -> {path.name}')
        results[(interval, direction)] = {'auc': auc_os, 'bl': bl_oos}

    return results


if __name__ == '__main__':
    all_res = {}
    for tf in ['15m', '5m']:
        r = run(tf)
        if r:
            all_res.update(r)

    print('\n' + '='*62)
    print('RESUMO FINAL — REVERSAO A MEDIA')
    print('='*62)
    print(f'  {"TF":>4}  {"Dir":>6}  {"AUC OOS":>8}  {"Baseline":>9}')
    print('-'*42)
    for (tf, direction), v in all_res.items():
        flag = '<<' if v['auc'] > 0.56 else ('~' if v['auc'] > 0.53 else '')
        print(f'  {tf:>4}  {direction.upper():>6}  {v["auc"]:.4f}    {v["bl"]:.1%}  {flag}')
    print()
    print('Ref S1 momentum 1H: AUC=0.5613 (LONG)  AUC=0.5506 (SHORT)')
    print('Ref scalper   15m:  AUC=0.5463 (LONG)  AUC=0.5604 (SHORT)')

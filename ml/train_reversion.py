"""
train_reversion.py — Treina modelos ML de reversao a media para TODOS os ativos (15m).

Para cada ativo (MNQ, ES, CL, BTC), treina LONG e SHORT separadamente usando:
  - Gate de entrada: RSI(14) < 35 + close < BB inferior (LONG)
                     RSI(14) > 65 + close > BB superior (SHORT)
  - Features: proprias + outros 3 ativos como contexto + divergencias RSI
  - Split temporal 70/15/15

Modelos salvos: ml/model_rev_15m_{ativo}_{long|short}.pkl  (8 modelos total)

Uso: python ml/train_reversion.py
"""
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd, numpy as np, pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from reversion_utils import TICKERS, INTERVAL, FWD_BARS, TARGET_PCT, RSI_OS, RSI_OB, compute, build_features

BASE     = Path(__file__).parent
DATA_DIR = BASE / 'data_hist'


def load_hist(nome):
    path = DATA_DIR / f'{nome}_{INTERVAL}.csv.gz'
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True, compression='gzip')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def download(ticker, period='59d'):
    df = yf.download(ticker, period=period, interval=INTERVAL,
                     auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()


def get_data(nome, ticker):
    hist = load_hist(nome)
    if not hist.empty:
        print(f'  {nome:>4}: {len(hist):>6} barras  {hist.index[0].date()} -> {hist.index[-1].date()}  [hist]')
        return hist
    df = download(ticker)
    if df is not None:
        print(f'  {nome:>4}: {len(df):>6} barras  {df.index[0].date()} -> {df.index[-1].date()}  [yfinance]')
    return df


def train_asset(primary, all_computed, all_raws, results):
    print(f'\n{"="*62}')
    print(f'ATIVO: {primary.upper()}  ({TICKERS[primary]})')
    print('='*62)

    # Indice comum a todos os ativos
    idx = all_computed['mnq'].index
    for n in TICKERS:
        idx = idx.intersection(all_computed[n].index)

    # Realinha todos ao indice comum
    comp = {n: all_computed[n].loc[idx] for n in TICKERS}
    raws = {n: all_raws[n].reindex(idx, method='ffill') for n in TICKERS}

    f = build_features(primary, comp, raws, idx)

    # Targets baseados no ativo primario
    p       = comp[primary]
    fwd_ret = p['close'].shift(-FWD_BARS) / p['close'] - 1
    f['target_long']  = (fwd_ret >  TARGET_PCT).astype(int)
    f['target_short'] = (fwd_ret < -TARGET_PCT).astype(int)
    f['filtro_long']  = ((f['rsi'] < RSI_OS) & (f['bb_pctb'] < 0)).astype(int)
    f['filtro_short'] = ((f['rsi'] > RSI_OB) & (f['bb_pctb'] > 1)).astype(int)

    # Filtro de rollover (barra de ajuste de contrato)
    ret_abs = p['ret1'].abs() * 100
    f = f.dropna()
    f = f[ret_abs.reindex(f.index) <= 0.8].copy()

    print(f'Dataset: {len(f)} barras  {f.index[0].date()} -> {f.index[-1].date()}')
    bl_l = float(f['target_long'].mean())
    bl_s = float(f['target_short'].mean())
    print(f'Baseline LONG={bl_l:.1%}  SHORT={bl_s:.1%}')

    # Estatistica do filtro RSI+BB sem ML
    for direction, filt_col, tgt_col, bl in [
        ('LONG',  'filtro_long',  'target_long',  bl_l),
        ('SHORT', 'filtro_short', 'target_short', bl_s),
    ]:
        n_f = int(f[filt_col].sum())
        if n_f > 0:
            wr = float(f.loc[f[filt_col]==1, tgt_col].mean())
            print(f'  RSI+BB sem ML  {direction:<5}: N={n_f:>4}  WR={wr:.1%}  edge={wr-bl:>+.1%}')

    # Split temporal 70/15/15
    n       = len(f)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train   = f.iloc[:n_train]
    val     = f.iloc[n_train:n_train+n_val]
    oos     = f.iloc[n_train+n_val:]
    print(f'Split: train={len(train)}  val={len(val)}  OOS={len(oos)}')
    print(f'OOS: {oos.index[0].date()} -> {oos.index[-1].date()}')

    feat_cols = [c for c in f.columns
                 if c not in ['target_long','target_short','filtro_long','filtro_short']]

    for direction in ['long', 'short']:
        tgt_col  = f'target_{direction}'
        filt_col = f'filtro_{direction}'
        bl_oos   = float(oos[tgt_col].mean())

        print(f'\n  [{direction.upper()}]  baseline OOS={bl_oos:.1%}')

        X_tr = train[feat_cols].fillna(0)
        y_tr = train[tgt_col]
        X_vl = val[feat_cols].fillna(0)
        y_vl = val[tgt_col]
        X_os = oos[feat_cols].fillna(0)
        y_os = oos[tgt_col]

        model = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            eval_metric='auc', early_stopping_rounds=40,
            use_label_encoder=False, random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        prob_os = model.predict_proba(X_os)[:, 1]
        auc_os  = roc_auc_score(y_os, prob_os)
        print(f'  AUC OOS (todas as barras): {auc_os:.4f}')

        # Threshold table — todas as barras
        print(f'  {"thr":>6}  {"N":>5}  {"WR":>6}  {"Edge":>7}')
        for thr in [0.45, 0.50, 0.55, 0.60, 0.65]:
            m = prob_os >= thr
            nt = int(m.sum())
            if nt < 3: continue
            wr = float(y_os[m].mean())
            print(f'  >={thr:.2f}  {nt:>5}  {wr:.1%}  {wr-bl_oos:>+.1%}')

        # Threshold table — dentro do filtro RSI+BB
        fmask = oos[filt_col] == 1
        nf = int(fmask.sum())
        if nf >= 3:
            pf  = prob_os[fmask.values]
            yf  = y_os[fmask]
            blf = float(yf.mean())
            print(f'\n  +filtro RSI+BB: N={nf}  baseline={blf:.1%}')
            print(f'  {"thr":>6}  {"N":>5}  {"WR":>6}  {"Edge":>7}')
            for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
                m2 = pf >= thr
                n2 = int(m2.sum())
                if n2 < 2: continue
                wr2 = float(yf[m2].mean())
                print(f'  >={thr:.2f}  {n2:>5}  {wr2:.1%}  {wr2-blf:>+.1%}')
        else:
            print(f'  +filtro RSI+BB: N={nf} no OOS (poucos sinais)')

        # Top 5 features
        imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        top5 = '  '.join(f'{k}:{v:.3f}' for k, v in imp.head(5).items())
        print(f'  Top5: {top5}')

        # Salvar modelo
        path = BASE / f'model_rev_15m_{primary}_{direction}.pkl'
        pickle.dump({
            'model':     model,
            'features':  feat_cols,
            'baseline':  float(train[tgt_col].mean()),
            'baseline_oos': bl_oos,
            'interval':  INTERVAL,
            'fwd_bars':  FWD_BARS,
            'auc_oos':   round(auc_os, 4),
            'direction': direction,
            'asset':     primary,
            'ticker':    TICKERS[primary],
        }, open(path, 'wb'))
        print(f'  -> {path.name}  (AUC={auc_os:.4f})')
        results[(primary, direction)] = {'auc': auc_os, 'bl': bl_oos}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(BASE))

    print('Carregando dados...')
    raws = {}
    for nome, ticker in TICKERS.items():
        df = get_data(nome, ticker)
        if df is None or (hasattr(df, 'empty') and df.empty):
            print(f'ERRO: {ticker} nao disponivel'); exit(1)
        raws[nome] = df

    all_computed = {n: compute(raws[n]) for n in TICKERS}
    all_results  = {}

    for primary in TICKERS:
        train_asset(primary, all_computed, raws, all_results)

    print('\n' + '='*62)
    print('RESUMO FINAL — REVERSAO 15M TODOS OS ATIVOS')
    print('='*62)
    print(f'  {"Ativo":<5}  {"Dir":>6}  {"AUC OOS":>8}  {"Baseline":>9}')
    print('-'*45)
    for (asset, direction), v in all_results.items():
        flag = ' <<' if v['auc'] > 0.56 else (' ~' if v['auc'] > 0.53 else '')
        print(f'  {asset.upper():<5}  {direction.upper():>6}  {v["auc"]:.4f}    {v["bl"]:.1%}{flag}')
    print()
    print('Referencia: MNQ SHORT 15m AUC=0.6504 (backtest anterior)')
    print('Para usar ao vivo: python ml/predict_reversion.py')

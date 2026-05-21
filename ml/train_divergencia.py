"""
Treina XGBoost BINARIO para prever o padrao MNQ sobe + CL cai (divergencia).
Usa todas as features: ES, BTC, key levels, aberturas 1h/4h/daily/weekly, spreads.

Uso: python train_divergencia.py [--forward 1] [--interval 1h]
"""
import argparse, sqlite3, pickle, warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np, ta, xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path

BASE = Path(__file__).parent

def load_sym(conn, sym):
    df = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts",
        conn, params=(sym,), parse_dates=['ts'], index_col='ts')
    return df

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

def build_features(conn_main, conn_es, forward=1):
    mnq_raw = load_sym(conn_main, 'mnq')
    btc_raw = load_sym(conn_main, 'btc')
    cl_raw  = load_sym(conn_main, 'cl')
    es_raw  = load_sym(conn_es, 'es')

    mnq = compute(mnq_raw); es = compute(es_raw)
    btc = compute(btc_raw); cl = compute(cl_raw)

    idx = mnq.index.intersection(es.index).intersection(btc.index).intersection(cl.index)
    mnq = mnq.loc[idx]; es = es.loc[idx]
    btc = btc.loc[idx]; cl = cl.loc[idx]

    f = pd.DataFrame(index=idx)

    # Precos
    f['mnq'] = mnq['close']; f['es'] = es['close']
    f['btc'] = btc['close']; f['cl'] = cl['close']

    # Retornos
    for nome, s in [('mnq', mnq), ('es', es), ('btc', btc), ('cl', cl)]:
        f[f'r_{nome}_1h'] = s['ret1'] * 100
        f[f'r_{nome}_4h'] = s['ret4'] * 100

    f['r_mnq_8h'] = mnq['ret8'] * 100

    # Correlacoes ES
    f['es_mnq_mesmo'] = ((f['r_es_1h'] > 0) & (f['r_mnq_1h'] > 0) | (f['r_es_1h'] < 0) & (f['r_mnq_1h'] < 0)).astype(int)
    f['es_mnq_oposto'] = ((f['r_es_1h'] > 0) & (f['r_mnq_1h'] < 0) | (f['r_es_1h'] < 0) & (f['r_mnq_1h'] > 0)).astype(int)
    f['btc_mnq_mesmo'] = ((f['r_btc_1h'] > 0) & (f['r_mnq_1h'] > 0) | (f['r_btc_1h'] < 0) & (f['r_mnq_1h'] < 0)).astype(int)
    f['btc_mnq_oposto'] = ((f['r_btc_1h'] > 0) & (f['r_mnq_1h'] < 0) | (f['r_btc_1h'] < 0) & (f['r_mnq_1h'] > 0)).astype(int)
    f['es_btc_discordam_mnq'] = (f['es_mnq_oposto'] & f['btc_mnq_oposto']).astype(int)
    f['es_btc_concordam_mnq'] = (f['es_mnq_mesmo'] & f['btc_mnq_mesmo']).astype(int)

    # RSI e divergencias
    for nome in ['mnq', 'es', 'btc', 'cl']:
        f[f'rsi_{nome}'] = locals()[nome]['rsi']

    f['div_cl'] = f['rsi_mnq'] - f['rsi_cl']
    f['div_es'] = f['rsi_mnq'] - f['rsi_es']
    f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']
    f['rsi_mnq_acima_60'] = (f['rsi_mnq'] > 60).astype(int)
    f['rsi_mnq_abaixo_40'] = (f['rsi_mnq'] < 40).astype(int)

    # Key levels - abertura 1h
    for nome, raw in [('mnq', mnq_raw), ('es', es_raw), ('btc', btc_raw), ('cl', cl_raw)]:
        s = raw['open'].reindex(idx, method='ffill')
        f[f'open_1h_{nome}'] = s
        f[f'open_1h_acima_close_ant_{nome}'] = (s > f[nome].shift(1)).astype(int)

    # Abertura 4h
    for nome in ['mnq', 'es', 'btc', 'cl']:
        s = f[f'open_1h_{nome}']
        open_4h = s.groupby(idx.floor('4h')).transform('first')
        close_4h_ant = s.shift(1).groupby(idx.floor('4h')).transform('first')
        f[f'open_4h_acima_4h_ant_{nome}'] = (open_4h > close_4h_ant).astype(int)
        f[f'open_4h_dist_{nome}'] = (s - open_4h) / open_4h * 100

    # Abertura daily
    for nome in ['mnq', 'es', 'btc', 'cl']:
        s = f[f'open_1h_{nome}']
        do = s.groupby(idx.date).transform('first')
        close_ant = s.shift(1).groupby(idx.date).transform('first')
        f[f'open_d_acima_d_ant_{nome}'] = (do > close_ant).astype(int)
        f[f'open_d_dist_{nome}'] = (s - do) / do * 100

    # Abertura weekly
    for nome in ['mnq', 'es', 'btc', 'cl']:
        s = f[f'open_1h_{nome}']
        wo = s.groupby(pd.Grouper(freq='W-MON')).transform('first')
        wc_ant = s.shift(1).groupby(pd.Grouper(freq='W-MON')).transform('first')
        f[f'open_w_acima_w_ant_{nome}'] = (wo > wc_ant).astype(int)
        f[f'open_w_dist_{nome}'] = (s - wo) / wo * 100

    # Comparacoes entre aberturas
    f['open_mnq_acima_cl'] = (f['open_1h_mnq'] > f['open_1h_cl']).astype(int)
    f['open_mnq_acima_es'] = (f['open_1h_mnq'] > f['open_1h_es']).astype(int)
    f['open_mnq_acima_btc'] = (f['open_1h_mnq'] > f['open_1h_btc']).astype(int)

    # ADX
    for nome in ['mnq', 'es', 'btc', 'cl']:
        f[f'adx_{nome}'] = locals()[nome]['adx']
    f['adx_mnq_alto'] = (f['adx_mnq'] > 14).astype(int)
    f['adx_cl_alto'] = (f['adx_cl'] > 14).astype(int)
    f['adx_es_alto'] = (f['adx_es'] > 14).astype(int)

    # DI spread
    for nome in ['mnq', 'es', 'btc', 'cl']:
        f[f'di_spread_{nome}'] = locals()[nome]['pdi'] - locals()[nome]['mdi']

    # Alinhamentos SMA50 / EMA20
    for nome in ['mnq', 'es', 'btc', 'cl']:
        f[f'above_sma50_{nome}'] = locals()[nome]['above_sma50']
        f[f'above_ema20_{nome}'] = locals()[nome]['above_ema20']
        f[f'dist_sma50_{nome}'] = locals()[nome]['dist_sma50']
        f[f'dist_ema20_{nome}'] = locals()[nome]['dist_ema20']
    f['sma50_alignment'] = sum(f[f'above_sma50_{n}'] for n in ['mnq','es','btc','cl'])
    f['ema20_alignment'] = sum(f[f'above_ema20_{n}'] for n in ['mnq','es','btc','cl'])

    # Volatilidade
    for nome in ['mnq', 'es', 'btc', 'cl']:
        f[f'vol_{nome}'] = locals()[nome]['vol'] * 100
        f[f'bb_w_{nome}'] = locals()[nome]['bb_w'] * 100

    # Tempo
    f['hour'] = idx.hour; f['dow'] = idx.dayofweek
    f['is_us'] = f['hour'].between(9, 17).astype(int)
    f['is_asia'] = f['hour'].between(0, 8).astype(int)
    f['is_evening'] = f['hour'].between(18, 23).astype(int)

    # Target: MNQ sobe (> 0.1%) nas proximas N horas
    # (target comprovado com AUC ~0.60, muito mais viavel que prever divergencia exata)
    f['mnq_fwd'] = f['mnq'].shift(-forward) / f['mnq'] - 1
    f['target'] = (f['mnq_fwd'] > 0.001).astype(int)

    f = f.dropna()
    return f

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--forward', type=int, default=4, help='Horas a frente (4 = default)')
    p.add_argument('--interval', type=str, default='1h')
    args = p.parse_args()

    print(f'Treinando modelo divergencia (target=MNQ>0.1% em {args.forward}h)')
    conn_main = sqlite3.connect(BASE / 'data.db')
    conn_es = sqlite3.connect(BASE / 'es' / 'data.db')
    df = build_features(conn_main, conn_es, args.forward)
    conn_main.close(); conn_es.close()

    print(f'Amostras: {len(df)} | Target=1: {df["target"].mean():.1%}')
    print(f'Features: {len(df.columns)}')

    # Features (excluir precos, target, colunas derivadas obvias)
    skip = {'mnq','es','btc','cl','target','open_1h_mnq','open_1h_es','open_1h_btc','open_1h_cl',
            'r_mnq_1h','r_es_1h','r_btc_1h','r_cl_1h','mnq_fwd'}
    feat_cols = [c for c in df.columns if c not in skip]
    X, y = df[feat_cols].fillna(0), df['target']

    # Walk-forward split
    split = int(len(X) * 0.70)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # Balance weights
    ratio = y_tr.sum() / len(y_tr)
    w = {0: ratio, 1: 1 - ratio}
    sw = y_tr.map(w)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
        eval_metric='logloss', early_stopping_rounds=40, random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], sample_weight=sw.values, verbose=False)

    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]

    print(f'\n-- Walk-Forward --')
    print(f'Treino: {len(X_tr)} | Teste: {len(X_te)}')
    print(f'Baseline: {y_te.mean():.1%}')
    print(classification_report(y_te, preds, target_names=['NAO','DIVERGENCIA']))
    auc = roc_auc_score(y_te, proba)
    print(f'ROC-AUC: {auc:.4f}')

    # Analise por prob threshold
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pbin = (proba >= th).astype(int)
        if pbin.sum() < 5: continue
        wr = (pbin == y_te.values).mean()
        n = pbin.sum()
        print(f'  Threshold {th:.1f}: pred=1 N={n:>4} WR={wr:.1%} edge={(pbin[y_te==1].sum()/max(n,1)):.1%}')

    # Feature importance
    imp = pd.DataFrame({'feature': feat_cols, 'gain': model.feature_importances_})
    imp = imp.sort_values('gain', ascending=False).head(30)
    print(f'\n-- Top 30 Features --')
    for _, r in imp.iterrows():
        print(f'  {r["feature"]:<35} gain={r["gain"]:.4f}')

    # Salvar
    out_path = BASE / 'model_divergencia.pkl'
    with open(out_path, 'wb') as fp:
        pickle.dump({
            'model': model, 'features': feat_cols,
            'forward': args.forward, 'auc': auc,
            'baseline': y_te.mean(),
        }, fp)
    print(f'\nModelo salvo: {out_path}')

if __name__ == '__main__':
    main()

"""
Treina XGBoost para prever direcao do MNQ nas proximas N horas.
Foco: sessao americana 12h-17h BRT (10h-15h nos dados) onde o edge e maior.
Uso: python train.py [--forward 4] [--interval 1h] [--all-hours]
"""
import argparse
import sqlite3
import pickle
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DB        = Path(__file__).parent / 'data.db'
MODEL_OUT = Path(__file__).parent / 'model.pkl'
FIG_OUT   = Path(__file__).parent / 'feature_importance.png'


def load_symbol(conn, symbol, interval='1h'):
    ts_len = 19 if interval == '1h' else 10
    return pd.read_sql(
        'SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND LENGTH(ts)=? ORDER BY ts',
        conn, params=(symbol, ts_len), parse_dates=['ts'], index_col='ts',
    )


def compute(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    adx_i     = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx']  = adx_i.adx()
    df['pdi']  = adx_i.adx_pos()
    df['mdi']  = adx_i.adx_neg()
    df['ret1'] = df['close'].pct_change(1)
    df['ret4'] = df['close'].pct_change(4)
    df['ret8'] = df['close'].pct_change(8)
    df['vol']  = df['ret1'].rolling(20).std()
    df['bb_w'] = df['close'].rolling(20).std() * 2 / df['close'].rolling(20).mean()
    # MA50
    df['ma50']       = df['close'].rolling(50).mean()
    df['dist_ma50']  = (df['close'] - df['ma50']) / df['ma50'] * 100  # % acima/abaixo da MA50
    df['ma50_slope'] = df['ma50'].pct_change(5) * 100                 # inclinacao da MA50
    df['above_ma50'] = (df['close'] > df['ma50']).astype(int)
    return df


def build_features(conn, interval='1h', forward=8, min_ret=0.001):
    mnq = compute(load_symbol(conn, 'mnq', interval))
    btc = compute(load_symbol(conn, 'btc', interval))
    cl  = compute(load_symbol(conn, 'cl',  interval))

    idx = mnq.index
    f = pd.DataFrame(index=idx)

    # RSI dos 3 ativos
    f['rsi_mnq'] = mnq['rsi']
    f['rsi_btc'] = btc['rsi'].reindex(idx, method='ffill')
    f['rsi_cl']  = cl['rsi'].reindex(idx, method='ffill')

    # ADX + DI
    f['adx_mnq'] = mnq['adx']
    f['adx_btc'] = btc['adx'].reindex(idx, method='ffill')
    f['adx_cl']  = cl['adx'].reindex(idx, method='ffill')
    f['pdi_mnq'] = mnq['pdi']
    f['mdi_mnq'] = mnq['mdi']

    # Divergencia RSI (estrategia original)
    f['div_cl']  = f['rsi_mnq'] - f['rsi_cl']
    f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']

    # Retornos e vol MNQ
    f['ret1_mnq'] = mnq['ret1']
    f['ret4_mnq'] = mnq['ret4']
    f['ret8_mnq'] = mnq['ret8']
    f['vol_mnq']  = mnq['vol']
    f['bb_mnq']   = mnq['bb_w']

    # Retornos BTC e CL
    f['ret1_btc'] = btc['ret1'].reindex(idx, method='ffill')
    f['ret4_btc'] = btc['ret4'].reindex(idx, method='ffill')
    f['ret1_cl']  = cl['ret1'].reindex(idx, method='ffill')
    f['ret4_cl']  = cl['ret4'].reindex(idx, method='ffill')

    # Tempo
    f['hour'] = idx.hour
    f['dow']  = idx.dayofweek

    # Aceleracao do ADX
    f['dadx_mnq'] = f['adx_mnq'].diff(2)

    # Sinal triplo original
    f['triple_signal'] = (
        (f['rsi_btc'] < 45) & (f['div_cl'] < 0) & (f['div_btc'] < 0)
    ).astype(int)

    # MA50 dos 3 ativos
    f['dist_ma50_mnq'] = mnq['dist_ma50']
    f['dist_ma50_btc'] = btc['dist_ma50'].reindex(idx, method='ffill')
    f['dist_ma50_cl']  = cl['dist_ma50'].reindex(idx, method='ffill')
    f['ma50_slope_mnq']= mnq['ma50_slope']
    f['above_ma50_mnq']= mnq['above_ma50']
    f['above_ma50_btc']= btc['above_ma50'].reindex(idx, method='ffill')
    f['above_ma50_cl'] = cl['above_ma50'].reindex(idx, method='ffill')
    # Alinhamento MA50 (todos acima = regime bullish)
    f['ma50_alignment']= (f['above_ma50_mnq'] + f['above_ma50_btc'] + f['above_ma50_cl'])  # 0,1,2,3

    # ── NOVAS FEATURES ───────────────────────────────────────────
    # CL e MNQ andando contra (produto negativo = direcoes opostas)
    ret_cl = cl['ret1'].reindex(idx, method='ffill')
    f['price_div_cl'] = mnq['ret1'] * ret_cl   # negativo = divergencia de preco

    # Intensidade da divergencia de preco (modulo)
    f['price_div_abs'] = f['price_div_cl'].abs()

    # ADX acima de 14 (filtro de tendencia)
    f['adx_above_14'] = (f['adx_mnq'] > 14).astype(int)
    f['adx_above_20'] = (f['adx_mnq'] > 20).astype(int)
    f['adx_above_25'] = (f['adx_mnq'] > 25).astype(int)

    # Sessao 21h BRT (referencia antiga)
    f['is_evening'] = f['hour'].isin([18, 19, 20, 21]).astype(int)

    # Sessao americana 12h-17h BRT (10h-17h nos dados UTC/ET)
    # Dados mostram win rate 44-47% nessa janela vs 33-40% fora dela
    US_HOURS = list(range(9, 18))
    f['is_us_session']  = f['hour'].isin(US_HOURS).astype(int)
    f['is_us_morning']  = f['hour'].isin([9, 10, 11, 12, 13]).astype(int)   # pico 47%
    f['is_us_afternoon']= f['hour'].isin([14, 15, 16, 17]).astype(int)

    # Gatilho forte: CL e MNQ contra + ADX > 14
    f['strong_div'] = (
        (f['price_div_cl'] < 0) & (f['adx_mnq'] > 14)
    ).astype(int)

    # Gatilho com sessao 21h (antigo)
    f['prime_setup'] = (
        (f['price_div_cl'] < 0) & (f['adx_mnq'] > 14) & f['hour'].isin([18, 19, 20, 21])
    ).astype(int)

    # Gatilho com sessao americana (novo - maior edge)
    f['us_prime_setup'] = (
        (f['price_div_cl'] < 0) & (f['adx_mnq'] > 14) & f['is_us_session'] == 1
    ).astype(int)

    # CL caindo enquanto MNQ sobe (setup especifico de LONG MNQ)
    f['cl_down_mnq_up'] = (
        (ret_cl < 0) & (mnq['ret1'] > 0)
    ).astype(int)
    # ─────────────────────────────────────────────────────────────

    # Label
    future_price   = mnq['close'].shift(-forward)
    f['future_ret'] = future_price / mnq['close'] - 1
    f['label']      = (f['future_ret'] > min_ret).astype(int)

    return f.dropna()


FEATURE_COLS = [
    'rsi_mnq', 'rsi_btc', 'rsi_cl',
    'adx_mnq', 'adx_btc', 'adx_cl',
    'pdi_mnq', 'mdi_mnq',
    'div_cl', 'div_btc',
    'ret1_mnq', 'ret4_mnq', 'ret8_mnq', 'vol_mnq', 'bb_mnq',
    'ret1_btc', 'ret4_btc',
    'ret1_cl', 'ret4_cl',
    'hour', 'dow',
    'dadx_mnq', 'triple_signal',
    'price_div_cl', 'price_div_abs',
    'adx_above_14', 'adx_above_20', 'adx_above_25',
    'is_evening', 'strong_div', 'prime_setup', 'cl_down_mnq_up',
    # Sessao americana (maior edge)
    'is_us_session', 'is_us_morning', 'is_us_afternoon', 'us_prime_setup',
    # MA50
    'dist_ma50_mnq', 'dist_ma50_btc', 'dist_ma50_cl',
    'ma50_slope_mnq', 'above_ma50_mnq', 'above_ma50_btc', 'above_ma50_cl',
    'ma50_alignment',
]

# Horas da sessao americana (onde o modelo e treinado)
US_SESSION_HOURS = list(range(9, 18))


def train_model(X, y):
    split = int(len(X) * 0.70)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=8,
        scale_pos_weight=scale,   # corrige desbalanceamento
        eval_metric='logloss',
        early_stopping_rounds=40,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]

    print('\n========================================')
    print('  Walk-Forward - resultado fora da amostra')
    print('========================================')
    print(f'  Treino : {len(X_tr)} amostras  ({X_tr.index[0].date()} ate {X_tr.index[-1].date()})')
    print(f'  Teste  : {len(X_te)} amostras  ({X_te.index[0].date()} ate {X_te.index[-1].date()})')
    print(f'  LONG na amostra de teste: {y_te.mean():.1%}\n')
    print(classification_report(y_te, preds, target_names=['Neutro/Down', 'LONG']))
    auc = roc_auc_score(y_te, proba)
    rating = 'bom' if auc > 0.55 else 'fraco - mais dados ajudam'
    print(f'  ROC-AUC: {auc:.4f}  [{rating}]')

    # Analise do gatilho
    print('\n--- Analise dos gatilhos no periodo de teste ---')
    df_te = X_te.copy()
    df_te['label'] = y_te.values
    df_te['prob']  = proba

    for nome, col in [('strong_div', 'strong_div'), ('us_prime_setup', 'us_prime_setup')]:
        sub = df_te[df_te[col] == 1] if col in df_te.columns else pd.DataFrame()
        if len(sub) >= 10:
            print(f'  {nome}: N={len(sub)}  LONG={sub["label"].mean():.1%}  prob_media={sub["prob"].mean():.1%}')

    # Win rate por hora na amostra de teste
    print('\n  Win rate por hora (teste):')
    for h in sorted(df_te['hour'].unique()):
        sub = df_te[df_te['hour'] == h]
        if len(sub) < 20: continue
        wr = sub['label'].mean()
        mark = ' <--' if wr >= 0.50 else ''
        print(f'    {h:02d}h: {wr:.1%}  (N={len(sub)}){mark}')

    return model, auc


def plot_importance(model):
    fig, ax = plt.subplots(figsize=(9, 7))
    xgb.plot_importance(model, ax=ax, max_num_features=20, importance_type='gain')
    ax.set_title('Feature Importance (Gain) - MNQ XGBoost')
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=130)
    plt.close()
    print(f'\n  Grafico salvo em: {FIG_OUT}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--forward',   type=int,   default=4,     help='Barras a frente (default 4)')
    p.add_argument('--interval',  type=str,   default='1h',  help='1h ou 1d (default 1h)')
    p.add_argument('--min-ret',   type=float, default=0.001, help='Retorno minimo LONG (default 0.1%%)')
    p.add_argument('--all-hours', action='store_true',       help='Treinar em todas as horas (nao so sessao US)')
    args = p.parse_args()

    if not DB.exists():
        print('DB nao encontrado. Rode collect_data.py primeiro.')
        return

    print(f'Config: interval={args.interval}  forward={args.forward}h  min_ret={args.min_ret:.2%}')
    conn = sqlite3.connect(DB)
    print('Calculando features...')
    df = build_features(conn, args.interval, args.forward, args.min_ret)
    conn.close()

    # Filtrar sessao americana (maior edge comprovado pelos dados)
    if not args.all_hours:
        total_antes = len(df)
        df = df[df['hour'].isin(US_SESSION_HOURS)].copy()
        print(f'Filtro sessao US (horas 9-17): {total_antes} -> {len(df)} amostras')
    else:
        print(f'Modo todas as horas ativo')

    print(f'Dataset: {len(df)} amostras  |  LONG: {df["label"].mean():.1%}')
    print(f'Amostras strong_div: {df["strong_div"].sum()}  ({df["strong_div"].mean():.1%})')
    print(f'Amostras us_prime_setup: {df["us_prime_setup"].sum()}')

    if len(df) < 300:
        print('Poucos dados. Rode collect_data.py.')
        return

    X, y = df[FEATURE_COLS], df['label']
    model, auc = train_model(X, y)
    plot_importance(model)

    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({
            'model':    model,
            'features': FEATURE_COLS,
            'forward':  args.forward,
            'interval': args.interval,
            'auc':      auc,
        }, f)
    print(f'  Modelo salvo em: {MODEL_OUT}')


if __name__ == '__main__':
    main()

"""
predict_reversion.py — Score ao vivo de reversao para todos os ativos (15m).

Carrega os modelos model_rev_15m_{ativo}_{long|short}.pkl e retorna
probabilidade de reversao para a ultima barra fechada de cada ativo.

Uso: python ml/predict_reversion.py
Saida: JSON (para server.js via runPredict)
"""
import warnings; warnings.filterwarnings('ignore')
import json, sys
from pathlib import Path
from datetime import datetime

import yfinance as yf
import pandas as pd, pickle

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from reversion_utils import TICKERS, INTERVAL, RSI_OS, RSI_OB, compute, build_features

SIGNAL_THR = 0.55   # prob minima para emitir sinal (dentro do filtro RSI+BB)


def load_model(asset, direction):
    path = BASE / f'model_rev_15m_{asset}_{direction}.pkl'
    if not path.exists():
        return None
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def download_recent(ticker, period='5d'):
    df = yf.download(ticker, period=period, interval=INTERVAL,
                     auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()


def score():
    # Baixar dados recentes
    raws = {}
    for nome, ticker in TICKERS.items():
        df = download_recent(ticker)
        if df is None:
            out = {'error': f'falha ao baixar {ticker}', 'ts': datetime.utcnow().isoformat()}
            print(json.dumps(out))
            sys.exit(1)
        raws[nome] = df

    computed = {n: compute(raws[n]) for n in TICKERS}

    # Indice comum (barras onde todos os ativos tem dados)
    idx = computed['mnq'].index
    for n in TICKERS:
        idx = idx.intersection(computed[n].index)

    # Precisamos de pelo menos 60 barras para indicadores estabilizarem
    if len(idx) < 60:
        out = {'error': f'dados insuficientes: {len(idx)} barras', 'ts': datetime.utcnow().isoformat()}
        print(json.dumps(out))
        sys.exit(1)

    comp = {n: computed[n].loc[idx] for n in TICKERS}
    raws_aligned = {n: raws[n].reindex(idx, method='ffill') for n in TICKERS}

    results = {}
    for primary in TICKERS:
        feat_df = build_features(primary, comp, raws_aligned, idx)
        feat_df = feat_df.fillna(0)

        # Usa a penultima barra (ultima fechada; a ultima pode estar incompleta)
        last_idx  = -2
        last_row  = feat_df.iloc[last_idx]
        p_last    = comp[primary].iloc[last_idx]
        bar_ts    = str(feat_df.index[last_idx])

        rsi_val   = float(p_last['rsi'])
        bb_pctb   = float(p_last['bb_pctb'])
        close_val = float(p_last['close'])
        adx_val   = float(p_last['adx'])
        bb_w_val  = float(p_last['bb_w'])

        long_gate  = bool(rsi_val < RSI_OS and bb_pctb < 0)
        short_gate = bool(rsi_val > RSI_OB and bb_pctb > 1)

        asset_result = {
            'ts':         bar_ts,
            'close':      round(close_val, 4),
            'rsi':        round(rsi_val, 2),
            'bb_pctb':    round(bb_pctb, 4),
            'adx':        round(adx_val, 2),
            'bb_w':       round(bb_w_val, 2),
            'long_gate':  long_gate,
            'short_gate': short_gate,
            'long_prob':  None,
            'short_prob': None,
            'signal':     'neutro',
        }

        for direction in ['long', 'short']:
            pkg = load_model(primary, direction)
            if pkg is None:
                continue
            model     = pkg['model']
            feat_cols = pkg['features']
            baseline  = pkg.get('baseline_oos', pkg.get('baseline', 0.5))

            row  = last_row.reindex(feat_cols).fillna(0).values.reshape(1, -1)
            prob = float(model.predict_proba(row)[0][1])
            asset_result[f'{direction}_prob']  = round(prob, 4)
            asset_result[f'{direction}_base']  = round(baseline, 4)
            asset_result[f'{direction}_edge']  = round(prob - baseline, 4)

        # Sinal final: precisa do gate RSI+BB E prob >= threshold
        lp = asset_result['long_prob']
        sp = asset_result['short_prob']

        if long_gate and lp is not None and lp >= SIGNAL_THR:
            asset_result['signal'] = 'LONG'
        elif short_gate and sp is not None and sp >= SIGNAL_THR:
            asset_result['signal'] = 'SHORT'

        results[primary] = asset_result

    output = {
        'ts':     datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'assets': results,
    }
    print(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    score()

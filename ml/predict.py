"""
Busca dados atuais, computa features e retorna predicao ML.
Chamado pelo server.js via child_process. Saida: JSON no stdout.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import ta
import pickle
from pathlib import Path
from datetime import datetime

MODEL_PATH = Path(__file__).parent / 'model.pkl'
SYMS = {'mnq': 'MNQ=F', 'btc': 'BTC-USD', 'cl': 'CL=F'}


def fetch(ticker):
    df = yf.download(ticker, period='5d', interval='1h',
                     auto_adjust=True, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])


def compute(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    adx_i     = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_i.adx()
    df['pdi'] = adx_i.adx_pos()
    df['mdi'] = adx_i.adx_neg()
    df['ret1'] = df['close'].pct_change(1)
    df['ret4'] = df['close'].pct_change(4)
    df['ret8'] = df['close'].pct_change(8)
    df['vol']  = df['ret1'].rolling(20).std()
    df['bb_w'] = df['close'].rolling(20).std() * 2 / df['close'].rolling(20).mean()
    df['ma50']       = df['close'].rolling(50).mean()
    df['dist_ma50']  = (df['close'] - df['ma50']) / df['ma50'] * 100
    df['ma50_slope'] = df['ma50'].pct_change(5) * 100
    df['above_ma50'] = (df['close'] > df['ma50']).astype(int)
    return df


def last(df, col):
    v = df[col].iloc[-1]
    return float(v) if pd.notna(v) else None


def main():
    if not MODEL_PATH.exists():
        print(json.dumps({'error': 'model.pkl nao encontrado. Rode train.py primeiro.'}))
        return

    try:
        mnq = compute(fetch('MNQ=F'))
        btc = compute(fetch('BTC-USD'))
        cl  = compute(fetch('CL=F'))

        if mnq is None or cl is None or btc is None:
            print(json.dumps({'error': 'Falha ao buscar dados do Yahoo Finance'}))
            return

        # Monta vetor de features (mesmo que train.py)
        r = {}
        r['rsi_mnq']      = last(mnq, 'rsi')
        r['rsi_btc']      = last(btc, 'rsi')
        r['rsi_cl']       = last(cl,  'rsi')
        r['adx_mnq']      = last(mnq, 'adx')
        r['adx_btc']      = last(btc, 'adx')
        r['adx_cl']       = last(cl,  'adx')
        r['pdi_mnq']      = last(mnq, 'pdi')
        r['mdi_mnq']      = last(mnq, 'mdi')
        r['div_cl']       = r['rsi_mnq'] - r['rsi_cl']
        r['div_btc']      = r['rsi_mnq'] - r['rsi_btc']
        r['ret1_mnq']     = last(mnq, 'ret1')
        r['ret4_mnq']     = last(mnq, 'ret4')
        r['ret8_mnq']     = last(mnq, 'ret8')
        r['vol_mnq']      = last(mnq, 'vol')
        r['bb_mnq']       = last(mnq, 'bb_w')
        r['ret1_btc']     = last(btc, 'ret1')
        r['ret4_btc']     = last(btc, 'ret4')
        r['ret1_cl']      = last(cl,  'ret1')
        r['ret4_cl']      = last(cl,  'ret4')
        r['hour']         = int(mnq.index[-1].hour)
        r['dow']          = int(mnq.index[-1].dayofweek)
        r['dadx_mnq']      = (last(mnq, 'adx') - float(mnq['adx'].iloc[-3])) if len(mnq) >= 3 else 0.0
        r['dist_ma50_mnq'] = last(mnq, 'dist_ma50') or 0.0
        r['dist_ma50_btc'] = last(btc, 'dist_ma50') or 0.0
        r['dist_ma50_cl']  = last(cl,  'dist_ma50') or 0.0
        r['ma50_slope_mnq']= last(mnq, 'ma50_slope') or 0.0
        r['above_ma50_mnq']= int((last(mnq, 'above_ma50') or 0))
        r['above_ma50_btc']= int((last(btc, 'above_ma50') or 0))
        r['above_ma50_cl'] = int((last(cl,  'above_ma50') or 0))
        r['ma50_alignment']= r['above_ma50_mnq'] + r['above_ma50_btc'] + r['above_ma50_cl']
        r['triple_signal'] = int(
            (r['rsi_btc'] or 100) < 45 and (r['div_cl'] or 0) < 0 and (r['div_btc'] or 0) < 0
        )

        ret1_cl_v  = r['ret1_cl'] or 0
        ret1_mnq_v = r['ret1_mnq'] or 0
        adx_v      = r['adx_mnq'] or 0

        r['price_div_cl']  = ret1_mnq_v * ret1_cl_v
        r['price_div_abs'] = abs(r['price_div_cl'])
        r['adx_above_14']  = int(adx_v > 14)
        r['adx_above_20']  = int(adx_v > 20)
        r['adx_above_25']  = int(adx_v > 25)
        r['is_evening']    = int(r['hour'] in [18, 19, 20, 21])
        r['strong_div']    = int(r['price_div_cl'] < 0 and adx_v > 14)
        r['prime_setup']   = int(r['price_div_cl'] < 0 and adx_v > 14 and r['hour'] in [18, 19, 20, 21])
        r['cl_down_mnq_up'] = int(ret1_cl_v < 0 and ret1_mnq_v > 0)
        us_hours = list(range(9, 18))
        r['is_us_session']   = int(r['hour'] in us_hours)
        r['is_us_morning']   = int(r['hour'] in [9, 10, 11, 12, 13])
        r['is_us_afternoon'] = int(r['hour'] in [14, 15, 16, 17])
        r['us_prime_setup']  = int(r['price_div_cl'] < 0 and adx_v > 14 and r['hour'] in us_hours)

        model_data = pickle.load(open(MODEL_PATH, 'rb'))
        model      = model_data['model']
        feat_cols  = model_data['features']

        X    = pd.DataFrame([r])[feat_cols]
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(model.predict(X)[0])

        # Contexto legivel para o dashboard
        cl_dir  = 'BAIXO' if ret1_cl_v < 0 else 'CIMA'
        mnq_dir = 'CIMA'  if ret1_mnq_v > 0 else 'BAIXO'

        output = {
            'prob_long':     round(prob, 4),
            'signal':        pred,
            'adx_mnq':       round(adx_v, 2),
            'adx_active':    bool(adx_v > 14),
            'price_div_cl':  round(r['price_div_cl'], 6),
            'cl_dir':        cl_dir,
            'mnq_dir':       mnq_dir,
            'moving_against':  bool(r['price_div_cl'] < 0),
            'strong_div':      bool(r['strong_div']),
            'prime_setup':     bool(r['prime_setup']),
            'us_prime_setup':  bool(r['us_prime_setup']),
            'is_evening':      bool(r['is_evening']),
            'is_us_session':   bool(r['is_us_session']),
            'is_us_morning':   bool(r['is_us_morning']),
            'hour':            r['hour'],
            'model_forward': model_data.get('forward', 8),
            'model_auc':     round(model_data.get('auc', 0), 4),
            'ts':            datetime.now().isoformat(),
            'dist_ma50_mnq': round(r['dist_ma50_mnq'], 3),
            'above_ma50_mnq': bool(r['above_ma50_mnq']),
            'ma50_alignment': r['ma50_alignment'],
        }
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({'error': str(e)}))


if __name__ == '__main__':
    main()

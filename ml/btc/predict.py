"""
Busca dados atuais, computa features e retorna predicao ML para BTC.
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
SYMS = {'btc': 'BTC-USD', 'mnq': 'MNQ=F', 'cl': 'CL=F'}


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
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
    adx_i     = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=17)
    df['adx'] = adx_i.adx()
    df['pdi'] = adx_i.adx_pos()
    df['mdi'] = adx_i.adx_neg()
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


def last(df, col):
    v = df[col].iloc[-1]
    return float(v) if pd.notna(v) else None


def main():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        print(json.dumps({'error': 'model.pkl nao encontrado ou vazio. Rode train.py primeiro.'}))
        return

    try:
        btc_df = fetch('BTC-USD')
        mnq_df = fetch('MNQ=F')
        cl_df  = fetch('CL=F')

        if btc_df is None or mnq_df is None or cl_df is None:
            print(json.dumps({'error': 'Falha ao buscar dados do Yahoo Finance'}))
            return

        btc = compute(btc_df)
        mnq = compute(mnq_df)
        cl  = compute(cl_df)

        r = {}
        r['rsi_mnq']      = last(btc, 'rsi')
        r['rsi_btc']      = last(mnq, 'rsi')
        r['rsi_cl']       = last(cl,  'rsi')
        r['adx_mnq']      = last(btc, 'adx')
        r['adx_btc']      = last(mnq, 'adx')
        r['adx_cl']       = last(cl,  'adx')
        r['pdi_mnq']      = last(btc, 'pdi')
        r['mdi_mnq']      = last(btc, 'mdi')
        r['div_cl']       = r['rsi_mnq'] - r['rsi_cl']
        r['div_btc']      = r['rsi_mnq'] - r['rsi_btc']
        r['ret1_mnq']     = last(btc, 'ret1')
        r['ret4_mnq']     = last(btc, 'ret4')
        r['ret8_mnq']     = last(btc, 'ret8')
        r['vol_mnq']      = last(btc, 'vol')
        r['bb_mnq']       = last(btc, 'bb_w')
        r['ret1_btc']     = last(mnq, 'ret1')
        r['ret4_btc']     = last(mnq, 'ret4')
        r['ret1_cl']      = last(cl,  'ret1')
        r['ret4_cl']      = last(cl,  'ret4')
        r['hour']         = int(btc.index[-1].hour)
        r['dow']          = int(btc.index[-1].dayofweek)
        r['dadx_mnq']     = (last(btc, 'adx') - float(btc['adx'].iloc[-3])) if len(btc) >= 3 else 0.0
        r['dist_sma50_mnq'] = last(btc, 'dist_sma50') or 0.0
        r['dist_sma50_btc'] = last(mnq, 'dist_sma50') or 0.0
        r['dist_sma50_cl']  = last(cl,  'dist_sma50') or 0.0
        r['sma50_slope_mnq']= last(btc, 'sma50_slope') or 0.0
        r['above_sma50_mnq']= int((last(btc, 'above_sma50') or 0))
        r['above_sma50_btc']= int((last(mnq, 'above_sma50') or 0))
        r['above_sma50_cl'] = int((last(cl,  'above_sma50') or 0))
        r['sma50_alignment']= r['above_sma50_mnq'] + r['above_sma50_btc'] + r['above_sma50_cl']
        r['dist_ema20_mnq'] = last(btc, 'dist_ema20') or 0.0
        r['dist_ema20_btc'] = last(mnq, 'dist_ema20') or 0.0
        r['dist_ema20_cl']  = last(cl,  'dist_ema20') or 0.0
        r['above_ema20_mnq']= int((last(btc, 'above_ema20') or 0))
        r['above_ema20_btc']= int((last(mnq, 'above_ema20') or 0))
        r['above_ema20_cl'] = int((last(cl,  'above_ema20') or 0))
        r['ema20_bias_mnq_btc'] = r['above_ema20_mnq'] + r['above_ema20_btc']
        r['ema20_alignment'] = r['above_ema20_mnq'] + r['above_ema20_btc'] + r['above_ema20_cl']
        ret1_cl_v  = r['ret1_cl'] or 0
        ret1_mnq_v = r['ret1_mnq'] or 0
        ret1_btc_v = r['ret1_btc'] or 0
        adx_v      = r['adx_mnq'] or 0

        r['triple_signal'] = int(
            (r['rsi_btc'] or 100) < 45 and (r['div_cl'] or 0) < 0 and (r['div_btc'] or 0) < 0
        )

        # Spreads completos
        r['rsi_spread_btc_cl']   = (r['rsi_btc'] or 50) - (r['rsi_cl'] or 50)
        r['rsi_abs_mnq_btc']     = abs(r['div_btc'] or 0)
        r['rsi_abs_mnq_cl']      = abs(r['div_cl'] or 0)
        r['rsi_abs_btc_cl']      = abs(r['rsi_spread_btc_cl'])

        r['adx_spread_mnq_btc']  = (r['adx_mnq'] or 0) - (r['adx_btc'] or 0)
        r['adx_spread_mnq_cl']   = (r['adx_mnq'] or 0) - (r['adx_cl'] or 0)
        r['adx_spread_btc_cl']   = (r['adx_btc'] or 0) - (r['adx_cl'] or 0)
        r['adx_abs_mnq_btc']     = abs(r['adx_spread_mnq_btc'])
        r['adx_abs_mnq_cl']      = abs(r['adx_spread_mnq_cl'])
        r['adx_abs_btc_cl']      = abs(r['adx_spread_btc_cl'])

        r['ret1_spread_mnq_btc'] = ret1_mnq_v - ret1_btc_v
        r['ret1_spread_mnq_cl']  = ret1_mnq_v - ret1_cl_v
        r['ret1_spread_btc_cl']  = ret1_btc_v - ret1_cl_v
        ret4_mnq_v = r['ret4_mnq'] or 0
        ret4_btc_v = r['ret4_btc'] or 0
        ret4_cl_v  = r['ret4_cl'] or 0
        r['ret4_spread_mnq_btc'] = ret4_mnq_v - ret4_btc_v
        r['ret4_spread_mnq_cl']  = ret4_mnq_v - ret4_cl_v
        r['ret4_spread_btc_cl']  = ret4_btc_v - ret4_cl_v

        r['ret1_prod_mnq_btc'] = ret1_mnq_v * ret1_btc_v
        r['ret1_prod_btc_cl']  = ret1_btc_v * ret1_cl_v
        r['price_div_cl']      = ret1_mnq_v * ret1_cl_v
        r['price_div_abs']     = abs(r['price_div_cl'])
        r['ret4_prod_mnq_btc'] = ret4_mnq_v * ret4_btc_v
        r['ret4_prod_mnq_cl']  = ret4_mnq_v * ret4_cl_v
        r['ret4_prod_btc_cl']  = ret4_btc_v * ret4_cl_v

        vol_mnq_v = r['vol_mnq'] or 0
        vol_btc_v = last(mnq, 'vol') or 0
        vol_cl_v  = last(cl, 'vol') or 0
        r['vol_spread_mnq_btc'] = vol_mnq_v - vol_btc_v
        r['vol_spread_mnq_cl']  = vol_mnq_v - vol_cl_v
        r['vol_spread_btc_cl']  = vol_btc_v - vol_cl_v

        bb_mnq_v = r['bb_mnq'] or 0
        bb_btc_v = last(mnq, 'bb_w') or 0
        bb_cl_v  = last(cl,  'bb_w') or 0
        r['bb_spread_mnq_btc'] = bb_mnq_v - bb_btc_v
        r['bb_spread_mnq_cl']  = bb_mnq_v - bb_cl_v
        r['bb_spread_btc_cl']  = bb_btc_v - bb_cl_v

        m50_mnq = r['dist_sma50_mnq']
        m50_btc = r['dist_sma50_btc']
        m50_cl  = r['dist_sma50_cl']
        r['sma50_dist_spread_mnq_btc'] = m50_mnq - m50_btc
        r['sma50_dist_spread_mnq_cl']  = m50_mnq - m50_cl
        r['sma50_dist_spread_btc_cl']  = m50_btc - m50_cl

        e50_mnq = r['dist_ema20_mnq']
        e50_btc = r['dist_ema20_btc']
        e50_cl  = r['dist_ema20_cl']
        r['ema20_dist_spread_mnq_btc'] = e50_mnq - e50_btc
        r['ema20_dist_spread_mnq_cl']  = e50_mnq - e50_cl
        r['ema20_dist_spread_btc_cl']  = e50_btc - e50_cl

        am_mnq = r['above_sma50_mnq']
        am_btc = r['above_sma50_btc']
        am_cl  = r['above_sma50_cl']
        r['sma50_align_mnq_btc'] = am_mnq + am_btc
        r['sma50_align_mnq_cl']  = am_mnq + am_cl
        r['sma50_align_btc_cl']  = am_btc + am_cl

        ae_mnq = r['above_ema20_mnq']
        ae_btc = r['above_ema20_btc']
        ae_cl  = r['above_ema20_cl']
        r['ema20_align_mnq_btc'] = ae_mnq + ae_btc
        r['ema20_align_mnq_cl']  = ae_mnq + ae_cl
        r['ema20_align_btc_cl']  = ae_btc + ae_cl

        pdi_mnq_v = r['pdi_mnq'] or 0
        mdi_mnq_v = r['mdi_mnq'] or 0
        pdi_btc_v = last(mnq, 'pdi') or 0
        mdi_btc_v = last(mnq, 'mdi') or 0
        pdi_cl_v  = last(cl, 'pdi') or 0
        mdi_cl_v  = last(cl, 'mdi') or 0
        r['di_spread_mnq'] = pdi_mnq_v - mdi_mnq_v
        r['di_spread_btc'] = pdi_btc_v - mdi_btc_v
        r['di_spread_cl']  = pdi_cl_v - mdi_cl_v

        # Composite triggers
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
        proba = model.predict_proba(X)[0]
        pred  = int(model.predict(X)[0])
        prob_short = float(proba[0])
        prob_long  = float(proba[2])

        cl_dir  = 'BAIXO' if ret1_cl_v < 0 else 'CIMA'
        mnq_dir = 'CIMA'  if ret1_mnq_v > 0 else 'BAIXO'
        signal_map = {0: 'SHORT', 1: 'NEUTRO', 2: 'LONG'}

        output = {
            'prob_short':    round(prob_short, 4),
            'prob_long':     round(prob_long, 4),
            'signal':        pred,
            'signal_label':  signal_map.get(pred, '?'),
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
            'dow':             r['dow'],
            'model_forward': model_data.get('forward', 8),
            'model_auc':     round(model_data.get('auc', 0), 4),
            'ts':            datetime.now().isoformat(),
            'dist_sma50_mnq': round(r['dist_sma50_mnq'], 3),
            'above_sma50_mnq': bool(r['above_sma50_mnq']),
            'sma50_alignment': r['sma50_alignment'],
            'ema20_bias_mnq_btc': r['ema20_bias_mnq_btc'],
        }
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({'error': str(e)}))


if __name__ == '__main__':
    main()

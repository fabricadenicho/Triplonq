"""
Busca dados atuais, computa features (MGC primario) e retorna predicao ML.
Chamado pelo server.js via child_process. Saida: JSON no stdout.
As chaves de saida sao identicas ao predict.py original para compatibilidade do dashboard.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pickle
from pathlib import Path
from datetime import datetime

MODEL_PATH = Path(__file__).parent / 'model.pkl'
SYMS = {'mgc': 'MGC=F', 'mnq': 'MNQ=F', 'btc': 'BTC-USD'}


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


def fetch_long(ticker):
    df = yf.download(ticker, period='60d', interval='1h',
                     auto_adjust=True, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])


def compute_key_levels_live(df):
    idx    = df.index
    last_c = float(df['close'].iloc[-1])

    def spct(last_val, ref_ser):
        ref_aligned = ref_ser.reindex(idx, method='ffill')
        if len(ref_aligned) == 0:
            return 0.0
        ref = ref_aligned.iloc[-1]
        if pd.isna(ref) or float(ref) == 0:
            return 0.0
        return round((last_val - float(ref)) / float(ref) * 100, 4)

    def sabove(last_val, ref_ser):
        ref_aligned = ref_ser.reindex(idx, method='ffill')
        if len(ref_aligned) == 0:
            return 0
        ref = ref_aligned.iloc[-1]
        return int(last_val > float(ref)) if not pd.isna(ref) else 0

    result = {}

    daily = df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pdh_s = daily['high'].shift(1); pdl_s = daily['low'].shift(1); do_s = daily['open']
    result['dist_to_pdh'] = spct(last_c, pdh_s)
    result['dist_to_pdl'] = spct(last_c, pdl_s)
    result['dist_to_do']  = spct(last_c, do_s)
    result['above_do']    = sabove(last_c, do_s)
    result['above_pdh']   = sabove(last_c, pdh_s)
    result['above_pdl']   = sabove(last_c, pdl_s)
    pdh_v = pdh_s.iloc[-1]; pdl_v = pdl_s.iloc[-1]
    result['prev_day_range_pct'] = round((float(pdh_v) - float(pdl_v)) / float(pdl_v) * 100, 4) \
        if not (pd.isna(pdh_v) or pd.isna(pdl_v) or float(pdl_v) == 0) else 0.0

    weekly = df.resample('W-SUN').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pwh_s = weekly['high'].shift(1); pwl_s = weekly['low'].shift(1); wo_s = weekly['open']
    result['dist_to_pwh'] = spct(last_c, pwh_s)
    result['dist_to_pwl'] = spct(last_c, pwl_s)
    result['dist_to_wo']  = spct(last_c, wo_s)
    result['above_wo']    = sabove(last_c, wo_s)
    result['above_pwh']   = sabove(last_c, pwh_s)
    result['above_pwl']   = sabove(last_c, pwl_s)

    monthly = df.resample('MS').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pmh_s = monthly['high'].shift(1); pml_s = monthly['low'].shift(1); mo_s = monthly['open']
    result['dist_to_pmh'] = spct(last_c, pmh_s)
    result['dist_to_pml'] = spct(last_c, pml_s)
    result['dist_to_mo']  = spct(last_c, mo_s)
    result['above_mo']    = sabove(last_c, mo_s)
    result['above_pmh']   = sabove(last_c, pmh_s)
    result['above_pml']   = sabove(last_c, pml_s)

    monday_bars = df[df.index.dayofweek == 0]
    if len(monday_bars) >= 4:
        mday_h_s = monday_bars['high'].resample('W-SUN').max()
        mday_l_s = monday_bars['low'].resample('W-SUN').min()
        result['dist_to_mday_h'] = spct(last_c, mday_h_s)
        result['dist_to_mday_l'] = spct(last_c, mday_l_s)
        result['above_mday_h']   = sabove(last_c, mday_h_s)
        result['above_mday_l']   = sabove(last_c, mday_l_s)
    else:
        result['dist_to_mday_h'] = 0.0; result['dist_to_mday_l'] = 0.0
        result['above_mday_h']   = 0;   result['above_mday_l']   = 0

    return result


def main():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        print(json.dumps({'error': 'model.pkl nao encontrado. Rode train.py primeiro.'}))
        return

    try:
        mgc_df = fetch('MGC=F')
        mnq_df = fetch('MNQ=F')
        btc_df = fetch('BTC-USD')
        mgc_long = fetch_long('MGC=F')

        if mgc_df is None or mnq_df is None or btc_df is None:
            print(json.dumps({'error': 'Falha ao buscar dados do Yahoo Finance'}))
            return

        mgc = compute(mgc_df)
        mnq = compute(mnq_df)
        btc = compute(btc_df)

        # Monta vetor de features (MGC primario, chaves mantidas com sufixo _cl para compatibilidade)
        r = {}
        r['rsi_cl']       = last(mgc, 'rsi')
        r['rsi_mnq']      = last(mnq, 'rsi')
        r['rsi_btc']      = last(btc, 'rsi')
        r['adx_cl']       = last(mgc, 'adx')
        r['adx_mnq']      = last(mnq, 'adx')
        r['adx_btc']      = last(btc, 'adx')
        r['pdi_cl']       = last(mgc, 'pdi')
        r['mdi_cl']       = last(mgc, 'mdi')
        r['div_mnq']      = r['rsi_cl'] - r['rsi_mnq']
        r['div_btc']      = r['rsi_cl'] - r['rsi_btc']
        r['ret1_cl']      = last(mgc, 'ret1')
        r['ret4_cl']      = last(mgc, 'ret4')
        r['ret8_cl']      = last(mgc, 'ret8')
        r['vol_cl']       = last(mgc, 'vol')
        r['bb_cl']        = last(mgc, 'bb_w')
        r['ret1_mnq']     = last(mnq, 'ret1')
        r['ret4_mnq']     = last(mnq, 'ret4')
        r['ret1_btc']     = last(btc, 'ret1')
        r['ret4_btc']     = last(btc, 'ret4')
        r['hour']         = int(mgc.index[-1].hour)
        r['dow']          = int(mgc.index[-1].dayofweek)
        r['dadx_cl']      = (last(mgc, 'adx') - float(mgc['adx'].iloc[-3])) if len(mgc) >= 3 else 0.0
        r['dist_sma50_cl']   = last(mgc, 'dist_sma50') or 0.0
        r['dist_sma50_mnq']  = last(mnq, 'dist_sma50') or 0.0
        r['dist_sma50_btc']  = last(btc, 'dist_sma50') or 0.0
        r['sma50_slope_cl']  = last(mgc, 'sma50_slope') or 0.0
        r['above_sma50_cl']  = int((last(mgc, 'above_sma50') or 0))
        r['above_sma50_mnq'] = int((last(mnq, 'above_sma50') or 0))
        r['above_sma50_btc'] = int((last(btc, 'above_sma50') or 0))
        r['sma50_alignment'] = r['above_sma50_cl'] + r['above_sma50_mnq'] + r['above_sma50_btc']
        r['dist_ema20_cl']   = last(mgc, 'dist_ema20') or 0.0
        r['dist_ema20_mnq']  = last(mnq, 'dist_ema20') or 0.0
        r['dist_ema20_btc']  = last(btc, 'dist_ema20') or 0.0
        r['above_ema20_cl']  = int((last(mgc, 'above_ema20') or 0))
        r['above_ema20_mnq'] = int((last(mnq, 'above_ema20') or 0))
        r['above_ema20_btc'] = int((last(btc, 'above_ema20') or 0))
        r['ema20_bias_mnq_btc'] = r['above_ema20_cl'] + r['above_ema20_btc']
        r['ema20_alignment']    = r['above_ema20_cl'] + r['above_ema20_mnq'] + r['above_ema20_btc']
        ret1_cl_v  = r['ret1_cl'] or 0
        ret1_mnq_v = r['ret1_mnq'] or 0
        ret1_btc_v = r['ret1_btc'] or 0
        adx_v      = r['adx_cl'] or 0

        r['triple_signal'] = int(
            (r['rsi_btc'] or 100) < 45 and (r['div_mnq'] or 0) < 0 and (r['div_btc'] or 0) < 0
        )

        # ── Spreads completos ──
        r['rsi_spread_mnq_btc']  = (r['rsi_mnq'] or 50) - (r['rsi_btc'] or 50)
        r['rsi_abs_cl_mnq']      = abs(r['div_mnq'] or 0)
        r['rsi_abs_cl_btc']      = abs(r['div_btc'] or 0)
        r['rsi_abs_mnq_btc']     = abs(r['rsi_spread_mnq_btc'])

        r['adx_spread_cl_mnq']   = (r['adx_cl'] or 0) - (r['adx_mnq'] or 0)
        r['adx_spread_cl_btc']   = (r['adx_cl'] or 0) - (r['adx_btc'] or 0)
        r['adx_spread_mnq_btc']  = (r['adx_mnq'] or 0) - (r['adx_btc'] or 0)
        r['adx_abs_cl_mnq']      = abs(r['adx_spread_cl_mnq'])
        r['adx_abs_cl_btc']      = abs(r['adx_spread_cl_btc'])
        r['adx_abs_mnq_btc']     = abs(r['adx_spread_mnq_btc'])

        r['ret1_spread_cl_mnq']  = ret1_cl_v - ret1_mnq_v
        r['ret1_spread_cl_btc']  = ret1_cl_v - ret1_btc_v
        r['ret1_spread_mnq_btc'] = ret1_mnq_v - ret1_btc_v
        ret4_cl_v  = r['ret4_cl'] or 0
        ret4_mnq_v = r['ret4_mnq'] or 0
        ret4_btc_v = r['ret4_btc'] or 0
        r['ret4_spread_cl_mnq']  = ret4_cl_v - ret4_mnq_v
        r['ret4_spread_cl_btc']  = ret4_cl_v - ret4_btc_v
        r['ret4_spread_mnq_btc'] = ret4_mnq_v - ret4_btc_v

        r['ret1_prod_cl_mnq']  = ret1_cl_v * ret1_mnq_v
        r['ret1_prod_cl_btc']  = ret1_cl_v * ret1_btc_v
        r['price_div_cl']      = ret1_cl_v * ret1_mnq_v
        r['price_div_abs']     = abs(r['price_div_cl'])
        r['ret4_prod_cl_mnq']  = ret4_cl_v * ret4_mnq_v
        r['ret4_prod_cl_btc']  = ret4_cl_v * ret4_btc_v
        r['ret4_prod_mnq_btc'] = ret4_mnq_v * ret4_btc_v

        vol_cl_v  = r['vol_cl'] or 0
        vol_mnq_v = last(mnq, 'vol') or 0
        vol_btc_v = last(btc, 'vol') or 0
        r['vol_spread_cl_mnq']  = vol_cl_v - vol_mnq_v
        r['vol_spread_cl_btc']  = vol_cl_v - vol_btc_v
        r['vol_spread_mnq_btc'] = vol_mnq_v - vol_btc_v

        bb_cl_v  = r['bb_cl'] or 0
        bb_mnq_v = last(mnq, 'bb_w') or 0
        bb_btc_v = last(btc, 'bb_w') or 0
        r['bb_spread_cl_mnq']  = bb_cl_v - bb_mnq_v
        r['bb_spread_cl_btc']  = bb_cl_v - bb_btc_v
        r['bb_spread_mnq_btc'] = bb_mnq_v - bb_btc_v

        m50_cl  = r['dist_sma50_cl']
        m50_mnq = r['dist_sma50_mnq']
        m50_btc = r['dist_sma50_btc']
        r['sma50_dist_spread_cl_mnq']  = m50_cl - m50_mnq
        r['sma50_dist_spread_cl_btc']  = m50_cl - m50_btc
        r['sma50_dist_spread_mnq_btc'] = m50_mnq - m50_btc

        e50_cl  = r['dist_ema20_cl']
        e50_mnq = r['dist_ema20_mnq']
        e50_btc = r['dist_ema20_btc']
        r['ema20_dist_spread_cl_mnq']  = e50_cl - e50_mnq
        r['ema20_dist_spread_cl_btc']  = e50_cl - e50_btc
        r['ema20_dist_spread_mnq_btc'] = e50_mnq - e50_btc

        ac_cl  = r['above_sma50_cl']
        ac_mnq = r['above_sma50_mnq']
        ac_btc = r['above_sma50_btc']
        r['sma50_align_cl_mnq']  = ac_cl + ac_mnq
        r['sma50_align_cl_btc']  = ac_cl + ac_btc
        r['sma50_align_mnq_btc'] = ac_mnq + ac_btc

        ae_cl  = r['above_ema20_cl']
        ae_mnq = r['above_ema20_mnq']
        ae_btc = r['above_ema20_btc']
        r['ema20_align_cl_mnq']  = ae_cl + ae_mnq
        r['ema20_align_cl_btc']  = ae_cl + ae_btc
        r['ema20_align_mnq_btc'] = ae_mnq + ae_btc

        pdi_cl_v  = r['pdi_cl'] or 0
        mdi_cl_v  = r['mdi_cl'] or 0
        pdi_mnq_v = last(mnq, 'pdi') or 0
        mdi_mnq_v = last(mnq, 'mdi') or 0
        pdi_btc_v = last(btc, 'pdi') or 0
        mdi_btc_v = last(btc, 'mdi') or 0
        r['di_spread_cl']  = pdi_cl_v - mdi_cl_v
        r['di_spread_mnq'] = pdi_mnq_v - mdi_mnq_v
        r['di_spread_btc'] = pdi_btc_v - mdi_btc_v

        # ── Composite triggers ──
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

        # ── Kill zone features ──
        ASIA_HOURS   = [0, 1, 2, 3, 4, 5, 6, 7]
        LONDON_HOURS = [8, 9, 10, 11, 12, 13, 14]
        NY_HOURS     = [13, 14, 15, 16, 17, 18, 19]
        OVERLAP_HOURS = [13, 14]

        mgc_hour = mgc.index.hour
        in_asia   = pd.Series(mgc_hour.isin(ASIA_HOURS), index=mgc.index)
        in_london = pd.Series(mgc_hour.isin(LONDON_HOURS), index=mgc.index)
        in_ny     = pd.Series(mgc_hour.isin(NY_HOURS), index=mgc.index)
        in_any    = in_asia | in_london | in_ny
        prev_in_any = in_any.shift(1).fillna(False)
        kz_session_id = (in_any & ~prev_in_any).cumsum()
        kz_session_id[~in_any] = -1

        session_high = mgc['high'].groupby(kz_session_id).cummax()
        session_low  = mgc['low'].groupby(kz_session_id).cummin()

        r['is_asia']   = int(in_asia.iloc[-1])
        r['is_london'] = int(in_london.iloc[-1])
        r['is_ny']     = int(in_ny.iloc[-1])
        r['kz_overlap'] = int(r['hour'] in OVERLAP_HOURS)
        r['kz_dist_high'] = float(((mgc['close'] - session_high) / session_high * 100).iloc[-1]) if in_any.iloc[-1] else 0.0
        r['kz_dist_low']  = float(((mgc['close'] - session_low) / session_low * 100).iloc[-1]) if in_any.iloc[-1] else 0.0
        r['kz_range']     = float(((session_high - session_low) / session_low * 100).iloc[-1]) if in_any.iloc[-1] else 0.0
        r['kz_progress']  = 0.0
        if in_any.iloc[-1]:
            cur_grp = kz_session_id.iloc[-1]
            grp_count = (kz_session_id == cur_grp).sum()
            grp_pos = int((kz_session_id == cur_grp).cumsum().iloc[-1]) - 1
            r['kz_progress'] = grp_pos / max(grp_count - 1, 1)

        prev_sh = session_high.shift(1)
        prev_sl = session_low.shift(1)
        close_ = mgc['close']
        r['kz_breakout_up'] = int(bool(in_any.iloc[-1] and close_.iloc[-1] > prev_sh.iloc[-1] and close_.iloc[-2] <= prev_sh.iloc[-1]))
        r['kz_breakout_dn'] = int(bool(in_any.iloc[-1] and close_.iloc[-1] < prev_sl.iloc[-1] and close_.iloc[-2] >= prev_sl.iloc[-1]))

        r.update(compute_key_levels_live(mgc_long if mgc_long is not None else mgc_df))

        model_data = pickle.load(open(MODEL_PATH, 'rb'))
        model      = model_data['model']
        feat_cols  = model_data['features']

        X     = pd.DataFrame([r])[feat_cols]
        proba = model.predict_proba(X)[0]
        pred  = int(model.predict(X)[0])

        # Ensemble com especialista kill zone
        KZ_HOURS_ENS = {'asia': [0,1,2,3,4,5,6,7], 'london': [8,9,10,11,12,13,14], 'ny': [13,14,15,16,17,18,19]}
        current_kz = next((k for k, v in KZ_HOURS_ENS.items() if r['hour'] in v), None)
        if current_kz:
            kz_path = Path(__file__).parent / f'model_kz_{current_kz}.pkl'
            if kz_path.exists():
                kz_data  = pickle.load(open(kz_path, 'rb'))
                kz_model = kz_data['model']
                kz_feats = kz_data['features']
                X_kz = pd.DataFrame([r])[kz_feats]
                proba_kz = kz_model.predict_proba(X_kz)[0]
                w_main, kz_w = 0.4, 0.6
                proba = w_main * proba + kz_w * proba_kz
                pred = int(np.argmax(proba))
                r['ensemble_kz'] = current_kz

        prob_short = float(proba[0])
        prob_long  = float(proba[2])

        mgc_dir = 'BAIXO' if ret1_cl_v < 0 else 'CIMA'
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
            'cl_dir':        mgc_dir,
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
            'dist_sma50_mnq': round(r['dist_sma50_cl'], 3),
            'above_sma50_mnq': bool(r['above_sma50_cl']),
            'sma50_alignment': r['sma50_alignment'],
            'ema20_bias_mnq_btc': r['ema20_bias_mnq_btc'],
            'is_asia': bool(r['is_asia']),
            'is_london': bool(r['is_london']),
            'is_ny': bool(r['is_ny']),
            'kz_overlap': bool(r['kz_overlap']),
            'kz_dist_high': round(r['kz_dist_high'], 4),
            'kz_dist_low': round(r['kz_dist_low'], 4),
            'kz_range': round(r['kz_range'], 4),
            'kz_breakout_up': bool(r['kz_breakout_up']),
            'kz_breakout_dn': bool(r['kz_breakout_dn']),
            'ensemble_kz': r.get('ensemble_kz'),
        }
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({'error': str(e)}))


if __name__ == '__main__':
    main()

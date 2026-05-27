"""
Predicao ao vivo -- Estrategia S2: ROT-SHORT + abaixo 4H open + ML + Regime BEAR

Sinal SHORT quando:
  1. rot_score_short >= 60%  (9 condicoes espelhadas do S1)
  2. div_cl < 0              (CL RSI > MNQ RSI = vies SHORT)
  3. close < 4H open         (preco abaixo da abertura do candle de 4H)
  4. ml_prob >= 0.50         (modelo XGBoost treinado para queda)
  5. regime_bear == 1        (MNQ close < SMA200 1h -- filtro de regime)

Saida: JSON no stdout. Loga em ml/signals_s2_4h.csv.
"""
import sys, json, csv, warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd, numpy as np, ta, pickle
from pathlib import Path
from datetime import datetime

BASE       = Path(__file__).parent
MODEL_PATH = BASE / 'model_s2_4h.pkl'
LOG_PATH   = BASE / 'signals_s2_4h.csv'
ML_THR     = 0.50

def fetch(ticker):
    df = yf.download(ticker, period='15d', interval='1h', auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()

def compute(df):
    d = df.copy()
    d['rsi']        = ta.momentum.RSIIndicator(d['close'], window=21).rsi()
    adx_i           = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=17)
    d['adx']        = adx_i.adx()
    d['pdi']        = adx_i.adx_pos()
    d['mdi']        = adx_i.adx_neg()
    d['ret1']       = d['close'].pct_change(1)
    d['ret4']       = d['close'].pct_change(4)
    d['ret8']       = d['close'].pct_change(8)
    d['vol']        = d['ret1'].rolling(20).std()
    d['bb_w']       = d['close'].rolling(20).std() * 2 / d['close'].rolling(20).mean()
    d['sma50']      = d['close'].rolling(50).mean()
    d['dist_sma50'] = (d['close'] - d['sma50']) / d['sma50'] * 100
    d['above_sma50']= (d['close'] > d['sma50']).astype(int)
    d['ema20']      = d['close'].ewm(span=20, adjust=False).mean()
    d['dist_ema20'] = (d['close'] - d['ema20']) / d['ema20'] * 100
    d['above_ema20']= (d['close'] > d['ema20']).astype(int)
    return d

def main():
    if not MODEL_PATH.exists():
        print(json.dumps({'error': 'model_s2_4h.pkl nao encontrado'}))
        return

    try:
        mnq_raw = fetch('MNQ=F')
        es_raw  = fetch('ES=F')
        btc_raw = fetch('BTC-USD')
        cl_raw  = fetch('CL=F')
        if any(x is None for x in [mnq_raw, es_raw, btc_raw, cl_raw]):
            print(json.dumps({'error': 'Falha ao baixar dados'})); return

        mnq = compute(mnq_raw); es  = compute(es_raw)
        btc = compute(btc_raw); cl  = compute(cl_raw)

        idx = mnq.index.intersection(es.index).intersection(btc.index).intersection(cl.index)
        mnq = mnq.loc[idx]; es  = es.loc[idx]
        btc = btc.loc[idx]; cl  = cl.loc[idx]

        # ── Features (identicas ao S1) ────────────────────────────────────────
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

        for nome in ['mnq','es','btc','cl']:
            f[f'rsi_{nome}'] = locals()[nome]['rsi']
        f['div_cl']  = f['rsi_mnq'] - f['rsi_cl']
        f['div_es']  = f['rsi_mnq'] - f['rsi_es']
        f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']
        f['rsi_mnq_acima_60']  = (f['rsi_mnq'] > 60).astype(int)
        f['rsi_mnq_abaixo_40'] = (f['rsi_mnq'] < 40).astype(int)

        for nome in ['mnq','es','btc','cl']:
            raw_ref = {'mnq': mnq_raw,'es': es_raw,'btc': btc_raw,'cl': cl_raw}[nome]
            s = raw_ref['open'].reindex(idx, method='ffill')
            f[f'open_1h_{nome}'] = s
            f[f'open_1h_acima_close_ant_{nome}'] = (s > f[nome].shift(1)).astype(int)

        for nome in ['mnq','es','btc','cl']:
            s   = f[f'open_1h_{nome}']
            o4  = s.groupby(idx.floor('4h')).transform('first')
            ca4 = s.shift(1).groupby(idx.floor('4h')).transform('first')
            f[f'open_4h_acima_4h_ant_{nome}'] = (o4 > ca4).astype(int)
            f[f'open_4h_dist_{nome}']         = (s - o4) / o4 * 100

        for nome in ['mnq','es','btc','cl']:
            s  = f[f'open_1h_{nome}']
            d  = s.groupby(idx.date).transform('first')
            ca = s.shift(1).groupby(idx.date).transform('first')
            f[f'open_d_acima_d_ant_{nome}'] = (d > ca).astype(int)
            f[f'open_d_dist_{nome}']        = (s - d) / d * 100

        for nome in ['mnq','es','btc','cl']:
            s  = f[f'open_1h_{nome}']
            w  = s.groupby(pd.Grouper(freq='W-MON')).transform('first')
            wc = s.shift(1).groupby(pd.Grouper(freq='W-MON')).transform('first')
            f[f'open_w_acima_w_ant_{nome}'] = (w > wc).astype(int)
            f[f'open_w_dist_{nome}']        = (s - w) / w * 100

        f['open_mnq_acima_cl']  = (f['open_1h_mnq'] > f['open_1h_cl']).astype(int)
        f['open_mnq_acima_es']  = (f['open_1h_mnq'] > f['open_1h_es']).astype(int)
        f['open_mnq_acima_btc'] = (f['open_1h_mnq'] > f['open_1h_btc']).astype(int)

        for nome in ['mnq','es','btc','cl']:
            f[f'adx_{nome}']       = locals()[nome]['adx']
            f[f'di_spread_{nome}'] = locals()[nome]['pdi'] - locals()[nome]['mdi']
            f[f'above_sma50_{nome}']= locals()[nome]['above_sma50']
            f[f'above_ema20_{nome}']= locals()[nome]['above_ema20']
            f[f'dist_sma50_{nome}'] = locals()[nome]['dist_sma50']
            f[f'dist_ema20_{nome}'] = locals()[nome]['dist_ema20']
            f[f'vol_{nome}']        = locals()[nome]['vol'] * 100
            f[f'bb_w_{nome}']       = locals()[nome]['bb_w'] * 100

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

        f = f.dropna()
        if len(f) == 0:
            print(json.dumps({'error': 'Sem dados apos dropna'})); return

        # ── Predicao ML ───────────────────────────────────────────────────────
        model_data = pickle.load(open(MODEL_PATH, 'rb'))
        model      = model_data['model']
        feat_cols  = model_data['features']
        baseline   = model_data.get('baseline', 0.33)
        fwd        = model_data.get('forward', 4)

        last = f.iloc[-1:]
        X    = last[[c for c in feat_cols if c in last.columns]].fillna(0)
        prob = float(model.predict_proba(X)[0, 1])
        edge = round(prob - baseline, 4)

        # ── Score rotacao SHORT (9 condicoes espelhadas) ──────────────────────
        div_cl_v    = float(last['div_cl'].iloc[0])
        rsi_mnq_v   = float(last['rsi_mnq'].iloc[0])
        rsi_cl_v    = float(last['rsi_cl'].iloc[0])
        adx_v       = float(last['adx_mnq'].iloc[0])
        bb_cl_v     = float(last['bb_w_cl'].iloc[0])
        sma50_v     = int(last['sma50_alignment'].iloc[0])
        es_oposto_v = int(last['es_mnq_oposto'].iloc[0])
        h           = int(last['hour'].iloc[0])
        dw          = int(last['dow'].iloc[0])

        div_delta_2h  = float(f['div_cl'].iloc[-1] - f['div_cl'].iloc[-3]) if len(f) >= 3 else 0.0
        rsi_mnq_d2    = float(f['rsi_mnq'].iloc[-1] - f['rsi_mnq'].iloc[-3]) if len(f) >= 3 else 0.0
        rsi_cl_d2     = float(f['rsi_cl'].iloc[-1]  - f['rsi_cl'].iloc[-3])  if len(f) >= 3 else 0.0
        prev_div_cl   = float(f['div_cl'].iloc[-2]) if len(f) >= 2 else div_cl_v
        div_mudou     = int((div_cl_v > 0 and prev_div_cl <= 0) or (div_cl_v < 0 and prev_div_cl >= 0))
        ret1_mnq_v    = float(last['r_mnq_1h'].iloc[0])
        ret1_es_v     = float(f['r_es_1h'].iloc[-1])

        rot_c1 = div_mudou
        rot_c2 = int(abs(div_delta_2h) > 5.0)
        # SHORT: MNQ RSI caiu >2 E CL RSI subiu >2
        rot_c3 = int(rsi_mnq_d2 < -2.0 and rsi_cl_d2 > 2.0)
        rot_c4 = int((ret1_mnq_v > 0 and ret1_es_v > 0) or (ret1_mnq_v < 0 and ret1_es_v < 0))
        rot_c5 = int(bb_cl_v > 1.5)
        # SHORT: CL diario abrindo ABAIXO do anterior
        open_d_cl_v = int(last['open_d_acima_d_ant_cl'].iloc[0])
        rot_c6 = 1 - open_d_cl_v
        rot_c7 = int(12 <= adx_v <= 20)
        # SHORT: maioria ABAIXO da SMA50
        rot_c8 = int(sma50_v <= 2)
        rot_c9 = int(rsi_mnq_v > 55 or rsi_mnq_v < 45)

        rot_raw   = rot_c1*3.0 + rot_c2*2.5 + rot_c3*2.5 + rot_c4*2.0 + rot_c5*1.5 + rot_c6*1.5 + rot_c7*1.5 + rot_c8*1.5 + rot_c9*1.0
        rot_score = round(rot_raw / 17.0 * 100, 1)

        # ── 4H open ───────────────────────────────────────────────────────────
        open_4h_v     = float(mnq_raw['open'].resample('4h').first().reindex(idx, method='ffill').iloc[-1])
        mnq_close_v   = float(f['mnq'].iloc[-1])
        below_4h_open = int(mnq_close_v < open_4h_v)
        dist_4h_pct   = round((mnq_close_v - open_4h_v) / open_4h_v * 100, 3)

        # ── Regime: close < SMA200 (1h) ──────────────────────────────────────
        sma200_v    = float(mnq['close'].rolling(200).mean().iloc[-1])
        regime_bear = int(mnq_close_v < sma200_v)
        dist_sma200 = round((mnq_close_v - sma200_v) / sma200_v * 100, 2)

        # ── Sinal S2 ──────────────────────────────────────────────────────────
        dir_short   = int(div_cl_v < 0)
        s2_ativo    = int(rot_score >= 60.0 and dir_short and below_4h_open and prob >= ML_THR and regime_bear)
        s2_parcial  = int(rot_score >= 60.0 and dir_short and below_4h_open)  # sem ML e sem regime
        sinal       = 'SHORT' if s2_ativo else 'AGUARDAR'

        output = {
            # Sinal principal
            'sinal':         sinal,
            's2_ativo':      s2_ativo,
            's2_parcial':    s2_parcial,
            # ML
            'ml_prob':       round(prob, 4),
            'ml_edge':       edge,
            'ml_baseline':   round(baseline, 4),
            'ml_threshold':  ML_THR,
            'ml_ok':         int(prob >= ML_THR),
            # Rotacao SHORT
            'rot_score':     rot_score,
            'rot_ok':        int(rot_score >= 60),
            'rot_c1': rot_c1, 'rot_c2': rot_c2, 'rot_c3': rot_c3,
            'rot_c4': rot_c4, 'rot_c5': rot_c5, 'rot_c6': rot_c6,
            'rot_c7': rot_c7, 'rot_c8': rot_c8, 'rot_c9': rot_c9,
            # Direcao
            'div_cl':        round(div_cl_v, 2),
            'dir_short':     dir_short,
            'dir_ok':        dir_short,
            # 4H open
            'open_4h':       round(open_4h_v, 2),
            'mnq_close':     round(mnq_close_v, 2),
            'below_4h_open': below_4h_open,
            'dist_4h_pct':   dist_4h_pct,
            '4h_ok':         below_4h_open,
            # Regime
            'regime_bear':   regime_bear,
            'regime_ok':     regime_bear,
            'sma200':        round(sma200_v, 2),
            'dist_sma200':   dist_sma200,
            # Contexto
            'rsi_mnq':       round(rsi_mnq_v, 1),
            'rsi_cl':        round(rsi_cl_v, 1),
            'adx_mnq':       round(adx_v, 1),
            'sma50_align':   sma50_v,
            'bb_w_cl':       round(bb_cl_v, 3),
            'hour':          h,
            'dow':           dw,
            'ts':            str(last.index[0]),
            'forward_h':     fwd,
        }

        # ── Log ───────────────────────────────────────────────────────────────
        try:
            log_exists = LOG_PATH.exists()
            with open(LOG_PATH, 'a', newline='') as lf:
                w = csv.DictWriter(lf, fieldnames=[
                    'ts','sinal','rot_score','dir_short','below_4h_open',
                    'ml_prob','dist_4h_pct','div_cl','rsi_mnq','adx_mnq',
                    'sma50_align','bb_w_cl','regime_bear','dist_sma200','hour','dow'])
                if not log_exists:
                    w.writeheader()
                w.writerow({
                    'ts': output['ts'], 'sinal': sinal,
                    'rot_score': rot_score, 'dir_short': dir_short,
                    'below_4h_open': below_4h_open, 'ml_prob': round(prob, 4),
                    'dist_4h_pct': dist_4h_pct, 'div_cl': round(div_cl_v, 2),
                    'rsi_mnq': round(rsi_mnq_v, 1), 'adx_mnq': round(adx_v, 1),
                    'sma50_align': sma50_v, 'bb_w_cl': round(bb_cl_v, 3),
                    'regime_bear': regime_bear, 'dist_sma200': dist_sma200,
                    'hour': h, 'dow': dw,
                })
        except Exception:
            pass

        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({'error': str(e)}))

if __name__ == '__main__':
    main()

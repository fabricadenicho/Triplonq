"""
Predicao ao vivo do modelo divergencia (MNQ > 0.1% em 4h).
Chamado pelo server.js via child_process. Saida: JSON no stdout.
Loga predicoes para tracking de performance.

Uso: python predict_divergencia.py
"""
import sys, json, csv, warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd, numpy as np, ta, pickle
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
MODEL_PATH = BASE / 'model_divergencia.pkl'
LOG_PATH = BASE / 'logs_divergencia.csv'

def fetch(ticker):
    df = yf.download(ticker, period='10d', interval='1h', auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()

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

def last(df, col):
    v = df[col].iloc[-1]
    return float(v) if pd.notna(v) else 0.0

def main():
    if not MODEL_PATH.exists():
        print(json.dumps({'error': 'model_divergencia.pkl nao encontrado. Rode train_divergencia.py primeiro.'}))
        return

    try:
        mnq_df = fetch('MNQ=F')
        es_df  = fetch('ES=F')
        btc_df = fetch('BTC-USD')
        cl_df  = fetch('CL=F')
        if any(x is None for x in [mnq_df, es_df, btc_df, cl_df]):
            print(json.dumps({'error': 'Falha ao baixar dados'})); return

        mnq = compute(mnq_df); es = compute(es_df)
        btc = compute(btc_df); cl = compute(cl_df)

        # Sincronizar indices
        idx = mnq.index.intersection(es.index).intersection(btc.index).intersection(cl.index)
        mnq = mnq.loc[idx]; es = es.loc[idx]
        btc = btc.loc[idx]; cl = cl.loc[idx]

        f = pd.DataFrame(index=idx)
        f['mnq'] = mnq['close']; f['es'] = es['close']
        f['btc'] = btc['close']; f['cl'] = cl['close']

        for nome, s in [('mnq',mnq),('es',es),('btc',btc),('cl',cl)]:
            f[f'r_{nome}_1h'] = s['ret1'] * 100
            f[f'r_{nome}_4h'] = s['ret4'] * 100
        f['r_mnq_8h'] = mnq['ret8'] * 100

        f['es_mnq_mesmo'] = (((f['r_es_1h']>0)&(f['r_mnq_1h']>0))|((f['r_es_1h']<0)&(f['r_mnq_1h']<0))).astype(int)
        f['es_mnq_oposto'] = (((f['r_es_1h']>0)&(f['r_mnq_1h']<0))|((f['r_es_1h']<0)&(f['r_mnq_1h']>0))).astype(int)
        f['btc_mnq_mesmo'] = (((f['r_btc_1h']>0)&(f['r_mnq_1h']>0))|((f['r_btc_1h']<0)&(f['r_mnq_1h']<0))).astype(int)
        f['btc_mnq_oposto'] = (((f['r_btc_1h']>0)&(f['r_mnq_1h']<0))|((f['r_btc_1h']<0)&(f['r_mnq_1h']>0))).astype(int)
        f['es_btc_discordam_mnq'] = (f['es_mnq_oposto'] & f['btc_mnq_oposto']).astype(int)
        f['es_btc_concordam_mnq'] = (f['es_mnq_mesmo'] & f['btc_mnq_mesmo']).astype(int)

        for nome in ['mnq','es','btc','cl']:
            f[f'rsi_{nome}'] = locals()[nome]['rsi']
        f['div_cl'] = f['rsi_mnq'] - f['rsi_cl']
        f['div_es'] = f['rsi_mnq'] - f['rsi_es']
        f['div_btc'] = f['rsi_mnq'] - f['rsi_btc']
        f['rsi_mnq_acima_60'] = (f['rsi_mnq'] > 60).astype(int)
        f['rsi_mnq_abaixo_40'] = (f['rsi_mnq'] < 40).astype(int)

        # Key levels
        for nome, raw in [('mnq',mnq_df),('es',es_df),('btc',btc_df),('cl',cl_df)]:
            s = raw['open'].reindex(idx, method='ffill')
            f[f'open_1h_{nome}'] = s
            f[f'open_1h_acima_close_ant_{nome}'] = (s > f[nome].shift(1)).astype(int)

        for nome in ['mnq','es','btc','cl']:
            s = f[f'open_1h_{nome}']
            o4 = s.groupby(idx.floor('4h')).transform('first')
            ca4 = s.shift(1).groupby(idx.floor('4h')).transform('first')
            f[f'open_4h_acima_4h_ant_{nome}'] = (o4 > ca4).astype(int)
            f[f'open_4h_dist_{nome}'] = (s - o4) / o4 * 100

        for nome in ['mnq','es','btc','cl']:
            s = f[f'open_1h_{nome}']
            d = s.groupby(idx.date).transform('first')
            ca = s.shift(1).groupby(idx.date).transform('first')
            f[f'open_d_acima_d_ant_{nome}'] = (d > ca).astype(int)
            f[f'open_d_dist_{nome}'] = (s - d) / d * 100

        for nome in ['mnq','es','btc','cl']:
            s = f[f'open_1h_{nome}']
            w = s.groupby(pd.Grouper(freq='W-MON')).transform('first')
            wc = s.shift(1).groupby(pd.Grouper(freq='W-MON')).transform('first')
            f[f'open_w_acima_w_ant_{nome}'] = (w > wc).astype(int)
            f[f'open_w_dist_{nome}'] = (s - w) / w * 100

        f['open_mnq_acima_cl'] = (f['open_1h_mnq'] > f['open_1h_cl']).astype(int)
        f['open_mnq_acima_es'] = (f['open_1h_mnq'] > f['open_1h_es']).astype(int)
        f['open_mnq_acima_btc'] = (f['open_1h_mnq'] > f['open_1h_btc']).astype(int)

        for nome in ['mnq','es','btc','cl']:
            f[f'adx_{nome}'] = locals()[nome]['adx']
        f['adx_mnq_alto'] = (f['adx_mnq'] > 14).astype(int)
        f['adx_cl_alto'] = (f['adx_cl'] > 14).astype(int)
        f['adx_es_alto'] = (f['adx_es'] > 14).astype(int)

        for nome in ['mnq','es','btc','cl']:
            f[f'di_spread_{nome}'] = locals()[nome]['pdi'] - locals()[nome]['mdi']

        for nome in ['mnq','es','btc','cl']:
            f[f'above_sma50_{nome}'] = locals()[nome]['above_sma50']
            f[f'above_ema20_{nome}'] = locals()[nome]['above_ema20']
            f[f'dist_sma50_{nome}'] = locals()[nome]['dist_sma50']
            f[f'dist_ema20_{nome}'] = locals()[nome]['dist_ema20']
        f['sma50_alignment'] = sum(f[f'above_sma50_{n}'] for n in ['mnq','es','btc','cl'])
        f['ema20_alignment'] = sum(f[f'above_ema20_{n}'] for n in ['mnq','es','btc','cl'])

        for nome in ['mnq','es','btc','cl']:
            f[f'vol_{nome}'] = locals()[nome]['vol'] * 100
            f[f'bb_w_{nome}'] = locals()[nome]['bb_w'] * 100

        f['hour'] = idx.hour; f['dow'] = idx.dayofweek
        f['is_us'] = f['hour'].between(9, 17).astype(int)
        f['is_asia'] = f['hour'].between(0, 8).astype(int)
        f['is_evening'] = f['hour'].between(18, 23).astype(int)

        f = f.dropna()
        if len(f) == 0:
            print(json.dumps({'error': 'Sem dados apos dropna'})); return

        # Deltas de divergencia — detecta inicio de rotacao (usa historico disponivel)
        div_cl_delta_2h = float(f['div_cl'].iloc[-1] - f['div_cl'].iloc[-3]) if len(f) >= 3 else 0.0
        div_cl_delta_3h = float(f['div_cl'].iloc[-1] - f['div_cl'].iloc[-4]) if len(f) >= 4 else 0.0
        rsi_mnq_delta_2h = float(f['rsi_mnq'].iloc[-1] - f['rsi_mnq'].iloc[-3]) if len(f) >= 3 else 0.0
        rsi_cl_delta_2h  = float(f['rsi_cl'].iloc[-1]  - f['rsi_cl'].iloc[-3])  if len(f) >= 3 else 0.0
        # rsi_rotation: MNQ subindo E CL caindo ao mesmo tempo (>=2 pts em 2h)
        rsi_rotation = int(rsi_mnq_delta_2h > 2.0 and rsi_cl_delta_2h < -2.0)
        div_cl_snapshot = float(f['div_cl'].iloc[-1])
        # divergencia_starting: rotacao ativa + aceleracao significativa + div ainda nao extrema
        divergencia_starting = int(
            abs(div_cl_delta_2h) > 5.0 and
            rsi_rotation == 1 and
            abs(div_cl_snapshot) < 20.0
        )

        # C1: div_cl mudou de sinal na ultima barra
        prev_div_cl = float(f['div_cl'].iloc[-2]) if len(f) >= 2 else div_cl_snapshot
        div_sinal_mudou = int(
            (div_cl_snapshot > 0 and prev_div_cl <= 0) or
            (div_cl_snapshot < 0 and prev_div_cl >= 0)
        )

        last_row = f.iloc[-1:]

        model_data = pickle.load(open(MODEL_PATH, 'rb'))
        model = model_data['model']
        feat_cols = model_data['features']
        fwd = model_data.get('forward', 1)
        baseline = model_data.get('baseline', 0.246)

        X = last_row[[c for c in feat_cols if c in last_row.columns]].fillna(0)
        if len(X.columns) == 0:
            print(json.dumps({'error': 'Nenhuma feature encontrada'})); return

        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)

        edg = round(prob - baseline, 4)

        # Valores para o checklist estendido
        rsi_cl_v = float(last_row['rsi_cl'].iloc[0])
        bb_cl_v = float(last_row['bb_w_cl'].iloc[0])
        vol_cl_v = float(last_row['vol_cl'].iloc[0])
        open_d_acima_cl = int(last_row['open_d_acima_d_ant_cl'].iloc[0])
        open_1h_acima_cl = int(last_row['open_1h_acima_close_ant_cl'].iloc[0])
        div_cl_abs = abs(float(last_row['div_cl'].iloc[0]))
        rsi_mnq_v = float(last_row['rsi_mnq'].iloc[0])
        es_oposto = int(last_row['es_mnq_oposto'].iloc[0])
        btc_oposto = int(last_row['btc_mnq_oposto'].iloc[0])
        adx_v = float(last_row['adx_mnq'].iloc[0])
        sma50_v = int(last_row['sma50_alignment'].iloc[0])
        es_acima_sma50 = int(last_row['above_sma50_es'].iloc[0])
        h = int(last_row['hour'].iloc[0])
        dw = int(last_row['dow'].iloc[0])
        is_us_v = int(last_row['is_us'].iloc[0])
        vol_mnq_v = float(last_row['vol_mnq'].iloc[0])

        # Checklist items com pesos baseados na analise
        checks = [
            ('es_mnq_oposto',       es_oposto,      2.4),
            ('div_rsi_abs_alta',    int(div_cl_abs > 10), 1.6),
            ('rsi_mnq_acima_55',    int(rsi_mnq_v > 55), 1.6),
            ('adx_mnq_alto',        int(adx_v > 14), 1.2),
            ('sma50_alignment_3',   int(sma50_v >= 3), 1.5),
            ('bb_w_cl_alto',        int(bb_cl_v > 1.5), 2.7),
            ('open_d_acima_cl',     open_d_acima_cl, 2.0),
            ('rsi_cl_acima_55',     int(rsi_cl_v > 55), 1.2),
            ('vol_cl_alto',         int(vol_cl_v > 0.4), 2.6),
            ('is_us',               is_us_v, 1.1),
        ]
        score_max = sum(w for _, _, w in checks)
        score_ok = sum(w for _, ok, w in checks if ok)
        score_pct = round(score_ok / score_max * 100, 1) if score_max > 0 else 0
        score7_ok = sum(1 for _, ok, _ in checks if ok)

        # Rotacao score — 9 condicoes ponderadas (trigger-rotacao.md / backtest_rotacao.py)
        es_mesmo = 1 - es_oposto
        rot_c1 = div_sinal_mudou
        rot_c2 = int(abs(div_cl_delta_2h) > 5.0)
        rot_c3 = rsi_rotation
        rot_c4 = es_mesmo
        rot_c5 = int(bb_cl_v > 1.5)
        rot_c6 = open_d_acima_cl
        rot_c7 = int(12 <= adx_v <= 20)
        rot_c8 = int(sma50_v >= 2)
        rot_c9 = int(rsi_mnq_v > 55 or rsi_mnq_v < 45)
        rot_raw = (rot_c1*3.0 + rot_c2*2.5 + rot_c3*2.5 + rot_c4*2.0 +
                   rot_c5*1.5 + rot_c6*1.5 + rot_c7*1.5 + rot_c8*1.5 + rot_c9*1.0)
        rot_score = round(rot_raw / 17.0 * 100, 1)
        rot_dir = 'LONG' if div_cl_snapshot > 0 else 'SHORT'

        output = {
            'prob_long': round(prob, 4),
            'prob_divergencia': round(prob, 4),
            'pred': pred,
            'pred_label': 'LONG' if pred else 'NEUTRO/SHORT',
            'target_desc': f'MNQ > 0.1% em {fwd}h',
            'forward': fwd,
            'baseline': round(baseline, 4),
            'edge': edg,
            'hour': h, 'dow': dw,
            'ts': str(last_row.index[0]),
            'es_mnq_oposto': es_oposto,
            'btc_mnq_oposto': btc_oposto,
            'div_cl': round(float(last_row['div_cl'].iloc[0]), 2),
            'rsi_mnq': round(rsi_mnq_v, 1),
            'rsi_cl': round(rsi_cl_v, 1),
            'adx_mnq': round(adx_v, 1),
            'sma50_alignment': sma50_v,
            'es_acima_sma50': es_acima_sma50,
            'bb_w_cl': round(bb_cl_v, 3),
            'vol_cl': round(vol_cl_v, 3),
            'vol_mnq': round(vol_mnq_v, 3),
            'open_d_acima_ant_cl': open_d_acima_cl,
            'open_1h_acima_ant_cl': open_1h_acima_cl,
            'is_us': is_us_v,
            'div_cl_delta_2h': round(div_cl_delta_2h, 2),
            'div_cl_delta_3h': round(div_cl_delta_3h, 2),
            'rsi_mnq_delta_2h': round(rsi_mnq_delta_2h, 1),
            'rsi_cl_delta_2h': round(rsi_cl_delta_2h, 1),
            'rsi_rotation': rsi_rotation,
            'divergencia_starting': divergencia_starting,
            'checklist_score': score_pct,
            'checklist_max': score_max,
            'checklist_ok': round(score_ok, 1),
            'checklist_items': score7_ok,
            'checklist_total': len(checks),
            'rotacao_score': rot_score,
            'rotacao_ativo': int(rot_score >= 60),
            'rotacao_direcao': rot_dir,
            'rot_c1': rot_c1, 'rot_c2': rot_c2, 'rot_c3': rot_c3,
            'rot_c4': rot_c4, 'rot_c5': rot_c5, 'rot_c6': rot_c6,
            'rot_c7': rot_c7, 'rot_c8': rot_c8, 'rot_c9': rot_c9,
        }

        # Log para tracking de performance
        try:
            log_file = LOG_PATH
            log_exists = log_file.exists()
            with open(log_file, 'a', newline='') as lf:
                w = csv.DictWriter(lf, fieldnames=[
                    'ts','hour','dow','prob','pred','mnq_ret_fwd',
                    'es_oposto','div_cl','rsi_mnq','adx_mnq',
                    'sma50_align','bb_w_cl','open_d_cl','is_us',
                    'checklist_score','checklist_ok'])
                if not log_exists:
                    w.writeheader()
                w.writerow({
                    'ts': output['ts'], 'hour': h, 'dow': dw,
                    'prob': round(prob, 4), 'pred': pred,
                    'mnq_ret_fwd': '',  # preenchido depois pelo stats
                    'es_oposto': es_oposto, 'div_cl': round(div_cl_abs, 2),
                    'rsi_mnq': round(rsi_mnq_v, 1), 'adx_mnq': round(adx_v, 1),
                    'sma50_align': sma50_v, 'bb_w_cl': round(bb_cl_v, 3),
                    'open_d_cl': open_d_acima_cl, 'is_us': is_us_v,
                    'checklist_score': score_pct,
                    'checklist_ok': round(score_ok, 1),
                })
        except Exception:
            pass  # log nao critico

        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({'error': str(e)}))

if __name__ == '__main__':
    main()

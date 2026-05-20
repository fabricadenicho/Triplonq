"""
Live 2.0 — Predicao ML multi-ativo com sinais de execucao.
Usa os novos modelos (propfirm_model_*.pkl, forward=8h).
Gera sinais: direcao, entry, stop, target, risco, contratos.

Chamado pelo server.js via child_process. Saida: JSON no stdout.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pickle
from pathlib import Path

BASE = Path(__file__).parent.parent
TESTE_DIR = Path(__file__).parent

MODELOS = {
    'mnq': TESTE_DIR / 'propfirm_model_mnq.pkl',
    'btc': TESTE_DIR / 'propfirm_model_btc.pkl',
    'cl':  TESTE_DIR / 'propfirm_model_cl.pkl',
    'mgc': TESTE_DIR / 'propfirm_model_mgc.pkl',
    'es':  TESTE_DIR / 'propfirm_model_es.pkl',
}

SYMS = {'mnq': 'MNQ=F', 'btc': 'BTC-USD', 'cl': 'CL=F', 'mgc': 'MGC=F', 'es': 'ES=F'}

# Setup prop firm por ativo (da otimizacao)
SETUP = {
    'mnq': {'direcao': 'both', 'stop_r': 1.5, 'target_r': 3.0, 'ml_min': 0.5},
    'btc': {'direcao': 'short','stop_r': 1.5, 'target_r': 3.0, 'ml_min': 0.5},
    'cl':  {'direcao': 'both', 'stop_r': 1.5, 'target_r': 2.0, 'ml_min': 0.5},
    'mgc': {'direcao': 'both', 'stop_r': 1.5, 'target_r': 2.0, 'ml_min': 0.5},
    'es':  {'direcao': 'both', 'stop_r': 1.5, 'target_r': 3.0, 'ml_min': 0.5},
}

RISCO_PCT = 0.005  # 0.5% por trade
CONTA = 50000      # $50k prop firm

# Features-chave por ativo (baseado nas arvores XGBoost — arvore_propfirm.md)
# sec1/sec2 por ativo: mnq→(btc,cl) btc→(mnq,cl) cl→(mnq,btc) mgc→(mnq,btc)
KEY_FEATURES = {
    'mnq': ['vol_p', 'sma50_alignment', 'div_1', 'di_spread_p', 'di_spread_2', 'dist_to_mo', 'hour', 'vol_spread_p_1'],
    'btc': ['dow', 'vol_p', 'rsi_p', 'bb_p', 'adx_1', 'di_spread_1', 'dist_to_mday_h', 'ret4_1'],
    'cl':  ['hour', 'vol_p', 'prev_day_range_pct', 'di_spread_1', 'dist_to_pwh', 'dist_to_mo', 'dist_to_pdl', 'dist_to_mday_l'],
    'mgc': ['vol_p', 'dist_to_mo', 'dist_to_mday_l', 'di_spread_1', 'dist_to_pdl', 'di_spread_2', 'rsi_p', 'bb_p'],
}


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


def compute(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
    adx_i     = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=17)
    df['adx']  = adx_i.adx()
    df['pdi']  = adx_i.adx_pos()
    df['mdi']  = adx_i.adx_neg()
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
    # ATR
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    df['atr14'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    return df


def last(df, col):
    v = df[col].iloc[-1]
    return float(v) if pd.notna(v) else None


def compute_key_levels_live(df):
    """Key level features da ultima barra."""
    idx    = df.index
    last_c = float(df['close'].iloc[-1])

    def spct(last_val, ref_ser):
        ref_aligned = ref_ser.reindex(idx, method='ffill')
        if len(ref_aligned) == 0: return 0.0
        ref = ref_aligned.iloc[-1]
        if pd.isna(ref) or float(ref) == 0: return 0.0
        return round((last_val - float(ref)) / float(ref) * 100, 4)

    def sabove(last_val, ref_ser):
        ref_aligned = ref_ser.reindex(idx, method='ffill')
        if len(ref_aligned) == 0: return 0
        ref = ref_aligned.iloc[-1]
        return int(last_val > float(ref)) if not pd.isna(ref) else 0

    result = {}
    daily = df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pdh_s = daily['high'].shift(1); pdl_s = daily['low'].shift(1); do_s = daily['open']
    result['dist_to_pdh'] = spct(last_c, pdh_s); result['dist_to_pdl'] = spct(last_c, pdl_s)
    result['dist_to_do']  = spct(last_c, do_s);  result['above_do'] = sabove(last_c, do_s)
    result['above_pdh']   = sabove(last_c, pdh_s); result['above_pdl'] = sabove(last_c, pdl_s)
    pdh_v = pdh_s.iloc[-1]; pdl_v = pdl_s.iloc[-1]
    result['prev_day_range_pct'] = round((float(pdh_v)-float(pdl_v))/float(pdl_v)*100, 4) \
        if not (pd.isna(pdh_v) or pd.isna(pdl_v) or float(pdl_v)==0) else 0.0

    weekly = df.resample('W-SUN').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pwh_s = weekly['high'].shift(1); pwl_s = weekly['low'].shift(1); wo_s = weekly['open']
    result['dist_to_pwh'] = spct(last_c, pwh_s); result['dist_to_pwl'] = spct(last_c, pwl_s)
    result['dist_to_wo']  = spct(last_c, wo_s);  result['above_wo'] = sabove(last_c, wo_s)
    result['above_pwh']   = sabove(last_c, pwh_s); result['above_pwl'] = sabove(last_c, pwl_s)

    monthly = df.resample('MS').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pmh_s = monthly['high'].shift(1); pml_s = monthly['low'].shift(1); mo_s = monthly['open']
    result['dist_to_pmh'] = spct(last_c, pmh_s); result['dist_to_pml'] = spct(last_c, pml_s)
    result['dist_to_mo']  = spct(last_c, mo_s);  result['above_mo'] = sabove(last_c, mo_s)
    result['above_pmh']   = sabove(last_c, pmh_s); result['above_pml'] = sabove(last_c, pml_s)

    monday_bars = df[df.index.dayofweek == 0]
    if len(monday_bars) >= 4:
        mday_h_s = monday_bars['high'].resample('W-SUN').max()
        mday_l_s = monday_bars['low'].resample('W-SUN').min()
        result['dist_to_mday_h'] = spct(last_c, mday_h_s); result['dist_to_mday_l'] = spct(last_c, mday_l_s)
        result['above_mday_h']   = sabove(last_c, mday_h_s); result['above_mday_l']   = sabove(last_c, mday_l_s)
    else:
        result['dist_to_mday_h'] = 0.0; result['dist_to_mday_l'] = 0.0
        result['above_mday_h']   = 0;   result['above_mday_l']   = 0
    return result


def build_features_live(pri, sec1, sec2, key_levels):
    """Monta feature vector identico ao do train.py para a ultima barra."""
    pri, sec1, sec2 = pri.copy(), sec1.copy(), sec2.copy()
    
    # Alinhar indices
    idx = pri.index
    sec1 = sec1.reindex(idx, method='ffill')
    sec2 = sec2.reindex(idx, method='ffill')

    f = pd.DataFrame(index=idx)
    f['rsi_p'] = pri['rsi']; f['rsi_1'] = sec1['rsi']; f['rsi_2'] = sec2['rsi']
    f['div_1'] = f['rsi_p'] - f['rsi_1']; f['div_2'] = f['rsi_p'] - f['rsi_2']
    f['rsi_spread_1_2'] = f['rsi_1'] - f['rsi_2']
    f['rsi_abs_p_1'] = (f['rsi_p'] - f['rsi_1']).abs()
    f['rsi_abs_p_2'] = (f['rsi_p'] - f['rsi_2']).abs()
    f['rsi_abs_1_2'] = (f['rsi_1'] - f['rsi_2']).abs()

    f['adx_p'] = pri['adx']; f['adx_1'] = sec1['adx']; f['adx_2'] = sec2['adx']
    f['pdi_p'] = pri['pdi']; f['mdi_p'] = pri['mdi']
    f['adx_spread_p_1'] = f['adx_p'] - f['adx_1']; f['adx_spread_p_2'] = f['adx_p'] - f['adx_2']
    f['adx_spread_1_2'] = f['adx_1'] - f['adx_2']
    f['adx_abs_p_1'] = (f['adx_p'] - f['adx_1']).abs()
    f['adx_abs_p_2'] = (f['adx_p'] - f['adx_2']).abs()
    f['adx_abs_1_2'] = (f['adx_1'] - f['adx_2']).abs()

    f['di_spread_p'] = f['pdi_p'] - f['mdi_p']
    f['di_spread_1'] = sec1['pdi'] - sec1['mdi']
    f['di_spread_2'] = sec2['pdi'] - sec2['mdi']
    f['dadx_p'] = f['adx_p'].diff(2)

    f['ret1_p'] = pri['ret1']; f['ret4_p'] = pri['ret4']; f['ret8_p'] = pri['ret8']
    f['vol_p'] = pri['vol']; f['bb_p'] = pri['bb_w']
    f['ret1_1'] = sec1['ret1']; f['ret4_1'] = sec1['ret4']
    f['ret1_2'] = sec2['ret1']; f['ret4_2'] = sec2['ret4']

    f['ret1_spread_p_1'] = f['ret1_p'] - sec1['ret1']; f['ret1_spread_p_2'] = f['ret1_p'] - sec2['ret1']
    f['ret1_spread_1_2'] = sec1['ret1'] - sec2['ret1']
    f['ret4_spread_p_1'] = f['ret4_p'] - sec1['ret4']; f['ret4_spread_p_2'] = f['ret4_p'] - sec2['ret4']
    f['ret4_spread_1_2'] = sec1['ret4'] - sec2['ret4']

    f['ret1_prod_p_1'] = f['ret1_p'] * sec1['ret1']; f['ret1_prod_1_2'] = sec1['ret1'] * sec2['ret1']
    f['price_div_p_2'] = f['ret1_p'] * sec2['ret1']; f['price_div_abs'] = f['price_div_p_2'].abs()
    f['ret4_prod_p_1'] = f['ret4_p'] * sec1['ret4']; f['ret4_prod_p_2'] = f['ret4_p'] * sec2['ret4']
    f['ret4_prod_1_2'] = sec1['ret4'] * sec2['ret4']

    f['vol_spread_p_1'] = f['vol_p'] - sec1['vol']; f['vol_spread_p_2'] = f['vol_p'] - sec2['vol']
    f['vol_spread_1_2'] = sec1['vol'] - sec2['vol']
    f['bb_spread_p_1'] = f['bb_p'] - sec1['bb_w']; f['bb_spread_p_2'] = f['bb_p'] - sec2['bb_w']
    f['bb_spread_1_2'] = sec1['bb_w'] - sec2['bb_w']

    f['dist_sma50_p'] = pri['dist_sma50']; f['dist_sma50_1'] = sec1['dist_sma50']; f['dist_sma50_2'] = sec2['dist_sma50']
    f['sma50_slope_p'] = pri['sma50_slope']
    f['above_sma50_p'] = pri['above_sma50']; f['above_sma50_1'] = sec1['above_sma50']; f['above_sma50_2'] = sec2['above_sma50']
    f['sma50_alignment'] = f['above_sma50_p'] + f['above_sma50_1'] + f['above_sma50_2']
    f['sma50_dist_spread_p_1'] = f['dist_sma50_p'] - f['dist_sma50_1']
    f['sma50_dist_spread_p_2'] = f['dist_sma50_p'] - f['dist_sma50_2']
    f['sma50_dist_spread_1_2'] = f['dist_sma50_1'] - f['dist_sma50_2']
    f['sma50_align_p_1'] = f['above_sma50_p'] + f['above_sma50_1']
    f['sma50_align_p_2'] = f['above_sma50_p'] + f['above_sma50_2']
    f['sma50_align_1_2'] = f['above_sma50_1'] + f['above_sma50_2']

    f['dist_ema20_p'] = pri['dist_ema20']; f['dist_ema20_1'] = sec1['dist_ema20']; f['dist_ema20_2'] = sec2['dist_ema20']
    f['above_ema20_p'] = pri['above_ema20']; f['above_ema20_1'] = sec1['above_ema20']; f['above_ema20_2'] = sec2['above_ema20']
    f['ema20_bias_p_1'] = f['above_ema20_p'] + f['above_ema20_1']
    f['ema20_alignment'] = f['above_ema20_p'] + f['above_ema20_1'] + f['above_ema20_2']
    f['ema20_dist_spread_p_1'] = f['dist_ema20_p'] - f['dist_ema20_1']
    f['ema20_dist_spread_p_2'] = f['dist_ema20_p'] - f['dist_ema20_2']
    f['ema20_dist_spread_1_2'] = f['dist_ema20_1'] - f['dist_ema20_2']
    f['ema20_align_p_1'] = f['above_ema20_p'] + f['above_ema20_1']
    f['ema20_align_p_2'] = f['above_ema20_p'] + f['above_ema20_2']
    f['ema20_align_1_2'] = f['above_ema20_1'] + f['above_ema20_2']

    now = idx[-1]
    f['hour'] = idx.hour; f['dow'] = idx.dayofweek
    f['hour_sin'] = np.sin(2*np.pi*idx.hour/24); f['hour_cos'] = np.cos(2*np.pi*idx.hour/24)
    f['dow_sin']  = np.sin(2*np.pi*idx.dayofweek/7); f['dow_cos']  = np.cos(2*np.pi*idx.dayofweek/7)

    for k, v in key_levels.items():
        f[k] = v

    return f.iloc[-1:]


def main():
    saida = {'ts': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), 'assets': {}}

    try:
        # Fetch ALL data first (4 assets + 3 cross-ref)
        raw = {}
        for sym, ticker in SYMS.items():
            raw[sym] = fetch(ticker)
            if raw[sym] is None:
                saida['assets'][sym] = {'erro': f'Falha ao baixar {ticker}'}
                continue

        # Para key levels, precisamos de 60d
        raw_long = {}
        for sym in ['mnq', 'btc', 'cl', 'mgc', 'es']:
            t = SYMS[sym]
            raw_long[sym] = fetch_long(t)

        for asset in ['mnq', 'btc', 'cl', 'mgc', 'es']:
            if asset not in raw or raw[asset] is None:
                continue

            try:
                s = SETUP[asset]
                model_path = MODELOS[asset]
                if not model_path.exists():
                    saida['assets'][asset] = {'erro': 'Modelo nao encontrado'}
                    continue

                md = pickle.load(open(model_path, 'rb'))
                model = md['model']
                feats = md['features']
                fwd = md.get('forward', 8)

                # Primary
                pri = compute(raw[asset])

                # Secundarios: para primary=mnq, usamos btc e cl
                # Para primary=btc, usamos mnq e cl
                # Para primary=cl, usamos mnq e btc
                # Para primary=mgc, usamos mnq e btc
                # Para primary=es, usamos mnq e btc
                if asset == 'mnq':
                    s1 = compute(raw['btc']); s2 = compute(raw['cl'])
                elif asset == 'btc':
                    s1 = compute(raw['mnq']); s2 = compute(raw['cl'])
                elif asset == 'cl':
                    s1 = compute(raw['mnq']); s2 = compute(raw['btc'])
                elif asset == 'es':
                    s1 = compute(raw['mnq']); s2 = compute(raw['btc'])
                else:  # mgc
                    s1 = compute(raw['mnq']); s2 = compute(raw['btc'])

                # Key levels (precisa de 60d)
                if asset in raw_long and raw_long[asset] is not None:
                    kl = compute_key_levels_live(raw_long[asset])
                else:
                    # fallback: usar os proprios dados de 5d
                    kl = compute_key_levels_live(raw[asset])

                # Montar features
                X = build_features_live(pri, s1, s2, kl)
                common = [c for c in feats if c in X.columns]
                if len(common) < len(feats):
                    saida['assets'][asset] = {
                        'erro': f'Features incompletas: {len(common)}/{len(feats)}'
                    }
                    continue

                X_clean = X[common].fillna(0)

                # Extrai features-chave para exibir no live2 (usa X completo, nao so common)
                features_live = {}
                for feat in KEY_FEATURES.get(asset, []):
                    val = X[feat].iloc[0] if feat in X.columns else None
                    features_live[feat] = round(float(val), 4) if (val is not None and pd.notna(val)) else None

                proba = model.predict_proba(X_clean)[0]  # [SHORT, NEUTRO, LONG]
                prob_short = round(float(proba[0]) * 100, 1)
                prob_neutro = round(float(proba[1]) * 100, 1)
                prob_long = round(float(proba[2]) * 100, 1)

                is_long = prob_long > prob_short
                conf = max(prob_long, prob_short) / 100

                # Preco atual
                price = float(pri['close'].iloc[-1])
                atr = float(pri['atr14'].iloc[-1])

                # Performance do dia (UTC)
                perf_dia = None
                try:
                    today_dt = pri.index[-1].date()
                    today_bars = pri[[d.date() == today_dt for d in pri.index]]
                    yest_bars  = pri[[d.date() <  today_dt for d in pri.index]]
                    if len(today_bars) > 0 and len(yest_bars) > 0:
                        prev_close = float(yest_bars['close'].iloc[-1])
                        open_dia   = float(today_bars['open'].iloc[0])
                        high_dia   = float(today_bars['high'].max())
                        low_dia    = float(today_bars['low'].min())
                        pct_dia    = round((price - prev_close) / prev_close * 100, 2)
                        rng        = high_dia - low_dia
                        pos_range  = round((price - low_dia) / rng * 100, 1) if rng > 0 else 50.0
                        perf_dia   = {
                            'pct': pct_dia,
                            'open': round(open_dia, 2),
                            'high': round(high_dia, 2),
                            'low':  round(low_dia, 2),
                            'prev_close': round(prev_close, 2),
                            'pos_range': pos_range,
                        }
                except Exception:
                    pass
                if np.isnan(atr) or atr <= 0:
                    saida['assets'][asset] = {'erro': 'ATR invalido'}
                    continue

                # Check setup
                sinal = 'NO_TRADE'
                stop_price = target_price = risco_dolar = contratos = None

                if conf >= s['ml_min']:
                    dir_ok = False
                    if s['direcao'] == 'both':
                        dir_ok = True
                    elif s['direcao'] == 'long' and is_long:
                        dir_ok = True
                    elif s['direcao'] == 'short' and not is_long:
                        dir_ok = True

                    if dir_ok:
                        sinal = 'LONG' if is_long else 'SHORT'
                        R = atr
                        if sinal == 'LONG':
                            stop_price = round(price - s['stop_r'] * R, 2)
                            target_price = round(price + s['target_r'] * R, 2)
                        else:
                            stop_price = round(price + s['stop_r'] * R, 2)
                            target_price = round(price - s['target_r'] * R, 2)

                        # Risco em $
                        stop_dist_pct = s['stop_r'] * R / price
                        risco_dolar = round(CONTA * RISCO_PCT, 2)
                        # Contratos (valor nominal / preco)
                        valor_risco_por_contrato = stop_dist_pct * price
                        if valor_risco_por_contrato > 0:
                            contratos_raw = (risco_dolar / (stop_dist_pct * price))
                            contratos = max(1, round(contratos_raw))
                        else:
                            contratos = 1

                saida['assets'][asset] = {
                    'sinal': sinal,
                    'direcao': 'LONG' if is_long else 'SHORT',
                    'preco': price,
                    'stop': stop_price,
                    'target': target_price,
                    'conf_long': prob_long,
                    'conf_short': prob_short,
                    'conf': round(conf * 100, 1),
                    'atr': round(atr, 2),
                    'risco_dolar': risco_dolar,
                    'contratos': contratos,
                    'hora': pri.index[-1].strftime('%Y-%m-%d %H:%M'),
                    'forward_h': fwd,
                    'setup': s,
                    'features_live': features_live,
                }

            except Exception as e:
                saida['assets'][asset] = {'erro': str(e)}

        print(json.dumps(saida, default=str))

    except Exception as e:
        print(json.dumps({'erro': str(e)}, default=str))


if __name__ == '__main__':
    main()

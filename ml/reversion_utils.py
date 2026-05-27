"""
Utilitarios compartilhados entre train_reversion.py e predict_reversion.py.
Garantia: compute() e build_features() sao identicas em treino e predicao.
"""
import pandas as pd
import ta

TICKERS = {
    'mnq': 'MNQ=F',
    'es':  'ES=F',
    'cl':  'CL=F',
    'btc': 'BTC-USD',
}

INTERVAL   = '15m'
FWD_BARS   = 4        # 4 barras x 15m = 1H a frente
TARGET_PCT = 0.001    # 0.1%
RSI_OS     = 35
RSI_OB     = 65


def compute(df, rsi_w=14, adx_w=14, bb_w=20):
    d = df.copy()
    d['rsi']  = ta.momentum.RSIIndicator(d['close'], window=rsi_w).rsi()
    adx_i     = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=adx_w)
    d['adx']  = adx_i.adx()
    d['ret1'] = d['close'].pct_change(1)
    d['ret4'] = d['close'].pct_change(4)
    d['ret8'] = d['close'].pct_change(8)
    d['vol']  = d['ret1'].rolling(20).std()

    bb          = ta.volatility.BollingerBands(d['close'], window=bb_w, window_dev=2)
    d['bb_mid'] = bb.bollinger_mavg()
    d['bb_upp'] = bb.bollinger_hband()
    d['bb_low'] = bb.bollinger_lband()
    d['bb_w']   = (d['bb_upp'] - d['bb_low']) / d['bb_mid'] * 100
    d['bb_pctb']= bb.bollinger_pband()

    d['ema9']        = d['close'].ewm(span=9,  adjust=False).mean()
    d['ema20']       = d['close'].ewm(span=20, adjust=False).mean()
    d['sma50']       = d['close'].rolling(50).mean()
    d['dist_ema9']   = (d['close'] - d['ema9'])  / d['ema9']  * 100
    d['dist_ema20']  = (d['close'] - d['ema20']) / d['ema20'] * 100
    d['dist_sma50']  = (d['close'] - d['sma50']) / d['sma50'] * 100
    d['above_sma50'] = (d['close'] > d['sma50']).astype(int)
    d['vol_ratio']   = d['volume'] / d['volume'].rolling(20).mean()
    return d


def build_features(primary_name, computed, raws, idx):
    """
    Constroi feature matrix para um ativo primario, usando os outros 3 como contexto.
    DEVE ser identica entre treino e predicao — nao alterar sem atualizar ambos os scripts.
    """
    p     = computed[primary_name]
    p_raw = raws[primary_name]
    others = [n for n in TICKERS if n != primary_name]

    f = pd.DataFrame(index=idx)

    # Ativo primario — features de reversao
    f['rsi']         = p['rsi']
    f['bb_pctb']     = p['bb_pctb']
    f['bb_w']        = p['bb_w']
    f['adx']         = p['adx']
    f['dist_ema9']   = p['dist_ema9']
    f['dist_ema20']  = p['dist_ema20']
    f['dist_sma50']  = p['dist_sma50']
    f['ret1']        = p['ret1'] * 100
    f['ret4']        = p['ret4'] * 100
    f['ret8']        = p['ret8'] * 100
    f['vol_ratio']   = p['vol_ratio']
    f['dist_bb_low'] = p['bb_pctb'].clip(upper=0).abs()   # quanto abaixo da BB inf
    f['dist_bb_upp'] = (p['bb_pctb'] - 1).clip(lower=0)   # quanto acima da BB sup
    f['above_sma50'] = p['above_sma50']
    f['adx_baixo']   = (p['adx'] < 20).astype(int)
    f['vol_spike']   = (p['vol_ratio'] > 1.5).astype(int)
    f['rsi_extreme'] = ((p['rsi'] < 30) | (p['rsi'] > 70)).astype(int)

    # Contexto dos outros 3 ativos
    for n in others:
        f[f'rsi_{n}']  = computed[n]['rsi']
        f[f'ret1_{n}'] = computed[n]['ret1'] * 100
        f[f'adx_{n}']  = computed[n]['adx']
        f[f'div_{n}']  = p['rsi'] - computed[n]['rsi']  # divergencia RSI

    # Primario isolado no extremo (outros neutros) — sinal mais forte
    o_rsi = [computed[n]['rsi'] for n in others]
    f['sozinho_os'] = ((p['rsi'] < RSI_OS) &
                       (o_rsi[0] > 40) & (o_rsi[1] > 40) & (o_rsi[2] > 40)).astype(int)
    f['sozinho_ob'] = ((p['rsi'] > RSI_OB) &
                       (o_rsi[0] < 60) & (o_rsi[1] < 60) & (o_rsi[2] < 60)).astype(int)

    # Alinhamento SMA50 de todos os ativos (0-4)
    f['sma50_align'] = sum(computed[n]['above_sma50'] for n in TICKERS)

    # Distancia do open da hora atual
    open_1h = p_raw['open'].resample('1h').first().reindex(idx, method='ffill')
    f['dist_1h_open'] = (p['close'] - open_1h) / open_1h * 100

    # Tempo
    f['hour']    = idx.hour
    f['dow']     = idx.dayofweek
    f['is_us']   = ((idx.hour >= 13) & (idx.hour <= 21)).astype(int)
    f['is_asia'] = ((idx.hour >= 0)  & (idx.hour <= 8)).astype(int)

    return f

"""
Resolve outcomes dos sinais reais enviados pelo Telegram (signals_log.csv).
Para cada sinal pendente, baixa dados de preco e verifica se bateu stop ou target.
P&L em R: WIN = +target_r, LOSS = -stop_r, TIMEOUT = 0.

Uso:
  python ml/validate_live.py           # resolve todos os pendentes
  python ml/validate_live.py --backfill --days 30   # re-simula (sem log real)
"""
import argparse, warnings, pickle, time
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import ta

warnings.filterwarnings('ignore')

TESTE_DIR   = Path(__file__).parent / 'teste'
SIGNALS_LOG = Path(__file__).parent / 'signals_log.csv'
OUTPUT_CSV  = Path(__file__).parent / 'live_performance.csv'
MAX_BARS    = 8

SYMS = {'mnq': 'MNQ=F', 'btc': 'BTC-USD', 'cl': 'CL=F', 'es': 'ES=F'}

ASSET_CONFIG = {
    'mnq': {'ticker': 'MNQ=F',   'sec': ['btc', 'cl'], 'stop_r': 1.5, 'target_r': 3.0, 'direcao': 'both'},
    'btc': {'ticker': 'BTC-USD', 'sec': ['mnq', 'cl'], 'stop_r': 1.5, 'target_r': 3.0, 'direcao': 'short'},
    'cl':  {'ticker': 'CL=F',    'sec': ['mnq', 'btc'], 'stop_r': 1.5, 'target_r': 2.0, 'direcao': 'both'},
    'es':  {'ticker': 'ES=F',    'sec': ['mnq', 'btc'], 'stop_r': 1.5, 'target_r': 3.0, 'direcao': 'both'},
}


# ── Download ──────────────────────────────────────────────────────────────────

def fetch(ticker, period='60d', retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval='1h',
                             auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = df.columns.str.lower()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(3 * (attempt + 1))
    return pd.DataFrame()


def compute_atr(df):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()


# ── Resolucao de outcome ──────────────────────────────────────────────────────

def resolve_signal(df_raw, signal_ts, direction, entry, stop, target):
    """
    Verifica as proximas MAX_BARS barras e retorna 'WIN', 'LOSS' ou 'TIMEOUT'.
    O pnl_r deve ser calculado pelo chamador usando stop_r/target_r do sinal.
    """
    try:
        pos = df_raw.index.searchsorted(signal_ts)
    except Exception:
        return 'TIMEOUT'

    if pos >= len(df_raw):
        return 'TIMEOUT'

    entry  = float(entry)
    stop   = float(stop)
    target = float(target)

    for i in range(pos + 1, min(pos + 1 + MAX_BARS, len(df_raw))):
        h = float(df_raw['high'].iloc[i])
        l = float(df_raw['low'].iloc[i])
        if direction == 'LONG':
            if h >= target: return 'WIN'
            if l <= stop:   return 'LOSS'
        else:
            if l <= target: return 'WIN'
            if h >= stop:   return 'LOSS'

    return 'TIMEOUT'


# ── Modo principal: resolver sinais do log ────────────────────────────────────

def resolve_from_log():
    if not SIGNALS_LOG.exists():
        print('[validate_live] signals_log.csv nao encontrado.')
        print('  Os sinais serao gravados automaticamente a partir do proximo envio pelo Telegram.')
        return

    log = pd.read_csv(SIGNALS_LOG)
    log['ts'] = pd.to_datetime(log['ts'])
    print(f'[validate_live] {len(log)} sinais no log')

    # Carregar ja resolvidos
    if OUTPUT_CSV.exists():
        perf = pd.read_csv(OUTPUT_CSV)
        if 'asset' in perf.columns:
            resolved_keys = set(zip(perf['ts'], perf['asset'], perf['direction']))
        else:
            resolved_keys = set()
            perf = pd.DataFrame()
    else:
        perf = pd.DataFrame()
        resolved_keys = set()

    now = datetime.now()
    cutoff = now - timedelta(hours=MAX_BARS)

    # Sinais pendentes: nao resolvidos e com tempo suficiente passado
    pending = log[
        ~log.apply(lambda r: (r['ts'].strftime('%Y-%m-%d %H:%M'), r['asset'], r['direction']) in resolved_keys, axis=1) &
        (log['ts'] <= cutoff)
    ].copy()

    if pending.empty:
        print(f'[validate_live] Nenhum sinal pendente para resolver.')
        ainda = log[log['ts'] > cutoff]
        if not ainda.empty:
            print(f'  {len(ainda)} sinal(is) ainda em progresso (ultimas {MAX_BARS}h) — aguardar.')
        return

    print(f'[validate_live] {len(pending)} sinal(is) para resolver...')

    # Baixar dados por ativo necessario
    assets_needed = pending['asset'].unique()
    raw = {}
    for asset in assets_needed:
        ticker = SYMS.get(asset)
        if not ticker:
            continue
        print(f'  Baixando {ticker}...')
        df = fetch(ticker)
        if not df.empty:
            raw[asset] = df
        else:
            print(f'  ERRO ao baixar {ticker}')

    # Resolver cada sinal
    new_rows = []
    for _, sig in pending.iterrows():
        asset = sig['asset']
        if asset not in raw:
            continue

        df_raw    = raw[asset]
        sig_ts    = sig['ts']
        direction = sig['direction']
        entry     = float(sig['entry'])  if pd.notna(sig.get('entry', ''))  else None
        stop      = float(sig['stop'])   if pd.notna(sig.get('stop', ''))   else None
        target    = float(sig['target']) if pd.notna(sig.get('target', '')) else None
        stop_r    = float(sig['stop_r']) if pd.notna(sig.get('stop_r', 1.5)) else 1.5
        target_r  = float(sig['target_r']) if pd.notna(sig.get('target_r', 2.0)) else 2.0

        if entry is None or stop is None or target is None:
            continue

        result = resolve_signal(df_raw, sig_ts, direction, entry, stop, target)
        if result == 'WIN':
            pnl_r = target_r
        elif result == 'LOSS':
            pnl_r = -stop_r
        else:
            pnl_r = 0.0

        # Calcular ATR na hora do sinal para referencia
        pos = df_raw.index.searchsorted(sig_ts)
        atr_ser = compute_atr(df_raw)
        atr_val = float(atr_ser.iloc[min(pos, len(atr_ser)-1)]) if pos < len(atr_ser) else 0.0

        new_rows.append({
            'ts':        sig_ts.strftime('%Y-%m-%d %H:%M'),
            'asset':     asset,
            'direction': direction,
            'entry':     round(entry, 3),
            'target':    round(target, 3),
            'stop':      round(stop, 3),
            'result':    result,
            'prob':      round(float(sig.get('prob', 0) or 0), 4),
            'hour':      sig_ts.hour,
            'dow':       sig_ts.dayofweek,
            'atr':       round(atr_val, 4),
            'stop_r':    stop_r,
            'target_r':  target_r,
            'pnl_r':     round(pnl_r, 2),
        })
        print(f'  {asset.upper()} {direction} {sig_ts.strftime("%m/%d %H:%M")} -> {result}  ({pnl_r:+.2f}R)')

    if not new_rows:
        print('[validate_live] Nenhum resultado calculado.')
        return

    new_df = pd.DataFrame(new_rows)

    if not perf.empty and 'asset' in perf.columns:
        combined = pd.concat([perf, new_df]).drop_duplicates(subset=['ts', 'asset', 'direction'])
    else:
        combined = new_df

    combined = combined.sort_values('ts').reset_index(drop=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    # Resumo
    print(f'\n{"="*55}')
    print(f'  RESUMO — {len(new_rows)} sinal(is) resolvidos')
    print(f'{"="*55}')
    for asset in ['mnq', 'btc', 'cl', 'es']:
        sub = new_df[new_df['asset'] == asset]
        if sub.empty: continue
        w = (sub['result'] == 'WIN').sum()
        l = (sub['result'] == 'LOSS').sum()
        t = (sub['result'] == 'TIMEOUT').sum()
        wr = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f'  {asset.upper():<5} N={len(sub):>3}  W={w} L={l} T={t}  WR={wr:.0f}%  P&L={sub["pnl_r"].sum():+.1f}R')
    print(f'\n  CSV: {len(combined)} entradas acumuladas -> {OUTPUT_CSV.name}')
    print(f'{"="*55}')


# ── Modo backfill: re-simulacao (sem log real) ────────────────────────────────

def compute_indicators(df):
    d = df.copy()
    d['rsi'] = ta.momentum.RSIIndicator(d['close'], window=21).rsi()
    adx_i    = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=17)
    d['adx'] = adx_i.adx(); d['pdi'] = adx_i.adx_pos(); d['mdi'] = adx_i.adx_neg()
    d['ret1'] = d['close'].pct_change(1); d['ret4'] = d['close'].pct_change(4)
    d['ret8'] = d['close'].pct_change(8)
    d['vol']  = d['ret1'].rolling(20).std()
    d['bb_w'] = d['close'].rolling(20).std() * 2 / d['close'].rolling(20).mean()
    d['sma50']       = d['close'].rolling(50).mean()
    d['dist_sma50']  = (d['close'] - d['sma50']) / d['sma50'] * 100
    d['sma50_slope'] = d['sma50'].pct_change(5) * 100
    d['above_sma50'] = (d['close'] > d['sma50']).astype(int)
    d['ema20']       = d['close'].ewm(span=20, adjust=False).mean()
    d['dist_ema20']  = (d['close'] - d['ema20']) / d['ema20'] * 100
    d['above_ema20'] = (d['close'] > d['ema20']).astype(int)
    hl = d['high'] - d['low']
    hc = (d['high'] - d['close'].shift()).abs()
    lc = (d['low']  - d['close'].shift()).abs()
    d['atr14'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    return d


def add_key_levels(primary_df, f):
    idx = f.index
    c   = primary_df['close'].reindex(idx, method='ffill')
    def pct(series, ref):
        return (series - ref.replace(0, float('nan'))) / ref.replace(0, float('nan')) * 100
    daily = primary_df.resample('1D').agg({'open':'first','high':'max','low':'min'})
    pdh = daily['high'].shift(1).reindex(idx, method='ffill')
    pdl = daily['low'].shift(1).reindex(idx, method='ffill')
    do_ = daily['open'].reindex(idx, method='ffill')
    f['dist_to_pdh']=pct(c,pdh); f['dist_to_pdl']=pct(c,pdl)
    f['dist_to_do']=pct(c,do_); f['above_do']=(c>do_).astype(int)
    f['above_pdh']=(c>pdh).astype(int); f['above_pdl']=(c>pdl).astype(int)
    f['prev_day_range_pct']=pct(daily['high'].shift(1),daily['low'].shift(1)).reindex(idx,method='ffill')
    weekly=primary_df.resample('W-SUN').agg({'open':'first','high':'max','low':'min'})
    pwh=weekly['high'].shift(1).reindex(idx,method='ffill'); pwl=weekly['low'].shift(1).reindex(idx,method='ffill')
    wo=weekly['open'].reindex(idx,method='ffill')
    f['dist_to_pwh']=pct(c,pwh); f['dist_to_pwl']=pct(c,pwl)
    f['dist_to_wo']=pct(c,wo); f['above_wo']=(c>wo).astype(int)
    f['above_pwh']=(c>pwh).astype(int); f['above_pwl']=(c>pwl).astype(int)
    monthly=primary_df.resample('MS').agg({'open':'first','high':'max','low':'min'})
    pmh=monthly['high'].shift(1).reindex(idx,method='ffill'); pml=monthly['low'].shift(1).reindex(idx,method='ffill')
    mo=monthly['open'].reindex(idx,method='ffill')
    f['dist_to_pmh']=pct(c,pmh); f['dist_to_pml']=pct(c,pml)
    f['dist_to_mo']=pct(c,mo); f['above_mo']=(c>mo).astype(int)
    f['above_pmh']=(c>pmh).astype(int); f['above_pml']=(c>pml).astype(int)
    mb=primary_df[primary_df.index.dayofweek==0]
    if len(mb)>=4:
        mh=mb['high'].resample('W-SUN').max().reindex(idx,method='ffill')
        ml_=mb['low'].resample('W-SUN').min().reindex(idx,method='ffill')
        f['dist_to_mday_h']=pct(c,mh); f['dist_to_mday_l']=pct(c,ml_)
        f['above_mday_h']=(c>mh).astype(int); f['above_mday_l']=(c>ml_).astype(int)
    else:
        for col in ['dist_to_mday_h','dist_to_mday_l','above_mday_h','above_mday_l']: f[col]=0.0


def build_features(pri, sec1, sec2):
    idx=pri.index; sec1=sec1.reindex(idx,method='ffill'); sec2=sec2.reindex(idx,method='ffill')
    f=pd.DataFrame(index=idx)
    f['rsi_p']=pri['rsi']; f['rsi_1']=sec1['rsi']; f['rsi_2']=sec2['rsi']
    f['div_1']=f['rsi_p']-f['rsi_1']; f['div_2']=f['rsi_p']-f['rsi_2']
    f['rsi_spread_1_2']=f['rsi_1']-f['rsi_2']
    f['adx_p']=pri['adx']; f['adx_1']=sec1['adx']; f['adx_2']=sec2['adx']
    f['pdi_p']=pri['pdi']; f['mdi_p']=pri['mdi']
    f['di_spread_p']=f['pdi_p']-f['mdi_p']; f['di_spread_1']=sec1['pdi']-sec1['mdi']; f['di_spread_2']=sec2['pdi']-sec2['mdi']
    f['dadx_p']=f['adx_p'].diff(2)
    f['ret1_p']=pri['ret1']; f['ret4_p']=pri['ret4']; f['ret8_p']=pri['ret8']
    f['vol_p']=pri['vol']; f['bb_p']=pri['bb_w']
    f['ret1_1']=sec1['ret1']; f['ret4_1']=sec1['ret4']
    f['ret1_2']=sec2['ret1']; f['ret4_2']=sec2['ret4']
    f['ret1_spread_p_1']=f['ret1_p']-sec1['ret1']; f['ret1_spread_p_2']=f['ret1_p']-sec2['ret1']
    f['ret1_spread_1_2']=sec1['ret1']-sec2['ret1']
    f['ret4_spread_p_1']=f['ret4_p']-sec1['ret4']; f['ret4_spread_p_2']=f['ret4_p']-sec2['ret4']
    f['ret1_prod_p_1']=f['ret1_p']*sec1['ret1']; f['ret1_prod_1_2']=sec1['ret1']*sec2['ret1']
    f['price_div_p_2']=f['ret1_p']*sec2['ret1']; f['price_div_abs']=f['price_div_p_2'].abs()
    f['ret4_prod_p_1']=f['ret4_p']*sec1['ret4']; f['ret4_prod_p_2']=f['ret4_p']*sec2['ret4']
    f['ret4_prod_1_2']=sec1['ret4']*sec2['ret4']
    f['vol_spread_p_1']=f['vol_p']-sec1['vol']; f['vol_spread_p_2']=f['vol_p']-sec2['vol']
    f['vol_spread_1_2']=sec1['vol']-sec2['vol']
    f['bb_spread_p_1']=f['bb_p']-sec1['bb_w']; f['bb_spread_p_2']=f['bb_p']-sec2['bb_w']
    f['bb_spread_1_2']=sec1['bb_w']-sec2['bb_w']
    f['dist_sma50_p']=pri['dist_sma50']; f['dist_sma50_1']=sec1['dist_sma50']; f['dist_sma50_2']=sec2['dist_sma50']
    f['sma50_slope_p']=pri['sma50_slope']
    f['above_sma50_p']=pri['above_sma50']; f['above_sma50_1']=sec1['above_sma50']; f['above_sma50_2']=sec2['above_sma50']
    f['sma50_alignment']=f['above_sma50_p']+f['above_sma50_1']+f['above_sma50_2']
    f['sma50_dist_spread_p_1']=f['dist_sma50_p']-sec1['dist_sma50']; f['sma50_dist_spread_p_2']=f['dist_sma50_p']-sec2['dist_sma50']
    f['sma50_align_p_1']=f['above_sma50_p']+f['above_sma50_1']; f['sma50_align_p_2']=f['above_sma50_p']+f['above_sma50_2']
    f['dist_ema20_p']=pri['dist_ema20']; f['dist_ema20_1']=sec1['dist_ema20']; f['dist_ema20_2']=sec2['dist_ema20']
    f['above_ema20_p']=pri['above_ema20']; f['above_ema20_1']=sec1['above_ema20']; f['above_ema20_2']=sec2['above_ema20']
    f['ema20_bias_p_1']=f['above_ema20_p']+f['above_ema20_1']
    f['ema20_alignment']=f['above_ema20_p']+f['above_ema20_1']+f['above_ema20_2']
    f['ema20_dist_spread_p_1']=f['dist_ema20_p']-sec1['dist_ema20']; f['ema20_dist_spread_p_2']=f['dist_ema20_p']-sec2['dist_ema20']
    f['ema20_align_p_1']=f['above_ema20_p']+f['above_ema20_1']; f['ema20_align_p_2']=f['above_ema20_p']+f['above_ema20_2']
    f['ema20_align_1_2']=f['above_ema20_1']+f['above_ema20_2']
    f['hour']=idx.hour; f['dow']=idx.dayofweek
    f['hour_sin']=np.sin(2*np.pi*idx.hour/24); f['hour_cos']=np.cos(2*np.pi*idx.hour/24)
    f['dow_sin']=np.sin(2*np.pi*idx.dayofweek/7); f['dow_cos']=np.cos(2*np.pi*idx.dayofweek/7)
    add_key_levels(pri, f)
    return f


def simulate_atr(df_raw, pos, direction, stop_r, target_r):
    entry = float(df_raw['close'].iloc[pos])
    atr   = float(df_raw['atr14'].iloc[pos])
    if np.isnan(atr) or atr <= 0:
        return 'TIMEOUT', entry, entry, entry, atr, 0.0
    tgt = entry + target_r*atr if direction=='LONG' else entry - target_r*atr
    stp = entry - stop_r*atr  if direction=='LONG' else entry + stop_r*atr
    for i in range(pos+1, min(pos+1+MAX_BARS, len(df_raw))):
        h = float(df_raw['high'].iloc[i]); l = float(df_raw['low'].iloc[i])
        if direction=='LONG':
            if h>=tgt: return 'WIN',  entry, tgt, stp, atr,  target_r
            if l<=stp: return 'LOSS', entry, tgt, stp, atr, -stop_r
        else:
            if l<=tgt: return 'WIN',  entry, tgt, stp, atr,  target_r
            if h>=stp: return 'LOSS', entry, tgt, stp, atr, -stop_r
    return 'TIMEOUT', entry, tgt, stp, atr, 0.0


def backfill(days, min_prob):
    print(f'[backfill] Re-simulando {days} dias (sem log real)...')
    now = datetime.now()
    cutoff_start = now - timedelta(days=days)
    cutoff_end   = now - timedelta(hours=MAX_BARS)

    raw = {}
    for sym, ticker in SYMS.items():
        df = fetch(ticker)
        raw[sym] = compute_indicators(df) if not df.empty else None

    computed = {k: v for k, v in raw.items() if v is not None}
    all_rows = []

    for asset in ['mnq','btc','cl','es']:
        model_path = TESTE_DIR / f'propfirm_model_{asset}.pkl'
        if not model_path.exists() or asset not in computed: continue
        md = pickle.load(open(model_path,'rb'))
        model = md['model']; feats = md['features']
        cfg = ASSET_CONFIG[asset]
        s1, s2 = cfg['sec']
        f = build_features(computed[asset], computed[s1], computed[s2])
        common = [c for c in feats if c in f.columns]
        X = f[common].fillna(0)
        valid = X.dropna()
        if valid.empty: continue
        probas = model.predict_proba(valid)
        mask = (valid.index >= cutoff_start) & (valid.index <= cutoff_end)
        for ts in valid.index[mask]:
            loc = valid.index.get_loc(ts)
            if isinstance(loc, slice): loc = loc.start
            prob_l = float(probas[loc][2]); prob_s = float(probas[loc][0])
            is_long = prob_l > prob_s
            direction = 'LONG' if is_long else 'SHORT'
            conf = prob_l if is_long else prob_s
            if conf < min_prob: continue
            d = cfg['direcao']
            if d=='long' and not is_long: continue
            if d=='short' and is_long: continue
            pos_raw = raw[asset].index.searchsorted(ts)
            if pos_raw >= len(raw[asset])-1: continue
            result, entry, tgt, stp, atr, pnl_r = simulate_atr(raw[asset], pos_raw, direction, cfg['stop_r'], cfg['target_r'])
            all_rows.append({'ts':ts.strftime('%Y-%m-%d %H:%M'),'asset':asset,'direction':direction,
                'entry':round(entry,3),'target':round(tgt,3),'stop':round(stp,3),
                'result':result,'prob':round(conf,4),'hour':ts.hour,'dow':ts.dayofweek,
                'atr':round(atr,4),'stop_r':cfg['stop_r'],'target_r':cfg['target_r'],'pnl_r':round(pnl_r,2)})

    if not all_rows:
        print('Nenhum sinal encontrado.'); return

    new_df = pd.DataFrame(all_rows)
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        if 'asset' in existing.columns:
            combined = pd.concat([existing, new_df]).drop_duplicates(subset=['ts','asset','direction'])
        else:
            combined = new_df
    else:
        combined = new_df

    combined = combined.sort_values('ts').reset_index(drop=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    for asset in ['mnq','btc','cl','es']:
        sub = new_df[new_df['asset']==asset]
        if sub.empty: continue
        w=(sub['result']=='WIN').sum(); l=(sub['result']=='LOSS').sum(); t=(sub['result']=='TIMEOUT').sum()
        wr = w/(w+l)*100 if (w+l)>0 else 0
        print(f'  {asset.upper():<5} N={len(sub):>3}  W={w} L={l} T={t}  WR={wr:.0f}%  P&L={sub["pnl_r"].sum():+.1f}R')
    print(f'  CSV: {len(combined)} entradas -> {OUTPUT_CSV.name}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backfill',  action='store_true', help='Re-simular sem log real')
    parser.add_argument('--days',      type=int,   default=30)
    parser.add_argument('--min-prob',  type=float, default=0.55)
    args = parser.parse_args()

    if args.backfill:
        backfill(args.days, args.min_prob)
    else:
        resolve_from_log()


if __name__ == '__main__':
    main()

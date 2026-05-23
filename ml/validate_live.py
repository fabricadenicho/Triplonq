"""
Valida sinais ML do CL contra dados reais.

Para cada bar com sinal LONG/SHORT, simula trade com stop $1.00 / target $3.00
e regista resultado em ml/live_performance.csv.

Uso:
  python ml/validate_live.py             # ultimos 2 dias
  python ml/validate_live.py --days 7   # ultimos 7 dias
  python ml/validate_live.py --days 30  # acumulado 30 dias
"""
import argparse, warnings, pickle
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import ta

warnings.filterwarnings('ignore')

MODEL_PATH = Path(__file__).parent / 'cl' / 'model.pkl'
OUTPUT_CSV = Path(__file__).parent / 'live_performance.csv'
STOP_PTS   = 1.00   # stop em $ por barril
TARGET_PTS = 3.00   # target em $ por barril
MAX_BARS   = 8      # max horas em trade antes de TIMEOUT


# ── Dados ────────────────────────────────────────────────────────────────────

def fetch(ticker, period='60d'):
    df = yf.download(ticker, period=period, interval='1h',
                     auto_adjust=True, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])


# ── Indicadores ───────────────────────────────────────────────────────────────

def indicators(df):
    d = df.copy()
    d['rsi'] = ta.momentum.RSIIndicator(d['close'], window=21).rsi()
    adx_i    = ta.trend.ADXIndicator(d['high'], d['low'], d['close'], window=17)
    d['adx'] = adx_i.adx()
    d['pdi'] = adx_i.adx_pos()
    d['mdi'] = adx_i.adx_neg()
    d['ret1'] = d['close'].pct_change(1)
    d['ret4'] = d['close'].pct_change(4)
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
    return d


# ── Key levels vetorizados ────────────────────────────────────────────────────

def key_levels_vec(df):
    idx = df.index
    c   = df['close']

    def spct(ref_s):
        al = ref_s.reindex(idx, method='ffill')
        return (c - al) / al.replace(0, np.nan) * 100

    def sabove(ref_s):
        al = ref_s.reindex(idx, method='ffill')
        return (c > al).astype(int)

    out = pd.DataFrame(index=idx)

    daily  = df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pdh_s  = daily['high'].shift(1)
    pdl_s  = daily['low'].shift(1)
    do_s   = daily['open']
    out['dist_to_pdh']      = spct(pdh_s)
    out['dist_to_pdl']      = spct(pdl_s)
    out['dist_to_do']       = spct(do_s)
    out['above_do']         = sabove(do_s)
    out['above_pdh']        = sabove(pdh_s)
    out['above_pdl']        = sabove(pdl_s)
    pdh_al = pdh_s.reindex(idx, method='ffill')
    pdl_al = pdl_s.reindex(idx, method='ffill')
    out['prev_day_range_pct'] = (pdh_al - pdl_al) / pdl_al.replace(0, np.nan) * 100

    weekly = df.resample('W-SUN').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pwh_s  = weekly['high'].shift(1)
    pwl_s  = weekly['low'].shift(1)
    wo_s   = weekly['open']
    out['dist_to_pwh'] = spct(pwh_s)
    out['dist_to_pwl'] = spct(pwl_s)
    out['dist_to_wo']  = spct(wo_s)
    out['above_wo']    = sabove(wo_s)
    out['above_pwh']   = sabove(pwh_s)
    out['above_pwl']   = sabove(pwl_s)

    monthly = df.resample('MS').agg({'open': 'first', 'high': 'max', 'low': 'min'})
    pmh_s   = monthly['high'].shift(1)
    pml_s   = monthly['low'].shift(1)
    mo_s    = monthly['open']
    out['dist_to_pmh'] = spct(pmh_s)
    out['dist_to_pml'] = spct(pml_s)
    out['dist_to_mo']  = spct(mo_s)
    out['above_mo']    = sabove(mo_s)
    out['above_pmh']   = sabove(pmh_s)
    out['above_pml']   = sabove(pml_s)

    monday_bars = df[df.index.dayofweek == 0]
    if len(monday_bars) >= 4:
        mday_h_s = monday_bars['high'].resample('W-SUN').max()
        mday_l_s = monday_bars['low'].resample('W-SUN').min()
        out['dist_to_mday_h'] = spct(mday_h_s)
        out['dist_to_mday_l'] = spct(mday_l_s)
        out['above_mday_h']   = sabove(mday_h_s)
        out['above_mday_l']   = sabove(mday_l_s)
    else:
        out[['dist_to_mday_h', 'dist_to_mday_l']] = 0.0
        out[['above_mday_h', 'above_mday_l']]      = 0

    return out.fillna(0)


# ── Kill zones vetorizados ────────────────────────────────────────────────────

def kill_zone_vec(df):
    ASIA   = list(range(0, 8))
    LONDON = list(range(8, 15))
    NY     = list(range(13, 20))
    OVLP   = [13, 14]

    h       = df.index.hour
    in_asia = pd.Series(np.isin(h, ASIA),   index=df.index)
    in_lon  = pd.Series(np.isin(h, LONDON), index=df.index)
    in_ny   = pd.Series(np.isin(h, NY),     index=df.index)
    in_any  = in_asia | in_lon | in_ny

    kz_id = (in_any & ~in_any.shift(1).fillna(False)).cumsum()
    kz_id[~in_any] = -1

    sh = df['high'].groupby(kz_id).cummax()
    sl = df['low'].groupby(kz_id).cummin()

    out = pd.DataFrame(index=df.index)
    out['is_asia']   = in_asia.astype(int)
    out['is_london'] = in_lon.astype(int)
    out['is_ny']     = in_ny.astype(int)
    out['kz_overlap'] = pd.Series(np.isin(h, OVLP), index=df.index).astype(int)
    out['kz_dist_high'] = np.where(in_any, (df['close'] - sh) / sh.replace(0, np.nan) * 100, 0.0)
    out['kz_dist_low']  = np.where(in_any, (df['close'] - sl) / sl.replace(0, np.nan) * 100, 0.0)
    out['kz_range']     = np.where(in_any, (sh - sl) / sl.replace(0, np.nan) * 100, 0.0)

    c  = df['close']
    c1 = df['close'].shift(1)
    ps = sh.shift(1); pl = sl.shift(1)
    out['kz_breakout_up'] = (in_any & (c > ps) & (c1 <= ps)).astype(int)
    out['kz_breakout_dn'] = (in_any & (c < pl) & (c1 >= pl)).astype(int)

    grp_count = kz_id.groupby(kz_id).transform('count')
    grp_pos   = kz_id.groupby(kz_id).cumcount()
    out['kz_progress'] = np.where(in_any, grp_pos / (grp_count - 1).clip(lower=1), 0.0)

    return out.fillna(0)


# ── Feature matrix completa ───────────────────────────────────────────────────

def build_features(cl, mnq, btc):
    idx = cl.index
    mnq_r = mnq.reindex(idx, method='ffill')
    btc_r = btc.reindex(idx, method='ffill')

    def col(df_r, c):
        return df_r[c].fillna(0)

    f = pd.DataFrame(index=idx)

    # Individuais
    for name, src in [('cl', cl), ('mnq', mnq_r), ('btc', btc_r)]:
        f[f'rsi_{name}']  = col(src, 'rsi')
        f[f'adx_{name}']  = col(src, 'adx')
        f[f'ret1_{name}'] = col(src, 'ret1')
        f[f'ret4_{name}'] = col(src, 'ret4')
        f[f'dist_sma50_{name}'] = col(src, 'dist_sma50')
        f[f'above_sma50_{name}'] = col(src, 'above_sma50')
        f[f'dist_ema20_{name}'] = col(src, 'dist_ema20')
        f[f'above_ema20_{name}'] = col(src, 'above_ema20')

    f['pdi_cl']   = col(cl, 'pdi')
    f['mdi_cl']   = col(cl, 'mdi')
    f['ret8_cl']  = col(cl, 'ret8')
    f['vol_cl']   = col(cl, 'vol')
    f['bb_cl']    = col(cl, 'bb_w')
    f['div_mnq']  = f['rsi_cl'] - f['rsi_mnq']
    f['div_btc']  = f['rsi_cl'] - f['rsi_btc']
    f['hour']     = idx.hour
    f['dow']      = idx.dayofweek
    f['dadx_cl']  = col(cl, 'adx') - col(cl, 'adx').shift(2).fillna(0)
    f['sma50_slope_cl'] = col(cl, 'sma50_slope')

    f['sma50_alignment']    = f['above_sma50_cl'] + f['above_sma50_mnq'] + f['above_sma50_btc']
    f['ema20_bias_mnq_btc'] = f['above_ema20_cl'] + f['above_ema20_btc']
    f['ema20_alignment']    = f['above_ema20_cl'] + f['above_ema20_mnq'] + f['above_ema20_btc']

    # Spreads
    r1c = f['ret1_cl']; r1m = f['ret1_mnq']; r1b = f['ret1_btc']
    r4c = f['ret4_cl']; r4m = f['ret4_mnq']; r4b = f['ret4_btc']

    f['triple_signal'] = ((f['rsi_btc'] < 45) & (f['div_mnq'] < 0) & (f['div_btc'] < 0)).astype(int)
    f['rsi_spread_mnq_btc'] = f['rsi_mnq'] - f['rsi_btc']
    f['rsi_abs_cl_mnq']     = f['div_mnq'].abs()
    f['rsi_abs_cl_btc']     = f['div_btc'].abs()
    f['rsi_abs_mnq_btc']    = f['rsi_spread_mnq_btc'].abs()

    f['adx_spread_cl_mnq']  = f['adx_cl'] - f['adx_mnq']
    f['adx_spread_cl_btc']  = f['adx_cl'] - f['adx_btc']
    f['adx_spread_mnq_btc'] = f['adx_mnq'] - f['adx_btc']
    f['adx_abs_cl_mnq']     = f['adx_spread_cl_mnq'].abs()
    f['adx_abs_cl_btc']     = f['adx_spread_cl_btc'].abs()
    f['adx_abs_mnq_btc']    = f['adx_spread_mnq_btc'].abs()

    f['ret1_spread_cl_mnq']  = r1c - r1m
    f['ret1_spread_cl_btc']  = r1c - r1b
    f['ret1_spread_mnq_btc'] = r1m - r1b
    f['ret4_spread_cl_mnq']  = r4c - r4m
    f['ret4_spread_cl_btc']  = r4c - r4b
    f['ret4_spread_mnq_btc'] = r4m - r4b

    f['ret1_prod_cl_mnq']  = r1c * r1m
    f['ret1_prod_cl_btc']  = r1c * r1b
    f['price_div_cl']      = r1c * r1m
    f['price_div_abs']     = f['price_div_cl'].abs()
    f['ret4_prod_cl_mnq']  = r4c * r4m
    f['ret4_prod_cl_btc']  = r4c * r4b
    f['ret4_prod_mnq_btc'] = r4m * r4b

    vc = col(cl, 'vol'); vm = col(mnq_r, 'vol'); vb = col(btc_r, 'vol')
    f['vol_spread_cl_mnq']  = vc - vm
    f['vol_spread_cl_btc']  = vc - vb
    f['vol_spread_mnq_btc'] = vm - vb

    bc = col(cl, 'bb_w'); bm = col(mnq_r, 'bb_w'); bb = col(btc_r, 'bb_w')
    f['bb_spread_cl_mnq']  = bc - bm
    f['bb_spread_cl_btc']  = bc - bb
    f['bb_spread_mnq_btc'] = bm - bb

    m50c = f['dist_sma50_cl']; m50m = f['dist_sma50_mnq']; m50b = f['dist_sma50_btc']
    f['sma50_dist_spread_cl_mnq']  = m50c - m50m
    f['sma50_dist_spread_cl_btc']  = m50c - m50b
    f['sma50_dist_spread_mnq_btc'] = m50m - m50b

    e20c = f['dist_ema20_cl']; e20m = f['dist_ema20_mnq']; e20b = f['dist_ema20_btc']
    f['ema20_dist_spread_cl_mnq']  = e20c - e20m
    f['ema20_dist_spread_cl_btc']  = e20c - e20b
    f['ema20_dist_spread_mnq_btc'] = e20m - e20b

    ac = f['above_sma50_cl']; am = f['above_sma50_mnq']; ab = f['above_sma50_btc']
    f['sma50_align_cl_mnq']  = ac + am
    f['sma50_align_cl_btc']  = ac + ab
    f['sma50_align_mnq_btc'] = am + ab

    ae = f['above_ema20_cl']; aem = f['above_ema20_mnq']; aeb = f['above_ema20_btc']
    f['ema20_align_cl_mnq']  = ae + aem
    f['ema20_align_cl_btc']  = ae + aeb
    f['ema20_align_mnq_btc'] = aem + aeb

    f['di_spread_cl']  = f['pdi_cl'] - f['mdi_cl']
    f['di_spread_mnq'] = col(mnq_r, 'pdi') - col(mnq_r, 'mdi')
    f['di_spread_btc'] = col(btc_r, 'pdi') - col(btc_r, 'mdi')

    # Composite triggers
    adx = f['adx_cl']
    f['adx_above_14']  = (adx > 14).astype(int)
    f['adx_above_20']  = (adx > 20).astype(int)
    f['adx_above_25']  = (adx > 25).astype(int)
    f['is_evening']    = pd.Series(np.isin(idx.hour, [18,19,20,21]), index=idx).astype(int)
    f['strong_div']    = ((f['price_div_cl'] < 0) & (adx > 14)).astype(int)
    f['prime_setup']   = ((f['price_div_cl'] < 0) & (adx > 14) & f['is_evening'].astype(bool)).astype(int)
    f['cl_down_mnq_up'] = ((r1c < 0) & (r1m > 0)).astype(int)
    f['is_us_session']   = pd.Series(np.isin(idx.hour, range(9, 18)), index=idx).astype(int)
    f['is_us_morning']   = pd.Series(np.isin(idx.hour, range(9, 14)), index=idx).astype(int)
    f['is_us_afternoon'] = pd.Series(np.isin(idx.hour, range(14, 18)), index=idx).astype(int)
    f['us_prime_setup']  = ((f['price_div_cl'] < 0) & (adx > 14) & f['is_us_session'].astype(bool)).astype(int)

    # Key levels e kill zones
    f = f.join(key_levels_vec(cl)).join(kill_zone_vec(cl))

    return f


# ── Simulação de trade ────────────────────────────────────────────────────────

def simulate(cl_raw, signal_pos, direction):
    entry = float(cl_raw['close'].iloc[signal_pos])
    if direction == 'LONG':
        tgt = entry + TARGET_PTS
        stp = entry - STOP_PTS
    else:
        tgt = entry - TARGET_PTS
        stp = entry + STOP_PTS

    n = len(cl_raw)
    for i in range(signal_pos + 1, min(signal_pos + 1 + MAX_BARS, n)):
        h = float(cl_raw['high'].iloc[i])
        l = float(cl_raw['low'].iloc[i])
        if direction == 'LONG':
            if h >= tgt: return 'WIN',  entry, tgt, stp
            if l <= stp: return 'LOSS', entry, tgt, stp
        else:
            if l <= tgt: return 'WIN',  entry, tgt, stp
            if h >= stp: return 'LOSS', entry, tgt, stp

    return 'TIMEOUT', entry, tgt, stp


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Valida sinais ML do CL ao vivo')
    parser.add_argument('--days', type=int, default=2,
                        help='Dias passados para avaliar (default: 2)')
    parser.add_argument('--min-prob', type=float, default=0.55,
                        help='Probabilidade minima para aceitar sinal (default: 0.55)')
    args = parser.parse_args()

    print(f'[validate_live] Buscando 60d de dados 1h...')
    cl_raw  = fetch('CL=F')
    mnq_raw = fetch('MNQ=F')
    btc_raw = fetch('BTC-USD')

    if any(d is None for d in [cl_raw, mnq_raw, btc_raw]):
        print('Erro: falha ao buscar dados do Yahoo Finance.')
        return

    print(f'[validate_live] Calculando indicadores e features...')
    cl  = indicators(cl_raw)
    mnq = indicators(mnq_raw)
    btc = indicators(btc_raw)

    feats = build_features(cl, mnq, btc)

    if not MODEL_PATH.exists():
        print(f'Erro: modelo nao encontrado em {MODEL_PATH}')
        return

    model_data = pickle.load(open(MODEL_PATH, 'rb'))
    model      = model_data['model']
    feat_cols  = model_data['features']

    # bars com features completas
    valid = feats[feat_cols].dropna()
    if valid.empty:
        print('Sem bars com features completas.')
        return

    print(f'[validate_live] Rodando modelo em {len(valid)} bars...')
    preds  = model.predict(valid)
    probas = model.predict_proba(valid)

    # janela de avaliacao: [agora - days, agora - MAX_BARS horas]
    now          = datetime.now()
    cutoff_start = now - timedelta(days=args.days)
    cutoff_end   = now - timedelta(hours=MAX_BARS)

    eval_mask = (valid.index >= cutoff_start) & (valid.index <= cutoff_end)
    eval_ts   = valid.index[eval_mask]

    signal_map = {0: 'SHORT', 1: 'NEUTRO', 2: 'LONG'}
    rows = []

    for ts in eval_ts:
        pos_pred = valid.index.get_loc(ts)
        if isinstance(pos_pred, slice):
            pos_pred = pos_pred.start

        direction = signal_map.get(int(preds[pos_pred]))
        if direction == 'NEUTRO':
            continue

        prob_s = float(probas[pos_pred][0])
        prob_l = float(probas[pos_pred][2])
        prob   = prob_l if direction == 'LONG' else prob_s

        if prob < args.min_prob:
            continue

        # posicao no raw para simular com high/low reais
        pos_raw = cl_raw.index.searchsorted(ts)
        if pos_raw >= len(cl_raw) - 1:
            continue

        result, entry, tgt, stp = simulate(cl_raw, pos_raw, direction)

        rows.append({
            'ts':        ts.strftime('%Y-%m-%d %H:%M'),
            'direction': direction,
            'entry':     round(entry, 3),
            'target':    round(tgt, 3),
            'stop':      round(stp, 3),
            'result':    result,
            'prob':      round(prob, 4),
            'hour':      ts.hour,
            'dow':       ts.dayofweek,
        })

    if not rows:
        print(f'Nenhum sinal LONG/SHORT nos ultimos {args.days} dia(s) avaliados.')
        return

    new_df = pd.DataFrame(rows)

    # acumula no CSV sem duplicar
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        combined = pd.concat([existing, new_df]).drop_duplicates(subset=['ts', 'direction'])
    else:
        combined = new_df

    combined.to_csv(OUTPUT_CSV, index=False)

    # ── Resumo ──
    total    = len(new_df)
    wins     = (new_df['result'] == 'WIN').sum()
    losses   = (new_df['result'] == 'LOSS').sum()
    timeouts = (new_df['result'] == 'TIMEOUT').sum()
    wr       = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
    pnl      = wins * TARGET_PTS - losses * STOP_PTS

    print(f'\n{"="*50}')
    print(f'  Periodo avaliado: ultimos {args.days} dia(s)')
    print(f'  Filtro prob:      >= {args.min_prob}')
    print(f'  Sinais:   {total}  '
          f'(LONG: {(new_df["direction"]=="LONG").sum()}  '
          f'SHORT: {(new_df["direction"]=="SHORT").sum()})')
    print(f'  WIN: {wins}   LOSS: {losses}   TIMEOUT: {timeouts}')
    print(f'  Win Rate: {wr:.1f}%  (excl. timeout)')
    print(f'  P&L bruto: {pnl:+.2f} pts  '
          f'(stop $1 / target $3 por barril)')
    print(f'  CSV total: {len(combined)} entradas  ->  {OUTPUT_CSV.name}')
    print(f'{"="*50}')

    # detalhe por sinal
    print(f'\n{"TS":<17} {"DIR":<6} {"ENTRY":>7}  {"TGT":>7}  {"STP":>7}  {"RESULT":<8} {"PROB":>6}')
    print('-' * 62)
    for _, r in new_df.sort_values('ts').iterrows():
        res_sym = '+' if r['result'] == 'WIN' else ('-' if r['result'] == 'LOSS' else '~')
        print(f'{r["ts"]:<17} {r["direction"]:<6} {r["entry"]:>7.3f}  '
              f'{r["target"]:>7.3f}  {r["stop"]:>7.3f}  '
              f'{res_sym} {r["result"]:<7} {r["prob"]:>6.4f}')

    # acumulado historico (se houver mais dias no CSV)
    if len(combined) > total:
        all_w = (combined['result'] == 'WIN').sum()
        all_l = (combined['result'] == 'LOSS').sum()
        all_wr = all_w / (all_w + all_l) * 100 if (all_w + all_l) > 0 else 0
        all_pnl = all_w * TARGET_PTS - all_l * STOP_PTS
        print(f'\nHistorico acumulado ({len(combined)} sinais): '
              f'WR {all_wr:.1f}%  P&L {all_pnl:+.2f} pts')


if __name__ == '__main__':
    main()

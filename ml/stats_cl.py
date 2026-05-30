"""
stats_cl.py — Estatisticas comportamentais do CL (Crude Oil).
Analisa padroes de price action para day/swing traders.
Saida: JSON para stats_cl.html

Uso: python ml/stats_cl.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np

TICKER = 'CL=F'


def download(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[['open','high','low','close','volume']].dropna()


def pct(series):
    v = float(series.mean() * 100)
    return round(v, 1)

def safe_pct(n, d):
    if d == 0: return 0.0
    return round(n / d * 100, 1)

def r(v, dec=3):
    return round(float(v), dec)


def compute():
    df_1h = download(TICKER, '730d', '1h')
    df_d  = download(TICKER, '5y',   '1d')
    if df_1h is None or df_d is None:
        return {'error': 'falha ao baixar dados CL'}

    # Filtro rollover — CL rola mensalmente, jumps maiores que NQ
    df_1h = df_1h[df_1h['close'].pct_change().abs() < 0.04].copy()
    df_d  = df_d[df_d['close'].pct_change().abs() < 0.08].copy()

    # 1H → 4H
    df_4h = df_1h.resample('4h').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'),     close=('close','last'),
        volume=('volume','sum')
    ).dropna()
    df_4h = df_4h[df_4h['volume'] > 0].copy()

    out = {}

    out['periodo'] = {
        'inicio_1h':      str(df_1h.index[0].date()),
        'fim_1h':         str(df_1h.index[-1].date()),
        'barras_1h':      len(df_1h),
        'barras_4h':      len(df_4h),
        'inicio_diario':  str(df_d.index[0].date()),
        'fim_diario':     str(df_d.index[-1].date()),
        'barras_diarias': len(df_d),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 1. BARRA DE 4H
    # ══════════════════════════════════════════════════════════════════════════
    f = df_4h.copy()
    f['bull']       = f['close'] > f['open']
    f['body_pct']   = (f['close'] - f['open']) / f['open'] * 100
    f['range_pct']  = (f['high']  - f['low'])  / f['open'] * 100
    f['close_pos']  = (f['close'] - f['low'])  / (f['high'] - f['low'])
    f['gap_pct']    = (f['open']  - f['close'].shift(1)) / f['close'].shift(1) * 100
    f['next_bull']  = f['bull'].shift(-1)
    f['prev_bull']  = f['bull'].shift(1).fillna(False).astype(bool)
    f['prev2_bull'] = f['bull'].shift(2).fillna(False).astype(bool)
    f['wick_up']    = (f['high'] - f[['open','close']].max(axis=1)) / f['open'] * 100
    f['wick_dn']    = (f[['open','close']].min(axis=1) - f['low'])  / f['open'] * 100

    green4 = f[f['bull'] == True]
    red4   = f[f['bull'] == False]
    gap_up = f[f['gap_pct'] >  0.1]
    gap_dn = f[f['gap_pct'] < -0.1]

    out['barra_4h'] = {
        'total':                    len(f),
        'pct_bullish':              pct(f['bull']),
        'avg_range_pct':            r(f['range_pct'].mean()),
        'avg_body_pct':             r(f['body_pct'].abs().mean()),
        'avg_wick_up_pct':          r(f['wick_up'].mean()),
        'avg_wick_dn_pct':          r(f['wick_dn'].mean()),
        'avg_close_pos_green':      r(green4['close_pos'].mean()),
        'avg_close_pos_red':        r(red4['close_pos'].mean()),
        'pct_verde_apos_verde':     pct(green4['next_bull'].dropna()),
        'pct_verde_apos_vermelho':  pct(red4['next_bull'].dropna()),
        'pct_verde_apos_2v':        pct(f[(~f['prev_bull']) & (~f['prev2_bull'])]['bull'].dropna()),
        'pct_vermelho_apos_2g':     round(100 - pct(f[f['prev_bull'] & f['prev2_bull']]['bull'].dropna()), 1),
        'avg_contra_em_bull_pct':   r(green4['wick_dn'].mean()),
        'avg_contra_em_bear_pct':   r(red4['wick_up'].mean()),
        'pct_bull_fecha_acima_0.2': pct(green4['body_pct'] >= 0.2),
        'pct_bull_fecha_acima_0.5': pct(green4['body_pct'] >= 0.5),
        'pct_bear_fecha_abaixo_0.2': pct(red4['body_pct'] <= -0.2),
        'pct_bear_fecha_abaixo_0.5': pct(red4['body_pct'] <= -0.5),
        'dist_body': {
            'abaixo_0.5':        safe_pct((f['body_pct'] < -0.5).sum(), len(f)),
            'entre_0.5_0.2_neg': safe_pct(((f['body_pct'] >= -0.5) & (f['body_pct'] < -0.2)).sum(), len(f)),
            'neutro':            safe_pct((f['body_pct'].abs() < 0.2).sum(), len(f)),
            'entre_0.2_0.5':     safe_pct(((f['body_pct'] >= 0.2) & (f['body_pct'] < 0.5)).sum(), len(f)),
            'acima_0.5':         safe_pct((f['body_pct'] >= 0.5).sum(), len(f)),
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 2. LDH / LDL
    # ══════════════════════════════════════════════════════════════════════════
    dd = df_d.copy()
    dd['dow']       = dd.index.dayofweek
    dd['ldh']       = dd['high'].shift(1)
    dd['ldl']       = dd['low'].shift(1)
    dd['day_ret']   = (dd['close'] / dd['open'] - 1) * 100
    dd['range_pct'] = (dd['high'] - dd['low']) / dd['open'] * 100
    dd['bull']      = dd['close'] > dd['open']
    dd['close_pos'] = (dd['close'] - dd['low']) / (dd['high'] - dd['low'])

    dd['toca_ldh']    = dd['high'] >= dd['ldh']
    dd['rompe_ldh']   = dd['close'] > dd['ldh']
    dd['rejeita_ldh'] = dd['toca_ldh'] & ~dd['rompe_ldh'].fillna(False)
    dd['toca_ldl']    = dd['low'] <= dd['ldl']
    dd['rompe_ldl']   = dd['close'] < dd['ldl']
    dd['rejeita_ldl'] = dd['toca_ldl'] & ~dd['rompe_ldl'].fillna(False)

    n_ldh = int(dd['toca_ldh'].sum())
    n_ldl = int(dd['toca_ldl'].sum())

    out['ldh_ldl'] = {
        'total_dias':           len(dd),
        'pct_toca_ldh':         pct(dd['toca_ldh']),
        'pct_ldh_rejeitado':    safe_pct(int(dd['rejeita_ldh'].sum()), n_ldh),
        'pct_ldh_rompido':      safe_pct(int(dd['rompe_ldh'].sum()), n_ldh),
        'n_toca_ldh':           n_ldh,
        'pct_toca_ldl':         pct(dd['toca_ldl']),
        'pct_ldl_rejeitado':    safe_pct(int(dd['rejeita_ldl'].sum()), n_ldl),
        'pct_ldl_rompido':      safe_pct(int(dd['rompe_ldl'].sum()), n_ldl),
        'n_toca_ldl':           n_ldl,
        'adr_medio_pct':        r(dd['range_pct'].mean()),
        'adr_mediana_pct':      r(dd['range_pct'].median()),
        'pct_acima_1adr':       pct(dd['range_pct'] > dd['range_pct'].rolling(20, min_periods=5).mean()),
        'avg_close_pos':        r(dd['close_pos'].mean()),
        'pct_fecha_top_tercil': pct(dd['close_pos'] > 0.667),
        'pct_fecha_mid_tercil': pct((dd['close_pos'] >= 0.333) & (dd['close_pos'] <= 0.667)),
        'pct_fecha_bot_tercil': pct(dd['close_pos'] < 0.333),
        'avg_close_pos_verde':  r(dd.loc[dd['bull'], 'close_pos'].mean()),
        'avg_close_pos_verm':   r(dd.loc[~dd['bull'], 'close_pos'].mean()),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. MONDAY LOW / HIGH
    # ══════════════════════════════════════════════════════════════════════════
    dd['week'] = dd.index.to_period('W')

    monday_levels = {}
    monday_dir    = {}
    for week, grp in dd.groupby('week'):
        mon = grp[grp['dow'] == 0]
        if not mon.empty:
            monday_levels[week] = {'low': float(mon['low'].iloc[0]), 'high': float(mon['high'].iloc[0])}
            monday_dir[week]    = bool(mon['close'].iloc[0] > mon['open'].iloc[0])

    dd['monday_low']  = dd['week'].map(lambda w: monday_levels.get(w, {}).get('low',  np.nan))
    dd['monday_high'] = dd['week'].map(lambda w: monday_levels.get(w, {}).get('high', np.nan))
    dd['mon_bull']    = dd['week'].map(lambda w: monday_dir.get(w, None))

    week_bull = {}
    for week, grp in dd.groupby('week'):
        if len(grp) >= 2:
            week_bull[week] = bool(grp['close'].iloc[-1] > grp['open'].iloc[0])
    dd['week_bull'] = dd['week'].map(week_bull)

    tf = dd[dd['dow'] > 0].copy()
    tf = tf.dropna(subset=['monday_low', 'monday_high'])

    tf['toca_ml']  = tf['low'] <= tf['monday_low']
    tf['rompe_ml'] = tf['close'] < tf['monday_low']
    tf['toca_mh']  = tf['high'] >= tf['monday_high']
    tf['rompe_mh'] = tf['close'] > tf['monday_high']

    n_ml = int(tf['toca_ml'].sum())
    n_mh = int(tf['toca_mh'].sum())

    dd_w   = dd.dropna(subset=['mon_bull', 'week_bull'])
    mon_up = dd_w[dd_w['mon_bull'] == True]
    mon_dn = dd_w[dd_w['mon_bull'] == False]

    out['monday'] = {
        'total_semanas':                     len(monday_levels),
        'pct_dias_toca_mon_low':             safe_pct(n_ml, len(tf)),
        'pct_mon_low_respeitado':            safe_pct(n_ml - int(tf['rompe_ml'].sum()), n_ml),
        'pct_mon_low_rompido':               safe_pct(int(tf['rompe_ml'].sum()), n_ml),
        'n_toca_mon_low':                    n_ml,
        'pct_dias_toca_mon_high':            safe_pct(n_mh, len(tf)),
        'pct_mon_high_respeitado':           safe_pct(n_mh - int(tf['rompe_mh'].sum()), n_mh),
        'pct_mon_high_rompido':              safe_pct(int(tf['rompe_mh'].sum()), n_mh),
        'n_toca_mon_high':                   n_mh,
        'pct_semana_verde_se_seg_verde':     pct(mon_up['week_bull']) if len(mon_up) > 5 else None,
        'pct_semana_verm_se_seg_verm':       round(100 - pct(mon_dn['week_bull']), 1) if len(mon_dn) > 5 else None,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 4. PRIMEIRO CANDLE 4H
    # ══════════════════════════════════════════════════════════════════════════
    f['date'] = f.index.normalize()
    first4h = f.groupby('date').first().copy()
    first4h['fc_bull'] = first4h['close'] > first4h['open']

    dd_idx = dd.copy()
    dd_idx.index = pd.to_datetime(dd_idx.index).normalize()
    merged = first4h.join(dd_idx[['bull']].rename(columns={'bull': 'day_bull'}), how='inner')

    fc_g = merged[merged['fc_bull'] == True]
    fc_r = merged[merged['fc_bull'] == False]

    out['primeiro_candle'] = {
        'total':                            len(merged),
        'pct_primeiro_4h_verde':            pct(merged['fc_bull']),
        'pct_dia_verde_se_1c_verde':        pct(fc_g['day_bull']),
        'pct_dia_verde_se_1c_vermelho':     pct(fc_r['day_bull']),
        'pct_dia_vermelho_se_1c_vermelho':  round(100 - pct(fc_r['day_bull']), 1),
        'pct_dia_vermelho_se_1c_verde':     round(100 - pct(fc_g['day_bull']), 1),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 5. ORB — Primeira hora NY RTH (13:00 UTC ≈ 9:00 ET)
    # ══════════════════════════════════════════════════════════════════════════
    h1 = df_1h.copy()
    h1['hour'] = h1.index.hour
    h1['date'] = h1.index.normalize()

    orb = h1[h1['hour'] == 13].copy()
    orb['orb_bull'] = (orb['close'] > orb['open']).astype(int)
    orb_d = orb.set_index('date')[['orb_bull', 'high', 'low']].rename(
        columns={'high': 'orb_high', 'low': 'orb_low'})

    dd_idx2 = dd.copy()
    dd_idx2.index = pd.to_datetime(dd_idx2.index).normalize()
    orb_m = orb_d.join(dd_idx2[['bull']].rename(columns={'bull': 'day_bull'}), how='inner')

    orb_g = orb_m[orb_m['orb_bull'] == 1]
    orb_r = orb_m[orb_m['orb_bull'] == 0]

    out['orb'] = {
        'total':                           len(orb_m),
        'pct_1h_rth_verde':                pct(orb_m['orb_bull']),
        'pct_dia_verde_se_1h_verde':       pct(orb_g['day_bull']),
        'pct_dia_verde_se_1h_vermelho':    pct(orb_r['day_bull']),
        'pct_dia_vermelho_se_1h_vermelho': round(100 - pct(orb_r['day_bull']), 1),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 6. SESSOES (Asia / London / NY)
    # ══════════════════════════════════════════════════════════════════════════
    def sess(hour_s, hour_e, nome):
        sub = h1[(h1['hour'] >= hour_s) & (h1['hour'] < hour_e)].copy()
        if sub.empty: return {}
        s = sub.groupby('date').agg(
            open=('open','first'), close=('close','last'),
            high=('high','max'),   low=('low','min')
        ).dropna()
        s = s[(s['high'] - s['low']) > 0]
        s['bull']      = s['close'] > s['open']
        s['range_pct'] = (s['high'] - s['low']) / s['open'] * 100
        s['ret_pct']   = (s['close'] / s['open'] - 1) * 100
        s['close_pos'] = (s['close'] - s['low']) / (s['high'] - s['low'])
        return {
            'nome':          nome,
            'pct_bullish':   pct(s['bull']),
            'avg_range_pct': r(s['range_pct'].mean()),
            'avg_ret_pct':   r(s['ret_pct'].mean(), 4),
            'avg_close_pos': r(s['close_pos'].mean()),
            'n':             len(s),
        }

    ldn = h1[(h1['hour'] >= 8) & (h1['hour'] < 13)].groupby('date').agg(
        ldn_open=('open','first'), ldn_close=('close','last')).dropna()
    ldn['ldn_bull'] = ldn['ldn_close'] > ldn['ldn_open']
    ny = h1[(h1['hour'] >= 13) & (h1['hour'] < 20)].groupby('date').agg(
        ny_open=('open','first'), ny_close=('close','last')).dropna()
    ny['ny_bull'] = ny['ny_close'] > ny['ny_open']
    ldn_ny = ldn.join(ny, how='inner')

    out['sessoes'] = {
        'stats': [
            sess(0,  8,  'Asia (00-08 UTC)'),
            sess(8,  13, 'London (08-13 UTC)'),
            sess(13, 20, 'NY RTH (13-20 UTC)'),
            sess(20, 24, 'After-Hours (20-24 UTC)'),
        ],
        'london_ny_continua': pct(ldn_ny[ldn_ny['ldn_bull']]['ny_bull']) if len(ldn_ny) > 10 else None,
        'london_ny_reverte':  round(100 - pct(ldn_ny[ldn_ny['ldn_bull']]['ny_bull']), 1) if len(ldn_ny) > 10 else None,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 7. POR DIA DA SEMANA — Quarta = EIA Inventory (10:30 ET / 14:30 UTC)
    # ══════════════════════════════════════════════════════════════════════════
    dow_names = ['Segunda', 'Terca', 'Quarta (EIA)', 'Quinta', 'Sexta']
    dow_stats = []
    for i, nome in enumerate(dow_names):
        sub = dd[dd['dow'] == i]
        if len(sub) < 5: continue
        dow_stats.append({
            'dia':           nome,
            'n':             len(sub),
            'pct_bullish':   pct(sub['bull']),
            'avg_ret_pct':   r(sub['day_ret'].mean(), 3),
            'avg_range_pct': r(sub['range_pct'].mean()),
            'pct_fecha_top': pct(sub['close_pos'] > 0.667),
            'pct_fecha_bot': pct(sub['close_pos'] < 0.333),
            'eia':           i == 2,
        })
    out['por_dia_semana'] = dow_stats

    # ══════════════════════════════════════════════════════════════════════════
    # 8. POR HORA DO DIA (UTC)
    # ══════════════════════════════════════════════════════════════════════════
    h1['ret_pct']    = (h1['close'] / h1['open'] - 1) * 100
    h1['range_pct2'] = (h1['high'] - h1['low']) / h1['open'] * 100
    h1['bull1h']     = (h1['close'] > h1['open']).astype(int)

    hour_stats = []
    for hr in sorted(h1['hour'].unique()):
        sub = h1[h1['hour'] == hr]
        if len(sub) < 20: continue
        hour_stats.append({
            'hour':          int(hr),
            'n':             len(sub),
            'pct_bullish':   pct(sub['bull1h']),
            'avg_ret_pct':   r(sub['ret_pct'].mean(), 4),
            'avg_range_pct': r(sub['range_pct2'].mean(), 4),
        })
    out['por_hora'] = hour_stats

    # ══════════════════════════════════════════════════════════════════════════
    # 9. PREVIOUS WEEK HIGH / LOW
    # ══════════════════════════════════════════════════════════════════════════
    week_hl = dd.groupby('week').agg(wk_high=('high','max'), wk_low=('low','min')).reset_index()
    week_hl['pwh'] = week_hl['wk_high'].shift(1)
    week_hl['pwl'] = week_hl['wk_low'].shift(1)
    week_hl = week_hl.dropna()
    week_hl['toca_pwh'] = week_hl['wk_high'] >= week_hl['pwh']
    week_hl['toca_pwl'] = week_hl['wk_low']  <= week_hl['pwl']

    out['prev_week'] = {
        'total_semanas': len(week_hl),
        'pct_toca_pwh':  pct(week_hl['toca_pwh']),
        'pct_toca_pwl':  pct(week_hl['toca_pwl']),
        'pct_ambos':     pct(week_hl['toca_pwh'] & week_hl['toca_pwl']),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 10. ABERTURA 4H — REVERSAO VS CONTINUACAO
    # ══════════════════════════════════════════════════════════════════════════
    f['prev_close']  = f['close'].shift(1)
    f['abre_acima']  = f['open'] > f['prev_close']
    f['abre_abaixo'] = f['open'] < f['prev_close']

    above = f[f['abre_acima']].copy()
    below = f[f['abre_abaixo']].copy()

    def bucket_close(sub):
        n = len(sub)
        if n == 0: return {}
        return {
            'fecha_0.5_abaixo':             safe_pct((sub['body_pct'] < -0.5).sum(), n),
            'fecha_0.2_0.5_abx':            safe_pct(((sub['body_pct'] >= -0.5) & (sub['body_pct'] < -0.2)).sum(), n),
            'fecha_0.2_acima_e_abaixo':     safe_pct((sub['body_pct'].abs() < 0.2).sum(), n),
            'fecha_0.2_0.5_cma':            safe_pct(((sub['body_pct'] >= 0.2) & (sub['body_pct'] < 0.5)).sum(), n),
            'fecha_0.5_acima':              safe_pct((sub['body_pct'] >= 0.5).sum(), n),
        }

    out['abertura_4h'] = {
        'n_abre_acima':               len(above),
        'n_abre_abaixo':              len(below),
        'pct_abre_acima':             safe_pct(len(above), len(f)),
        'pct_abre_abaixo':            safe_pct(len(below), len(f)),
        'pct_continua_se_abre_acima': pct(above['bull']) if len(above) > 5 else None,
        'pct_reverte_se_abre_acima':  round(100 - pct(above['bull']), 1) if len(above) > 5 else None,
        'pct_continua_se_abre_abaixo':round(100 - pct(below['bull']), 1) if len(below) > 5 else None,
        'pct_reverte_se_abre_abaixo': pct(below['bull']) if len(below) > 5 else None,
        'dist_gap_up':  bucket_close(above),
        'dist_gap_dn':  bucket_close(below),
    }

    out['ts'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return out


if __name__ == '__main__':
    data = compute()
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(data, default=str, ensure_ascii=False))

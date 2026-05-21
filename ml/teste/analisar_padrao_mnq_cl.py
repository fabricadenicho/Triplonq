"""
Analisa o padrao MNQ sobe + CL cai.
O usuario observou: divergencia comecou as 06h BRT, mercado abriu 19h BRT.
Vamos procurar no historico quando isso aconteceu e qual foi a sequencia.
"""
import sqlite3, json
import pandas as pd
import numpy as np
import ta
from pathlib import Path

DB = Path(__file__).parent.parent / 'data.db'

def load_sym(conn, sym):
    df = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND LENGTH(ts)=19 ORDER BY ts",
        conn, params=(sym,), parse_dates=['ts'], index_col='ts')
    return df

def compute(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
    adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=17)
    df['adx'] = adx_i.adx()
    df['pdi'] = adx_i.adx_pos()
    df['mdi'] = adx_i.adx_neg()
    df['ret1'] = df['close'].pct_change(1)
    df['ret4'] = df['close'].pct_change(4)
    df['ret8'] = df['close'].pct_change(8)
    df['vol'] = df['ret1'].rolling(20).std()
    df['sma50'] = df['close'].rolling(50).mean()
    df['dist_sma50'] = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['above_sma50'] = (df['close'] > df['sma50']).astype(int)
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['above_ema20'] = (df['close'] > df['ema20']).astype(int)
    return df

def main():
    conn = sqlite3.connect(DB)
    print("Carregando dados...")
    mnq = compute(load_sym(conn, 'mnq'))
    btc = compute(load_sym(conn, 'btc'))
    cl  = compute(load_sym(conn, 'cl'))
    conn.close()

    idx = mnq.index
    f = pd.DataFrame(index=idx)

    f['hour'] = idx.hour
    f['dow']  = idx.dayofweek

    # RSI
    f['rsi_mnq'] = mnq['rsi']
    f['rsi_cl']  = cl['rsi'].reindex(idx, method='ffill')
    f['rsi_btc'] = btc['rsi'].reindex(idx, method='ffill')

    # Retornos
    f['ret1_mnq'] = mnq['ret1']
    f['ret1_cl']  = cl['ret1'].reindex(idx, method='ffill')
    f['ret1_btc'] = btc['ret1'].reindex(idx, method='ffill')

    # ADX
    f['adx_mnq'] = mnq['adx']
    f['adx_cl']  = cl['adx'].reindex(idx, method='ffill')

    # Divergencias
    f['div_cl']  = f['rsi_mnq'] - f['rsi_cl']
    f['price_div_cl'] = f['ret1_mnq'] * f['ret1_cl']
    f['cl_down_mnq_up'] = ((f['ret1_cl'] < 0) & (f['ret1_mnq'] > 0)).astype(int)

    # SMA50
    f['above_sma50_mnq'] = mnq['above_sma50']
    f['above_sma50_cl']  = cl['above_sma50'].reindex(idx, method='ffill')
    f['above_sma50_btc'] = btc['above_sma50'].reindex(idx, method='ffill')
    f['sma50_alignment'] = f['above_sma50_mnq'] + f['above_sma50_btc'] + f['above_sma50_cl']

    # EMA20
    f['above_ema20_mnq'] = mnq['above_ema20']
    f['above_ema20_cl']  = cl['above_ema20'].reindex(idx, method='ffill')
    f['above_ema20_btc'] = btc['above_ema20'].reindex(idx, method='ffill')

    # Precos
    f['close_mnq'] = mnq['close']
    f['close_cl']  = cl['close'].reindex(idx, method='ffill')
    f['close_btc'] = btc['close'].reindex(idx, method='ffill')

    f['mnq_pct_4h'] = mnq['close'].pct_change(4) * 100
    f['cl_pct_4h']  = cl['close'].reindex(idx, method='ffill').pct_change(4) * 100

    f = f.dropna()

    print(f"\nTotal de candles: {len(f)}")
    print(f"Periodo: {f.index[0]} ate {f.index[-1]}")

    # ============================================================
    # 1. QUANTAS VEZES ACONTECEU O PADRAO "MNQ SOBE + CL CAI"?
    # ============================================================
    total_cl_down_mnq_up = f['cl_down_mnq_up'].sum()
    print(f"\n--- PADRAO 'MNQ SOBE + CL CAI' ---")
    print(f"Total de ocorrencias (cada hora): {total_cl_down_mnq_up}")
    print(f"Frequencia: {total_cl_down_mnq_up/len(f)*100:.1f}% do tempo")

    # ============================================================
    # 2. ANALISE POR HORARIO (BRT = UTC-3)
    # ============================================================
    # Descobrir timezone: comparar horario dos dados com NY
    # Vamos assumir que os dados estao em US/Eastern (ET)
    # 19h BRT = 18h ET (EDT) ou 17h ET (EST)
    # mas vamos usar as horas como estao no banco

    print(f"\n--- OCORRENCIAS POR HORA ---")
    for h in range(24):
        sub = f[f['hour'] == h]
        if len(sub) < 10: continue
        n = sub['cl_down_mnq_up'].sum()
        pct = n/len(sub)*100
        print(f"  Hora {h:02d}: {n:>4} vezes ({pct:.1f}%) em {len(sub)} candles")

    # ============================================================
    # 3. BUSCAR DIAS COM O PADRAO DESDE A ABERTURA
    # ============================================================
    # Estratégia: encontrar dias onde:
    # - Entre 19h e 06h (proximo dia) houve MNQ subindo + CL caindo consistentemente
    # - A divergencia comecou cedo (06h) e continuou

    # Agrupar por "sessao": de 19h ate 18h do dia seguinte (futures session)
    # A sessao de futuros comeca as 18h ET (19h BRT) e vai ate 17h ET do dia seguinte

    # Vamos criar um identificador de sessao baseado na data as 19h
    f['session_date'] = f.index.date.astype(str)
    mask = f['hour'] < 19
    f.loc[mask, 'session_date'] = (f.index[mask] - pd.Timedelta(days=1)).date.astype(str)

    print(f"\n\n========================================")
    print(f"BUSCA DE SESSOES COM PADRAO FORTE")
    print(f"========================================")

    results = []
    for sess, grp in f.groupby('session_date'):
        grp = grp.sort_index()
        if len(grp) < 6: continue

        # Horarios chave
        hours_in_sess = set(grp['hour'])

        # Verificar se tem o padrao na janela 19h-23h (abertura)
        evening = grp[grp['hour'].between(19, 23)]
        # Verificar se tem o padrao na janela 00h-06h (asia / pre-US)
        early = grp[grp['hour'].between(0, 10)]

        n_evening = evening['cl_down_mnq_up'].sum()
        n_early = early['cl_down_mnq_up'].sum()

        if n_evening >= 1 or n_early >= 1:
            # Calcular consistencia
            hrs_div = grp[grp['price_div_cl'] < 0]
            hrs_tot = len(grp)
            pct_div = len(hrs_div) / hrs_tot * 100

            # Preco change na sessao
            first_close_mnq = grp['close_mnq'].iloc[0]
            last_close_mnq = grp['close_mnq'].iloc[-1]
            first_close_cl = grp['close_cl'].iloc[0]
            last_close_cl = grp['close_cl'].iloc[-1]
            mnq_change = (last_close_mnq / first_close_mnq - 1) * 100
            cl_change = (last_close_cl / first_close_cl - 1) * 100

            # RSI divergence na janela 06h-10h
            early_hrs = grp[grp['hour'].between(6, 10)]
            if len(early_hrs) > 0:
                div_cl_early = early_hrs['div_cl'].iloc[0] if len(early_hrs) > 0 else 0
            else:
                div_cl_early = 0

            results.append({
                'session': sess,
                'total_hrs': hrs_tot,
                'n_divergencia': len(hrs_div),
                'pct_divergencia': round(pct_div, 1),
                'n_cl_down_mnq_up_evening': int(n_evening),
                'n_cl_down_mnq_up_early': int(n_early),
                'n_cl_down_mnq_up_total': int(grp['cl_down_mnq_up'].sum()),
                'mnq_change_pct': round(mnq_change, 2),
                'cl_change_pct': round(cl_change, 2),
                'div_cl_early_06_10': round(div_cl_early, 2),
                'hrs': sorted(list(hours_in_sess)),
            })

    df_r = pd.DataFrame(results)
    if len(df_r) == 0:
        print("Nenhuma sessao encontrada.")
        return

    # Sessoes com mais divergencia
    top_div = df_r.sort_values('pct_divergencia', ascending=False).head(20)
    print(f"\n--- TOP 20 SESSOES COM MAIS DIVERGENCIA (MNQ x CL) ---")
    print(f"{'Sessao':<12} {'%Div':>5} {'N':>3} {'MNQ%':>7} {'CL%':>7} {'Evening':>7} {'Early':>6} {'Div06-10':>8}")
    print(f"{'------':<12} {'----':>5} {'---':>3} {'-----':>7} {'-----':>7} {'-------':>7} {'-----':>6} {'-------':>8}")
    for _, r in top_div.iterrows():
        print(f"{r['session']:<12} {r['pct_divergencia']:>5} {r['total_hrs']:>3} "
              f"{r['mnq_change_pct']:>+7.2f} {r['cl_change_pct']:>+7.2f} "
              f"{r['n_cl_down_mnq_up_evening']:>7} {r['n_cl_down_mnq_up_early']:>6} "
              f"{r['div_cl_early_06_10']:>+8.2f}")

    # ============================================================
    # 4. PADRAO ESPECIFICO: divergencia comeca 06h, continua ate US session
    # ============================================================
    print(f"\n\n========================================")
    print(f"SESSOES COM PADRAO SIMILAR AO OBSERVADO")
    print(f"(divergencia 06h + abertura 19h + MNQ sobe / CL cai)")
    print(f"========================================")

    # Filtrar: teve divergencia entre 06-10h E teve na abertura 19-23h
    similar = df_r[(df_r['n_cl_down_mnq_up_early'] >= 1) &
                   (df_r['n_cl_down_mnq_up_evening'] >= 1) &
                   (df_r['mnq_change_pct'] > 0) &
                   (df_r['cl_change_pct'] < 0)]
    similar = similar.sort_values('pct_divergencia', ascending=False)

    if len(similar) == 0:
        print("Nenhuma sessao exatamente igual encontrada.")
        print("Mas mostrando sessoes com divergencia forte em qualquer horario:")
        similar = df_r[df_r['pct_divergencia'] > 50].sort_values('pct_divergencia', ascending=False)

    print(f"{'Sessao':<12} {'%Div':>5} {'N':>3} {'MNQ%':>7} {'CL%':>7} {'Eve':>4} {'Early':>5}")
    print(f"{'------':<12} {'----':>5} {'---':>3} {'-----':>7} {'-----':>7} {'----':>4} {'----':>5}")
    for _, r in similar.iterrows():
        print(f"{r['session']:<12} {r['pct_divergencia']:>5} {r['total_hrs']:>3} "
              f"{r['mnq_change_pct']:>+7.2f} {r['cl_change_pct']:>+7.2f} "
              f"{r['n_cl_down_mnq_up_evening']:>4} {r['n_cl_down_mnq_up_early']:>5}")

    # ============================================================
    # 5. DETALHAR UMA SESSAO ESPECIFICA (a mais recente)
    # ============================================================
    if len(similar) > 0:
        top_sess = similar.iloc[0]['session']
        print(f"\n\n--- DETALHAMENTO DA SESSAO {top_sess} ---")
        grp = f[f['session_date'] == top_sess].sort_index()
        print(f"{'Hora':>5} {'MNQ':>10} {'CL':>10} {'BTC':>10} {'MNQ%1h':>7} {'CL%1h':>7} "
              f"{'div':>6} {'price_div':>9} {'padrao':>6} {'ADX':>5} {'RSI_m':>6} {'RSI_c':>6}")
        for _, r in grp.iterrows():
            pad = "SIM" if r['cl_down_mnq_up'] else ""
            print(f"{r.name.hour:>5} {r['close_mnq']:>10.2f} {r['close_cl']:>10.2f} "
                  f"{r['close_btc']:>10.0f} {r['ret1_mnq']*100:>+6.2f} {r['ret1_cl']*100:>+6.2f} "
                  f"{r['div_cl']:>+6.1f} {r['price_div_cl']:>+9.6f} {pad:>6} "
                  f"{r['adx_mnq']:>5.1f} {r['rsi_mnq']:>6.1f} {r['rsi_cl']:>6.1f}")

    # ============================================================
    # 6. ESTATISTICAS GLOBAIS DO PADRAO
    # ============================================================
    print(f"\n\n========================================")
    print(f"ESTATISTICAS GLOBAIS")
    print(f"========================================")
    print(f"Dados de {f.index[0].date()} a {f.index[-1].date()}")
    print(f"Total de candles: {len(f)}")
    print(f"Total 'cl_down_mnq_up': {total_cl_down_mnq_up} ({total_cl_down_mnq_up/len(f)*100:.1f}%)")

    # Win rate: quando MNQ sobe e CL cai, o que acontece nas proximas 4h?
    # Calcular forward return
    f['mnq_fwd_4h'] = f['close_mnq'].shift(-4) / f['close_mnq'] - 1
    f_valid = f.dropna(subset=['mnq_fwd_4h'])

    padrao = f_valid[f_valid['cl_down_mnq_up'] == 1]
    resto = f_valid[f_valid['cl_down_mnq_up'] == 0]

    if len(padrao) > 10:
        wr_pad = (padrao['mnq_fwd_4h'] > 0.001).mean()
        wr_rest = (resto['mnq_fwd_4h'] > 0.001).mean()
        avg_pad = padrao['mnq_fwd_4h'].mean() * 100
        avg_rest = resto['mnq_fwd_4h'].mean() * 100
        print(f"\nQuando MNQ sobe e CL cai (mesma hora):")
        print(f"  Win rate LONG 4h: {wr_pad:.1%} (vs {wr_rest:.1%} quando nao ocorre)")
        print(f"  Retorno medio 4h: {avg_pad:+.3f}% (vs {avg_rest:+.3f}%)")
        print(f"  Amostras: {len(padrao)}")

        # Por hora
        print(f"\n  Win rate por hora:")
        for h in range(24):
            sub = padrao[padrao['hour'] == h]
            if len(sub) < 5: continue
            wr = (sub['mnq_fwd_4h'] > 0.001).mean()
            avg = sub['mnq_fwd_4h'].mean() * 100
            print(f"    Hora {h:02d}: WR={wr:.1%} avg_ret={avg:+.3f}% N={len(sub)}")

        # Quando combinado com ADX > 14
        strong = padrao[padrao['adx_mnq'] > 14]
        if len(strong) > 5:
            wr_s = (strong['mnq_fwd_4h'] > 0.001).mean()
            avg_s = strong['mnq_fwd_4h'].mean() * 100
            print(f"\n  + ADX > 14: WR={wr_s:.1%} avg_ret={avg_s:+.3f}% N={len(strong)}")

    # Exportar CSV
    out_dir = Path(__file__).parent
    similar.to_csv(out_dir / 'sessoes_divergencia_mnq_cl.csv', index=False)
    print(f"\n\nResultados salvos em: {out_dir}/sessoes_divergencia_mnq_cl.csv")

if __name__ == '__main__':
    main()

"""
Analisa o padrao das 7:15 AM BRT (10:15 UTC).
Verifica: correlacao MNQ vs CL, ES, BTC por hora.
Procura hedge real.
"""
import sqlite3, pandas as pd, numpy as np
from pathlib import Path

base = Path(__file__).parent.parent
c = sqlite3.connect(base / 'data.db')
mnq = pd.read_sql("SELECT ts,close FROM candles WHERE symbol='mnq' AND LENGTH(ts)=19 ORDER BY ts", c, parse_dates=['ts'], index_col='ts')
cl  = pd.read_sql("SELECT ts,close FROM candles WHERE symbol='cl'  AND LENGTH(ts)=19 ORDER BY ts", c, parse_dates=['ts'], index_col='ts')
btc = pd.read_sql("SELECT ts,close FROM candles WHERE symbol='btc' AND LENGTH(ts)=19 ORDER BY ts", c, parse_dates=['ts'], index_col='ts')
c.close()
c2 = sqlite3.connect(base / 'es/data.db')
es  = pd.read_sql("SELECT ts,close FROM candles WHERE symbol='es' AND LENGTH(ts)=19 ORDER BY ts", c2, parse_dates=['ts'], index_col='ts')
c2.close()

idx = mnq.index.intersection(cl.index).intersection(btc.index).intersection(es.index)
mnq=mnq.loc[idx]; cl=cl.loc[idx]; btc=btc.loc[idx]; es=es.loc[idx]

r_mnq = mnq.pct_change()*100
r_cl = cl.pct_change()*100
r_btc = btc.pct_change()*100
r_es = es.pct_change()*100

print("=" * 70)
print("  ANALISE DO PADRAO DAS 7:15 AM BRT (10 UTC)")
print("=" * 70)

# Correlacoes moveis 20h
corr_mnq_cl = r_mnq['close'].rolling(20).corr(r_cl['close'])
corr_mnq_es = r_mnq['close'].rolling(20).corr(r_es['close'])
corr_mnq_btc = r_mnq['close'].rolling(20).corr(r_btc['close'])

horarios = [(9,'06h BRT'),(10,'07h BRT'),(11,'08h BRT'),(14,'11h BRT'),
            (13,'10h BRT'),(12,'09h BRT'),(15,'12h BRT')]

print("\n--- CORRELACAO MEDIA MNQ vs CL por hora ---")
for h, nome in horarios:
    sub = corr_mnq_cl[idx.hour==h].dropna()
    if len(sub)<10: continue
    print(f"  {nome}: media={sub.mean():.3f} mediana={sub.median():.3f} >0.0={sub.gt(0).mean()*100:.0f}% N={len(sub)}")

print("\n--- CORRELACAO MEDIA MNQ vs ES por hora ---")
for h, nome in horarios:
    sub = corr_mnq_es[idx.hour==h].dropna()
    if len(sub)<10: continue
    print(f"  {nome}: media={sub.mean():.3f} >0.0={sub.gt(0).mean()*100:.0f}% N={len(sub)}")

print("\n--- CENARIOS MNQ vs CL por hora (% do tempo) ---")
for h, nome in horarios:
    mask = idx.hour == h
    sm = r_mnq['close'][mask].dropna()
    sc = r_cl['close'][mask].dropna()
    n = len(sm)
    up_up    = ((sm>0)&(sc>0)).sum()/n*100
    dn_dn    = ((sm<0)&(sc<0)).sum()/n*100
    up_dn    = ((sm>0)&(sc<0)).sum()/n*100
    dn_up    = ((sm<0)&(sc>0)).sum()/n*100
    print(f"  {nome}:  UP/UP={up_up:.0f}%  DN/DN={dn_dn:.0f}%  MNQup/CLdn={up_dn:.0f}%  MNQdn/CLup={dn_up:.0f}%")

print("\n--- QUANDO MNQ SOBE + CL CAI, o que ES e BTC fazem? ---")
for h, nome in horarios:
    mask = (idx.hour==h) & (r_mnq['close']>0) & (r_cl['close']<0)
    n = mask.sum()
    if n<5: continue
    es_sobe = r_es['close'][mask].gt(0).mean()*100
    es_cai  = r_es['close'][mask].lt(0).mean()*100
    bt_sobe = r_btc['close'][mask].gt(0).mean()*100
    bt_cai  = r_btc['close'][mask].lt(0).mean()*100
    print(f"  {nome} (N={n}): ES sobre={es_sobe:.0f}%/cai={es_cai:.0f}%  BTC sobre={bt_sobe:.0f}%/cai={bt_cai:.0f}%")

# Forward return quando MNQ sobe + CL cai nas 3h seguintes
print("\n--- RETORNO MNQ NAS 3H SEGUINTES apos divergencia por hora ---")
for h, nome in horarios:
    mask = (idx.hour==h) & (r_mnq['close']>0) & (r_cl['close']<0)
    idx_div = idx[mask]
    rets = []
    for t in idx_div:
        pos = list(idx).index(t)
        if pos+3 < len(idx):
            ret = float(mnq['close'].iloc[pos+3]) / float(mnq['close'].iloc[pos]) - 1
            rets.append(ret)
    if len(rets)<5: continue
    wr = sum(1 for r in rets if r>0.001)/len(rets)*100
    avg = np.mean(rets)*100
    print(f"  {nome}: N={len(rets)} WR={wr:.0f}% ret_medio={avg:+.2f}%")

print("\n--- CORRELACAO CONDICIONAL: quando divergem, MNQ segue ES ou BTC? ---")
# Em vez de correlacao fixa, ver: quando diverge, o MNQ segue qual ativo?
divergindo = (r_mnq['close']>0) & (r_cl['close']<0)
for h, nome in horarios:
    mask = (idx.hour==h) & divergindo
    sub_es = r_es['close'][mask].dropna()
    sub_bt = r_btc['close'][mask].dropna()
    n = len(sub_es)
    if n<5: continue
    # MNQ segue ES (mesmo sinal) ou BTC?
    segue_es = (sub_es>0).mean()*100
    segue_bt = (sub_bt>0).mean()*100
    print(f"  {nome} (N={n}): MNQ segue ES={segue_es:.0f}%  MNQ segue BTC={segue_bt:.0f}%")

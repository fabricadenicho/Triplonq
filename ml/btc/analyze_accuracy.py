"""
Analise pratica do modelo BTC: acertividade por threshold, sessao, regime.
"""
import sqlite3, pickle, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path
from train import DB, build_features, FEATURE_COLS_OPTIMIZED, US_SESSION_HOURS

model_data = pickle.load(open(Path(__file__).parent / 'model.pkl', 'rb'))
model = model_data['model']
feat_cols = model_data['features']

conn = sqlite3.connect(DB)
df = build_features(conn, '1h', 4, 0.001)
conn.close()

df = df[df['hour'].isin(US_SESSION_HOURS)].copy()
print(f'Total amostras (sessao US): {len(df)}')
split = int(len(df) * 0.70)
df_te = df.iloc[split:].copy()

X_te = df_te[feat_cols]
y_te = df_te['label']
proba = model.predict_proba(X_te)
preds = model.predict(X_te)

df_te['prob_short'] = proba[:, 0]
df_te['prob_long']  = proba[:, 2]
df_te['pred']       = preds

print(f'\n{"="*60}')
print(f'  ACERTIVIDADE PRATICA - Sessao US (BTC)')
print(f'{"="*60}')
print(f'  Amostras de teste: {len(df_te)}')
print(f'  Distribuicao: SHORT={y_te.value_counts().get(0,0)/len(y_te):.1%}  NEUTRO={y_te.value_counts().get(1,0)/len(y_te):.1%}  LONG={y_te.value_counts().get(2,0)/len(y_te):.1%}')

print(f'\n{"─"*60}')
print(f'  1. ACERTIVIDADE GERAL')
print(f'{"─"*60}')
for target, nome in [(0, 'SHORT'), (1, 'NEUTRO'), (2, 'LONG')]:
    sub = df_te[df_te['pred'] == target]
    correct = (sub['label'] == target).sum()
    total = len(sub)
    if total > 0:
        print(f'  {nome:<8}: {correct}/{total} = {correct/total:.1%}')

total_correct = (df_te['pred'] == df_te['label']).sum()
print(f'  TOTAL   : {total_correct}/{len(df_te)} = {total_correct/len(df_te):.1%}')

print(f'\n{"─"*60}')
print(f'  2. ACERTIVIDADE POR NIVEL DE CONFIANCA (prob_long)')
print(f'{"─"*60}')
print(f'  {"Threshold":<12} {"N":<8} {"LONG%":<10} {"SHORT%":<10} {"Acertou%":<10}')
print(f'  {"-"*48}')
for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    sub = df_te[df_te['prob_long'] >= thresh]
    if len(sub) < 10: continue
    long_actual = (sub['label'] == 2).mean()
    short_actual = (sub['label'] == 0).mean()
    pred_long = (sub['pred'] == 2).mean()
    print(f'  prob>={thresh:.1f}  {len(sub):<8} {long_actual:<10.1%} {short_actual:<10.1%} {pred_long:<10.1%}')

print(f'\n  {"Threshold":<12} {"N":<8} {"SHORT%":<10} {"LONG%":<10} {"Acertou%":<10}')
print(f'  {"-"*48}')
for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    sub = df_te[df_te['prob_short'] >= thresh]
    if len(sub) < 10: continue
    short_actual = (sub['label'] == 0).mean()
    long_actual = (sub['label'] == 2).mean()
    pred_short = (sub['pred'] == 0).mean()
    print(f'  prob>={thresh:.1f}  {len(sub):<8} {short_actual:<10.1%} {long_actual:<10.1%} {pred_short:<10.1%}')

print(f'\n{"─"*60}')
print(f'  3. ACERTIVIDADE POR DIFERENCA DE CONFIANCA (long - short)')
print(f'{"─"*60}')
df_te['conf_diff'] = df_te['prob_long'] - df_te['prob_short']
for diff in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
    sub = df_te[df_te['conf_diff'] > diff]
    if len(sub) < 5: continue
    long_actual = (sub['label'] == 2).mean()
    short_actual = (sub['label'] == 0).mean()
    print(f'  LONG (dif>{diff:.2f}): N={len(sub):<6} LONG={long_actual:.1%}  SHORT={short_actual:.1%}  edge={long_actual-short_actual:.1%}')
    sub = df_te[df_te['conf_diff'] < -diff]
    if len(sub) < 5: continue
    short_actual = (sub['label'] == 0).mean()
    long_actual = (sub['label'] == 2).mean()
    print(f'  SHORT(dif<{-diff:.2f}): N={len(sub):<6} SHORT={short_actual:.1%}  LONG={long_actual:.1%}  edge={short_actual-long_actual:.1%}')

print(f'\n{"─"*60}')
print(f'  4. us_prime_setup + confianca')
print(f'{"─"*60}')
sub = df_te[df_te['us_prime_setup'] == 1]
if len(sub) > 0:
    for thresh in [0.0, 0.3, 0.4, 0.5]:
        sub2 = sub[sub['prob_long'] >= thresh]
        if len(sub2) < 3: continue
        long_actual = (sub2['label'] == 2).mean()
        short_actual = (sub2['label'] == 0).mean()
        print(f'  prob>={thresh:.1f}: N={len(sub2):<5} LONG={long_actual:.1%}  SHORT={short_actual:.1%}')

print(f'\n{"─"*60}')
print(f'  5. ACERTIVIDADE POR HORA')
print(f'{"─"*60}')
for h in sorted(df_te['hour'].unique()):
    sub = df_te[df_te['hour'] == h]
    if len(sub) < 20: continue
    long_actual = (sub['label'] == 2).mean()
    short_actual = (sub['label'] == 0).mean()
    pred_correct = (sub['pred'] == sub['label']).mean()
    if long_actual > 0.45:
        print(f'  {h:02d}h: acertou={pred_correct:.1%}  LONG_real={long_actual:.1%}  SHORT_real={short_actual:.1%}  ← melhor LONG')
    elif short_actual > 0.35:
        print(f'  {h:02d}h: acertou={pred_correct:.1%}  LONG_real={long_actual:.1%}  SHORT_real={short_actual:.1%}')

print(f'\n{"─"*60}')
print(f'  6. VIÉS EMA20 + CONFIANCA')
print(f'{"─"*60}')
for bias, lbl in [(0, 'ambos abaixo'), (1, 'misturado'), (2, 'ambos acima')]:
    sub = df_te[df_te['ema20_bias_mnq_btc'] == bias]
    if len(sub) < 10: continue
    long_actual = (sub['label'] == 2).mean()
    short_actual = (sub['label'] == 0).mean()
    pred_correct = (sub['pred'] == sub['label']).mean()
    print(f'  {lbl:<15}: N={len(sub):<6} LONG={long_actual:.1%}  SHORT={short_actual:.1%}  acertou={pred_correct:.1%}')

print(f'\n{"─"*60}')
print(f'  RESUMO - QUANDO O MODELO DA SINAL DE LONG/SHORT')
print(f'{"─"*60}')
for target, nome in [(2, 'LONG'), (0, 'SHORT')]:
    sub = df_te[df_te['pred'] == target]
    correct = (sub['label'] == target).sum()
    total = len(sub)
    wrong_opposite = (sub['label'] == (0 if target == 2 else 2)).sum()
    if total > 0:
        print(f'  {nome}:')
        print(f'    Total sinais: {total}')
        print(f'    Acertou: {correct} ({correct/total:.1%})')
        print(f'    Errou contrario: {wrong_opposite} ({wrong_opposite/total:.1%})')
        print(f'    Fator de acerto (acertou/errou_contrario): {correct/max(wrong_opposite,1):.2f}x')

print(f'\n{"="*60}')
print(f'  RESUMO FINAL')
print(f'{"="*60}')
print(f'  Acertou classe exata: {total_correct}/{len(df_te)} = {total_correct/len(df_te):.1%}')
sub_dir = df_te[df_te['label'] != 1]
dir_correct = (sub_dir['pred'] == sub_dir['label']).sum()
print(f'  Acertou direcao (ignorando NEUTRO): {dir_correct}/{len(sub_dir)} = {dir_correct/len(sub_dir):.1%}')
close_count = ((df_te['pred'] == 1) & (df_te['label'] != 1)).sum()
print(f'  Neutro quando deveria ser direcao: {close_count} ({close_count/len(df_te):.1%})')

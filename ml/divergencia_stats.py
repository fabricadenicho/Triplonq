"""
Calcula acertos/erros do modelo divergencia.
Lê logs_divergencia.csv, consulta dados reais do MNQ, atualiza performance.

Uso: python divergencia_stats.py
Saida: JSON com stats de acerto
"""
import json, csv, sqlite3, warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
LOG_PATH = BASE / 'logs_divergencia.csv'
DB_PATH = BASE / 'data.db'

def main():
    if not LOG_PATH.exists():
        print(json.dumps({'error': 'Nenhum log encontrado'}))
        return

    df = pd.read_csv(LOG_PATH)
    if len(df) == 0:
        print(json.dumps({'error': 'Log vazio'}))
        return

    # Obter dados reais do MNQ para preencher mnq_ret_fwd
    conn = sqlite3.connect(DB_PATH)
    mnq = pd.read_sql(
        "SELECT ts,close FROM candles WHERE symbol='mnq' AND LENGTH(ts)=19 ORDER BY ts",
        conn, parse_dates=['ts'], index_col='ts')
    conn.close()

    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values('ts')

    # Preencher retorno futuro (4h) para cada predicao
    fwd_hours = 4
    retornos = []
    for ts in df['ts']:
        if ts in mnq.index:
            idx_pos = mnq.index.get_loc(ts)
            fwd_idx = idx_pos + fwd_hours
            if fwd_idx < len(mnq):
                ret = mnq['close'].iloc[fwd_idx] / mnq['close'].iloc[idx_pos] - 1
                retornos.append(ret)
            else:
                retornos.append(None)
        else:
            retornos.append(None)

    df['mnq_ret_fwd'] = retornos
    df['acertou'] = ((df['pred'] == 1) & (df['mnq_ret_fwd'] > 0.001)).astype(int)
    df['errou'] = ((df['pred'] == 1) & (df['mnq_ret_fwd'] <= 0.001)).astype(int)

    # Salvar de volta com resultados
    df.to_csv(LOG_PATH, index=False)

    # Estatisticas
    total = len(df)
    total_pred_1 = df['pred'].sum()
    total_acertos = df['acertou'].sum()
    total_erros = df['errou'].sum()
    pendentes = total_pred_1 - total_acertos - total_erros

    wr = total_acertos / total_pred_1 * 100 if total_pred_1 > 0 else 0
    wr_overall = (df['mnq_ret_fwd'] > 0.001).mean() * 100 if df['mnq_ret_fwd'].notna().sum() > 0 else 0

    # Por threshold
    thresholds = {}
    for th in [0.4, 0.5, 0.6]:
        sub = df[df['prob'] >= th].copy()
        if len(sub) == 0: continue
        sub['acertou_th'] = ((sub['mnq_ret_fwd'] > 0.001)).astype(int)
        acertos_th = sub['acertou_th'].sum()
        thresholds[str(th)] = {
            'n': int(len(sub)),
            'acertos': int(acertos_th),
            'wr': round(acertos_th / len(sub) * 100, 1),
        }

    # Checklist score performance
    checklist_bins = {}
    for lo, hi in [(0, 40), (40, 60), (60, 80), (80, 101)]:
        sub = df[(df['checklist_score'] >= lo) & (df['checklist_score'] < hi)]
        if len(sub) < 3: continue
        ac = sub['acertou'].sum()
        checklist_bins[f'{lo}-{hi}'] = {
            'n': int(len(sub)), 'acertos': int(ac), 'wr': round(ac / len(sub) * 100, 1) if len(sub) > 0 else 0
        }

    # Ultimos 20 registros
    recent = df.tail(20).to_dict('records')
    recent_out = []
    for r in recent:
        recent_out.append({
            'ts': str(r['ts']), 'prob': round(r['prob'], 3),
            'pred': int(r['pred']), 'ret_fwd': round(r['mnq_ret_fwd'], 4) if pd.notna(r.get('mnq_ret_fwd')) else None,
            'acertou': int(r['acertou']) if pd.notna(r.get('acertou')) else None,
        })

    print(json.dumps({
        'total_predicoes': int(total),
        'total_pred_1': int(total_pred_1),
        'acertos': int(total_acertos),
        'erros': int(total_erros),
        'pendentes': int(pendentes),
        'wr': round(wr, 1),
        'baseline_global': round(wr_overall, 1),
        'por_threshold': thresholds,
        'por_checklist': checklist_bins,
        'recentes': recent_out,
        'ts': str(pd.Timestamp.now()),
    }))

if __name__ == '__main__':
    main()

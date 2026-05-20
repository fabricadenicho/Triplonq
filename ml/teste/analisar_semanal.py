"""
Analisa relacao MNQ x CL nos ultimos 12 meses.
Saida: JSON com estatisticas por dia da semana e insights.
Chamado pelo server.js via child_process.
"""
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
import json, sys
from datetime import datetime, timedelta

fim = datetime.now()
inicio = fim - timedelta(days=370)

try:
    mnq = yf.download('MNQ=F', start=inicio, end=fim, interval='1d', auto_adjust=True, progress=False)
    cl  = yf.download('CL=F',  start=inicio, end=fim, interval='1d', auto_adjust=True, progress=False)

    for df in [mnq, cl]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.lower()

    mnq_ret = mnq['close'].pct_change() * 100
    cl_ret  = cl['close'].pct_change() * 100

    df = pd.DataFrame({
        'mnq_ret': mnq_ret, 'cl_ret': cl_ret,
        'dow': mnq_ret.index.dayofweek,
        'dow_name': mnq_ret.index.strftime('%A'),
    }).dropna()

    # Classificar
    def classif(r):
        m, c = r['mnq_ret'], r['cl_ret']
        if m > 0.3 and c > 0.3: return 'AMBOS_POS'
        if m < -0.3 and c < -0.3: return 'AMBOS_NEG'
        if m > 0.3 and c < -0.3: return 'MNQ_POS_CL_NEG'
        if m < -0.3 and c > 0.3: return 'MNQ_NEG_CL_POS'
        return 'NEUTRO'
    df['classe'] = df.apply(classif, axis=1)
    df['divergente'] = df['classe'].str.contains('POS_CL_NEG|NEG_CL_POS')
    df['alinhado'] = df['classe'].isin(['AMBOS_POS', 'AMBOS_NEG'])

    dias_pt = ['Segunda', 'Terca', 'Quarta', 'Quinta', 'Sexta']
    dias = []

    for i, nome_pt in enumerate(dias_pt):
        sub = df[df['dow'] == i]
        if len(sub) < 5:
            continue
        tot = len(sub)
        c = sub['classe'].value_counts()
        distribuicao = {}
        for cls in ['AMBOS_POS', 'AMBOS_NEG', 'MNQ_POS_CL_NEG', 'MNQ_NEG_CL_POS']:
            qtd = int(c.get(cls, 0))
            distribuicao[cls] = {'dias': qtd, 'pct': round(qtd/tot*100, 1)}

        mais_comum = max(distribuicao, key=lambda k: distribuicao[k]['dias'])
        div_qtd = int(sub['divergente'].sum())
        ali_qtd = int(sub['alinhado'].sum())

        dias.append({
            'nome': nome_pt,
            'nome_en': ['Monday','Tuesday','Wednesday','Thursday','Friday'][i],
            'total_dias': tot,
            'distribuicao': distribuicao,
            'mais_comum': mais_comum,
            'divergentes': div_qtd,
            'divergentes_pct': round(div_qtd/tot*100, 1),
            'alinhados': ali_qtd,
            'alinhados_pct': round(ali_qtd/tot*100, 1),
            'mnq_ret_medio': round(float(sub['mnq_ret'].mean()), 2),
            'cl_ret_medio': round(float(sub['cl_ret'].mean()), 2),
        })

    # Gerais
    correlacao = round(float(df['mnq_ret'].corr(df['cl_ret'])), 3)
    distribuicao_geral = {}
    for cls in ['AMBOS_POS', 'AMBOS_NEG', 'MNQ_POS_CL_NEG', 'MNQ_NEG_CL_POS']:
        qtd = int((df['classe'] == cls).sum())
        distribuicao_geral[cls] = {'dias': qtd, 'pct': round(qtd/len(df)*100, 1)}

    # Quando MNQ forte
    mnq_alta = df[df['mnq_ret'] > 1]
    mnq_baixa = df[df['mnq_ret'] < -1]

    resultado = {
        'periodo': {'inicio': str(inicio.date()), 'fim': str(fim.date()), 'dias_totais': len(df)},
        'correlacao': correlacao,
        'distribuicao_geral': distribuicao_geral,
        'dias': dias,
        'insights': {
            'mnq_alta': {
                'dias': len(mnq_alta),
                'cl_positivo_pct': round((mnq_alta['cl_ret'] > 0).mean()*100, 1) if len(mnq_alta) > 0 else 0,
                'cl_retorno_medio': round(float(mnq_alta['cl_ret'].mean()), 2) if len(mnq_alta) > 0 else 0,
            },
            'mnq_baixa': {
                'dias': len(mnq_baixa),
                'cl_positivo_pct': round((mnq_baixa['cl_ret'] > 0).mean()*100, 1) if len(mnq_baixa) > 0 else 0,
                'cl_retorno_medio': round(float(mnq_baixa['cl_ret'].mean()), 2) if len(mnq_baixa) > 0 else 0,
            },
            'melhor_dia_mnq': '',
            'melhor_dia_cl': '',
            'pior_dia_mnq': '',
            'pior_dia_cl': '',
        }
    }

    # Media por dia
    ret_medio = df.groupby('dow_name')[['mnq_ret','cl_ret']].mean()
    resultado['retorno_medio_por_dia'] = {}
    for nome_pt, nome_en in zip(dias_pt, ['Monday','Tuesday','Wednesday','Thursday','Friday']):
        if nome_en in ret_medio.index:
            resultado['retorno_medio_por_dia'][nome_pt] = {
                'mnq': round(float(ret_medio.loc[nome_en, 'mnq_ret']), 2),
                'cl': round(float(ret_medio.loc[nome_en, 'cl_ret']), 2),
            }

    # Top divergentes
    div_mask = df['classe'].isin(['MNQ_POS_CL_NEG', 'MNQ_NEG_CL_POS'])
    top_div = df[div_mask].nlargest(8, 'mnq_ret' if df[div_mask]['mnq_ret'].abs().mean() > df[div_mask]['cl_ret'].abs().mean() else 'cl_ret')
    resultado['top_divergentes'] = []
    for _, r in top_div.iterrows():
        resultado['top_divergentes'].append({
            'data': str(r.name.date()),
            'mnq_ret': round(float(r['mnq_ret']), 2),
            'cl_ret': round(float(r['cl_ret']), 2),
            'dia': dias_pt[int(r['dow'])],
            'classe': r['classe'],
        })

    print(json.dumps(resultado, ensure_ascii=False))

except Exception as e:
    print(json.dumps({'erro': str(e)}, ensure_ascii=False))

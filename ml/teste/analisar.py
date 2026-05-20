"""
Analisa modelo treinado: importancia, padroes, periodo de treino.
Uso: python analisar.py --asset mnq
"""
import argparse, pickle, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent

ASSET_NAMES = {
    'mnq': 'MNQ (Micro Nasdaq)',
    'btc': 'BTC (Bitcoin)',
    'cl':  'CL (Crude Oil)',
    'mgc': 'MGC (Micro Gold)',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', type=str, default='mnq', choices=['mnq', 'btc', 'cl', 'mgc'])
    ap.add_argument('--n', type=int, default=25, help='Top N features (default 25)')
    args = ap.parse_args()

    pkl_path = MODEL_DIR / f'propfirm_model_{args.asset}.pkl'
    if not pkl_path.exists():
        print(f'Modelo nao encontrado: {pkl_path}')
        return

    data = pickle.load(open(pkl_path, 'rb'))
    model = data['model']
    feats = data['features']
    meta  = data.get('metadata', {})

    asset_name = ASSET_NAMES.get(args.asset, args.asset.upper())

    print(f'==============================================')
    print(f'  ANALISE DO MODELO - {asset_name}')
    print(f'==============================================')

    print(f'\n>> PERIODO DE TREINO')
    if meta:
        print(f'  Treino: {meta["train_start"]} ate {meta["train_end"]}  ({meta["train_samples"]} amostras)')
        print(f'  Teste:  {meta["test_start"]} ate {meta["test_end"]}  ({meta["test_samples"]} amostras)')
    print(f'  Forward: {data.get("forward", "?")}h')
    print(f'  AUC: {data.get("auc", "?"):.4f}  |  Acurácia: {data.get("acc", "?"):.2%}')
    print(f'  Total features: {len(feats)}')

    # ── Feature Importance ──
    top_n = args.n
    imp_gain = pd.DataFrame({
        'feature': feats,
        'gain': model.feature_importances_,
    }).sort_values('gain', ascending=False).head(top_n)

    w_dict = model.get_booster().get_score(importance_type='weight')
    imp_weight = pd.DataFrame([{'feature': f, 'weight': w_dict.get(f, 0)} for f in feats])
    imp_weight = imp_weight.sort_values('weight', ascending=False).head(top_n)

    print(f'\n>> TOP {top_n} FEATURES POR IMPORTANCIA (GAIN)')
    print(f'  {"#":>3}  {"Feature":<30} {"Gain":>8}  {"Acum":>8}')
    print(f'  {"---":>3}  {"-"*30} {"------":>8} {"------":>8}')
    total_gain = imp_gain['gain'].sum()
    cum = 0
    for i, (_, r) in enumerate(imp_gain.iterrows(), 1):
        cum += r['gain']
        print(f'  {i:>3}  {r["feature"]:<30} {r["gain"]:.4f}  {cum/total_gain:.1%}')

    print(f'\n>> TOP {top_n} FEATURES POR FREQUENCIA (WEIGHT)')
    print(f'  {"#":>3}  {"Feature":<30} {"Vezes":>8}')
    print(f'  {"---":>3}  {"-"*30} {"------":>8}')
    for i, (_, r) in enumerate(imp_weight.iterrows(), 1):
        print(f'  {i:>3}  {r["feature"]:<30} {r["weight"]:>8.0f}')

    # ── Feature categories ──
    cats = {
        'RSI / Divergencias': ['rsi_p', 'rsi_1', 'rsi_2', 'div_1', 'div_2', 'rsi_spread_1_2',
                                  'rsi_abs_p_1', 'rsi_abs_p_2', 'rsi_abs_1_2'],
        'ADX / Tendencia': ['adx_p', 'adx_1', 'adx_2', 'pdi_p', 'mdi_p',
                              'adx_spread_p_1', 'adx_spread_p_2', 'adx_spread_1_2',
                              'adx_abs_p_1', 'adx_abs_p_2', 'adx_abs_1_2',
                              'di_spread_p', 'di_spread_1', 'di_spread_2', 'dadx_p'],
        'Retornos / Volatilidade': ['ret1_p', 'ret4_p', 'ret8_p', 'vol_p', 'bb_p',
                                      'ret1_1', 'ret4_1', 'ret1_2', 'ret4_2',
                                      'ret1_spread_p_1', 'ret1_spread_p_2', 'ret1_spread_1_2',
                                      'ret4_spread_p_1', 'ret4_spread_p_2', 'ret4_spread_1_2',
                                      'ret1_prod_p_1', 'ret1_prod_1_2',
                                      'price_div_p_2', 'price_div_abs',
                                      'ret4_prod_p_1', 'ret4_prod_p_2', 'ret4_prod_1_2',
                                      'vol_spread_p_1', 'vol_spread_p_2', 'vol_spread_1_2',
                                      'bb_spread_p_1', 'bb_spread_p_2', 'bb_spread_1_2'],
        'Medias (SMA50/EMA20)': ['dist_sma50_p', 'dist_sma50_1', 'dist_sma50_2',
                                   'sma50_slope_p', 'above_sma50_p', 'above_sma50_1', 'above_sma50_2',
                                   'sma50_alignment', 'sma50_dist_spread_p_1', 'sma50_dist_spread_p_2',
                                   'sma50_dist_spread_1_2', 'sma50_align_p_1', 'sma50_align_p_2',
                                   'sma50_align_1_2',
                                   'dist_ema20_p', 'dist_ema20_1', 'dist_ema20_2',
                                   'above_ema20_p', 'above_ema20_1', 'above_ema20_2',
                                   'ema20_bias_p_1', 'ema20_alignment', 'ema20_dist_spread_p_1',
                                   'ema20_dist_spread_p_2', 'ema20_dist_spread_1_2',
                                   'ema20_align_p_1', 'ema20_align_p_2', 'ema20_align_1_2'],
        'Tempo': ['hour', 'dow', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
    }

    # Key levels extra
    kl_feats = [f for f in feats if f.startswith(('dist_to_', 'above_', 'prev_day'))]
    if kl_feats:
        cats['Key Levels'] = kl_feats

    print(f'\n>> DISTRIBUICAO DE IMPORTANCIA POR CATEGORIA')
    full_imp = pd.DataFrame({
        'feature': feats,
        'gain': model.feature_importances_,
    })
    imp_map = dict(zip(full_imp['feature'], full_imp['gain']))
    total = sum(imp_map.values())
    for cat_name, cat_feats in cats.items():
        cat_imp = sum(imp_map.get(f, 0) for f in cat_feats if f in feats)
        if cat_imp > 0:
            print(f'  {cat_name}: {cat_imp/total:.1%}')

    # ── Top patterns (direcao das features mais importantes) ──
    print(f'\n>> PADROES APRENDIDOS (TOP 5 features)')
    print(f'  (Baseado nas arvores de decisao do XGBoost)')
    top5 = imp_gain.head(5)
    booster = model.get_booster()
    for _, r in top5.iterrows():
        f = r['feature']
        g = r['gain']
        # Analisa splits da feature nas arvores
        trees = booster.get_dump(with_stats=True)
        splits_above = 0
        splits_below = 0
        total_splits = 0
        for t in trees:
            for line in t.split('\n'):
                if f'[{f}<' in line:
                    total_splits += 1
                    if 'yes' in line and 'no' in line:
                        pass
        print(f'  {f:<28} gain={g:.4f}  (usada em ~{total_splits} splits)')

    # ── Dicas praticas ──
    print(f'\n>> INSIGHTS')
    rsi_p_gain = imp_map.get('rsi_p', 0)
    adx_p_gain = imp_map.get('adx_p', 0)
    vol_p_gain = imp_map.get('vol_p', 0)
    hour_sin_g = imp_map.get('hour_sin', 0)
    hour_cos_g = imp_map.get('hour_cos', 0)
    time_gain = hour_sin_g + hour_cos_g + imp_map.get('hour', 0)
    print(f'  • RSI do ativo principal representa {rsi_p_gain/total:.1%} da importancia total')
    print(f'  • ADX (forca da tendencia): {adx_p_gain/total:.1%}')
    print(f'  • Volatilidade: {vol_p_gain/total:.1%}')
    print(f'  • Componentes horarios (hora do dia): {time_gain/total:.1%}')


if __name__ == '__main__':
    main()

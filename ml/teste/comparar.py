"""
Compara modelo antigo (com vies) vs novo (sem vies) para todos os ativos.
Carrega dados do DB existente — nao baixa nada novo.
Uso: python comparar.py
"""
import sys, importlib.util, warnings
warnings.filterwarnings('ignore')

import sqlite3
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score

BASE = Path(__file__).parent.parent
TESTE_DIR = Path(__file__).parent

ASSET_CONFIG = {
    'mnq': {'db': 'data.db',         'syms': ['mnq', 'btc', 'cl'], 'old_dir': ''},
    'btc': {'db': 'btc/data.db',     'syms': ['btc', 'mnq', 'cl'], 'old_dir': 'btc'},
    'cl':  {'db': 'cl/data.db',      'syms': ['cl',  'mnq', 'btc'], 'old_dir': 'cl'},
    'mgc': {'db': 'mgc/data.db',     'syms': ['mgc', 'mnq', 'btc'], 'old_dir': 'mgc'},
}


def import_old_module(asset):
    """Importa o train.py antigo do ativo como modulo."""
    old_file = BASE / ASSET_CONFIG[asset]['old_dir'] / 'train.py'
    if not old_file.exists():
        return None
    spec = importlib.util.spec_from_file_location(f'old_train_{asset}', old_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_old_model(asset):
    """Carrega model.pkl antigo."""
    old_dir = ASSET_CONFIG[asset]['old_dir']
    pkl_path = BASE / old_dir / 'model.pkl' if old_dir else BASE / 'model.pkl'
    if not pkl_path.exists():
        return None
    return pickle.load(open(pkl_path, 'rb'))


def load_new_model(asset):
    """Carrega propfirm_model_{asset}.pkl novo."""
    pkl_path = TESTE_DIR / f'propfirm_model_{asset}.pkl'
    if not pkl_path.exists():
        return None
    return pickle.load(open(pkl_path, 'rb'))


def evaluate(y_true, proba, preds, model_name):
    acc = (preds == y_true).mean()
    try:
        auc = roc_auc_score(y_true, proba, multi_class='ovr')
    except:
        auc = 0.0
    n = len(y_true)
    return {'model': model_name, 'auc': round(auc, 4), 'acc': round(acc, 4), 'n': n}


def test_model(model_data, X, y_true, name):
    model = model_data['model']
    feats = model_data['features']
    common = [c for c in feats if c in X.columns]
    if len(common) < len(feats):
        missing = set(feats) - set(X.columns)
        print(f'  [!] {name}: {len(missing)} features faltando: {list(missing)[:5]}...')
    X_sub = X[common]
    proba = model.predict_proba(X_sub)
    preds = model.predict(X_sub)
    return evaluate(y_true, proba, preds, name)


def main():
    results = []

    for asset in ['mnq', 'btc', 'cl', 'mgc']:
        cfg = ASSET_CONFIG[asset]
        print(f'\n{"="*60}')
        print(f'  {asset.upper()}')
        print(f'{"="*60}')

        db_path = BASE / cfg['db']
        if not db_path.exists():
            print(f'  DB nao encontrado: {db_path}')
            continue

        # --- Carrega modelos ---
        old_model = load_old_model(asset)
        new_model = load_new_model(asset)

        if old_model is None:
            print(f'  Modelo antigo nao encontrado')
        if new_model is None:
            print(f'  Modelo novo nao encontrado')
        if old_model is None or new_model is None:
            continue

        old_fwd = old_model.get('forward', 4)
        new_fwd = new_model.get('forward', 4)
        fwd = min(old_fwd, new_fwd)
        print(f'  Forward: {fwd}h  (old={old_fwd}, new={new_fwd})')

        conn = sqlite3.connect(db_path)

        # --- Features VELHAS (com vies) ---
        old_mod = import_old_module(asset)
        if old_mod is not None:
            print(f'  Gerando features (old / com vies)...')
            df_old = old_mod.build_features(conn, interval='1h', forward=fwd)
        else:
            df_old = None

        # --- Features NOVAS (sem vies) ---
        from train import build_features as new_build_features
        print(f'  Gerando features (new / sem vies)...')
        df_new = new_build_features(conn, interval='1h', forward=fwd, syms=cfg['syms'])
        conn.close()

        # --- Avalia old ---
        if df_old is not None and 'label' in df_old.columns:
            feat_cols_old = old_model['features']
            y_old = df_old['label'].values
            X_old = df_old[[c for c in feat_cols_old if c in df_old.columns]]
            if len(X_old) > 0:
                r = test_model(old_model, X_old, y_old, f'{asset.upper()} (antigo)')
                results.append(r)
                print(f'  >> {asset} antigo: AUC={r["auc"]:.4f}  Acc={r["acc"]:.2%}  N={r["n"]}')

        # --- Avalia new ---
        if 'label' in df_new.columns:
            feat_cols_new = new_model['features']
            y_new = df_new['label'].values
            X_new = df_new[[c for c in feat_cols_new if c in df_new.columns]]
            r = test_model(new_model, X_new, y_new, f'{asset.upper()} (novo)')
            results.append(r)
            print(f'  >> {asset} novo: AUC={r["auc"]:.4f}  Acc={r["acc"]:.2%}  N={r["n"]}')

    # --- Tabela resumo ---
    print(f'\n{"="*60}')
    print(f'  RESUMO COMPARATIVO')
    print(f'{"="*60}')
    print(f'  {"Ativo":<8} {"Modelo":<12} {"AUC":<8} {"Acurácia":<10} {"Amostras":<10}')
    print(f'  {"-"*8} {"-"*12} {"-"*8} {"-"*10} {"-"*10}')
    for r in results:
        acc_str = f'{r["acc"]:.2%}'
        print(f'  {r["model"][:7]:<8} {r["model"][9:12]:<12} {r["auc"]:<8.4f} {acc_str:<10} {r["n"]:<10}')

    # Diferenca
    print(f'\n  DIFERENCA (novo - antigo):')
    old_by_asset = {r['model'].split()[0].lower(): r for r in results if 'antigo' in r['model']}
    new_by_asset = {r['model'].split()[0].lower(): r for r in results if 'novo' in r['model']}
    for asset in ['mnq', 'btc', 'cl', 'mgc']:
        o = old_by_asset.get(asset)
        n = new_by_asset.get(asset)
        if o and n:
            print(f'  {asset.upper():<6} AUC: {n["auc"]-o["auc"]:+.4f}  Acc: {n["acc"]-o["acc"]:+.2%}')


if __name__ == '__main__':
    main()

"""
Retreina todos os modelos ML Prop Firm em um comando so.
Fonte: yfinance (auto_adjust=True) — sem contaminacao de rollover.
Metodologia: walk-forward por ano + blind OOS genuino (early stopping interno).
"""
import subprocess, sys, time
from pathlib import Path

BASE   = Path(__file__).parent
PYTHON = sys.executable
ASSETS = ['mnq', 'btc', 'cl', 'es']


def step(msg):
    print(f'\n{"="*60}')
    print(f'  {msg}')
    print(f'{"="*60}')


def run(cmd, cwd=None, timeout=600):
    print(f'  -> {cmd}')
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=cwd or str(BASE),
                       capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    out = (r.stdout or '').strip()
    err = (r.stderr or '').strip()
    for line in out.split('\n'):
        if any(kw in line.lower() for kw in ['auc', 'acuracia', 'modelo salvo',
                                              'dataset', 'blind oos', 'live 2026',
                                              'treino real', 'filtro rollover',
                                              'resumo', 'ok', 'fraco']):
            print(f'    {line.strip()}')
    if r.returncode != 0:
        print(f'  [ERRO] returncode={r.returncode}')
        if err:
            print(f'  {err[:500]}')
        return False
    print(f'  [OK] {dt:.0f}s')
    return True


def main():
    t0 = time.time()

    step('TREINANDO PROPFIRM MODELS (MNQ/BTC/CL/ES) — SQLite limpo (5 anos)')
    ok1 = run(f'"{PYTHON}" train_sqlite_clean.py --all --forward 8', cwd=str(BASE))

    step('TREINANDO MODEL DIVERGENCIA (MNQ>0.1% em 4h) — yfinance')
    ok2 = run(f'"{PYTHON}" train_divergencia_yfinance.py --forward 4', cwd=str(BASE / '..'))

    dt = time.time() - t0
    ok = ok1 and ok2
    print(f'\n{"="*60}')
    status = 'COMPLETO' if ok else 'COM ERROS'
    print(f'  RETREINO {status} em {dt:.0f}s')
    print(f'  PropFirm: {", ".join(a.upper() for a in ASSETS)}')
    print(f'  Divergencia: model_divergencia.pkl')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()

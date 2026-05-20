"""
Retreina todos os modelos ML Prop Firm em um comando so.
1. Coleta dados atualizados dos 4 DBs
2. Treina os 4 modelos (MNQ, BTC, CL, MGC) com forward=8h
"""
import subprocess, sys, time
from pathlib import Path

BASE = Path(__file__).parent
PYTHON = sys.executable

ASSETS = ['mnq', 'btc', 'cl', 'mgc']

# Subdiretorios de cada DB
DB_DIRS = {
    'mnq': BASE / '..',
    'btc': BASE / '..' / 'btc',
    'cl':  BASE / '..' / 'cl',
    'mgc': BASE / '..' / 'mgc',
}

def step(msg):
    print(f'\n{"="*60}')
    print(f'  {msg}')
    print(f'{"="*60}')

def run(cmd, cwd=None, timeout=300):
    print(f'  -> {cmd}')
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=cwd or str(BASE),
                       capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    out = (r.stdout or '').strip()
    err = (r.stderr or '').strip()
    # So mostra erros e linhas importantes
    for line in out.split('\n'):
        if any(kw in line.lower() for kw in ['error', 'auc', 'acuracia', 'modelo salvo',
                                              'dataset', 'f1-score', 'precis', 'recall',
                                              'walk-forward', 'treino', 'teste',
                                              'novos', 'candles', 'db salvo']):
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

    # Step 1: Coletar dados de cada DB
    step('COLETANDO DADOS — MNQ/BTC/CL (main)')
    run(f'"{PYTHON}" collect_data.py --force', cwd=str(BASE / '..'))

    for asset in ['btc', 'cl', 'mgc']:
        step(f'COLETANDO DADOS — {asset.upper()}')
        d = DB_DIRS[asset]
        run(f'"{PYTHON}" collect_data.py --force', cwd=str(d))

    # Step 2: Treinar modelos
    for asset in ASSETS:
        step(f'TREINANDO MODELO {asset.upper()} (forward=8h)')
        run(f'"{PYTHON}" train.py --asset {asset} --forward 8', cwd=str(BASE))

    dt = time.time() - t0
    print(f'\n{"="*60}')
    print(f'  RETREINO COMPLETO em {dt:.0f}s')
    print(f'  4 modelos atualizados: {", ".join(a.upper() for a in ASSETS)}')
    print(f'{"="*60}')

if __name__ == '__main__':
    main()

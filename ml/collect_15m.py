"""
Coletor incremental — 15m e 5m
Roda diariamente, baixa os ultimos 7 dias e faz merge com historico acumulado.
Armazena em ml/data_hist/{ticker}_{interval}.csv.gz

Uso:
  python ml/collect_15m.py           # coleta 15m e 5m
  python ml/collect_15m.py --status  # mostra resumo do historico atual
"""
import argparse, warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / 'data_hist'
DATA_DIR.mkdir(exist_ok=True)

TICKERS = {
    'MNQ=F':    'mnq',
    'ES=F':     'es',
    'CL=F':     'cl',
    'BTC-USD':  'btc',
}

INTERVALS = ['15m', '5m']


def fetch(ticker, interval, period='7d'):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.lower()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[['open','high','low','close','volume']].dropna()
    except Exception as e:
        print(f'  ERRO ao baixar {ticker} {interval}: {e}')
        return None


def load_existing(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True, compression='gzip')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def save(df, path):
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df.to_csv(path, compression='gzip')
    return df


def collect():
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f'[{now}] Iniciando coleta...')
    total_new = 0

    for interval in INTERVALS:
        print(f'\n-- {interval} --')
        for ticker, nome in TICKERS.items():
            path = DATA_DIR / f'{nome}_{interval}.csv.gz'
            existing = load_existing(path)

            new = fetch(ticker, interval)
            if new is None:
                print(f'  {nome:>4} {interval}  FALHOU')
                continue

            if existing.empty:
                merged = new
                added = len(new)
            else:
                merged = pd.concat([existing, new])
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                added = len(merged) - len(existing)

            save(merged, path)
            total_new += max(added, 0)
            print(f'  {nome:>4} {interval}  total={len(merged):>6}  novos={max(added,0):>4}  '
                  f'de {merged.index[0].date()} ate {merged.index[-1].date()}')

    print(f'\nColeta concluida. {total_new} barras novas adicionadas.')
    return total_new


def status():
    print('\nHistorico acumulado em ml/data_hist/\n')
    print(f'  {"Ativo":<8} {"TF":<5} {"Barras":>7}  {"De":>12}  {"Ate":>12}  {"Dias":>5}')
    print('-' * 60)
    for interval in INTERVALS:
        for ticker, nome in TICKERS.items():
            path = DATA_DIR / f'{nome}_{interval}.csv.gz'
            df = load_existing(path)
            if df.empty:
                print(f'  {nome:<8} {interval:<5} {"---":>7}  {"sem dados":>12}')
                continue
            dias = (df.index[-1] - df.index[0]).days
            print(f'  {nome:<8} {interval:<5} {len(df):>7}  '
                  f'{str(df.index[0].date()):>12}  '
                  f'{str(df.index[-1].date()):>12}  {dias:>5}d')

    print()
    # Recomendacao para retreinar
    for interval in INTERVALS:
        path_ref = DATA_DIR / f'mnq_{interval}.csv.gz'
        df = load_existing(path_ref)
        if df.empty:
            continue
        dias = (df.index[-1] - df.index[0]).days
        if dias >= 180:
            print(f'  {interval}: {dias} dias acumulados -> PRONTO para retreinar modelo')
        elif dias >= 90:
            print(f'  {interval}: {dias} dias acumulados -> mais {180-dias} dias para retreino ideal')
        else:
            print(f'  {interval}: {dias} dias acumulados -> acumulando... (meta: 180 dias)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()

    if args.status:
        status()
    else:
        collect()
        status()

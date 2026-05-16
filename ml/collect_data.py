"""
Baixa dados históricos OHLCV e salva em SQLite.
Uso: python collect_data.py
"""
import sqlite3
import yfinance as yf
import pandas as pd
from pathlib import Path

DB   = Path(__file__).parent / 'data.db'
SYMS = {'mnq': 'MNQ=F', 'btc': 'BTC-USD', 'cl': 'CL=F'}


def init_db(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            ts     TEXT NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (symbol, ts)
        )
    ''')
    conn.commit()


def fetch(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False, repair=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])


def save(conn, symbol, df):
    rows = [
        (symbol, str(ts), r.open, r.high, r.low, r.close, r.volume)
        for ts, r in df.iterrows()
    ]
    conn.executemany('INSERT OR IGNORE INTO candles VALUES (?,?,?,?,?,?,?)', rows)
    conn.commit()


def main():
    conn = sqlite3.connect(DB)
    init_db(conn)

    configs = [
        ('1h',  '730d',  'Horário 2 anos'),
        ('1d',  '5y',    'Diário  5 anos'),
    ]

    for key, ticker in SYMS.items():
        print(f'\n{key.upper()} ({ticker})')
        for interval, period, label in configs:
            df = fetch(ticker, interval, period)
            if df.empty:
                print(f'  {label}: sem dados')
                continue
            before = conn.execute(
                'SELECT COUNT(*) FROM candles WHERE symbol=?', (key,)
            ).fetchone()[0]
            save(conn, key, df)
            after = conn.execute(
                'SELECT COUNT(*) FROM candles WHERE symbol=?', (key,)
            ).fetchone()[0]
            print(f'  {label}: {after - before} novos  '
                  f'({df.index[0].date()} → {df.index[-1].date()})')

    for key in SYMS:
        total = conn.execute(
            'SELECT COUNT(*) FROM candles WHERE symbol=?', (key,)
        ).fetchone()[0]
        print(f'{key}: {total} candles no DB')

    conn.close()
    print(f'\nDB salvo em: {DB}')


if __name__ == '__main__':
    main()

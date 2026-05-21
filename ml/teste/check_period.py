import sqlite3, pandas as pd
from pathlib import Path
base = Path(__file__).parent.parent
c1 = sqlite3.connect(base / 'data.db')
c2 = sqlite3.connect(base / 'es/data.db')
m = pd.read_sql("SELECT ts FROM candles WHERE symbol='mnq' AND LENGTH(ts)=19 ORDER BY ts", c1, parse_dates=['ts'], index_col='ts')
e = pd.read_sql("SELECT ts FROM candles WHERE symbol='es' AND LENGTH(ts)=19 ORDER BY ts", c2, parse_dates=['ts'], index_col='ts')
i = m.index.intersection(e.index)
s = int(len(i) * 0.7)
print(f'Dados sincronizados: {i[0].date()} ate {i[-1].date()}')
print(f'Treino (70%): {i[0].date()} ate {i[s-1].date()} ({s} amostras)')
print(f'Teste  (30%): {i[s].date()} ate {i[-1].date()} ({len(i)-s} amostras)')
c1.close(); c2.close()

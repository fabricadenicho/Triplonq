import pickle, sys
from pathlib import Path

model_data = pickle.load(open(Path(__file__).parent / 'model.pkl', 'rb'))
model = model_data['model']
features = model_data['features']

importance = model.get_booster().get_score(importance_type='gain')
total = sum(importance.values())
rows = [(f, importance.get(f, 0), importance.get(f, 0) / total * 100) for f in features]
rows.sort(key=lambda x: x[1], reverse=True)

print('Top 20 features por importancia (gain) - BTC:')
print(f'{"Feature":<35} {"Gain":<10} {"%":<8}')
print('-' * 55)
for feat, gain, pct in rows[:20]:
    print(f'{feat:<35} {gain:<10.0f} {pct:.1f}%')

top5 = sum(r[1] for r in rows[:5])
print(f'\nTotal features: {len(rows)}')
print(f'Top 5 concentram: {top5 / total * 100:.1f}% da importancia total')
print(f'Top 10 concentram: {sum(r[1] for r in rows[:10]) / total * 100:.1f}%')

print('\n\nTop spreads (filtrando por spread_):')
spreads = [(f, g, p) for f, g, p in rows if 'spread' in f]
for feat, gain, pct in spreads[:15]:
    print(f'{feat:<35} {gain:<10.0f} {pct:.1f}%')

cats = {
    'Tempo (hora/sessao)': ['hour', 'dow', 'is_us', 'is_evening'],
    'Divergencia preco': ['price_div'],
    'Divergencia RSI': ['div_cl', 'div_btc', 'rsi_abs'],
    'EMA20 alignment': ['ema20_align', 'ema20_bias'],
    'MA50 alignment': ['sma50_align'],
    'Co-movement (prod)': ['prod'],
    'Spreads ADX': ['adx_spread', 'adx_abs'],
    'Spreads volatilidade': ['vol_spread'],
    'Spreads Bollinger': ['bb_spread'],
    'DI spread': ['di_spread'],
}
print('\n\nImportancia por categoria:')
cat_totals = {}
for cat, patterns in cats.items():
    total_cat = sum(g for f, g, p in rows if any(p in f for p in patterns))
    if total_cat > 0:
        cat_totals[cat] = total_cat
        print(f'  {cat:<30} {total_cat/total*100:.1f}%')
print(f'  {"(outros)":<30} {(total - sum(cat_totals.values()))/total*100:.1f}%')

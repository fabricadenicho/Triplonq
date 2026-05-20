"""
Gera Pine Script v5 para backtest visual no TradingView.
Filtra ultimo 1 mes do CSV de trades.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

BASE = Path(__file__).parent

def ts_ms(dt_str):
    """Converte string datetime para Unix timestamp em milissegundos (UTC)."""
    dt = pd.Timestamp(dt_str)
    if dt.tzinfo is None:
        dt = dt.tz_localize('UTC')
    return int(dt.timestamp() * 1000)

def gerar_pine(asset, df_asset, out_path):
    trades = df_asset.sort_values('entry_time').reset_index(drop=True)
    n = len(trades)
    if n == 0:
        print(f"  {asset.upper()}: sem trades no periodo")
        return

    wins   = (trades['result'] == 'WIN').sum()
    losses = (trades['result'] == 'LOSS').sum()
    exps   = (trades['result'] == 'EXPIRED').sum()
    wr     = wins / n * 100
    avg_r  = trades['ret_r'].mean()
    date_start = pd.Timestamp(trades['entry_time'].min()).strftime('%Y-%m-%d')
    date_end   = pd.Timestamp(trades['exit_time'].max()).strftime('%Y-%m-%d')

    asset_upper = asset.upper()
    sign = '+' if avg_r >= 0 else ''
    wr_color  = '#2ecc71' if wr >= 50 else '#e74c3c'
    avg_color = '#2ecc71' if avg_r >= 0 else '#e74c3c'

    lines = []
    lines.append('//@version=5')
    lines.append(f'// Backtest ML Prop Firm — {asset_upper}')
    lines.append(f'// {n} trades — Ultimo mes ({date_start} a {date_end})')
    lines.append(f'// WR={wr:.1f}%  AvgR={sign}{avg_r:.2f}R  W/L/E={wins}/{losses}/{exps}')
    lines.append('')
    lines.append(f'strategy("[BT] {asset_upper} PropFirm", overlay=true, pyramiding=0, default_qty_type=strategy.fixed, default_qty_value=1, commission_type=strategy.commission.percent, commission_value=0)')
    lines.append('')

    for i, t in trades.iterrows():
        entry_ms = ts_ms(t['entry_time'])
        exit_ms  = ts_ms(t['exit_time'])
        is_long  = t['direction'] == 'LONG'
        result   = t['result']
        stop_px  = round(float(t['stop_price']), 4)
        tgt_px   = round(float(t['target_price']), 4)
        tid      = f"T{i+1}"
        xid      = f"X{i+1}"
        dir_str  = 'strategy.long' if is_long else 'strategy.short'
        lbl      = 'L' if is_long else 'S'

        lines.append(f'if time == {entry_ms}')
        lines.append(f'    strategy.entry("{tid}", {dir_str}, comment="{lbl}")')
        lines.append(f'    strategy.exit("{xid}", from_entry="{tid}", stop={stop_px}, limit={tgt_px})')
        if result == 'EXPIRED':
            lines.append(f'if time == {exit_ms}')
            lines.append(f'    strategy.close("{tid}", comment="EXP")')
        lines.append('')

    # Tabela de stats
    lines.append('// Tabela de estatisticas')
    lines.append('if barstate.islast')
    lines.append(f'    tbl = table.new(position.top_right, 2, 6, bgcolor=color.new(#0b0e14,80), border_width=1, border_color=color.new(#2a3342,0))')
    lines.append(f'    table.cell(tbl, 0, 0, "{asset_upper} PropFirm ML", text_color=color.white, text_size=size.small)')
    lines.append(f'    table.cell(tbl, 1, 0, "1 mes", text_color=color.gray, text_size=size.small)')
    lines.append(f'    table.cell(tbl, 0, 1, "Trades", text_color=color.gray, text_size=size.small)')
    lines.append(f'    table.cell(tbl, 1, 1, "{n}", text_color=color.white, text_size=size.small)')
    lines.append(f'    table.cell(tbl, 0, 2, "Win Rate", text_color=color.gray, text_size=size.small)')
    lines.append(f'    table.cell(tbl, 1, 2, "{wr:.1f}%", text_color=color.new({wr_color},0), text_size=size.small)')
    lines.append(f'    table.cell(tbl, 0, 3, "Avg R", text_color=color.gray, text_size=size.small)')
    lines.append(f'    table.cell(tbl, 1, 3, "{sign}{avg_r:.2f}R", text_color=color.new({avg_color},0), text_size=size.small)')
    lines.append(f'    table.cell(tbl, 0, 4, "W / L / E", text_color=color.gray, text_size=size.small)')
    lines.append(f'    table.cell(tbl, 1, 4, "{wins} / {losses} / {exps}", text_color=color.white, text_size=size.small)')

    code = '\n'.join(lines)
    out_path.write_text(code, encoding='utf-8')
    print(f"  {asset_upper}: {n} trades  WR={wr:.1f}%  AvgR={sign}{avg_r:.2f}R  -> {out_path.name}")


def main():
    csv_path = BASE / 'trades_multi_asset.csv'
    df = pd.read_csv(csv_path, parse_dates=['entry_time', 'exit_time'])

    # Filtro: ultimo 1 mes
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(days=30)
    df = df[df['entry_time'] >= cutoff].copy()

    if df.empty:
        print("Nenhum trade no ultimo mes.")
        return

    print(f"Periodo filtrado: {df['entry_time'].min().date()} a {df['exit_time'].max().date()}")
    print(f"Total trades: {len(df)}\n")

    for asset in ['mnq', 'btc', 'cl', 'mgc']:
        df_a = df[df['asset'] == asset].copy()
        out  = BASE / f'pine_{asset}.txt'
        gerar_pine(asset, df_a, out)

    print("\nPronto! Cole cada arquivo no TradingView (Ctrl+A, Delete, Ctrl+V).")


if __name__ == '__main__':
    main()

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


HIST_FILES = {
    'mnq': BASE.parent.parent.parent / 'histogramas' / 'mnq' / 'histograma - mnq.pine',
    'btc': BASE.parent.parent.parent / 'histogramas' / 'btc' / 'histograma - btc.pine',
    'cl':  BASE.parent.parent.parent / 'histogramas' / 'cl'  / 'histograma - cl.pine',
    'mgc': BASE.parent.parent.parent / 'histogramas' / 'mgc' / 'histograma - mgc.pine',
}

BACKTEST_DIR = BASE.parent.parent.parent / 'backtest'

ML_SIGNAL_MARKER = '// ── ML PropFirm Signals'


def gerar_bloco_sinais(asset, df_asset):
    """Gera bloco Pine Script com sinais ML para adicionar ao histograma."""
    trades = df_asset.sort_values('entry_time').reset_index(drop=True)
    if len(trades) == 0:
        return None

    lines = []
    lines.append('')
    lines.append(ML_SIGNAL_MARKER + ' ─────────────────────────────────────────')
    lines.append('show_ml = input.bool(true, "ML Signals", group="ML PropFirm")')
    lines.append('var bool ml_pos    = false')
    lines.append('var bool ml_islong = false')
    lines.append('')

    for i, t in trades.iterrows():
        entry_ms = ts_ms(t['entry_time'])
        exit_ms  = ts_ms(t['exit_time'])
        is_long  = t['direction'] == 'LONG'
        result   = t['result']

        lines.append(f'if time == {entry_ms}')
        lines.append(f'    ml_pos := true')
        lines.append(f'    ml_islong := {str(is_long).lower()}')
        lines.append(f'if time == {exit_ms}')
        lines.append(f'    ml_pos := false')
        lines.append('')

    # bgcolor durante posicao aberta
    lines.append('ml_bg = ml_pos ? (ml_islong ? color.new(#2ecc71, 88) : color.new(#e74c3c, 88)) : na')
    lines.append('bgcolor(show_ml ? ml_bg : na, title="ML Posicao")')
    lines.append('')

    # plotshapes de entrada e saida
    for i, t in trades.iterrows():
        entry_ms = ts_ms(t['entry_time'])
        exit_ms  = ts_ms(t['exit_time'])
        is_long  = t['direction'] == 'LONG'
        result   = t['result']

        if is_long:
            lines.append(f'plotshape(show_ml and time=={entry_ms}, "ML-L{i+1}", shape.triangleup, location.bottom, color.new(#2ecc71,0), size=size.small)')
        else:
            lines.append(f'plotshape(show_ml and time=={entry_ms}, "ML-S{i+1}", shape.triangledown, location.top, color.new(#e74c3c,0), size=size.small)')

        if result == 'WIN':
            loc = 'location.bottom' if is_long else 'location.top'
            lines.append(f'plotshape(show_ml and time=={exit_ms}, "ML-W{i+1}", shape.diamond, {loc}, color.new(#2ecc71,0), size=size.tiny)')
        elif result == 'LOSS':
            loc = 'location.top' if is_long else 'location.bottom'
            lines.append(f'plotshape(show_ml and time=={exit_ms}, "ML-X{i+1}", shape.xcross, {loc}, color.new(#e74c3c,0), size=size.tiny)')
        else:
            lines.append(f'plotshape(show_ml and time=={exit_ms}, "ML-E{i+1}", shape.cross, location.top, color.new(#f1c40f,0), size=size.tiny)')

    return '\n'.join(lines)


def injetar_sinais_histograma(asset, df_asset):
    hist_path = HIST_FILES.get(asset)
    if not hist_path or not hist_path.exists():
        print(f"  {asset.upper()}: histograma nao encontrado em {hist_path}")
        return

    bloco = gerar_bloco_sinais(asset, df_asset)
    if bloco is None:
        return

    codigo = hist_path.read_text(encoding='utf-8')

    # Remove bloco anterior se existir
    if ML_SIGNAL_MARKER in codigo:
        idx = codigo.index(ML_SIGNAL_MARKER)
        # volta ate a linha vazia antes do marcador
        codigo = codigo[:max(0, idx-1)].rstrip()

    codigo = codigo + '\n' + bloco + '\n'
    hist_path.write_text(codigo, encoding='utf-8')
    n = len(df_asset)
    print(f"  {asset.upper()}: {n} sinais ML injetados -> {hist_path.name}")


def gerar_overlay(asset, df_asset, out_path):
    """Indicador overlay=true: sinais ML nas velas do grafico de preco."""
    trades = df_asset.sort_values('entry_time').reset_index(drop=True)
    n = len(trades)
    if n == 0:
        print(f"  {asset.upper()}: sem trades")
        return

    wins   = (trades['result'] == 'WIN').sum()
    losses = (trades['result'] == 'LOSS').sum()
    exps   = (trades['result'] == 'EXPIRED').sum()
    wr     = wins / n * 100
    avg_r  = trades['ret_r'].mean()
    sign   = '+' if avg_r >= 0 else ''
    asset_upper = asset.upper()

    lines = []
    lines.append('//@version=5')
    lines.append(f'// ML PropFirm Signals — {asset_upper} (overlay)')
    lines.append(f'// {n} trades  WR={wr:.1f}%  AvgR={sign}{avg_r:.2f}R')
    lines.append('')
    lines.append(f'indicator("[ML] {asset_upper} Signals", overlay=true, max_labels_count=500)')
    lines.append('')
    lines.append('// Rastreia posicao ativa para bgcolor no painel de preco')
    lines.append('var bool ml_pos    = false')
    lines.append('var bool ml_islong = false')
    lines.append('')

    for i, t in trades.iterrows():
        entry_ms = ts_ms(t['entry_time'])
        exit_ms  = ts_ms(t['exit_time'])
        is_long  = t['direction'] == 'LONG'
        result   = t['result']
        stop_px  = round(float(t['stop_price']), 4)
        tgt_px   = round(float(t['target_price']), 4)

        lines.append(f'if time == {entry_ms}')
        lines.append(f'    ml_pos := true')
        lines.append(f'    ml_islong := {str(is_long).lower()}')
        lines.append(f'if time == {exit_ms}')
        lines.append(f'    ml_pos := false')
        lines.append('')

    # bgcolor enquanto posicao ativa (muito leve, so hint)
    lines.append('ml_bg = ml_pos ? (ml_islong ? color.new(#2ecc71, 93) : color.new(#e74c3c, 93)) : na')
    lines.append('bgcolor(ml_bg, title="ML Posicao")')
    lines.append('')

    # Linhas de stop e target durante posicao ativa
    lines.append('var float ml_stop = na')
    lines.append('var float ml_tgt  = na')
    lines.append('')

    for i, t in trades.iterrows():
        entry_ms = ts_ms(t['entry_time'])
        exit_ms  = ts_ms(t['exit_time'])
        stop_px  = round(float(t['stop_price']), 4)
        tgt_px   = round(float(t['target_price']), 4)
        lines.append(f'if time == {entry_ms}')
        lines.append(f'    ml_stop := {stop_px}')
        lines.append(f'    ml_tgt  := {tgt_px}')
        lines.append(f'if time == {exit_ms}')
        lines.append(f'    ml_stop := na')
        lines.append(f'    ml_tgt  := na')
        lines.append('')

    lines.append('plot(ml_pos ? ml_stop : na, "ML Stop",   color=color.new(#e74c3c, 20), style=plot.style_linebr, linewidth=1)')
    lines.append('plot(ml_pos ? ml_tgt  : na, "ML Target", color=color.new(#2ecc71, 20), style=plot.style_linebr, linewidth=1)')
    lines.append('')

    # Plotshapes nas velas
    for i, t in trades.iterrows():
        entry_ms = ts_ms(t['entry_time'])
        exit_ms  = ts_ms(t['exit_time'])
        is_long  = t['direction'] == 'LONG'
        result   = t['result']

        if is_long:
            lines.append(f'plotshape(time=={entry_ms}?close:na, "ML-L{i+1}", shape.triangleup, location.belowbar, color.new(#2ecc71,0), size=size.normal, text="ML")')
        else:
            lines.append(f'plotshape(time=={entry_ms}?close:na, "ML-S{i+1}", shape.triangledown, location.abovebar, color.new(#e74c3c,0), size=size.normal, text="ML")')

        if result == 'WIN':
            loc = 'location.belowbar' if is_long else 'location.abovebar'
            lines.append(f'plotshape(time=={exit_ms}?close:na, "ML-W{i+1}", shape.diamond, {loc}, color.new(#2ecc71,0), size=size.small)')
        elif result == 'LOSS':
            loc = 'location.abovebar' if is_long else 'location.belowbar'
            lines.append(f'plotshape(time=={exit_ms}?close:na, "ML-X{i+1}", shape.xcross, {loc}, color.new(#e74c3c,0), size=size.small)')
        else:
            lines.append(f'plotshape(time=={exit_ms}?close:na, "ML-E{i+1}", shape.cross, location.abovebar, color.new(#f1c40f,0), size=size.small)')

    code = '\n'.join(lines)
    out_path.write_text(code, encoding='utf-8')
    print(f"  {asset_upper}: {n} trades -> {out_path.name}")


def main():
    csv_path = BASE / 'trades_multi_asset.csv'
    df = pd.read_csv(csv_path, parse_dates=['entry_time', 'exit_time'])

    # Filtro: ultimo 1 mes
    cutoff = pd.Timestamp.now() - timedelta(days=30)
    df = df[df['entry_time'] >= cutoff].copy()

    if df.empty:
        print("Nenhum trade no ultimo mes.")
        return

    print(f"Periodo filtrado: {df['entry_time'].min().date()} a {df['exit_time'].max().date()}")
    print(f"Total trades: {len(df)}\n")

    print("-- Strategy files --")
    for asset in ['mnq', 'btc', 'cl', 'mgc']:
        df_a = df[df['asset'] == asset].copy()
        out  = BACKTEST_DIR / asset / 'strategy.pine'
        gerar_pine(asset, df_a, out)

    print("\n-- Histogramas (sinais ML) --")
    for asset in ['mnq', 'btc', 'cl', 'mgc']:
        df_a = df[df['asset'] == asset].copy()
        injetar_sinais_histograma(asset, df_a)

    print("\n-- Overlay (sinais nas velas) --")
    for asset in ['mnq', 'btc', 'cl', 'mgc']:
        df_a = df[df['asset'] == asset].copy()
        out  = BACKTEST_DIR / asset / 'overlay.pine'
        gerar_overlay(asset, df_a, out)

    print("\nPronto!")


if __name__ == '__main__':
    main()

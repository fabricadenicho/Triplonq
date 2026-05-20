"""
Refatora os histogramas Pine Script para alinhar com a arvore ML PropFirm.

Adiciona por ativo:
  - Monthly Open (mo) e Weekly Open (wo) — KEY LEVELS top feature (33-37%)
  - Intraday high/low tracking (id_h, id_l)
  - BB position (bb_pos) onde necessario
  - ret4 (retorno 4 barras) para BTC
  - ML Score baseado nas condicoes reais da arvore de decisao
  - plotshape e bgcolor para sinal ML no pane do histograma
  - Painel atualizado mostrando ML Score

Preserva toda a logica existente, apenas adiciona/atualiza sinais.
"""
from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent  # Triplonq root
ML_MARKER = '// -- ML PropFirm Signals'
ML_MARKER2 = '// ── ML PropFirm Signals'

def limpar_bloco_ml(text):
    for marker in [ML_MARKER, ML_MARKER2]:
        if marker in text:
            idx = text.index(marker)
            text = text[:max(0, idx - 1)].rstrip()
            break
    return text


# ─────────────────────────────────────────────────────────────────────────────
# CL — LONG only | KEY LEVELS 37% | VOL 9% | TEMPORAL 13%
# Arvore: hour<14, vol_p<0.004, prev_day_range<0.54, dist_to_mo, dist_to_pdl
# ─────────────────────────────────────────────────────────────────────────────
def refatorar_cl():
    path = BASE / 'histograma - cl.pine'
    text = limpar_bloco_ml(path.read_text(encoding='utf-8'))

    # 1. Adiciona mo, wo, id_h/l apos KEY LEVELS
    kl_anchor = 'pml = request.security(syminfo.tickerid, "M", low[1],  lookahead=barmerge.lookahead_on)'
    kl_add = '''
mo  = request.security(syminfo.tickerid, "M", open, lookahead=barmerge.lookahead_on)
wo  = request.security(syminfo.tickerid, "W", open, lookahead=barmerge.lookahead_on)
var float id_h = high
var float id_l = low
id_h := ta.change(time("D")) != 0 ? high : math.max(id_h, high)
id_l := ta.change(time("D")) != 0 ? low  : math.min(id_l, low)'''
    text = text.replace(kl_anchor, kl_anchor + '\n' + kl_add)

    # 2. Adiciona dist_to_mo, id_h/l, bb_pos apos strong_div
    sd_anchor = 'strong_div     = diverging and adx_val > adx_thr'
    sd_add = '''
// KEY LEVELS ML: dist_to_mo, intraday H/L, bb_pos
dist_to_mo_v  = mo  != 0 ? (close - mo)  / mo  * 100 : 0.0
dist_to_wo_v  = wo  != 0 ? (close - wo)  / wo  * 100 : 0.0
dist_to_idh_v = id_h != 0 ? (close - id_h) / id_h * 100 : 0.0
dist_to_idl_v = id_l != 0 ? (close - id_l) / id_l * 100 : 0.0
bb_pos_v      = bb_up != bb_dn ? (close - bb_dn) / (bb_up - bb_dn) : 0.5'''
    text = text.replace(sd_anchor, sd_anchor + '\n' + sd_add)

    # 3. Insere ML Score antes de // HISTOGRAMA
    ml_block = '''// ML SCORE CL — hora(2)+vol(2)+prev_rng(2)+dist_mo(1)+dist_pdl(1)+DI(1) = max 9
ml_vol_ok  = vol_cl < 0.402
ml_hour_ok = h < 14
ml_rng_ok  = prev_day_range_pct < 0.54
ml_mo_ok   = math.abs(dist_to_mo_v) < 3.0
ml_pdl_ok  = dist_pdl_v > -0.5 and dist_pdl_v < 1.5
ml_di_ok   = di_plus > di_minus
ml_score   = (ml_hour_ok ? 2 : 0) + (ml_vol_ok ? 2 : 0) + (ml_rng_ok ? 2 : 0) + (ml_mo_ok ? 1 : 0) + (ml_pdl_ok ? 1 : 0) + (ml_di_ok ? 1 : 0)
ml_signal  = ml_score >= 6

'''
    text = text.replace('// HISTOGRAMA\n', ml_block + '// HISTOGRAMA\n', 1)

    # 4. Adiciona visualizacao ML antes de // PAINEL
    ml_viz = '''plotshape(ml_signal and not ml_signal[1], "ML Long CL", shape.labelup, location.bottom, color.new(#62b0ff, 0), size=size.small, text="ML")
bgcolor(ml_signal ? color.new(#2ecc71, 89) : ml_score >= 4 ? color.new(#ffd166, 95) : na, title="ML Score BG")

'''
    text = text.replace('// PAINEL\n', ml_viz + '// PAINEL\n', 1)

    # 5. Atualiza dec_str/dec_col
    text = text.replace(
        'sig_lng = cl_down_mnq_up and in_pf\n    dec_str = sig_lng and strong_div ? "COMPRAR CL ++" : sig_lng ? "COMPRAR CL" : cl_down_mnq_up ? "cl_dn_mnq_up" : "AGUARDAR"\n    dec_col = sig_lng and strong_div ? C_GRN : sig_lng ? C_YEL : cl_down_mnq_up ? color.new(col_signal, 30) : C_MUT',
        'sig_lng = cl_down_mnq_up and in_pf\n    dec_str = ml_signal ? "ML COMPRAR CL" : ml_score >= 4 ? "parcial " + str.tostring(ml_score) + "/9" : cl_down_mnq_up ? "div CL-MNQ" : "AGUARDAR"\n    dec_col = ml_signal ? C_GRN : ml_score >= 4 ? C_YEL : cl_down_mnq_up ? color.new(col_signal, 30) : C_MUT'
    )

    # 6. Expande tabela +1 row e adiciona linha ML Score
    text = text.replace(
        'table.new(position.top_right, 2, 20,',
        'table.new(position.top_right, 2, 21,'
    )
    old_auc = '    table.cell(t, 0, 19, "AUC 0.474 | Acc 52%"'
    new_rows = (
        '    ml_sc_s = str.tostring(ml_score) + "/9 " + (ml_signal ? "COMPRAR" : ml_score >= 4 ? "parcial" : "aguardar")\n'
        '    ml_sc_c = ml_signal ? C_GRN : ml_score >= 4 ? C_YEL : C_MUT\n'
        '    table.cell(t, 0, 19, "ML SCORE", text_color=C_MUT, bgcolor=C_LINE, text_size=size.tiny)\n'
        '    table.cell(t, 1, 19, ml_sc_s, text_color=ml_sc_c, bgcolor=C_LINE, text_size=size.small, text_halign=text.align_right)\n'
        '    table.cell(t, 0, 20, "AUC 0.474 | Acc 52%"'
    )
    text = text.replace(old_auc, new_rows)
    text = text.replace(
        'table.cell(t, 1, 19, str.tostring(close, "#.##"), text_color=C_WHT, bgcolor=C_BG2, text_size=size.small, text_halign=text.align_right)',
        'table.cell(t, 1, 20, str.tostring(close, "#.##"), text_color=C_WHT, bgcolor=C_BG2, text_size=size.small, text_halign=text.align_right)'
    )

    path.write_text(text, encoding='utf-8')
    print("  CL: ok")


# ─────────────────────────────────────────────────────────────────────────────
# BTC — SHORT only | KEY LEVELS 32% | TEMPORAL 14% | RETORNOS 19%
# Arvore: dow_sin, ret4, rsi, bb_pos, adx<28.5, dist_to_mday_h, di_spread
# ─────────────────────────────────────────────────────────────────────────────
def refatorar_btc():
    path = BASE / 'histograma - btc.pine'
    text = limpar_bloco_ml(path.read_text(encoding='utf-8'))

    # 1. Adiciona KEY LEVELS apos dados externos
    kl_anchor = 'cl_above50  = request.security(sym_cl,  timeframe.period, close > ta.sma(close, 50) ? 1 : 0)'
    kl_add = '''
// KEY LEVELS
pdh = request.security(syminfo.tickerid, "D", high[1], lookahead=barmerge.lookahead_on)
pdl = request.security(syminfo.tickerid, "D", low[1],  lookahead=barmerge.lookahead_on)
mo  = request.security(syminfo.tickerid, "M", open,    lookahead=barmerge.lookahead_on)
wo  = request.security(syminfo.tickerid, "W", open,    lookahead=barmerge.lookahead_on)
var float id_h = high
var float id_l = low
id_h := ta.change(time("D")) != 0 ? high : math.max(id_h, high)
id_l := ta.change(time("D")) != 0 ? low  : math.min(id_l, low)'''
    text = text.replace(kl_anchor, kl_anchor + '\n' + kl_add)

    # 2. Adiciona ret4, bb_pos, intraday dist apos strong_div_btc
    sd_anchor = 'strong_div_btc = diverging and adx_val > adx_thr'
    sd_add = '''
// ML features BTC: ret4, bb_pos, intraday H/L, dist_to_mo
ret4_btc      = close[4] != 0 ? (close - close[4]) / close[4] : 0.0
bb_pos_btc    = bb_up != bb_dn ? (close - bb_dn) / (bb_up - bb_dn) : 0.5
dist_to_mo_v  = mo  != 0 ? (close - mo)  / mo  * 100 : 0.0
dist_to_wo_v  = wo  != 0 ? (close - wo)  / wo  * 100 : 0.0
dist_to_idh_v = id_h != 0 ? (close - id_h) / id_h * 100 : 0.0
dist_to_idl_v = id_l != 0 ? (close - id_l) / id_l * 100 : 0.0
prev_day_range_pct = pdl != 0 ? (pdh - pdl) / pdl * 100 : 0.0'''
    text = text.replace(sd_anchor, sd_anchor + '\n' + sd_add)

    # 3. ML Score BTC (SHORT) antes de // HISTOGRAMA
    ml_block = '''// ML SCORE BTC SHORT — rsi(2)+ret4(2)+DI(2)+adx(1)+bb(1)+vol(1) = max 9
ml_rsi_ok  = rsi_val >= 55
ml_ret4_ok = ret4_btc < -0.005
ml_adx_ok  = adx_val >= adx_thr
ml_di_ok   = di_minus > di_plus
ml_bb_ok   = bb_pos_btc > 0.7
ml_vol_ok  = vol_btc < 0.4
ml_score   = (ml_rsi_ok ? 2 : 0) + (ml_ret4_ok ? 2 : 0) + (ml_di_ok ? 2 : 0) + (ml_adx_ok ? 1 : 0) + (ml_bb_ok ? 1 : 0) + (ml_vol_ok ? 1 : 0)
ml_signal  = ml_score >= 6

'''
    text = text.replace('// HISTOGRAMA\n', ml_block + '// HISTOGRAMA\n', 1)

    # 4. Vizualizacao ML
    ml_viz = '''plotshape(ml_signal and not ml_signal[1], "ML Short BTC", shape.labeldown, location.top, color.new(#e74c3c, 0), size=size.small, text="ML")
bgcolor(ml_signal ? color.new(#e74c3c, 89) : ml_score >= 4 ? color.new(#ffd166, 95) : na, title="ML Score BG")

'''
    text = text.replace('// PAINEL\n', ml_viz + '// PAINEL\n', 1)

    # 5. Atualiza dec_str/dec_col
    text = text.replace(
        'sig_sht = strong_div_btc and in_pf\n    dec_str = sig_sht ? "VENDER BTC" : strong_div_btc ? "div SHORT" : sma50_align_v == 0 ? "BEARISH" : sma50_align_v == 2 ? "BULLISH" : "AGUARDAR"\n    dec_col = sig_sht ? C_RED : strong_div_btc ? C_YEL : sma50_align_v == 0 ? C_RED : sma50_align_v == 2 ? C_GRN : C_MUT',
        'sig_sht = strong_div_btc and in_pf\n    dec_str = ml_signal ? "ML VENDER BTC" : ml_score >= 4 ? "parcial " + str.tostring(ml_score) + "/9" : strong_div_btc ? "div SHORT" : "AGUARDAR"\n    dec_col = ml_signal ? C_RED : ml_score >= 4 ? C_YEL : strong_div_btc ? C_YEL : C_MUT'
    )

    # 6. Expande tabela +1 e adiciona row ML Score
    text = text.replace(
        'table.new(position.top_right, 2, 16,',
        'table.new(position.top_right, 2, 17,'
    )
    old_auc = '    table.cell(t, 0, 15, "AUC 0.555 | Acc 39%"'
    new_rows = (
        '    ml_sc_s = str.tostring(ml_score) + "/9 " + (ml_signal ? "VENDER" : ml_score >= 4 ? "parcial" : "aguardar")\n'
        '    ml_sc_c = ml_signal ? C_RED : ml_score >= 4 ? C_YEL : C_MUT\n'
        '    table.cell(t, 0, 15, "ML SCORE", text_color=C_MUT, bgcolor=C_LN, text_size=size.tiny)\n'
        '    table.cell(t, 1, 15, ml_sc_s, text_color=ml_sc_c, bgcolor=C_LN, text_size=size.small, text_halign=text.align_right)\n'
        '    table.cell(t, 0, 16, "AUC 0.555 | Acc 39%"'
    )
    text = text.replace(old_auc, new_rows)
    text = text.replace(
        'table.cell(t, 1, 15, str.tostring(close, "#.##"), text_color=C_WHT, bgcolor=C_BG2, text_size=size.small, text_halign=text.align_right)',
        'table.cell(t, 1, 16, str.tostring(close, "#.##"), text_color=C_WHT, bgcolor=C_BG2, text_size=size.small, text_halign=text.align_right)'
    )

    path.write_text(text, encoding='utf-8')
    print("  BTC: ok")


# ─────────────────────────────────────────────────────────────────────────────
# MNQ — BOTH | KEY LEVELS 34% | VOL 11% | TEMPORAL 11%
# Arvore: vol_p<0.334, sma50_alignment, div_cl, hour, di_spread
# ─────────────────────────────────────────────────────────────────────────────
def refatorar_mnq():
    path = BASE / 'histograma - mnq.pine'
    text = limpar_bloco_ml(path.read_text(encoding='utf-8'))

    # 1. Adiciona mo, wo, id_h/l apos KEY LEVELS
    kl_anchor = 'pml = request.security(syminfo.tickerid, "M", low[1],  lookahead=barmerge.lookahead_on)'
    kl_add = '''
mo  = request.security(syminfo.tickerid, "M", open, lookahead=barmerge.lookahead_on)
wo  = request.security(syminfo.tickerid, "W", open, lookahead=barmerge.lookahead_on)
var float id_h = high
var float id_l = low
id_h := ta.change(time("D")) != 0 ? high : math.max(id_h, high)
id_l := ta.change(time("D")) != 0 ? low  : math.min(id_l, low)'''
    text = text.replace(kl_anchor, kl_anchor + '\n' + kl_add)

    # 2. Adiciona dist_to_mo apos us_prime
    us_anchor = 'us_prime   = strong_div and in_us'
    us_add = '''
// KEY LEVELS ML: dist_to_mo, intraday H/L
dist_to_mo_v  = mo  != 0 ? (close - mo)  / mo  * 100 : 0.0
dist_to_wo_v  = wo  != 0 ? (close - wo)  / wo  * 100 : 0.0
dist_to_idh_v = id_h != 0 ? (close - id_h) / id_h * 100 : 0.0
dist_to_idl_v = id_l != 0 ? (close - id_l) / id_l * 100 : 0.0'''
    text = text.replace(us_anchor, us_anchor + '\n' + us_add)

    # 3. ML Score MNQ (BOTH) antes de // HISTOGRAMA
    ml_block = '''// ML SCORE MNQ — vol(2)+sma50/DI(2)+div(1)+hora(1) = max 6 por direcao
ml_vol_ok     = vol_mnq < 0.334
ml_long_ok    = sma50_align >= 2 and di_spread_val > 0
ml_short_ok   = sma50_align == 0 and di_spread_val < 0
ml_div_ok     = math.abs(price_div_cl) * 10000 > 2.0
ml_hour_ok    = in_us
ml_l_score    = (ml_vol_ok ? 2 : 0) + (ml_long_ok ? 2 : 0) + (ml_div_ok ? 1 : 0) + (ml_hour_ok ? 1 : 0)
ml_s_score    = (ml_vol_ok ? 2 : 0) + (ml_short_ok ? 2 : 0) + (ml_div_ok ? 1 : 0) + (ml_hour_ok ? 1 : 0)
ml_score      = math.max(ml_l_score, ml_s_score)
ml_is_long    = ml_l_score >= ml_s_score
ml_signal     = ml_score >= 4

'''
    text = text.replace('// HISTOGRAMA\n', ml_block + '// HISTOGRAMA\n', 1)

    # 4. Visualizacao ML
    ml_viz = '''plotshape(ml_signal and ml_is_long and not (ml_signal and ml_is_long)[1], "ML Long MNQ", shape.labelup, location.bottom, color.new(#2ecc71, 0), size=size.small, text="ML")
plotshape(ml_signal and not ml_is_long and not (ml_signal and not ml_is_long)[1], "ML Short MNQ", shape.labeldown, location.top, color.new(#e74c3c, 0), size=size.small, text="ML")
bgcolor(ml_signal and ml_is_long ? color.new(#2ecc71, 89) : ml_signal and not ml_is_long ? color.new(#e74c3c, 89) : ml_score >= 3 ? color.new(#ffd166, 95) : na, title="ML Score BG")

'''
    text = text.replace('// PAINEL\n', ml_viz + '// PAINEL\n', 1)

    # 5. Atualiza dec_str/dec_col
    text = text.replace(
        'dec_str = sig_lng ? "COMPRAR MNQ" : sig_sht ? "VENDER MNQ" : strong_div ? "strong_div" : "AGUARDAR"\n    dec_col = sig_lng ? C_GRN : sig_sht ? C_RED : strong_div ? C_YEL : C_MUT',
        'dec_str = ml_signal and ml_is_long ? "ML COMPRAR MNQ" : ml_signal and not ml_is_long ? "ML VENDER MNQ" : ml_score >= 3 ? "parcial " + str.tostring(ml_score) + "/6" : strong_div ? "div ativo" : "AGUARDAR"\n    dec_col = ml_signal and ml_is_long ? C_GRN : ml_signal and not ml_is_long ? C_RED : ml_score >= 3 ? C_YEL : strong_div ? color.new(col_signal, 30) : C_MUT'
    )

    # 6. Expande tabela +1 e adiciona row ML Score
    text = text.replace(
        'table.new(position.top_right, 2, 18,',
        'table.new(position.top_right, 2, 19,'
    )
    old_auc = '    table.cell(t, 0, 17, "AUC 0.602 | Acc 43%"'
    new_rows = (
        '    ml_dir = ml_is_long ? "LONG" : "SHORT"\n'
        '    ml_sc_s = str.tostring(ml_score) + "/6 " + (ml_signal ? ml_dir : ml_score >= 3 ? "parcial" : "aguardar")\n'
        '    ml_sc_c = ml_signal and ml_is_long ? C_GRN : ml_signal ? C_RED : ml_score >= 3 ? C_YEL : C_MUT\n'
        '    table.cell(t, 0, 17, "ML SCORE", text_color=C_MUT, bgcolor=C_LINE, text_size=size.tiny)\n'
        '    table.cell(t, 1, 17, ml_sc_s, text_color=ml_sc_c, bgcolor=C_LINE, text_size=size.small, text_halign=text.align_right)\n'
        '    table.cell(t, 0, 18, "AUC 0.602 | Acc 43%"'
    )
    text = text.replace(old_auc, new_rows)
    text = text.replace(
        'table.cell(t, 1, 17, str.tostring(close, "#.##"), text_color=C_WHT, bgcolor=C_BG2, text_size=size.small, text_halign=text.align_right)',
        'table.cell(t, 1, 18, str.tostring(close, "#.##"), text_color=C_WHT, bgcolor=C_BG2, text_size=size.small, text_halign=text.align_right)'
    )

    path.write_text(text, encoding='utf-8')
    print("  MNQ: ok")


# ─────────────────────────────────────────────────────────────────────────────
# MGC — BOTH | KEY LEVELS 37% | VOL 11% | TEMPORAL 10%
# Arvore: vol_p<0.256, dist_to_mo<0.49, prev_rng<0.977, di_spread, rsi, div
# ─────────────────────────────────────────────────────────────────────────────
def refatorar_mgc():
    path = BASE / 'histograma - mgc.pine'
    text = limpar_bloco_ml(path.read_text(encoding='utf-8'))

    # 1. Adiciona mo, wo, id_h/l apos KEY LEVELS (apos pd_low)
    kl_anchor = 'pd_low  = request.security(syminfo.tickerid, "D", low[1],  lookahead=barmerge.lookahead_on)'
    kl_add = '''
mo  = request.security(syminfo.tickerid, "M", open, lookahead=barmerge.lookahead_on)
wo  = request.security(syminfo.tickerid, "W", open, lookahead=barmerge.lookahead_on)
var float id_h = high
var float id_l = low
id_h := ta.change(time("D")) != 0 ? high : math.max(id_h, high)
id_l := ta.change(time("D")) != 0 ? low  : math.min(id_l, low)'''
    text = text.replace(kl_anchor, kl_anchor + '\n' + kl_add)

    # 2. Adiciona dist_to_mo, id_h/l apos prime_setup
    ps_anchor = 'prime_setup = above_pmh and strong_adx and in_us_pm'
    ps_add = '''
// KEY LEVELS ML: dist_to_mo (peso FORTE — arvore: dist_to_mo < 0.49!)
dist_to_mo_v  = mo  != 0 ? (close - mo)  / mo  * 100 : 0.0
dist_to_wo_v  = wo  != 0 ? (close - wo)  / wo  * 100 : 0.0
dist_to_idh_v = id_h != 0 ? (close - id_h) / id_h * 100 : 0.0
dist_to_idl_v = id_l != 0 ? (close - id_l) / id_l * 100 : 0.0'''
    text = text.replace(ps_anchor, ps_anchor + '\n' + ps_add)

    # 3. ML Score MGC (BOTH) antes de // HISTOGRAMA
    ml_block = '''// ML SCORE MGC — vol(2)+dist_mo(3)+prev_rng(1)+rsi(1)+DI(1) = max 8
ml_vol_ok     = vol_mgc < 0.256
ml_mo_ok      = math.abs(dist_to_mo_v) < 0.5
ml_rng_ok     = prev_day_range_v < 0.977
ml_rsi_long   = rsi_val < 50
ml_rsi_short  = rsi_val >= 50
ml_di_long    = di_plus > di_minus
ml_di_short   = di_minus > di_plus
ml_l_score    = (ml_vol_ok ? 2 : 0) + (ml_mo_ok ? 3 : 0) + (ml_rng_ok ? 1 : 0) + (ml_rsi_long ? 1 : 0) + (ml_di_long ? 1 : 0)
ml_s_score    = (ml_vol_ok ? 2 : 0) + (ml_mo_ok ? 3 : 0) + (ml_rng_ok ? 1 : 0) + (ml_rsi_short ? 1 : 0) + (ml_di_short ? 1 : 0)
ml_score      = math.max(ml_l_score, ml_s_score)
ml_is_long    = ml_l_score >= ml_s_score
ml_signal     = ml_score >= 5

'''
    text = text.replace('// HISTOGRAMA\n', ml_block + '// HISTOGRAMA\n', 1)

    # 4. Visualizacao ML
    ml_viz = '''plotshape(ml_signal and ml_is_long and not (ml_signal and ml_is_long)[1], "ML Long MGC", shape.labelup, location.bottom, color.new(#2ecc71, 0), size=size.small, text="ML")
plotshape(ml_signal and not ml_is_long and not (ml_signal and not ml_is_long)[1], "ML Short MGC", shape.labeldown, location.top, color.new(#e74c3c, 0), size=size.small, text="ML")
bgcolor(ml_signal and ml_is_long ? color.new(#2ecc71, 89) : ml_signal and not ml_is_long ? color.new(#e74c3c, 89) : ml_score >= 4 ? color.new(#ffd166, 95) : na, title="ML Score BG")

'''
    text = text.replace('// PAINEL\n', ml_viz + '// PAINEL\n', 1)

    # 5. Atualiza dec_str/dec_col
    text = text.replace(
        'dec_str = prime_setup ? "COMPRAR PRIME" : sig_lng ? "COMPRAR MGC" : sig_sht ? "VENDER MGC" : above_pmh ? "above PMH" : near_pml ? "near PML" : "AGUARDAR"\n    dec_col = prime_setup ? C_GRN : sig_lng ? C_YEL : sig_sht ? C_RED : above_pmh ? C_YEL : near_pml ? C_PRP : C_MUT',
        'dec_str = ml_signal and ml_is_long ? "ML COMPRAR MGC" : ml_signal and not ml_is_long ? "ML VENDER MGC" : ml_score >= 4 ? "parcial " + str.tostring(ml_score) + "/8" : prime_setup ? "PRIME" : above_pmh ? "above PMH" : "AGUARDAR"\n    dec_col = ml_signal and ml_is_long ? C_GRN : ml_signal and not ml_is_long ? C_RED : ml_score >= 4 ? C_YEL : prime_setup ? C_GRN : above_pmh ? C_YEL : C_MUT'
    )

    # 6. Adiciona dist_to_mo no painel (substitui linha BB% CL)
    text = text.replace(
        '    table.cell(t, 0, 13, "BB% CL", text_color=C_MUT, bgcolor=C_BG, text_size=size.tiny)\n    table.cell(t, 1, 13, str.tostring(cl_bb_w, "#.##") + "%", text_color=cl_bb_w > 5 ? C_RED : cl_bb_w > 3 ? C_YEL : C_GRN, bgcolor=C_BG, text_size=size.small, text_halign=text.align_right)',
        '    mo_s = (dist_to_mo_v >= 0 ? "+" : "") + str.tostring(dist_to_mo_v, "#.##") + "% " + (ml_mo_ok ? "KL!" : "")\n    mo_c = ml_mo_ok ? C_GRN : math.abs(dist_to_mo_v) < 2 ? C_YEL : C_MUT\n    table.cell(t, 0, 13, "DIST MO (KL)", text_color=C_MUT, bgcolor=C_BG, text_size=size.tiny)\n    table.cell(t, 1, 13, mo_s, text_color=mo_c, bgcolor=C_BG, text_size=size.small, text_halign=text.align_right)'
    )

    # 7. Expande tabela +1 e adiciona row ML Score
    text = text.replace(
        'table.new(position.top_right, 2, 19,',
        'table.new(position.top_right, 2, 20,'
    )
    old_auc = '    table.cell(t, 0, 18, "AUC 0.529 | US_tarde *1"'
    new_rows = (
        '    ml_dir = ml_is_long ? "LONG" : "SHORT"\n'
        '    ml_sc_s = str.tostring(ml_score) + "/8 " + (ml_signal ? ml_dir : ml_score >= 4 ? "parcial" : "aguardar")\n'
        '    ml_sc_c = ml_signal and ml_is_long ? C_GRN : ml_signal ? C_RED : ml_score >= 4 ? C_YEL : C_MUT\n'
        '    table.cell(t, 0, 18, "ML SCORE", text_color=C_MUT, bgcolor=C_LINE, text_size=size.tiny)\n'
        '    table.cell(t, 1, 18, ml_sc_s, text_color=ml_sc_c, bgcolor=C_LINE, text_size=size.small, text_halign=text.align_right)\n'
        '    table.cell(t, 0, 19, "AUC 0.529 | US_tarde *1"'
    )
    text = text.replace(old_auc, new_rows)
    text = text.replace(
        'table.cell(t, 1, 18, str.tostring(close, "#.##"), text_color=C_WHT, bgcolor=C_BG2, text_size=size.small, text_halign=text.align_right)',
        'table.cell(t, 1, 19, str.tostring(close, "#.##"), text_color=C_WHT, bgcolor=C_BG2, text_size=size.small, text_halign=text.align_right)'
    )

    path.write_text(text, encoding='utf-8')
    print("  MGC: ok")


if __name__ == '__main__':
    print("Refatorando histogramas...")
    refatorar_cl()
    refatorar_btc()
    refatorar_mnq()
    refatorar_mgc()
    print("\nReinjetando sinais ML...")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'gerar_pine.py')],
        capture_output=True, text=True
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print("ERRO:", result.stderr[-300:])

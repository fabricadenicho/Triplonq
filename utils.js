function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function getRsiBias(rsi) {
  if (rsi === null) return 'sem dados';
  if (rsi >= 70) return 'esticado para cima';
  if (rsi <= 30) return 'esticado para baixo';
  if (rsi >= 60) return 'forca compradora';
  if (rsi <= 40) return 'pressao vendedora';
  return 'zona neutra';
}

function getLsrBias(ratio) {
  if (ratio === null) return 'sem dados';
  if (ratio >= 1.8) return 'mercado bem inclinado para long';
  if (ratio >= 1.2) return 'leve inclinacao para long';
  if (ratio <= 0.7) return 'mercado bem inclinado para short';
  if (ratio <= 0.9) return 'leve inclinacao para short';
  return 'equilibrado';
}

function getOiBias(deltaPct) {
  if (!Number.isFinite(deltaPct)) return 'sem dados';
  if (deltaPct >= 3) return 'entrada forte de posicao';
  if (deltaPct > 0.5) return 'oi em expansao';
  if (deltaPct <= -3) return 'saida forte de posicao';
  if (deltaPct < -0.5) return 'oi em contracao';
  return 'oi estavel';
}

function getTakerBias(ratio) {
  if (!Number.isFinite(ratio)) return 'sem dados';
  if (ratio >= 1.35) return 'agressao compradora';
  if (ratio <= 0.75) return 'agressao vendedora';
  return 'fluxo equilibrado';
}

function getCvdBias(cvd) {
  const deltaPct = cvd?.deltaPct;
  if (!Number.isFinite(deltaPct)) return 'sem dados';
  if (deltaPct >= 15) return 'agressao compradora muito forte';
  if (deltaPct >= 5) return 'compradores dominando o candle';
  if (deltaPct <= -15) return 'agressao vendedora muito forte';
  if (deltaPct <= -5) return 'vendedores dominando o candle';
  return 'fluxo equilibrado no candle';
}

function buildLevelsFromKlines(monthly, weekly, daily) {
  const pm = monthly.length >= 2 ? monthly[monthly.length - 2] : monthly[0];
  const cm = monthly[monthly.length - 1];
  const monthlyPrevHigh = parseFloat(pm[2]);
  const monthlyPrevLow  = parseFloat(pm[3]);
  const monthlyOpen     = parseFloat(cm[1]);
  const monthlyMid      = (monthlyPrevHigh + monthlyPrevLow) / 2;

  const pw = weekly.length >= 2 ? weekly[weekly.length - 2] : weekly[0];
  const cw = weekly[weekly.length - 1];
  const weeklyPrevHigh = parseFloat(pw[2]);
  const weeklyPrevLow  = parseFloat(pw[3]);
  const weeklyOpen     = parseFloat(cw[1]);
  const weeklyMid      = (weeklyPrevHigh + weeklyPrevLow) / 2;

  const pd = daily[daily.length - 2];
  const cd = daily[daily.length - 1];
  const dailyPrevHigh = parseFloat(pd[2]);
  const dailyPrevLow  = parseFloat(pd[3]);
  const dailyOpen     = parseFloat(cd[1]);

  let mondayHigh = null, mondayLow = null;
  for (let i = daily.length - 1; i >= 0; i--) {
    if (new Date(daily[i][0]).getUTCDay() === 1) {
      mondayHigh = parseFloat(daily[i][2]);
      mondayLow  = parseFloat(daily[i][3]);
      break;
    }
  }
  const mondayMid = mondayHigh && mondayLow ? (mondayHigh + mondayLow) / 2 : null;

  return {
    monthlyPrevHigh, monthlyPrevLow, monthlyOpen, monthlyMid,
    weeklyPrevHigh,  weeklyPrevLow,  weeklyOpen,  weeklyMid,
    dailyPrevHigh,   dailyPrevLow,   dailyOpen,
    mondayHigh,      mondayLow,      mondayMid,
  };
}

function buildProximity(levels, price) {
  const proximity = {};
  Object.keys(levels).forEach(k => {
    proximity[k] = levels[k] ? (price - levels[k]) / levels[k] * 100 : null;
  });
  return proximity;
}

function getNearestLevel(levels, proximity) {
  let nearestKey = null;
  let nearestDist = Infinity;
  for (const key of Object.keys(levels)) {
    const dist = proximity[key];
    if (!Number.isFinite(dist)) continue;
    const abs = Math.abs(dist);
    if (abs < nearestDist) {
      nearestDist = abs;
      nearestKey = key;
    }
  }
  if (!nearestKey) return null;
  return {
    key: nearestKey,
    value: levels[nearestKey],
    distancePct: proximity[nearestKey],
  };
}

function computeLevelEvent(symbol, period, distPct, getPrevLevelDist) {
  if (!Number.isFinite(distPct)) return null;
  if (Math.abs(distPct) < 0.5) return 'toque';
  const prev = getPrevLevelDist ? getPrevLevelDist(symbol, period) : null;
  if (!prev || !Number.isFinite(prev.nearest_level_distance_pct)) return null;
  const p = prev.nearest_level_distance_pct;
  if (p < 0 && distPct > 0) return 'reclaim_long';
  if (p > 0 && distPct < 0) return 'reclaim_short';
  return null;
}

function getConfluenceSummary({ nearestLevel, lsr, rsi, oi, taker, cvd }) {
  const near = nearestLevel && Number.isFinite(nearestLevel.distancePct) ? Math.abs(nearestLevel.distancePct) <= 1 : false;
  const lsrLong  = lsr && Number.isFinite(lsr.ratio) && lsr.ratio >= 1.2;
  const lsrShort = lsr && Number.isFinite(lsr.ratio) && lsr.ratio <= 0.9;
  const rsiStrong = Number.isFinite(rsi?.value) && rsi.value >= 60;
  const rsiWeak   = Number.isFinite(rsi?.value) && rsi.value <= 40;
  const oiUp   = Number.isFinite(oi?.deltaPct) && oi.deltaPct > 0.5;
  const oiDown = Number.isFinite(oi?.deltaPct) && oi.deltaPct < -0.5;
  const takerBuy  = taker && Number.isFinite(taker.buySellRatio) && taker.buySellRatio >= 1.1;
  const takerSell = taker && Number.isFinite(taker.buySellRatio) && taker.buySellRatio <= 0.9;
  const cvdBuy  = Number.isFinite(cvd?.deltaPct) && cvd.deltaPct >= 5;
  const cvdSell = Number.isFinite(cvd?.deltaPct) && cvd.deltaPct <= -5;

  if (near && lsrLong  && rsiStrong && oiUp && takerBuy  && cvdBuy)  return 'rompimento comprador com confluencia forte';
  if (near && lsrShort && rsiWeak   && oiUp && takerSell && cvdSell) return 'rompimento vendedor com confluencia forte';
  if (near && oiDown) return 'movimento perto do nivel mais com cara de limpeza do que de entrada nova';
  if (near) return 'preco muito perto de nivel importante';
  if (oiUp && takerBuy  && rsiStrong && cvdBuy)  return 'continuidade compradora em construcao';
  if (oiUp && takerSell && rsiWeak   && cvdSell) return 'continuidade vendedora em construcao';
  return 'contexto misto, sem confluencia forte';
}

function buildScenarioOverview(payload) {
  const near = Number.isFinite(payload.nearestLevel?.distancePct) && Math.abs(payload.nearestLevel.distancePct) <= 1;
  const lsrRatio    = payload.lsr?.ratio;
  const rsiValue    = payload.rsi?.value;
  const oiDelta     = payload.oi?.deltaPct;
  const takerRatio  = payload.taker?.buySellRatio;
  const cvdDeltaPct = payload.cvd?.deltaPct;

  let regime = 'mercado neutro';
  if (Number.isFinite(oiDelta) && oiDelta > 0.5 && Number.isFinite(takerRatio) && takerRatio >= 1.1 && Number.isFinite(cvdDeltaPct) && cvdDeltaPct >= 5) {
    regime = 'expansao compradora';
  } else if (Number.isFinite(oiDelta) && oiDelta > 0.5 && Number.isFinite(takerRatio) && takerRatio <= 0.9 && Number.isFinite(cvdDeltaPct) && cvdDeltaPct <= -5) {
    regime = 'expansao vendedora';
  } else if (Number.isFinite(oiDelta) && oiDelta < -0.5) {
    regime = 'limpeza / desalavancagem';
  }

  let risk = 'risco controlado';
  if ((Number.isFinite(rsiValue) && (rsiValue >= 70 || rsiValue <= 30)) || (Number.isFinite(lsrRatio) && (lsrRatio >= 1.8 || lsrRatio <= 0.7))) {
    risk = 'mercado mais esticado';
  }
  if (near) risk = 'decisao perto de nivel importante';

  return {
    regime,
    flow:         payload.cvd?.bias || payload.taker?.bias || 'sem leitura de fluxo',
    positioning:  payload.lsr?.bias || 'sem leitura de posicionamento',
    risk,
    takeaway:     payload.summary,
  };
}

function computeExpansionScoreFromSignals(signals) {
  let score = 12;
  const positives = [];
  const warnings  = [];

  const nearestDistance = Math.abs(signals.nearestLevelDistancePct ?? Infinity);
  if (nearestDistance <= 0.25)      { score += 22; positives.push('gatilho muito perto do nivel'); }
  else if (nearestDistance <= 0.6)  { score += 15; positives.push('preco perto do gatilho estrutural'); }
  else if (nearestDistance <= 1.2)  { score += 8;  positives.push('ainda perto de nivel importante'); }
  else if (nearestDistance >= 3)    { score -= 8;  warnings.push('movimento longe do gatilho estrutural'); }
  else if (nearestDistance >= 2)    { score -= 4; }

  if (Number.isFinite(signals.oiDeltaPct)) {
    if (signals.oiDeltaPct >= 6)        { score += 24; positives.push('OI entrando muito forte'); }
    else if (signals.oiDeltaPct >= 3)   { score += 16; positives.push('OI entrando forte'); }
    else if (signals.oiDeltaPct >= 1.2) { score += 8;  positives.push('OI em expansao'); }
    else if (signals.oiDeltaPct <= -2)  { score -= 14; warnings.push('OI caindo forte'); }
    else if (signals.oiDeltaPct <= -0.5){ score -= 8;  warnings.push('OI sem sustentacao'); }
  }

  if (Number.isFinite(signals.takerBuySellRatio)) {
    if (signals.takerBuySellRatio >= 1.3)        { score += 20; positives.push('taker comprador muito agressivo'); }
    else if (signals.takerBuySellRatio >= 1.15)  { score += 13; positives.push('taker comprador forte'); }
    else if (signals.takerBuySellRatio >= 1.06)  { score += 8;  positives.push('taker comprador'); }
    else if (signals.takerBuySellRatio <= 0.9)   { score -= 14; warnings.push('taker vendedor forte'); }
    else if (signals.takerBuySellRatio <= 0.97)  { score -= 8;  warnings.push('taker vendedor'); }
  }

  if (Number.isFinite(signals.cvdDeltaPct)) {
    if (signals.cvdDeltaPct >= 18)      { score += 22; positives.push('CVD explosivo'); }
    else if (signals.cvdDeltaPct >= 10) { score += 15; positives.push('CVD confirma agressao compradora'); }
    else if (signals.cvdDeltaPct >= 5)  { score += 10; positives.push('CVD positivo'); }
    else if (signals.cvdDeltaPct <= -10){ score -= 14; warnings.push('CVD fortemente contra a compra'); }
    else if (signals.cvdDeltaPct <= -5) { score -= 8;  warnings.push('CVD contra a compra'); }
  }

  if (Number.isFinite(signals.rsi)) {
    if (signals.rsi >= 58 && signals.rsi <= 67) { score += 16; positives.push('RSI em faixa de ignicao'); }
    else if (signals.rsi > 67 && signals.rsi <= 72) { score += 8; positives.push('RSI forte'); }
    else if (signals.rsi >= 75) { score -= 14; warnings.push('RSI ja muito esticado'); }
    else if (signals.rsi < 50)  { score -= 10; warnings.push('RSI ainda sem aceleracao'); }
  }

  if (Number.isFinite(signals.lsrRatio)) {
    if (signals.lsrRatio >= 0.9 && signals.lsrRatio <= 1.35)      { score += 12; positives.push('LSR leve e saudavel'); }
    else if (signals.lsrRatio >= 0.75 && signals.lsrRatio < 0.9)  { score += 10; positives.push('LSR ainda favorece squeeze'); }
    else if (signals.lsrRatio > 1.35 && signals.lsrRatio <= 1.7)  { score += 2; }
    else if (signals.lsrRatio >= 1.9)  { score -= 14; warnings.push('crowding long elevado'); }
    else if (signals.lsrRatio <= 0.65) { score -= 4;  warnings.push('mercado ainda muito inclinado para short'); }
  }

  const ignitionAligned =
    Number.isFinite(signals.oiDeltaPct)       && signals.oiDeltaPct >= 3 &&
    Number.isFinite(signals.takerBuySellRatio) && signals.takerBuySellRatio >= 1.15 &&
    Number.isFinite(signals.cvdDeltaPct)       && signals.cvdDeltaPct >= 10;
  if (ignitionAligned) { score += 18; positives.push('OI, taker e CVD alinhados'); }

  const breakoutReady =
    nearestDistance <= 0.8 &&
    Number.isFinite(signals.rsi) && signals.rsi >= 58 && signals.rsi <= 72 &&
    Number.isFinite(signals.oiDeltaPct) && signals.oiDeltaPct >= 2;
  if (breakoutReady) { score += 12; positives.push('cara de rompimento com aceitacao'); }

  const squeezeFuel =
    Number.isFinite(signals.lsrRatio)          && signals.lsrRatio <= 0.95 &&
    Number.isFinite(signals.cvdDeltaPct)       && signals.cvdDeltaPct >= 8 &&
    Number.isFinite(signals.takerBuySellRatio) && signals.takerBuySellRatio >= 1.1;
  if (squeezeFuel) { score += 12; positives.push('potencial de short squeeze'); }

  const fakeMoveRisk =
    (!Number.isFinite(signals.oiDeltaPct) || signals.oiDeltaPct <= 1) &&
    Number.isFinite(signals.rsi) && signals.rsi >= 75 &&
    Number.isFinite(signals.cvdDeltaPct) && signals.cvdDeltaPct < 5;
  if (fakeMoveRisk) { score -= 12; warnings.push('alta sem sustentacao clara de fluxo e OI'); }

  const finalScore = clamp(Math.round(score), 0, 100);
  let label = 'fraco';
  if (finalScore >= 82)      label = 'explosivo';
  else if (finalScore >= 68) label = 'muito promissor';
  else if (finalScore >= 52) label = 'monitorar';

  return { score: finalScore, label, positives, warnings, reasons: [...positives.slice(0, 3), ...warnings.slice(0, 2)] };
}

// ── Risk Management (50k prop firm, max DD $2k) ──
const ASSET_SPECS = {
  mnq: { tickValue: 5,  label: 'MNQ',  defaultStop: 20, stopRange: [12, 35] },
  btc: { tickValue: 5,  label: 'BTC',  defaultStop: 200, stopRange: [100, 500] }, // BTC futures CME $5/tick
  cl:  { tickValue: 10, label: 'CL',   defaultStop: 50, stopRange: [30, 100] },   // Crude Oil $10/pt
};
const ACCOUNT_SIZE = 50000;
const MAX_DRAWDOWN = 2000; // 4%
const MAX_DAILY_LOSS = 500; // 1%

function calcRisk(mlData, asset) {
  const spec = ASSET_SPECS[asset] || ASSET_SPECS.mnq;
  const diff = (mlData.prob_long || 0) - (mlData.prob_short || 0);
  const inUs = mlData.is_us_session;
  const adxActive = mlData.adx_active;
  const strongDiv = mlData.strong_div;
  const bias = mlData.ema20_bias_mnq_btc; // 0=abaixo, 1=misto, 2=acima
  const signalIsLong = mlData.signal_label === 'LONG';
  const signalIsShort = mlData.signal_label === 'SHORT';

  // Confianca base
  let riskPct = 0;
  let tier = 'sem sinal';
  let maxContracts = 0;

  if (diff > 0.10 && signalIsLong) {
    riskPct = 0.004; tier = 'alta';
    maxContracts = 3;
  } else if (diff > 0.05 && signalIsLong) {
    riskPct = 0.0025; tier = 'moderada';
    maxContracts = 2;
  } else if (diff < -0.10 && signalIsShort) {
    riskPct = 0.003; tier = 'alta';
    maxContracts = 2;
  } else if (diff < -0.05 && signalIsShort) {
    riskPct = 0.002; tier = 'moderada';
    maxContracts = 1;
  } else {
    return {
      tradeable: false,
      reason: 'confianca baixa — edge insuficiente',
      tier,
      riskPerTrade: 0,
      maxContracts: 0,
      dailyLossRemaining: MAX_DAILY_LOSS,
      drawdownRemaining: MAX_DRAWDOWN,
    };
  }

  // Multiplicador de conviction por filtros extras
  let convictionMul = 1.0;
  const activeFilters = [];
  if (inUs && (signalIsLong || signalIsShort)) {
    convictionMul += signalIsLong && inUs ? 0.15 : 0.1;
    activeFilters.push('sessao US');
  }
  if (adxActive) {
    convictionMul += 0.1;
    activeFilters.push('ADX ativo');
  }
  if (strongDiv) {
    const aligned = signalIsLong ? mlData.price_div_cl < 0 : mlData.price_div_cl > 0;
    if (aligned) { convictionMul += 0.15; activeFilters.push('strong_div alinhado'); }
    else { convictionMul -= 0.1; activeFilters.push('strong_div contra-indicado'); }
  }
  if (signalIsLong && bias === 2) { convictionMul += 0.1; activeFilters.push('regime EMA20 favorece LONG'); }
  if (signalIsShort && bias === 0) { convictionMul += 0.1; activeFilters.push('regime EMA20 favorece SHORT'); }

  // Risk final com conviccao
  const riskAmount = Math.round(ACCOUNT_SIZE * riskPct * clamp(convictionMul, 0.5, 1.5));

  // Stop loss sugerido baseado no modelo (ADX define volatilidade)
  let suggestedStopPts = spec.defaultStop;
  if (adxActive) {
    const adxVal = mlData.adx_mnq || 17;
    suggestedStopPts = Math.round(clamp(adxVal * (asset === 'btc' ? 12 : asset === 'cl' ? 3 : 1.2), spec.stopRange[0], spec.stopRange[1]));
  }

  const contractsFromRisk = Math.floor(riskAmount / (suggestedStopPts * spec.tickValue));
  const contracts = clamp(contractsFromRisk, 1, maxContracts);

  const effectiveRisk = contracts * suggestedStopPts * spec.tickValue;
  const dailyLossRemaining = MAX_DAILY_LOSS;
  const drawdownRemaining = MAX_DRAWDOWN;

  let direction = signalIsLong ? 'LONG' : 'SHORT';

  return {
    tradeable: true,
    reason: `${direction} — ${tier} confianca (diff=${diff > 0 ? '+' : ''}${diff.toFixed(3)})`,
    tier,
    direction,
    riskPerTrade: effectiveRisk,
    riskPctDisplay: (effectiveRisk / ACCOUNT_SIZE * 100).toFixed(2) + '%',
    maxContracts: contracts,
    suggestedStopPts,
    suggestedStopDollars: suggestedStopPts * spec.tickValue,
    dailyLossRemaining,
    drawdownRemaining,
    drawdownPctDisplay: (drawdownRemaining / ACCOUNT_SIZE * 100).toFixed(1) + '%',
    convictionMul: +convictionMul.toFixed(2),
    activeFilters,
    accountSize: ACCOUNT_SIZE,
    asset: spec.label,
    tickValue: spec.tickValue,
  };
}

module.exports = {
  clamp, getRsiBias, getLsrBias, getOiBias, getTakerBias, getCvdBias,
  buildLevelsFromKlines, buildProximity, getNearestLevel, computeLevelEvent,
  getConfluenceSummary, buildScenarioOverview, computeExpansionScoreFromSignals,
  calcRisk,
};

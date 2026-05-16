const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const utils = require('./utils');

const app = express();
const PORT = process.env.PORT || 3000;

const PREDICT_SCRIPT = path.join(__dirname, 'ml', 'predict.py');
let mlCache = { data: null, ts: 0 };
const ML_TTL = 5 * 60 * 1000;

app.use(express.static(__dirname));

const YAHOO_BASE = 'https://query1.finance.yahoo.com/v8/finance/chart';
const SYMBOLS = {
  mnq: 'MNQ=F',
  btc: 'BTC-USD',
  cl:  'CL=F',
};

function ma50(closes) {
  if (!closes || closes.length < 50) return null;
  const slice = closes.slice(-50);
  return slice.reduce((a, b) => a + b, 0) / 50;
}

function rsi(closes, period) {
  if (!closes || closes.length < period + 1) return null;
  const gains = [], losses = [];
  for (let i = 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    gains.push(d > 0 ? d : 0);
    losses.push(d < 0 ? -d : 0);
  }
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
  }
  if (avgLoss === 0) return 100;
  return 100 - 100 / (1 + avgGain / avgLoss);
}

function adx(highs, lows, closes, period = 14) {
  if (!highs || highs.length < period * 2 + 1) return null;

  const trs = [], plusDMs = [], minusDMs = [];
  for (let i = 1; i < highs.length; i++) {
    const h = highs[i], l = lows[i], pc = closes[i - 1];
    const ph = highs[i - 1], pl = lows[i - 1];

    trs.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));

    const upMove = h - ph;
    const downMove = pl - l;
    plusDMs.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDMs.push(downMove > upMove && downMove > 0 ? downMove : 0);
  }

  if (trs.length < period) return null;

  let smTR  = trs.slice(0, period).reduce((a, b) => a + b, 0);
  let smPDM = plusDMs.slice(0, period).reduce((a, b) => a + b, 0);
  let smMDM = minusDMs.slice(0, period).reduce((a, b) => a + b, 0);

  const dxArr = [];
  const pushDX = () => {
    const pDI = smTR > 0 ? 100 * smPDM / smTR : 0;
    const mDI = smTR > 0 ? 100 * smMDM / smTR : 0;
    const s = pDI + mDI;
    dxArr.push(s > 0 ? 100 * Math.abs(pDI - mDI) / s : 0);
  };
  pushDX();

  for (let i = period; i < trs.length; i++) {
    smTR  = smTR  - smTR  / period + trs[i];
    smPDM = smPDM - smPDM / period + plusDMs[i];
    smMDM = smMDM - smMDM / period + minusDMs[i];
    pushDX();
  }

  if (dxArr.length < period) return null;

  let adxVal = dxArr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < dxArr.length; i++) {
    adxVal = (adxVal * (period - 1) + dxArr[i]) / period;
  }
  return adxVal;
}

async function fetchYahoo(symbol, interval = '5m', range = '5d') {
  const url = `${YAHOO_BASE}/${encodeURIComponent(symbol)}?interval=${interval}&range=${range}`;
  const res = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0' },
  });
  if (!res.ok) throw new Error(`Yahoo HTTP ${res.status} for ${symbol}`);
  const json = await res.json();
  if (json.chart.error) throw new Error(`Yahoo error for ${symbol}: ${json.chart.error}`);
  const r = json.chart.result?.[0];
  if (!r) throw new Error(`No result for ${symbol}`);
  const quotes = r.indicators?.quote?.[0];
  if (!quotes) throw new Error(`No quotes for ${symbol}`);

  const rawC = quotes.close ?? [], rawH = quotes.high ?? [], rawL = quotes.low ?? [];
  const closes = [], highs = [], lows = [];
  for (let i = 0; i < rawC.length; i++) {
    if (rawC[i] != null && rawH[i] != null && rawL[i] != null) {
      closes.push(rawC[i]);
      highs.push(rawH[i]);
      lows.push(rawL[i]);
    }
  }

  if (closes.length < 2) throw new Error(`Not enough data for ${symbol}`);
  return { closes, highs, lows, price: closes[closes.length - 1] };
}

let cache = {};
const CACHE_TTL = 60_000;

function buildInstrumentResult(closes, highs, lows, price, rsiPeriod, adxPeriod) {
  const ma50Val = ma50(closes);
  const priceVsMa50 = ma50Val ? +((price - ma50Val) / ma50Val * 100).toFixed(3) : null;
  return {
    rsi:         rsi(closes, rsiPeriod),
    adx:         adx(highs, lows, closes, adxPeriod),
    price,
    ma50:        ma50Val ? +ma50Val.toFixed(4) : null,
    priceVsMa50,
    aboveMa50:   ma50Val ? price > ma50Val : null,
  };
}

async function getInstrument(key, rsiPeriod, adxPeriod) {
  const now = Date.now();
  const cached = cache[key];
  if (cached && now - cached.ts < CACHE_TTL) {
    return buildInstrumentResult(cached.closes, cached.highs, cached.lows, cached.price, rsiPeriod, adxPeriod);
  }
  const data = await fetchYahoo(SYMBOLS[key]);
  cache[key] = { closes: data.closes, highs: data.highs, lows: data.lows, price: data.price, ts: now };
  return buildInstrumentResult(data.closes, data.highs, data.lows, data.price, rsiPeriod, adxPeriod);
}

app.get('/api/mnq-cl/scan', async (req, res) => {
  try {
    const rsiPeriod = parseInt(req.query.rsi) || 14;
    const adxPeriod = parseInt(req.query.adx) || 14;
    const [mnq, btc, cl] = await Promise.all([
      getInstrument('mnq', rsiPeriod, adxPeriod),
      getInstrument('btc', rsiPeriod, adxPeriod),
      getInstrument('cl',  rsiPeriod, adxPeriod),
    ]);
    const divergences = {
      mnq_cl_divergence: mnq.rsi != null && cl.rsi != null  ? +(mnq.rsi - cl.rsi).toFixed(2)  : null,
      mnq_btc_spread:    mnq.rsi != null && btc.rsi != null ? +(mnq.rsi - btc.rsi).toFixed(2) : null,
    };
    mnq.rsiBias = utils.getRsiBias(mnq.rsi);
    btc.rsiBias = utils.getRsiBias(btc.rsi);
    cl.rsiBias  = utils.getRsiBias(cl.rsi);
    res.json({ mnq, btc, cl, divergences, ts: new Date().toISOString() });
  } catch (err) {
    res.json({ error: err.message });
  }
});

// ── Binance BTC Derivatives ──────────────────────────────────────────────────
const BINANCE = 'https://fapi.binance.com';
const BTC_SYM = 'BTCUSDT';
let binanceCache = { data: null, ts: 0 };
const BINANCE_TTL = 5 * 60 * 1000;

async function fetchBinanceBtc() {
  const [lsrRes, oiRes, takerRes, klineRes] = await Promise.allSettled([
    fetch(`${BINANCE}/futures/data/topLongShortPositionRatio?symbol=${BTC_SYM}&period=1h&limit=1`).then(r => r.json()),
    fetch(`${BINANCE}/futures/data/openInterestHist?symbol=${BTC_SYM}&period=1h&limit=2`).then(r => r.json()),
    fetch(`${BINANCE}/futures/data/takerlongshortRatio?symbol=${BTC_SYM}&period=1h&limit=1`).then(r => r.json()),
    fetch(`${BINANCE}/fapi/v1/klines?symbol=${BTC_SYM}&interval=1h&limit=3`).then(r => r.json()),
  ]);

  const lsr = lsrRes.status === 'fulfilled' && lsrRes.value?.[0]
    ? { ratio: parseFloat(lsrRes.value[0].longShortRatio) }
    : null;

  let oi = null;
  if (oiRes.status === 'fulfilled' && Array.isArray(oiRes.value) && oiRes.value.length >= 2) {
    const curr = parseFloat(oiRes.value[oiRes.value.length - 1].sumOpenInterestValue);
    const prev = parseFloat(oiRes.value[oiRes.value.length - 2].sumOpenInterestValue);
    oi = { deltaPct: +((curr - prev) / prev * 100).toFixed(3) };
  }

  const taker = takerRes.status === 'fulfilled' && takerRes.value?.[0]
    ? { buySellRatio: parseFloat(takerRes.value[0].buySellRatio) }
    : null;

  let cvd = null;
  if (klineRes.status === 'fulfilled' && Array.isArray(klineRes.value) && klineRes.value.length >= 1) {
    const k = klineRes.value[klineRes.value.length - 1];
    const totalVol    = parseFloat(k[5]);
    const takerBuyVol = parseFloat(k[9]);
    const delta = takerBuyVol - (totalVol - takerBuyVol);
    cvd = { deltaPct: totalVol > 0 ? +((delta / totalVol) * 100).toFixed(2) : 0 };
  }

  const lsrBias   = utils.getLsrBias(lsr?.ratio ?? null);
  const oiBias    = utils.getOiBias(oi?.deltaPct ?? NaN);
  const takerBias = utils.getTakerBias(taker?.buySellRatio ?? NaN);
  const cvdBias   = utils.getCvdBias(cvd);

  if (lsr)   lsr.bias   = lsrBias;
  if (oi)    oi.bias    = oiBias;
  if (taker) taker.bias = takerBias;
  if (cvd)   cvd.bias   = cvdBias;

  return { lsr, oi, taker, cvd };
}

app.get('/api/btc/derivatives', async (req, res) => {
  try {
    const now = Date.now();
    if (!req.query.force && binanceCache.data && now - binanceCache.ts < BINANCE_TTL) {
      return res.json({ ...binanceCache.data, cached: true });
    }
    const data = await fetchBinanceBtc();
    binanceCache = { data, ts: now };
    res.json(data);
  } catch (err) {
    res.json({ error: err.message });
  }
});

// ── ML predict ───────────────────────────────────────────────────────────────
function runPredict() {
  return new Promise((resolve, reject) => {
    const proc = spawn('python', [PREDICT_SCRIPT]);
    let out = '';
    proc.stdout.on('data', d => { out += d; });
    proc.on('close', () => {
      try { resolve(JSON.parse(out.trim())); }
      catch { reject(new Error('predict parse error: ' + out.slice(0, 200))); }
    });
    proc.on('error', reject);
    setTimeout(() => reject(new Error('predict timeout')), 60_000);
  });
}

app.get('/api/ml/predict', async (req, res) => {
  try {
    const now = Date.now();
    if (!req.query.force && mlCache.data && now - mlCache.ts < ML_TTL) {
      return res.json({ ...mlCache.data, cached: true });
    }
    const data = await runPredict();
    mlCache = { data, ts: now };
    res.json(data);
  } catch (err) {
    res.json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`MNQ-CL server running on http://localhost:${PORT}`);
});

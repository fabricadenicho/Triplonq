const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const https = require('https');
const fs = require('fs');
const utils = require('./utils');

// ── Carregar .env ────────────────────────────────────────────────────────────
const envPath = path.join(__dirname, '.env');
if (fs.existsSync(envPath)) {
  fs.readFileSync(envPath, 'utf8').split('\n').forEach(line => {
    const m = line.match(/^([^=\s#][^=]*)=(.*)$/);
    if (m) process.env[m[1].trim()] = m[2].trim();
  });
}

// ── Telegram ─────────────────────────────────────────────────────────────────
function sendTelegram(text) {
  const token  = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  if (!token || !chatId || chatId === 'SEU_CHAT_ID_AQUI') return;
  const body = JSON.stringify({ chat_id: chatId, text, parse_mode: 'HTML' });
  const opts = {
    hostname: 'api.telegram.org',
    path: `/bot${token}/sendMessage`,
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
  };
  const req = https.request(opts, res => { res.resume(); });
  req.on('error', e => console.error('[Telegram]', e.message));
  req.write(body);
  req.end();
}

const TG_ASSET_LABEL = { mnq: 'MNQ 📈', btc: 'BTC 🟡', cl: 'CL 🛢', es: 'ES 📊' };

function buildTelegramMsg(asset, a) {
  const sinal = a.sinal;
  const emoji = sinal === 'LONG' ? '🟢' : '🔴';
  const dir   = sinal === 'LONG' ? 'COMPRAR' : 'VENDER';
  const lbl   = TG_ASSET_LABEL[asset] || asset.toUpperCase();
  const s     = a.setup || {};
  const f2    = v => (v != null ? v.toFixed(2) : '—');
  const f1    = v => (v != null ? v.toFixed(1) : '—');

  let msg = `${emoji} <b>${dir} ${lbl}</b>\n\n`;
  msg += `💵 Entrada: <code>${f2(a.preco)}</code>\n`;
  msg += `🛑 Stop:    <code>${f2(a.stop)}</code>   (${s.stop_r || 1.5}R)\n`;
  msg += `🎯 Target:  <code>${f2(a.target)}</code>   (${s.target_r || 2}R)\n\n`;
  msg += `📊 LONG ${f1(a.conf_long)}%  ·  SHORT ${f1(a.conf_short)}%\n`;
  msg += `⏰ ${a.hora || '—'} UTC\n`;
  if (a.risco_dolar) {
    msg += `💰 Risco: $${Math.round(a.risco_dolar)}  ·  ${a.contratos} contrato(s)\n`;
  }
  msg += `\n#Triplonq #${asset.toUpperCase()} #PropFirm`;
  return msg;
}

// ── Log de sinais enviados ───────────────────────────────────────────────────
const SIGNALS_LOG = path.join(__dirname, 'ml', 'signals_log.csv');
const SIGNALS_HEADER = 'ts,asset,direction,entry,stop,target,prob,atr,stop_r,target_r,conf_long,conf_short\n';

function logSignal(asset, a) {
  try {
    if (!fs.existsSync(SIGNALS_LOG)) {
      fs.writeFileSync(SIGNALS_LOG, SIGNALS_HEADER);
    }
    const s   = a.setup || {};
    const ts  = a.hora || new Date().toISOString().slice(0, 16).replace('T', ' ');
    const row = [
      ts,
      asset,
      a.sinal,
      a.preco    != null ? a.preco.toFixed(3)     : '',
      a.stop     != null ? a.stop.toFixed(3)      : '',
      a.target   != null ? a.target.toFixed(3)    : '',
      a.conf     != null ? (a.conf / 100).toFixed(4) : '',
      a.atr      != null ? a.atr.toFixed(4)       : '',
      s.stop_r   ?? 1.5,
      s.target_r ?? 2.0,
      a.conf_long  != null ? (a.conf_long  / 100).toFixed(4) : '',
      a.conf_short != null ? (a.conf_short / 100).toFixed(4) : '',
    ].join(',');
    fs.appendFileSync(SIGNALS_LOG, row + '\n');
    console.log(`[SignalLog] ${asset.toUpperCase()} ${a.sinal} salvo em signals_log.csv`);
  } catch (e) {
    console.error('[logSignal]', e.message);
  }
}

// ── Monitor de sinais (roda a cada 5 min) ───────────────────────────────────
const lastSignals = { mnq: null, btc: null, cl: null, es: null };
const shortCooldowns = {}; // asset -> timestamp da ultima vez q enviou SHORT

async function checkSignals() {
  try {
    const data = await runPredict(LIVE2_SCRIPT);
    if (!data || !data.assets) return;
    live2Cache = { data, ts: Date.now() };

    for (const asset of ['mnq', 'btc', 'cl', 'es']) {
      const a = data.assets[asset];
      if (!a || a.erro) continue;
      const prev = lastSignals[asset];
      const curr = a.sinal;
      if (curr !== 'NO_TRADE' && curr !== prev) {
        // Cooldown: SHORT de CL no max 1 por hora
        if (asset === 'cl' && curr === 'SHORT') {
          const lastShort = shortCooldowns['cl'] || 0;
          if (Date.now() - lastShort < 60 * 60 * 1000) {
            console.log(`[Telegram] CL SHORT ignorado (cooldown de 1h)`);
            lastSignals[asset] = curr;
            continue;
          }
          shortCooldowns['cl'] = Date.now();
        }
        logSignal(asset, a);
        const msg = buildTelegramMsg(asset, a);
        sendTelegram(msg);
        console.log(`[Telegram] ${asset.toUpperCase()} ${curr} enviado`);
      }
      lastSignals[asset] = curr;
    }
  } catch (e) {
    console.error('[checkSignals]', e.message);
  }
}

const app = express();
const PORT = process.env.PORT || 3000;

const PYTHON_PATH = process.env.PYTHON_PATH || 'C:\\Python314\\python.exe';

app.use(express.static(__dirname));

app.get('/', (req, res) => res.redirect('/live2'));

// ── ML predict ───────────────────────────────────────────────────────────────
function runPredict(script) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_PATH, [script]);
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

// ── Modelo Divergência (MNQ sobe + CL cai) ───────────────────────────────
const DIVERG_SCRIPT = path.join(__dirname, 'ml', 'predict_divergencia.py');
let divergCache = { data: null, ts: 0 };
const DIVERG_TTL = 5 * 60 * 1000;

app.get('/api/divergencia', async (req, res) => {
  try {
    const now = Date.now();
    if (!req.query.force && divergCache.data && now - divergCache.ts < DIVERG_TTL) {
      return res.json({ ...divergCache.data, cached: true });
    }
    const data = await runPredict(DIVERG_SCRIPT);
    divergCache = { data, ts: now };
    res.json(data);
  } catch (err) {
    res.json({ error: err.message });
  }
});

// ── S1: ROT-LONG + acima 4H + ML (WR=78.6% OOS) ────────────────────────────
const S1_SCRIPT = path.join(__dirname, 'ml', 'predict_s1_4h.py');
let s1Cache = { data: null, ts: 0 };
const S1_TTL  = 5 * 60 * 1000;
let lastS1Signal = null;

app.get('/api/s1-4h', async (req, res) => {
  try {
    const now = Date.now();
    if (!req.query.force && s1Cache.data && now - s1Cache.ts < S1_TTL) {
      return res.json({ ...s1Cache.data, cached: true });
    }
    const data = await runPredict(S1_SCRIPT);
    s1Cache = { data, ts: now };

    // Telegram: dispara so quando sinal muda para LONG
    if (data && data.sinal === 'LONG' && lastS1Signal !== 'LONG') {
      const msg = `<b>S1 LONG MNQ</b>\n\n` +
        `Score Rotacao: <code>${data.rot_score}%</code>\n` +
        `ML Prob:       <code>${(data.ml_prob * 100).toFixed(1)}%</code>  (edge ${data.ml_edge > 0 ? '+' : ''}${(data.ml_edge * 100).toFixed(1)}pp)\n` +
        `4H Open:       <code>${data.open_4h}</code>  |  Close: <code>${data.mnq_close}</code>  (${data.dist_4h_pct > 0 ? '+' : ''}${data.dist_4h_pct}%)\n` +
        `DIV CL:        <code>${data.div_cl}</code>  RSI MNQ: <code>${data.rsi_mnq}</code>\n` +
        `Hora: ${data.hour}h UTC  |  ADX: ${data.adx_mnq}\n\n` +
        `#S1 #MNQ #Rotacao`;
      sendTelegram(msg);
    }
    lastS1Signal = data ? data.sinal : null;

    res.json(data);
  } catch (err) {
    res.json({ error: err.message });
  }
});

// ── Divergencia Stats ────────────────────────────────────────────────────
const DIVERG_STATS_SCRIPT = path.join(__dirname, 'ml', 'divergencia_stats.py');

app.get('/api/divergencia/stats', async (req, res) => {
  try {
    const data = await runPredict(DIVERG_STATS_SCRIPT);
    res.json(data);
  } catch (err) {
    res.json({ error: err.message });
  }
});

// ── Live 2.0 (novos modelos forward=8h) ──────────────────────────────────
const LIVE2_SCRIPT = path.join(__dirname, 'ml', 'teste', 'predict_live.py');
let live2Cache = { data: null, ts: 0 };
const LIVE2_TTL = 5 * 60 * 1000;

app.get('/api/live2', async (req, res) => {
  try {
    const now = Date.now();
    if (!req.query.force && live2Cache.data && now - live2Cache.ts < LIVE2_TTL) {
      return res.json({ ...live2Cache.data, cached: true });
    }
    const data = await runPredict(LIVE2_SCRIPT);
    live2Cache = { data, ts: now };
    res.json(data);
  } catch (err) {
    res.json({ error: err.message });
  }
});

app.get('/live2', (req, res) => res.sendFile(path.join(__dirname, 'live2.html')));

const SEMANAL_SCRIPT = path.join(__dirname, 'ml', 'teste', 'analisar_semanal.py');
let semanalCache = { data: null, ts: 0 };
const SEMANAL_TTL = 7 * 24 * 60 * 60 * 1000; // 7 dias

app.get('/api/live2/semanal', async (req, res) => {
  try {
    const now = Date.now();
    if (!req.query.force && semanalCache.data && now - semanalCache.ts < SEMANAL_TTL) {
      return res.json({ ...semanalCache.data, cached: true });
    }
    const data = await runPredict(SEMANAL_SCRIPT);
    semanalCache = { data, ts: now };
    res.json(data);
  } catch (err) {
    res.json({ error: err.message });
  }
});

// ── Performance ao vivo ───────────────────────────────────────────────────────
const PERF_CSV    = path.join(__dirname, 'ml', 'live_performance.csv');
const VALIDATE_PY = path.join(__dirname, 'ml', 'validate_live.py');

function parsePerformanceCsv() {
  if (!fs.existsSync(PERF_CSV)) return [];
  const lines = fs.readFileSync(PERF_CSV, 'utf8').trim().split('\n');
  if (lines.length < 2) return [];
  const header = lines[0].split(',');
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const row = {};
    header.forEach((h, i) => { row[h.trim()] = vals[i]?.trim() ?? ''; });
    row.prob     = parseFloat(row.prob)    || 0;
    row.hour     = parseInt(row.hour)      || 0;
    row.dow      = parseInt(row.dow)       || 0;
    row.stop_r   = parseFloat(row.stop_r)  || 1.5;
    row.target_r = parseFloat(row.target_r)|| 2.0;
    row.pnl_r    = parseFloat(row.pnl_r)   ||
      (row.result === 'WIN' ? row.target_r : row.result === 'LOSS' ? -row.stop_r : 0);
    row.asset    = row.asset || 'cl';
    return row;
  }).filter(r => r.ts);
}

function calcStats(rows) {
  const wins     = rows.filter(r => r.result === 'WIN').length;
  const losses   = rows.filter(r => r.result === 'LOSS').length;
  const timeouts = rows.filter(r => r.result === 'TIMEOUT').length;
  const decided  = wins + losses;
  const pnl      = parseFloat(rows.reduce((s, r) => s + r.pnl_r, 0).toFixed(2));
  const wr       = decided > 0 ? parseFloat((wins / decided * 100).toFixed(1)) : 0;
  return { total: rows.length, wins, losses, timeouts, wr, pnl };
}

app.get('/api/performance', (req, res) => {
  try {
    const all  = parsePerformanceCsv();
    if (all.length === 0) return res.json({ trades: [], summary: {}, by_hour: [], by_dir: {}, by_asset: {}, equity: [] });

    const asset  = req.query.asset || 'all';
    const rows   = asset === 'all' ? all : all.filter(r => r.asset === asset);

    // equity acumulada
    let cum = 0;
    const equity = rows.map(r => { cum += r.pnl_r; return { ts: r.ts, cum: parseFloat(cum.toFixed(2)), asset: r.asset }; });

    // por hora
    const hours = {};
    rows.forEach(r => {
      const h = r.hour;
      if (!hours[h]) hours[h] = { hour: h, wins: 0, losses: 0, timeouts: 0, pnl: 0 };
      hours[h][r.result === 'WIN' ? 'wins' : r.result === 'LOSS' ? 'losses' : 'timeouts']++;
      hours[h].pnl = parseFloat((hours[h].pnl + r.pnl_r).toFixed(2));
    });
    const by_hour = Object.values(hours).sort((a, b) => a.hour - b.hour).map(h => ({
      ...h, wr: h.wins + h.losses > 0 ? parseFloat((h.wins / (h.wins + h.losses) * 100).toFixed(1)) : 0
    }));

    // por direcao
    const by_dir = {};
    rows.forEach(r => {
      if (!by_dir[r.direction]) by_dir[r.direction] = { wins: 0, losses: 0, timeouts: 0, pnl: 0 };
      by_dir[r.direction][r.result === 'WIN' ? 'wins' : r.result === 'LOSS' ? 'losses' : 'timeouts']++;
      by_dir[r.direction].pnl = parseFloat((by_dir[r.direction].pnl + r.pnl_r).toFixed(2));
    });

    // por ativo
    const by_asset = {};
    ['mnq','btc','cl','es'].forEach(a => {
      const sub = all.filter(r => r.asset === a);
      if (sub.length > 0) by_asset[a] = calcStats(sub);
    });

    res.json({
      summary: calcStats(rows),
      by_dir, by_hour, by_asset, equity,
      trades: rows.slice().reverse().slice(0, 100),
    });
  } catch (err) {
    res.json({ error: err.message });
  }
});

app.get('/api/performance/refresh', async (req, res) => {
  try {
    const days     = parseInt(req.query.days) || 30;
    const backfill = req.query.backfill !== '0'; // default true (backfill on manual refresh)
    const args     = backfill
      ? [VALIDATE_PY, '--backfill', '--days', String(days)]
      : [VALIDATE_PY];
    const data = await new Promise((resolve, reject) => {
      const proc = spawn(PYTHON_PATH, args, { cwd: __dirname });
      let out = '', err = '';
      proc.stdout.on('data', d => { out += d; });
      proc.stderr.on('data', d => { err += d; });
      proc.on('close', code => code === 0 ? resolve({ ok: true, log: out }) : reject(new Error(err || out)));
      setTimeout(() => reject(new Error('validate_live timeout')), 120_000);
    });
    res.json(data);
  } catch (err) {
    res.json({ error: err.message });
  }
});

app.get('/api/signals-log', (req, res) => {
  try {
    if (!fs.existsSync(SIGNALS_LOG)) return res.json({ signals: [] });
    const lines  = fs.readFileSync(SIGNALS_LOG, 'utf8').trim().split('\n');
    if (lines.length < 2) return res.json({ signals: [] });
    const header = lines[0].split(',');
    const signals = lines.slice(1).map(line => {
      const vals = line.split(',');
      const row  = {};
      header.forEach((h, i) => { row[h.trim()] = vals[i]?.trim() ?? ''; });
      return row;
    }).filter(r => r.ts).reverse().slice(0, 200);
    res.json({ signals, total: lines.length - 1 });
  } catch (e) {
    res.json({ error: e.message });
  }
});

app.get('/performance', (req, res) => res.sendFile(path.join(__dirname, 'performance.html')));

app.get('/api/telegram-test', (req, res) => {
  sendTelegram('🤖 <b>Triplonq Bot</b> conectado!\n\nSinais ML PropFirm ativos ✅\nMonitorando: MNQ · BTC · CL · ES\n\n#Triplonq #PropFirm');
  res.json({ ok: true });
});

function autoResolveSignals() {
  const proc = spawn(PYTHON_PATH, [VALIDATE_PY], { cwd: __dirname });
  let out = '';
  proc.stdout.on('data', d => { out += d; });
  proc.on('close', code => {
    if (code === 0 && out.trim()) console.log('[AutoResolve]', out.trim().split('\n')[0]);
    else if (code !== 0) console.error('[AutoResolve] erro código', code);
  });
  proc.on('error', e => console.error('[AutoResolve]', e.message));
  setTimeout(() => proc.kill(), 120_000);
}

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`MNQ-CL server running on http://localhost:${PORT}`);
    setTimeout(checkSignals, 15_000);
    setInterval(checkSignals, 1 * 60 * 1000);
    // Auto-resolve pending signals every 2h
    setTimeout(autoResolveSignals, 60_000);
    setInterval(autoResolveSignals, 2 * 60 * 60 * 1000);
  });
}

module.exports = app;

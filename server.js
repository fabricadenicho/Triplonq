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
            lastSignals[asset] = curr; // ainda atualiza pra evitar spam na proxima
            continue;
          }
          shortCooldowns['cl'] = Date.now();
        }
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
const SEMANAL_TTL = 6 * 60 * 60 * 1000; // 6 horas

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

app.get('/api/telegram-test', (req, res) => {
  sendTelegram('🤖 <b>Triplonq Bot</b> conectado!\n\nSinais ML PropFirm ativos ✅\nMonitorando: MNQ · BTC · CL · ES\n\n#Triplonq #PropFirm');
  res.json({ ok: true });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`MNQ-CL server running on http://localhost:${PORT}`);
    setTimeout(checkSignals, 15_000);
    setInterval(checkSignals, 1 * 60 * 1000);
  });
}

module.exports = app;

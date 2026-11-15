// capture harness: node shoot.mjs <outdir> [poses|fly|bench|probe] [args]
import puppeteer from 'puppeteer-core';
import { mkdirSync, writeFileSync } from 'node:fs';
const [out = 'shots', what = 'poses', ...rest] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const page_ = process.env.PAGE || 'dev.html', W = +(process.env.W || 2560), H = +(process.env.H || 1440);
const HEADED = !!process.env.HEADED;
const b = await puppeteer.launch({ executablePath: process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: HEADED ? false : 'new', protocolTimeout: 600000, defaultViewport: null,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPUDeveloperFeatures', '--enable-dawn-features=allow_unsafe_apis', HEADED ? `--window-size=${W / 2},${H / 2 + 90}` : `--window-size=${W},${H}`, '--disable-background-timer-throttling', '--disable-renderer-backgrounding', ...(HEADED ? [] : ['--disable-gpu-vsync', '--disable-frame-rate-limit'])] });
const p = await b.newPage(); await p.setViewport({ width: HEADED ? W / 2 : W, height: HEADED ? H / 2 : H, deviceScaleFactor: HEADED ? 2 : 1 });
const logs = [];
p.on('console', (m) => { const t = m.text(); logs.push(t); if (!/^\[DOM\]/.test(t) && !/^Error while parsing|^ - While/.test(t)) console.log('  >', t.split('\n')[0].slice(0, 300)); });
p.on('pageerror', (e) => console.log('  PAGEERROR', e.message));
const url = `${new URL('../', import.meta.url)}${page_}${process.env.Q ? '?' + process.env.Q : ''}`;
const t0 = Date.now();
await p.goto(url);
await p.waitForFunction(() => document.body.classList.contains('ready') || document.body.className === 'e', { timeout: 300000 });
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
if (await p.evaluate(() => document.body.className === 'e')) { console.log('FAILED:', await p.evaluate(() => document.getElementById('s').textContent)); await b.close(); process.exit(1); }
await p.evaluate(() => { document.getElementById('o').style.display = 'none'; });
await new Promise((r) => setTimeout(r, 1500));
const shot = async (name) => { await p.screenshot({ path: `${out}/${name}.png`, type: 'png' }); };
if (what === 'poses') {
  const list = rest.length ? rest.map(Number) : [0, 1, 2, 3, 4, 5];
  for (const i of list) { await p.evaluate((i) => window.__h.pose(i, 8), i); await new Promise((r) => setTimeout(r, 400)); await p.evaluate((i) => window.__h.pose(i, 8), i); await shot(`pose${i + 1}`); console.log('pose', i + 1); }
} else if (what === 'fly') {
  // fly: t0 t1 fps -> consecutive frames
  const [a = 0, z = 60, fps = 2] = rest.map(Number);
  let k = 0;
  for (let t = a; t <= z + 1e-6; t += 1 / fps) { await p.evaluate((t) => window.__h.fly(t), t); await p.evaluate((t) => window.__h.fly(t), t); await shot(`f${String(k++).padStart(4, '0')}`); }
  console.log('frames', k);
} else if (what === 'cam') {
  const v = rest.map(Number); await p.evaluate((v) => window.__h.cam(...v), v); await p.evaluate((v) => window.__h.cam(...v), v); await shot(process.env.NAME || 'cam');
  console.log(JSON.stringify(await p.evaluate(() => window.__h.info())).slice(0, 3000));
} else if (what === 'cams') {
  const C = JSON.parse(process.env.CAMS);
  for (let i = 0; i < C.length; i++) { await p.evaluate((v) => window.__h.cam(...v), C[i]); await p.evaluate((v) => window.__h.cam(...v), C[i]); await shot('cam' + i); }
} else if (what === 'burst') {
  // burst: t0 n fps -> consecutive frames named b<t0>_<k>
  const [a, n, fps] = rest.map(Number);
  for (let k = 0; k < n; k++) { const t = a + k / fps; await p.evaluate((t) => window.__h.fly(t), t); await p.evaluate((t) => window.__h.fly(t), t); await shot(`b${a}_${String(k).padStart(2, '0')}`); }
} else if (what === 'prof') {
  // prof: per pose, GPU ms with all on, then with each group skipped
  const M = { all: 0, solids: 1, leaves: 2, fx: 4, water: 8, fog: 16, bloom: 32, mainOnly: 56, none: 63 };
  const res = {};
  for (const i of (rest.length ? rest.map(Number) : [0, 1, 2, 3, 4, 5])) { res[i + 1] = {}; for (const [k, m] of Object.entries(M)) res[i + 1][k] = await p.evaluate((i, m) => window.__h.prof(i, m), i, m); console.log('pose', i + 1, JSON.stringify(res[i + 1])); }
  writeFileSync(`${out}/prof.json`, JSON.stringify(res, null, 1));
} else if (what === 'counts') {
  for (const i of [0, 1, 2, 3, 4, 5]) console.log('pose', i + 1, await p.evaluate((i) => window.__h.counts(i), i));
} else if (what === 'proffly') {
  const o = []; for (let t = 0; t <= 60; t += 4) { const v = await p.evaluate((i) => window.__h.prof(i, 0, 260), 100 + t); o.push([t, v]); console.log('t', t, v); }
  writeFileSync(`${out}/proffly.json`, JSON.stringify(o));
} else if (what === 'bench') {
  await p.evaluate(() => window.__h.free());
  await new Promise((r) => setTimeout(r, 3000));
  const r = await p.evaluate(() => window.__h.bench());
  console.log(JSON.stringify(r, null, 1));
  writeFileSync(`${out}/bench.json`, JSON.stringify(r, null, 1));
}
writeFileSync(`${out}/console.log`, logs.join('\n'));
await b.close();

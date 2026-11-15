// Bundles src/*.js into the single self-contained ../index.html
//   node build.mjs            minified  -> ../index.html
//   node build.mjs --readable unminified -> ../dev.html
import { readFileSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { transformSync } from 'esbuild';

const here = path.dirname(fileURLToPath(import.meta.url));
const FILES = ['math.js', 'shaders.js', 'world.js', 'geo.js', 'main.js', 'bench.js'];
const readable = process.argv.includes('--readable');
const out = readable ? 'dev.html' : 'index.html';
let js = FILES.map((f) => readFileSync(path.join(here, 'src', f), 'utf8')).join('\n');
const tpl = readFileSync(path.join(here, 'template.html'), 'utf8');
const wrap = (b) => `(async()=>{${b}\n})().catch(e=>{console.error(e);document.body.className='e';document.getElementById('s').textContent='Failed to start: '+e.message});`;
let html;
if (readable) html = tpl.replace('/*SCRIPT*/', () => wrap(js));
else {
  const minWgsl = (s) => s.replace(/\/\/[^\n]*/g, '').replace(/\s+/g, ' ').replace(/ ?([{}()\[\];,:=+\-*\/<>&|!%@^~?]) ?/g, '$1').trim();
  js = js.replace(/\/\*wgsl\*\/`([^`]*)`/g, (_, w) => '`' + minWgsl(w) + '`');
  const code = transformSync(wrap(js), { minify: true, target: 'es2022', legalComments: 'none' }).code.trim();
  const css = (_, s) => '<style>' + transformSync(s, { loader: 'css', minify: true }).code.trim();
  html = tpl.replace(/<style>([^]*?)(?=<\/style>)/, css).replace(/>\s+</g, '><').replace('/*SCRIPT*/', () => code).replace(/<script>\s+/, '<script>').replace(/\s+<\/script>/, '</script>');
}
writeFileSync(path.join(here, '..', out), html);
const gz = gzipSync(Buffer.from(html), { level: 9 }).length;
console.log(`${out}: ${(html.length / 1024).toFixed(1)} kB (${html.length} B), ${(gz / 1024).toFixed(1)} kB gzipped`);

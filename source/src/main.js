// ============================================================================
// renderer, world assembly, camera and input
// ============================================================================
const $ = (id) => document.getElementById(id);
const canvas = $('c'), info = $('i'), body = document.body;
const QS = new URLSearchParams(location.search);
const SCALE = +QS.get('scale') || 1;
const say = (m) => new Promise((r) => { $('s').textContent = m; setTimeout(r, 16); });
const T0 = performance.now(), stamp = (m) => console.log(`[${((performance.now() - T0) / 1000).toFixed(2)}s] ${m}`);
if (!navigator.gpu) throw new Error('this scene needs WebGPU (current Chrome or Edge)');
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('no WebGPU adapter');
const hasTS = adapter.features.has('timestamp-query') && !QS.has('nots');
const device = await adapter.requestDevice({
  requiredFeatures: hasTS ? ['timestamp-query'] : [],
  requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
});
device.addEventListener('uncapturederror', (e) => console.error('GPU: ' + e.error.message));
device.lost.then((l) => { if (l.reason !== 'destroyed') { body.className = 'e'; $('s').textContent = 'The graphics device was lost: ' + l.message; } });
const ctx = canvas.getContext('webgpu'), FMT = navigator.gpu.getPreferredCanvasFormat();
const Q = device.queue, HDR = 'rgba16float', DEPTH = 'depth32float', AA = 4, SHADOW = 8192;
const TU = GPUTextureUsage, BU = GPUBufferUsage, SS = GPUShaderStage, VF = SS.VERTEX | SS.FRAGMENT;
const mod = (code, label) => { const m = device.createShaderModule({ code, label }); m.getCompilationInfo().then((i) => { for (const x of i.messages) if (x.type === 'error') console.error(label, x.lineNum, x.message); }); return m; };
const buf = (data, usage) => { const b = device.createBuffer({ size: (data.byteLength + 3) & ~3, usage: usage | BU.COPY_DST }); Q.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength); return b; };
const tex = (w, h, format, usage, o = {}) => device.createTexture({ size: [w, h, o.d || 1], format, usage, sampleCount: o.s || 1, dimension: o.d ? '3d' : '2d' });

// ---------------------------------------------------------------- terrain
await say('Raising the karst…');
buildGround();
for (const c of CARVE) { c[1] += gAt(GH, c[0], c[2]); c[4] += gAt(GH, c[3], c[5]); }
let D;
{
  const m = mod(W_DENS, 'dens'), p = device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
  const size = NX * NY * NZ * 4;
  const bg = buf(GH, BU.STORAGE), bk = buf(karstBuf(), BU.STORAGE), bc = buf(carveBuf(), BU.STORAGE);
  const bd = device.createBuffer({ size, usage: BU.STORAGE | BU.COPY_SRC }), br = device.createBuffer({ size, usage: BU.MAP_READ | BU.COPY_DST });
  const g = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [bg, bd, bk, bc].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const e = device.createCommandEncoder(), c = e.beginComputePass();
  c.setPipeline(p); c.setBindGroup(0, g); c.dispatchWorkgroups(NX / 4, NY / 4, NZ / 4); c.end();
  e.copyBufferToBuffer(bd, 0, br, 0, size);
  Q.submit([e.finish()]);
  await br.mapAsync(GPUMapMode.READ);
  D = new Float32Array(br.getMappedRange().slice(0));
  br.unmap(); br.destroy(); bd.destroy(); bg.destroy(); bk.destroy(); bc.destroy();
}
stamp('density');
await say('Carving canyons…');
const SN = surfaceNets(D);
TOP = SN.TOP;
stamp(`surface nets: ${SN.nv} verts, ${SN.Q.length / 4} quads`);
// camera collision: signed distance to rock, quantised, same grid as the density
const COL = new Int8Array(D.length);
for (let i = 0; i < D.length; i++) COL[i] = clamp(round(D[i] * 30), -127, 127);
const dAt = (x, y, z) => { // trilinear density (metres, > 0 solid)
  const fx = clamp((x - OX) / VS, 0, NX - 1.001), fy = clamp((y - OY) / VS, 0, NY - 1.001), fz = clamp((z - OZ) / VS, 0, NZ - 1.001);
  const i = fx | 0, j = fy | 0, k = fz | 0, u = fx - i, v = fy - j, w = fz - k, b = i + NX * (j + NY * k), SXY = NX * NY;
  const c00 = mix(COL[b], COL[b + 1], u), c10 = mix(COL[b + NX], COL[b + NX + 1], u), c01 = mix(COL[b + SXY], COL[b + SXY + 1], u), c11 = mix(COL[b + SXY + NX], COL[b + SXY + NX + 1], u);
  return mix(mix(c00, c10, v), mix(c01, c11, v), w) / 30;
};
const DSN = (x, y, z) => D[clamp(round((x - OX) / VS), 0, NX - 1) + NX * (clamp(round((y - OY) / VS), 0, NY - 1) + NY * clamp(round((z - OZ) / VS), 0, NZ - 1))];

// ---------------------------------------------------------------- camera path and viewpoints
// ground under the camera: highest ground and water within a radius, so the path floats over ravines and pools
const camGround = (x, z, r = 4) => { let h = -1e9; for (let a = 0; a < 8; a++) { const px = x + cos(a * 0.785) * r, pz = z + sin(a * 0.785) * r; h = max(h, gy(px, pz)); } return max(h, gy(x, z)); };
// [x, z, height above ground, look x, look z, look height above ground]
const FLYP = [
  [-6, 20, 1.8, -8, -30, 4.5], [-3, 6, 2.1, -6, -30, 3.5], [-3, -12, 2.4, 0, -40, 3], [1, -24, 2.6, 13, -34, 3],
  [5, -42, 3.2, 4, -60, 3], [-4, -60, 3.4, -6, -82, 3], [-4, -82, 3.8, 2, -100, 4], [3, -102, 4.5, 6, -130, -4],
  [-6, -124, 3, -40, -144, -6], [-30, -138, -5, -58, -143, -6], [-58, -143, -6.5, -80, -140, -2],
  [-78, -126, 3, -81.5, -96, 3], [-78, -96, 2.5, -94.5, -72, 0], [-81.5, -84.5, 0, -107.5, -59.6, 0], [-107.5, -59.6, 0, -112, -30, 3], [-112, -36, 3, -100, -10, 3],
  [-100, -18, 4, -72, 18, 4], [-72, 18, 4, -45, 50, 5], [-42, 50, 5, -10, 70, 4], [-10, 70, 4, 22, 80, 3],
  [22, 80, 3, 48, 92, 3], [46, 90, 3, 72, 108, 10], [60, 99, 2.4, 77, 111, 14],
];
const POSES = [
  [-6, 20, 1.8, -8, -30, 4.5], [-1, -27, 1.7, 13, -33, 3.8], [60, 94, 3.6, 76, 110, 7],
  [120, -60, 1.6, 123, -52, 40], [-22, -126, 2.5, -58, -143, -2], [-85, -81, 1.8, -94.5, -72, 0],
];
const overRav = (x, z) => sInfo(x, z, 3) < RAV_RIM + 4;
const flyY = (x, z, h) => (h === 0 ? CARVE[0][1] - 0.5 : camGround(x, z, overRav(x, z) ? 20 : 6) + h);
const FLY = new Float32Array(FLYP.length * 6);
FLYP.forEach((p, i) => FLY.set([p[0], flyY(p[0], p[1], p[2]), p[1], p[3], camGround(p[3], p[4], overRav(p[3], p[4]) ? 20 : 2) + p[5], p[4]], i * 6));
for (const p of POSES) { p[2] = camGround(p[0], p[1], 2) + p[2]; p[5] = p[5] === 0 ? CARVE[0][1] : camGround(p[3], p[4], 2) + p[5]; }
POSES[3][5] = POSES[3][2] + 40;
for (let i = 0; i < FLYP.length - 1; i++) for (let k = 0; k < 8; k++) { const t = k / 8; FLY_CLEAR.push([mix(FLY[i * 6], FLY[i * 6 + 6], t), 0, mix(FLY[i * 6 + 2], FLY[i * 6 + 8], t)]); }
for (const p of POSES) for (let k = 0; k < 6; k++) FLY_CLEAR.push([p[0] + (p[3] - p[0]) * k * 0.05, 0, p[1] + (p[4] - p[1]) * k * 0.05]);

// ---------------------------------------------------------------- world content
await say('Growing the forest…');
const geo = new Geo();
{ // terrain into the static geometry
  const P = SN.P, N = SN.N, base = geo.nv;
  for (let i = 0; i < SN.nv; i++) {
    const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2], nx = N[i * 3], ny = N[i * 3 + 1], nz = N[i * 3 + 2];
    const [kd] = karstDist(x, z);
    const wd = waterDist(x, z), lev = sInfo(x, z, 1);
    let wet = y < lev + 1.2 ? sstep(3, 0, wd) : 0;
    wet = max(wet, 0.6 * sstep(14, 4, hypot(x - FALL_MAIN.x, z - FALL_MAIN.z)), sstep(8, 0, sInfo(x, z, 3) - RAV_HW) * sstep(-15, -25, y));
    const occ = DSN(x + nx * 2.5, y + ny * 2.5, z + nz * 2.5);
    const ao = clamp(1 - max(occ + 2.5, 0) / 4, 0.35, 1);
    geo.setMat(0, kd < 1.5 ? 1 : 0, sstep(0.3, 0.8, vn2(x * 0.05, z * 0.05)) * 0.3, wet).setSway(0, 0, 0, 0);
    geo.v(x, y, z, nx, ny, nz, ao);
  }
  const Qd = SN.Q;
  for (let q = 0; q < Qd.length; q += 4) { const a = Qd[q] + base; geo.at(P[(a - base) * 3], P[(a - base) * 3 + 2]); geo.quad(a, Qd[q + 1] + base, Qd[q + 2] + base, Qd[q + 3] + base); }
}
D = null;
stamp('terrain geo');
const rr = RNG(1234);
const nearPose = (x, z, r) => { for (const p of POSES) if (p !== POSES[3]) if (hypot(p[0] - x, p[1] - z) < r) return true; return false; };
const inGap = (x, z, pad) => { if (nearPose(x, z, 6 + pad)) return true; if (hypot(x - POOLS[0][0], z - POOLS[0][1]) < 24 + pad) return true; for (const g of GAPS) if (hypot(x - g[0], z - g[1]) < g[2] + pad) return true; return false; };
const trunkFree = (x, z, r) => { for (const t of TREES) if (hypot(t[0] - x, t[1] - z) < t[2] + r) return false; return true; };
const okGround = (x, z, o = {}) => {
  if (abs(x) > (o.lim ?? 248) || abs(z) > (o.lim ?? 248)) return false;
  if (waterDist(x, z) < (o.water ?? 2)) return false;
  if (sInfo(x, z, 3) < RAV_RIM + (o.rav ?? 3)) return false;
  if (slope(x, z) > (o.slope ?? 0.35)) return false;
  if (hypot(x - SHRINE.x, z - SHRINE.z) < (o.shrine ?? 12)) return false;
  if (abs(x - BRIDGE.x) < 4 && z < BRIDGE.z0 + 4 && z > BRIDGE.z1 - 4) return false;
  return true;
};
// set pieces first (so trees keep clear of them)
shrine(geo, RNG(7));
banyan(geo, BANYAN_SHRINE[0], BANYAN_SHRINE[1], RNG(8), { limbA: [atan2(SHRINE.z - BANYAN_SHRINE[1], SHRINE.x - BANYAN_SHRINE[0]) - 0.2, atan2(SHRINE.z - BANYAN_SHRINE[1], SHRINE.x - BANYAN_SHRINE[0]) + 0.5] });
bridge(geo, RNG(9));
// a hero dipterocarp and a bamboo grove in the opening views
dipterocarp(geo, -21, -2, RNG(31), 1.15);
bambooGrove(geo, 14, 2, 4.5, RNG(32));
POSES[4][0] = -30; POSES[4][1] = -141; POSES[4][2] = BRIDGE_Y + 3 - 2; POSES[4][3] = -58; POSES[4][4] = -143; POSES[4][5] = BRIDGE_Y - 1.5;
for (const [x, z] of [[-74, -24], [64, -6], [-34, 84], [104, 54], [-128, 36], [30, 150], [-150, -110], [140, -100], [-20, -170], [-200, 40], [190, 10], [80, -110]]) {
  let px = x, pz = z;
  for (let k = 0; k < 20 && !(okGround(px, pz, { water: 6, slope: 0.3 }) && !nearFly(px, pz, 6) && trunkFree(px, pz, 6)); k++) { px = x + rr.n() * 12; pz = z + rr.n() * 12; }
  if (okGround(px, pz, { water: 6, slope: 0.3 }) && !nearFly(px, pz, 6)) banyan(geo, px, pz, rr);
}
stamp('set pieces');
// emergent dipterocarps
for (let gz = -250; gz < 250; gz += 19) for (let gx = -250; gx < 250; gx += 19) {
  const x = gx + rr.r(0, 19), z = gz + rr.r(0, 19);
  if (!okGround(x, z, { water: 4, slope: 0.32 }) || inGap(x, z, 9) || hypot(x - CLEAR.x, z - CLEAR.z) < CLEAR.r + 10 || nearFly(x, z, 4) || !trunkFree(x, z, 5)) continue;
  if (karstDist(x, z)[0] < 0) { if (rr() < 0.5) canopyTree(geo, x, z, rr, rr.r(10, 16)); continue; }
  if (rr() < 0.82) dipterocarp(geo, x, z, rr);
}
stamp('emergents');
// main canopy and understory
for (let gz = -250; gz < 250; gz += 9.5) for (let gx = -250; gx < 250; gx += 9.5) {
  const x = gx + rr.r(0, 9.5), z = gz + rr.r(0, 9.5);
  if (!okGround(x, z, { water: 2.5, slope: 0.45 }) || inGap(x, z, 3) || hypot(x - CLEAR.x, z - CLEAR.z) < CLEAR.r + 3 || nearFly(x, z, 3) || !trunkFree(x, z, 1.5)) continue;
  const big = rr() < 0.55;
  canopyTree(geo, x, z, rr, big ? rr.r(22, 34) : rr.r(9, 17), { broad: !big && rr() < 0.6 });
}
stamp('canopy');
for (let gz = -246; gz < 246; gz += 9) for (let gx = -246; gx < 246; gx += 9) {
  const x = gx + rr.r(0, 9), z = gz + rr.r(0, 9);
  if (!okGround(x, z, { water: 1.5, slope: 0.5, rav: 1 }) || nearFly(x, z, 2) || !trunkFree(x, z, 1) || rr() < 0.45) continue;
  palm(geo, x, z, rr, true); TREES.push([x, z, 0.4, 5]);
}
stamp('palms');
// bamboo groves
for (const [x, z, r] of [[34, 8, 5], [-40, -2, 4.5], [38, -62, 5], [-30, 56, 4], [62, 64, 5], [-92, -122, 4], [96, -60, 5], [-120, 70, 5], [20, 104, 4], [-62, -60, 4], [150, 110, 5], [-170, -70, 6], [90, 160, 5], [-30, 130, 4], [120, -10, 4]]) {
  if (okGround(x, z, { water: 1, slope: 0.6 }) && !nearFly(x, z, r + 2)) bambooGrove(geo, x, z, r, rr);
}
stamp('bamboo');
// hanging curtains of roots and vines under the crowns
for (const c of CLUMPS) if (rr() < 0.015 && c[1] > 12 && !nearFly(c[0], c[2], 3)) hangCurtain(geo, [c[0], c[1] - c[3] * 0.5, c[2]], rr, rr());
// lianas between crowns
for (let i = 0; i < 260; i++) {
  const a = CLUMPS[rr.i(CLUMPS.length)], b = CLUMPS[rr.i(CLUMPS.length)];
  const d = hypot(a[0] - b[0], a[2] - b[2]);
  if (d < 8 || d > 26 || a[1] < 15 || b[1] < 15) continue;
  const n = 24, P = [], sag = d * rr.r(0.25, 0.6), A = [a[0], a[1] - a[3] * 0.5, a[2]], B = [b[0], b[1] - b[3] * 0.5, b[2]];
  let low = 1e9;
  for (let k = 0; k <= n; k++) { const f = k / n, p = vlerp(A, B, f); p[1] -= sag * 4 * f * (1 - f); P.push(p); low = min(low, p[1] - gy(p[0], p[2])); }
  if (low < 3 || nearFly(P[12][0], P[12][2], 2.5) && P[12][1] - gy(P[12][0], P[12][2]) < 8) continue;
  geo.at(P[12][0], P[12][2]).setMat(8, rr(), rr() * 0.5, 0);
  const r0 = rr.r(0.03, 0.09), ph = rr();
  geo.tube(P, (k, th) => r0 * (1 + 0.3 * sin(th * 2 + k)), 5, { sway: (k) => geo.setSway(0.6 * sin(PI * k / n), ph, 0.2 * sin(PI * k / n), 0) });
}
stamp('lianas');
// rocks: stream banks, karst talus, scattered and enormous boulders
for (let i = 0; i < STREAM.length; i += 1) {
  const [x, z] = STREAM[i];
  if (rr() < 0.55) { const a = rr() * TAU, hw = sInfo(x, z, 2), d = hw + rr.r(-0.5, 1.8); boulder(geo, x + cos(a) * d, z + sin(a) * d, rr.r(0.35, 1.3), rr, { moss: rr() * 0.5, sink: 0.25 }); }
  if (rr() < 0.15) boulder(geo, x + rr.n() * 1.5, z + rr.n() * 1.5, rr.r(0.25, 0.6), rr, { moss: 0.1, sink: 0.4 });
}
for (const k of KARST) {
  const L = hypot(k[2] - k[0], k[3] - k[1]) + k[4] * TAU, n = round(L / 6);
  for (let i = 0; i < n; i++) {
    const a = rr() * TAU, t = rr(), cx = mix(k[0], k[2], t), cz = mix(k[1], k[3], t), d = k[4] * 1.15 + rr.r(0, 9);
    const x = cx + cos(a) * d, z = cz + sin(a) * d;
    if (abs(x) > 250 || abs(z) > 250 || waterDist(x, z) < 1 || nearFly(x, z, 3)) continue;
    boulder(geo, x, z, rr.r(0.8, 3.5), rr, { moss: 0.4 + rr() * 0.5 });
  }
}
for (let gz = -240; gz < 240; gz += 12) for (let gx = -240; gx < 240; gx += 12) {
  const x = gx + rr.r(0, 12), z = gz + rr.r(0, 12);
  if (rr() > 0.3 || !okGround(x, z, { water: 1, slope: 0.6 }) || nearFly(x, z, 2.5) || !trunkFree(x, z, 1)) continue;
  boulder(geo, x, z, rr.r(0.4, 2.2), rr, { moss: 0.5 + rr() * 0.5 });
}
for (const [x, z, r] of [[-28, -40, 6], [24, -16, 5], [-52, 8, 7], [40, 40, 5.5], [-70, -70, 6], [70, 84, 5], [96, 96, 7], [-120, -10, 6], [-14, 40, 4.5], [-110, -120, 6]]) {
  if (okGround(x, z, { water: 3, slope: 0.6, shrine: 9 }) && !nearFly(x, z, r + 2)) { boulder(geo, x, z, r, rr, { moss: 0.75, squash: 0.7, sink: 0.25 }); TREES.push([x, z, r, 0]); }
}
stamp('rocks');
{ // plants clinging to ledges of the karst and the ravine walls, vines hanging over the edges
  const r3 = RNG(55), P = SN.P, N = SN.N;
  for (let i = 0; i < SN.nv; i++) {
    const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2], ny = N[i * 3 + 1];
    if (ny < 0.15 || abs(x) > 245 || abs(z) > 245) continue;
    const kd = karstDist(x, z)[0], rav = sInfo(x, z, 3) < RAV_RIM + 2 && y < gy(x, z) + 1;
    if (kd > 2 && !rav) continue;
    if (y < gy(x, z) - 1.5 && r3() < 0.5) continue;
    const q = r3();
    if (q < 0.09 && ny > 0.4) shrub([x, y - 0.2, z], r3, 0.7);
    else if (q < 0.2) fernPlant([x, y - 0.05, z], r3, 0.7);
    else if (q < 0.16) herbs([x, y, z], r3, [0.16, 0.34, 0.07]);
    if (ny < 0.75 && r3() < 0.006) hangCurtain(geo.at(x, z), [x + N[i * 3] * 0.5, y, z + N[i * 3 + 2] * 0.5], r3, r3());
  }
}
stamp('ledges');
// fallen trees
for (let i = 0; i < 26; i++) {
  const x = rr.r(-220, 220), z = rr.r(-220, 220);
  if (!okGround(x, z, { water: 1, slope: 0.35 }) || nearFly(x, z, 7)) continue;
  fallenLog(geo, x, z, rr() * TAU, rr.r(10, 22), rr.r(0.45, 1.0), rr);
}
fallenLog(geo, -9, -45, 0.35, 16, 0.75, rr); // a log over the stream north of the clearing
// mushrooms at tree bases
for (const t of TREES) if (rr() < 0.25 && t[3] > 0) { const a = rr() * TAU; mushrooms(geo.at(t[0], t[1]), [t[0] + cos(a) * (t[2] - 0.2), gy(t[0], t[1]) + 0.05, t[1] + sin(a) * (t[2] - 0.2)], rr); }
stamp('logs');
// forest floor: ferns, broad-leaved herbs, seedlings, flowers, wild bananas
{
  const r2 = RNG(99);
  for (let gz = -248; gz < 248; gz += 1.7) for (let gx = -248; gx < 248; gx += 1.7) {
    const x = gx + r2() * 1.7, z = gz + r2() * 1.7;
    const dens = 0.7 + 0.35 * vn2(x * 0.05, z * 0.05);
    if (r2() > dens || !okGround(x, z, { water: 0.4, slope: 0.62, rav: -2, shrine: 4.5 })) continue;
    const dc = hypot(x - CLEAR.x, z - CLEAR.z); if (dc < CLEAR.r * 0.55 && r2() < 0.8) continue;
    if (nearFly(x, z, 1.2) && r2() < 0.7) continue;
    fernPlant([x, gy(x, z) - 0.05, z], r2, 0.8 + r2() * 0.6);
  }
  stamp('ferns');
  for (let gz = -248; gz < 248; gz += 2.6) for (let gx = -248; gx < 248; gx += 2.6) {
    const x = gx + r2() * 2.6, z = gz + r2() * 2.6;
    const wet = sstep(10, 1, waterDist(x, z));
    if (r2() > 0.32 + wet * 0.4 || !okGround(x, z, { water: 0.4, slope: 0.55, shrine: 4.5 })) continue;
    if (nearFly(x, z, 1.5) && r2() < 0.6) continue;
    broadPlant([x, gy(x, z) - 0.05, z], r2, r2() < 0.45 + wet * 0.3 ? 1 : 0, 0.7 + r2() * 0.8);
  }
  stamp('broad');
  for (let gz = -248; gz < 248; gz += 3.4) for (let gx = -248; gx < 248; gx += 3.4) {
    const x = gx + r2() * 3.4, z = gz + r2() * 3.4;
    const dens = 0.45 + 0.4 * vn2(x * 0.03 + 9, z * 0.03 - 4);
    if (r2() > dens || !okGround(x, z, { water: 1.2, slope: 0.6, shrine: 6 }) || hypot(x - CLEAR.x, z - CLEAR.z) < CLEAR.r * 0.8) continue;
    if (nearFly(x, z, 2.2)) continue;
    shrub([x, gy(x, z), z], r2, 0.8 + r2() * 0.6);
  }
  stamp('shrubs');
  for (let gz = -246; gz < 246; gz += 1.0) for (let gx = -246; gx < 246; gx += 1.0) {
    const x = gx + r2(), z = gz + r2();
    if (r2() > 0.8 || !okGround(x, z, { water: 0.2, slope: 0.7, rav: -6, shrine: 3.6 })) continue;
    const c = r2() < 0.15 ? [0.32, 0.42, 0.1] : r2() < 0.5 ? [0.16, 0.34, 0.07] : [0.12, 0.27, 0.06];
    herbs([x, gy(x, z) - 0.03, z], r2, c);
  }
  stamp('herbs');
  // flowers where light reaches: the clearing, stream banks, canopy gaps
  const FL = [[0.95, 0.3, 0.55, 0], [0.98, 0.92, 0.85, 0], [0.7, 0.3, 0.9, 0], [0.95, 0.45, 0.1, 1], [0.9, 0.12, 0.15, 1], [1.0, 0.8, 0.2, 0]];
  for (let i = 0; i < 1000; i++) {
    let x, z;
    const m = r2();
    if (m < 0.3) { const a = r2() * TAU, d = sqrt(r2()) * (CLEAR.r + 4); x = CLEAR.x + cos(a) * d; z = CLEAR.z + sin(a) * d; }
    else if (m < 0.65) { const s = STREAM[r2.i(STREAM.length)], a = r2() * TAU, d = sInfo(s[0], s[1], 2) + 1 + r2() * 4; x = s[0] + cos(a) * d; z = s[1] + sin(a) * d; }
    else { const g = GAPS[r2.i(GAPS.length)], a = r2() * TAU, d = sqrt(r2()) * g[2]; x = g[0] + cos(a) * d; z = g[1] + sin(a) * d; }
    if (!okGround(x, z, { water: 0.5, slope: 0.5, shrine: 4 })) continue;
    flowers([x, gy(x, z), z], r2, FL[r2.i(FL.length)], 1 + r2.i(4));
  }
  for (let i = 0; i < 0; i++) {
    let x, z;
    if (r2() < 0.5) { const s = STREAM[r2.i(STREAM.length)], a = r2() * TAU, d = sInfo(s[0], s[1], 2) + 2 + r2() * 6; x = s[0] + cos(a) * d; z = s[1] + sin(a) * d; }
    else { const g = r2() < 0.3 ? [CLEAR.x, CLEAR.z, CLEAR.r + 5] : GAPS[r2.i(GAPS.length)], a = r2() * TAU, d = (0.6 + 0.5 * r2()) * g[2]; x = g[0] + cos(a) * d; z = g[1] + sin(a) * d; }
    if (!okGround(x, z, { water: 1.2, slope: 0.4, shrine: 6 }) || nearFly(x, z, 2) || !trunkFree(x, z, 1)) continue;
    banana(geo, x, z, r2);
  }
  stamp('flowers, bananas');
}
// water
{
  const hw = (i) => sInfo(STREAM[i][0], STREAM[i][1], 2);
  waterRibbon(STREAM, SLEV, hw, { pools: POOLS });
  // pools: round discs at the local water level, under the banks
  for (const [px, pz, pr] of POOLS) {
    const lv = SLEV[nearPoly(STREAM, px, pz)[1]];
    stamp(`pool ${px},${pz}: water ${lv.toFixed(2)} ground ${gy(px, pz).toFixed(2)} edge ${gy(px + pr, pz).toFixed(2)} ${gy(px - pr, pz).toFixed(2)} ${gy(px, pz + pr).toFixed(2)} ${gy(px, pz - pr).toFixed(2)}`);
    const c = wv(px, lv - 0.01, pz, 0.05, 0.05, 0, 0), n = 40, R = pr + 1.5;
    for (let k = 0; k < n; k++) { const a0 = (k / n) * TAU, a1 = ((k + 1) / n) * TAU; const v0 = wv(px + cos(a0) * R, lv - 0.01, pz + sin(a0) * R, 0.05, 0.05, 0, 0), v1 = wv(px + cos(a1) * R, lv - 0.01, pz + sin(a1) * R, 0.05, 0.05, 0, 0); WIDX.need(3); WIDX.a.set([c, v1, v0], WIDX.n); WIDX.n += 3; }
  }
  // the stream falls over the ravine rim at its end
  const e = STREAM[STREAM.length - 1], e0 = STREAM[STREAM.length - 3], d = vnorm([e[0] - e0[0], 0, e[1] - e0[1]]);
  waterfall([e[0] + d[0] * 0.3, SLEV[STREAM.length - 1] + 0.02, e[1] + d[2] * 0.3], RAV_FLOOR - 0.4, d, 2.6, 4.5);
  // ravine floor stream: levels follow the floor, always falling eastward
  const RL = new Float32Array(RAVINE.length); let lv = 1e9;
  for (let i = 0; i < RAVINE.length; i++) { const [x, z] = RAVINE[i]; lv = min(lv - 0.01, gy(x, z) + 0.25); RL[i] = lv; }
  waterRibbon(RAVINE, RL, () => 2.2, { speed: 0.5 });
  // the main fall from the ridge face into the pool
  const lip = [FALL_MAIN.x, 0, FALL_MAIN.z], pool = POOLS[0];
  const dirF = vnorm([pool[0] - lip[0], 0, pool[1] - lip[2]]);
  let ly = SLEV[0] + 18;
  waterfall([lip[0], ly, lip[2]], SLEV[0] - 0.2, dirF, 3.2, 8.5);
  FLY_CLEAR.push(lip);
}
stamp('water');

// ---------------------------------------------------------------- creatures, particles (storage data)
const FX = new Float32Array(4096 * 4);
for (let i = 0; i < min(FALLS.length, 8); i++) FX.set(FALLS[FALLS.length - 1 - i], (64 + i) * 4);
// bird perches sorted spatially so a territory is a run of nearby perches
PERCH.sort((a, b) => (floor((a[2] + 260) / 40) * 20 + floor((a[0] + 260) / 40)) - (floor((b[2] + 260) / 40) * 20 + floor((b[0] + 260) / 40)));
const nP = min(PERCH.length, 512);
// keep the perches near the action (camera path), spread over the map
const perchSel = []; { const step = max(1, floor(PERCH.length / 512)); for (let i = 0; i < PERCH.length && perchSel.length < 512; i += step) perchSel.push(PERCH[i]); }
perchSel.forEach((p, i) => FX.set([p[0], p[1], p[2], 0], (1024 + i) * 4));
const HORN = [[-60, 34, -142, 70], [10, 48, -150, 60], [-17, 45, -38, 35], [75, 52, 100, 45], [-100, 46, -120, 60], [60, 50, -140, 70]];
HORN.forEach((h, i) => FX.set(h, (1600 + i) * 4));
const BFLY = []; for (let i = 0; i < 64; i++) { const m = i % 3; let x, z; if (m === 0) { x = CLEAR.x + rr.n() * 10; z = CLEAR.z + rr.n() * 10; } else if (m === 1) { const s = STREAM[rr.i(STREAM.length)]; x = s[0] + rr.n() * 4; z = s[1] + rr.n() * 4; } else { const g = GAPS[rr.i(GAPS.length)]; x = g[0] + rr.n() * 5; z = g[1] + rr.n() * 5; } BFLY.push([x, gy(x, z) + 0.3, z, 0]); }
BFLY.forEach((b, i) => FX.set(b, (1700 + i) * 4));
for (let i = 0; i < 64; i++) { const s = STREAM[(i * 7) % STREAM.length]; FX.set([s[0], SLEV[(i * 7) % STREAM.length], s[1], 0], (1800 + i) * 4); }
let nFF = 0;
for (let i = 0; i < 4000 && nFF < 1500; i++) {
  const x = rr.r(-200, 200), z = rr.r(-200, 200); const vis = 1;
  if (!okGround(x, z, { water: 0, slope: 0.6 })) continue;
  if (rr() > 0.3 + 0.7 * sstep(12, 0, waterDist(x, z))) continue;
  FX.set([x, gy(x, z) + rr.r(0.4, 2.5), z, rr() * 100], (2048 + nFF++) * 4);
}
const NB_BIRD = 70, NB_HORN = HORN.length, NB_BFLY = 260, NB_DRAG = 48, NMOTE = 24000, NFALL = 900, NSPRAY = 1600;
const meshB = meshOut(birdMesh()), meshBf = meshOut(butterflyMesh()), meshDf = meshOut(dragonflyMesh());

// ---------------------------------------------------------------- world map texture: ground, canopy depth, canopy height, water
const MAPN = 512;
const MAP = new Float16Array(MAPN * MAPN * 4);
{
  const tau = new Float32Array(MAPN * MAPN), hs = new Float32Array(MAPN * MAPN), ws = new Float32Array(MAPN * MAPN), cs = (WH * 2) / MAPN;
  for (const [x, y, z, r] of CLUMPS) {
    const R = r * 1.05, i0 = floor((x - R + WH) / cs), i1 = ceil((x + R + WH) / cs), j0 = floor((z - R + WH) / cs), j1 = ceil((z + R + WH) / cs);
    for (let j = max(j0, 0); j <= min(j1, MAPN - 1); j++) for (let i = max(i0, 0); i <= min(i1, MAPN - 1); i++) {
      const px = (i + 0.5) * cs - WH, pz = (j + 0.5) * cs - WH, d = hypot(px - x, pz - z) / R;
      if (d >= 1) continue;
      const w = (1 - d * d) * 1.5; tau[i + j * MAPN] += w; hs[i + j * MAPN] += y * w; ws[i + j * MAPN] += w;
    }
  }
  const blur = (A) => { const B = new Float32Array(A.length); for (let p = 0; p < 2; p++) { for (let j = 0; j < MAPN; j++) for (let i = 0; i < MAPN; i++) { let s = 0; for (let k = -3; k <= 3; k++) s += A[clamp(i + k, 0, MAPN - 1) + j * MAPN]; B[i + j * MAPN] = s / 7; } for (let j = 0; j < MAPN; j++) for (let i = 0; i < MAPN; i++) { let s = 0; for (let k = -3; k <= 3; k++) s += B[i + clamp(j + k, 0, MAPN - 1) * MAPN]; A[i + j * MAPN] = s / 7; } } };
  blur(tau); blur(hs); blur(ws);
  for (let j = 0; j < MAPN; j++) for (let i = 0; i < MAPN; i++) {
    const k = i + j * MAPN, x = (i + 0.5) * cs - WH, z = (j + 0.5) * cs - WH;
    MAP[k * 4] = min(gy(x, z), gAt(GH, x, z) + 1); MAP[k * 4 + 1] = min(tau[k], 6); MAP[k * 4 + 2] = ws[k] > 0.05 ? hs[k] / ws[k] : gy(x, z) + 25; MAP[k * 4 + 3] = sstep(8, 0, waterDist(x, z));
  }
}
const canopyAt = (x, z) => { const i = clamp(floor((x + WH) / (WH * 2) * MAPN), 0, MAPN - 1), j = clamp(floor((z + WH) / (WH * 2) * MAPN), 0, MAPN - 1); return MAP[(i + j * MAPN) * 4 + 1]; };
stamp('map');
await say('Weaving the canopy…');
const FOL = buildClusters();
stamp('clusters: ' + FOL.map((f) => f.n * 64).join(', ') + ' leaf slots');

// ---------------------------------------------------------------- GPU resources
// static geometry: one vertex buffer, indices grouped by cell
const NCH = NCELL * NCELL, CH = new Float32Array(NCH * 4), CHI = new Uint32Array(NCH * 2);
let nIdx = 0; for (const g of geo.idx) nIdx += g.n;
let IDX = new Uint32Array(nIdx), VV = geo.f.view();
{
  let o = 0;
  for (let c = 0; c < NCH; c++) {
    const g = geo.idx[c]; IDX.set(g.view(), o); CHI[c * 2] = o; CHI[c * 2 + 1] = g.n; o += g.n;
    let mny = 1e9, mxy = -1e9; for (let i = 0; i < g.n; i += 3) { const y = VV[g.a[i] * 6 + 1]; if (y < mny) mny = y; if (y > mxy) mxy = y; }
    const cx = (c % NCELL + 0.5) * CELL - WH, cz = (floor(c / NCELL) + 0.5) * CELL - WH;
    // a cell's geometry can overhang its square by a crown radius
    CH[c * 4] = cx; CH[c * 4 + 1] = (mny + mxy) / 2; CH[c * 4 + 2] = cz; CH[c * 4 + 3] = hypot(CELL * 0.5 + 16, CELL * 0.5 + 16, (mxy - mny) / 2);
    geo.idx[c] = null;
  }
}
const vbStatic = buf(VV, BU.VERTEX), ibStatic = buf(IDX, BU.INDEX);
stamp(`static: ${(VV.length / 6 / 1e6).toFixed(2)}M verts, ${(nIdx / 3 / 1e6).toFixed(2)}M tris`);
geo.f = null;
const vbWater = buf(WATER.view(), BU.VERTEX), ibWater = buf(WIDX.view(), BU.INDEX), nWI = WIDX.n;
const vbBird = buf(meshB.v, BU.VERTEX), ibBird = buf(meshB.i, BU.INDEX), vbBf = buf(meshBf.v, BU.VERTEX), ibBf = buf(meshBf.i, BU.INDEX), vbDf = buf(meshDf.v, BU.VERTEX), ibDf = buf(meshDf.i, BU.INDEX);
const sbFX = buf(FX, BU.STORAGE);
// foliage
let NCL = 0; const KRNG = [];
for (let k = 0; k < NKIND; k++) { KRNG.push([NCL, NCL * 64, FOL[k].n * 64]); NCL += FOL[k].n; }
const CLD = new Float32Array(NCL * 4); FOL.forEach((f, k) => CLD.set(f.cl, KRNG[k][0] * 4));
const sbClusters = buf(CLD, BU.STORAGE), sbVis = device.createBuffer({ size: NCL * 64 * 4 + 16, usage: BU.STORAGE });
const sbInst = FOL.map((f) => buf(f.inst, BU.STORAGE));
// the GPU has its copies: drop the generator's working memory so the JS heap stays small (no long GC pauses later)
IDX = VV = null; for (const f of FOL) f.inst = f.cl = null; SN.P = SN.N = SN.Q = null; WATER.a = WIDX.a = null;
//            maxDist lodDist DFULL DMIN rowsNear
const KPRM = [[330, 38, 48, 0.3, 3], [170, 32, 34, 0.42, 5], [150, 34, 36, 0.42, 6], [62, 14, 13, 0.2, 2], [55, 12, 12, 0.3, 1]];
const FROWS = [1, 2, 2, 1, 1];
// indexed-indirect args (5 words each); leaf cards share one index buffer of ROWS x 2 quads over a (ROWS+1) x 3 vertex grid
const ARGS = new Uint32Array(NKIND * 2 * 5);
for (let k = 0; k < NKIND; k++) { ARGS[k * 10] = 12 * KPRM[k][4]; ARGS[k * 10 + 5] = 12 * FROWS[k]; }
const LIX = new Uint16Array(12 * 8);
for (let r = 0, o = 0; r < 8; r++) for (let h = 0; h < 2; h++) { const a = r * 3 + h; LIX.set([a, a + 1, a + 4, a, a + 4, a + 3], o); o += 6; }
const ibLeaf = buf(LIX, BU.INDEX);
const bArgs = device.createBuffer({ size: ARGS.byteLength, usage: BU.INDIRECT | BU.STORAGE | BU.COPY_DST | BU.COPY_SRC });
const CUB = new ArrayBuffer(384), CUF = new Float32Array(CUB), CUU = new Uint32Array(CUB);
CUU[28] = NCL; CUU[29] = NKIND;
for (let k = 0; k < NKIND; k++) { CUF[32 + k * 4] = KPRM[k][0]; CUF[32 + k * 4 + 1] = KPRM[k][1]; CUF[32 + k * 4 + 2] = KPRM[k][2]; CUF[32 + k * 4 + 3] = KPRM[k][3]; CUU[64 + k * 4] = KRNG[k][0]; CUU[64 + k * 4 + 1] = KRNG[k][1]; CUU[64 + k * 4 + 2] = KRNG[k][2]; }
const ubCull = device.createBuffer({ size: 384, usage: BU.UNIFORM | BU.COPY_DST });

// shared bindings
const sRep = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat', magFilter: 'linear', minFilter: 'linear' });
const sClamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
const sCmp = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
const tShadow = tex(SHADOW, SHADOW, 'depth16unorm', TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING);
const tNoise = tex(128, 128, 'rgba8unorm', TU.STORAGE_BINDING | TU.TEXTURE_BINDING, { d: 128 });
const tMap = tex(MAPN, MAPN, 'rgba16float', TU.TEXTURE_BINDING | TU.COPY_DST);
Q.writeTexture({ texture: tMap }, MAP, { bytesPerRow: MAPN * 8 }, [MAPN, MAPN]);
const UB = new Float32Array(96), UBS = new Float32Array(96);
const ubo = device.createBuffer({ size: 384, usage: BU.UNIFORM | BU.COPY_DST }), uboS = device.createBuffer({ size: 384, usage: BU.UNIFORM | BU.COPY_DST });
const bgl0 = device.createBindGroupLayout({ entries: [
  { binding: 0, visibility: VF, buffer: {} }, { binding: 1, visibility: VF, sampler: {} }, { binding: 2, visibility: VF, sampler: {} },
  { binding: 3, visibility: VF, sampler: { type: 'comparison' } }, { binding: 4, visibility: VF, texture: { sampleType: 'depth' } },
  { binding: 5, visibility: VF, texture: { viewDimension: '3d' } }, { binding: 6, visibility: VF, texture: {} }] });
const tDummy = tex(1, 1, 'depth16unorm', TU.TEXTURE_BINDING | TU.RENDER_ATTACHMENT);
const mkBG0 = (u, sh) => device.createBindGroup({ layout: bgl0, entries: [{ binding: 0, resource: { buffer: u } }, { binding: 1, resource: sRep }, { binding: 2, resource: sClamp }, { binding: 3, resource: sCmp }, { binding: 4, resource: sh.createView() }, { binding: 5, resource: tNoise.createView() }, { binding: 6, resource: tMap.createView() }] });
const bg0 = mkBG0(ubo, tShadow), bg0s = mkBG0(uboS, tDummy);
const L1 = (entries) => device.createBindGroupLayout({ entries });
const bglLeaf = L1([{ binding: 0, visibility: SS.VERTEX, buffer: { type: 'read-only-storage' } }, { binding: 1, visibility: SS.VERTEX, buffer: { type: 'read-only-storage' } }]);
const bglFx = L1([{ binding: 0, visibility: SS.VERTEX, buffer: { type: 'read-only-storage' } }]);
const bglWater = L1([{ binding: 0, visibility: SS.FRAGMENT, texture: {} }, { binding: 1, visibility: SS.FRAGMENT, texture: { sampleType: 'depth', multisampled: true } }]);
const bglFog = L1([{ binding: 0, visibility: SS.FRAGMENT, texture: { sampleType: 'depth', multisampled: true } }]);
const bglPost = L1([{ binding: 0, visibility: SS.FRAGMENT, texture: {} }, { binding: 1, visibility: SS.FRAGMENT, texture: {} }, { binding: 2, visibility: SS.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }, { binding: 3, visibility: SS.FRAGMENT, texture: {} }, { binding: 4, visibility: SS.FRAGMENT, texture: { sampleType: 'depth', multisampled: true } }]);
const bglBlur = L1([{ binding: 0, visibility: SS.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }, { binding: 1, visibility: SS.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }]);
const bglBloom = L1([{ binding: 0, visibility: SS.FRAGMENT, texture: {} }, { binding: 1, visibility: SS.FRAGMENT, sampler: {} }]);
const PL = (...g) => device.createPipelineLayout({ bindGroupLayouts: g });
const pl0 = PL(bgl0), plLeaf = PL(bgl0, bglLeaf), plFx = PL(bgl0, bglFx), plWater = PL(bgl0, bglWater), plFog = PL(bgl0, bglFog), plPost = PL(bgl0, bglPost), plBlur = PL(bglBlur), plBloom = PL(bglBloom);
const bgLeaf = sbInst.map((b) => device.createBindGroup({ layout: bglLeaf, entries: [{ binding: 0, resource: { buffer: b } }, { binding: 1, resource: { buffer: sbVis } }] }));
const bgFx = device.createBindGroup({ layout: bglFx, entries: [{ binding: 0, resource: { buffer: sbFX } }] });

// pipelines
await say('Compiling shaders…');
const mSolid = mod(W_SOLID, 'solid'), mLeaf = mod(W_LEAF, 'leaf'), mSky = mod(W_SKY, 'sky'), mWater = mod(W_WATER, 'water'), mFx = mod(W_FX, 'fx');
const mFog = mod(W_FOG, 'fog'), mBlur = mod(W_BLUR, 'blur'), mBloom = mod(W_BLOOM, 'bloom'), mPost = mod(W_POST, 'post'), mCull = mod(W_CULL, 'cull'), mNoise = mod(W_NOISE, 'noise');
const A = (shaderLocation, offset, format) => ({ shaderLocation, offset, format });
const vbS = { arrayStride: 24, attributes: [A(0, 0, 'float32x3'), A(1, 12, 'snorm8x4'), A(2, 16, 'unorm8x4'), A(3, 20, 'unorm8x4')] };
const vbW = { arrayStride: 28, attributes: [A(0, 0, 'float32x3'), A(1, 12, 'float32x4')] };
const ADD = { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } };
const DS = (write, cmp = 'greater') => ({ format: DEPTH, depthWriteEnabled: write, depthCompare: cmp });
const P = (o) => device.createRenderPipelineAsync(o);
const scene = (layout, m, vs, fs, buffers, o = {}) => P({ layout, vertex: { module: m, entryPoint: vs, buffers, constants: o.c },
  fragment: { module: m, entryPoint: fs, constants: o.c, targets: [{ format: HDR, blend: o.blend, writeMask: o.mask ?? GPUColorWrite.ALL }] },
  primitive: { topology: o.topo || 'triangle-list', cullMode: o.cull || 'none' }, depthStencil: DS(!o.blend && !o.noz, o.cmp), multisample: { count: AA, alphaToCoverageEnabled: !!o.a2c } });
const shadowP = (layout, m, vs, fs, buffers, c) => P({ layout, vertex: { module: m, entryPoint: vs, buffers, constants: c }, fragment: fs ? { module: m, entryPoint: fs, constants: c, targets: [] } : undefined,
  primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: { format: 'depth16unorm', depthWriteEnabled: true, depthCompare: 'less', depthBias: 3, depthBiasSlopeScale: 2.5 } });
const leafC = (k, lod) => ({ KIND: k, ROWS: lod ? FROWS[k] : KPRM[k][4], VBASE: KRNG[k][1] + lod * (KRNG[k][2] - 1), VREV: lod, DFULL: KPRM[k][2], DMIN: KPRM[k][3], DMAX: KPRM[k][0] });
const pend = {
  solid: scene(pl0, mSolid, 'vs', 'fs', [vbS], { cull: 'back', c: QS.has('cheap') ? { CHEAP: 1 } : undefined }),
  solidSh: P({ layout: pl0, vertex: { module: mSolid, entryPoint: 'vs', buffers: [vbS], constants: { SHADOW: 1 } }, primitive: { cullMode: 'none' }, depthStencil: { format: 'depth16unorm', depthWriteEnabled: true, depthCompare: 'less', depthBias: 3, depthBiasSlopeScale: 2.5 } }),
  sky: scene(pl0, mSky, 'vs', 'fs', [], { noz: true, cmp: 'equal' }),
  water: P({ layout: plWater, vertex: { module: mWater, entryPoint: 'vs', buffers: [vbW] }, fragment: { module: mWater, entryPoint: 'fs', targets: [{ format: HDR }] }, primitive: { cullMode: 'none' }, depthStencil: DS(false), multisample: { count: AA, alphaToCoverageEnabled: true } }),
  mote: scene(plFx, mFx, 'vsMote', 'fsGlow', [], { blend: ADD, topo: 'triangle-strip' }),
  fly: scene(plFx, mFx, 'vsFly', 'fsGlow', [], { blend: ADD, topo: 'triangle-strip' }),
  spray: scene(plFx, mFx, 'vsSpray', 'fsSoft', [], { blend: ADD, topo: 'triangle-strip' }),
  fall: scene(plFx, mFx, 'vsFall', 'fsFall', [], { topo: 'triangle-strip', a2c: true }),
  bird: scene(plFx, mFx, 'vsBird', 'fsBird', [vbS], { c: { CREATURE: 0, FXBASE: 1024, NPER: 12 } }),
  horn: scene(plFx, mFx, 'vsBird', 'fsBird', [vbS], { c: { CREATURE: 1, FXBASE: 1600 } }),
  bfly: scene(plFx, mFx, 'vsBird', 'fsBird', [vbS], { c: { CREATURE: 2, FXBASE: 1700 } }),
  drag: scene(plFx, mFx, 'vsBird', 'fsBird', [vbS], { c: { CREATURE: 3, FXBASE: 1800 } }),
  fog: P({ layout: plFog, vertex: { module: mFog, entryPoint: 'vs' }, fragment: { module: mFog, entryPoint: 'fs', targets: [{ format: HDR }, { format: 'r32float' }] } }),
  blurH: P({ layout: plBlur, vertex: { module: mBlur, entryPoint: 'vs' }, fragment: { module: mBlur, entryPoint: 'fs', constants: { DX: 1, DY: 0 }, targets: [{ format: HDR }, { format: 'r32float' }] } }),
  blurV: P({ layout: plBlur, vertex: { module: mBlur, entryPoint: 'vs' }, fragment: { module: mBlur, entryPoint: 'fs', constants: { DX: 0, DY: 1 }, targets: [{ format: HDR }, { format: 'r32float' }] } }),
  down0: P({ layout: plBloom, vertex: { module: mBloom, entryPoint: 'vs' }, fragment: { module: mBloom, entryPoint: 'fsDown0', targets: [{ format: HDR }] } }),
  down: P({ layout: plBloom, vertex: { module: mBloom, entryPoint: 'vs' }, fragment: { module: mBloom, entryPoint: 'fsDown', targets: [{ format: HDR }] } }),
  up: P({ layout: plBloom, vertex: { module: mBloom, entryPoint: 'vs' }, fragment: { module: mBloom, entryPoint: 'fsUp', targets: [{ format: HDR, blend: ADD }] } }),
  post: P({ layout: plPost, vertex: { module: mPost, entryPoint: 'vs' }, fragment: { module: mPost, entryPoint: 'fs', targets: [{ format: FMT }] } }),
};
const pLeaf = [], pLeafSh = [];
for (let k = 0; k < NKIND; k++) {
  pLeaf.push([0, 1].map((lod) => scene(plLeaf, mLeaf, 'vs', 'fs', [], { c: leafC(k, lod), a2c: true })));
  pLeafSh.push(shadowP(plLeaf, mLeaf, 'vs', 'fsShadow', [], { ...leafC(k, 1), SHADOW: 1 }));
}
const PP = {}; for (const k in pend) PP[k] = await pend[k];
for (let k = 0; k < NKIND; k++) { pLeaf[k] = await Promise.all(pLeaf[k]); pLeafSh[k] = await pLeafSh[k]; }
const pCull = device.createComputePipeline({ layout: 'auto', compute: { module: mCull, entryPoint: 'main' } });
const bgCull = device.createBindGroup({ layout: pCull.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubCull } }, { binding: 1, resource: { buffer: sbClusters } }, { binding: 2, resource: { buffer: sbVis } }, { binding: 3, resource: { buffer: bArgs } }] });
stamp('pipelines');

// ---------------------------------------------------------------- sun matrix, one-time bakes (noise, shadow map)
{
  const L = SUN, lr = vnorm(vcross([0, 1, 0], L)), lu = vcross(L, lr);
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, z0 = 1e9, z1 = -1e9;
  for (const cx of [-238, 238]) for (const cy of [-36, 105]) for (const cz of [-238, 238]) { const p = [cx, cy, cz]; const a = vdot(p, lr), b = vdot(p, lu), c = vdot(p, L); x0 = min(x0, a); x1 = max(x1, a); y0 = min(y0, b); y1 = max(y1, b); z0 = min(z0, c); z1 = max(z1, c); }
  const hx = (x1 - x0) / 2, hy = (y1 - y0) / 2, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, zr = z1 - z0 + 10;
  const m = UBS;
  m[0] = lr[0] / hx; m[4] = lr[1] / hx; m[8] = lr[2] / hx; m[12] = -cx / hx;
  m[1] = lu[0] / hy; m[5] = lu[1] / hy; m[9] = lu[2] / hy; m[13] = -cy / hy;
  m[2] = -L[0] / zr; m[6] = -L[1] / zr; m[10] = -L[2] / zr; m[14] = (z1 + 5) / zr;
  m[3] = 0; m[7] = 0; m[11] = 0; m[15] = 1;
  for (let i = 0; i < 16; i++) UB[16 + i] = UBS[16 + i] = m[i];
  UB[48] = UBS[48] = L[0]; UB[49] = UBS[49] = L[1]; UB[50] = UBS[50] = L[2];
  UBS[56] = 1; UBS[35] = 0; UBS[32] = -L[0] * 1000; UBS[33] = 1000; UBS[34] = -L[2] * 1000;
  Q.writeBuffer(uboS, 0, UBS);
  stamp(`shadow texel ${(hx * 2 / SHADOW * 100).toFixed(1)} x ${(hy * 2 / SHADOW * 100).toFixed(1)} cm`);
}
{
  const p = device.createComputePipeline({ layout: 'auto', compute: { module: mNoise, entryPoint: 'main' } });
  const e = device.createCommandEncoder(), c = e.beginComputePass();
  c.setPipeline(p); c.setBindGroup(0, device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: tNoise.createView() }] }));
  c.dispatchWorkgroups(32, 32, 32); c.end();
  const r = e.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: tShadow.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
  r.setBindGroup(0, bg0s);
  r.setPipeline(PP.solidSh); r.setVertexBuffer(0, vbStatic); r.setIndexBuffer(ibStatic, 'uint32'); r.drawIndexed(nIdx);
  for (let k = 0; k < NKIND; k++) { if (k === 3 || k === 4) continue; r.setPipeline(pLeafSh[k]); r.setBindGroup(1, bgLeaf[k]); r.setIndexBuffer(ibLeaf, 'uint16'); r.drawIndexed(12 * FROWS[k], FOL[k].n * 64); }
  r.end();
  Q.submit([e.finish()]);
  await Q.onSubmittedWorkDone();
}
stamp('bakes');
const expoAt = (x, y, z) => 1 / (0.6 + 2.2 * exp(-canopyAt(x, z) * 0.5) * sstep(-5, 25, y - gy(x, z)) + 0.4 * exp(-canopyAt(x, z) * 0.5));
const fogLumAt = (x, y, z) => 0.12 + 0.2 * exp(-canopyAt(x, z) * 0.6);

// ---------------------------------------------------------------- render targets
let W = 0, H = 0, targets = [], passMain, passWater, passFog, passBH, passBV, passPost, bgWaterT, bgFogT, bgBH, bgBV, bgPostT, bloomPass = [], bloomBG = [], tColor, tCopy;
const NBL = 6;
function resize() {
  const w = max(2, round(canvas.clientWidth * devicePixelRatio * SCALE) & ~1), h = max(2, round(canvas.clientHeight * devicePixelRatio * SCALE) & ~1);
  if (w === W && h === H) return;
  W = w; H = h; canvas.width = W; canvas.height = H;
  ctx.configure({ device, format: FMT, alphaMode: 'opaque' });
  for (const t of targets) t.destroy();
  const RA = TU.RENDER_ATTACHMENT, TB = TU.TEXTURE_BINDING;
  const tMS = tex(W, H, HDR, RA, { s: AA }), tDep = tex(W, H, DEPTH, RA | TB, { s: AA });
  tColor = tex(W, H, HDR, RA | TB | TU.COPY_SRC); tCopy = tex(W, H, HDR, TB | TU.COPY_DST);
  const hw = W / 2, hh = H / 2;
  const tF = tex(hw, hh, HDR, RA | TB), tFD = tex(hw, hh, 'r32float', RA | TB), tF2 = tex(hw, hh, HDR, RA | TB), tFD2 = tex(hw, hh, 'r32float', RA | TB);
  targets = [tMS, tDep, tColor, tCopy, tF, tFD, tF2, tFD2];
  const bl = []; for (let i = 0; i < NBL; i++) { const t = tex(max(1, W >> (i + 1)), max(1, H >> (i + 1)), HDR, RA | TB); bl.push(t); targets.push(t); }
  const vMS = tMS.createView(), vDep = tDep.createView(), vCol = tColor.createView();
  passMain = { colorAttachments: [{ view: vMS, resolveTarget: vCol, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }], depthStencilAttachment: { view: vDep, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' } };
  passWater = { colorAttachments: [{ view: vMS, resolveTarget: vCol, loadOp: 'load', storeOp: 'discard' }], depthStencilAttachment: { view: vDep, depthReadOnly: true } };
  const ca = (v) => ({ view: v, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] });
  passFog = { colorAttachments: [ca(tF.createView()), ca(tFD.createView())] };
  passBH = { colorAttachments: [ca(tF2.createView()), ca(tFD2.createView())] };
  passBV = { colorAttachments: [ca(tF.createView()), ca(tFD.createView())] };
  passPost = { colorAttachments: [{ view: null, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] };
  bgWaterT = device.createBindGroup({ layout: bglWater, entries: [{ binding: 0, resource: tCopy.createView() }, { binding: 1, resource: vDep }] });
  bgFogT = device.createBindGroup({ layout: bglFog, entries: [{ binding: 0, resource: vDep }] });
  bgBH = device.createBindGroup({ layout: bglBlur, entries: [{ binding: 0, resource: tF.createView() }, { binding: 1, resource: tFD.createView() }] });
  bgBV = device.createBindGroup({ layout: bglBlur, entries: [{ binding: 0, resource: tF2.createView() }, { binding: 1, resource: tFD2.createView() }] });
  const vBl = bl.map((t) => t.createView());
  bgPostT = device.createBindGroup({ layout: bglPost, entries: [{ binding: 0, resource: vCol }, { binding: 1, resource: tF.createView() }, { binding: 2, resource: tFD.createView() }, { binding: 3, resource: vBl[0] }, { binding: 4, resource: vDep }] });
  bloomPass = []; bloomBG = [];
  const bbg = (v) => device.createBindGroup({ layout: bglBloom, entries: [{ binding: 0, resource: v }, { binding: 1, resource: sClamp }] });
  for (let i = 0; i < NBL; i++) { bloomPass.push({ colorAttachments: [ca(vBl[i])] }); bloomBG.push(bbg(i ? vBl[i - 1] : vCol)); }
  for (let i = NBL - 1; i > 0; i--) { bloomPass.push({ colorAttachments: [{ view: vBl[i - 1], loadOp: 'load', storeOp: 'store' }] }); bloomBG.push(bbg(vBl[i])); }
  UB[52] = W; UB[53] = H; UB[54] = 1 / W; UB[55] = 1 / H;
}
resize();
addEventListener('resize', resize);

// ---------------------------------------------------------------- GPU timing
let qs, qRes, qRead = [], tsB, tsE;
const F = new Float64Array(48);
const CX = 0, CY = 1, CZ = 2, YAW = 3, PIT = 4, TIME = 5, GPU = 6, LAST = 7, VX = 8, VY = 9, VZ = 10, FLT = 11, SPL = 12, FOGL = 13, EXPO = 14, DT = 15, GPUN = 16, JST = 17, RAWDT = 18;
// every frame writes its begin/end timestamps into a 256-frame ring; each half is resolved and read back in one go
const QN = 256, GLOGN = 1 << 16, GLOG = new Float32Array(GLOGN), TSB = [], TSE = [];
let GLN = 0;
if (hasTS) {
  qs = device.createQuerySet({ type: 'timestamp', count: QN * 2 });
  qRes = device.createBuffer({ size: QN * 8, usage: BU.QUERY_RESOLVE | BU.COPY_SRC });
  for (let k = 0; k < QN; k++) { TSB.push({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: k * 2 } }); TSE.push({ querySet: qs, endOfPassWriteIndex: k * 2 + 1 }); }
  for (let i = 0; i < 2; i++) {
    const b = device.createBuffer({ size: QN * 8, usage: BU.MAP_READ | BU.COPY_DST }), o = { b, busy: false, done: null };
    o.done = () => {
      const a = new BigInt64Array(b.getMappedRange());
      for (let j = 0; j < QN / 2; j++) { const ms = Number(a[j * 2 + 1] - a[j * 2]) / 1e6; if (ms > 0 && ms < 1000) { GLOG[GLN++ & (GLOGN - 1)] = ms; F[GPU] = ms; } }
      F[GPUN]++; b.unmap(); o.busy = false;
    };
    qRead.push(o);
  }
}

// ---------------------------------------------------------------- camera
const TANY = 0.62, NEAR = 0.08;
const aimAt = (x, y, z, tx, ty, tz) => { F[CX] = x; F[CY] = y; F[CZ] = z; const dx = tx - x, dy = ty - y, dz = tz - z; F[YAW] = atan2(dx, -dz); F[PIT] = atan2(dy, hypot(dx, dz)); };
const setPose = (i) => { const p = POSES[i]; aimAt(p[0], p[2], p[1], p[3], p[5], p[4]); F[VX] = F[VY] = F[VZ] = 0; };
setPose(0);
F[TIME] = 8; F[FOGL] = fogLumAt(F[CX], F[CY], F[CZ]); F[EXPO] = 1;
// fly path: Catmull-Rom over position + target, arc-length parameterised
const FLYN = FLY.length / 6, FLYT = 60, LUTN = 1024, flyLUT = new Float32Array(LUTN + 1), P6 = new Float64Array(6);
function flyRaw(s) {
  let i = floor(s) | 0; if (i > FLYN - 2) i = FLYN - 2; const u = s - i;
  const i0 = (i > 0 ? i - 1 : 0) * 6, i1 = i * 6, i2 = (i + 1) * 6, i3 = (i + 2 < FLYN ? i + 2 : FLYN - 1) * 6;
  for (let o = 0; o < 6; o++) { const a = FLY[i0 + o], b = FLY[i1 + o], c = FLY[i2 + o], d = FLY[i3 + o]; P6[o] = 0.5 * (2 * b + (c - a) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (3 * b - a - 3 * c + d) * u * u * u); }
}
{
  const M = 4096, len = new Float32Array(M + 1); let px = 0, py = 0, pz = 0;
  for (let k = 0; k <= M; k++) { flyRaw((k / M) * (FLYN - 1)); if (k) len[k] = len[k - 1] + hypot(P6[0] - px, P6[1] - py, P6[2] - pz); px = P6[0]; py = P6[1]; pz = P6[2]; }
  for (let k = 0, j = 0; k <= LUTN; k++) { const d = (k / LUTN) * len[M]; while (j < M - 1 && len[j + 1] < d) j++; flyLUT[k] = ((j + (d - len[j]) / max(len[j + 1] - len[j], 1e-6)) / M) * (FLYN - 1); }
  stamp(`fly path ${len[M].toFixed(0)} m`);
}
// look target is smoothed separately so the view never snaps
const LK = new Float64Array(3);
function flyAt(t) {
  let s = clamp(t / FLYT, 0, 1); s = s * s * (3 - 2 * s) * 0.2 + s * 0.8; s *= LUTN;
  let k = floor(s) | 0; if (k > LUTN - 1) k = LUTN - 1;
  flyRaw(flyLUT[k] + (flyLUT[k + 1] - flyLUT[k]) * (s - k));
  aimAt(P6[0], P6[1], P6[2], P6[3], P6[4], P6[5]);
}

// ---------------------------------------------------------------- per-frame culling of the static cells
const PLN = new Float64Array(24);
function planes(cx, cy, cz, fx, fy, fz, rx, rz, ux, uy, uz, tx, ty) {
  const set = (i, nx, ny, nz) => { const l = 1 / hypot(nx, ny, nz); nx *= l; ny *= l; nz *= l; PLN[i * 4] = nx; PLN[i * 4 + 1] = ny; PLN[i * 4 + 2] = nz; PLN[i * 4 + 3] = -(nx * cx + ny * cy + nz * cz); };
  set(0, rx + fx * tx, fy * tx, rz + fz * tx); set(1, -rx + fx * tx, fy * tx, -rz + fz * tx);
  set(2, ux + fx * ty, uy + fy * ty, uz + fz * ty); set(3, -ux + fx * ty, -uy + fy * ty, -uz + fz * ty);
  set(4, fx, fy, fz); PLN[19] -= NEAR; set(5, -fx, -fy, -fz); PLN[23] += 360;
}
const VCI = new Uint16Array(NCH), VCD = new Float32Array(NCH);
const visCell = (i) => { const x = CH[i * 4], y = CH[i * 4 + 1], z = CH[i * 4 + 2], r = CH[i * 4 + 3]; for (let p = 0; p < 6; p++) if (PLN[p * 4] * x + PLN[p * 4 + 1] * y + PLN[p * 4 + 2] * z + PLN[p * 4 + 3] < -r) return false; return true; };

// ---------------------------------------------------------------- frame
const submitList = [null];
let hud = false, frameN = 0, SKIP = 0; const DS_ = +(QS.get('ds') || 1); // SKIP: profiling mask (1 solids, 2 leaves, 4 fx, 8 water, 16 fog, 32 blur+bloom)
function render() {
  const cx = F[CX], cy = F[CY], cz = F[CZ], pitch = F[PIT], yaw = F[YAW];
  const cp = cos(pitch), fy = sin(pitch), rx = cos(yaw), rz = sin(yaw), fx = cp * rz, fz = -cp * rx;
  const ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy;
  const asp = W / H, tx = TANY * asp, sx = 1 / tx, sy = 1 / TANY, m = UB;
  m[0] = rx * sx; m[4] = 0; m[8] = rz * sx; m[12] = -(rx * cx + rz * cz) * sx;
  m[1] = ux * sy; m[5] = uy * sy; m[9] = uz * sy; m[13] = -(ux * cx + uy * cy + uz * cz) * sy;
  m[2] = 0; m[6] = 0; m[10] = 0; m[14] = NEAR;
  m[3] = fx; m[7] = fy; m[11] = fz; m[15] = -(fx * cx + fy * cy + fz * cz);
  m[32] = cx; m[33] = cy; m[34] = cz; m[35] = F[TIME];
  m[36] = rx; m[37] = 0; m[38] = rz; m[39] = tx;
  m[40] = ux; m[41] = uy; m[42] = uz; m[43] = TANY;
  m[44] = fx; m[45] = fy; m[46] = fz; m[47] = NEAR;
  m[51] = 1.05 * F[EXPO];
  m[56] = DS_; m[57] = frameN % 64; m[58] = F[FOGL]; m[59] = 0.6;
  m[60] = -WH; m[61] = -WH; m[62] = 1 / (WH * 2);
  m[88] = 0.007; m[89] = 4; m[92] = FALL_MAIN.x - 2; m[93] = SLEV[0] + 2; m[94] = FALL_MAIN.z - 2; m[95] = 6;
  Q.writeBuffer(ubo, 0, UB);
  planes(cx, cy, cz, fx, fy, fz, rx, rz, ux, uy, uz, tx * 1.02, TANY * 1.02);
  for (let i = 0; i < 24; i++) CUF[i] = PLN[i];
  CUF[24] = cx; CUF[25] = cy; CUF[26] = cz; CUF[27] = DS_;
  Q.writeBuffer(ubCull, 0, CUB);
  Q.writeBuffer(bArgs, 0, ARGS);

  const e = device.createCommandEncoder();
  const qk = frameN % QN;
  const cpd = hasTS ? TSB[qk] : cullDesc;
  const c = e.beginComputePass(cpd);
  c.setPipeline(pCull); c.setBindGroup(0, bgCull); c.dispatchWorkgroups(ceil(NCL / 64)); c.end();
  let r = e.beginRenderPass(passMain);
  r.setBindGroup(0, bg0);
  // static geometry, merged runs of visible cells
  r.setPipeline(PP.solid); r.setVertexBuffer(0, vbStatic); r.setIndexBuffer(ibStatic, 'uint32');
  // visible cells nearest first, so early depth testing rejects what they hide
  let nv = 0;
  if (!(SKIP & 1)) for (let i = 0; i < NCH; i++) if (CHI[i * 2 + 1] && visCell(i)) {
    const dx = CH[i * 4] - cx, dz = CH[i * 4 + 2] - cz, d = dx * dx + dz * dz; let j = nv++;
    while (j > 0 && VCD[j - 1] > d) { VCD[j] = VCD[j - 1]; VCI[j] = VCI[j - 1]; j--; }
    VCD[j] = d; VCI[j] = i;
  }
  for (let j = 0; j < nv; j++) r.drawIndexed(CHI[VCI[j] * 2 + 1], 1, CHI[VCI[j] * 2]);
  // foliage, GPU-culled
  r.setIndexBuffer(ibLeaf, 'uint16');
  if (!(SKIP & 2)) for (let k = 0; k < NKIND; k++) {
    r.setBindGroup(1, bgLeaf[k]);
    r.setPipeline(pLeaf[k][0]); r.drawIndexedIndirect(bArgs, (k * 2) * 20);
    r.setPipeline(pLeaf[k][1]); r.drawIndexedIndirect(bArgs, (k * 2 + 1) * 20);
  }
  r.setBindGroup(1, bgFx);
  if (!(SKIP & 4)) {
  r.setPipeline(PP.fall); r.draw(4, NFALL);
  r.setPipeline(PP.bird); r.setVertexBuffer(0, vbBird); r.setIndexBuffer(ibBird, 'uint32'); r.drawIndexed(meshB.i.length, NB_BIRD);
  r.setPipeline(PP.horn); r.drawIndexed(meshB.i.length, NB_HORN);
  r.setPipeline(PP.bfly); r.setVertexBuffer(0, vbBf); r.setIndexBuffer(ibBf, 'uint32'); r.drawIndexed(meshBf.i.length, NB_BFLY);
  r.setPipeline(PP.drag); r.setVertexBuffer(0, vbDf); r.setIndexBuffer(ibDf, 'uint32'); r.drawIndexed(meshDf.i.length, NB_DRAG);
  }
  r.setPipeline(PP.sky); r.draw(3);
  if (!(SKIP & 4)) {
  r.setPipeline(PP.mote); r.draw(4, NMOTE);
  r.setPipeline(PP.fly); r.draw(4, nFF);
  r.setPipeline(PP.spray); r.draw(4, NSPRAY);
  }
  r.end();
  e.copyTextureToTexture({ texture: tColor }, { texture: tCopy }, [W, H]);
  r = e.beginRenderPass(passWater);
  r.setBindGroup(0, bg0); r.setBindGroup(1, bgWaterT);
  if (!(SKIP & 8)) { r.setPipeline(PP.water); r.setVertexBuffer(0, vbWater); r.setIndexBuffer(ibWater, 'uint32'); r.drawIndexed(nWI); }
  r.end();
  if (!(SKIP & 16)) { r = e.beginRenderPass(passFog); r.setPipeline(PP.fog); r.setBindGroup(0, bg0); r.setBindGroup(1, bgFogT); r.draw(3); r.end(); }
  if (!(SKIP & 32)) {
  r = e.beginRenderPass(passBH); r.setPipeline(PP.blurH); r.setBindGroup(0, bgBH); r.draw(3); r.end();
  r = e.beginRenderPass(passBV); r.setPipeline(PP.blurV); r.setBindGroup(0, bgBV); r.draw(3); r.end();
  for (let i = 0; i < bloomPass.length; i++) { r = e.beginRenderPass(bloomPass[i]); r.setPipeline(i === 0 ? PP.down0 : i < NBL ? PP.down : PP.up); r.setBindGroup(0, bloomBG[i]); r.draw(3); r.end(); }
  }
  passPost.colorAttachments[0].view = ctx.getCurrentTexture().createView();
  passPost.timestampWrites = hasTS ? TSE[qk] : undefined;
  r = e.beginRenderPass(passPost); r.setPipeline(PP.post); r.setBindGroup(0, bg0); r.setBindGroup(1, bgPostT); r.draw(3); r.end();
  let slot = null;
  if (hasTS && (qk === QN / 2 - 1 || qk === QN - 1)) {
    slot = qRead[qk < QN / 2 ? 0 : 1];
    if (slot.busy) slot = null;
    else { const h = qk < QN / 2 ? 0 : QN; e.resolveQuerySet(qs, h, QN, qRes, 0); e.copyBufferToBuffer(qRes, 0, slot.b, 0, QN * 8); }
  }
  submitList[0] = e.finish();
  Q.submit(submitList);
  if (slot) { slot.busy = true; slot.b.mapAsync(GPUMapMode.READ).then(slot.done); }
  frameN++;
}
const cullDesc = {};

// ---------------------------------------------------------------- input and movement
const KEY = new Uint8Array(256);
const KC = { KeyW: 1, KeyS: 2, KeyA: 3, KeyD: 4, KeyE: 5, KeyQ: 6, ShiftLeft: 7, ShiftRight: 7 };
let locked = false, flying = false, mdx = 0, mdy = 0;
addEventListener('keydown', (e) => {
  const k = KC[e.code]; if (k) KEY[k] = 1;
  if (e.repeat) return;
  if (/Digit[1-6]/.test(e.code)) { flying = false; setPose(+e.code[5] - 1); }
  if (e.code === 'KeyH') { hud = !hud; info.textContent = ''; }
});
addEventListener('keyup', (e) => { const k = KC[e.code]; if (k) KEY[k] = 0; });
addEventListener('blur', () => KEY.fill(0));
$('o').addEventListener('click', () => canvas.requestPointerLock());
document.addEventListener('pointerlockchange', () => { locked = document.pointerLockElement === canvas; body.classList.toggle('r', locked); if (!locked) $('s').textContent = 'Click to resume'; });
addEventListener('mousemove', (e) => { if (locked) { mdx += e.movementX; mdy += e.movementY; } });
function collide() {
  // push the camera out of rock (signed density gradient), keep it inside the world
  for (let k = 0; k < 3; k++) {
    const x = F[CX], y = F[CY], z = F[CZ], d = dAt(x, y, z) + 0.6;
    if (d <= 0) break;
    const gx = dAt(x + 0.4, y, z) - dAt(x - 0.4, y, z), gy_ = dAt(x, y + 0.4, z) - dAt(x, y - 0.4, z), gz = dAt(x, y, z + 0.4) - dAt(x, y, z - 0.4);
    const l = hypot(gx, gy_, gz) || 1;
    F[CX] -= (gx / l) * d; F[CY] -= (gy_ / l) * d; F[CZ] -= (gz / l) * d;
  }
  F[CX] = clamp(F[CX], -212, 212); F[CZ] = clamp(F[CZ], -212, 212); F[CY] = clamp(F[CY], -40, 130);
}
function move(dt) {
  F[YAW] += mdx * 0.0022; F[PIT] = clamp(F[PIT] - mdy * 0.0022, -1.55, 1.55); mdx = mdy = 0;
  const sp = KEY[7] ? 22 : 5, cp = cos(F[PIT]);
  const fx = cp * sin(F[YAW]), fy = sin(F[PIT]), fz = -cp * cos(F[YAW]), rx = cos(F[YAW]), rz = sin(F[YAW]);
  const iw = KEY[1] - KEY[2], ia = KEY[4] - KEY[3], iu = KEY[5] - KEY[6];
  const tx = (fx * iw + rx * ia) * sp, ty = (fy * iw + iu) * sp, tz = (fz * iw + rz * ia) * sp;
  const k = 1 - exp(-dt * 6);
  F[VX] += (tx - F[VX]) * k; F[VY] += (ty - F[VY]) * k; F[VZ] += (tz - F[VZ]) * k;
  F[CX] += F[VX] * dt; F[CY] += F[VY] * dt; F[CZ] += F[VZ] * dt;
  collide();
}
let hudT = 0, hudN = 0, hudG = 0, hudMax = 0;
let paused = false, tick = null; // paused: the harness renders on demand; tick: the benchmark / harness hook (bench.js) can take over the camera
function frame(now) {
  requestAnimationFrame(frame);
  if (paused) return;
  F[RAWDT] = F[LAST] ? now - F[LAST] : 16;
  const dt = F[LAST] ? min((now - F[LAST]) / 1000, 0.1) : 0.016; F[LAST] = now; F[DT] = dt;
  if (tick) tick(dt); else { F[TIME] += dt; move(dt); }
  // fog brightness and exposure follow how open the canopy is around the camera
  const ft = fogLumAt(F[CX], F[CY], F[CZ]); F[FOGL] += (ft - F[FOGL]) * min(dt * 1.5, 1);
  F[EXPO] += (expoAt(F[CX], F[CY], F[CZ]) - F[EXPO]) * min(dt * 1.2, 1);
  const j0 = performance.now();
  render();
  F[JST] = performance.now() - j0;
  if (hud) {
    hudN++; hudT += dt; hudG += F[GPU]; hudMax = max(hudMax, dt);
    if (hudT > 0.5) { info.textContent = `${(hudN / hudT).toFixed(0)} fps  frame ${(hudT / hudN * 1000).toFixed(2)} ms (max ${(hudMax * 1000).toFixed(1)})  GPU ${(hudG / hudN).toFixed(2)} ms  ${W}x${H}`; hudN = 0; hudT = 0; hudG = 0; hudMax = 0; }
  }
}
stamp('ready');

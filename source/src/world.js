// ============================================================================
// world layout: terrain, karst, stream, ravine, clearings, set-piece locations
// ============================================================================
// Coordinates: metres, y up. The camera starts near START looking north (-z) at a sunlit clearing.
const SUN = vnorm([-0.46, 0.8, -0.42]); // sun from the north-west, ~53° high
const START = [5, 0, 30], START_LOOK = [-6, 3.5, -24];
const CLEAR = { x: -6, z: -26, r: 15 }; // sunlit clearing
const GAPS = [[-17, -38, 21], [52, 30, 8], [-62, 18, 9], [30, -84, 9], [-48, -98, 8], [70, -40, 10], [-100, 60, 9], [-20, 96, 8], [110, 20, 7], [-140, -40, 9]]; // canopy openings x, z, r
const SHRINE = { x: 13, z: -33, rot: -1.93 };
const BANYAN_SHRINE = [23, -35.5];
// karst limestone: capsule ax, az, bx, bz, radius, top height, seed
const KARST = [
  [62, 150, 128, 88, 15, 64, 1], // the waterfall ridge (south-east), north-west face lit
  [-20, -205, 10, -192, 24, 78, 2], [88, -208, 110, -196, 20, 70, 3], [-150, -212, -160, -200, 19, 62, 4],
  [-162, -40, -150, -12, 22, 70, 5], [-140, 84, -132, 96, 17, 56, 6], [160, -44, 168, -30, 22, 72, 7], [172, 62, 166, 70, 18, 62, 8],
  [-60, 42, -60, 42, 7, 27, 9], [44, -64, 44, -64, 6, 21, 10], [-108, -88, -82, -56, 6.5, 31, 11], [130, 150, 150, 160, 16, 60, 12],
  [-190, 150, -170, 170, 22, 70, 13], [200, -150, 185, -170, 20, 66, 14], [-60, 205, -30, 210, 20, 64, 15], [60, 205, 80, 200, 20, 66, 16],
];
// carved holes (capsules): ax, ay, az, bx, by, bz, radius — an arch through the small fin, a cave under the ridge
const CARVE = [[-107, 6.5, -60, -82, 6.5, -84, 5.4], [108, 5, 96, 116, 6, 104, 6.5], [-160, 6, -20, -150, 6, -30, 6]];
// the stream: control points x, z (levels are derived from the terrain, always falling downstream)
const STREAM_P = [[72, 108], [60, 98], [46, 88], [34, 74], [22, 62], [14, 48], [12, 34], [8, 18], [2, 2], [-6, -12], [-12, -24], [-10, -38], [-6, -52], [-10, -66], [-6, -82], [2, -96], [8, -110], [10, -122]];
const POOLS = [[72, 108, 8.5], [-11, -25, 6.5], [14, 47, 4], [-8, -66, 4.5]]; // x, z, radius: wider water
const RAVINE_P = [[-262, -150], [-200, -142], [-130, -134], [-60, -144], [0, -140], [60, -150], [120, -146], [180, -156], [262, -150]];
const RAV_FLOOR = -27, RAV_HW = 6.5, RAV_RIM = 15;
const BRIDGE = { x: -58, z0: -126.5, z1: -160 };
const FALL_MAIN = { x: 77.5, z: 111.5 }; // lip of the waterfall on the ridge face

// ---------------------------------------------------------------- polylines
function resample(P, step) {
  const out = [];
  for (let i = 0; i < P.length - 1; i++) {
    const a = P[i], b = P[i + 1], n = max(1, ceil(hypot(b[0] - a[0], b[1] - a[1]) / step));
    for (let k = 0; k < n; k++) {
      const t = k / n, a0 = P[max(i - 1, 0)], b1 = P[min(i + 2, P.length - 1)];
      // Catmull-Rom
      const cr = (p0, p1, p2, p3) => 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t * t * t);
      out.push([cr(a0[0], a[0], b[0], b1[0]), cr(a0[1], a[1], b[1], b1[1])]);
    }
  }
  out.push(P[P.length - 1].slice());
  return out;
}
// nearest point on a polyline: returns [dist, segment index, t]
function nearPoly(L, x, z) {
  let best = 1e9, bi = 0, bt = 0;
  for (let i = 0; i < L.length - 1; i++) {
    const ax = L[i][0], az = L[i][1], dx = L[i + 1][0] - ax, dz = L[i + 1][1] - az, l2 = dx * dx + dz * dz;
    let t = ((x - ax) * dx + (z - az) * dz) / l2; t = t < 0 ? 0 : t > 1 ? 1 : t;
    const ex = ax + dx * t - x, ez = az + dz * t - z, d = ex * ex + ez * ez;
    if (d < best) { best = d; bi = i; bt = t; }
  }
  return [sqrt(best), bi, bt];
}
const STREAM = resample(STREAM_P, 2), RAVINE = resample(RAVINE_P, 4);
const capsule2 = (x, z, ax, az, bx, bz) => { const dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz || 1; let t = ((x - ax) * dx + (z - az) * dz) / l2; t = clamp(t, 0, 1); return hypot(ax + dx * t - x, az + dz * t - z); };
const karstDist = (x, z) => { let d = 1e9, k = -1; for (let i = 0; i < KARST.length; i++) { const c = KARST[i], e = capsule2(x, z, c[0], c[1], c[2], c[3]) - c[4]; if (e < d) { d = e; k = i; } } return [d, k]; };

// ---------------------------------------------------------------- base ground (JS, 1 m grid)
const GN = WH * 2 + 1; // samples per side
const G0 = new Float32Array(GN * GN); // natural ground before the water carves it
const GH = new Float32Array(GN * GN); // final ground the density volume uses
const SINFO = new Float32Array(GN * GN * 4); // per cell: stream distance, water level, half width, ravine distance
function natural(x, z) {
  let h = 4 + 7 * fbm2(x / 130 + 3.1, z / 130 - 1.7, 4) + 3.2 * fbm2(x / 42 + 7, z / 42 - 3, 3) + 0.7 * fbm2(x / 11, z / 11, 2);
  // steeper jungle slopes: ridged noise
  const r = 1 - abs(fbm2(x / 70 - 9, z / 70 + 4, 3)); h += 9 * r * r * sstep(40, 120, hypot(x - CLEAR.x, z - CLEAR.z));
  const rb = max(abs(x), abs(z)) + fbm2(x / 50, z / 50, 2) * 18;
  h += 80 * sstep(190, 252, rb) * (0.75 + 0.25 * fbm2(x / 30, z / 30, 3));
  // talus mounds around karst
  const [kd] = karstDist(x, z); h += 6 * sstep(26, 0, kd);
  return h;
}
let SLEV; // stream water level per resampled point
function buildGround() {
  for (let j = 0; j < GN; j++) for (let i = 0; i < GN; i++) G0[i + j * GN] = natural(i - WH, j - WH);
  const g0 = (x, z) => { const i = clamp(round(x + WH), 0, GN - 1), j = clamp(round(z + WH), 0, GN - 1); return G0[i + j * GN]; };
  // clearing: flattened
  // stream levels: follow the ground but never rise downstream; a few explicit cascades
  SLEV = new Float32Array(STREAM.length);
  let lv = 1e9;
  for (let i = 0; i < STREAM.length; i++) {
    const [x, z] = STREAM[i];
    let m = 1e9; for (let a = 0; a < 6; a++) m = min(m, g0(x + cos(a) * 3, z + sin(a) * 3));
    lv = min(lv - 0.03, m - 0.9);
    SLEV[i] = lv;
  }
  // smooth the levels a little but keep steps (cascades) crisp where the drop is > 0.6 m over 2 m
  for (let it = 0; it < 3; it++) for (let i = 1; i < STREAM.length - 1; i++) { if (SLEV[i - 1] - SLEV[i + 1] < 1.2) SLEV[i] = (SLEV[i - 1] + SLEV[i] * 2 + SLEV[i + 1]) / 4; }
  for (let i = 1; i < STREAM.length; i++) SLEV[i] = min(SLEV[i], SLEV[i - 1]);
  // pools hold one level across their whole basin; the stream drops out of them in a cascade
  for (const [px, pz, pr] of POOLS) {
    const [, si] = nearPoly(STREAM, px, pz), pl = SLEV[si];
    for (let i = 0; i < STREAM.length; i++) if (hypot(STREAM[i][0] - px, STREAM[i][1] - pz) < pr + 2.5) SLEV[i] = pl;
  }
  for (let i = 1; i < STREAM.length; i++) SLEV[i] = min(SLEV[i], SLEV[i - 1]);
  const clearLv = (() => { const [, si] = nearPoly(STREAM, CLEAR.x, CLEAR.z); return SLEV[si] + 0.7; })();
  for (let j = 0; j < GN; j++) for (let i = 0; i < GN; i++) {
    const x = i - WH, z = j - WH, k = i + j * GN;
    let h = G0[k];
    // clearing
    const dc = hypot(x - CLEAR.x, z - CLEAR.z); h = mix(clearLv + 0.25 * fbm2(x / 6, z / 6, 2), h, sstep(CLEAR.r * 0.7, CLEAR.r + 16, dc));
    // stream valley
    const [ds, si, st] = nearPoly(STREAM, x, z);
    const lev = mix(SLEV[si], SLEV[min(si + 1, STREAM.length - 1)], st);
    let hw = 1.6 + 0.9 * (0.5 + 0.5 * vn2(si * 0.13, 3.3));
    for (const p of POOLS) { const dp = hypot(x - p[0], z - p[1]); hw = max(hw, p[2] * sstep(p[2] * 1.6, p[2] * 0.4, dp) + hw * 0.0); }
    let pd = 0; for (const p of POOLS) pd = max(pd, sstep(p[2], p[2] * 0.25, hypot(x - p[0], z - p[1])));
    const bed = lev - 0.25 - 0.9 * sstep(hw, 0, ds) - 1.8 * pd; // channel, deeper in the pools
    const bank = lev + 0.35 + 0.18 * (ds - hw) + 0.35 * vn2(x * 0.4, z * 0.4);
    const tgt = ds < hw ? bed : max(bank, bed);
    h = mix(tgt, h, sstep(hw + 0.5, hw + 16, ds)) ;
    if (ds < hw + 1) h = min(h, ds < hw ? bed : lev + 0.3 * (ds - hw) + 0.05);
    // ravine
    const [dr] = nearPoly(RAVINE, x, z);
    const rr = RAV_RIM + 3 * vn2(x / 25, z / 25), fw = RAV_HW + 1.5 * vn2(x / 30 + 5, z / 30);
    if (dr < rr + 6) {
      const fl = RAV_FLOOR + 1.5 * vn2(x / 40, z / 40) - 0.4 * sstep(fw, 0, dr);
      const w = sstep(fw, rr, dr); const prof = pow(w, 0.35); // steep walls
      h = mix(fl, h, prof) + 0 * w;
      if (dr > rr) h = max(h, mix(h, G0[k], sstep(rr, rr + 6, dr)));
    }
    GH[k] = h;
    SINFO[k * 4] = ds; SINFO[k * 4 + 1] = lev; SINFO[k * 4 + 2] = hw; SINFO[k * 4 + 3] = dr;
  }
}
const gAt = (A, x, z) => { // bilinear sample of a 1 m grid
  const fx = clamp(x + WH, 0, GN - 1.001), fz = clamp(z + WH, 0, GN - 1.001), i = floor(fx), j = floor(fz), u = fx - i, v = fz - j, k = i + j * GN;
  return mix(mix(A[k], A[k + 1], u), mix(A[k + GN], A[k + GN + 1], u), v);
};
const sInfo = (x, z, c) => SINFO[(clamp(round(x + WH), 0, GN - 1) + clamp(round(z + WH), 0, GN - 1) * GN) * 4 + c];

// ---------------------------------------------------------------- density volume (GPU)
const VS = 1.25, NX = 416, NY = 112, NZ = 416, OX = -WH, OY = -44, OZ = -WH;
const W_DENS = /*wgsl*/`
struct P { n: vec4u }
@group(0) @binding(0) var<storage, read> gh: array<f32>;
@group(0) @binding(1) var<storage, read_write> outD: array<f32>;
@group(0) @binding(2) var<storage, read> karst: array<vec4f>;
@group(0) @binding(3) var<storage, read> carve: array<vec4f>;
fn hh(p: vec3i) -> f32 { var x = u32(p.x * 73856093) ^ u32(p.y * 19349663) ^ u32(p.z * 83492791); x = (x ^ (x >> 16u)) * 0x7feb352du; x = (x ^ (x >> 15u)) * 0x846ca68bu; x = x ^ (x >> 16u); return f32(x) / 4294967295.0; }
fn vn(p: vec3f) -> f32 {
  let i = vec3i(floor(p)); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(hh(i), hh(i + vec3i(1, 0, 0)), u.x), mix(hh(i + vec3i(0, 1, 0)), hh(i + vec3i(1, 1, 0)), u.x), u.y);
  let b = mix(mix(hh(i + vec3i(0, 0, 1)), hh(i + vec3i(1, 0, 1)), u.x), mix(hh(i + vec3i(0, 1, 1)), hh(i + vec3i(1, 1, 1)), u.x), u.y);
  return mix(a, b, u.z) * 2.0 - 1.0;
}
fn fbm(p: vec3f) -> f32 { var s = 0.0; var a = 0.5; var q = p; for (var i = 0; i < 4; i++) { s += a * vn(q); q = q * 2.03 + vec3f(5.1, 1.3, 7.7); a *= 0.5; } return s; }
fn ghAt(x: f32, z: f32) -> f32 {
  let fx = clamp(x + ${WH}.0, 0.0, ${GN - 1}.0 - 0.001); let fz = clamp(z + ${WH}.0, 0.0, ${GN - 1}.0 - 0.001);
  let i = u32(fx); let j = u32(fz); let u = fx - f32(i); let v = fz - f32(j); let k = i + j * ${GN}u;
  return mix(mix(gh[k], gh[k + 1u], u), mix(gh[k + ${GN}u], gh[k + ${GN + 1}u], u), v);
}
@compute @workgroup_size(4, 4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${NX}u || id.y >= ${NY}u || id.z >= ${NZ}u) { return; }
  let p = vec3f(${OX}.0, ${OY}.0, ${OZ}.0) + vec3f(id) * ${VS};
  let g = ghAt(p.x, p.z);
  // slope of the ground: steep ground (valley sides, ravine walls) becomes craggy rock with overhangs
  let gx = ghAt(p.x + 1.5, p.z) - ghAt(p.x - 1.5, p.z); let gz = ghAt(p.x, p.z + 1.5) - ghAt(p.x, p.z - 1.5);
  let sl = clamp(length(vec2f(gx, gz)) / 3.0, 0.0, 3.0);
  var d = g - p.y;
  let rock = smoothstep(0.7, 1.6, sl);
  d += rock * (fbm(p * vec3f(0.11, 0.07, 0.11)) * 5.5 + vn(p * 0.35) * 1.1);
  d += (1.0 - rock) * vn(p * 0.21 + 3.0) * 0.35;
  // karst towers and ridges: vertical, fluted, rounded tops
  for (var i = 0u; i < ${KARST.length}u; i++) {
    let a = karst[i * 2u]; let b = karst[i * 2u + 1u];
    let ab = a.zw - a.xy; let t = clamp(dot(p.xz - a.xy, ab) / max(dot(ab, ab), 1e-4), 0.0, 1.0);
    let dist = length(p.xz - (a.xy + ab * t));
    if (dist > b.x * 1.8 + 12.0) { continue; }
    let s = b.z * 17.0;
    let hy = clamp(p.y / b.y, 0.0, 1.2);
    var r = b.x * (1.0 + 0.22 * (1.0 - hy) * (1.0 - hy) - 0.08 * hy);
    let fl = fbm(vec3f(p.x * 0.09, p.y * 0.022, p.z * 0.09) + s);   // vertical flutes
    let rid = 1.0 - abs(vn(vec3f(p.x * 0.25, p.y * 0.035, p.z * 0.25) + s));
    r += b.x * 0.3 * fl + 2.0 * vn(p * 0.18 + s) + 0.9 * vn(p * 0.5 + s) - 2.2 * rid * rid;
    // horizontal bands and ledges
    r += 1.6 * sin(p.y * 0.21 + fl * 3.0) * smoothstep(0.3, 0.8, vn(p * vec3f(0.04, 0.0, 0.04) + s) * 0.5 + 0.5);
    let top = b.y + 9.0 * fbm(p * 0.03 + s + 11.0) + 4.0 * vn(p * 0.12 + s);
    let dome = top - p.y - max(dist - r * 0.6, 0.0) * 0.6;
    let kd = min(r - dist, dome);
    d = max(d, kd);
  }
  for (var i = 0u; i < ${CARVE.length}u; i++) {
    let a = carve[i * 2u].xyz; let b = carve[i * 2u + 1u];
    let ab = b.xyz - a; let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    let q = p - (a + ab * t);
    let cd = length(q * vec3f(1.0, 1.25, 1.0)) - b.w * (1.0 + 0.25 * vn(p * 0.3));
    d = min(d, cd);
  }
  outD[id.x + ${NX}u * (id.y + ${NY}u * id.z)] = clamp(d, -8.0, 8.0);
}`;
const karstBuf = () => { const a = new Float32Array(KARST.length * 8); KARST.forEach((k, i) => a.set([k[0], k[1], k[2], k[3], k[4], k[5], k[6], 0], i * 8)); return a; };
const carveBuf = () => { const a = new Float32Array(CARVE.length * 8); CARVE.forEach((c, i) => a.set([c[0], c[1], c[2], 0, c[3], c[4], c[5], c[6]], i * 8)); return a; };

// ---------------------------------------------------------------- surface nets
// D: density (> 0 solid), returns { pos, nrm, quads (vertex indices) }, plus the top-surface height map
function surfaceNets(D) {
  const SX = NX, SXY = NX * NY;
  const vid = new Int32Array(NX * NY * NZ).fill(-1);
  const P = new Grow(Float32Array, 1 << 22), Nn = new Grow(Float32Array, 1 << 22);
  const E = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const CO = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const v = new Float32Array(8);
  let nv = 0;
  for (let z = 0; z < NZ - 1; z++) for (let y = 0; y < NY - 1; y++) for (let x = 0; x < NX - 1; x++) {
    const i0 = x + SX * y + SXY * z;
    let mask = 0;
    for (let c = 0; c < 8; c++) { const q = D[i0 + CO[c][0] + SX * CO[c][1] + SXY * CO[c][2]]; v[c] = q; if (q > 0) mask |= 1 << c; }
    if (mask === 0 || mask === 255) continue;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const [a, b] of E) {
      if ((v[a] > 0) === (v[b] > 0)) continue;
      const t = v[a] / (v[a] - v[b]);
      sx += CO[a][0] + (CO[b][0] - CO[a][0]) * t; sy += CO[a][1] + (CO[b][1] - CO[a][1]) * t; sz += CO[a][2] + (CO[b][2] - CO[a][2]) * t; n++;
    }
    sx /= n; sy /= n; sz /= n;
    // gradient by trilinear derivative of the cell
    const gx = (v[1] - v[0]) * (1 - sy) * (1 - sz) + (v[3] - v[2]) * sy * (1 - sz) + (v[5] - v[4]) * (1 - sy) * sz + (v[7] - v[6]) * sy * sz;
    const gy = (v[2] - v[0]) * (1 - sx) * (1 - sz) + (v[3] - v[1]) * sx * (1 - sz) + (v[6] - v[4]) * (1 - sx) * sz + (v[7] - v[5]) * sx * sz;
    const gz = (v[4] - v[0]) * (1 - sx) * (1 - sy) + (v[5] - v[1]) * sx * (1 - sy) + (v[6] - v[2]) * (1 - sx) * sy + (v[7] - v[3]) * sx * sy;
    const gl = -1 / (hypot(gx, gy, gz) || 1);
    P.need(3); P.a[P.n++] = OX + (x + sx) * VS; P.a[P.n++] = OY + (y + sy) * VS; P.a[P.n++] = OZ + (z + sz) * VS;
    Nn.need(3); Nn.a[Nn.n++] = gx * gl; Nn.a[Nn.n++] = gy * gl; Nn.a[Nn.n++] = gz * gl;
    vid[i0] = nv++;
  }
  // smooth normals with the larger-scale gradient (central differences over the volume)
  const at = (x, y, z) => D[clamp(x, 0, NX - 1) + SX * clamp(y, 0, NY - 1) + SXY * clamp(z, 0, NZ - 1)];
  for (let i = 0; i < nv; i++) {
    const fx = (P.a[i * 3] - OX) / VS, fy = (P.a[i * 3 + 1] - OY) / VS, fz = (P.a[i * 3 + 2] - OZ) / VS, x = round(fx), y = round(fy), z = round(fz);
    const gx = at(x + 1, y, z) - at(x - 1, y, z), gy = at(x, y + 1, z) - at(x, y - 1, z), gz = at(x, y, z + 1) - at(x, y, z - 1);
    const l = -1 / (hypot(gx, gy, gz) || 1);
    let nx = Nn.a[i * 3] + gx * l, ny = Nn.a[i * 3 + 1] + gy * l, nz = Nn.a[i * 3 + 2] + gz * l; const m = 1 / (hypot(nx, ny, nz) || 1);
    Nn.a[i * 3] = nx * m; Nn.a[i * 3 + 1] = ny * m; Nn.a[i * 3 + 2] = nz * m;
  }
  // quads: for each edge crossing along an axis, connect the 4 cells sharing that edge
  const Q = new Grow(Uint32Array, 1 << 22);
  for (let z = 1; z < NZ - 1; z++) for (let y = 1; y < NY - 1; y++) for (let x = 1; x < NX - 1; x++) {
    const i0 = x + SX * y + SXY * z, s0 = D[i0] > 0;
    // edge to +x: cells (x, y-1..y, z-1..z)
    if ((D[i0 + 1] > 0) !== s0) { const a = vid[i0 - SX - SXY], b = vid[i0 - SXY], c = vid[i0], d = vid[i0 - SX]; if (a >= 0 && b >= 0 && c >= 0 && d >= 0) { Q.need(4); if (s0) Q.a.set([a, b, c, d], Q.n); else Q.a.set([a, d, c, b], Q.n); Q.n += 4; } }
    if ((D[i0 + SX] > 0) !== s0) { const a = vid[i0 - 1 - SXY], b = vid[i0 - 1], c = vid[i0], d = vid[i0 - SXY]; if (a >= 0 && b >= 0 && c >= 0 && d >= 0) { Q.need(4); if (s0) Q.a.set([a, b, c, d], Q.n); else Q.a.set([a, d, c, b], Q.n); Q.n += 4; } }
    if ((D[i0 + SXY] > 0) !== s0) { const a = vid[i0 - 1 - SX], b = vid[i0 - SX], c = vid[i0], d = vid[i0 - 1]; if (a >= 0 && b >= 0 && c >= 0 && d >= 0) { Q.need(4); if (s0) Q.a.set([a, b, c, d], Q.n); else Q.a.set([a, d, c, b], Q.n); Q.n += 4; } }
  }
  // top surface height map (1 m grid): highest solid -> air transition per column
  const TOP = new Float32Array(GN * GN);
  for (let j = 0; j < GN; j++) for (let i = 0; i < GN; i++) {
    const fx = clamp(i / VS, 0, NX - 1.001), fz = clamp(j / VS, 0, NZ - 1.001), x0 = floor(fx), z0 = floor(fz), u = fx - x0, w = fz - z0;
    const col = (y) => { const k = x0 + SX * y + SXY * z0; return mix(mix(D[k], D[k + 1], u), mix(D[k + SXY], D[k + SXY + 1], u), w); };
    let h = OY;
    for (let y = NY - 2; y >= 0; y--) { const a = col(y), b = col(y + 1); if (a > 0 && b <= 0) { h = OY + (y + a / (a - b)) * VS; break; } }
    TOP[i + j * GN] = h;
  }
  return { P: P.view(), N: Nn.view(), Q: Q.view(), nv, TOP };
}

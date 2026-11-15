// ============================================================================
// math, random numbers, noise and the static-geometry builder
// ============================================================================
const { sin, cos, tan, sqrt, abs, floor, ceil, min, max, PI, atan2, hypot, round, exp, pow, log, sign, acos, asin } = Math;
const TAU = PI * 2;
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const mix = (a, b, t) => a + (b - a) * t;
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// deterministic RNG (mulberry32)
function RNG(seed) {
  let s = seed >>> 0;
  const f = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  f.r = (a, b) => a + (b - a) * f();
  f.i = (n) => floor(f() * n);
  f.n = () => { let u = 0; for (let k = 0; k < 4; k++) u += f(); return (u - 2) * 0.866; }; // ~N(0,0.5)
  f.pick = (a) => a[floor(f() * a.length)];
  return f;
}
const ihash = (x) => { x = Math.imul(x ^ (x >>> 16), 0x7feb352d); x = Math.imul(x ^ (x >>> 15), 0x846ca68b); return (x ^ (x >>> 16)) >>> 0; };
const h2 = (x, z) => ihash((x | 0) * 73856093 ^ ((z | 0) * 19349663)) / 4294967296;
const h3 = (x, y, z) => ihash((x | 0) * 73856093 ^ ((y | 0) * 19349663) ^ ((z | 0) * 83492791)) / 4294967296;
// value noise in [-1,1]
function vn2(x, z) {
  const ix = floor(x), iz = floor(z), fx = x - ix, fz = z - iz, ux = fx * fx * (3 - 2 * fx), uz = fz * fz * (3 - 2 * fz);
  const a = h2(ix, iz), b = h2(ix + 1, iz), c = h2(ix, iz + 1), d = h2(ix + 1, iz + 1);
  return (a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz) * 2 - 1;
}
function vn3(x, y, z) {
  const ix = floor(x), iy = floor(y), iz = floor(z), fx = x - ix, fy = y - iy, fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy), uz = fz * fz * (3 - 2 * fz);
  const l = (a, b, t) => a + (b - a) * t;
  const x0 = l(l(h3(ix, iy, iz), h3(ix + 1, iy, iz), ux), l(h3(ix, iy + 1, iz), h3(ix + 1, iy + 1, iz), ux), uy);
  const x1 = l(l(h3(ix, iy, iz + 1), h3(ix + 1, iy, iz + 1), ux), l(h3(ix, iy + 1, iz + 1), h3(ix + 1, iy + 1, iz + 1), ux), uy);
  return l(x0, x1, uz) * 2 - 1;
}
const fbm2 = (x, z, o = 4) => { let s = 0, a = 0.5, f = 1; for (let i = 0; i < o; i++) { s += a * vn2(x * f + i * 17.3, z * f - i * 9.1); f *= 2.03; a *= 0.5; } return s; };

// small vector helpers (only used at load time)
const v3 = (x, y, z) => [x, y, z];
const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vmul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vlen = (a) => hypot(a[0], a[1], a[2]);
const vnorm = (a) => { const l = vlen(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const vlerp = (a, b, t) => [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)];
const anyPerp = (d) => vnorm(abs(d[1]) < 0.9 ? vcross(d, [0, 1, 0]) : vcross(d, [1, 0, 0]));
const rotAxis = (v, k, a) => { // Rodrigues
  const c = cos(a), s = sin(a), d = vdot(k, v), x = vcross(k, v);
  return [v[0] * c + x[0] * s + k[0] * d * (1 - c), v[1] * c + x[1] * s + k[1] * d * (1 - c), v[2] * c + x[2] * s + k[2] * d * (1 - c)];
};

// growable typed arrays
class Grow {
  constructor(T, n = 1 << 16) { this.T = T; this.a = new T(n); this.n = 0; }
  need(k) { if (this.n + k > this.a.length) { let m = this.a.length * 2; while (m < this.n + k) m *= 2; const b = new this.T(m); b.set(this.a.subarray(0, this.n)); this.a = b; } }
  push(v) { this.need(1); this.a[this.n++] = v; }
  view() { return this.a.subarray(0, this.n); }
}

// ---------------------------------------------------------------- static geometry
// Vertex (24 B): position f32x3 | normal snorm8x3 + ao | material unorm8x4 | sway unorm8x4
//   material: r = material id, g = tint/variation, b = moss, a = wetness
//   sway:     r = sway weight (×0.02 m per unit), g = phase, b = flex (fine motion), a = param
// The world is split into CELL-sized cells; each cell's geometry is one contiguous index range, culled on the CPU.
const CELL = 32, WH = 260, NCELL = Math.ceil((WH * 2) / CELL); // world spans [-WH, WH]
const cellOf = (x, z) => clamp(floor((x + WH) / CELL), 0, NCELL - 1) + NCELL * clamp(floor((z + WH) / CELL), 0, NCELL - 1);
const sn8 = (v) => (clamp(round(v * 127), -127, 127) & 255);
const un8 = (v) => clamp(round(v * 255), 0, 255);
class Geo {
  constructor() {
    this.f = new Grow(Float32Array, 1 << 22); // 6 words per vertex (3 floats + 3 packed u32 as float bits)
    this.u = null;
    this.idx = []; for (let i = 0; i < NCELL * NCELL; i++) this.idx.push(new Grow(Uint32Array, 1024));
    this.nv = 0; this.cell = 0;
    this.mat = 0; this.sway = 0; // current packed material / sway words
    this._b = new ArrayBuffer(4); this._f = new Float32Array(this._b); this._u = new Uint32Array(this._b);
  }
  at(x, z) { this.cell = cellOf(x, z); return this; }
  setMat(id, tint = 0.5, moss = 0, wet = 0) { this._u[0] = un8(id / 255) | (un8(tint) << 8) | (un8(moss) << 16) | (un8(wet) << 24); this.mat = this._f[0]; return this; }
  setSway(w = 0, ph = 0, flex = 0, prm = 0) { this._u[0] = un8(clamp(w, 0, 1)) | (un8(ph % 1) << 8) | (un8(clamp(flex, 0, 1)) << 16) | (un8(clamp(prm, 0, 1)) << 24); this.sway = this._f[0]; return this; }
  packN(nx, ny, nz, ao) { this._u[0] = sn8(nx) | (sn8(ny) << 8) | (sn8(nz) << 16) | (sn8(ao * 2 - 1) << 24); return this._f[0]; }
  // vertex with explicit packed words (mat/sway default to current)
  v(x, y, z, nx, ny, nz, ao = 1, mat = this.mat, sway = this.sway) {
    const f = this.f; f.need(6); const a = f.a, o = f.n;
    a[o] = x; a[o + 1] = y; a[o + 2] = z; a[o + 3] = this.packN(nx, ny, nz, ao); a[o + 4] = mat; a[o + 5] = sway; f.n += 6;
    return this.nv++;
  }
  tri(a, b, c) { const g = this.idx[this.cell]; g.need(3); g.a[g.n++] = a; g.a[g.n++] = b; g.a[g.n++] = c; }
  quad(a, b, c, d) { this.tri(a, b, c); this.tri(a, c, d); }
  // generalised tube along points P[i] with radius R[i] (or function r(i,θ)), `sides` around; swayW(i) per ring
  tube(P, R, sides, o = {}) {
    const n = P.length, base = this.nv;
    let T = vnorm(vsub(P[1], P[0])), N0 = anyPerp(T);
    const frames = [];
    for (let i = 0; i < n; i++) {
      const Ti = i === 0 ? T : i === n - 1 ? vnorm(vsub(P[i], P[i - 1])) : vnorm(vsub(P[i + 1], P[i - 1]));
      // parallel transport
      const d = vdot(N0, Ti); N0 = vnorm(vsub(N0, vmul(Ti, d)));
      frames.push([Ti, N0, vcross(Ti, N0)]);
    }
    const ring = sides + 1;
    for (let i = 0; i < n; i++) {
      const [Ti, Ni, Bi] = frames[i];
      if (o.sway) o.sway(i, P[i]);
      if (o.mat) o.mat(i, P[i]);
      for (let k = 0; k <= sides; k++) {
        const th = (k / sides) * TAU, c = cos(th), s = sin(th);
        const r = typeof R === 'function' ? R(i, th) : R[i];
        const dir = [Ni[0] * c + Bi[0] * s, Ni[1] * c + Bi[1] * s, Ni[2] * c + Bi[2] * s];
        const p = vadd(P[i], vmul(dir, r));
        let nn = dir;
        if (typeof R === 'function') { // approximate normal including radius change around
          const r2 = R(i, th + 0.05), dr = (r2 - r) / 0.05, tg = [-Ni[0] * s + Bi[0] * c, -Ni[1] * s + Bi[1] * c, -Ni[2] * s + Bi[2] * c];
          nn = vnorm(vsub(dir, vmul(tg, dr / max(r, 0.05))));
        }
        const ao = o.ao ? o.ao(i, th, r) : 1;
        this.v(p[0], p[1], p[2], nn[0], nn[1], nn[2], ao);
      }
    }
    for (let i = 0; i < n - 1; i++) for (let k = 0; k < sides; k++) {
      const a = base + i * ring + k, b = a + 1, c = a + ring + 1, d = a + ring;
      this.quad(a, b, c, d);
    }
    if (o.cap) { // close the far end with a fan
      const [Ti] = frames[n - 1], cidx = this.v(P[n - 1][0] + Ti[0] * 0.01, P[n - 1][1] + Ti[1] * 0.01, P[n - 1][2] + Ti[2] * 0.01, Ti[0], Ti[1], Ti[2], 1);
      for (let k = 0; k < sides; k++) this.tri(base + (n - 1) * ring + k, base + (n - 1) * ring + k + 1, cidx);
    }
    return frames;
  }
  // box from a centre, half-extents and basis (ax, ay, az), with optional upward taper
  box(c, hx, hy, hz, ax = [1, 0, 0], ay = [0, 1, 0], az = [0, 0, 1], taper = 1) {
    const F = [[ax, 1], [ax, -1], [ay, 1], [ay, -1], [az, 1], [az, -1]];
    const H = [hx, hy, hz], A = [ax, ay, az];
    for (let f = 0; f < 6; f++) {
      const ai = f >> 1, sgn = f & 1 ? -1 : 1, n = vmul(A[ai], sgn);
      const u = A[(ai + 1) % 3], w = A[(ai + 2) % 3], hu = H[(ai + 1) % 3], hw = H[(ai + 2) % 3];
      const ids = [];
      for (const [su, sw] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        let p = vadd(c, vadd(vmul(n, H[ai]), vadd(vmul(u, su * hu), vmul(w, sw * hw))));
        // taper: points above centre shrink toward the axis
        const up = vdot(vsub(p, c), ay);
        if (taper !== 1 && up > 0) { const off = vsub(p, vadd(c, vmul(ay, up))); p = vadd(vadd(c, vmul(ay, up)), vmul(off, taper)); }
        ids.push(this.v(p[0], p[1], p[2], n[0], n[1], n[2], f === 3 ? 0.55 : 1));
      }
      if (sgn > 0) this.quad(ids[0], ids[1], ids[2], ids[3]); else this.quad(ids[0], ids[3], ids[2], ids[1]);
    }
  }
  // noise-displaced sphere (boulders, mushroom caps); returns nothing
  blob(c, r, seed, o = {}) {
    const sub = o.sub ?? 2, sq = o.squash ?? 1;
    const { V, I } = icosphere(sub), base = this.nv, rr = RNG(seed), ox = rr() * 100, oy = rr() * 100, oz = rr() * 100;
    const P = [];
    const dsp = (d) => { const n = vn3(d[0] * 1.3 + ox, d[1] * 1.3 + oy, d[2] * 1.3 + oz) * 0.22 + vn3(d[0] * 3 + oy, d[1] * 3 + oz, d[2] * 3 + ox) * 0.08 + vn3(d[0] * 7 + oz, d[1] * 7, d[2] * 7) * 0.03; return 1 + n * (o.rough ?? 1); };
    for (const d of V) { const s = dsp(d) * r; let y = d[1] * s * sq; if (o.flat && y < -r * o.flat) y = -r * o.flat + (y + r * o.flat) * 0.2; P.push([c[0] + d[0] * s * (o.sx ?? 1), c[1] + y, c[2] + d[2] * s * (o.sz ?? 1)]); }
    // normals by accumulating faces
    const N = V.map(() => [0, 0, 0]);
    for (let i = 0; i < I.length; i += 3) { const a = P[I[i]], b = P[I[i + 1]], cc = P[I[i + 2]], n = vcross(vsub(b, a), vsub(cc, a)); for (let k = 0; k < 3; k++) { const q = N[I[i + k]]; q[0] += n[0]; q[1] += n[1]; q[2] += n[2]; } }
    for (let i = 0; i < V.length; i++) { const n = vnorm(N[i]); const ao = o.ao ? o.ao(P[i], n) : clamp(0.55 + 0.45 * (V[i][1] + 1) * 0.6, 0, 1); this.v(P[i][0], P[i][1], P[i][2], n[0], n[1], n[2], ao); }
    for (let i = 0; i < I.length; i += 3) this.tri(base + I[i], base + I[i + 1], base + I[i + 2]);
  }
  // lathe (surface of revolution) around an up axis: prof = [[r, y], ...]
  lathe(c, prof, sides, ay = [0, 1, 0], rot = 0) {
    const ax = anyPerp(ay), az = vcross(ax, ay), base = this.nv;
    for (let i = 0; i < prof.length; i++) {
      const [r, y] = prof[i], p0 = prof[max(i - 1, 0)], p1 = prof[min(i + 1, prof.length - 1)];
      const dr = p1[0] - p0[0], dy = p1[1] - p0[1], ln = hypot(dr, dy) || 1, nr = dy / ln, ny = -dr / ln;
      for (let k = 0; k <= sides; k++) {
        const th = (k / sides) * TAU + rot, cx = cos(th), sz = sin(th);
        const dir = vadd(vmul(ax, cx), vmul(az, sz));
        const p = vadd(c, vadd(vmul(dir, r), vmul(ay, y))), n = vnorm(vadd(vmul(dir, nr), vmul(ay, ny)));
        this.v(p[0], p[1], p[2], n[0], n[1], n[2], 1);
      }
    }
    const ring = sides + 1;
    for (let i = 0; i < prof.length - 1; i++) for (let k = 0; k < sides; k++) { const a = base + i * ring + k; this.quad(a, a + ring, a + ring + 1, a + 1); }
  }
}
let _ico = {};
function icosphere(sub) {
  if (_ico[sub]) return _ico[sub];
  const t = (1 + sqrt(5)) / 2;
  let V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map(vnorm);
  let I = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1];
  for (let s = 0; s < sub; s++) {
    const m = new Map(), J = [];
    const mid = (a, b) => { const k = a < b ? a * 65536 + b : b * 65536 + a; let r = m.get(k); if (r === undefined) { r = V.length; V.push(vnorm(vlerp(V[a], V[b], 0.5))); m.set(k, r); } return r; };
    for (let i = 0; i < I.length; i += 3) { const a = I[i], b = I[i + 1], c = I[i + 2], ab = mid(a, b), bc = mid(b, c), ca = mid(c, a); J.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca); }
    I = J;
  }
  return (_ico[sub] = { V, I });
}

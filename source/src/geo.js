// ============================================================================
// content: trees, plants, rocks, set pieces, water, creatures, world map
// ============================================================================
// ---------------------------------------------------------------- foliage instances
// 32 B: f32 x,y,z,size | snorm fwd.xyz,droop | snorm faceN.xyz,ao | snorm bend.xyz,sway | rgb565 var:4 phase:6 aspect:6
const NKIND = 5;
const KINDS = []; // per kind: { u: Grow(Uint32Array) }
const _cv = new Float32Array(1), _cu = new Uint32Array(_cv.buffer);
const fbits = (f) => { _cv[0] = f; return _cu[0]; };
const p4s = (a, b, c, d) => (sn8(a) | (sn8(b) << 8) | (sn8(c) << 16) | (sn8(d) << 24)) >>> 0;
for (let k = 0; k < NKIND; k++) KINDS.push(new Grow(Uint32Array, 1 << 20));
function leaf(kind, p, size, fwd, nf, bend, ao, sw, droop, rgb, variant = 0, phase = 0, aspect = 0.5) {
  const g = KINDS[kind]; g.need(8); const a = g.a, o = g.n;
  a[o] = fbits(p[0]); a[o + 1] = fbits(p[1]); a[o + 2] = fbits(p[2]); a[o + 3] = fbits(size);
  a[o + 4] = p4s(fwd[0], fwd[1], fwd[2], droop);
  a[o + 5] = p4s(nf[0], nf[1], nf[2], ao * 2 - 1);
  a[o + 6] = p4s(bend[0], bend[1], bend[2], clamp(sw, 0, 1) * 2 - 1);
  const r = clamp(round(rgb[0] * 31), 0, 31), gg = clamp(round(rgb[1] * 63), 0, 63), b = clamp(round(rgb[2] * 31), 0, 31);
  a[o + 7] = ((r << 11) | (gg << 5) | b | ((variant & 15) << 16) | ((round((phase % 1) * 63) & 63) << 20) | ((round(clamp(aspect, 0, 1) * 63) & 63) << 26)) >>> 0;
  g.n += 8;
}
// sort each kind into 3D bins and cut the bins into clusters of 64 (shuffled, padded with zero-size leaves)
function buildClusters() {
  const out = [];
  for (let k = 0; k < NKIND; k++) {
    const g = KINDS[k], n = g.n / 8, A = g.a, F = new Float32Array(A.buffer, 0, g.n);
    const B = [6, 4, 6, 3, 5][k];
    const map = new Map();
    for (let i = 0; i < n; i++) {
      const key = floor((F[i * 8] + 300) / B) + floor((F[i * 8 + 1] + 100) / B) * 1000 + floor((F[i * 8 + 2] + 300) / B) * 1000000;
      let l = map.get(key); if (!l) map.set(key, (l = [])); l.push(i);
    }
    const keys = [...map.keys()].sort((a, b) => a - b);
    let nc = 0; for (const key of keys) nc += ceil(map.get(key).length / 64);
    const U = new Uint32Array(nc * 64 * 8), UF = new Float32Array(U.buffer), C = new Float32Array(nc * 4);
    const rr = RNG(77 + k); let c = 0;
    for (const key of keys) {
      const l = map.get(key);
      for (let i = l.length - 1; i > 0; i--) { const j = floor(rr() * (i + 1)); const t = l[i]; l[i] = l[j]; l[j] = t; }
      for (let s = 0; s < l.length; s += 64, c++) {
        let mnx = 1e9, mny = 1e9, mnz = 1e9, mxx = -1e9, mxy = -1e9, mxz = -1e9, ms = 0;
        for (let j = 0; j < 64; j++) {
          const o = (c * 64 + j) * 8;
          if (s + j < l.length) {
            const src = l[s + j] * 8; for (let w = 0; w < 8; w++) U[o + w] = A[src + w];
            const x = UF[o], y = UF[o + 1], z = UF[o + 2], sz = UF[o + 3];
            mnx = min(mnx, x); mny = min(mny, y); mnz = min(mnz, z); mxx = max(mxx, x); mxy = max(mxy, y); mxz = max(mxz, z); ms = max(ms, sz);
          } else { U.set(U.subarray((c * 64) * 8, (c * 64) * 8 + 8), o); UF[o + 3] = 0; }
        }
        C[c * 4] = (mnx + mxx) / 2; C[c * 4 + 1] = (mny + mxy) / 2; C[c * 4 + 2] = (mnz + mxz) / 2;
        C[c * 4 + 3] = hypot(mxx - mnx, mxy - mny, mxz - mnz) / 2 + ms * 1.4 + 0.5;
      }
    }
    out.push({ inst: U, cl: C, n: nc });
    KINDS[k] = null;
  }
  return out;
}

// ---------------------------------------------------------------- helpers
let TOP; // top-surface height map (1 m grid) from the density volume
const gy = (x, z) => gAt(TOP, x, z);
const gnorm = (x, z) => vnorm([gy(x - 1, z) - gy(x + 1, z), 2, gy(x, z - 1) - gy(x, z + 1)]);
const slope = (x, z) => 1 - gnorm(x, z)[1];
const waterDist = (x, z) => sInfo(x, z, 0) - sInfo(x, z, 2);
const randDir = (r) => { const z = r() * 2 - 1, a = r() * TAU, s = sqrt(1 - z * z); return [s * cos(a), z, s * sin(a)]; };
const jit = (c, r, k) => [clamp(c[0] + (r() - 0.5) * k, 0, 1), clamp(c[1] + (r() - 0.5) * k, 0, 1), clamp(c[2] + (r() - 0.5) * k * 0.6, 0, 1)];
const PAL = [[0.10, 0.27, 0.07], [0.17, 0.33, 0.07], [0.27, 0.40, 0.09], [0.21, 0.29, 0.09], [0.09, 0.25, 0.15], [0.13, 0.30, 0.10], [0.30, 0.42, 0.12], [0.07, 0.20, 0.07]];
const FLUSH = [[0.48, 0.20, 0.08], [0.55, 0.17, 0.13], [0.45, 0.52, 0.12], [0.62, 0.42, 0.12]];
const CLUMPS = []; // canopy clumps: x, y, z, r (for the canopy map, perches and the flight path)
const TREES = []; // trunk positions: x, z, radius, height (collision of placement)
const PERCH = [];
const FLY_CLEAR = []; // points the camera path passes, kept free of trunks
const nearFly = (x, z, r) => { for (const p of FLY_CLEAR) if (hypot(p[0] - x, p[2] - z) < r) return true; return false; };

// a tapered branch: points P with radii from r0 to r1; returns tip
function limb(geo, P, r0, r1, sides, tree, mat = tree.mat) {
  geo.setMat(mat, tree.tint, tree.moss, 0);
  geo.tube(P, P.map((_, i) => mix(r0, r1, i / (P.length - 1))), sides, {
    sway: (i, p) => geo.setSway(tree.sw(p[1]), tree.ph, 0.15 * (i / P.length), 0),
    ao: (i, th, r) => 1, cap: true,
  });
}
// curved path from a, direction d, length L, n points, bending toward `pull`
function curve(a, d, L, n, pull, rr, wob = 0.15) {
  const P = [a]; let p = a, dir = vnorm(d);
  for (let i = 1; i < n; i++) {
    dir = vnorm(vadd(dir, vadd(vmul(pull, 1 / n), [rr.n() * wob, rr.n() * wob * 0.5, rr.n() * wob])));
    p = vadd(p, vmul(dir, L / (n - 1))); P.push(p);
  }
  return P;
}
// a clump of canopy leaf-sprays around c
function crownClump(c, rc, tree, rr, o = {}) {
  const n = round(rc * rc * (o.dens ?? 13) * 1.15);
  const flat = o.flat ?? 0.62, size = (o.size ?? 1.15) * 1.3, kind = o.kind ?? 0;
  CLUMPS.push([c[0], c[1], c[2], rc]);
  for (let i = 0; i < n; i++) {
    const d = randDir(rr); d[1] = d[1] * 0.85 + 0.15;
    const sh = 0.5 + 0.5 * Math.cbrt(rr());
    const p = [c[0] + d[0] * rc * sh, c[1] + d[1] * rc * sh * flat, c[2] + d[2] * rc * sh];
    const dn = vnorm([d[0], d[1] * flat + 0.1, d[2]]);
    let nf = vnorm(vadd(vadd([0, 1.1, 0], vmul(dn, 0.9)), [rr.n() * 0.7, rr.n() * 0.4, rr.n() * 0.7]));
    let fw = vsub(dn, vmul(nf, vdot(dn, nf))); if (vlen(fw) < 0.1) fw = anyPerp(nf); fw = vnorm(vadd(vnorm(fw), [rr.n() * 0.5, -0.1, rr.n() * 0.5]));
    const ao = clamp(0.25 + 0.55 * sh * sh + 0.25 * (d[1] * 0.5 + 0.5), 0, 1);
    const top = d[1] > 0.4 && sh > 0.85;
    let col = jit(tree.leaf, rr, 0.07);
    if (tree.flush && top && rr() < 0.55) col = jit(tree.flush, rr, 0.08);
    else if (rr() < 0.04) col = [0.5, 0.42, 0.12]; // an old yellowing leaf
    leaf(kind, p, size * (0.8 + 0.45 * rr()), fw, nf, dn, ao, tree.sw(p[1]), 0.15 + rr() * 0.45, col, o.variant ?? 0, tree.ph + rr() * 0.3, rr());
  }
}

// ---------------------------------------------------------------- trees
let TREE_ID = 0;
function newTree(x, z, H, rr, o = {}) {
  const ph = rr(), base = gy(x, z);
  const amp = o.amp ?? 0.5;
  return { x, z, base, H, ph, tint: rr(), moss: o.moss ?? rr() * 0.3, mat: o.mat ?? 1, leaf: o.leaf ?? rr.pick(PAL), flush: o.flush, id: TREE_ID++,
    sw: (y) => amp * pow(clamp((y - base) / H, 0, 1.2), 1.6) };
}
// emergent dipterocarp: huge straight bole, buttress fins, umbrella crown of cauliflower clumps
function dipterocarp(geo, x, z, rr, sc = 1) {
  const H = rr.r(42, 58) * sc, r0 = rr.r(1.0, 1.65) * sc;
  const t = newTree(x, z, H, rr, { mat: rr() < 0.7 ? 1 : 4, flush: rr() < 0.4 ? rr.pick(FLUSH) : null, amp: 0.55 });
  TREES.push([x, z, r0 + 1.5, H]);
  geo.at(x, z);
  const nf = 3 + rr.i(3), fins = [];
  for (let i = 0; i < nf; i++) fins.push([(i / nf) * TAU + rr.n() * 0.5, rr.r(2.8, 6) * sc, rr.r(3.5, 8) * sc]);
  const lean = [rr.n() * 0.03, 0, rr.n() * 0.03];
  const trunkTop = t.base + H * 0.68;
  const hs = [-1.2, -0.3, 0.3, 0.8, 1.4, 2.1, 3, 4.1, 5.4, 7];
  const PB = hs.map((h) => [x + lean[0] * h, t.base + h, z + lean[2] * h]);
  const R = (i, th) => {
    const h = hs[i]; let r = r0 * (1 + 0.25 * sstep(3, -1, h));
    for (const [a, L, fh] of fins) { const e = L * pow(clamp(1 - h / fh, 0, 1), 1.7); const c = cos(th - a); r += e * pow(max(c, 0), 30 + 12 / (e + 0.3)) + e * 0.12 * pow(max(c, 0), 4); }
    return r * (1 + 0.04 * sin(th * 7 + h));
  };
  geo.tube(PB, R, 40, { mat: (i) => geo.setMat(t.mat, t.tint, mix(0.6, t.moss, sstep(-1, 7, hs[i])), 0.1), sway: () => geo.setSway(0, t.ph, 0, 0), ao: (i) => clamp(0.45 + hs[i] * 0.12, 0, 1) });
  // upper bole
  const PU = []; const n = 10;
  for (let i = 0; i <= n; i++) { const h = 6.6 + (trunkTop - t.base - 6.6) * (i / n); PU.push([x + lean[0] * h + sin(h * 0.15 + t.ph * 9) * 0.25, t.base + h, z + lean[2] * h + cos(h * 0.12 + t.ph * 7) * 0.25]); }
  geo.setMat(t.mat, t.tint, t.moss, 0);
  geo.tube(PU, PU.map((_, i) => r0 * (1.02 - 0.32 * (i / n))), 14, { sway: (i, p) => geo.setSway(t.sw(p[1]), t.ph, 0, 0), ao: (i) => 0.75 + 0.25 * i / n });
  if (rr() < 0.55) trunkVine(geo, t, PB[0], trunkTop, r0, rr);
  if (rr() < 0.5) epiphytes(t, PU, r0 * 0.75, rr);
  // crown: 4-6 limbs, each forking, clumps at the tips
  const top = PU[n], nl = 4 + rr.i(3), crownR = rr.r(9, 14) * sc;
  for (let i = 0; i < nl; i++) {
    const a = (i / nl) * TAU + rr.n() * 0.4, el = rr.r(0.55, 1.1);
    const d = [cos(a), el, sin(a)], L = crownR * rr.r(0.55, 0.85);
    const P = curve(vadd(top, [0, -rr() * 3, 0]), d, L, 6, [0, 0.8, 0], rr, 0.12);
    limb(geo, P, r0 * 0.45, r0 * 0.18, 9, t);
    const tip = P[P.length - 1];
    for (let j = 0; j < 2 + rr.i(2); j++) {
      const s = P[3 + rr.i(3) - 1], aa = a + rr.n() * 1.4;
      const Q = curve(s, [cos(aa), rr.r(0.3, 1.0), sin(aa)], rr.r(3, 6) * sc, 4, [0, 0.4, 0], rr, 0.2);
      limb(geo, Q, r0 * 0.16, 0.05, 6, t);
      crownClump(vadd(Q[3], [0, 1.2, 0]), rr.r(2.8, 4.2) * sc, t, rr);
      PERCH.push(vadd(Q[2], [0, 0.3, 0]));
    }
    crownClump(vadd(tip, [0, 1.5, 0]), rr.r(3.2, 4.6) * sc, t, rr);
    if (rr() < 0.5) lianaDrop(geo, tip, rr, t);
  }
  return t;
}
// main canopy and understory trees
function canopyTree(geo, x, z, rr, H, o = {}) {
  const r0 = o.r0 ?? rr.r(0.35, 0.75) * (0.4 + 0.6 * H / 28);
  const t = newTree(x, z, H, rr, { mat: rr.pick([1, 2, 2, 4, 1, 3]), flush: rr() < 0.25 ? rr.pick(FLUSH) : null, amp: 0.45 });
  TREES.push([x, z, r0 + 0.6, H]);
  geo.at(x, z);
  const forkH = H * rr.r(0.5, 0.68), n = 8;
  const lean = [rr.n() * 0.12, 0, rr.n() * 0.12];
  const P = [];
  for (let i = 0; i <= n; i++) { const h = -0.6 + (forkH + 0.6) * (i / n); P.push([x + lean[0] * h + sin(h * 0.3 + t.ph * 5) * 0.2, t.base + h, z + lean[2] * h + cos(h * 0.25 + t.ph * 3) * 0.2]); }
  geo.setMat(t.mat, t.tint, t.moss + 0.2, 0.05);
  const fins = H > 24 && rr() < 0.5 ? 3 : 0, fa = rr() * TAU;
  geo.tube(P, (i, th) => { const h = P[i][1] - t.base; let r = r0 * (1 - 0.3 * i / n); if (fins) r += r0 * 1.6 * pow(clamp(1 - h / 2.2, 0, 1), 2) * pow(max(cos((th - fa) * 1.5), 0), 8); return r; }, fins ? 18 : 10,
    { sway: (i, p) => geo.setSway(t.sw(p[1]), t.ph, 0, 0), ao: (i) => 0.55 + 0.45 * i / n });
  if (rr() < 0.25) trunkVine(geo, t, P[0], P[n][1], r0, rr);
  if (rr() < 0.3) epiphytes(t, P, r0, rr);
  const top = P[n], nl = 2 + rr.i(3), CR = H * rr.r(0.2, 0.3);
  const leafKind = o.broad ? 1 : 0;
  for (let i = 0; i < nl; i++) {
    const a = (i / nl) * TAU + rr.n() * 0.6;
    const Q = curve(top, [cos(a) * 0.8, 1, sin(a) * 0.8], (H - forkH) * rr.r(0.6, 0.9), 5, [0, 0.5, 0], rr, 0.2);
    limb(geo, Q, r0 * 0.55, r0 * 0.15, 7, t);
    for (let j = 0; j < 2; j++) {
      const s = Q[2 + j], aa = a + rr.n() * 1.6;
      const B = curve(s, [cos(aa), 0.5, sin(aa)], CR * rr.r(0.6, 1), 3, [0, 0.3, 0], rr, 0.25);
      limb(geo, B, r0 * 0.2, 0.03, 5, t);
      crownClump(vadd(B[2], [0, 0.6, 0]), CR * rr.r(0.45, 0.7), t, rr, leafKind ? { kind: 1, size: 0.55, dens: 9, variant: 0 } : { size: 1.0 });
      PERCH.push(B[1]);
    }
    crownClump(vadd(Q[4], [0, 0.8, 0]), CR * rr.r(0.55, 0.8), t, rr, leafKind ? { kind: 1, size: 0.55, dens: 9, variant: 0 } : { size: 1.0 });
  }
  return t;
}
// banyan: braided trunk strands, long horizontal limbs, curtains of aerial roots, sprawling surface roots
function banyan(geo, x, z, rr, o = {}) {
  const H = rr.r(24, 30), t = newTree(x, z, H, rr, { mat: 3, amp: 0.3, leaf: [0.10, 0.25, 0.08] });
  t.moss = 0.35;
  TREES.push([x, z, 4.5, H]);
  geo.at(x, z);
  const ns = 6 + rr.i(4), th = 13 + rr() * 3;
  for (let s = 0; s < ns; s++) {
    const a0 = (s / ns) * TAU + rr.n() * 0.3, rad = rr.r(1.2, 2.6), tw = rr.r(-0.6, 0.6);
    const P = []; for (let i = 0; i <= 9; i++) { const h = -1 + (th + 1) * (i / 9), a = a0 + tw * (h / th), rr2 = rad * (1 - 0.55 * (h / th)) + (h < 1.5 ? (1.5 - h) * 0.9 : 0); P.push([x + cos(a) * rr2, t.base + h, z + sin(a) * rr2]); }
    geo.setMat(3, t.tint, 0.5, 0.05);
    geo.tube(P, P.map((_, i) => rr.r(0.45, 0.85) * (1 - 0.3 * i / 9) + (i === 0 ? 0.4 : 0)), 10, { sway: (i, p) => geo.setSway(t.sw(p[1]) * 0.5, t.ph, 0, 0), ao: (i) => 0.5 + 0.5 * i / 9 });
  }
  const top = [x, t.base + th, z];
  const nl = o.limbs ?? 6 + rr.i(3);
  for (let i = 0; i < nl; i++) {
    const a = (o.limbA && i < o.limbA.length) ? o.limbA[i] : (i / nl) * TAU + rr.n() * 0.3;
    const L = (o.limbL ?? rr.r(11, 17));
    const P = curve(vadd(top, [cos(a) * 1.5, -rr() * 2, sin(a) * 1.5]), [cos(a), rr.r(0.15, 0.4), sin(a)], L, 7, [0, -0.2, 0], rr, 0.1);
    limb(geo, P, 0.7, 0.18, 9, t, 3);
    for (let k = 1; k < P.length; k++) {
      crownClump(vadd(P[k], [rr.n() * 2, 2 + rr() * 3, rr.n() * 2]), rr.r(3, 4.6), t, rr, { dens: 12, size: 1.0 });
      if (k % 2 === 0) PERCH.push(P[k]);
      // aerial roots: many thin, some thick pillar roots that reached the ground
      const nr = 2 + rr.i(4);
      for (let r = 0; r < nr; r++) {
        const s = vlerp(P[k - 1], P[k], rr());
        const thick = rr() < 0.18;
        const g = gy(s[0], s[2]);
        const len = thick ? s[1] - g + 0.5 : rr.r(3, s[1] - g - 1.5);
        if (len < 1) continue;
        const Q = []; const m = max(4, round(len / 1.5));
        for (let j = 0; j <= m; j++) { const f = j / m; Q.push([s[0] + sin(f * 3 + r) * 0.15 * f, s[1] - len * f, s[2] + cos(f * 2.3 + r) * 0.15 * f]); }
        geo.setMat(3, t.tint * 0.7, 0.2, 0);
        const r0 = thick ? rr.r(0.18, 0.4) : rr.r(0.025, 0.06);
        geo.tube(Q, Q.map((_, j) => r0 * (thick ? 1 + (j === m ? 0.6 : 0) : 1 - 0.6 * j / m)), thick ? 8 : 4, { sway: (j) => geo.setSway(thick ? 0 : 0.25 * (j / m), t.ph + r * 0.1, thick ? 0 : 0.6 * j / m, 0) });
      }
    }
  }
  // crown top
  for (let i = 0; i < 6; i++) crownClump(vadd(top, [rr.n() * 8, rr.r(6, 11), rr.n() * 8]), rr.r(3.5, 5), t, rr, { dens: 12 });
  // surface roots
  for (let i = 0; i < 9; i++) {
    const a = rr() * TAU, L = rr.r(5, 11), P = [];
    for (let j = 0; j <= 8; j++) { const f = j / 8, px = x + cos(a + sin(f * 3) * 0.3) * (2 + L * f), pz = z + sin(a + sin(f * 3) * 0.3) * (2 + L * f); P.push([px, gy(px, pz) + 0.05 - f * 0.25 + (1 - f) * 0.5, pz]); }
    geo.setMat(3, t.tint, 0.6, 0.1);
    geo.tube(P, P.map((_, j) => 0.35 * (1 - j / 9) + 0.04), 7, { sway: () => geo.setSway(0, 0, 0, 0) });
  }
  return t;
}
// philodendron-type climbers spiralling up a trunk, with big leaves pressed to the bark
function trunkVine(geo, t, b, topY, r0, rr) {
  const P = [], a0 = rr() * TAU, turns = rr.r(0.6, 1.8), h0 = rr.r(0, 2), h1 = min(topY - t.base, rr.r(8, 30));
  for (let i = 0; i <= 24; i++) { const f = i / 24, h = mix(h0, h1, f), a = a0 + f * turns * TAU, r = r0 * (1 - 0.2 * f) + 0.12 + (h < 6 ? sstep(6, 0, h) * 0.6 : 0); P.push([b[0] + cos(a) * r, t.base + h, b[2] + sin(a) * r]); }
  geo.setMat(8, 0.2, 0.3, 0);
  geo.tube(P, P.map(() => 0.035), 4, { sway: (i, p) => geo.setSway(t.sw(p[1]), t.ph, 0, 0) });
  const col = rr() < 0.5 ? [0.12, 0.30, 0.06] : [0.18, 0.36, 0.08];
  for (let i = 1; i < 24; i += 1) {
    if (rr() < 0.3) continue;
    const p = P[i], out = vnorm([p[0] - b[0], 0, p[2] - b[2]]);
    const nf = vnorm(vadd(out, [rr.n() * 0.3, 0.3, rr.n() * 0.3]));
    const fw = vnorm(vadd([0, -0.6, 0], vmul(vcross(nf, [0, 1, 0]), rr.n())));
    leaf(1, vadd(p, vmul(out, 0.05)), rr.r(0.28, 0.55) * (1 + i / 30), fw, nf, out, 0.7, t.sw(p[1]), 0.1, jit(col, rr, 0.06), rr() < 0.5 ? 1 : 0, rr(), rr());
  }
}
// bird's-nest ferns and orchids on trunks and forks
function epiphytes(t, P, r0, rr) {
  const n = 1 + rr.i(3);
  for (let e = 0; e < n; e++) {
    const i = 2 + rr.i(P.length - 3), p = P[i], a = rr() * TAU, out = [cos(a), 0, sin(a)];
    const c = vadd(p, vmul(out, r0 + 0.1));
    if (rr() < 0.6) {
      for (let k = 0; k < 12; k++) {
        const b = rr() * TAU, d = vnorm([cos(b) * 0.7 + out[0] * 0.5, 0.9, sin(b) * 0.7 + out[2] * 0.5]);
        leaf(1, c, rr.r(0.6, 1.1), d, vnorm(vadd([0, 1, 0], vmul(d, -0.6))), d, 0.7, t.sw(c[1]), 0.6, jit([0.32, 0.48, 0.1], rr, 0.08), 3, rr(), rr());
      }
    } else {
      const col = rr.pick([[0.9, 0.35, 0.75], [0.95, 0.95, 0.92], [0.75, 0.3, 0.85], [0.98, 0.7, 0.2]]);
      for (let k = 0; k < 5; k++) {
        const d = vnorm([out[0] + rr.n() * 0.5, -0.7, out[2] + rr.n() * 0.5]);
        leaf(1, c, rr.r(0.25, 0.4), d, vnorm(vadd(out, [0, 0.5, 0])), out, 0.7, t.sw(c[1]), 0.2, [0.22, 0.38, 0.1], 3, rr(), rr());
      }
      for (let k = 0; k < 7; k++) { const q = vadd(c, [out[0] * (0.2 + k * 0.05) + rr.n() * 0.1, -0.1 - k * 0.07, out[2] * (0.2 + k * 0.05) + rr.n() * 0.1]); leaf(4, q, rr.r(0.08, 0.11), vnorm(vadd(out, [0, 0.4, 0])), out, out, 1, t.sw(c[1]), 0, jit(col, rr, 0.05), rr() < 0.7 ? 0 : 1, rr(), 0.5); }
    }
  }
}
// a liana dropping from a limb in a long loop
function lianaDrop(geo, a, rr, t) {
  const g = gy(a[0], a[2]);
  const b = [a[0] + rr.n() * 8, g + rr.r(-0.2, 6), a[2] + rr.n() * 8];
  if (b[1] > a[1] - 4) return;
  const n = 26, P = [], sag = rr.r(2, 8);
  for (let i = 0; i <= n; i++) { const f = i / n; const p = vlerp(a, b, f); p[1] -= sag * 4 * f * (1 - f); p[0] += sin(f * 9 + a[0]) * 0.2; p[2] += cos(f * 7 + a[2]) * 0.2; P.push(p); }
  const r0 = rr.r(0.03, 0.08), twist = rr() < 0.4;
  geo.setMat(8, rr(), rr() * 0.4, 0);
  geo.tube(P, (i, th) => r0 * (twist ? 1 + 0.35 * sin(th * 2 + i * 1.3) : 1), 5, { sway: (i) => geo.setSway(0.55 * sin(PI * i / n) * (b[1] > g + 0.5 ? 1 : 0.4), t.ph + 0.37, 0.25 * sin(PI * i / n), 0) });
  if (rr() < 0.5) for (let i = 3; i < n - 2; i += 2) { const p = P[i], d = randDir(rr); leaf(3, p, rr.r(0.15, 0.3), vnorm([d[0], -0.5, d[2]]), vnorm([d[0], 0.6, d[2]]), d, 0.8, 0.55 * sin(PI * i / n), 0.2, jit([0.16, 0.34, 0.07], rr, 0.08), 0, t.ph, rr()); }
}
// curtains of thin hanging roots / vines from a high point
function hangCurtain(geo, a, rr, ph) {
  const n = 3 + rr.i(6);
  for (let k = 0; k < n; k++) {
    const s = [a[0] + rr.n() * 2, a[1] - rr() * 1.5, a[2] + rr.n() * 2], g = gy(s[0], s[2]);
    const len = min(rr.r(3, 14), s[1] - g - 0.3); if (len < 2) continue;
    const m = max(6, round(len / 1.2)), Q = [], wx = rr.n() * 1.2, wz = rr.n() * 1.2;
    for (let j = 0; j <= m; j++) { const f = j / m; Q.push([s[0] + sin(f * 3 + k) * 0.3 + wx * f * f, s[1] - len * f, s[2] + cos(f * 2.3 + k) * 0.3 + wz * f * f]); }
    geo.setMat(8, 0.3 + rr() * 0.5, 0.15, 0);
    geo.tube(Q, Q.map(() => rr.r(0.012, 0.025)), 3, { sway: (j) => geo.setSway(0.5 * (j / m), ph + k * 0.05, 0.7 * j / m, 0) });
  }
}
// bamboo grove: arching culms with node rings, sprays of narrow leaves on the upper half
function bambooGrove(geo, cx, cz, R, rr) {
  const n = round(R * R * 1.6 + 8), ph = rr();
  geo.at(cx, cz);
  for (let i = 0; i < n; i++) {
    const a = rr() * TAU, rd = sqrt(rr()) * R, x = cx + cos(a) * rd, z = cz + sin(a) * rd;
    if (waterDist(x, z) < 0.6) continue;
    const g = gy(x, z), H = rr.r(9, 17), r0 = rr.r(0.04, 0.075), lean = rr.r(0.05, 0.35) * (0.4 + rd / R);
    const out = [cos(a), 0, sin(a)], yel = rr() < 0.25 ? rr.r(0.6, 1) : rr.r(0, 0.3);
    const m = round(H / 0.42), P = [];
    for (let j = 0; j <= m; j++) { const f = j / m, h = H * f; P.push([x + out[0] * lean * h * f * 1.4, g - 0.2 + h * (1 - lean * f * 0.5), z + out[2] * lean * h * f * 1.4]); }
    const sw = (j) => 0.75 * pow(j / m, 1.8);
    const tph = ph + rr() * 0.15;
    geo._bt = 0;
    geo.tube(P, P.map((_, j) => r0 * (1 - 0.55 * j / m) * (1 + 0.08 * (j % 1))), 6, {
      mat: () => geo.setMat(9, 1, 0, 0),
      sway: (j) => geo.setSway(sw(j), tph, 0.25 * j / m, yel),
    });
    // node rings: a slim darker band at every node
    geo.setMat(9, 0.4, 0, 0);
    for (let j = 2; j < m; j += 1) {
      const p = P[j], q = P[j + 1], d = vnorm(vsub(q, p)), rn = r0 * (1 - 0.55 * j / m) * 1.18;
      geo.setSway(sw(j), tph, 0.25 * j / m, yel);
      geo.tube([vsub(p, vmul(d, 0.02)), vadd(p, vmul(d, 0.02))], [rn, rn], 6);
    }
    // leaves
    for (let j = round(m * 0.4); j < m; j++) {
      const k = rr() < 0.65 ? 2 : 1;
      for (let s = 0; s < k; s++) {
        const p = P[j], b = rr() * TAU, d = vnorm([cos(b) + out[0] * 0.8, rr.r(-0.6, 0.2), sin(b) + out[2] * 0.8]);
        const nf = vnorm(vadd([0, 1, 0], vmul(d, -0.3)));
        leaf(0, p, rr.r(0.7, 1.05), d, nf, vnorm(vadd(d, [0, 0.4, 0])), 0.4 + 0.6 * (j / m), sw(j), rr.r(0.4, 0.8), jit(rr() < 0.7 ? [0.26, 0.42, 0.09] : [0.38, 0.48, 0.12], rr, 0.08), 1, tph, rr());
      }
    }
    if (rr() < 0.15) PERCH.push(P[round(m * 0.7)]);
  }
}
// palms (fishtail) and tree ferns: slender trunk and a crown of fronds
function palm(geo, x, z, rr, fern) {
  const g = gy(x, z), H = fern ? rr.r(1.5, 3.5) : rr.r(3, 9), r0 = fern ? rr.r(0.12, 0.18) : rr.r(0.08, 0.13);
  const t = newTree(x, z, H, rr, { amp: 0.25 });
  geo.at(x, z);
  const lean = [rr.n() * 0.25, 0, rr.n() * 0.25], P = [];
  for (let i = 0; i <= 8; i++) { const h = -0.3 + (H + 0.3) * (i / 8); P.push([x + lean[0] * h * h / H, g + h, z + lean[2] * h * h / H]); }
  geo.setMat(12, rr(), fern ? 0.5 : 0.2, 0);
  geo.tube(P, P.map((_, i) => r0 * (fern ? 1 + 0.2 * (i / 8) : 1 - 0.2 * i / 8)), 7, { sway: (i, p) => geo.setSway(t.sw(p[1]), t.ph, 0, 0) });
  const top = P[8], nfr = fern ? 11 + rr.i(5) : 7 + rr.i(5);
  for (let i = 0; i < nfr; i++) {
    const a = (i / nfr) * TAU + rr.n() * 0.2, el = fern ? rr.r(0.35, 1.3) : rr.r(0.9, 2.2);
    const d = vnorm([cos(a), el, sin(a)]);
    const nf = vnorm(vadd([0, 1, 0], vmul(d, -0.7)));
    const L = fern ? rr.r(2.0, 3.0) : rr.r(2.2, 3.4);
    leaf(2, top, L, d, nf, d, 0.75, t.sw(top[1]) + 0.1, fern ? rr.r(0.9, 1.0) : rr.r(0.85, 1.0), jit(fern ? [0.24, 0.42, 0.09] : [0.12, 0.27, 0.06], rr, 0.06), fern ? 0 : 2, t.ph + rr() * 0.2, rr());
  }
  if (!fern) PERCH.push(top);
}
// a fallen, moss-covered trunk lying along the ground
function fallenLog(geo, x, z, a, L, r, rr) {
  if (nearPose(x, z, L * 0.5 + 6)) return;
  const P = [], dx = cos(a), dz = sin(a), n = round(L / 1.2);
  let maxg = -1e9;
  for (let i = 0; i <= n; i++) { const f = i / n, px = x + dx * L * (f - 0.5), pz = z + dz * L * (f - 0.5); maxg = max(maxg, gy(px, pz)); }
  for (let i = 0; i <= n; i++) { const f = i / n, px = x + dx * L * (f - 0.5), pz = z + dz * L * (f - 0.5); P.push([px, max(gy(px, pz) + r * 0.55, maxg - 2.5 + r) + sin(f * 3) * 0.1, pz]); }
  geo.at(x, z).setMat(11, rr(), 0.8, 0.3);
  geo.tube(P, (i, th) => r * (1 - 0.25 * i / n) * (1 + 0.06 * sin(th * 5 + i)), 12, { sway: () => geo.setSway(0, 0, 0, 0), cap: true, ao: (i, th) => 0.6 + 0.4 * max(sin(th), 0) });
  // ferns, mushrooms on the log
  for (let i = 1; i < n; i += 2) {
    const p = P[i];
    if (rr() < 0.5) fernPlant([p[0], p[1] + r * 0.8, p[2]], rr, 0.6);
    if (rr() < 0.5) mushrooms(geo, [p[0] + rr.n() * r, p[1] + r * 0.6, p[2] + rr.n() * r], rr, 0.6);
  }
}
function mushrooms(geo, c, rr, s = 1) {
  const n = 2 + rr.i(6), tint = rr();
  for (let i = 0; i < n; i++) {
    const p = [c[0] + rr.n() * 0.25, c[1], c[2] + rr.n() * 0.25], h = rr.r(0.03, 0.12) * s, cr = rr.r(0.025, 0.07) * s;
    geo.setMat(10, tint, 0, 0).setSway(0, 0, 0, 0);
    geo.lathe(p, [[0.008, -0.03], [cr * 0.25, 0], [cr * 0.22, h], [cr * 1.0, h], [cr * 0.9, h + cr * 0.35], [cr * 0.5, h + cr * 0.6], [0.001, h + cr * 0.7]], 7);
  }
}
// ground plants
function fernPlant(c, rr, s = 1) {
  const n = 6 + rr.i(8), col = jit(rr.pick([[0.22, 0.42, 0.08], [0.16, 0.36, 0.08], [0.30, 0.46, 0.10], [0.13, 0.30, 0.10]]), rr, 0.06), ph = rr();
  for (let i = 0; i < n; i++) {
    const a = (i / n) * TAU + rr.n() * 0.3, el = rr.r(0.3, 1.1);
    const d = vnorm([cos(a), el, sin(a)]);
    leaf(2, c, rr.r(0.8, 1.7) * s, d, vnorm(vadd([0, 1, 0], vmul(d, -el * 0.8))), d, 0.6 + 0.4 * rr(), 0.35, rr.r(0.4, 0.9), jit(col, rr, 0.04), 0, ph, rr());
  }
}
function broadPlant(c, rr, variant, s = 1) {
  const n = variant === 2 ? 0 : 3 + rr.i(5), ph = rr();
  const col = variant === 1 ? jit([0.10, 0.28, 0.08], rr, 0.05) : jit(rr.pick([[0.14, 0.32, 0.06], [0.24, 0.40, 0.08], [0.10, 0.24, 0.10]]), rr, 0.06);
  for (let i = 0; i < n; i++) {
    const a = rr() * TAU, el = rr.r(0.35, 1.2), h = (variant === 1 ? rr.r(0.25, 0.8) : rr.r(0.05, 0.35)) * s;
    const p = [c[0] + cos(a) * 0.35 * h, c[1] + h, c[2] + sin(a) * 0.35 * h];
    const d = vnorm([cos(a), el * 0.5, sin(a)]);
    leaf(1, p, rr.r(0.5, 1.0) * s * (variant === 1 ? 1.3 : 1), d, vnorm(vadd([0, 1, 0], vmul(d, 0.2))), vnorm([d[0], 0.6, d[2]]), 0.65 + 0.3 * rr(), 0.3, rr.r(0.1, 0.4), jit(col, rr, 0.04), variant, ph, rr());
  }
}
function banana(geo, x, z, rr) {
  const g = gy(x, z), H = rr.r(2.2, 3.6), ph = rr();
  geo.at(x, z).setMat(12, 0.6, 0.2, 0);
  const P = [[x, g - 0.2, z], [x, g + H * 0.5, z], [x + rr.n() * 0.1, g + H, z + rr.n() * 0.1]];
  geo.tube(P, [0.13, 0.11, 0.08], 8, { sway: (i) => geo.setSway(0.08 * i, ph, 0, 0) });
  const n = 6 + rr.i(4);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * TAU + rr.n() * 0.4, el = rr.r(0.5, 1.4);
    const d = vnorm([cos(a), el, sin(a)]);
    leaf(1, P[2], rr.r(1.6, 2.6), d, vnorm(vadd([0, 1, 0], vmul(d, -0.8))), vnorm([d[0], 0.5, d[2]]), 0.8, 0.35, rr.r(0.7, 0.95), jit(rr() < 0.15 ? [0.36, 0.36, 0.12] : [0.14, 0.30, 0.07], rr, 0.05), 2, ph, rr());
  }
}
// a dense leafy shrub: the undergrowth that makes the forest feel impenetrable
function shrub(c, rr, s = 1) {
  const R = rr.r(0.7, 1.5) * s, h = rr.r(0.6, 1.6) * s, n = round(R * R * rr.r(110, 160)), ph = rr();
  const col = jit(rr.pick([[0.10, 0.26, 0.06], [0.16, 0.33, 0.07], [0.08, 0.22, 0.08], [0.22, 0.36, 0.08], [0.13, 0.28, 0.12]]), rr, 0.05);
  const ctr = [c[0], c[1] + h, c[2]];
  for (let i = 0; i < n; i++) {
    const d = randDir(rr); d[1] = d[1] * 0.75 + 0.1;
    const sh = 0.55 + 0.45 * Math.cbrt(rr());
    const p = [ctr[0] + d[0] * R * sh, max(ctr[1] + d[1] * h * sh, c[1] + 0.15), ctr[2] + d[2] * R * sh];
    const nf = vnorm(vadd(vadd([0, 1, 0], vmul(d, 0.8)), [rr.n() * 0.6, 0, rr.n() * 0.6]));
    let fw = vsub(d, vmul(nf, vdot(d, nf))); if (vlen(fw) < 0.1) fw = anyPerp(nf); fw = vnorm(fw);
    leaf(1, p, rr.r(0.22, 0.42) * s, fw, nf, d, clamp(0.2 + 0.75 * sh * sh * (0.6 + 0.4 * d[1]), 0, 1), 0.25, rr.r(0.1, 0.4), jit(col, rr, 0.05), 0, ph, rr());
  }
}
function herbs(c, rr, col) {
  const n = 3 + rr.i(6), ph = rr();
  for (let i = 0; i < n; i++) {
    const a = rr() * TAU, el = rr.r(0.1, 1.0), d = vnorm([cos(a), el, sin(a)]);
    const p = [c[0] + cos(a) * 0.05, c[1] + rr.r(0, 0.25), c[2] + sin(a) * 0.05];
    leaf(3, p, rr.r(0.12, 0.32), d, vnorm(vadd([0, 1, 0], vmul(d, -el * 0.5))), vnorm([d[0], 0.7, d[2]]), 0.55 + 0.35 * rr(), 0.15, rr.r(0.1, 0.5), jit(col, rr, 0.07), 0, ph, rr());
  }
}
function flowers(c, rr, col, n = 5) {
  const ph = rr(), variant = col[3] ?? 0;
  for (let i = 0; i < n; i++) {
    const p = [c[0] + rr.n() * 0.5, c[1] + rr.r(0.1, 0.5), c[2] + rr.n() * 0.5];
    leaf(4, p, rr.r(0.05, 0.1), vnorm([rr.n(), 1, rr.n()]), [0, 1, 0], [0, 1, 0], 1, 0.15, 0, jit(col, rr, 0.06), variant, ph + i * 0.1, 0.5);
    leaf(3, [p[0], p[1] - 0.1, p[2]], 0.2, vnorm([rr.n(), 0.3, rr.n()]), [0, 1, 0], [0, 1, 0], 0.7, 0.15, 0.3, [0.2, 0.4, 0.08], 0, ph, 0.5);
  }
}
function boulder(geo, x, z, r, rr, o = {}) {
  if (slope(x, z) > 0.45 && !o.force) return;
  if (typeof nearPose === 'function' && nearPose(x, z, r + 7)) return;
  const g = gy(x, z), w = waterDist(x, z) < 1.5 ? 0.85 : 0.1;
  geo.at(x, z).setMat(5, rr(), o.moss ?? rr() * 0.6, w).setSway(0, 0, 0, 0);
  geo.blob([x, g + r * (o.sink ?? 0.15), z], r, rr() * 1e6 | 0, { sub: r > 2.5 ? 3 : 2, squash: o.squash ?? rr.r(0.55, 0.9), flat: 0.55, sx: rr.r(0.8, 1.3), sz: rr.r(0.8, 1.3) });
}

// ---------------------------------------------------------------- set pieces
// the shrine: stepped plinth, small sanctuary with a dark doorway, two-tier roof with upturned eaves, stele
function shrine(geo, rr) {
  const { x, z, rot } = SHRINE, g = gy(x, z) - 0.25;
  const ax = [cos(rot), 0, sin(rot)], az = [-sin(rot), 0, cos(rot)], up = [0, 1, 0];
  const K = 1.6, P = (u, y, w) => [x + (ax[0] * u + az[0] * w) * K, g + y * K, z + (ax[2] * u + az[2] * w) * K];
  const B = (u, y, w, hx, hy, hz, tilt = 0, taper = 1) => {
    hx *= K; hy *= K; hz *= K;
    const a = rotAxis(ax, up, tilt), c = rotAxis(az, up, tilt);
    geo.setMat(6, rr(), 0.4 + rr() * 0.3, 0.1);
    geo.box(P(u, y, w), hx, hy, hz, a, up, c, taper);
  };
  geo.at(x, z).setSway(0, 0, 0, 0);
  // plinth: blocks, some slightly displaced
  for (let i = -2; i <= 2; i++) for (let k = -2; k <= 2; k++) B(i * 1.25, 0.3, k * 1.25, 0.62, 0.42, 0.62, rr.n() * 0.03);
  for (let i = -1; i <= 1; i++) for (let k = -1; k <= 1; k++) B(i * 1.2, 0.95, k * 1.2, 0.6, 0.25, 0.6, rr.n() * 0.02);
  // steps toward the clearing (-az side)
  for (let s = 0; s < 3; s++) B(0, 0.12 + s * 0.28, -3.6 + s * 0.45, 1.0, 0.14 + s * 0.14, 0.25, rr.n() * 0.04);
  // sanctuary walls with a doorway facing the clearing
  const wy = 1.2 + 1.1, wh = 1.1;
  B(0, wy, 1.35, 1.5, wh, 0.18); B(-1.35, wy, 0, 0.18, wh, 1.5); B(1.35, wy, 0, 0.18, wh, 1.5);
  B(-0.95, wy, -1.35, 0.55, wh, 0.18); B(0.95, wy, -1.35, 0.55, wh, 0.18); B(0, wy + 0.75, -1.35, 0.42, 0.35, 0.18);
  // inner dark niche + altar stone
  B(0, 1.55, 0.6, 0.45, 0.35, 0.35);
  // roof tiers with upturned corner horns
  const roof = (y, h, s, s2) => {
    B(0, y, 0, s, h, s, 0, s2 / s);
    for (const [cu, cw] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const c = P(cu * s * 0.98, y - h * 0.4, cw * s * 0.98);
      const d = vnorm([ax[0] * cu + az[0] * cw, 0.9, ax[2] * cu + az[2] * cw]);
      geo.setMat(6, 0.4, 0.5, 0.1); geo.tube([c, vadd(c, vmul(d, 0.55)), vadd(c, vmul(vnorm(vadd(d, [0, 1.2, 0])), 1.0))], [0.18, 0.12, 0.04], 5);
    }
  };
  B(0, 3.5, 0, 1.75, 0.12, 1.75);
  roof(3.95, 0.35, 1.85, 1.1);
  B(0, 4.45, 0, 0.95, 0.18, 0.95);
  roof(4.85, 0.28, 1.2, 0.55);
  // finial: stacked lotus
  geo.setMat(6, 0.6, 0.6, 0.1);
  geo.lathe(P(0, 5.1, 0), [[0.3, 0], [0.36, 0.1], [0.2, 0.25], [0.24, 0.35], [0.12, 0.55], [0.05, 0.8], [0.0, 0.9]].map(([a, b]) => [a * K, b * K]), 10);
  // stele and two guardian posts
  B(-2.0, 0.9, -4.6, 0.38, 0.9, 0.1, 0.2); B(-2.0, 1.85, -4.6, 0.42, 0.08, 0.14, 0.2);
  for (const s of [-1, 1]) { B(s * 2.0, 0.55, -2.7, 0.22, 0.55, 0.22); geo.setMat(6, 0.5, 0.7, 0.1); geo.lathe(P(s * 2.0, 1.1, -2.7), [[0.2, 0], [0.24, 0.1], [0.12, 0.3], [0.0, 0.4]].map(([a, b]) => [a * K, b * K]), 8); }
  // tumbled blocks around
  for (let i = 0; i < 9; i++) { const a = rr() * TAU, d = rr.r(3.4, 5.5); B(cos(a) * d, 0.15, sin(a) * d, rr.r(0.25, 0.45), rr.r(0.15, 0.3), rr.r(0.25, 0.45), rr() * 3); }
  // moss, ferns and small plants growing on the shrine
  for (let i = 0; i < 10; i++) { const u0 = rr.n() * 1.6, w0 = rr.n() * 1.6; fernPlant(P(u0, 1.25, w0 + (abs(w0) < 1.4 && abs(u0) < 1.4 ? 2 : 0)), rr, 0.5); }
  for (let i = 0; i < 6; i++) herbs(P(rr.n() * 1.5, 3.65, rr.n() * 1.5), rr, [0.25, 0.45, 0.1]);
  for (let i = 0; i < 4; i++) fernPlant(P(rr.n() * 0.8, 4.65, rr.n() * 0.8), rr, 0.4);
  flowers(P(-0.4, 1.0, -3.6), rr, [0.95, 0.75, 0.15, 1], 3);
  FLY_CLEAR.push(P(0, 2, -7));
  // roots of the banyan draping over the roof and down the walls
  const bx = BANYAN_SHRINE[0], bz = BANYAN_SHRINE[1];
  for (let i = 0; i < 9; i++) {
    const side = rr.i(4), su = [1, -1, 0, 0][side], sw = [0, 0, 1, -1][side];
    const along = rr.r(-1.3, 1.3);
    const eu = su !== 0 ? su * 2.0 : along, ew = sw !== 0 ? sw * 2.0 : along;
    const Q = [vadd(P(along * 0.3 + 2, 7.5 + rr() * 2, 2.5), [bx - x, 0, bz - z].map((v) => v * 0.4)), P(along * 0.4, 5.25, along * 0.2), P(eu * 0.62, 4.35, ew * 0.62), P(eu * 0.93, 3.9, ew * 0.93), P(eu * 0.85, 3.3, ew * 0.85), P(eu * 0.8, 2.0, ew * 0.8), P(eu * 0.95, 1.2, ew * 0.95), P(eu * 1.55, 0.8, ew * 1.55), P(eu * 2.3, -0.1, ew * 2.3)];
    for (const q of Q.slice(1, -1)) { q[0] += rr.n() * 0.25; q[2] += rr.n() * 0.25; q[1] += rr.n() * 0.1; }
    const R = Q.map((_, j) => (0.16 - j * 0.012) * rr.r(0.7, 1.3));
    geo.setMat(3, 0.2 + rr() * 0.3, 0.25, 0.05).setSway(0, 0, 0, 0);
    geo.tube(resampleCurve(Q, 30), (j, th) => (R[min(floor(j / 3.4), R.length - 1)] * 1.3 + 0.03) * (1 + 0.15 * sin(th * 3 + j)), 7);
  }
}
const resampleCurve = (Q, n) => { const out = []; for (let i = 0; i <= n; i++) { const s = (i / n) * (Q.length - 1), k = min(floor(s), Q.length - 2), t = s - k; const a = Q[max(k - 1, 0)], b = Q[k], c = Q[k + 1], d = Q[min(k + 2, Q.length - 1)]; out.push([0, 1, 2].map((o) => 0.5 * (2 * b[o] + (c[o] - a[o]) * t + (2 * a[o] - 5 * b[o] + 4 * c[o] - d[o]) * t * t + (3 * b[o] - a[o] - 3 * c[o] + d[o]) * t * t * t))); } return out; };
// suspension bridge: planks on a catenary, rope handrails and hangers, A-frame posts at both ends
let BRIDGE_Y = 0;
function bridge(geo, rr) {
  const { x, z0, z1 } = BRIDGE;
  const y0 = gy(x, z0) + 0.35, y1 = gy(x, z1) + 0.35, L = abs(z1 - z0), sag = L * 0.07;
  BRIDGE_Y = (y0 + y1) / 2 - sag;
  const deck = (f) => [x + sin(f * PI) * 0.6, mix(y0, y1, f) - sag * 4 * f * (1 - f), mix(z0, z1, f)];
  const sw = (f) => 4 * f * (1 - f) * 0.6;
  geo.at(x, (z0 + z1) / 2);
  const n = round(L / 0.42);
  for (let i = 1; i < n; i++) {
    const f = i / n; if (rr() < 0.06) continue; // missing plank
    const p = deck(f), q = deck(f + 0.01), d = vnorm(vsub(q, p));
    const ax = [1, 0, 0], ay = vnorm(vcross(d, ax)), az = vcross(ax, ay);
    const tw = rr.n() * (rr() < 0.1 ? 0.25 : 0.04);
    geo.setMat(7, rr(), rr() * 0.4, 0.2).setSway(sw(f), 0.21, 0.3 * sw(f), 1);
    geo.box(vadd(p, [rr.n() * 0.05, 0, 0]), 0.8 + rr.n() * 0.05, 0.055, 0.17, rotAxis(ax, ay, tw), ay, rotAxis(az, ay, tw));
  }
  const rope = (off, yo, r) => {
    const P = []; for (let i = 0; i <= 40; i++) { const f = i / 40, p = deck(f); P.push([p[0] + off, p[1] + yo + (yo > 0 ? sag * 0.15 * 4 * f * (1 - f) : 0), p[2]]); }
    geo.setMat(8, 0.7, 0.2, 0);
    geo.tube(P, P.map(() => r), 5, { sway: (i) => geo.setSway(sw(i / 40), 0.21, 0.3 * sw(i / 40), 1) });
    return P;
  };
  for (const s of [-1, 1]) {
    rope(s * 0.78, -0.08, 0.05);
    const H = rope(s * 0.88, 1.05, 0.045);
    for (let i = 2; i < 40; i += 2) {
      const f = i / 40, p = deck(f);
      geo.setMat(8, 0.5, 0, 0);
      geo.tube([[p[0] + s * 0.72, p[1], p[2]], [p[0] + s * 0.8, p[1] + 0.5, p[2]], H[i]], [0.012, 0.012, 0.012], 3, { sway: () => geo.setSway(sw(f), 0.21, 0.3 * sw(f), 1) });
    }
  }
  // posts
  for (const [zz, yy, dir] of [[z0, y0, -1], [z1, y1, 1]]) {
    geo.setMat(7, 0.3, 0.6, 0.2).setSway(0, 0, 0, 0);
    for (const s of [-1, 1]) {
      geo.tube([[x + s * 0.95, yy - 1.2, zz + dir * 0.4], [x + s * 0.9, yy + 2.4, zz]], [0.24, 0.2], 8, { cap: true });
      geo.tube([[x + s * 0.9, yy + 1.1, zz], [x + s * 1.3, gy(x + s * 1.3, zz + dir * 2.5), zz + dir * 2.5]], [0.025, 0.025], 4);
    }
    geo.box([x, yy + 1.45, zz], 1.05, 0.08, 0.08);
    for (let i = 0; i < 3; i++) fernPlant([x + rr.n() * 2, gy(x, zz + dir * 2) , zz + dir * rr.r(1, 3)], rr, 0.8);
    hangCurtain(geo, [x + rr.n() * 2, yy + 9, zz + dir * 2], rr, 0.3);
  }
  FLY_CLEAR.push(deck(0.5));
}

// ---------------------------------------------------------------- water surfaces
// vertex: f32 x,y,z, flow.xy, fall, foam (28 B)
const WATER = new Grow(Float32Array, 1 << 16), WIDX = new Grow(Uint32Array, 1 << 16);
let nWV = 0;
const wv = (x, y, z, fx, fz, fall, foam) => { WATER.need(7); WATER.a.set([x, y, z, fx, fz, fall, foam], WATER.n); WATER.n += 7; return nWV++; };
const wq = (a, b, c, d) => { WIDX.need(6); WIDX.a.set([a, b, c, a, c, d], WIDX.n); WIDX.n += 6; };
const FALLS = []; // spray sources: x, y, z, scale
function waterRibbon(L, lev, hw, o = {}) {
  let prev = null;
  for (let i = 0; i < L.length; i++) {
    const a = L[max(i - 1, 0)], b = L[min(i + 1, L.length - 1)];
    let dx = b[0] - a[0], dz = b[1] - a[1]; const dl = hypot(dx, dz) || 1; dx /= dl; dz /= dl;
    const drop = i < L.length - 1 ? lev[i] - lev[i + 1] : 0;
    const sp = 0.25 + clamp(drop * 2.5, 0, 2.5) + (o.speed ?? 0);
    const foam = clamp((drop - 0.25) * 1.2, 0, 0.8);
    const w = hw(i) + 0.9;
    const l = wv(L[i][0] - dz * w, lev[i], L[i][1] + dx * w, dx * sp, dz * sp, 0, foam);
    const r = wv(L[i][0] + dz * w, lev[i], L[i][1] - dx * w, dx * sp, dz * sp, 0, foam);
    const inPool = o.pools && o.pools.some((p) => hypot(L[i][0] - p[0], L[i][1] - p[1]) < p[2] - 1 && hypot(L[max(i - 1, 0)][0] - p[0], L[max(i - 1, 0)][1] - p[1]) < p[2] - 1);
    if (prev && !inPool) wq(prev[0], l, r, prev[1]);
    prev = [l, r];
    if (drop > 0.8) FALLS.push([L[i][0] + dx, lev[i + 1], L[i][1] + dz, 0.5]);
  }
}
function waterfall(top, bottomY, dir, w0, w1) {
  // a curtain bending outward from the lip
  const n = 24, H = top[1] - bottomY;
  const side = vnorm(vcross(dir, [0, 1, 0]));
  let prev = null;
  for (let i = 0; i <= n; i++) {
    const f = i / n, y = top[1] - H * f, out = 2.2 * sqrt(f * H / 9.8) * 1.2;
    const c = [top[0] + dir[0] * out, y, top[2] + dir[2] * out], w = mix(w0, w1, f) / 2;
    const l = wv(c[0] - side[0] * w, c[1], c[2] - side[2] * w, -1, 0, 1, f > 0.85 ? 1 : 0.3);
    const r = wv(c[0] + side[0] * w, c[1], c[2] + side[2] * w, 1, 0, 1, f > 0.85 ? 1 : 0.3);
    if (prev) wq(prev[0], prev[1], r, l); // double-sided via cull none
    prev = [l, r];
  }
  FALLS.push([top[0] + dir[0] * 2.5, bottomY, top[2] + dir[2] * 2.5, 1.6]);
}

// ---------------------------------------------------------------- creatures (meshes in the static vertex format)
function birdMesh() {
  const g = new Geo(); g.setSway(0, 0, 0, 0);
  g._u[0] = 0; g.mat = g._f[0];
  g.lathe([0, 0, -0.1], [[0.0, 0], [0.035, 0.03], [0.05, 0.09], [0.045, 0.15], [0.035, 0.19], [0.04, 0.22], [0.035, 0.26], [0.0, 0.28]], 8, [0, 0, 1]);
  g._u[0] = 3 << 8; g.mat = g._f[0];
  g.lathe([0, 0.0, 0.17], [[0.013, 0], [0.0, 0.05]], 5, [0, 0, 1]);
  g._u[0] = 2 << 8; g.mat = g._f[0];
  // tail
  const t0 = g.v(-0.03, 0, -0.1, 0, 1, 0), t1 = g.v(0.03, 0, -0.1, 0, 1, 0), t2 = g.v(0.05, -0.01, -0.22, 0, 1, 0), t3 = g.v(-0.05, -0.01, -0.22, 0, 1, 0); g.quad(t0, t1, t2, t3);
  g._u[0] = 1 << 8; g.mat = g._f[0];
  for (const s of [-1, 1]) {
    const span = [0.02, 0.12, 0.2];
    const ids = span.map((x, i) => { g.setSway(i / 2 + 0.01, 0, 0, 0); const a = g.v(s * x, 0.02, 0.06 - i * 0.015, 0, 1, 0); const b = g.v(s * x, 0.02, -0.04 - i * 0.02 + (i === 2 ? 0.07 : 0), 0, 1, 0); return [a, b]; });
    g._u[0] = 1 << 8; g.mat = g._f[0];
    for (let i = 0; i < 2; i++) if (s > 0) g.quad(ids[i][0], ids[i][1], ids[i + 1][1], ids[i + 1][0]); else g.quad(ids[i][0], ids[i + 1][0], ids[i + 1][1], ids[i][1]);
    g.setSway(0, 0, 0, 0);
  }
  return g;
}
function butterflyMesh() {
  const g = new Geo(); g._u[0] = 3 << 8; g.mat = g._f[0]; g.setSway(0, 0, 0, 0);
  g.lathe([0, 0, -0.012], [[0, 0], [0.002, 0.005], [0.002, 0.025], [0, 0.03]], 4, [0, 0, 1]);
  g._u[0] = 0; g.mat = g._f[0];
  for (const s of [-1, 1]) {
    for (const [z0, z1, w] of [[0.004, 0.045, 0.05], [-0.004, -0.04, 0.042]]) {
      g.setSway(0.01, 0, 0, 0); const a = g.v(s * 0.002, 0, z0 * 0.2, 0, 1, 0), b = g.v(s * 0.002, 0, z1 * 0.3, 0, 1, 0);
      g.setSway(1, 0, 0, 0); const c = g.v(s * w, 0, z1, 0, 1, 0), d = g.v(s * w * 0.9, 0, z0, 0, 1, 0);
      if (s > 0) g.quad(a, b, c, d); else g.quad(a, d, c, b);
    }
  }
  return g;
}
function dragonflyMesh() {
  const g = new Geo(); g._u[0] = 0; g.mat = g._f[0]; g.setSway(0, 0, 0, 0);
  g.lathe([0, 0, -0.05], [[0, 0], [0.003, 0.01], [0.003, 0.06], [0.006, 0.075], [0.005, 0.085], [0, 0.09]], 4, [0, 0, 1]);
  g._u[0] = 2 << 8; g.mat = g._f[0];
  for (const s of [-1, 1]) for (const z0 of [0.03, 0.015]) {
    g.setSway(0.01, 0, 0, 0); const a = g.v(s * 0.003, 0.004, z0, 0, 1, 0), b = g.v(s * 0.003, 0.004, z0 - 0.008, 0, 1, 0);
    g.setSway(1, 0, 0, 0); const c = g.v(s * 0.045, 0.004, z0 - 0.01, 0, 1, 0), d = g.v(s * 0.045, 0.004, z0 - 0.002, 0, 1, 0);
    if (s > 0) g.quad(a, b, c, d); else g.quad(a, d, c, b);
  }
  return g;
}
const meshOut = (g) => { const v = new Float32Array(g.f.view()); const I = []; for (const c of g.idx) for (let i = 0; i < c.n; i++) I.push(c.a[i]); return { v, i: new Uint32Array(I) }; };

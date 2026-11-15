// ============================================================================
// WGSL: shared prelude, scene shaders, post-processing
// ============================================================================
// Uniform layout (floats): 0 vp | 16 shadow | 32 cam.xyz,time | 36 right,tanX | 40 up,tanY | 44 fwd,near
//   48 sun.xyz,exposure | 52 W,H,1/W,1/H | 56 lod,frame,fogLum,shaftK | 60 map ox,oz,1/size,mapH | 64 planes[6] | 88 misc
const W_COMMON = /*wgsl*/`
diagnostic(off, derivative_uniformity);
struct U {
  vp: mat4x4f, sm: mat4x4f,
  cam: vec4f, right: vec4f, up: vec4f, fwd: vec4f,
  sun: vec4f, scr: vec4f, prm: vec4f, wld: vec4f,
  planes: array<vec4f, 6>, misc: vec4f, misc2: vec4f,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var sRep: sampler;
@group(0) @binding(2) var sClamp: sampler;
@group(0) @binding(3) var sCmp: sampler_comparison;
@group(0) @binding(4) var tShadow: texture_depth_2d;
@group(0) @binding(5) var tNoise: texture_3d<f32>;
@group(0) @binding(6) var tMap: texture_2d<f32>;
const PI = 3.14159265;
const SUNC = vec3f(1.0, 0.82, 0.58) * 7.0;
fn ih(x0: u32) -> u32 { var x = x0; x = (x ^ (x >> 16u)) * 0x7feb352du; x = (x ^ (x >> 15u)) * 0x846ca68bu; return x ^ (x >> 16u); }
fn hf(x: u32) -> f32 { return f32(ih(x)) * (1.0 / 4294967295.0); }
fn n3(p: vec3f) -> vec4f { return textureSampleLevel(tNoise, sRep, p, 0.0); }
// world map: r ground height, g canopy optical depth, b canopy mid height, a water proximity
fn mapAt(xz: vec2f) -> vec4f { return textureSampleLevel(tMap, sClamp, (xz - u.wld.xy) * u.wld.z, 0.0); }
fn srgb(c: vec3f) -> vec3f { return pow(c, vec3f(2.2)); }
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
// HDR values are stored compressed so the hardware MSAA resolve does not alias bright edges
fn enc(c0: vec3f) -> vec4f { var c = max(c0, vec3f(0.0)); if (any(c != c) || any(c > vec3f(1e5))) { c = vec3f(0.0); } let m = max(c.r, max(c.g, c.b)); return vec4f(c / (1.0 + m), 1.0); }
fn dec(c0: vec3f) -> vec3f { var c = max(c0, vec3f(0.0)); let m0 = max(c.r, max(c.g, c.b)); if (m0 > 0.95) { c *= 0.95 / m0; } let m = max(c.r, max(c.g, c.b)); return c / (1.0 - m); }
fn windAt(p: vec3f, ph: f32) -> vec3f {
  let t = u.cam.w;
  let g = 0.6 + 0.4 * sin(t * 0.23 + dot(p.xz, vec2f(0.011, 0.007)));
  let s = sin(t * 0.55 + ph * 6.283 + dot(p.xz, vec2f(0.021, 0.014))) * 0.6 + sin(t * 1.37 + ph * 17.0) * 0.22 + 0.35;
  return (vec3f(0.82, 0.0, 0.57) * s + vec3f(-0.57, 0.0, 0.82) * sin(t * 0.71 + ph * 9.0) * 0.3) * g;
}
// sun visibility: static shadow map, 5-tap PCF, with a wind-driven wobble that keeps the dapples alive
fn shadowAt(wp: vec3f, soft: f32) -> f32 {
  let t = u.cam.w;
  let j = (n3(wp * 0.031 + vec3f(t * 0.012, 0.0, t * 0.009)).xy - 0.5) * 0.22;
  let p = wp + vec3f(j.x, 0.0, j.y);
  let l = u.sm * vec4f(p, 1.0);
  let uv = l.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 1.0; }
  let z = l.z - 0.0004;
  let o = soft / 8192.0;
  var s = textureSampleCompareLevel(tShadow, sCmp, uv, z) * 2.0;
  s += textureSampleCompareLevel(tShadow, sCmp, uv + vec2f(o, o * 0.4), z);
  s += textureSampleCompareLevel(tShadow, sCmp, uv + vec2f(-o * 0.4, o), z);
  s += textureSampleCompareLevel(tShadow, sCmp, uv + vec2f(-o, -o * 0.4), z);
  s += textureSampleCompareLevel(tShadow, sCmp, uv + vec2f(o * 0.4, -o), z);
  return s / 6.0;
}
// fraction of open sky above a point (canopy optical depth from the map, fading out above the crowns)
fn skyVis(p: vec3f) -> f32 {
  let m = mapAt(p.xz);
  let above = smoothstep(m.b - 10.0, m.b + 14.0, p.y);
  return exp(-m.g * (1.0 - above));
}
fn ambient(n: vec3f, p: vec3f) -> vec3f { return ambientV(n, skyVis(p)); }
fn ambientV(n: vec3f, v: f32) -> vec3f {
  let sky = vec3f(0.42, 0.62, 0.78) * (0.75 + 0.25 * n.y);
  let leaf = vec3f(0.10, 0.17, 0.07) * (0.7 + 0.3 * n.y);   // light filtered through layers of leaves
  let bounce = vec3f(0.05, 0.06, 0.03) * (0.6 - 0.4 * n.y);
  return sky * v * 1.25 + leaf * (0.35 + 0.65 * sqrt(1.0 - v)) * (0.4 + 0.6 * v + 0.35) + bounce;
}
// analytic height fog: warm golden-green close by, cool blue-green far away
fn fogT(p: vec3f) -> f32 {
  let d = p - u.cam.xyz; let dist = length(d); let rd = d / max(dist, 1e-3);
  let b = 1.0 / 22.0;
  let a = u.misc.x * exp(-b * (u.cam.y - u.misc.y));
  let k = rd.y * b;
  let tau = select(a * dist, a * (1.0 - exp(-k * dist)) / k, abs(k) > 1e-4);
  return exp(-tau - max(dist - 150.0, 0.0) * 0.004);
}
fn fogCol(rd: vec3f, dist: f32) -> vec3f {
  let warm = vec3f(0.34, 0.40, 0.20);
  let cool = vec3f(0.09, 0.19, 0.25);
  var c = mix(warm, cool, 1.0 - exp(-dist / 55.0));
  let mu = max(dot(rd, u.sun.xyz), 0.0);
  c += vec3f(1.0, 0.75, 0.42) * (pow(mu, 8.0) * 0.3 + pow(mu, 40.0) * 0.5);
  return c * u.prm.z;
}
fn fogApply(c: vec3f, p: vec3f) -> vec3f {
  let d = p - u.cam.xyz; let dist = length(d);
  let T = fogT(p);
  return c * T + fogCol(d / max(dist, 1e-3), dist) * (1.0 - T);
}
fn hg(mu: f32, g: f32) -> f32 { let g2 = g * g; return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * mu, 1.5)); }
fn sky(rd: vec3f) -> vec3f {
  let y = max(rd.y, 0.0);
  var c = mix(vec3f(0.95, 1.05, 1.05), vec3f(0.30, 0.55, 1.1), pow(y, 0.45)) * 1.7;
  let mu = dot(rd, u.sun.xyz);
  c += vec3f(1.0, 0.85, 0.6) * (pow(max(mu, 0.0), 6.0) * 2.5 + pow(max(mu, 0.0), 64.0) * 6.0);
  c += vec3f(1.0, 0.9, 0.7) * smoothstep(0.99975, 0.99985, mu) * 60.0;
  return c;
}
`;

// ---------------------------------------------------------------- noise texture bake (compute)
const W_NOISE = /*wgsl*/`
@group(0) @binding(0) var dst: texture_storage_3d<rgba8unorm, write>;
fn ih(x0: u32) -> u32 { var x = x0; x = (x ^ (x >> 16u)) * 0x7feb352du; x = (x ^ (x >> 15u)) * 0x846ca68bu; return x ^ (x >> 16u); }
fn grad(c: vec3i, per: i32) -> vec3f {
  let w = ((c % per) + per) % per;
  let h = ih((u32(w.x) * 73856093u) ^ (u32(w.y) * 19349663u) ^ (u32(w.z) * 83492791u + u32(per) * 7919u));
  let a = f32(h & 1023u) / 1023.0 * 6.2831853; let z = f32((h >> 10u) & 1023u) / 1023.0 * 2.0 - 1.0; let r = sqrt(1.0 - z * z);
  return vec3f(r * cos(a), r * sin(a), z);
}
fn pn(p: vec3f, per: i32) -> f32 {
  let i = vec3i(floor(p)); let f = fract(p); let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var acc = 0.0;
  let n000 = dot(grad(i, per), f); let n100 = dot(grad(i + vec3i(1, 0, 0), per), f - vec3f(1, 0, 0));
  let n010 = dot(grad(i + vec3i(0, 1, 0), per), f - vec3f(0, 1, 0)); let n110 = dot(grad(i + vec3i(1, 1, 0), per), f - vec3f(1, 1, 0));
  let n001 = dot(grad(i + vec3i(0, 0, 1), per), f - vec3f(0, 0, 1)); let n101 = dot(grad(i + vec3i(1, 0, 1), per), f - vec3f(1, 0, 1));
  let n011 = dot(grad(i + vec3i(0, 1, 1), per), f - vec3f(0, 1, 1)); let n111 = dot(grad(i + vec3i(1, 1, 1), per), f - vec3f(1, 1, 1));
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y), mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}
fn cell(p: vec3f, per: i32) -> f32 {
  let i = vec3i(floor(p)); let f = fract(p); var d = 9.0;
  for (var z = -1; z <= 1; z++) { for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    let o = vec3i(x, y, z); let w = (((i + o) % per) + per) % per;
    let h = ih(u32(w.x) * 7u + u32(w.y) * 1013u + u32(w.z) * 104729u + 99u);
    let r = vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0;
    d = min(d, length(vec3f(o) + r - f));
  } } }
  return d;
}
@compute @workgroup_size(4, 4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = vec3f(id) / 128.0;
  let a = pn(p * 8.0, 8) * 0.5 + pn(p * 16.0, 16) * 0.25 + pn(p * 32.0, 32) * 0.125;
  let b = pn(p * 16.0 + 3.0, 16) * 0.6 + pn(p * 32.0 + 3.0, 32) * 0.3;
  let c = pn(p * 32.0 + 7.0, 32) * 0.6 + pn(p * 64.0 + 7.0, 64) * 0.35;
  let d = cell(p * 16.0, 16);
  textureStore(dst, id, vec4f(a * 1.1 + 0.5, b * 0.9 + 0.5, c * 0.9 + 0.5, clamp(d, 0.0, 1.0)));
}`;

// ---------------------------------------------------------------- static geometry (terrain, trunks, rocks, set pieces)
const W_SOLID = W_COMMON + /*wgsl*/`
override SHADOW: bool = false;
override CHEAP: bool = false;
struct VI { @location(0) p: vec3f, @location(1) n: vec4f, @location(2) m: vec4f, @location(3) s: vec4f }
struct VO { @builtin(position) pos: vec4f, @location(0) wp: vec3f, @location(1) n: vec3f, @location(2) @interpolate(flat) m: vec4f, @location(3) ao: f32, @location(4) s: vec4f }
fn sway(p: vec3f, s: vec4f) -> vec3f {
  var o = windAt(p, s.y) * s.x * 1.2;
  o += vec3f(sin(u.cam.w * 2.1 + s.y * 40.0 + p.y * 0.7), 0.0, cos(u.cam.w * 1.7 + s.y * 31.0 + p.y * 0.5)) * s.z * 0.06;
  return o;
}
@vertex fn vs(i: VI) -> VO {
  var o: VO;
  let wp = i.p + sway(i.p, i.s);
  o.pos = u.vp * vec4f(wp, 1.0);
  o.wp = wp; o.n = i.n.xyz; o.ao = i.n.w * 0.5 + 0.5; o.m = i.m; o.s = i.s;
  return o;
}
@fragment fn fsShadow() {}
fn bump(p: vec3f, n: vec3f, k: f32, fw: f32) -> vec3f {
  let a = n3(p * 0.37).xyz - 0.5;
  let fade = clamp(1.0 - fw * 3.0, 0.0, 1.0);
  var b = vec3f(0.0); if (fade > 0.0) { b = n3(p * 1.9).xyz - 0.5; }
  return normalize(n + (a * 0.6 + b * 0.5 * fade) * k);
}
struct Mat { alb: vec3f, n: vec3f, rough: f32, spec: f32, sss: f32 }
// two rotated, incommensurate samples: hides the period of the noise texture
fn nt(p: vec3f) -> vec4f {
  let q = vec3f(p.x * 0.8 - p.z * 0.6, p.y, p.x * 0.6 + p.z * 0.8) * 1.371 + vec3f(0.31, 0.17, 0.53);
  let a = (n3(p) + n3(q)) * 0.5;
  return clamp(a * 1.45 - 0.225, vec4f(0.0), vec4f(1.0));
}
// fallen leaves: three layers of randomly rotated leaf shapes on a jittered grid
fn litter(p: vec2f, fw: f32) -> vec4f {
  var col = vec3f(0.0); var m = 0.0;
  for (var l = 0; l < 3; l++) {
    let sc = 5.0 + f32(l) * 1.9;
    let q = p * sc + vec2f(f32(l) * 17.3, f32(l) * 7.1);
    let c = vec2i(floor(q)); let f = fract(q) - 0.5;
    let h = ih((u32(c.x + 65536) * 7919u) ^ (u32(c.y + 65536) * 104729u) ^ (u32(l) * 2654435761u));
    let a = f32(h & 1023u) / 1023.0 * 6.283;
    let off = (vec2f(f32((h >> 10u) & 255u), f32((h >> 18u) & 255u)) / 255.0 - 0.5) * 0.3;
    let r = mat2x2f(cos(a), sin(a), -sin(a), cos(a)) * (f - off);
    let e = length(r * vec2f(1.0 / 0.44, 1.0 / (0.17 + 0.06 * f32((h >> 7u) & 7u) / 7.0)));
    let d = (1.0 - e) * 0.17 / sc;
    let cov = clamp(d / max(fw * 0.7, 1e-4) + 0.5, 0.0, 1.0) * step(f32((h >> 26u) & 63u) / 63.0, 0.9);
    let k = f32((h >> 3u) & 63u) / 63.0;
    var lc = mix(vec3f(0.14, 0.065, 0.028), vec3f(0.24, 0.13, 0.05), k);
    if (k > 0.78) { lc = vec3f(0.05, 0.03, 0.015); } else if (k < 0.1) { lc = vec3f(0.30, 0.20, 0.05); } else if (k < 0.16) { lc = vec3f(0.12, 0.13, 0.04); }
    lc *= 0.8 + 0.35 * smoothstep(0.0, 0.25, abs(r.y) * sc * 2.0) - 0.25 * (1.0 - e);
    col = mix(col, lc, cov); m = max(m, cov);
  }
  // far away the pattern dissolves into its average colour
  let fade = smoothstep(0.004, 0.02, fw);
  return vec4f(mix(col, vec3f(0.13, 0.07, 0.03), fade), mix(m, 0.85, fade));
}
@fragment fn fs(i: VO, @builtin(front_facing) ff: bool) -> @location(0) vec4f {
  if (CHEAP) { return enc(abs(i.n) * 0.3); }
  let p = i.wp; var n0 = normalize(i.n); if (!ff) { n0 = -n0; }
  let fw = length(fwidth(p));
  let id = u32(i.m.x * 255.0 + 0.5); let tint = i.m.y; var moss = i.m.z; let wet0 = i.m.w;
  let nA = nt(p * 0.045); let nB = nt(p * 0.19); let nC = n3(p * 0.83);
  let fade = clamp(1.0 - fw * 2.5, 0.0, 1.0);
  var m: Mat; m.n = n0; m.rough = 0.8; m.spec = 0.03; m.sss = 0.0;
  var wet = wet0;
  if (id == 0u) {
    // terrain: leaf litter over dark soil on gentle ground, limestone on steep ground, moss in between
    let rocky = max(smoothstep(0.8, 0.52, n0.y + (nB.x - 0.5) * 0.35), tint * smoothstep(0.95, 0.7, n0.y));
    var soil = mix(vec3f(0.045, 0.03, 0.018), vec3f(0.08, 0.05, 0.028), nA.x);
    var lt = vec4f(0.13, 0.07, 0.03, 0.85); if (fw < 0.02) { lt = litter(p.xz, fw); }
    soil = mix(soil, lt.rgb, lt.a * (0.55 + 0.45 * smoothstep(0.3, 0.6, nA.z)));
    let gm = smoothstep(0.5, 0.75, nA.y + nB.y * 0.3) * smoothstep(0.6, 0.9, n0.y);
    soil = mix(soil, vec3f(0.04, 0.07, 0.02) * (0.7 + 0.6 * nC.y), gm * 0.6);
    // limestone: grey with dark vertical rain streaks, green algae and orange iron stains
    let streak = n3(vec3f(p.x * 0.035, p.y * 0.004, p.z * 0.035)).x;
    let big = n3(p * 0.006).x;
    var rock = mix(vec3f(0.12, 0.12, 0.11), vec3f(0.22, 0.215, 0.19), smoothstep(0.3, 0.7, big)) * (0.85 + 0.3 * nB.x);
    rock = mix(rock, vec3f(0.03, 0.035, 0.03), smoothstep(0.45, 0.65, streak) * 0.9);
    rock = mix(rock, vec3f(0.12, 0.16, 0.08), smoothstep(0.5, 0.7, nA.y) * 0.45);
    rock = mix(rock, vec3f(0.30, 0.18, 0.09), smoothstep(0.64, 0.8, nA.z) * 0.45);
    rock *= 0.7 + 0.6 * nC.x * fade + 0.3 * (1.0 - fade);
    let ledge = smoothstep(0.3, 0.7, n0.y) * smoothstep(0.3, 0.55, nA.w + nB.z * 0.5);
    moss = max(moss, ledge * 0.95 + smoothstep(0.55, 0.8, nA.y) * 0.35 * (1.0 - smoothstep(0.0, 0.5, -n0.y)));
    m.alb = mix(soil, rock, rocky);
    m.n = bump(p, n0, mix(0.3, 0.55, rocky), fw);
    wet = max(wet, 0.2 * (1.0 - rocky));
    m.rough = mix(0.85, 0.55, rocky);
    moss = moss * mix(0.5, 1.0, rocky);
  } else if (id <= 4u || id == 12u) {
    // bark: 1 pale dipterocarp, 2 dark wet, 3 banyan, 4 reddish, 12 palm / tree-fern trunk
    let fiss = n3(vec3f(p.x * 0.35, p.y * 0.03, p.z * 0.35)).x;
    var b = vec3f(0.22, 0.20, 0.16);
    if (id == 2u) { b = vec3f(0.07, 0.055, 0.04); }
    if (id == 3u) { b = vec3f(0.19, 0.18, 0.15); }
    if (id == 4u) { b = vec3f(0.17, 0.08, 0.045); }
    if (id == 12u) { b = vec3f(0.06, 0.045, 0.03); }
    b *= 0.75 + 0.5 * tint;
    b *= 0.8 + 0.4 * nB.y;
    b = mix(b, b * 0.45, smoothstep(0.45, 0.65, fiss) * fade);
    // lichen blotches (white / mint) on the pale trunks
    let lich = smoothstep(0.62, 0.72, n3(p * 0.33 + 5.0).z) * select(0.35, 0.9, id == 1u || id == 3u);
    b = mix(b, mix(vec3f(0.36, 0.37, 0.31), vec3f(0.2, 0.3, 0.22), nC.y), lich);
    m.alb = b;
    m.n = bump(p, n0, 0.55, fw);
    // moss: thick at the base and on upward faces
    moss = max(moss, smoothstep(0.1, 0.9, n0.y + nA.y * 0.6 - 0.2) * 0.8);
    m.rough = 0.8;
  } else if (id == 5u) {
    var r = mix(vec3f(0.17, 0.17, 0.155), vec3f(0.27, 0.26, 0.24), nB.x) * (0.8 + 0.4 * nC.y * fade);
    r = mix(r, vec3f(0.07, 0.07, 0.065), smoothstep(0.55, 0.75, n3(vec3f(p.x * 0.06, p.y * 0.012, p.z * 0.06)).x) * 0.6);
    m.alb = r * (0.8 + 0.4 * tint); m.n = bump(p, n0, 0.5, fw); m.rough = 0.55;
    moss = max(moss, smoothstep(0.2, 0.8, n0.y + nA.y * 0.5 - 0.1));
  } else if (id == 6u) {
    // shrine stone: weathered grey-brown blocks, black water stains, lichen
    var r = mix(vec3f(0.17, 0.155, 0.13), vec3f(0.26, 0.24, 0.2), nB.x) * (0.75 + 0.5 * nC.x * fade);
    r = mix(r, vec3f(0.05, 0.05, 0.045), smoothstep(0.5, 0.75, n3(vec3f(p.x * 0.12, p.y * 0.025, p.z * 0.12)).x) * 0.7);
    r = mix(r, vec3f(0.34, 0.33, 0.26), smoothstep(0.7, 0.8, nC.z) * 0.6);
    m.alb = r * (0.85 + 0.3 * tint); m.n = bump(p, n0, 0.4, fw); m.rough = 0.7;
    moss = max(moss, smoothstep(0.25, 0.85, n0.y + nA.y * 0.7 - 0.25));
  } else if (id == 7u) {
    // weathered wood: grey, grain along the plank
    let grain = n3(vec3f(p.x * 0.2, p.y * 3.0, p.z * 3.0)).x;
    m.alb = mix(vec3f(0.10, 0.085, 0.065), vec3f(0.2, 0.18, 0.15), grain) * (0.7 + 0.5 * tint);
    m.n = bump(p, n0, 0.25, fw); m.rough = 0.8;
    moss = max(moss, smoothstep(0.55, 0.8, nA.y) * 0.6 * smoothstep(0.3, 0.9, n0.y));
  } else if (id == 8u) {
    // rope, liana
    m.alb = mix(vec3f(0.07, 0.05, 0.03), vec3f(0.16, 0.12, 0.07), tint) * (0.8 + 0.4 * nC.x);
    m.rough = 0.9;
  } else if (id == 9u) {
    // bamboo culm: green to yellow, darker node bands carried by tint
    let g = mix(vec3f(0.07, 0.14, 0.025), vec3f(0.26, 0.24, 0.05), i.s.w);
    m.alb = g * (0.75 + 0.5 * nB.x) * (0.45 + 0.55 * tint);
    m.rough = 0.35; m.spec = 0.06;
    moss = 0.0;
  } else if (id == 10u) {
    // mushroom: tint selects the colour family
    let c = select(select(vec3f(0.85, 0.80, 0.70), vec3f(0.85, 0.45, 0.12), tint > 0.33), vec3f(0.65, 0.42, 0.22), tint > 0.66);
    m.alb = c * (0.8 + 0.3 * nC.x); m.rough = 0.5; m.sss = 0.3; moss = 0.0;
  } else if (id == 11u) {
    m.alb = vec3f(0.16, 0.12, 0.08); m.n = bump(p, n0, 0.6, fw); moss = max(moss, 0.75 + 0.25 * nA.y);
  } else {
    m.alb = vec3f(0.5);
  }
  // moss layer: deep to bright green, fuzzy
  let mc = mix(vec3f(0.03, 0.065, 0.015), vec3f(0.085, 0.16, 0.035), nB.z) * (0.75 + 0.5 * nC.y * fade);
  let mk = smoothstep(0.35, 0.65, moss + (nB.z - 0.5) * 0.5 + (nA.x - 0.5) * 0.4 + (nC.y - 0.5) * 0.15);
  m.alb = mix(m.alb, mc, mk);
  m.n = normalize(mix(m.n, n0, mk * 0.5));
  m.rough = mix(m.rough, 0.95, mk);
  // wetness darkens and adds gloss
  wet = clamp(wet + 0.15, 0.0, 1.0);
  m.alb *= 1.0 - 0.35 * wet;
  m.rough = mix(m.rough, 0.25, wet * (1.0 - mk * 0.7));
  // lighting
  let L = u.sun.xyz; let V = normalize(u.cam.xyz - p);
  let ndl = dot(m.n, L);
  var sh = 0.0; if (dot(n0, L) > -0.2) { sh = shadowAt(p + n0 * 0.08, 1.6); }
  let diff = max(ndl, 0.0) * sh;
  let H = normalize(L + V);
  let sp = pow(max(dot(m.n, H), 0.0), 2.0 / (m.rough * m.rough * m.rough + 0.002)) * (1.0 - m.rough) * 0.6 * sh;
  let f = 0.04 + 0.96 * pow(1.0 - max(dot(m.n, V), 0.0), 5.0);
  let sv = skyVis(p);
  let amb = ambientV(m.n, sv) * i.ao;
  let R = reflect(-V, m.n);
  let env = ambientV(R, sv) * f * (1.0 - m.rough) * (1.0 - m.rough) * 0.8 * i.ao;
  var c = m.alb * (SUNC * diff + amb) + SUNC * sp * (0.04 + f) + env;
  c += m.alb * SUNC * m.sss * sh * max(-ndl, 0.0) * 0.3;
  return enc(fogApply(c, p));
}`;

// ---------------------------------------------------------------- foliage (instanced cards with analytic leaf shapes)
// instance: f32 x,y,z,size | fwd.xyz,droop | faceN.xyz,ao | bend.xyz,swayW | rgb565, variant:4, phase:6, aspect:6
const W_LEAF = W_COMMON + /*wgsl*/`
override KIND: u32 = 0;
override ROWS: u32 = 4;
override SHADOW: bool = false;
override VBASE: u32 = 0;
override VREV: bool = false;
override DFULL: f32 = 40.0;
override DMIN: f32 = 0.3;
override DMAX: f32 = 300.0;
@group(1) @binding(0) var<storage, read> inst: array<vec4u>;
@group(1) @binding(1) var<storage, read> vis: array<u32>;
struct VO {
  @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) wp: vec3f, @location(2) n: vec3f,
  @location(3) @interpolate(flat) col: vec4f, @location(4) @interpolate(flat) bend: vec4f, @location(5) @interpolate(flat) kv: vec4f,
}
fn aspect() -> f32 {
  switch (KIND) { case 0u: { return 0.62; } case 1u: { return 0.5; } case 2u: { return 0.34; } case 3u: { return 0.42; } default: { return 1.0; } }
}
// outline: half width (in leaf lengths) at position v along the leaf
fn outline(v: f32, var_: u32) -> f32 {
  let a = aspect();
  switch (KIND) {
    case 0u: { return a * 0.5 * (0.55 + 0.45 * sin(PI * clamp(v * 0.9 + 0.1, 0.0, 1.0))); }
    case 1u: {
      if (var_ == 1u) { return a * 0.62 * pow(sin(PI * clamp(v * 0.85 + 0.15, 0.0, 1.0)), 0.6); }
      if (var_ == 2u) { return a * 0.55 * pow(sin(PI * clamp(v, 0.0, 1.0)), 0.45); }
      if (var_ == 3u) { return a * 0.26 * pow(sin(PI * clamp(v, 0.0, 1.0)), 0.5); }
      return a * 0.5 * pow(sin(PI * clamp(v, 0.0, 1.0)), 0.75);
    }
    case 2u: { return a * 0.5 * (1.0 - 0.75 * v) * smoothstep(0.0, 0.12, v) + 0.01; }
    case 3u: { return a * 0.5 * pow(sin(PI * clamp(v, 0.0, 1.0)), 0.7); }
    default: { return 0.5; }
  }
}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO;
  var id = ii; var keep = 1.0;
  if (!SHADOW) { id = vis[select(VBASE + ii, VBASE - ii, VREV)]; }
  let w0 = inst[id * 2u]; let w1 = inst[id * 2u + 1u];
  let base = bitcast<vec3f>(w0.xyz); var size = bitcast<f32>(w0.w);
  let fa = unpack4x8snorm(w1.x); let fb = unpack4x8snorm(w1.y); let fc = unpack4x8snorm(w1.z);
  let var_ = (w1.w >> 16u) & 15u; let ph = f32((w1.w >> 20u) & 63u) / 63.0; let asp = 0.6 + f32((w1.w >> 26u) & 63u) / 63.0 * 0.8;
  if (!SHADOW) {
    // density LOD: far clusters draw a random subset, the survivors grow to keep the coverage
    let d = distance(base, u.cam.xyz) * u.prm.x;
    let f = clamp(pow(DFULL / max(d, 1.0), 1.6), DMIN, 1.0);
    let k = f * smoothstep(DMAX, DMAX * 0.7, d) * 64.0; let li = f32(id & 63u);
    keep = clamp((k - li) / 3.0, 0.0, 1.0);
    size *= keep * min(inverseSqrt(f), 1.8);
    if (keep <= 0.0) { o.pos = vec4f(0.0, 0.0, -1.0, 1.0); return o; }
  }
  var fwd = normalize(fa.xyz); let droop = fa.w;
  var nf = normalize(fb.xyz - fwd * dot(fb.xyz, fwd));
  let side = cross(nf, fwd);
  // corner of the strip: ROWS rows, two halves (left/right of the midrib)
  // indexed grid: (ROWS+1) rows of 3 vertices (left edge, midrib, right edge)
  let v = f32(vi / 3u) / f32(ROWS);
  let us = f32(vi % 3u) - 1.0; // -1..1 across
  let hs = 0.5 / f32(ROWS);
  let gw = max(max(outline(v, var_), outline(clamp(v - hs, 0.0, 1.0), var_)), outline(clamp(v + hs, 0.0, 1.0), var_)) * asp + 0.02;
  let x = us * gw;
  // wind: the whole card sways with its plant, the blade flutters around its midrib
  let t = u.cam.w;
  let sw = windAt(base, ph) * (fc.w * 0.5 + 0.5) * 1.2;
  let fl = sin(t * (5.0 + ph * 4.0) + ph * 40.0 + base.y) * 0.18 + sin(t * 2.3 + ph * 13.0) * 0.1;
  var lp = base + sw;
  let down = vec3f(0.0, -1.0, 0.0);
  let along = v * size;
  lp += fwd * along * (1.0 - abs(droop) * v * 0.35) + down * droop * size * v * v * 0.6;
  let fold = select(0.18, 0.08, KIND == 0u);
  lp += side * (x * size) + nf * (abs(x) * size * fold + x * size * fl * v * 0.6);
  if (KIND == 4u) { lp = base + sw + fwd * (us * 0.5 * size * asp) + side * ((v - 0.5) * size); }
  let nn = normalize(nf - side * sign(us) * fold * 1.5 + fwd * droop * v * 0.6);
  o.pos = u.vp * vec4f(lp, 1.0);
  o.wp = lp; o.n = nn;
  o.uv = vec2f(x / max(asp, 0.01), v);
  if (KIND == 4u) { o.uv = vec2f(us, v * 2.0 - 1.0); }
  let rgb = vec3f(f32((w1.w >> 11u) & 31u) / 31.0, f32((w1.w >> 5u) & 63u) / 63.0, f32(w1.w & 31u) / 31.0);
  o.col = vec4f(rgb, fb.w * 0.5 + 0.5);
  o.bend = vec4f(fc.xyz, ph);
  o.kv = vec4f(f32(var_), asp, size, keep);
  return o;
}
// signed coverage of one elliptic leaflet with a drip tip: base b, direction d, length l, half width w
fn leaflet(p: vec2f, b: vec2f, d: vec2f, l: f32, w: f32) -> f32 {
  let q = p - b; let a = dot(q, d) / l; let c = abs(q.x * d.y - q.y * d.x);
  if (a < 0.0 || a > 1.0) { return -0.05; }
  let prof = pow(sin(PI * a), 0.7) * (1.0 - 0.25 * a) * smoothstep(1.0, 0.82, a) + smoothstep(0.8, 1.0, a) * 0.03 * (1.0 - a) * 10.0 * 0.0;
  return w * prof - c;
}
struct Shape { d: f32, vein: f32, tone: f32 }
fn shape(uv: vec2f, var_: u32, asp: f32, ph: f32) -> Shape {
  var s: Shape; s.vein = 0.0; s.tone = 0.0;
  let X = uv.x * asp; let Y = uv.y; // metric coordinates in leaf lengths
  switch (KIND) {
    case 0u: {
      // twig spray: alternate leaflets along a stem plus a terminal leaf
      // far cards (VREV) use half as many, wider leaflets: same coverage, half the cost
      var d = -1.0; let n = select(select(6, 9, var_ == 1u), select(3, 4, var_ == 1u), VREV);
      let L = select(0.42, 0.34, var_ == 1u); let Wd = select(0.11, 0.045, var_ == 1u) * select(1.0, 1.7, VREV);
      for (var k = 0; k < n; k++) {
        let fk = f32(k) * select(1.0, 2.0, VREV); let sd = select(-1.0, 1.0, (k & 1) == 0);
        let vy = 0.08 + fk * select(0.1, 0.075, var_ == 1u);
        let ang = select(0.85, 0.65, var_ == 1u) - fk * 0.04 + sin(ph * 30.0 + fk) * 0.15;
        let dd = vec2f(sd * sin(ang), cos(ang));
        d = max(d, leaflet(vec2f(X, Y), vec2f(0.0, vy), dd, L * (1.0 - fk * 0.03), Wd));
      }
      d = max(d, leaflet(vec2f(X, Y), vec2f(0.0, select(0.6, 0.64, var_ == 1u)), vec2f(0.0, 1.0), select(0.4, 0.36, var_ == 1u), Wd));
      s.d = d;
      s.vein = smoothstep(0.012, 0.0, abs(X)) * step(Y, 0.7);
    }
    case 1u: {
      let w = outline(Y, var_) / 0.5 * 0.5 * asp;
      var d = w - abs(X);
      if (var_ == 3u) { d += sin(Y * 60.0) * 0.004; }
      if (var_ == 0u) { // occasional insect bite holes
        let hc = vec2f(X, Y) * 18.0; let hid = floor(hc);
        let r = hf(u32(hid.x + 50.0) * 131u + u32(hid.y + 50.0) * 977u + u32(ph * 1000.0));
        if (r > 0.93) { d = min(d, length(fract(hc) - 0.5) / 18.0 - 0.016); }
      }
      s.d = d;
      s.vein = smoothstep(0.01, 0.0, abs(X)) + smoothstep(0.08, 0.0, abs(fract((Y - abs(X) * 0.9) * 9.0) - 0.5)) * 0.35;
      s.tone = abs(X) / max(w, 1e-3);
    }
    case 2u: {
      // pinnate frond: lanceolate pinnae angled toward the tip; fern pinnae are lobed, fishtail-palm leaflets are wedges
      let N = select(select(26.0, 11.0, var_ == 1u), 18.0, var_ == 2u);
      let ax = abs(X);
      let ang = select(select(0.35, 0.25, var_ == 1u), 0.75, var_ == 2u);
      let yy = (Y - ax * ang) * N; let k = floor(yy); let fy = abs(fract(yy) - 0.5);
      let yk = (k + 0.5) / N;
      let len = 0.2 * pow(max(1.0 - yk, 0.0), 0.7) * smoothstep(-0.05, 0.12, yk) * select(1.0, 0.85, var_ == 1u) + 0.002;
      let s0 = clamp(ax / len, 0.0, 1.0);
      var w = 0.44 * pow(1.0 - s0, 0.55) * smoothstep(0.0, 0.08, s0 + 0.05);
      if (var_ == 0u) { w *= 0.72 + 0.28 * abs(sin(s0 * 22.0)); }
      if (var_ == 1u) { w = (0.12 + 0.36 * s0) * step(s0, 0.97 - 0.15 * abs(sin(yy * 3.1 + ph * 20.0))); }
      if (var_ == 2u) { w = 0.2 * pow(1.0 - s0, 0.4); }
      s.d = min((w - fy) / N, (len - ax) * 0.5);
      s.d = max(s.d, 0.0045 - ax);
      s.vein = smoothstep(0.006, 0.0, ax) + smoothstep(0.08, 0.0, fy) * 0.25;
      s.tone = s0;
    }
    case 3u: {
      let w = outline(Y, var_) * asp;
      s.d = w - abs(X);
      s.vein = smoothstep(0.01, 0.0, abs(X));
      s.tone = abs(X) / max(w, 1e-3);
    }
    default: {
      // flower: radial petals around a bright centre
      let r = length(uv); let a = atan2(uv.y, uv.x);
      let np = select(5.0, 6.0, var_ == 1u);
      let pet = 0.62 + 0.38 * pow(abs(cos(a * np * 0.5 + ph * 6.0)), select(1.5, 0.6, var_ == 1u));
      s.d = (pet * 0.95 - r) * 0.5;
      s.tone = r;
      s.vein = smoothstep(0.25, 0.1, r);
    }
  }
  return s;
}
@fragment fn fsShadow(i: VO) {
  let s = shape(i.uv, u32(i.kv.x), i.kv.y, i.bend.w);
  if (s.d < 0.0) { discard; }
}
@fragment fn fs(i: VO, @builtin(front_facing) ff: bool) -> @location(0) vec4f {
  let var_ = u32(i.kv.x);
  let s = shape(i.uv, var_, i.kv.y, i.bend.w);
  let px = max(fwidth(s.d), 1e-5);
  let cov = clamp(s.d / px + 0.5, 0.0, 1.0);
  if (cov <= 0.0) { discard; }
  var alb = srgb(i.col.rgb);
  let lv = hf(u32(i.bend.w * 4093.0) + u32(i.kv.z * 977.0));
  let mot = n3(i.wp * 0.9).y;
  alb *= (0.8 + 0.4 * lv) * (0.85 + 0.3 * mot) * (1.0 + s.vein * 0.3 - s.tone * 0.15) * (0.82 + 0.18 * smoothstep(0.0, 0.6, i.uv.y));
  if (KIND == 4u) {
    alb = mix(alb, vec3f(1.0, 0.85, 0.35), s.vein * 0.85);
  }
  // underside of a blade: paler, less saturated
  if (!ff && KIND != 4u) { alb = mix(alb, vec3f(lum(alb)), 0.25) * 1.25; }
  let p = i.wp;
  var n = normalize(i.n); if (!ff) { n = -n; }
  let bend = i.bend.xyz;
  let bn = normalize(n * 0.6 + bend * 0.8 + vec3f(0.0, 0.1, 0.0));
  let L = u.sun.xyz; let V = normalize(u.cam.xyz - p);
  let sh = shadowAt(p + L * 0.35 + bn * 0.08, 1.6);
  let ndl = dot(bn, L);
  let diff = clamp(ndl * 0.85 + 0.15, 0.0, 1.0) * sh;
  // light through the blade: strongest looking toward the sun from the shaded side
  let back = clamp(-dot(n, L), 0.0, 1.0);
  let trans = (pow(max(dot(-V, L), 0.0), 4.0) * 1.2 + 0.25) * back;
  let tc = alb * vec3f(1.15, 1.3, 0.45);
  let ao = i.col.a;
  let sv = skyVis(p);
  let amb = ambientV(bn, sv) * (0.35 + 0.65 * ao) + ambientV(-bn, sv) * tc * 0.6 * ao;
  let H = normalize(L + V);
  let gloss = select(0.15, 0.45, ff);
  let sp = pow(max(dot(n, H), 0.0), 60.0) * gloss * sh * (1.0 - s.tone * 0.4);
  let f = 0.03 + 0.4 * pow(1.0 - abs(dot(n, V)), 5.0);
  var c = alb * (SUNC * diff + amb) + tc * SUNC * sh * trans * 0.35 + SUNC * sp * 0.35 + ambientV(reflect(-V, n), sv) * f * 0.3 * ao;
  c = fogApply(c, p);
  return vec4f(enc(c).rgb, cov);
}`;

// ---------------------------------------------------------------- per-frame culling of foliage clusters (compute)
const W_CULL = /*wgsl*/`
struct CU { planes: array<vec4f, 6>, cam: vec4f, total: u32, ncat: u32, pad0: u32, pad1: u32, cat: array<vec4f, 8>, rng: array<vec4u, 8> }
@group(0) @binding(0) var<uniform> c: CU;
@group(0) @binding(1) var<storage, read> cl: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vis: array<u32>;
@group(0) @binding(3) var<storage, read_write> args: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= c.total) { return; }
  var k = 0u; for (var j = 1u; j < c.ncat; j++) { if (i >= c.rng[j].x) { k = j; } }
  let s = cl[i]; let prm = c.cat[k];
  let d = distance(s.xyz, c.cam.xyz);
  if ((d - s.w) * c.cam.w > prm.x) { return; }
  for (var p = 0u; p < 6u; p++) { if (dot(c.planes[p].xyz, s.xyz) + c.planes[p].w < -s.w) { return; } }
  let lod = select(1u, 0u, d - s.w < prm.y);
  // density LOD (same rule as the leaf vertex shader, at the cluster's nearest point): only the survivors are drawn
  let dn = max(d - s.w, 1.0) * c.cam.w;
  let f = clamp(pow(prm.z / dn, 1.6), prm.w, 1.0);
  let n = min(64u, u32(ceil(f * smoothstep(prm.x, prm.x * 0.7, dn) * 64.0)));
  if (n == 0u) { return; }
  let idx = atomicAdd(&args[(k * 2u + lod) * 5u + 1u], n);
  let cid = (i - c.rng[k].x) * 64u;
  // near list fills its range from the front, far list from the back
  for (var j = 0u; j < n; j++) { vis[select(c.rng[k].y + idx + j, c.rng[k].y + c.rng[k].z - 1u - idx - j, lod == 1u)] = cid + j; }
}`;

// ---------------------------------------------------------------- sky (fills pixels no geometry covered)
const W_SKY = W_COMMON + /*wgsl*/`
struct VO { @builtin(position) pos: vec4f, @location(0) q: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var o: VO; let q = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  o.pos = vec4f(q, 0.0, 1.0); o.q = q; return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let rd = normalize(u.fwd.xyz + u.right.xyz * i.q.x * u.right.w + u.up.xyz * i.q.y * u.up.w);
  var c = sky(rd);
  // the boundary haze: far below the horizon is always fog
  let T = exp(-u.misc.x * 60.0 * exp(-(u.cam.y - u.misc.y) / 22.0) / max(rd.y * 1.6 + 0.08, 0.02));
  c = c * T + fogCol(rd, 400.0) * (1.0 - T);
  return enc(c);
}`;

// ---------------------------------------------------------------- water (streams, pools, falls) in a second pass
const W_WATER = W_COMMON + /*wgsl*/`
@group(1) @binding(0) var tCol: texture_2d<f32>;
@group(1) @binding(1) var tDep: texture_depth_multisampled_2d;
struct VI { @location(0) p: vec3f, @location(1) f: vec4f }
struct VO { @builtin(position) pos: vec4f, @location(0) wp: vec3f, @location(1) f: vec4f }
@vertex fn vs(i: VI) -> VO { var o: VO; o.pos = u.vp * vec4f(i.p, 1.0); o.wp = i.p; o.f = i.f; return o; }
fn linZ(d: f32) -> f32 { return u.fwd.w / max(d, 1e-7); }
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let p = i.wp; let t = u.cam.w;
  let fall = i.f.z; let flow = select(i.f.xy, vec2f(i.f.x, 0.0), fall > 0.5); let foam0 = i.f.w;
  let V = normalize(u.cam.xyz - p); let dist = length(u.cam.xyz - p);
  let pix = vec2i(i.pos.xy);
  let sceneD = linZ(textureLoad(tDep, pix, 0));
  let myD = linZ(i.pos.z);
  let depth = max(sceneD - myD, 0.0) * abs(V.y) + fall * 0.5;
  // ripples advected along the flow
  let q = vec3f(p.x, 0.0, p.z);
  let o1 = vec3f(flow.x, 0.0, flow.y) * t;
  let r1 = n3(q * 0.09 - o1 * 0.09 + vec3f(0.0, t * 0.01, 0.0)).xyz - 0.5;
  let r2 = n3(q * 0.31 - o1 * 0.31 * 1.3 + vec3f(0.3, t * 0.03, 0.0)).xyz - 0.5;
  let fw = clamp(1.0 - fwidth(p.x) * 4.0, 0.0, 1.0);
  var n = normalize(vec3f(r1.x * 0.8 + r2.x * 0.5 * fw, 1.0, r1.z * 0.8 + r2.z * 0.5 * fw));
  // rings spreading from where the falls hit the water
  let fo = p.xz - u.misc2.xz; let fd = length(fo);
  n = normalize(n + vec3f(fo.x, 0.0, fo.y) / max(fd, 0.1) * sin(fd * 5.0 - t * 7.0) * exp(-fd / 5.0) * 0.6 * fw);
  n = normalize(mix(n, vec3f(0.0, 1.0, 0.0), 0.2));
  // refraction
  let scr = i.pos.xy * u.scr.zw;
  var ro = scr + n.xz * 0.04 * clamp(depth, 0.0, 1.0) / max(myD * 0.08, 1.0);
  let rd = linZ(textureLoad(tDep, vec2i(ro * u.scr.xy), 0));
  if (rd < myD) { ro = scr; }
  let below = dec(textureSampleLevel(tCol, sClamp, ro, 0.0).rgb);
  let absorb = exp(-depth * vec3f(2.8, 1.4, 1.6));
  let deepc = vec3f(0.02, 0.05, 0.04) * (ambient(vec3f(0.0, 1.0, 0.0), p) * 1.5);
  var c = below * absorb + deepc * (1.0 - absorb);
  // screen-space reflection, with the canopy / sky as the fallback
  let R = reflect(-V, n);
  var refl = mix(vec3f(0.025, 0.045, 0.03) * u.prm.z * 3.0, sky(R) * 0.22, skyVis(p) * smoothstep(0.1, 0.6, R.y));
  var hit = 0.0;
  var rp = p; let stp = 0.6 + myD * 0.03;
  for (var k = 0; k < 20; k++) {
    rp += R * stp * (1.0 + f32(k) * 0.25);
    let cp = u.vp * vec4f(rp, 1.0);
    if (cp.w <= 0.0) { break; }
    let su = cp.xy / cp.w * vec2f(0.5, -0.5) + 0.5;
    if (any(su < vec2f(0.0)) || any(su > vec2f(1.0))) { break; }
    let sz = linZ(textureLoad(tDep, vec2i(su * u.scr.xy), 0));
    if (sz < cp.w && cp.w - sz < 3.0 + f32(k)) {
      let e = smoothstep(0.0, 0.12, min(min(su.x, 1.0 - su.x), min(su.y, 1.0 - su.y)));
      refl = mix(refl, dec(textureSampleLevel(tCol, sClamp, su, 0.0).rgb), e); hit = 1.0; break;
    }
  }
  let F = min(0.02 + 0.98 * pow(1.0 - max(dot(n, V), 0.0), 5.0), 0.55);
  c = mix(c, refl * vec3f(0.5, 0.62, 0.6), F);
  // a pale line where the water meets the bank
  c = mix(c, c * 0.7, smoothstep(0.08, 0.0, depth) * 0.5);
  // sun glints
  let L = u.sun.xyz; let sh = shadowAt(p + vec3f(0.0, 0.05, 0.0), 1.0);
  let H = normalize(L + V);
  c += SUNC * sh * (pow(max(dot(n, H), 0.0), 900.0) * 2.5 + pow(max(dot(n, H), 0.0), 90.0) * 0.25);
  // foam on falls and in shallow fast water, white water streaks on the falls
  let streak = n3(vec3f(p.x * 0.09, p.y * 0.008 + t * 0.06, p.z * 0.09)).x;
  let fm = clamp(foam0 * smoothstep(0.35, 0.7, n3(q * 0.05 - o1 * 0.05).y + n3(q * 0.2 - o1 * 0.2).z * 0.4) + smoothstep(0.1, 0.0, depth) * 0.15, 0.0, 1.0);
  let foamc = vec3f(0.8, 0.86, 0.85) * (ambient(vec3f(0.0, 1.0, 0.0), p) * 1.1 + SUNC * sh * 0.12);
  c = mix(c, foamc, fm * 0.7);
  let st2 = n3(vec3f(p.x * 0.25, p.y * 0.02 + t * 0.15, p.z * 0.25)).y;
  c = mix(c, foamc * (0.55 + 0.5 * streak + 0.35 * st2), fall * (0.75 + 0.25 * smoothstep(0.3, 0.7, streak)));
  var a = 1.0;
  if (fall > 0.5) { a = clamp((1.0 - abs(flow.x) - (streak - 0.5) * 1.4 - (st2 - 0.5) * 0.6) * 2.5, 0.0, 1.0); }
  c = fogApply(c, p);
  return vec4f(enc(c).rgb, a);
}`;

// ---------------------------------------------------------------- particles and wildlife
const W_FX = W_COMMON + /*wgsl*/`
@group(1) @binding(0) var<storage, read> fx: array<vec4f>;
struct VO { @builtin(position) pos: vec4f, @location(0) q: vec2f, @location(1) c: vec4f, @location(2) wp: vec3f }
fn quadCorner(i: u32) -> vec2f { return vec2f(f32(i & 1u), f32((i >> 1u) & 1u)) * 2.0 - 1.0; }
fn billboard(p: vec3f, q: vec2f, r: f32) -> vec4f {
  // never smaller than ~1.3 px, so motes do not twinkle
  let w = dot(p - u.cam.xyz, u.fwd.xyz);
  let rp = max(r, w * u.up.w * 2.6 * u.scr.w);
  return u.vp * vec4f(p + (u.right.xyz * q.x + u.up.xyz * q.y) * rp, 1.0);
}
// pollen / dust: a box of motes that wraps around the camera; only those in a sunbeam really show
@vertex fn vsMote(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO; let q = quadCorner(vi); let t = u.cam.w;
  let B = 36.0;
  let h = vec3f(hf(ii * 3u), hf(ii * 3u + 1u), hf(ii * 3u + 2u));
  let drift = vec3f(sin(t * 0.13 + h.x * 30.0) * 0.6 + t * 0.12, sin(t * 0.21 + h.y * 20.0) * 0.4 + t * 0.03, cos(t * 0.17 + h.z * 25.0) * 0.6 + t * 0.08);
  let p = u.cam.xyz + (fract(h + (drift - u.cam.xyz) / B) - 0.5) * B;
  let r = 0.006 + 0.01 * hf(ii * 7u);
  o.pos = billboard(p, q, r);
  let d = length(p - u.cam.xyz);
  let sh = shadowAt(p, 1.0);
  let mu = dot(normalize(p - u.cam.xyz), u.sun.xyz);
  let bright = sh * (0.25 + 2.5 * pow(max(mu, 0.0), 6.0)) * 2.2 + 0.015;
  let fade = smoothstep(B * 0.5, B * 0.3, d) * smoothstep(0.3, 1.2, d);
  let w = dot(p - u.cam.xyz, u.fwd.xyz);
  let px = r / (w * u.up.w * 2.0 * u.scr.w);
  o.c = vec4f(vec3f(1.0, 0.9, 0.65) * bright * fade * min(px * px, 1.0) * 0.45, 1.0);
  o.q = q; o.wp = p;
  return o;
}
@fragment fn fsGlow(i: VO) -> @location(0) vec4f {
  let a = max(1.0 - dot(i.q, i.q), 0.0);
  let c = min(i.c.rgb * a * a * fogT(i.wp), vec3f(3.0));
  return vec4f(c / (1.0 + max(c.r, max(c.g, c.b))) * 0.25, 0.0);
}
// fireflies in the shade: wander around their home point, blink
@vertex fn vsFly(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO; let q = quadCorner(vi); let t = u.cam.w;
  let home = fx[2048u + ii];
  let ph = home.w;
  let p = home.xyz + vec3f(sin(t * 0.31 + ph * 7.0) * 1.6 + sin(t * 0.9 + ph * 3.0) * 0.4, sin(t * 0.47 + ph * 5.0) * 0.5, cos(t * 0.27 + ph * 11.0) * 1.6);
  let blink = pow(max(sin(t * (0.8 + fract(ph * 3.7) * 0.9) + ph * 40.0), 0.0), 6.0);
  let sh = shadowAt(p, 1.0);
  o.pos = billboard(p, q, 0.06);
  o.c = vec4f(vec3f(0.75, 1.0, 0.3) * blink * 12.0 * (1.0 - sh * 0.9) * smoothstep(80.0, 30.0, distance(p, u.cam.xyz)), 1.0);
  o.q = q; o.wp = p;
  return o;
}
// waterfall spray: particles shot out at the foot of each fall
@vertex fn vsSpray(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO; let q = quadCorner(vi); let t = u.cam.w;
  let src = fx[ii % 8u + 64u];
  let ph = hf(ii * 13u);
  let life = fract(t * (0.25 + 0.2 * hf(ii * 5u)) + ph);
  let a = hf(ii * 7u) * 6.283;
  let sp = 1.0 + 2.5 * hf(ii * 11u);
  let p = src.xyz + vec3f(cos(a) * sp * life * src.w, life * (3.5 - 4.0 * life) * src.w, sin(a) * sp * life * src.w);
  let r = (0.3 + life * 1.1) * src.w;
  o.pos = u.vp * vec4f(p + (u.right.xyz * q.x + u.up.xyz * q.y) * r, 1.0);
  let sh = shadowAt(p, 1.0);
  o.c = vec4f((ambient(vec3f(0.0, 1.0, 0.0), p) * 0.6 + SUNC * sh * 0.12) * (1.0 - life) * 0.18 * step(0.01, src.w), 1.0);
  o.q = q; o.wp = p;
  return o;
}
@fragment fn fsSoft(i: VO) -> @location(0) vec4f {
  let a = max(1.0 - dot(i.q, i.q), 0.0);
  let c = i.c.rgb * a * fogT(i.wp);
  return vec4f(c / (1.0 + max(c.r, max(c.g, c.b))), 0.0);
}
// falling leaves: tumble down from the canopy around the camera
struct LO { @builtin(position) pos: vec4f, @location(0) q: vec2f, @location(1) c: vec3f, @location(2) wp: vec3f, @location(3) n: vec3f }
@vertex fn vsFall(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> LO {
  var o: LO; let q = quadCorner(vi); let t = u.cam.w;
  let B = 70.0;
  let h = vec3f(hf(ii * 3u + 100u), hf(ii * 3u + 101u), hf(ii * 3u + 102u));
  let H = 32.0;
  let fallT = fract(t * (0.022 + 0.012 * h.y) + h.y);
  var xz = u.cam.xz + (fract(h.xz + (vec2f(t * 0.03, t * 0.02) - u.cam.xz) / B) - 0.5) * B;
  let gy = mapAt(xz).r;
  let y = gy + H * (1.0 - fallT) + 0.05;
  xz += vec2f(sin(t * 1.1 + h.x * 40.0), cos(t * 0.9 + h.z * 40.0)) * 0.6;
  let p = vec3f(xz.x, y, xz.y);
  let a = t * (1.5 + h.x * 2.0) + h.z * 30.0; let b = t * (2.1 + h.y) + h.x * 20.0;
  let ax = vec3f(cos(a), sin(a) * 0.5, sin(a));
  let ay = normalize(cross(ax, vec3f(sin(b), cos(b), 0.3)));
  let s = 0.035 + 0.035 * h.z;
  let wp = p + (ax * q.x * 0.45 + ay * q.y) * s;
  o.pos = u.vp * vec4f(wp, 1.0);
  o.q = q; o.wp = wp; o.n = cross(ax, ay);
  let k = hf(ii * 17u);
  o.c = srgb(select(select(vec3f(0.55, 0.35, 0.1), vec3f(0.75, 0.6, 0.15), k > 0.5), vec3f(0.35, 0.5, 0.12), k > 0.8)) * smoothstep(80.0, 50.0, distance(p, u.cam.xyz)) ;
  return o;
}
@fragment fn fsFall(i: LO, @builtin(front_facing) ff: bool) -> @location(0) vec4f {
  let d = (0.95 - abs(i.q.x) / max(pow(1.0 - i.q.y * i.q.y, 0.7), 1e-3)) * 0.5;
  let cov = clamp(d / max(fwidth(d), 1e-4) + 0.5, 0.0, 1.0);
  if (cov <= 0.0 || dot(i.c, i.c) < 1e-6) { discard; }
  var n = normalize(i.n); if (!ff) { n = -n; }
  let sh = shadowAt(i.wp, 1.0);
  let V = normalize(u.cam.xyz - i.wp);
  let c = i.c * (SUNC * sh * (abs(dot(n, u.sun.xyz)) * 0.6 + pow(max(dot(-V, u.sun.xyz), 0.0), 3.0) * 0.6) + ambient(n, i.wp));
  return vec4f(enc(fogApply(c, i.wp)).rgb, cov);
}
// birds, hornbills, butterflies, dragonflies: small procedural meshes animated here
// fx layout: [0..63] spray sources etc. Birds read their territory from fx[128 + ...]
struct BI { @location(0) p: vec3f, @location(1) n: vec4f, @location(2) m: vec4f, @location(3) s: vec4f }
struct BO { @builtin(position) pos: vec4f, @location(0) wp: vec3f, @location(1) n: vec3f, @location(2) @interpolate(flat) c: vec3f, @location(3) @interpolate(flat) e: f32 }
override CREATURE: u32 = 0;
override NPER: u32 = 12;
override FXBASE: u32 = 128;
fn perch(i: u32) -> vec3f { return fx[FXBASE + i].xyz; }
fn spline(a: vec3f, b: vec3f, c: vec3f, d: vec3f, t: f32) -> vec3f {
  return 0.5 * (2.0 * b + (c - a) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t * t + (3.0 * b - a - 3.0 * c + d) * t * t * t);
}
@vertex fn vsBird(i: BI, @builtin(instance_index) ii: u32) -> BO {
  var o: BO; let t = u.cam.w;
  var pos: vec3f; var dir: vec3f; var flap = 0.0; var scale = 1.0; var col = vec3f(0.5);
  let hb = hf(ii * 31u + CREATURE * 1000u);
  if (CREATURE == 0u) {
    // small birds hop between perches in their own territory; the sequence comes from a hash, so it never repeats
    let T = 4.0 + 3.0 * hb; let tt = t + hb * 50.0; let seg = u32(floor(tt / T)); let f = fract(tt / T);
    let base = (ii * 7u) % 40u;
    let pa = perch((base + ih(seg * 9u + ii) % NPER) % 512u);
    let pb = perch((base + ih((seg + 1u) * 9u + ii) % NPER) % 512u);
    let fly = smoothstep(0.35, 0.4, f);
    let ft = clamp((f - 0.4) / 0.6, 0.0, 1.0);
    let e = ft * ft * (3.0 - 2.0 * ft);
    pos = mix(pa, pb, e) + vec3f(0.0, sin(ft * PI) * (1.5 + distance(pa, pb) * 0.12), 0.0);
    dir = normalize(pb - pa + vec3f(0.0, cos(ft * PI) * 3.0, 0.0) + vec3f(1e-3, 0.0, 0.0));
    flap = select(0.15 * sin(t * 3.0 + hb * 10.0), sin(t * 38.0 + hb * 20.0) * (0.6 + 0.4 * step(0.5, fract(ft * 3.0 + hb))), ft > 0.0 && ft < 1.0);
    let k = ih(ii) % 5u;
    var pal = array<vec3f, 5>(vec3f(0.9, 0.12, 0.05), vec3f(0.1, 0.45, 0.85), vec3f(0.95, 0.75, 0.05), vec3f(0.15, 0.6, 0.2), vec3f(0.95, 0.4, 0.05)); col = pal[k];
    scale = 1.4 + 0.4 * hb;
  } else if (CREATURE == 1u) {
    // hornbills: slow, wide, never-repeating loops over the ravine and the big openings, flap then glide
    let c = fx[FXBASE + ii];
    let w = 0.045 + 0.02 * hb;
    pos = c.xyz + vec3f(sin(t * w + hb * 9.0) * c.w + sin(t * w * 2.7 + hb) * c.w * 0.3, sin(t * w * 1.9 + hb * 4.0) * 4.0, cos(t * w * 1.13 + hb * 9.0) * c.w * 0.8);
    let p2 = c.xyz + vec3f(sin((t + 0.1) * w + hb * 9.0) * c.w + sin((t + 0.1) * w * 2.7 + hb) * c.w * 0.3, sin((t + 0.1) * w * 1.9 + hb * 4.0) * 4.0, cos((t + 0.1) * w * 1.13 + hb * 9.0) * c.w * 0.8);
    dir = normalize(p2 - pos);
    let cyc = fract(t * 0.12 + hb);
    flap = select(0.08, sin(t * 9.0 + hb * 7.0) * 0.9, cyc < 0.45);
    col = vec3f(0.04, 0.04, 0.04);
    scale = 2.6;
  } else if (CREATURE == 2u) {
    // butterflies: erratic flutter around flowers in sun
    let c = fx[FXBASE + ii % 64u];
    let s = t * (0.5 + hb * 0.3) + hb * 30.0;
    pos = c.xyz + vec3f(sin(s * 1.3) * 2.2 + sin(s * 3.7) * 0.5, 0.8 + sin(s * 2.1) * 0.6 + sin(s * 7.3) * 0.15, cos(s * 1.1) * 2.2 + cos(s * 4.1) * 0.5);
    let s2 = s + 0.05;
    let p2 = c.xyz + vec3f(sin(s2 * 1.3) * 2.2 + sin(s2 * 3.7) * 0.5, 0.8 + sin(s2 * 2.1) * 0.6, cos(s2 * 1.1) * 2.2 + cos(s2 * 4.1) * 0.5);
    dir = normalize(p2 - pos + vec3f(1e-3, 0.0, 0.0));
    flap = sin(t * 22.0 + hb * 40.0);
    let k = ih(ii * 3u) % 5u;
    var pal = array<vec3f, 5>(vec3f(0.05, 0.35, 1.0), vec3f(1.0, 0.45, 0.02), vec3f(1.0, 0.9, 0.15), vec3f(0.95, 0.95, 0.9), vec3f(0.1, 0.8, 0.6)); col = pal[k];
    scale = 0.8 + 0.4 * hb;
  } else {
    // dragonflies: hover then dart over the water
    let c = fx[FXBASE + ii % 64u];
    let T = 2.5; let tt = t + hb * 20.0; let seg = floor(tt / T); let f = fract(tt / T);
    let a = vec3f(hf(u32(seg) * 3u + ii * 101u), hf(u32(seg) * 3u + ii * 101u + 1u), hf(u32(seg) * 3u + ii * 101u + 2u)) - 0.5;
    let b = vec3f(hf(u32(seg + 1.0) * 3u + ii * 101u), hf(u32(seg + 1.0) * 3u + ii * 101u + 1u), hf(u32(seg + 1.0) * 3u + ii * 101u + 2u)) - 0.5;
    let e = smoothstep(0.6, 0.85, f);
    pos = c.xyz + mix(a, b, e) * vec3f(8.0, 1.0, 8.0) + vec3f(0.0, 0.6 + sin(t * 5.0 + hb) * 0.03, 0.0);
    dir = normalize(vec3f(b.x - a.x, 0.0, b.z - a.z) + vec3f(1e-3, 0.0, 0.0));
    flap = sin(t * 60.0 + hb * 9.0);
    col = select(vec3f(0.85, 0.12, 0.05), vec3f(0.1, 0.6, 0.75), hb > 0.5);
    scale = 1.0;
  }
  // never let a creature fill the screen: shrink away within a few metres of the camera
  scale *= smoothstep(1.5, 5.0, distance(pos, u.cam.xyz));
  // mesh: x right, y up, z forward; s.x = |wing span coordinate|, m.y = part (0 body, 1 wing, 2 accent)
  var lp = i.p * scale;
  let span = i.s.x;
  if (span > 0.0) {
    let ang = flap * 0.9 * select(1.0, 1.3, CREATURE == 2u);
    let sx = sign(lp.x);
    let r = abs(lp.x);
    lp = vec3f(sx * r * cos(ang), lp.y + r * sin(ang) * (1.0 + span * 0.4), lp.z);
  }
  let fw = dir; let rt = normalize(cross(fw, vec3f(0.0, 1.0, 0.0)) + vec3f(1e-4, 0.0, 0.0)); let up = cross(rt, fw);
  let wp = pos + rt * lp.x + up * lp.y + fw * lp.z;
  o.pos = u.vp * vec4f(wp, 1.0);
  o.wp = wp; o.n = normalize(rt * i.n.x + up * i.n.y + fw * i.n.z);
  let part = u32(i.m.y * 255.0 + 0.5);
  var c = col;
  if (CREATURE == 1u) { c = select(select(vec3f(0.05), vec3f(0.85, 0.85, 0.8), part == 2u), vec3f(0.95, 0.75, 0.15), part == 3u); }
  else if (part == 2u) { c = vec3f(0.95, 0.9, 0.2); }
  else if (part == 3u) { c = vec3f(0.08, 0.06, 0.05); }
  o.c = srgb(c); o.e = 0.0;
  return o;
}
@fragment fn fsBird(i: BO, @builtin(front_facing) ff: bool) -> @location(0) vec4f {
  var n = normalize(i.n); if (!ff) { n = -n; }
  let sh = shadowAt(i.wp, 1.0);
  let c = i.c * (SUNC * sh * (max(dot(n, u.sun.xyz), 0.0) * 0.8 + 0.2) + ambient(n, i.wp) * 1.3);
  return enc(fogApply(c, i.wp));
}`;

// ---------------------------------------------------------------- volumetric light shafts and mist (half resolution)
const W_FOG = W_COMMON + /*wgsl*/`
@group(1) @binding(0) var tDep: texture_depth_multisampled_2d;
struct VO { @builtin(position) pos: vec4f, @location(0) q: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var o: VO; let q = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  o.pos = vec4f(q, 0.0, 1.0); o.q = q; return o;
}
struct FO { @location(0) c: vec4f, @location(1) d: f32 }
@fragment fn fs(i: VO) -> FO {
  let pix = vec2i(i.pos.xy) * 2;
  // farthest of the 2x2 full-res samples keeps thin leaves from blocking the shafts behind them
  let dz = textureLoad(tDep, pix + vec2i(i32(u.prm.y) & 1, (i32(u.prm.y) >> 1) & 1) * 0, 0);
  let rd0 = u.fwd.xyz + u.right.xyz * i.q.x * u.right.w + u.up.xyz * i.q.y * u.up.w;
  let wz = u.fwd.w / max(dz, 1e-7);
  let rd = normalize(rd0);
  let tmax = min(wz * length(rd0), 170.0);
  let N = 32;
  let jit = fract(52.9829189 * fract(dot(vec2f(i.pos.xy), vec2f(0.06711056, 0.00583715))));
  var T = 1.0; var Tm = 1.0; var ins = vec3f(0.0); var amb = vec3f(0.0);
  let mu = dot(rd, u.sun.xyz);
  let ph = hg(mu, 0.6) * 0.7 + hg(mu, -0.2) * 0.3 + 0.03;
  let t = u.cam.w;
  var prev = 0.0;
  for (var k = 0; k < N; k++) {
    let a = (f32(k) + jit) / f32(N);
    let tt = tmax * a * a;
    let dt = tt - prev; prev = tt;
    let p = u.cam.xyz + rd * tt;
    let m = mapAt(p.xz);
    // base haze (already in the analytic fog) only scatters sunlight here; ground mist and ravine mist also absorb
    let base = u.misc.x * exp(-(p.y - u.misc.y) / 22.0);
    let nz = n3(p * 0.021 + vec3f(t * 0.004, -t * 0.002, t * 0.003));
    let hg0 = max(p.y - m.r, 0.0);
    var mist = 0.035 * exp(-hg0 / 2.2) * smoothstep(0.42, 0.7, nz.x + nz.y * 0.3);
    mist += 0.05 * exp(-max(p.y + 6.0, 0.0) / 5.0) * (0.5 + nz.x);
    mist += 0.004 * smoothstep(0.45, 0.75, nz.y);
    let fd = distance(p, u.misc2.xyz);
    mist += 0.025 * exp(-fd * fd / (u.misc2.w * u.misc2.w)) * (0.6 + 0.8 * nz.z);
    let sh = shadowAt(p, 1.0);
    let sun = sh * (base + mist) * u.prm.w;
    ins += T * sun * dt;
    let sv = skyVis(p);
    amb += T * mist * dt * (0.25 + 0.75 * sv);
    T *= exp(-(base + mist) * dt);
    Tm *= exp(-mist * dt);
  }
  var o: FO;
  let ambc = (vec3f(0.42, 0.62, 0.78) * 1.1 + vec3f(0.12, 0.2, 0.08)) * 0.6;
  o.c = vec4f(SUNC * ph * ins + ambc * amb * u.prm.z * 1.6, Tm);
  o.d = wz;
  return o;
}`;

// separable depth-aware blur on the half-res fog
const W_BLUR = /*wgsl*/`
@group(0) @binding(0) var tC: texture_2d<f32>;
@group(0) @binding(1) var tD: texture_2d<f32>;
override DX: i32 = 1;
override DY: i32 = 0;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let q = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0; return vec4f(q, 0.0, 1.0); }
struct FO { @location(0) c: vec4f, @location(1) d: f32 }
@fragment fn fs(@builtin(position) pp: vec4f) -> FO {
  let p = vec2i(pp.xy); let sz = vec2i(textureDimensions(tC));
  let d0 = textureLoad(tD, p, 0).r;
  var s = vec4f(0.0); var w = 0.0;
  for (var k = -4; k <= 4; k++) {
    let q = clamp(p + vec2i(DX, DY) * k * 2, vec2i(0), sz - 1);
    let d = textureLoad(tD, q, 0).r;
    let wk = exp(-f32(k * k) / 10.0) * exp(-abs(d - d0) / (0.06 * d0 + 0.3));
    s += textureLoad(tC, q, 0) * wk; w += wk;
  }
  var o: FO; o.c = s / w; o.d = d0; return o;
}`;

// bloom: downsample chain and additive upsample
const W_BLOOM = /*wgsl*/`
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
fn dec(c0: vec3f) -> vec3f { var c = max(c0, vec3f(0.0)); let m0 = max(c.r, max(c.g, c.b)); if (m0 > 0.95) { c *= 0.95 / m0; } let m = max(c.r, max(c.g, c.b)); return c / (1.0 - m); }
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let q = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0; return vec4f(q, 0.0, 1.0); }
fn uvOf(pp: vec4f) -> vec2f { return pp.xy / vec2f(textureDimensions(t)) * 0.5; }
fn tap(uv: vec2f) -> vec3f { return textureSampleLevel(t, s, uv, 0.0).rgb; }
fn down(uv: vec2f, k: vec2f) -> vec3f {
  return (tap(uv) * 4.0 + tap(uv + k * vec2f(1.0, 1.0)) + tap(uv + k * vec2f(-1.0, 1.0)) + tap(uv + k * vec2f(1.0, -1.0)) + tap(uv + k * vec2f(-1.0, -1.0))) / 8.0;
}
@fragment fn fsDown0(@builtin(position) pp: vec4f) -> @location(0) vec4f {
  let k = 1.0 / vec2f(textureDimensions(t)); let uv = pp.xy * 2.0 * k;
  var c = vec3f(0.0);
  for (var j = 0; j < 4; j++) { let o = vec2f(f32(j & 1), f32(j >> 1)) * 2.0 - 1.0; let x = dec(tap(uv + o * k)); c += x / (1.0 + lum(x) * 0.25); }
  return vec4f(c * 0.25, 1.0);
}
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
@fragment fn fsDown(@builtin(position) pp: vec4f) -> @location(0) vec4f {
  let k = 1.0 / vec2f(textureDimensions(t)); return vec4f(down(pp.xy * 2.0 * k, k), 1.0);
}
@fragment fn fsUp(@builtin(position) pp: vec4f) -> @location(0) vec4f {
  let k = 1.0 / vec2f(textureDimensions(t)); let uv = pp.xy * k * 0.5;
  return vec4f(down(uv, k * 0.75) * 0.8, 1.0);
}`;

// final composite: fog shafts (bilateral upsample), bloom, exposure, tone map, grade, dither
const W_POST = W_COMMON + /*wgsl*/`
@group(1) @binding(0) var tCol: texture_2d<f32>;
@group(1) @binding(1) var tFog: texture_2d<f32>;
@group(1) @binding(2) var tFogD: texture_2d<f32>;
@group(1) @binding(3) var tBloom: texture_2d<f32>;
@group(1) @binding(4) var tDep: texture_depth_multisampled_2d;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let q = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0; return vec4f(q, 0.0, 1.0); }
fn tonemap(x: vec3f) -> vec3f {
  // ACES fitted (Narkowicz) with a softer toe
  let a = x * (2.51 * x + 0.03); let b = x * (2.43 * x + 0.59) + 0.14;
  return clamp(a / b, vec3f(0.0), vec3f(1.0));
}
@fragment fn fs(@builtin(position) pp: vec4f) -> @location(0) vec4f {
  let p = vec2i(pp.xy);
  var c = dec(textureLoad(tCol, p, 0).rgb);
  // bilateral upsample of the half-res shafts / mist
  let wz = u.fwd.w / max(textureLoad(tDep, p, 0), 1e-7);
  let fsz = vec2f(textureDimensions(tFog));
  let hp = pp.xy * 0.5 - 0.5; let b = vec2i(floor(hp)); let f = fract(hp);
  var fs = vec4f(0.0); var ws = 0.0; var bestD = 1e9; var best = vec4f(0.0, 0.0, 0.0, 1.0);
  for (var k = 0; k < 4; k++) {
    let o = vec2i(k & 1, k >> 1); let q = clamp(b + o, vec2i(0), vec2i(fsz) - 1);
    let bw = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    let d = textureLoad(tFogD, q, 0).r;
    let dd = abs(d - wz) / (0.04 * wz + 0.3);
    let w = bw * exp(-dd * dd);
    let v = textureLoad(tFog, q, 0);
    fs += v * w; ws += w;
    if (dd < bestD) { bestD = dd; best = v; }
  }
  // no neighbour at this depth (a silhouette edge): take the closest-depth sample instead of blending across it
  fs = select(best, fs / max(ws, 1e-6), ws > 0.05);
  c = c * fs.a + fs.rgb;
  let bl = textureSampleLevel(tBloom, sClamp, pp.xy * u.scr.zw, 0.0).rgb;
  c = mix(c, bl, 0.07);
  c *= u.sun.w;
  // grade: cool shadows, warm highlights, a touch more saturation
  let l = lum(c);
  c = mix(vec3f(l), c, 1.12);
  c *= mix(vec3f(0.92, 1.0, 1.04), vec3f(1.06, 1.0, 0.9), smoothstep(0.05, 1.2, l));
  var o = tonemap(c);
  // vignette
  let v = pp.xy * u.scr.zw - 0.5; o *= 1.0 - dot(v, v) * 0.45;
  o = pow(o, vec3f(1.0 / 2.2));
  // dither (blue-ish noise) against banding in the fog gradients
  let dn = fract(52.9829189 * fract(dot(pp.xy, vec2f(0.06711056, 0.00583715))) + u.prm.y * 0.618034);
  o += (dn - 0.5) / 255.0;
  return vec4f(o, 1.0);
}`;

// ============================================================================
// fly-through (F), benchmark (B), and hooks for the capture harness
// ============================================================================
{
  const LOGN = 1 << 15, logF = new Float32Array(LOGN), logG = new Float32Array(LOGN), logJ = new Float32Array(LOGN), logH = new Float32Array(LOGN);
  let mode = 0, nF = 0, nG = 0, lastGpu = -1; // mode 1: fly, 2: benchmark
  const stats = (a, n) => {
    const s = Array.from(a.subarray(0, n)).sort((x, y) => x - y); let sum = 0; for (const v of s) sum += v;
    return { n, avg: sum / n, p99: s[min(n - 1, floor(n * 0.99))], max: s[n - 1] };
  };
  const report = () => {
    const f = stats(logF, nF), g = nG ? stats(logG, nG) : null, fm = (o) => `avg ${o.avg.toFixed(2)}  p99 ${o.p99.toFixed(2)}  max ${o.max.toFixed(2)} ms`;
    const big = []; for (let i = 0; i < nF; i++) if (logF[i] > 25) big.push([i, +logF[i].toFixed(1), +logJ[i].toFixed(1), +(logH[i - 2] || 0).toFixed(1), +logH[i].toFixed(1)]);
    let jm = 0, js = 0; for (let i = 0; i < nF; i++) { jm = max(jm, logJ[i]); js += logJ[i]; }
    const r = { W, H, jsMax: jm, jsAvg: js / nF, big: big.filter((b) => b[1] > 50).slice(0, 20), frames: f, gpu: g, over16: Array.from(logF.subarray(0, nF)).filter((v) => v > 16.7).length, gpuOver16: g ? Array.from(logG.subarray(0, nG)).filter((v) => v > 16.7).length : 0 };
    info.textContent = `Fly-through ${W}x${H}, ${f.n} frames\nframe ${fm(f)}\nGPU   ${g ? fm(g) : 'timestamp-query unavailable'}`;
    window.__bench = r; console.log('BENCH ' + JSON.stringify(r));
  };
  const start = (m) => { mode = m; F[FLT] = 0; nF = nG = 0; lastGpu = GLN; tick = run; if (m === 2) { hud = false; info.textContent = ''; } };
  const stop = () => { mode = 0; tick = null; };
  const run = (dt) => {
    F[FLT] += dt; F[TIME] += dt;
    flyAt(F[FLT]);
    if (mode === 2) {
      if (nF < LOGN) { logH[nF] = F[RAWDT];
        logJ[nF] = F[JST]; logF[nF++] = dt * 1000; }
      
    }
    if (F[FLT] >= FLYT) { if (mode === 2) { setTimeout(() => { for (let i = lastGpu; i < GLN && nG < LOGN; i++) logG[nG++] = GLOG[i & (GLOGN - 1)]; report(); }, 600); } stop(); }
  };
  addEventListener('keydown', (e) => {
    if (e.repeat) return;
    if (e.code === 'KeyF' || e.code === 'KeyB') { if (mode) stop(); else start(e.code === 'KeyB' ? 2 : 1); }
    else if (/Digit[1-6]/.test(e.code)) stop();
  });
  // harness: render a fixed pose / a fixed fly-through time deterministically, then wait for the GPU
  const still = (t) => { paused = true; F[TIME] = t; };
  const settle = () => { F[FOGL] = fogLumAt(F[CX], F[CY], F[CZ]); F[EXPO] = expoAt(F[CX], F[CY], F[CZ]); };
  window.__h = {
    pose: async (i, t = 8) => { still(t); setPose(i); settle(); for (let k = 0; k < 3; k++) { render(); await Q.onSubmittedWorkDone(); } return [F[CX], F[CY], F[CZ]]; },
    fly: async (t) => { still(t); flyAt(t); settle(); render(); await Q.onSubmittedWorkDone(); },
    bench: () => new Promise((res) => { paused = false; window.__bench = null; start(2); const w = () => (window.__bench ? res(window.__bench) : setTimeout(w, 250)); w(); }),
    gpu: () => F[GPU],
    cam: async (x, y, z, tx, ty, tz, t = 8) => { still(t); aimAt(x, y, z, tx, ty, tz); settle(); for (let k = 0; k < 2; k++) { render(); await Q.onSubmittedWorkDone(); } },
    info: () => ({ shrine: [SHRINE.x, gy(SHRINE.x, SHRINE.z), SHRINE.z], bridgeY: BRIDGE_Y, poses: POSES, carve: CARVE, slev0: SLEV[0], fly: Array.from(FLY) }),
    counts: async (i) => { still(8); setPose(i); settle(); render(); const rb = device.createBuffer({ size: ARGS.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder(); e.copyBufferToBuffer(bArgs, 0, rb, 0, ARGS.byteLength); Q.submit([e.finish()]); await rb.mapAsync(GPUMapMode.READ);
      const a = new Uint32Array(rb.getMappedRange()).slice(); rb.unmap(); const o = []; let v = 0;
      for (let k = 0; k < NKIND; k++) for (let l = 0; l < 2; l++) { const n = a[(k * 2 + l) * 5 + 1]; o.push(`k${k}l${l}:${n}x${a[(k * 2 + l) * 5]}`); v += n * a[(k * 2 + l) * 5]; }
      return o.join(' ') + ` verts ${(v / 1e6).toFixed(1)}M`; },
    free: () => { paused = false; tick = null; },
    prof: async (i, mask, n = 300) => { still(8); if (i >= 100) flyAt(i - 100); else setPose(i); settle(); SKIP = mask; const g0 = GLN;
      for (let k = 0; k < n; k++) { render(); if (k % 4 === 3) await Q.onSubmittedWorkDone(); }
      await Q.onSubmittedWorkDone(); while (GLN < g0 + 200) await new Promise((r) => setTimeout(r, 50));
      const v = []; for (let j = GLN - 120; j < GLN; j++) v.push(GLOG[j & (GLOGN - 1)]); v.sort((x, y) => x - y); SKIP = 0;
      return +v[60].toFixed(2); },
  };
  if (QS.has('bench')) setTimeout(() => start(2), 3000);
}
// warm-up: draw every viewpoint and stretches of the path once so every pipeline variant is compiled before flight
await say('Warming up…');
for (let i = 0; i < POSES.length; i++) { setPose(i); render(); await Q.onSubmittedWorkDone(); }
for (let t = 0; t <= FLYT; t += 3) { flyAt(t); render(); await Q.onSubmittedWorkDone(); }
setPose(0);
$('s').textContent = 'Click to enter';
body.classList.add('ready');
requestAnimationFrame(frame);

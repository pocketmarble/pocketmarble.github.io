---
layout: page
title: III. the art of training
description: Diagnosing dead neurons, taming activations, and adding batch normalization
img: assets/img/makemore_batchnorm.png
importance: 4
category: makemore
---

This is part 3 of [makemore](https://github.com/karpathy/makemore), following the [lecture](https://www.youtube.com/watch?v=P6sfmUTpUmc) by Andrej Karpathy. In [part 2]({{ '/projects/makemore_mlp/' | relative_url }}) I built the multi-layer perceptron that generates names three characters at a time. The model worked — but it was quietly fighting itself at startup. This project looks under the hood at what happens in the very first steps of training: why a carelessly initialized network wastes time (or fails to train entirely), and how the fixes for it turn out to be the very tools that make deep learning possible.

Two problems hide at initialization. The output logits are too large, so the model starts out *confidently wrong* and burns the first hundreds of steps just deflating that false confidence. Worse, the `tanh` hidden activations are saturated — pinned at −1 or +1, where the gradient `1 − h²` vanishes and the neuron stops learning entirely. The fixes are a principled weight scaling ([Kaiming initialization](https://arxiv.org/abs/1502.01852)) and, ultimately, [batch normalization](https://arxiv.org/pdf/1502.03167) — the technique that made training deep networks practical.

The notebook below works through the diagnosis and the fixes from first principles:

- **A loss that starts too high** — the untrained model should sit near `-log(1/27) ≈ 3.29`, but it starts much higher because the initial logits are large and arbitrary. Squashing the output weights toward zero removes the confidently-wrong opening.
- **Saturated `tanh` and dead neurons** — visualizing the hidden activations shows most of them jammed against ±1, where the gradient vanishes. A neuron whose inputs are all saturated can never learn.
- **Kaiming initialization** — scaling the weights by `gain / √(fan-in)` keeps the variance of the pre-activations stable through the layer, with a `5/3` gain to compensate for the shape of `tanh`.
- **Batch normalization** — normalizing each batch's pre-activations during the forward pass, then learning a scale (`γ`) and shift (`β`). It keeps activations healthy without hand-tuning every weight matrix — the trick that makes truly deep networks trainable.
- **Inference after batch norm** — maintaining running estimates of the mean and variance during training so the model can still process a single example at test time.

The one-hidden-layer model above is shallow enough that a sloppy initialization only nudges the final loss. To feel why initialization actually *matters*, you need depth. Below is a **5-hidden-layer** `tanh` network — the same architecture, just stacked — measured at initialization (and over its first 1000 training steps) as a single knob is turned: the **gain** applied to each layer's weights.

Drag the slider and watch four panels move together — the top row shows the network's signals at initialization, the bottom row how large a step those signals translate into:

- **activations** — the `tanh` outputs of each layer. Too little gain and they collapse toward 0 (σ shrinks layer by layer); too much and they saturate at ±1.
- **activation gradients** — the gradient flowing back through each layer; it vanishes or explodes in a deep neural network when the activations are unhealthy, making training impossible.
- **update:data ratio** — `log₁₀( (lr·∇).std / weight.std )` per weight matrix over the first 1000 training steps; the dashed line marks the `-3` (≈1e-3) sweet spot.
- **grad:data ratio** — the same ratio at initialization, one bar per weight matrix. The **output layer (`out`) towers over the `-3` line**: it was deliberately shrunk (×0.1) to start unconfident, so one global learning rate moves it far faster than its size — the case for per-parameter learning rates.

Colors map to the layers via the legend under each widget (`emb`, `L1`–`L5`, `out`).

<div class="initviz" data-net="hand">
  <div class="iv-controls">
    <span class="iv-name">by-hand initialization</span>
    <label class="iv-label">gain&nbsp;<input type="range" class="iv-gain"></label>
    <span class="iv-gainval"></span>
  </div>
  <div class="iv-grid">
    <canvas class="iv-canvas iv-act"></canvas>
    <canvas class="iv-canvas iv-grad"></canvas>
    <canvas class="iv-canvas iv-ud"></canvas>
    <canvas class="iv-canvas iv-gd"></canvas>
  </div>
  <div class="iv-legend"></div>
</div>

Notice how **every** metric lurches as you scrub: the activations vanish then saturate, the gradients follow, and the update ratios drift off the `-3` line. Getting a deep network to train this way means hand-balancing `gain / √(fan-in)` for every layer — and re-balancing whenever the architecture changes. It's intractable.

Now the same network with a **batch-normalization** layer after each linear layer. Drag the same slider:

<div class="initviz" data-net="bn">
  <div class="iv-controls">
    <span class="iv-name">with batch normalization</span>
    <label class="iv-label">gain&nbsp;<input type="range" class="iv-gain"></label>
    <span class="iv-gainval"></span>
  </div>
  <div class="iv-grid">
    <canvas class="iv-canvas iv-act"></canvas>
    <canvas class="iv-canvas iv-grad"></canvas>
    <canvas class="iv-canvas iv-ud"></canvas>
    <canvas class="iv-canvas iv-gd"></canvas>
  </div>
  <div class="iv-legend"></div>
</div>

The activations barely move. σ and saturation remain consistent across the *entire* gain range, and the gradients and update ratios stay put with them. **Batch norm decouples the health of the network from the precise scale of the initialization, which is exactly what makes training deep networks practical.**

<style>
.initviz { border: 1px solid #e3e3e3; border-radius: 8px; padding: 12px 14px; margin: 1.2rem 0; }
.initviz .iv-controls { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 8px; }
.initviz .iv-name { font-weight: 600; }
.initviz .iv-label { display: flex; align-items: center; }
.initviz .iv-gain { width: 170px; max-width: 45vw; margin-left: 6px; }
.initviz .iv-gainval { font-family: ui-monospace, monospace; min-width: 2.4em; }
.initviz .iv-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.initviz .iv-canvas { width: 100%; aspect-ratio: 1.9; display: block; }
.initviz .iv-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; color: #555; }
.initviz .iv-legend span { display: inline-flex; align-items: center; gap: 4px; }
.initviz .iv-legend i { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
</style>

<script>
(function () {
  const PAL = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2'];
  const ENT = ['emb', 'L1', 'L2', 'L3', 'L4', 'L5', 'out'];

  function fmt(v) {
    if (v == null) return '—';
    const a = Math.abs(v);
    if (a !== 0 && (a < 0.01 || a >= 1000)) return v.toExponential(0);
    return (Math.round(v * 100) / 100).toString();
  }

  // hi-DPI: size the backing buffer to CSS pixels × devicePixelRatio so lines stay crisp
  function prep(cv) {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    return { ctx, W, H };
  }
  function frame(ctx, m, W, H) { ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1; ctx.strokeRect(m.l, m.t, W - m.l - m.r, H - m.t - m.b); }
  function title(ctx, m, t) { ctx.fillStyle = '#333'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(t, m.l, 1); }
  function ylab(ctx, m, py, lo, hi) {
    ctx.fillStyle = '#999'; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(fmt(hi), m.l - 3, py(hi)); ctx.fillText(fmt(lo), m.l - 3, py(lo));
  }
  function yticks(ctx, m, W, py, lo, hi, step) {
    ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      const Y = py(v);
      ctx.strokeStyle = '#eee'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(m.l, Y); ctx.lineTo(W - m.r, Y); ctx.stroke();
      ctx.fillStyle = '#999'; ctx.fillText(fmt(v), m.l - 3, Y);
    }
  }

  function lineChart(cv, xs, series, colors, opts) {
    const r = prep(cv), ctx = r.ctx, W = r.W, H = r.H, m = { l: 34, r: 6, t: 14, b: 14 };
    const x0 = xs[0], x1 = xs[xs.length - 1], px = x => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r);
    let lo, hi;
    if (opts.yRange) { lo = opts.yRange[0]; hi = opts.yRange[1]; }
    else { hi = 0; for (const s of series) for (const v of s) if (v != null && v > hi) hi = v; lo = 0; hi = hi > 0 ? hi * 1.08 : 1; }
    const py = y => (H - m.b) - (y - lo) / (hi - lo) * (H - m.t - m.b);
    frame(ctx, m, W, H);
    if (opts.ytickStep) yticks(ctx, m, W, py, lo, hi, opts.ytickStep); else ylab(ctx, m, py, lo, hi);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(fmt(x0), px(x0), H - m.b + 2); ctx.fillText(fmt(x1), px(x1), H - m.b + 2);
    (opts.hlines || []).forEach(h => { ctx.strokeStyle = h.color; ctx.setLineDash(h.dash || []); ctx.beginPath(); ctx.moveTo(px(x0), py(h.y)); ctx.lineTo(px(x1), py(h.y)); ctx.stroke(); ctx.setLineDash([]); });
    series.forEach((s, si) => {
      ctx.strokeStyle = colors[si]; ctx.lineWidth = 1.3; ctx.beginPath(); let st = false;
      for (let i = 0; i < xs.length; i++) { const v = s[i]; if (v == null) { st = false; continue; } const X = px(xs[i]), Y = py(Math.max(lo, Math.min(hi, v))); if (!st) { ctx.moveTo(X, Y); st = true; } else ctx.lineTo(X, Y); }
      ctx.stroke();
    });
    title(ctx, m, opts.title || '');
  }

  function barChart(cv, labels, vals, colors, opts) {
    const r = prep(cv), ctx = r.ctx, W = r.W, H = r.H, m = { l: 30, r: 6, t: 14, b: 20 };
    const lo = opts.yRange[0], hi = opts.yRange[1], n = labels.length, plotW = W - m.l - m.r;
    const py = y => (H - m.b) - (y - lo) / (hi - lo) * (H - m.t - m.b);
    frame(ctx, m, W, H); yticks(ctx, m, W, py, lo, hi, opts.ytickStep || 1);
    if (opts.refY != null) { ctx.strokeStyle = '#000'; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(m.l, py(opts.refY)); ctx.lineTo(W - m.r, py(opts.refY)); ctx.stroke(); ctx.setLineDash([]); }
    const bw = plotW / n * 0.6, base = py(lo);
    for (let i = 0; i < n; i++) {
      const v = vals[i], cx = m.l + (i + 0.5) * plotW / n;
      if (v != null && v > 0) {
        const top = py(Math.max(lo, Math.min(hi, Math.log10(v))));
        ctx.fillStyle = colors[i]; ctx.fillRect(cx - bw / 2, Math.min(top, base), bw, Math.abs(base - top));
      }
      ctx.fillStyle = '#666'; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(labels[i], cx, H - m.b + 2);
    }
    title(ctx, m, opts.title || '');
  }

  function makeWidget(root, key, D) {
    const F = D[key], q = s => root.querySelector(s);
    const slider = q('.iv-gain'), gv = q('.iv-gainval');
    const cAct = q('.iv-act'), cGrad = q('.iv-grad'), cUd = q('.iv-ud'), cGd = q('.iv-gd');
    q('.iv-legend').innerHTML = ENT.map((e, i) => '<span><i style="background:' + PAL[i] + '"></i>' + e + '</span>').join('');
    slider.min = 0; slider.max = D.gains.length - 1; slider.step = 1;
    slider.value = Math.round((5 / 3 - D.gains[0]) / (D.gains[1] - D.gains[0]));
    const colT = PAL.slice(1, 6), colW = PAL.slice(0, 7);
    function update() {
      const gi = +slider.value;
      gv.textContent = D.gains[gi].toFixed(1);
      lineChart(cAct, D.x.act, F.act[gi], colT, { title: 'activations' });
      lineChart(cGrad, D.x.actgrad, F.actgrad[gi], colT, { title: 'activation gradients' });
      lineChart(cUd, D.udSteps, F.ud[gi], colW, { title: 'update:data over training (log₁₀)', yRange: [-6, 0.5], hlines: [{ y: -3, color: '#000', dash: [4, 3] }], ytickStep: 2 });
      barChart(cGd, ENT, F.ratio[gi], colW, { title: 'grad:data ratio (log₁₀)', yRange: [-4, 1], refY: -3 });
    }
    root._update = update;
    slider.addEventListener('input', update);
    update();
  }

  fetch("{{ '/assets/json/makemore_initviz.json' | relative_url }}")
    .then(r => r.json())
    .then(D => {
      const roots = Array.prototype.slice.call(document.querySelectorAll('.initviz'));
      roots.forEach(root => makeWidget(root, root.dataset.net, D));
      let t; window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(() => roots.forEach(r => r._update && r._update()), 120); });
    })
    .catch(e => console.error('initviz load failed', e));
})();
</script>

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/makemore_batchnorm.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/makemore_batchnorm.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}
<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

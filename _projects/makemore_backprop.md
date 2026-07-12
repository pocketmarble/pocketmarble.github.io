---
layout: page
title: IV. backprop, by hand
description: Deriving the backward pass of an entire MLP without autograd, then collapsing it
img: assets/img/makemore_backprop.png
importance: 5
category: makemore
---

This is part 4 of [makemore](https://github.com/karpathy/makemore), following the [lecture](https://www.youtube.com/watch?v=q8SA3rM6ckI) by Andrej Karpathy. [Part 3]({{ '/projects/makemore_batchnorm/' | relative_url }}) added batch normalization but called `loss.backward()` and trusted it. Here I delete it: backpropagate the whole network by hand — every intermediate tensor and every parameter, from `loss` back to the embedding table `C` — check each gradient against PyTorch's `.grad`, then collapse the two ugliest blocks into single lines.

One principle generates every gradient below:

> **Think in single scalars. Trace the routes an input takes to the loss. Sum over them. Read the index pattern back as a matrix operation.**

## The gradient checker

`cmp` compares a hand-computed gradient against PyTorch's `.grad`: **exact** (bit-for-bit), **approximate** (`torch.allclose`), **maxdiff** (largest disagreement).

`exact: True` is a bonus, not the target. Float addition isn't associative, so the same gradient computed in a different operation order lands a few ULPs off — `exact: False`, `approximate: True`, `maxdiff ≈ 1e-9`. That's a pass.

## Exercise 1 — every node by hand

The forward pass is split into atomic steps so every intermediate has a `.grad` to check against. Walking it in reverse, the cross-entropy chain:

- **`logprobs`** — loss is `-logprobs[range(n), Yb].mean()`, so only correct-label entries matter, each with derivative $$-1/n$$; the rest are zero.
- **`probs`** — through `probs.log()`, derivative $$1/\text{probs}$$.
- **`counts_sum_inv`** — `probs = counts * counts_sum_inv` broadcasts a `[n,1]` tensor across 27 columns, so its gradient sums along the row.
- **`counts`** — the first multivariable node: it reaches the loss through **two** routes, via `probs` and via `counts_sum`. The contributions add.
- then **`norm_logits`**, **`logit_maxes`**, **`logits`**.

`logit_maxes` contributes nothing — the routes through it cancel (proved in Exercise 2). The hand-code carries it anyway and `cmp` confirms it comes out zero.

### Linear layer

For $$\text{logits} = h\,W_2 + b_2$$, the notebook expands all ten $$\partial h_{ij}$$ and all fifteen $$\partial W_{jb}$$ term by term. The general entries:

$$
\partial h_{ij} = \sum_{b} \partial l_{ib}\, W_{jb},
\qquad
\partial W_{jb} = \sum_{i} \partial l_{ib}\, h_{ij},
\qquad
\partial b_k = \sum_i \partial l_{ik}
$$

which read back as

$$
\partial h = \partial l \, W^{\top}, \qquad \partial W = h^{\top}\, \partial l, \qquad \partial b = \textstyle\sum_i \partial l_{i,:}
$$

The transpose isn't a convention to memorize — it's forced by which index is summed. In $$\partial h_{ij}$$ the contracted index $$b$$ sits in the **second** slot of both factors, and matmul contracts the second slot of the left against the first of the right, so $$W$$ transposes. In $$\partial W_{jb}$$ the contracted index is $$i$$, in the **first** slot of both, so $$h$$ transposes instead.

**Forward fan-out becomes backward summation**: matmul reuse and a broadcast bias both reverse into a sum.

### The rest of the graph

- **`tanh`** — elementwise, derivative $$1 - h^2$$.
- **the batch-norm block** — node by node. `bn_var` carries Bessel's $$1/(n-1)$$, not $$1/n$$ — differentiate the code you actually wrote. And `z_pre` reaches the loss through two routes: directly via `bn_del`, and via `bn_mean`, which is a function of every `z_pre` in the column. That batch-crossing route is the difficulty in Exercise 3.
- **`.view`** — pure reshape; its backward is a reshape.
- **`C[Xb]`** — a row of `C` is read many times across the batch (the same character in many contexts). That's fan-out, so gradients **scatter-add** back — `+=`, not `=`. Every appearance of "a" piles onto row `ctoi['a']`.

Every gradient in Exercise 1 comes back **`exact: True`, `maxdiff: 0.0`** — bit-for-bit identical to autograd, not merely close. The only two that miss are the shortcuts below, at `~1e-9`: same quantity, different operation order.

## Exercise 2 — cross-entropy in one line

Six intermediates (`logit_maxes`, `norm_logits`, `counts`, `counts_sum`, `counts_sum_inv`, `probs`), each an `exp`, a `log`, or a division. Substituting each forward line into the next collapses the loss to a log-sum-exp:

$$
\text{loss}_i = \ln\Big(\sum_k e^{\text{logit}_{ik}}\Big) - \text{logit}_{iy}
$$

Differentiating: the label term gives $$-1$$ at $$j = y_i$$. In the log-sum term only the $$k=j$$ term survives, and the fraction left behind is the softmax probability already computed in the forward pass:

$$
\frac{\partial \text{loss}_i}{\partial \text{logit}_{ij}} = \text{probs}_{ij} - \mathbb{1}[j = y_i]
$$

Predicted minus target — take `probs`, subtract 1 at the correct class, divide by $$n$$.

The same algebra kills the max-subtraction. Substituting $$\text{logit}_{ik} \to \text{logit}_{ik} - m_i$$, the constant $$e^{-m_i}$$ factors out of the sum, so the log-sum term contributes $$-m_i$$ and the label term $$+m_i$$. They cancel: the loss and every gradient are unchanged, which is why it can be dropped when differentiating.

The one-liner agrees to `~1e-9` with the `dlogits` computed the long way.

## Exercise 3 — batch norm in one line

`tanh` is elementwise — output $$a$$ depends only on input $$a$$. Batch norm isn't. The mean and variance are computed **down the column**, across the batch, so nudging one input moves **every** output in its column, through three channels: **directly**, through the shared **mean**, and through the shared **variance**.

Written as a grid — one cell per (output $$a$$, input $$i$$) pair, with $$x$$ for `z_pre` and $$\hat z$$ for the normalized value:

$$
\frac{\partial \hat z_a}{\partial x_i} = (\sigma^2+\epsilon)^{-1/2}\Big[\;\underbrace{\mathbb{1}[a{=}i]}_{\text{direct}} \;-\; \underbrace{\tfrac{1}{n}}_{\text{mean}} \;-\; \underbrace{\tfrac{1}{n-1}\,\hat z_a\,\hat z_i}_{\text{variance}}\;\Big]
$$

`tanh` gives a **diagonal** Jacobian. Batch norm gives a **full** one — every cell nonzero.

Below is that grid from the real network at initialization ($$n = 32$$, one hidden column), `tanh` on the left. Switch the three routes **off** to see what each contributes: **direct** is the diagonal, **mean** is a uniform wash, **variance** is the outer product $$\hat z_a \hat z_i$$ that gives the grid its texture. Hover any cell for its value.

<div class="jacviz">
  <div class="jv-controls">
    <label><input type="checkbox" class="jv-route" data-route="direct" checked> <span class="jv-sw jv-sw-d"></span>direct <code>1[a=i]</code></label>
    <label><input type="checkbox" class="jv-route" data-route="mean" checked> <span class="jv-sw jv-sw-m"></span>mean <code>−1/n</code></label>
    <label><input type="checkbox" class="jv-route" data-route="var" checked> <span class="jv-sw jv-sw-v"></span>variance <code>−ẑₐẑᵢ/(n−1)</code></label>
  </div>
  <div class="jv-grid">
    <figure class="jv-fig">
      <figcaption>tanh — <b>diagonal</b></figcaption>
      <canvas class="jv-canvas jv-tanh"></canvas>
      <div class="jv-count jv-count-t"></div>
    </figure>
    <figure class="jv-fig">
      <figcaption>batch norm — <b class="jv-shape">full</b></figcaption>
      <canvas class="jv-canvas jv-bn"></canvas>
      <div class="jv-count jv-count-b"></div>
    </figure>
  </div>
  <div class="jv-scale">
    <span>−max</span><span class="jv-ramp"></span><span>+max</span>
    <em>signed √ scale · axes: output <b>a</b> (rows) × input <b>i</b> (columns)</em>
  </div>
  <div class="jv-tip" hidden></div>
</div>

Two facts make the full Jacobian tractable. The 64 columns are independent — 64 parallel batch-norm ops stored side by side — so fix one column and solve an $$n$$-in / $$n$$-out problem. And $$\sum_a (x_a - \mu) = 0$$: deviations from the mean sum to zero, which is what makes the variance route collapse.

Summing the Jacobian against the gradient arriving at each of the $$n$$ outputs:

$$
\frac{\partial \mathcal L}{\partial x_i}
= \gamma\,(\sigma^2+\epsilon)^{-1/2}\Big[\,\partial z_i \;-\; \tfrac1n\textstyle\sum_a \partial z_a \;-\; \tfrac{\hat z_i}{n-1}\textstyle\sum_a \partial z_a\,\hat z_a\,\Big]
$$

The two sums are one number per column — computed once, reused for every row. In code, two `.sum(0)` reductions:

```python
dz_pre = bn_gain * bn_var_inv * (dz - (1/n)*dz.sum(0,keepdim=True)
                                - (bn_raw/(n-1))*(dz*bn_raw).sum(0,keepdim=True))
```

That replaces the whole `bn_raw → bn_var_inv → bn_var → bn_del2 → bn_del → bn_mean → z_pre` chain and never materializes an intermediate.

## The method

Three collapses — `probs - onehot`, $$\partial l\,W^{\top}$$ and $$h^{\top}\partial l$$, and the batch-norm expression — all from one procedure:

1. **Think in single scalars.** How much does one output-number move when I nudge one input-number. Matrices are storage; the calculus is scalar.
2. **Trace the routes.** Which downstream quantities does the nudged input reach? One route for an elementwise op; a whole column for a batch statistic.
3. **Sum over the routes**, each weighted by the gradient already arriving there.
4. **Read the index pattern back as a matrix operation.** Where the contracted index sits determines the transpose; a matched index means elementwise; a summed index means a reduction.

Two ideas do all the work: forward fan-out becomes backward summation, and the shape of the Jacobian gives the shape of the answer — diagonal means elementwise, full means a sum over the coupled dimension.

<style>
.jacviz { border: 1px solid #e3e3e3; border-radius: 8px; padding: 14px; margin: 1.4rem 0; position: relative; }
.jacviz .jv-controls { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; font-size: 0.85rem; }
.jacviz .jv-controls label { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; margin: 0; font-weight: 400; }
.jacviz .jv-controls code { font-size: 0.8em; color: #666; }
.jacviz .jv-sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.jacviz .jv-sw-d { background: #892c2a; }
.jacviz .jv-sw-m { background: #9ec5f4; }
.jacviz .jv-sw-v { background: #d75852; }
.jacviz .jv-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.jacviz .jv-fig { margin: 0; }
.jacviz .jv-fig figcaption { font-size: 0.8rem; color: #555; text-align: center; margin-bottom: 6px; }
.jacviz .jv-canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border: 1px solid #e1e0d9; border-radius: 3px; }
.jacviz .jv-count { font: 10.5px ui-monospace, SFMono-Regular, Menlo, monospace; color: #777; text-align: center; margin-top: 6px; }
.jacviz .jv-scale { display: flex; align-items: center; gap: 7px; margin-top: 12px; font: 10px ui-monospace, monospace; color: #999; }
.jacviz .jv-ramp { width: 110px; height: 8px; border-radius: 2px; border: 1px solid #e1e0d9;
  background: linear-gradient(to right, #0d366b, #256abf, #6da7ec, #cde2fb, #f0efec, #f1aea8, #d75852, #b13f3c, #621b1a); }
.jacviz .jv-scale em { font-style: normal; margin-left: auto; text-align: right; }
.jacviz .jv-tip { position: absolute; pointer-events: none; background: #0b0b0b; color: #fff; padding: 4px 7px;
  border-radius: 4px; font: 11px ui-monospace, monospace; white-space: nowrap; z-index: 10; transform: translate(-50%, -130%); }
@media (max-width: 600px) {
  .jacviz .jv-grid { grid-template-columns: 1fr; }
  .jacviz .jv-scale em { display: none; }
}
</style>

<script>
(function () {
  const N = 32, VINV = 0.639895;
  // ẑ and 1−h² for one real hidden column (col 35) of the network at initialization, seed 1994
  const ZHAT = [1.6109,-2.5259,-0.6343,0.8843,-0.2790,-1.7162,-0.1278,-0.2416,-0.0424,0.1764,1.3924,-0.4648,0.3391,1.3924,0.7540,-1.5263,-1.7162,-0.4594,0.7673,0.0257,0.2383,0.7540,0.0257,0.0257,0.6681,1.0207,0.0257,-0.7811,0.0257,-0.8374,1.7703,-0.5443];
  const DTANH = [0.1003,0.0192,0.6998,0.4013,0.9487,0.1061,0.9956,0.9647,0.9993,0.9340,0.1561,0.8346,0.8295,0.1561,0.4950,0.1557,0.1061,0.8385,0.4849,0.9901,0.8986,0.4950,0.9901,0.9901,0.5624,0.3167,0.9901,0.5782,0.9901,0.5330,0.0721,0.7733];

  // diverging ramp: blue (−) → neutral gray (0) → red (+). Lightness monotonic per arm.
  const RAMP = ['#0d366b','#184f95','#256abf','#3987e5','#6da7ec','#9ec5f4','#cde2fb',
                '#f0efec','#fad6d2','#f1aea8','#e4857d','#d75852','#b13f3c','#892c2a','#621b1a'];
  const RGB = RAMP.map(h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)]);

  // map v ∈ [−1,1] through the ramp with linear interpolation between stops
  function color(t) {
    const x = Math.max(0, Math.min(1, (t + 1) / 2)) * (RGB.length - 1);
    const i = Math.min(RGB.length - 2, Math.floor(x)), f = x - i;
    const a = RGB[i], b = RGB[i + 1];
    return [a[0] + (b[0]-a[0])*f, a[1] + (b[1]-a[1])*f, a[2] + (b[2]-a[2])*f];
  }

  // the Jacobian ∂ẑ_a/∂z_pre_i, one route at a time — exactly the boxed formula
  function jacBN(routes) {
    const J = new Float64Array(N * N);
    for (let a = 0; a < N; a++) for (let i = 0; i < N; i++) {
      let v = 0;
      if (routes.direct && a === i) v += 1;
      if (routes.mean)             v -= 1 / N;
      if (routes.var)              v -= ZHAT[a] * ZHAT[i] / (N - 1);
      J[a*N + i] = VINV * v;
    }
    return J;
  }
  function jacTanh() {
    const J = new Float64Array(N * N);
    for (let a = 0; a < N; a++) J[a*N + a] = DTANH[a];  // off-diagonal is exactly 0
    return J;
  }

  function draw(cv, J, vmax) {
    const dpr = window.devicePixelRatio || 1, S = cv.clientWidth;
    if (cv.width !== Math.round(S*dpr)) { cv.width = Math.round(S*dpr); cv.height = Math.round(S*dpr); }
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(N, N);
    for (let k = 0; k < N*N; k++) {
      const v = J[k];
      const t = vmax > 0 ? Math.sign(v) * Math.sqrt(Math.abs(v) / vmax) : 0;  // shared signed-√ scale
      const c = color(t);
      img.data[k*4] = c[0]; img.data[k*4+1] = c[1]; img.data[k*4+2] = c[2]; img.data[k*4+3] = 255;
    }
    // blit the N×N image up to the canvas size without smoothing
    const off = document.createElement('canvas');
    off.width = N; off.height = N;
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(off, 0, 0, cv.width, cv.height);
  }

  function nonzeroOff(J) {
    let c = 0;
    for (let a = 0; a < N; a++) for (let i = 0; i < N; i++) if (a !== i && J[a*N+i] !== 0) c++;
    return c;
  }

  document.querySelectorAll('.jacviz').forEach(function (root) {
    const cvT = root.querySelector('.jv-tanh'), cvB = root.querySelector('.jv-bn');
    const cntT = root.querySelector('.jv-count-t'), cntB = root.querySelector('.jv-count-b');
    const shape = root.querySelector('.jv-shape'), tip = root.querySelector('.jv-tip');
    const boxes = Array.prototype.slice.call(root.querySelectorAll('.jv-route'));
    const JT = jacTanh();
    let JB = null;

    function render() {
      const routes = {};
      boxes.forEach(b => routes[b.dataset.route] = b.checked);
      JB = jacBN(routes);
      // one shared scale across both panels — the comparison must be honest
      let vmax = 0;
      for (let k = 0; k < N*N; k++) { const m = Math.abs(JB[k]); if (m > vmax) vmax = m; }
      for (let k = 0; k < N*N; k++) { const m = Math.abs(JT[k]); if (m > vmax) vmax = m; }
      draw(cvT, JT, vmax);
      draw(cvB, JB, vmax);
      const nzB = nonzeroOff(JB), OFF = N*N - N;
      cntT.textContent = '0 / ' + OFF + ' off-diagonal cells nonzero';
      cntB.textContent = nzB + ' / ' + OFF + ' off-diagonal cells nonzero';
      shape.textContent = nzB === 0 ? 'diagonal' : 'full';
    }

    function hover(cv, getJ, label) {
      cv.addEventListener('mousemove', function (e) {
        const r = cv.getBoundingClientRect();
        const i = Math.floor((e.clientX - r.left) / r.width * N);
        const a = Math.floor((e.clientY - r.top) / r.height * N);
        if (a < 0 || a >= N || i < 0 || i >= N) return;
        const v = getJ()[a*N + i];
        tip.hidden = false;
        tip.textContent = label + '  a=' + a + ', i=' + i + '  →  ' + (v === 0 ? '0' : v.toFixed(4));
        const rr = root.getBoundingClientRect();
        tip.style.left = (e.clientX - rr.left) + 'px';
        tip.style.top = (e.clientY - rr.top) + 'px';
      });
      cv.addEventListener('mouseleave', function () { tip.hidden = true; });
    }
    hover(cvT, () => JT, 'tanh');
    hover(cvB, () => JB, 'bn');

    boxes.forEach(b => b.addEventListener('change', render));
    let t; window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(render, 120); });
    render();
  });
})();
</script>

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/makemore_backprop.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/makemore_backprop.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}
<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

---
layout: page
title: building an MLP language model from scratch
description: A neural network that learns to dream up names, three characters at a time
img: assets/img/makemore_mlp.png
importance: 3
category: work
---

This is part 2 of [makemore](https://github.com/karpathy/makemore), following Andrej Karpathy's [walkthrough](https://www.youtube.com/watch?v=TCH_1BHY58I). Where [part 1]({{ '/projects/makemore/' | relative_url }}) showed the limits of a bigram model — it can only ever look one character back — this project implements the multi-layer perceptron (MLP) language model described by [Bengio et al. (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), adapted to the character level for continuity with part 1.

Instead of a lookup table of counts, each character is mapped into a learned embedding space, a block of 3 characters of context is fed through a hidden layer, and the network predicts the next character. Trained with gradient descent, it reaches a loss of ~2.1 — comfortably better than the bigram model — and generates names that genuinely look plausible.

The model below is the **trained MLP** from the notebook (10-dimensional embeddings, a 200-neuron hidden layer, ~12k parameters), running its full forward pass right in your browser. Click to generate some names:

<div class="row justify-content-sm-center mt-3 mb-4">
  <div class="col-sm-10 text-center">
    <button id="mlp-btn" class="btn btn-primary" disabled>loading model…</button>
    <pre id="mlp-output" style="margin-top: 1rem; min-height: 7.5rem; font-size: 1.1rem; line-height: 1.5;"></pre>
  </div>
</div>

<script>
(function () {
  const itoc = '.abcdefghijklmnopqrstuvwxyz';
  const btn = document.getElementById('mlp-btn');
  const out = document.getElementById('mlp-output');
  let M = null; // model weights, loaded asynchronously

  // forward pass of the MLP for a single 3-character context, returning a probability distribution
  function forward(ctx) {
    // embed each context character and concatenate into one vector
    const x = [];
    for (const ci of ctx) for (const v of M.C[ci]) x.push(v);

    // hidden layer: h = tanh(x @ w1 + b1)
    const H = M.b1.length;
    const h = new Array(H);
    for (let j = 0; j < H; j++) {
      let s = M.b1[j];
      for (let k = 0; k < x.length; k++) s += x[k] * M.w1[k][j];
      h[j] = Math.tanh(s);
    }

    // output layer: logits = h @ w2 + b2
    const L = M.b2.length;
    const logits = new Array(L);
    for (let j = 0; j < L; j++) {
      let s = M.b2[j];
      for (let k = 0; k < H; k++) s += h[k] * M.w2[k][j];
      logits[j] = s;
    }

    // softmax into a probability distribution
    let mx = -Infinity;
    for (const v of logits) if (v > mx) mx = v;
    let tot = 0;
    const p = logits.map(v => { const e = Math.exp(v - mx); tot += e; return e; });
    for (let j = 0; j < L; j++) p[j] /= tot;
    return p;
  }

  function sample(p) {
    let r = Math.random();
    for (let j = 0; j < p.length; j++) { r -= p[j]; if (r < 0) return j; }
    return p.length - 1;
  }

  function makeName() {
    let ctx = new Array(M.block_size).fill(0);
    let name = '';
    while (true) {
      const idx = sample(forward(ctx));
      if (idx === 0) break;          // sampled the end token
      name += itoc[idx];
      ctx = ctx.slice(1).concat(idx);
      if (name.length > 20) break;   // safety cap
    }
    return name;
  }

  fetch("{{ '/assets/json/makemore_mlp_weights.json' | relative_url }}")
    .then(r => r.json())
    .then(data => {
      M = data;
      btn.disabled = false;
      btn.textContent = '✨ generate ..."names"';
    })
    .catch(() => { btn.textContent = 'failed to load model'; });

  btn.addEventListener('click', function () {
    if (!M) return;
    const names = [];
    for (let i = 0; i < 5; i++) names.push(makeName());
    out.textContent = names.join('\n');
  });
})();
</script>

The notebook below builds the model up from first principles:

- **Character embeddings** — mapping each of the 27 characters into a learned, low-dimensional feature space (the idea at the heart of the Bengio paper).
- **The context window** — using a block of 3 previous characters, concatenating their embeddings as the network's input.
- **The MLP** — a `tanh` hidden layer feeding a softmax output over the 27 possible next characters, trained with cross-entropy loss.
- **Practical training** — minibatching, finding a good learning rate, and splitting the data into train/dev/test to diagnose under- vs. overfitting.
- **Tuning** — growing the hidden layer and the embedding dimension to push the loss down and improve the quality of generated names.

A nice side effect of training is that the learned character embeddings become meaningful. Below is the 2D embedding space from one of the intermediate models — notice how the vowels cluster together, while rare characters like `q` drift off as outliers:

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/makemore_mlp.png" title="learned 2D character embeddings — vowels cluster together" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/makemore_mlp.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/makemore_mlp.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}
<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

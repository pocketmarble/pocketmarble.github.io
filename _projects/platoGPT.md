---
layout: page
title: a transformer, from scratch
description: building a GPT and training it to dream in Plato
img: assets/img/platoGPT.png
importance: 1
category: nanoGPT
---

Following [*Let's build GPT: from scratch*](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy. After the [makemore](https://github.com/karpathy/makemore) series built up to an MLP and [backpropagated it by hand]({{ '/projects/makemore_backprop/' | relative_url }}), this project assembles the model everything was pointing at — a **decoder-only transformer**: self-attention, multiple heads, residual connections, layer norm. Then I train it, one character at a time, on the complete dialogues of **Plato**. The aim is to understand the transformer as a model; [the walkthrough below](#the-transformer) builds it from the makemore toolkit and nothing more.

The result is ~10.8M parameters that learn to *dream* in Plato — the cadence of the dialogues, the named speakers, the em-dashes and endless qualifications, all hallucinated character by character. Below, the model does exactly that, forever.

<div class="plato-scroll-wrap">
  <div id="plato-scroll" class="plato-scroll" aria-label="endless stream of machine-generated pseudo-Plato">
    <div id="plato-tape" class="plato-tape">summoning the ghost of Plato…</div>
  </div>
  <div class="plato-scroll-caption">↑ every character above was hallucinated by the model, one at a time · <a href="{{ '/assets/plato/pseudo_plato.txt' | relative_url }}" target="_blank" rel="noopener">raw sample</a></div>
</div>

<style>
  .plato-scroll-wrap { margin: 1.5rem 0 2rem; }
  .plato-scroll {
    position: relative;
    height: min(58vh, 520px);
    overflow: hidden;
    border: 1px solid var(--global-divider-color, rgba(0,0,0,.1));
    border-radius: 8px;
    background: var(--global-code-bg-color, rgba(0,0,0,.02));
    /* fade the text out at the top and bottom edges */
    -webkit-mask-image: linear-gradient(to bottom, transparent 0, #000 14%, #000 86%, transparent 100%);
            mask-image: linear-gradient(to bottom, transparent 0, #000 14%, #000 86%, transparent 100%);
  }
  .plato-tape {
    padding: 1.4rem 1.6rem;
    font-family: Georgia, Cambria, "Times New Roman", serif;
    font-size: 1.02rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--global-text-color, #333);
    will-change: transform;
  }
  .plato-cursor {
    display: inline-block;
    width: .5ch;
    margin-left: 1px;
    background: currentColor;
    opacity: .8;
    animation: plato-blink 1.05s steps(1) infinite;
  }
  @keyframes plato-blink { 50% { opacity: 0; } }
  .plato-scroll-caption {
    margin-top: .55rem;
    font-size: .78rem;
    text-align: center;
    color: var(--global-text-color-light, #888);
  }
  @media (prefers-reduced-motion: reduce) {
    .plato-cursor { animation: none; }
  }
  .transformer-fig { max-width: 880px; margin: 1.6rem auto 0.4rem; overflow-x: auto; }
  .transformer-fig svg { width: 100%; height: auto; display: block; min-width: 560px; }
  .transformer-fig-cap {
    margin: .6rem auto 0; max-width: 780px;
    font-size: .8rem; line-height: 1.5; text-align: center;
    color: var(--global-text-color-light, #888);
  }
</style>

<script>
(function () {
  var CORPUS_URL = "{{ '/assets/plato/pseudo_plato.txt' | relative_url }}";
  var viewport = document.getElementById('plato-scroll');
  var tape     = document.getElementById('plato-tape');
  if (!viewport || !tape) return;

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var corpus = '';
  var read = 0;                 // read cursor into the (looping) corpus
  var textNode = document.createTextNode('');
  var cursor = document.createElement('span');
  cursor.className = 'plato-cursor';
  cursor.textContent = ' ';

  var charsPerSec = 42;         // typing speed
  var scrollEase  = 0.07;       // how quickly the view glides toward the newest line
  var maxChars    = 9000;       // keep the DOM light: trim once the buffer grows past this
  var trimTo      = 6000;

  var acc = 0, last = 0, paused = false, running = false;

  function nextChar() {
    if (!corpus.length) return '';
    var ch = corpus[read];
    read = (read + 1) % corpus.length;   // loop the corpus seamlessly
    return ch;
  }

  function trimIfNeeded() {
    if (textNode.data.length <= maxChars) return;
    var drop = textNode.data.length - trimTo;
    // only cut at a newline so we never slice a word mid-character
    var nl = textNode.data.indexOf('\n', drop);
    drop = nl === -1 ? drop : nl + 1;
    var before = tape.scrollHeight;
    textNode.data = textNode.data.slice(drop);
    var after = tape.scrollHeight;
    // compensate the scroll position so the visible text doesn't jump
    viewport.scrollTop -= (before - after);
  }

  function frame(now) {
    if (!running) return;
    if (paused) { last = now; requestAnimationFrame(frame); return; }
    var dt = last ? (now - last) / 1000 : 0;
    last = now;

    // reveal characters at a steady rate (typewriter)
    acc += dt * charsPerSec;
    var n = acc | 0;
    if (n > 0) {
      acc -= n;
      var chunk = '';
      for (var i = 0; i < n; i++) chunk += nextChar();
      textNode.data += chunk;
      trimIfNeeded();
    }

    // glide the viewport toward the bottom (newest text), so older text rises and fades out the top
    var target = tape.scrollHeight - viewport.clientHeight;
    viewport.scrollTop += (target - viewport.scrollTop) * scrollEase;

    requestAnimationFrame(frame);
  }

  function start() {
    tape.textContent = '';
    tape.appendChild(textNode);
    tape.appendChild(cursor);

    if (reduceMotion) {
      // no animation: just show a static passage
      textNode.data = corpus.slice(0, 1600);
      cursor.remove();
      return;
    }
    running = true;
    last = 0;
    requestAnimationFrame(frame);

    // don't burn CPU while off-screen or on a hidden tab
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        paused = !entries[0].isIntersecting || document.hidden;
      }, { threshold: 0 }).observe(viewport);
    }
    document.addEventListener('visibilitychange', function () {
      paused = document.hidden;
    });
  }

  fetch(CORPUS_URL)
    .then(function (r) { return r.text(); })
    .then(function (t) {
      corpus = t.replace(/\r\n/g, '\n').trim();
      if (!corpus) throw new Error('empty corpus');
      start();
    })
    .catch(function () {
      tape.textContent = 'The model is resting. (Could not load the text sample.)';
    });
})();
</script>

## The transformer

**Why it exists.** Before 2017 the strongest sequence models were recurrent — RNNs and LSTMs — which read a sentence one token at a time, folding everything seen so far into a single fixed-size hidden state. Two problems followed. Training can't be parallelized along the sequence (step *t* must wait for step *t*−1), and information from a distant token has to survive a long chain of updates to still sway a prediction — the same gradient-health trouble the [training notebook]({{ '/projects/makemore_batchnorm/' | relative_url }}) fought at initialization, now stretched across time. Attention had already been *bolted onto* these models (Bahdanau et al., 2014) so a decoder could look back at any input position directly; ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) took the decisive step of discarding the recurrence and keeping only the attention. The payoff is structural: every position reaches every earlier position in a **single step**, and the whole sequence is processed **in parallel**.

**Self-attention** is the one genuinely new idea beyond makemore. Each token projects its embedding into three vectors — a **query** (*what am I looking for?*), a **key** (*what do I contain?*), and a **value** (*what will I pass on?*). The relevance of token *j* to token *i* is the dot product of *i*'s query with *j*'s key; those scores are divided by $$\sqrt{d_k}$$ to keep them in softmax's sensitive range, softmaxed into weights that sum to one, and used to take a weighted average of the values. Linear layers, dot products, softmax — every piece is already yours from makemore; attention just wires them into a lookup where the tokens themselves decide what to fetch. Two details make it a *language* model: the weights are **masked** so a token attends only backward, never to the future it must predict, and every position is computed at once.

**Heads, blocks, and depth.** One set of query/key/value is a single line of inquiry; six run in parallel (each in a smaller subspace) so the model can track several kinds of relationship at once, then concatenate and mix their outputs. This attention sublayer is paired with a small per-token MLP — attention lets tokens *communicate*, the feed-forward lets each one *think* — and each of the two sublayers is wrapped in a **residual connection** (a gradient shortcut straight through the block) and **layer normalization** (a cousin of the batch norm from [part III]({{ '/projects/makemore_batchnorm/' | relative_url }})). Six such blocks, with token and position embeddings at the bottom and one linear layer + softmax at the top, is the whole model.

**A GPT is half of the original.** The 2017 architecture is an *encoder–decoder*, built to turn one sequence into another (say, English into German). A model that only ever continues a single stream keeps just the **decoder** — and drops even the decoder's cross-attention, the sublayer that existed solely to read from the encoder. What remains is the right-hand column below, mapped one-to-one onto the paper's own schematic:

{::nomarkdown}
<div class="transformer-fig">
<svg id="txfig" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 590" width="100%" role="img" aria-label="The original encoder-decoder Transformer beside the decoder-only platoGPT: a GPT is the decoder, with the encoder and cross-attention removed.">

<style>
  #txfig .bl { text-anchor: middle; dominant-baseline: middle;
        font-family: system-ui,-apple-system,'Segoe UI',Roboto,sans-serif; fill: currentColor; }
  #txfig rect.keep { fill: rgba(37,150,90,0.13); stroke: currentColor; stroke-width: 1.3; }
  #txfig rect.drop { fill: rgba(140,140,150,0.09); stroke: #9aa0a6; stroke-width: 1.2; stroke-dasharray: 5 4; }
  #txfig rect.stackbox { fill: none; stroke: currentColor; stroke-width: 1; stroke-opacity: .45; }
  #txfig rect.dropstack { fill: rgba(140,140,150,0.05); stroke: #9aa0a6; stroke-width: 1; stroke-dasharray: 5 4; }
  #txfig .keept { fill: currentColor; } #txfig .dropt { fill: #9aa0a6; }
  #txfig .sub { fill: currentColor; opacity: .6; }
  #txfig .arrow { stroke: currentColor; stroke-width: 1.4; stroke-opacity: .8; fill: none; }
  #txfig .cross { stroke: #9aa0a6; stroke-width: 1.4; stroke-dasharray: 5 4; fill: none; }
  #txfig .plus { fill: var(--global-bg-color,#fff); stroke: currentColor; stroke-width: 1.3; }
  #txfig .pl { stroke: currentColor; stroke-width: 1.3; }
  #txfig .cap { fill: currentColor; opacity: .75; }
  #txfig .ttl { fill: currentColor; font-weight: 600; }
  #txfig .divider { stroke: currentColor; stroke-opacity: .2; stroke-width: 1; }
</style>
<defs>
  <marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" fill-opacity=".8"/>
  </marker>
</defs>

<text x="255" y="26" class="bl ttl" font-size="14">The Transformer — Vaswani et al. (2017)</text>
<text x="255" y="43" class="bl cap" font-size="10.5">encoder – decoder, built for translation</text>
<text x="150" y="70" class="bl cap" font-size="10.5">encoder</text>
<text x="360" y="70" class="bl cap" font-size="10.5">decoder</text>
<text x="150" y="524" class="bl cap" font-size="10">Inputs</text>
<text x="360" y="520" class="bl cap" font-size="10">Outputs</text>
<text x="360" y="533" class="bl cap" font-size="8.5">(shifted right)</text>
<rect x="62" y="484" width="176" height="26" rx="5" class="drop"/>
<text x="150" y="497" class="bl dropt" font-size="10.5">Input Embedding</text>
<rect x="272" y="484" width="176" height="26" rx="5" class="keep"/>
<text x="360" y="497" class="bl keept" font-size="10.5">Output Embedding</text>
<circle cx="150" cy="462" r="9" class="plus"/>
<line x1="144" y1="462" x2="156" y2="462" class="pl"/>
<line x1="150" y1="456" x2="150" y2="468" class="pl"/>
<circle cx="360" cy="462" r="9" class="plus"/>
<line x1="354" y1="462" x2="366" y2="462" class="pl"/>
<line x1="360" y1="456" x2="360" y2="468" class="pl"/>
<text x="32" y="462" class="bl cap" font-size="9">Positional</text>
<text x="32" y="473" class="bl cap" font-size="9">Encoding</text>
<line x1="150" y1="516" x2="150" y2="510" class="arrow" marker-end="url(#ah)"/>
<line x1="360" y1="512" x2="360" y2="510" class="arrow" marker-end="url(#ah)"/>
<line x1="150" y1="484" x2="150" y2="471" class="arrow" marker-end="url(#ah)"/>
<line x1="360" y1="484" x2="360" y2="471" class="arrow" marker-end="url(#ah)"/>
<rect x="62" y="424" width="176" height="28" rx="5" class="drop"/>
<text x="150" y="438" class="bl dropt" font-size="11.5">Multi-Head Attention</text>
<rect x="62" y="389" width="176" height="28" rx="5" class="drop"/>
<text x="150" y="403" class="bl dropt" font-size="11.5">Add &amp; Norm</text>
<rect x="62" y="354" width="176" height="28" rx="5" class="drop"/>
<text x="150" y="368" class="bl dropt" font-size="11.5">Feed Forward</text>
<rect x="62" y="319" width="176" height="28" rx="5" class="drop"/>
<text x="150" y="333" class="bl dropt" font-size="11.5">Add &amp; Norm</text>
<rect x="50" y="311" width="200" height="145" rx="7" class="dropstack"/>
<text x="262" y="386" class="bl cap" font-size="12">N×</text>
<line x1="150" y1="453" x2="150" y2="452" class="arrow" marker-end="url(#ah)"/>
<rect x="272" y="424" width="176" height="28" rx="5" class="keep"/>
<text x="360" y="438" class="bl keept" font-size="11.5">Masked Multi-Head Attn</text>
<rect x="272" y="389" width="176" height="28" rx="5" class="keep"/>
<text x="360" y="403" class="bl keept" font-size="11.5">Add &amp; Norm</text>
<rect x="272" y="354" width="176" height="28" rx="5" class="drop"/>
<text x="360" y="368" class="bl dropt" font-size="11.5">Multi-Head Attention</text>
<rect x="272" y="319" width="176" height="28" rx="5" class="drop"/>
<text x="360" y="333" class="bl dropt" font-size="11.5">Add &amp; Norm</text>
<rect x="272" y="284" width="176" height="28" rx="5" class="keep"/>
<text x="360" y="298" class="bl keept" font-size="11.5">Feed Forward</text>
<rect x="272" y="249" width="176" height="28" rx="5" class="keep"/>
<text x="360" y="263" class="bl keept" font-size="11.5">Add &amp; Norm</text>
<rect x="260" y="241" width="200" height="215" rx="7" class="stackbox"/>
<text x="472" y="350" class="bl cap" font-size="12">N×</text>
<line x1="360" y1="453" x2="360" y2="452" class="arrow" marker-end="url(#ah)"/>
<path d="M 250 337 C 330 337, 210 368.0, 260 368.0" class="cross" marker-end="url(#ah)"/>
<rect x="272" y="215" width="176" height="26" rx="5" class="keep"/>
<text x="360" y="228" class="bl keept" font-size="10.5">Linear</text>
<line x1="360" y1="249" x2="360" y2="241" class="arrow" marker-end="url(#ah)"/>
<rect x="272" y="179" width="176" height="26" rx="5" class="keep"/>
<text x="360" y="192" class="bl keept" font-size="10.5">Softmax</text>
<line x1="360" y1="215" x2="360" y2="205" class="arrow" marker-end="url(#ah)"/>
<text x="360" y="165" class="bl cap" font-size="9.5">Output Probabilities</text>
<line x1="510" y1="60" x2="510" y2="560" class="divider"/>
<text x="730" y="26" class="bl ttl" font-size="14">platoGPT — decoder-only</text>
<text x="730" y="43" class="bl cap" font-size="10.5">the same decoder, minus the greyed parts</text>
<text x="730" y="524" class="bl cap" font-size="10">characters</text>
<rect x="630" y="484" width="200" height="26" rx="5" class="keep"/>
<text x="730" y="497" class="bl keept" font-size="10.5">Token Embedding</text>
<circle cx="730" cy="462" r="9" class="plus"/>
<line x1="724" y1="462" x2="736" y2="462" class="pl"/>
<line x1="730" y1="456" x2="730" y2="468" class="pl"/>
<text x="600" y="462" class="bl cap" font-size="9">Position</text>
<text x="600" y="473" class="bl cap" font-size="9">Embedding</text>
<line x1="730" y1="516" x2="730" y2="510" class="arrow" marker-end="url(#ah)"/>
<line x1="730" y1="484" x2="730" y2="471" class="arrow" marker-end="url(#ah)"/>
<rect x="630" y="424" width="200" height="28" rx="5" class="keep"/>
<text x="730" y="438" class="bl keept" font-size="11.5">Masked Multi-Head Attn</text>
<rect x="630" y="389" width="200" height="28" rx="5" class="keep"/>
<text x="730" y="403" class="bl keept" font-size="11.5">Add &amp; Norm</text>
<rect x="630" y="354" width="200" height="28" rx="5" class="keep"/>
<text x="730" y="368" class="bl keept" font-size="11.5">Feed Forward</text>
<rect x="630" y="319" width="200" height="28" rx="5" class="keep"/>
<text x="730" y="333" class="bl keept" font-size="11.5">Add &amp; Norm</text>
<rect x="618" y="311" width="224" height="145" rx="7" class="stackbox"/>
<text x="854" y="386" class="bl cap" font-size="12">6×</text>
<line x1="730" y1="453" x2="730" y2="452" class="arrow" marker-end="url(#ah)"/>
<rect x="630" y="285" width="200" height="26" rx="5" class="keep"/>
<text x="730" y="298" class="bl keept" font-size="10.5">Linear  (lm_head)</text>
<line x1="730" y1="319" x2="730" y2="311" class="arrow" marker-end="url(#ah)"/>
<rect x="630" y="249" width="200" height="26" rx="5" class="keep"/>
<text x="730" y="262" class="bl keept" font-size="10.5">Softmax</text>
<line x1="730" y1="285" x2="730" y2="275" class="arrow" marker-end="url(#ah)"/>
<text x="730" y="235" class="bl cap" font-size="9.5">next-character probabilities</text>
<rect x="30" y="561" width="16" height="12" rx="3" class="keep"/>
<text x="96" y="570" class="bl cap" font-size="10">kept in a GPT</text>
<rect x="180" y="561" width="16" height="12" rx="3" class="drop"/>
<text x="268" y="570" class="bl cap" font-size="10">dropped (no encoder)</text>
</svg>
<p class="transformer-fig-cap">Left, the Transformer as Vaswani&nbsp;et&nbsp;al. drew it (redrawn); right, platoGPT. Grey out the encoder and the decoder&rsquo;s cross-attention sublayer and the two columns collapse into one &mdash; a decoder-only transformer. (One later refinement this model adopts: layer&nbsp;norm is applied <em>before</em> each sublayer, not after.)</p>
</div>
{:/nomarkdown}

Trained on ~1.2 MB of Plato for 5,000 steps, the loss falls from a random-guess $$\ln(78)\approx 4.36$$ down to about $$1.0$$ nat per character — the point where the samples stop being noise and start sounding like a philosopher who has lost the thread. The stream above isn't computed live in your browser (running a 10.8M-parameter transformer forever on every visitor's phone would be unkind) — I generated a large corpus offline and the page loops it. Everything you read is the model's own output; the notebook below builds and trains it from first principles.

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/platoGPT.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/platoGPT.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}
<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

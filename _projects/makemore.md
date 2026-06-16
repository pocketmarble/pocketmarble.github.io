---
layout: page
title: building a bigram language model from scratch
description: Teaching a tiny model to dream up new names, one character at a time
img: assets/img/makemore.png
importance: 2
category: work
---

This project is my hands-on implementation of [makemore](https://github.com/karpathy/makemore), following Andrej Karpathy's [walkthrough](https://www.youtube.com/watch?v=PaCmpygFfXo). The goal is to build a character-level language model that, trained on a list of 32,033 real US names, learns to generate new ones that *sound* like names. I start from raw bigram frequency counts and then rebuild the exact same model as a single-layer neural network trained with gradient descent — showing that the two approaches converge to the same result.

The model below is the **bigram model** from the notebook, running entirely in your browser. It looks at the current character and samples the next one from the probability distribution it learned from real names. Click the button to dream up a few:

<div class="row justify-content-sm-center mt-3 mb-4">
  <div class="col-sm-10 text-center">
    <button id="makemore-btn" class="btn btn-primary">✨ generate ..."names"</button>
    <pre id="makemore-output" style="margin-top: 1rem; min-height: 7.5rem; font-size: 1.1rem; line-height: 1.5;">click the button to dream up some names…</pre>
  </div>
</div>

<script>
(function () {
  // bigram counts learned from names.txt: N[i][j] = times character j followed character i.
  // index 0 is the start/end token '.'; indices 1-26 are 'a'-'z'.
  const N = [[0,4410,1306,1542,1690,1531,417,669,874,591,2422,2963,1572,2538,1146,394,515,92,1639,2055,1308,78,376,307,134,535,929],[6640,556,541,470,1042,692,134,168,2332,1650,175,568,2528,1634,5438,63,82,60,3264,1118,687,381,834,161,182,2050,435],[114,321,38,1,65,655,0,0,41,217,1,0,103,0,4,105,0,0,842,8,2,45,0,0,0,83,0],[97,815,0,42,1,551,0,2,664,271,3,316,116,0,0,380,1,11,76,5,35,35,0,0,3,104,4],[516,1303,1,3,149,1283,5,25,118,674,9,3,60,30,31,378,0,1,424,29,4,92,17,23,0,317,1],[3983,679,121,153,384,1271,82,125,152,818,55,178,3248,769,2675,269,83,14,1958,861,580,69,463,50,132,1070,181],[80,242,0,0,0,123,44,1,1,160,0,2,20,0,4,60,0,0,114,6,18,10,0,4,0,14,2],[108,330,3,0,19,334,1,25,360,190,3,0,32,6,27,83,0,0,201,30,31,85,1,26,0,31,1],[2409,2244,8,2,24,674,2,2,1,729,9,29,185,117,138,287,1,1,204,31,71,166,39,10,0,213,20],[2489,2445,110,509,440,1653,101,428,95,82,76,445,1345,427,2126,588,53,52,849,1316,541,109,269,8,89,779,277],[71,1473,1,4,4,440,0,0,45,119,2,2,9,5,2,479,1,0,11,7,2,202,5,6,0,10,0],[363,1731,2,2,2,895,1,0,307,509,2,20,139,9,26,344,0,0,109,95,17,50,2,34,0,379,2],[1314,2623,52,25,138,2921,22,6,19,2480,6,24,1345,60,14,692,15,3,18,94,77,324,72,16,0,1588,10],[516,2590,112,51,24,818,1,0,5,1256,7,1,5,168,20,452,38,0,97,35,4,139,3,2,0,287,11],[6763,2977,8,213,704,1359,11,273,26,1725,44,58,195,19,1906,496,5,2,44,278,443,96,55,11,6,465,145],[855,149,140,114,190,132,34,44,171,69,16,68,619,261,2411,115,95,3,1059,504,118,275,176,114,45,103,54],[33,209,2,1,0,197,1,0,204,61,1,1,16,1,1,59,39,0,151,16,17,4,0,0,0,12,0],[28,13,0,0,0,1,0,0,0,13,0,0,1,2,0,2,0,0,1,2,0,206,0,3,0,0,0],[1377,2356,41,99,187,1697,9,76,121,3033,25,90,413,162,140,869,14,16,425,190,208,252,80,21,3,773,23],[1169,1201,21,60,9,884,2,2,1285,684,2,82,279,90,24,531,51,1,55,461,765,185,14,24,0,215,10],[483,1027,1,17,0,716,2,2,647,532,3,0,134,4,22,667,0,0,352,35,374,78,15,11,2,341,105],[155,163,103,103,136,169,19,47,58,121,14,93,301,154,275,10,16,10,414,474,82,3,37,86,34,13,45],[88,642,1,0,1,568,0,0,1,911,0,3,14,0,8,153,0,0,48,0,0,7,7,0,0,121,0],[51,280,1,0,8,149,2,1,23,148,0,6,13,2,58,36,0,0,22,20,8,25,0,2,0,73,1],[164,103,1,4,5,36,3,0,1,102,0,0,39,1,1,41,0,0,0,31,70,5,0,3,38,30,19],[2007,2143,27,115,272,301,12,30,22,192,23,86,1104,148,1826,271,15,6,291,401,104,141,106,4,28,23,78],[160,860,4,2,2,373,0,1,43,364,2,2,123,35,4,110,2,0,32,4,4,73,2,3,1,147,45]];
  const itoc = '.abcdefghijklmnopqrstuvwxyz';

  // sample an index from a row of counts, weighted by frequency (torch.multinomial in the notebook)
  function sample(row) {
    let total = 0;
    for (const c of row) total += c;
    let r = Math.random() * total;
    for (let j = 0; j < row.length; j++) {
      r -= row[j];
      if (r < 0) return j;
    }
    return row.length - 1;
  }

  function makeName() {
    let name = '';
    let idx = 0;
    while (true) {
      idx = sample(N[idx]);
      if (idx === 0) break;          // sampled the end token
      name += itoc[idx];
      if (name.length > 20) break;   // safety cap
    }
    return name;
  }

  const btn = document.getElementById('makemore-btn');
  const out = document.getElementById('makemore-output');
  if (btn && out) {
    btn.addEventListener('click', function () {
      const names = [];
      for (let i = 0; i < 5; i++) names.push(makeName());
      out.textContent = names.join('\n');
    });
  }
})();
</script>

The notebook below walks through the whole build, each piece from first principles, implemented in pytorch:

- **Bigram counts** — tallying every adjacent pair of characters across all names into a [27,27] tensor
- **From counts to probabilities** — normalizing each row so it sums to 1, then sampling with `torch.multinomial` to generate names
- **Loss** — the average negative log-likelihood, a single number measuring how well the model predicts the real names
- **The neural-network version** — one-hot encoding the inputs, a single 27×27 weight matrix, softmax, and gradient descent
- **The punchline** — the trained weight matrix converges to (the log of) the count table. The two models are mathematically the same, but the neural-network framing scales to far richer models

The 27×27 bigram frequency table the model learns, visualized — brighter cells are more common transitions:

<div class="row justify-content-sm-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/makemore.png" title="bigram frequency table learned from 32,033 names" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/makemore.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/makemore.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}
<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

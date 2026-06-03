---
layout: page
title: micrograd — building a neural net from scratch
description: A from-scratch tour of the autograd engine and gradient descent behind neural networks, built by following Andrej Karpathy's micrograd video
img: assets/img/micrograd.png
importance: 1
category: work
---

This project is my hands-on walkthrough of [micrograd](https://github.com/karpathy/micrograd), Andrej Karpathy's tiny autograd engine, built while following his [introductory machine learning video](https://www.youtube.com/watch?v=VMj-3S1tku0). Starting from a single scalar `Value` that tracks its own gradient, I work up to neurons, layers, and a small multi-layer perceptron — then train it end to end with gradient descent.

The notebook below demonstrates the core ideas, each implemented from first principles with no deep-learning frameworks:

- **The `Value` object** — a scalar that records the operations performed on it, forming a computation graph.
- **Forward pass** — evaluating an expression while building that graph.
- **Backward pass** — reverse-mode automatic differentiation (backpropagation) to compute every gradient via the chain rule.
- **Neurons, layers, and MLPs** — composing `Value`s into the building blocks of a network.
- **Gradient descent** — nudging the parameters down the loss gradient to actually train the model.

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/micrograd_sandbox.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/micrograd_sandbox.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}
<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

---
layout: page
title: building a neural network from scratch
description: A from-scratch tour of the fundamentals of machine learning
img: assets/img/decision_boundary.gif
importance: 1
category: work
---

This project is my hands-on implementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), a bare-bones engine styled after PyTorch which handles the core functionality of ML models, following his [walkthrough](https://www.youtube.com/watch?v=VMj-3S1tku0). Starting from a single scalar object `Value` that contains data and keeps track of its own gradient and connections, I build up to `Neuron`, `Layer`, and finally `MLP`, a multi-layer perceptron. Forward and reverse passes are manually computed for `Neuron` before being implemented as class methods, eventually composing into a gradient descent loop to train a mock network as a binary classifier on mock data.

The notebook below demonstrates the core ideas, each implemented from first principles:

- **The `Value` object** — a scalar that records the operations performed on it, forming a computation graph.
- **The `Neuron`, `Layer`, and `MLP` objects** - `MLP`s are built from `Layer`s, groups of `Neuron`s built from `Value`s 
- **Forward pass** — computes the output of an `MLP`.
- **Backward pass** — computes the gradient of each `Value` in an `MLP` as the depending on its arrangement within the network.
- **Loss** - mean square error of MLP output and actual output for a given training input. Used to compute gradient of network
- **Gradient descent** — nudging the parameters down the loss gradient to train the network.

The result: as the network trains, its decision boundary sharpens to separate the two classes.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/decision_boundary.gif" title="decision boundary evolving during training" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/micrograd_sandbox.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/micrograd_sandbox.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}
<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

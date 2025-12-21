---
layout: post
title: Multilayer Perceptron From Scratch — and Why CNNs Were Needed
date: 2025-01-12
---

## Motivation

Before convolutions, attention, or transformers, the **Multilayer Perceptron (MLP)** was the primary way to build trainable nonlinear models.

To understand why CNNs exist, we must first understand **what MLPs can do well — and what they fundamentally cannot**.

This post builds an MLP from first principles and uses it to expose those limits.

---

## The MLP: Structure

An MLP is a composition of affine transformations and nonlinearities.

For a single hidden-layer MLP:

\[
\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
\]
\[
\hat{\mathbf{y}} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2
\]

Where:
- \(\mathbf{x} \in \mathbb{R}^d\) is the input
- \(\sigma\) is a nonlinear activation (e.g. ReLU)
- Parameters are learned via gradient descent

This structure is **fully connected**.

---

## From-Scratch Implementation (Minimal)

Below is a minimal MLP implementation using only NumPy-style operations.

```python
import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.h = self.relu(self.z1)
        self.out = self.h @ self.W2 + self.b2
        return self.out
```

This is the **purest form** of an MLP:

* no frameworks
* no abstractions
* no optimizers yet

---

## Applying an MLP to Images

Suppose we take a grayscale image of size (28 \times 28).

To feed it into an MLP, we must **flatten it**:

[
\mathbf{x} \in \mathbb{R}^{784}
]

This immediately introduces problems.

---

## Core Limitation 1: Loss of Spatial Structure

Flattening destroys spatial locality.

Pixels that were neighbors in 2D space become arbitrarily distant in the input vector.

As a result:

* the model has **no notion of edges**
* no understanding of shapes
* no translation awareness

The MLP treats:

* a pixel in the corner
* and a pixel in the center
  as equally unrelated unless it *learns that from scratch*.

---

## Core Limitation 2: Parameter Explosion

For a modest hidden layer of size 512:

[
784 \times 512 \approx 400{,}000 \text{ parameters}
]

This is:

* expensive
* data-hungry
* prone to overfitting

And this is **before** adding depth.

---

## Core Limitation 3: No Inductive Bias

MLPs assume **every input dimension is independent**.

Images violate this assumption:

* nearby pixels are correlated
* patterns repeat across locations

The MLP has **no built-in bias** to exploit this.

---

## Why CNNs Were Inevitable

CNNs were not invented to be clever.

They were invented because MLPs:

1. discard spatial structure
2. scale poorly with image size
3. lack translation-aware inductive bias

Convolutions introduce:

* local connectivity
* weight sharing
* spatial hierarchy

All three directly address MLP failures.

---

## Takeaway

MLPs are:

* powerful function approximators
* foundational to modern deep learning

But for structured inputs like images, **they are the wrong tool**.

CNNs are not an upgrade.
They are a **correction**.

---

## What Comes Next

The next step is to:

* formalize spatial locality
* introduce receptive fields
* show convolution as constrained linear layers

That transition will be made explicit in the next post.

---

### Code Reference

The full implementation and experiments live in:

[https://github.com/ScratchMind/ScratchVision](https://github.com/ScratchMind/ScratchVision)
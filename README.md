# Biologically Plausible Learning for Medical AI

This repository accompanies our study, [Investigating the Application of Feedback Alignment to Medical AI]().

This work explores biologically plausible alternatives to backpropagation (BP), specifically Direct Feedback Alignment (DFA), tested in the context of medical machine learning on the benchmark tabular medical data classification, Wisconsin Breast Cancer dataset.

**Objectives:**

1. Evaluate the performance of backpropagation vs. feedback-alignment-based learning in a feedforward MLP.


2. Investigate generalisation / robustness, convergence behaviour, and accuracy.


3. Explore learning dynamics and feature attribution of biologically inspired learning algorithms in medical AI.

---

## Core Principle

<br>

**1. Standard Backpropagation (BP):**

Let a feedforward network have $L$ layers, activations, $a^l = \phi(W^{l-1}a^{l-1} + b^l)$ , and loss, $\mathcal{L}(y, \hat{y})$. 

The gradient update is:

$$
\delta^L = \nabla_{\hat{y}} \mathcal{L} \odot \phi'(z^L), \quad
\delta^l = \big(W^{l+1}\big)^\top \delta^{l+1} \odot \phi'(z^l)
$$
$$
W^l \gets W^l - \eta \, \delta^l (a^{l-1})^\top
$$

Error signals are propagated back through the transpose of forward weights, ensuring exact gradient flow and error signals.

<br>

**2. Direct Feedback Alignment (DFA):**

Introduces biologically plausible credit assignment by bypassing symmetric weight transport.

Error at the output layer is `directly` projected backwards using fixed random feedback matrices, $B^l$, instead of $(W^{l+1})^T$ :

$$
\delta^l = B^l \delta^L \odot \phi'(z^l)
$$

This breaks symmetry by giving random singal but is observed to still lead to learning due to gradual alignment between $B^l$ and $W^l$.

---

| Feature                 | Backpropagation            | Feedback Alignment            |
| ----------------------- | -------------------------- | ----------------------------- |
| Gradient flow           | Exact                      | Approximate, random-projected |
| Weight symmetry         | Required (transpose usage) | Not required                  |
| Biological plausibility | Low; brain doesn't know forward weights                        | Higher                        |
| Convergence speed       | Typically faster           | Slower, depends on alignment  |

---

### Empirical Observations

1. Backpropagation achieved higher and more stable accuracy, reflecting its exact gradient optimisation.


2. DFA converged slower, taking slightly more epochs, and exhibited lower **but reasonably simiar** peak performance, consistent with the literature on feedback misalignment in small MLPs.


3.  DFA remains feasible but is less data-efficient, aligning with prior claims that alignment improves with wider or deeper networks; **needs information-rich representations**.

In resource-constrained or neuromorphic contexts, DFA could offer a path to on-chip learning without transpose-weight transport.

---
**Repository Structure:**

1. `FF_DFA_learning.ipynb` – Core training and learning dynamics comparison

2. `FF_DFA_tests.ipynb` – Metric experiments and tests across seeds, widths, and imbalance handling.

3. `results` - Contains expected behaviour as per tests

The code and the notebook are self-explainatory with interpretations and empirical findings.


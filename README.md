# Biologically Plausible Learning in Neural Networks

This repository accompanies our study, [Limits and Alignment of Biologically Plausible Learning in Clinically Relevant Neural Networks](#).

This work explores biologically plausible alternatives to backpropagation (BP), specifically Direct Feedback Alignment (DFA), on the Wisconsin Diagnostic Breast Cancer (WDBC) and McKinsey stroke datasets, and on PatchCamelyon (PCam) histopathology images. The comparisons cover six multilayer perceptron (MLP) architectures, class-imbalance handling, and compact CNNs with residual and dense connectivity. Further stroke experiments examine feedback initialization and the relation between error reduction and gradient alignment.

**Objectives:**

1. Evaluate the performance of backpropagation vs. direct feedback alignment in feedforward MLPs and convolutional networks for medical classification.


2. Investigate generalisation / robustness, learning dynamics, discrimination, and calibration under different architectures, imbalance strategies, and feedback initializations.


3. Explore learning dynamics and feature attribution of biologically plausible learning algorithms in medical AI.

---

## Core Principle

<br>

**1. Standard Backpropagation (BP):**

For one input $x$, let a feedforward network have $L$ layers, input activation $a^0=x$, preactivations $z^l=W^l a^{l-1}+b^l$, elementwise activations $a^l=\phi^l(z^l)$, and loss $\mathcal{L}(a^L,y)$. The output activation is the identity, so $a^L$ contains logits. Write $\delta^l=\nabla_{z^l}\mathcal{L}$ and let $\eta$ denote the learning rate.

The gradient update is:

$$
\delta^L = \nabla_{a^L} \mathcal{L} \odot (\phi^L)'(z^L), \quad
\delta^l = \big(W^{l+1}\big)^\top \delta^{l+1} \odot (\phi^l)'(z^l), \quad l<L
$$
$$
W^l \gets W^l - \eta \, \delta^l (a^{l-1})^\top
$$

Error signals are propagated back through the transpose of forward weights, giving the chain-rule gradient. The bias gradient is $\delta^l$. The update above writes a gradient-descent step; the main MLP and CNN comparisons apply their gradients through Adam and AdamW, respectively.

<br>

**2. Direct Feedback Alignment (DFA):** Based on [Lillicrap et al. (2016)](https://doi.org/10.1038/ncomms13276) and [Nøkland (2016)](https://papers.nips.cc/paper/2016/hash/d490d7b4576290fa60eb31b5fc917ad1-Abstract.html).

DFA extends feedback alignment by sending the output error directly to hidden layers, relaxing the requirement for transpose-weight transport.

For a hidden layer with $n_l$ units and an output layer with $n_L$ units, the fixed random feedback matrix $B^l\in\mathbb{R}^{n_l\times n_L}$ projects the output error $\delta^L$:

$$
\delta^l_{\mathrm{DFA}} = B^l \delta^L \odot (\phi^l)'(z^l), \quad l<L
$$

The MLP output layer uses BP. In the CNNs, projections enter composite feature-map boundaries; operations within each unit and the classifier retain local and exact BP, respectively. Fixed feedback can provide useful update directions, but their alignment with BP gradients need not improve monotonically.

---

| Feature                 | Backpropagation            | Direct Feedback Alignment     |
| ----------------------- | -------------------------- | ----------------------------- |
| Gradient flow           | Chain-rule gradient        | Fixed output-error projections at hidden boundaries |
| Weight symmetry         | Uses transposed forward weights | Not required at projected boundaries |
| Biological plausibility | Requires weight transport and nonlocal error signals | Relaxes weight transport; still requires output-error signals |
| Convergence speed       | Faster loss reduction in the reported comparisons | Slower loss reduction in the reported comparisons |

---

### Empirical Observations

1. BP generally achieved higher and more stable predictive performance. DFA remained competitive on WDBC, while the gap increased under severe stroke imbalance. Class weighting was more favorable to DFA than synthetic oversampling in the reported comparisons. SHAP attributions were broadly similar between the MLPs trained with BP and DFA, although calibration and predictive performance differed.


2. DFA showed slower loss reduction and nonmonotonic gradient alignment. Controlled stroke experiments and continuous linear references showed that decreasing projected error can coexist with gradient-angle reversals. The restricted linear convergence result does not establish convergence for the nonlinear, multilayer, Adam-trained classifiers.


3. The 144-run stroke initialization study compared Gaussian, Uniform, Orthogonal, and Sparse feedback across six architectures, two imbalance strategies, and three seeds. Sparse feedback had lower mean ROC-AUC and higher mean Brier/ECE than Gaussian under both strategies, with differences significant after Holm correction. No Uniform or Orthogonal comparison rejected after correction; this does not establish equivalence between distributions.

4. BP outperformed DFA in both PCam convolutional families, with limited BP–DFA Grad-CAM agreement. At 32 epochs, compact DenseNet-BC achieved 88.44% accuracy with BP and 76.61% with DFA; DFA had higher Brier score but lower ECE. DFA reached 76.44% at epoch 48. Epochs 33–48 used BF16 after a nonfinite FP16 loss, so this continuation changes both duration and precision. These single-seed comparisons describe the implemented models; they do not isolate connectivity as the cause or establish depth/width scalability.

---

**Repository Structure:**

1. `wisconsin_ff/` and `mckinsey_ff/` – Tabular studies, each with `FF_DFA_learning.ipynb` for learning dynamics and attribution, `FF_DFA_tests.ipynb` for comparisons across seeds, widths, and imbalance strategies, and corresponding `learning_results/` and `test_results/` folders.

2. `pcam_cnn/` – The original ResNet-style PCam study in `CNN_FA.ipynb`, with its recorded outputs and local PCam data in `pcam_data/`.

3. `reviewer7.ipynb` and `reviewer7_results/` – Executed stroke initialization, controlled alignment, and PCam DenseNet studies. The results folder contains the stroke summaries, linear references, models, and plots; its `reviewer7_image_results/` subfolder contains the DenseNet checkpoints, epoch histories, predictions, per-patch attribution scores, metric summaries, and figures.

The notebooks contain the executed outputs, interpretations, and experimental settings. `requirements.txt` lists the project dependencies. Run the additional studies in `reviewer7.ipynb` with `code/` as the working directory; its stroke and image sections can be run independently from their imports. Point `IMAGE_RESULT_DIR` to `reviewer7_results/reviewer7_image_results/` when reanalysing the saved DenseNet run.


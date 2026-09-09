# Tower Research Capital — ML Engineer Interview Prep

## Table of Contents
1. [Dijkstra with XOR Distance Metric](#1-dijkstra-with-xor-distance-metric)
2. [Adapting Tokenization for Time-Series](#2-adapting-tokenization-for-time-series)
3. [Principal Component Analysis — Theory, Explanation & Code](#3-principal-component-analysis)
4. [Normalization, Dropout & Weight Initialization](#4-normalization-dropout--weight-initialization)
5. [DeepSpeed ZeRO Stages](#5-deepspeed-zero-stages)
6. [Python Internals](#6-python-internals)
7. [Bias-Variance Tradeoff, Hypothesis Testing & MLE vs MAP](#7-bias-variance-tradeoff-hypothesis-testing--mle-vs-map)

---

## 1. Dijkstra with XOR Distance Metric

### The Question

> "Assume you apply Dijkstra's shortest path algorithm on a graph, where the distance metric uses bitwise XOR instead of adding distances. Will you get the correct answer?"

### The Answer: **No.**

### Why Dijkstra Works in the Normal Case

Dijkstra's algorithm relies on a **greedy invariant**: once a node is popped from the priority queue (i.e., "finalized"), the shortest distance to it has already been found. This invariant holds **only** when:

1. **Edge weights are non-negative** (w(u, v) ≥ 0)
2. **The path cost function is monotonically non-decreasing** — extending a path can never reduce its cost. Formally: d(s, u) ≤ d(s, u) + w(u, v) for all edges (u, v).

Property 2 is what matters here. With addition over non-negative weights, if you've found a path of cost 7 to node `u`, then any extension through `u` to another node `v` costs at least 7. So you can safely finalize `u` — no future path through unfinalized nodes could beat 7.

### Why XOR Breaks This

XOR is **not monotonically non-decreasing**. Extending a path with XOR can *decrease* the accumulated cost.

**Concrete Counterexample:**

```
Graph:  s ---5--- A ---2--- t
        s --------6-------- t

Edge weights: s→A = 5, A→t = 2, s→t = 6
Distance metric: XOR (not addition)
```

- Path s → t directly: cost = 6
- Path s → A → t: cost = 5 XOR 2 = 7

Dijkstra would first pop `s` (cost 0), then explore neighbors:
- A gets tentative cost 5
- t gets tentative cost 6

Pop `A` (cost 5), explore A→t:
- New cost to t via A = 5 XOR 2 = 7, which is > 6, so no update.
- Dijkstra returns d(s,t) = 6. ✓ here, but let's change the example:

```
Graph:  s ---6--- A ---3--- t
        s --------4-------- t
```

- Path s → t directly: cost = 4
- Path s → A → t: cost = 6 XOR 3 = 5

Dijkstra pops `s`, sets A=6, t=4. Pops `t` (cost 4), finalizes it.
But the actual XOR-path s → A → t has cost 5 > 4, so Dijkstra happened to get it right.

Now consider:

```
Graph:  s ---7--- A ---7--- B ---1--- t
        s ---------------5----------- t
```

- Direct path s → t: cost = 5
- Path s → A → B → t: cost = 7 XOR 7 XOR 1 = 0 XOR 1 = 1

Dijkstra pops s, sets A=7, t=5. Pops t (cost 5), finalizes it.
**But the true minimum XOR-cost path is s→A→B→t = 1.**

The problem is that 7 XOR 7 = 0 — XOR can "cancel out" previous costs, making a longer path cheaper. Dijkstra never discovers this because it finalized `t` too early.

### The Core Insight

Dijkstra requires the **optimal substructure with monotonicity**: if the shortest path from s to t goes through u, then the sub-path s→u must also be shortest. With XOR, this breaks because a "more expensive" intermediate path can cancel out later, yielding a globally cheaper result. The path cost function must satisfy the **triangle inequality** and **monotonicity** for Dijkstra's greedy approach to work.

### What Would Work Instead?

For XOR-distance, you'd need algorithms that explore all paths — Bellman-Ford style relaxation over all edges (though it may not converge either since XOR can create "negative-cycle"-like behavior), or a modified BFS/DFS approach. In practice, XOR-based shortest paths arise in linear algebra over GF(2), where you'd use Gaussian elimination on a basis of path XOR values.

---

## 2. Adapting Tokenization for Time-Series

### Context: What You Did at Krutrim

At Krutrim, you built a multilingual tokenizer for 22 Indic languages, improving fertility scores (tokens per word) by 39.5% over LLaMA4 and 18% over Sutra. The core idea: a tokenizer that understands the structure of the input domain produces fewer tokens → faster inference.

### The Interviewer's Follow-Up: "How would you adapt this for time-series tokenization?"

This question tests whether you can transfer the conceptual framework of tokenization (segmenting raw input into meaningful discrete units) from text to a fundamentally different modality.

### Why Tokenization Matters for Time-Series

Raw time-series data is continuous and high-frequency. Feeding raw values directly to a transformer is:
- **Expensive**: a 1-second signal at 1kHz = 1000 tokens, attention is O(n²)
- **Noisy**: individual ticks carry little semantic meaning
- **Missing structure**: the model has to learn segmentation from scratch

Tokenization = finding a good **vocabulary of patterns** that compresses the signal while preserving information. Same goal as text tokenization: reduce sequence length, improve fertility (information per token), capture domain structure.

### Approach 1: Patch-Based Tokenization (PatchTST Style)

The most direct analogy to subword tokenization.

**How it works:**
- Divide the time-series into fixed-length, non-overlapping (or overlapping) patches of length P
- Each patch of P consecutive values becomes one "token"
- A linear projection maps each patch (R^P) → (R^d) into the model's embedding space

**Analogy to text:** A patch is like a "word" — a meaningful chunk. Patch length P is analogous to choosing vocabulary granularity. Shorter patches = character-level (fine-grained but long sequences). Longer patches = word-level (compact but may miss detail).

**Fertility connection:** If your original series has length L, you go from L tokens to L/P tokens. Fertility improvement = P×. Just like improving Indic tokenizer fertility meant fewer tokens per word → throughput gains, larger patches mean fewer tokens per time window → throughput gains. The tradeoff: too-large patches lose temporal resolution.

### Approach 2: Learned Discrete Tokenization (VQ-VAE Style)

Closer to BPE in spirit — learn a **codebook** of prototypical patterns.

**How it works:**
1. Train a VQ-VAE (Vector Quantized Variational Autoencoder) on your time-series data
2. The encoder maps windowed segments to a continuous latent space
3. A **codebook** of K discrete vectors quantizes each latent to its nearest codebook entry
4. Each time-series window is now represented by a codebook index (a "token")

**Analogy to BPE:** BPE builds a vocabulary by iteratively merging frequent byte-pairs. VQ-VAE builds a vocabulary by clustering frequent patterns in latent space. Both produce a fixed-size vocabulary that covers the data distribution. The codebook size K is like the vocabulary size — 256 tokens (byte-level) vs 32K tokens (BPE) vs 1024 codebook entries.

**Financial time-series patterns the codebook might learn:**
- Uptrend with increasing volume
- Mean-reversion bounce
- Volatility expansion
- Consolidation/range-bound movement

### Approach 3: Adaptive Segmentation (Event-Driven)

**How it works:**
- Instead of fixed-length patches, segment at **semantically meaningful boundaries**: regime changes, volatility breakpoints, significant price moves
- Each variable-length segment becomes one token
- Use change-point detection algorithms (PELT, BOCPD) or learned segmentation

**Analogy to text:** This is like using a morphological analyzer instead of BPE — segmenting at linguistically meaningful boundaries rather than statistical frequency boundaries. For Indic languages, this would be like splitting at morpheme boundaries rather than arbitrary subword boundaries.

### Approach 4: Multi-Scale / Hierarchical Tokenization

**How it works:**
- Tokenize at multiple time scales simultaneously: tick-level, second-level, minute-level
- Each scale has its own vocabulary/embedding
- Cross-scale attention lets the model reason across granularities

**Analogy:** Like having character, subword, and word embeddings simultaneously — the model can attend to fine-grained detail when needed and coarse patterns for context.

### Key Design Decisions (What the Interviewer Really Wants to Hear)

| Decision | Text Tokenizer | Time-Series Tokenizer |
|---|---|---|
| Vocabulary size | 32K-128K subwords | Codebook size K or patch length P |
| Training data | Text corpus | Historical price/signal data |
| Fertility metric | Tokens per word | Tokens per time window |
| Domain adaptation | Language-specific merges | Asset/regime-specific codebooks |
| Quality signal | Downstream task perf | Reconstruction error + downstream |
| Compression vs info | Rare scripts need own tokens | Rare regimes need own codes |

**The punchline:** The same principles apply — domain-aware segmentation, balancing compression with information preservation, and evaluating via downstream task performance. Your experience building tokenizers for morphologically rich Indic languages (where naive BPE over-fragments) directly transfers to financial time-series (where naive fixed-window tokenization misses regime structure).

---

## 3. Principal Component Analysis

### Intuition

Imagine you have a cloud of data points in high-dimensional space. PCA asks: **what are the directions along which this data varies the most?**

Think of it physically: if you had a 3D cloud of data points shaped like a cigar, PCA would find that the long axis of the cigar is the "first principal component" — the direction capturing the most variance. The second PC would be the widest direction perpendicular to the first, and so on.

PCA is a **linear dimensionality reduction** technique that finds an orthogonal basis where the axes are ranked by how much variance they capture.

### The Math

Given a data matrix **X** of shape (n × d) where n = samples, d = features:

**Step 1: Center the data**

Subtract the mean of each feature:

$$\bar{x}_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}$$

$$\tilde{X} = X - \mathbf{1}\bar{x}^T$$

**Step 2: Compute the covariance matrix**

$$C = \frac{1}{n-1} \tilde{X}^T \tilde{X}$$

C is a d × d symmetric positive semi-definite matrix. Entry C_{jk} measures how features j and k co-vary.

**Step 3: Eigendecomposition**

$$C = V \Lambda V^T$$

Where:
- V = [v₁, v₂, ..., v_d] is the matrix of eigenvectors (principal components)
- Λ = diag(λ₁, λ₂, ..., λ_d) with λ₁ ≥ λ₂ ≥ ... ≥ λ_d ≥ 0

Each eigenvalue λᵢ represents the variance captured by the i-th principal component.

**Step 4: Project**

To reduce to k dimensions, take the top-k eigenvectors:

$$Z = \tilde{X} V_k$$

where V_k = [v₁, ..., v_k] is (d × k), giving Z of shape (n × k).

### Why Eigenvectors of the Covariance Matrix?

We want to find a unit vector **w** that maximizes the variance of the projected data:

$$\text{Var}(\tilde{X}w) = w^T C w$$

Subject to ||w|| = 1. Using Lagrange multipliers:

$$\mathcal{L} = w^T C w - \lambda(w^T w - 1)$$

$$\frac{\partial \mathcal{L}}{\partial w} = 2Cw - 2\lambda w = 0$$

$$Cw = \lambda w$$

This is the eigenvalue equation. The maximum variance direction is the eigenvector with the largest eigenvalue.

### Equivalent View: SVD

In practice, PCA is computed via SVD of the centered data matrix:

$$\tilde{X} = U \Sigma V^T$$

- The columns of V are the principal component directions (same eigenvectors of C)
- The singular values σᵢ relate to eigenvalues: λᵢ = σᵢ² / (n-1)
- The columns of UΣ (or equivalently XV) are the projected coordinates

SVD is numerically more stable than explicitly forming C and eigendecomposing it.

### Explained Variance Ratio

The fraction of total variance captured by the first k components:

$$\text{EVR}(k) = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

A common heuristic: choose k such that EVR(k) ≥ 0.95 (95% of variance explained).

### Limitations and Assumptions

- **Linearity**: PCA finds linear projections. If the data lies on a nonlinear manifold (e.g., Swiss roll), PCA fails — use kernel PCA or t-SNE/UMAP instead.
- **Variance = importance**: PCA assumes directions of maximum variance are the most informative. This may not align with discriminative power (high-variance direction might just be noise). For classification, LDA might be more appropriate.
- **Sensitive to scaling**: If features have different units/scales, the feature with the largest absolute values dominates. Always standardize (zero mean, unit variance) before PCA unless features are already comparable.
- **Orthogonality constraint**: PCs are forced to be orthogonal, which may not match the true data structure.

### PCA from Scratch in Python

```python
import numpy as np

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None    # (n_components, n_features) — the PC directions
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
    
    def fit(self, X: np.ndarray):
        """
        X: (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # Step 1: Center the data
        self.mean = X.mean(axis=0)                   # (n_features,)
        X_centered = X - self.mean                    # (n_samples, n_features)
        
        # Step 2: Covariance matrix
        # Using (1/(n-1)) for unbiased estimate
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)  # (d, d)
        
        # Step 3: Eigendecomposition
        # np.linalg.eigh is for symmetric matrices — returns sorted ascending
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Reverse to get descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Store the top-k components (each row is a PC direction)
        self.components = eigenvectors[:, :self.n_components].T   # (k, d)
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = (
            self.explained_variance / eigenvalues.sum()
        )
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto the principal components."""
        X_centered = X - self.mean
        return X_centered @ self.components.T    # (n, d) @ (d, k) = (n, k)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct data from reduced representation."""
        return Z @ self.components + self.mean   # (n, k) @ (k, d) + (d,)


# --- SVD-based implementation (more numerically stable) ---

class PCA_SVD:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
    
    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        
        # Full SVD: X = U @ diag(S) @ Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Vt rows are the principal components
        self.components = Vt[:self.n_components]           # (k, d)
        
        # Eigenvalues of covariance matrix = S^2 / (n-1)
        self.explained_variance = (S[:self.n_components] ** 2) / (n_samples - 1)
        total_var = (S ** 2).sum() / (n_samples - 1)
        self.explained_variance_ratio = self.explained_variance / total_var
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# --- Usage & Verification ---

if __name__ == "__main__":
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.datasets import load_iris
    
    X = load_iris().data    # (150, 4)
    
    # Our implementation
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    
    # Sklearn
    sk_pca = SklearnPCA(n_components=2)
    Z_sk = sk_pca.fit_transform(X)
    
    print("Our explained variance ratio:", pca.explained_variance_ratio)
    print("Sklearn explained variance ratio:", sk_pca.explained_variance_ratio_)
    # Should match (up to sign flips in eigenvectors, which is normal)
    
    print(f"\nOur PCA shape: {Z.shape}")
    print(f"Reconstruction error: {np.mean((pca.inverse_transform(Z) - X)**2):.6f}")
```

### Common Interview Follow-Ups

**Q: What's the time complexity?**
- Covariance matrix: O(n·d²)
- Eigendecomposition: O(d³)
- Total: O(n·d² + d³)
- SVD of X directly: O(min(n·d², n²·d)), better when n >> d or d >> n

**Q: How does PCA relate to autoencoders?**
A linear autoencoder with a bottleneck of size k learns the same subspace as PCA with k components. The encoder weights converge to a rotation of the top-k eigenvectors. Nonlinear autoencoders generalize PCA to nonlinear manifolds.

**Q: When would you NOT use PCA?**
When features have very different scales and you forget to standardize. When the relationship between features is nonlinear. When interpretability of individual features matters (PCA components are linear combinations — hard to interpret). When you need discriminative dimensionality reduction (use LDA instead).

---

## 4. Normalization, Dropout & Weight Initialization

### Batch Normalization

**What it does:** For a mini-batch of inputs, normalize each feature across the batch dimension to have zero mean and unit variance, then apply a learned affine transformation.

**Math:** Given a mini-batch B = {x₁, ..., x_m} for a particular feature/channel:

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

Where γ (scale) and β (shift) are **learned parameters**. They allow the network to undo the normalization if that's optimal — the network can learn γ = σ and β = μ to recover the original distribution.

**Why it works (intuitions, plural — no single accepted explanation):**
- Reduces internal covariate shift (original Ioffe & Szegedy claim, now debated)
- Smooths the loss landscape, making optimization easier (Santurkar et al., 2018)
- Acts as a regularizer (batch statistics introduce noise)
- Allows higher learning rates without divergence

**The problem with BatchNorm:**
- **Batch size dependency**: With small batches (e.g., batch size 1-2), the batch statistics are noisy and unreliable
- **Sequence models**: In RNNs/Transformers, the batch dimension mixes unrelated sequences. Normalizing across the batch means a token's normalization depends on what other sequences happen to be in the batch — this is semantically wrong
- **Variable-length sequences**: Different sequences have different lengths; batch statistics are ill-defined
- **Train/test discrepancy**: At test time, you use running averages, not batch stats

### Layer Normalization

**What it does:** Normalize across the **feature dimension** for each individual sample, instead of across the batch.

**Math:** For a single sample x (a vector of d features):

$$\mu = \frac{1}{d} \sum_{j=1}^{d} x_j$$

$$\sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2$$

$$\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y_j = \gamma_j \hat{x}_j + \beta_j$$

**Key difference:** Each sample is normalized independently. No dependence on other samples in the batch.

### Why Layer Norm for Transformers

| Property | BatchNorm | LayerNorm |
|---|---|---|
| Normalizes across | Batch dimension | Feature dimension |
| Depends on batch? | Yes | No |
| Works with batch=1? | Poorly | Yes |
| Variable-length seqs? | Problematic | Fine |
| Train/test behavior | Different (running stats) | Same |
| Autoregressive generation? | Broken (batch=1) | Works |

In transformer inference, you generate one token at a time (batch size 1, sequence length grows). BatchNorm's statistics would be meaningless. LayerNorm treats each token's representation independently — it normalizes the d-dimensional hidden vector for that token, regardless of what else is in the batch or sequence.

**RMSNorm (modern variant):** Many recent LLMs (LLaMA, etc.) use RMSNorm, which drops the mean-centering:

$$\hat{x}_j = \frac{x_j}{\text{RMS}(x)} \cdot \gamma_j, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2}$$

Simpler, cheaper, works just as well empirically.

**Pre-Norm vs Post-Norm:** Original transformer uses post-norm (normalize after residual addition). Modern LLMs use pre-norm (normalize before the sub-layer). Pre-norm is more stable for deep networks because the residual connection carries un-normalized information directly.

### Dropout

**What it does:** During training, randomly set each neuron's output to zero with probability p. Scale the remaining activations by 1/(1-p) to maintain expected values.

**Math (inverted dropout — the standard implementation):**

During training:
$$m_j \sim \text{Bernoulli}(1-p)$$
$$\tilde{h}_j = \frac{m_j \cdot h_j}{1-p}$$

During inference: no dropout, use all neurons as-is.

The 1/(1-p) scaling during training (inverted dropout) means you don't need to scale at test time — the expected value of each neuron is the same during training and testing.

**Why it works:**

1. **Ensemble interpretation**: Each training step uses a random sub-network. Dropout trains an exponential number of sub-networks (2^d for d neurons) that share parameters. At test time, using all neurons with no dropout approximates the ensemble average.

2. **Prevents co-adaptation**: Without dropout, neurons can develop complex co-dependencies ("neuron A only works when neuron B fires a specific way"). Dropout forces each neuron to be independently useful, learning more robust features.

3. **Implicit regularization**: Acts like L2 regularization with adaptive coefficients (Wager et al., 2013). Reduces effective model capacity during training.

**Practical details:**
- Typical p: 0.1 to 0.5 (p = probability of dropping, not keeping)
- In transformers: usually p = 0.1 for attention weights and FFN outputs
- Don't apply dropout to embeddings in modern LLMs (some older architectures did)
- Dropout is typically NOT used during fine-tuning of very large pretrained models — the model is already well-regularized by pretraining

### Weight Initialization Strategies

**Why it matters:** If weights are too large → activations explode → gradients explode. If weights are too small → activations vanish → gradients vanish. Good initialization keeps the variance of activations stable across layers.

#### Xavier/Glorot Initialization (2010)

**Designed for:** Layers with symmetric activations (tanh, sigmoid, linear)

**Goal:** Keep the variance of activations and gradients the same across layers.

**Derivation sketch:** For a layer y = Wx (no bias, no activation), if inputs xᵢ are i.i.d. with variance Var(x):

$$\text{Var}(y_j) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

To maintain Var(y) = Var(x), we need Var(w) = 1/n_in.

Similarly, for gradient flow backward, we need Var(w) = 1/n_out.

Xavier compromises:

$$\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$$

- **Uniform version**: w ~ U[-a, a] where a = √(6 / (n_in + n_out))
- **Normal version**: w ~ N(0, 2/(n_in + n_out))

#### Kaiming/He Initialization (2015)

**Designed for:** ReLU and variants (which break the symmetry assumption of Xavier)

**Key insight:** ReLU zeros out ~half the activations. This halves the variance at each layer. To compensate:

$$\text{Var}(w) = \frac{2}{n_{in}}$$

- **Normal version**: w ~ N(0, 2/n_in)
- **Uniform version**: w ~ U[-a, a] where a = √(6 / n_in)

For Leaky ReLU with slope α: Var(w) = 2 / ((1 + α²) · n_in)

#### Modern Transformer Initialization

Large transformer models use specific patterns:

- **Embeddings**: N(0, 0.02) typically
- **Attention/FFN weights**: Xavier or N(0, 0.02)
- **Output projection of residual blocks**: Scale by 1/√(2N) where N = number of layers (GPT-2 style). This accounts for the accumulation of variance through residual additions — each of the 2N sub-layers (attention + FFN per layer) contributes to the residual stream.
- **Biases**: Initialized to zero (standard)

**μP (Maximal Update Parameterization)**: A principled approach where initialization scales and learning rates are set so that optimal hyperparameters transfer across model widths. Key idea: wider layers need smaller weight updates to have the same effect on the output.

---

## 5. DeepSpeed ZeRO Stages

### The Memory Problem

Training a 7B parameter model in fp16:
- **Model parameters**: 7B × 2 bytes = 14 GB
- **Gradients**: 7B × 2 bytes = 14 GB  
- **Optimizer states** (Adam): 7B × 4 bytes (fp32 copy) + 7B × 4 bytes (momentum) + 7B × 4 bytes (variance) = 84 GB

**Total**: ~112 GB for a single copy. A single A100 has 80GB. You can't even fit the model on one GPU.

Standard Data Parallelism (DP) replicates ALL of this on every GPU. 8 GPUs? 8 × 112 GB = 896 GB of redundant memory. Wasteful.

### ZeRO: Zero Redundancy Optimizer

ZeRO's insight: in data parallelism, each GPU computes gradients on different data but maintains a **full copy** of parameters, gradients, and optimizer states. This is redundant — we can **partition** these across GPUs and gather only when needed.

### ZeRO Stage 1: Optimizer State Partitioning

**What's partitioned:** Only the optimizer states (Adam's momentum and variance, fp32 master weights)

**What's replicated:** Parameters (fp16) and gradients on every GPU

**How it works:**
- Each GPU is responsible for 1/N_th of the optimizer states
- After backward pass (all-reduce gradients as usual), each GPU only updates its partition of the parameters using its partition of optimizer states
- Then all-gather the updated parameters so every GPU has the full model

**Memory savings:** Optimizer states go from 84 GB per GPU → 84/N GB per GPU. With 8 GPUs: 84 → 10.5 GB.

**Communication:** Same as standard DP (one all-reduce for gradients) + one all-gather for updated parameters. Roughly 1.5× the communication of standard DP.

**Total memory per GPU (8 GPUs):** 14 (params) + 14 (grads) + 10.5 (opt states) = 38.5 GB (down from 112 GB)

### ZeRO Stage 2: Gradient Partitioning

**What's partitioned:** Optimizer states AND gradients

**What's replicated:** Only parameters (fp16)

**How it works:**
- During backward pass, instead of all-reducing gradients, use **reduce-scatter**: each GPU gets the reduced gradient for only its partition
- Each GPU only stores gradients for the parameters it's responsible for updating
- After update, all-gather the updated parameters

**Memory savings:** Gradients go from 14 GB per GPU → 14/N per GPU.

**Communication:** Replace all-reduce with reduce-scatter (half the communication of all-reduce) + all-gather. Net communication = same volume as standard DP but reorganized.

**Total memory per GPU (8 GPUs):** 14 (params) + 1.75 (grads) + 10.5 (opt states) = 26.25 GB

### ZeRO Stage 3: Parameter Partitioning

**What's partitioned:** EVERYTHING — optimizer states, gradients, AND parameters

**What's replicated:** Nothing (each GPU holds 1/N_th of each)

**How it works:**
- Each GPU stores only 1/N_th of the model parameters
- For the forward pass: before computing a layer, all-gather its parameters from all GPUs, compute, then discard the non-local parameters
- For the backward pass: same — all-gather parameters for each layer, compute gradients, reduce-scatter gradients, discard non-local params
- Each GPU updates only its partition

**Memory savings:** Everything is 1/N. With 8 GPUs: (14 + 14 + 84) / 8 = 14 GB total per GPU

**Communication:** ~3× the communication of standard DP (all-gather in forward, all-gather in backward, reduce-scatter for gradients). This is the tradeoff: maximum memory savings but highest communication overhead.

**Key optimization — communication overlap:** All-gathers for layer i+1 overlap with computation on layer i, hiding latency. Prefetching makes the communication overhead much less than 3× in practice.

### Summary Table

| | Partitioned | Memory per GPU (7B, 8 GPUs) | Communication vs DP |
|---|---|---|---|
| Standard DP | Nothing | 112 GB | 1× |
| ZeRO-1 | Optimizer states | ~38.5 GB | ~1.5× |
| ZeRO-2 | + Gradients | ~26.25 GB | ~1× (reorganized) |
| ZeRO-3 | + Parameters | ~14 GB | ~1.5× (with overlap) |

### ZeRO-Offload and ZeRO-Infinity

- **ZeRO-Offload**: Offload optimizer states and computation to CPU, freeing GPU memory further. Useful for training large models on fewer GPUs.
- **ZeRO-Infinity**: Extends offloading to NVMe SSDs. Can theoretically train trillion-parameter models on a single node by using the entire memory hierarchy (GPU → CPU → NVMe).

### When to Use Which Stage

- **ZeRO-1**: Default choice. Minimal overhead, significant savings. Use when your model almost fits but optimizer states push it over.
- **ZeRO-2**: When gradients are also a bottleneck. Common for models in the 3B-13B range on consumer/mid-range GPUs.
- **ZeRO-3**: When the model parameters themselves don't fit on a single GPU. Essential for 13B+ models on 80GB GPUs, or any large model on smaller GPUs. Combine with gradient checkpointing for maximum savings.

### Comparison with Other Parallelism Strategies

- **Tensor Parallelism** (Megatron-LM): Splits individual layers across GPUs. Requires fast interconnect (NVLink). Complementary to ZeRO.
- **Pipeline Parallelism**: Splits the model by layers across GPUs. Introduces pipeline bubbles. Complementary to ZeRO.
- **FSDP** (PyTorch Fully Sharded Data Parallel): PyTorch's native implementation of ZeRO Stage 3 concepts. Very similar in principle.

Modern large-scale training typically combines: ZeRO-3 (or FSDP) + tensor parallelism within a node + pipeline parallelism across nodes.

---

## 6. Python Internals

### `__slots__`

By default, Python objects store attributes in a `__dict__` (a hash table). `__slots__` replaces this with a fixed-size array, like a C struct.

```python
class PointDict:
    def __init__(self, x, y):
        self.x = x
        self.y = y
# Each instance has a __dict__: {'x': 1, 'y': 2}
# ~300 bytes per instance (dict overhead)

class PointSlots:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y
# Each instance stores x, y as fixed offsets, no dict
# ~60 bytes per instance
```

**Tradeoffs:**
- ✅ 4-5× less memory per instance
- ✅ Slightly faster attribute access (no hash lookup)
- ❌ Can't add arbitrary attributes at runtime (`p.z = 3` raises AttributeError)
- ❌ No `__dict__` means no `vars(obj)` or `obj.__dict__`
- ❌ Must be defined in every class in the inheritance chain, or you lose the benefit

**When to use:** Data classes with millions of instances (e.g., nodes in a graph, order book entries in a trading system).

### Metaclasses

A metaclass is the "class of a class." When you write `class Foo:`, Python calls `type('Foo', bases, namespace)` to create the class object. You can customize this by providing your own metaclass.

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "connected"

db1 = Database()
db2 = Database()
assert db1 is db2  # True — same instance
```

**The lookup chain:** `instance.__class__` → class, `class.__class__` → metaclass (usually `type`).

**When metaclasses are used in practice:** ORMs (Django's models), ABCs (`abc.ABCMeta`), validation frameworks, automatic registration of plugins/subclasses.

### The GIL (Global Interpreter Lock)

**What:** A mutex in CPython that allows only one thread to execute Python bytecode at a time.

**Why it exists:** CPython's memory management (reference counting) is not thread-safe. Without the GIL, two threads could simultaneously modify a reference count, causing memory corruption.

**Implications:**
- **CPU-bound** multithreaded Python code runs **slower** than single-threaded (threads fight for the GIL, adding context-switch overhead)
- **I/O-bound** multithreading works fine (GIL is released during I/O operations: network, disk, sleep)
- **C extensions** can release the GIL (NumPy, PyTorch operations do this — that's why numerical code IS fast despite the GIL)

**Workarounds:**
- `multiprocessing` — separate processes, each with its own GIL
- C extensions that release the GIL (Cython, ctypes, pybind11)
- `asyncio` — cooperative concurrency for I/O-bound work (no threads needed)
- **PEP 703 / Free-threaded Python (3.13+)**: Experimental no-GIL mode. Reference counting replaced with biased reference counting + deferred reference counting. Still experimental as of 2026.

### asyncio Basics

asyncio is **cooperative multitasking** — coroutines voluntarily yield control at `await` points.

```python
import asyncio

async def fetch_data(url: str) -> str:
    print(f"Start fetching {url}")
    await asyncio.sleep(1)  # Simulate I/O — yields control here
    print(f"Done fetching {url}")
    return f"Data from {url}"

async def main():
    # Runs concurrently, not in parallel
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3"),
    )
    # Total time: ~1 second (not 3), because all three sleep concurrently

asyncio.run(main())
```

**Key concepts:**
- **Coroutine**: An `async def` function. Calling it returns a coroutine object (doesn't execute).
- **Event loop**: Manages and schedules coroutines. Only one event loop per thread.
- **`await`**: Suspends the coroutine and gives control back to the event loop, which can run other coroutines.
- **Not parallelism**: Everything runs in one thread. Concurrency through cooperation, not preemption.

### Context Managers

The `with` statement calls `__enter__` and guarantees `__exit__` is called, even on exceptions.

```python
class Timer:
    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.perf_counter() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")
        return False  # Don't suppress exceptions

# Or with contextlib:
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.perf_counter()
    try:
        yield    # Control returns to the `with` block body
    finally:
        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.4f}s")

with timer():
    sum(range(10**7))
```

**`__exit__` return value:** If `__exit__` returns `True`, the exception is suppressed. If `False` or `None`, the exception propagates. Use `return True` carefully.

### `__getattr__` vs `__getattribute__`

```python
class Demo:
    def __init__(self):
        self.x = 10
    
    def __getattribute__(self, name):
        # Called for EVERY attribute access, including existing ones
        print(f"__getattribute__ called for '{name}'")
        return super().__getattribute__(name)
    
    def __getattr__(self, name):
        # Called ONLY when normal lookup fails (attribute doesn't exist)
        print(f"__getattr__ called for '{name}'")
        return f"default_{name}"

d = Demo()
d.x           # Calls __getattribute__('x') → returns 10
d.missing     # Calls __getattribute__('missing') → fails → calls __getattr__('missing')
```

**Warning:** Inside `__getattribute__`, if you access `self.anything`, it triggers `__getattribute__` again → infinite recursion. Always use `super().__getattribute__()` or `object.__getattribute__(self, name)`.

### Python Dict Hashing

Python dicts are open-addressing hash tables (not separate chaining).

**Hash → probe sequence:**
1. Compute `hash(key)` (calls `__hash__()`)
2. Index = hash % table_size
3. If slot is empty → insert. If slot is occupied:
   - Compare keys using `__eq__()` — if equal, update value
   - If not equal → **probe** to next slot (perturbation-based probing, not linear)
4. Resize when load factor exceeds ~2/3

**Perturbation probing** (CPython specific): `next_index = (5 * index + 1 + perturb) % size; perturb >>= 5`. This distributes collisions better than linear probing.

**Requirements for dict keys:**
- Must be **hashable**: implement `__hash__()` and `__eq__()`
- **Invariant**: if `a == b`, then `hash(a) == hash(b)` (but not vice versa)
- Mutable objects (lists, dicts, sets) are not hashable — if you mutate a key after insertion, the hash changes and you can never find it again

**Compact dict (Python 3.6+):** Dicts maintain insertion order (implementation detail in 3.6, language guarantee in 3.7+). Two arrays: a dense array of key-value entries (insertion order) and a sparse hash table of indices into the dense array.

### Mutable Default Argument Gotcha

```python
def append_to(element, target=[]):
    target.append(element)
    return target

append_to(1)    # [1]
append_to(2)    # [1, 2]  ← NOT [2]!
append_to(3)    # [1, 2, 3]
```

**Why:** Default arguments are evaluated **once** at function definition time, not at each call. The list `[]` is created once and shared across all calls. Each call mutates the same list object.

**Fix:**
```python
def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target
```

This pattern (None sentinel, create fresh inside) is the standard Python idiom.

---

## 7. Bias-Variance Tradeoff, Hypothesis Testing & MLE vs MAP

### Bias-Variance Tradeoff

**Setup:** You have a true function f(x) generating data y = f(x) + ε where ε ~ N(0, σ²). You train a model f̂(x) on a random training set D. The expected prediction error at a point x₀ is:

$$E[(y - \hat{f}(x_0))^2] = \text{Bias}[\hat{f}(x_0)]^2 + \text{Var}[\hat{f}(x_0)] + \sigma^2$$

Each term:

**Bias²** = [E[f̂(x₀)] - f(x₀)]²

The systematic error — how far the average prediction (over all possible training sets) is from the truth. High bias means the model consistently misses the target, regardless of training data.

**Intuition:** A linear model fitting a quadratic relationship will always be wrong in the same direction — it's structurally incapable of capturing the curve. That systematic miss is bias.

**Variance** = E[(f̂(x₀) - E[f̂(x₀)])²]

How much the prediction fluctuates across different training sets. High variance means the model is very sensitive to the specific training data it saw.

**Intuition:** A degree-20 polynomial fit to 25 data points will look wildly different depending on which 25 points you sample. That instability is variance.

**Irreducible error** = σ²

Noise in the data that no model can capture. The floor on achievable error.

**The tradeoff in practice:**

- **Underfitting** (high bias, low variance): Model too simple. Linear regression on nonlinear data. Training AND test error are high.
- **Overfitting** (low bias, high variance): Model too complex. Memorizes training noise. Training error is low, test error is high.
- **Sweet spot**: Model complex enough to capture the true pattern, but regularized enough to not fit noise.

**How to diagnose:**
- Training error high → underfitting → need more capacity (bigger model, more features)
- Training error low but test error high → overfitting → need regularization (dropout, weight decay, early stopping, more data)
- Both errors low and close → good fit

**In deep learning, the classical tradeoff is complicated:**
Modern overparameterized neural networks (more parameters than data points) achieve zero training error yet still generalize well — this was unexpected. The "double descent" phenomenon: test error first increases (classical regime), then decreases as you add more parameters past the interpolation threshold. This challenges the simple bias-variance picture but doesn't invalidate it — the implicit regularization of SGD + architecture choices effectively controls variance even in overparameterized models.

### Hypothesis Testing Intuition

**The core question:** "Is this result real or could it have happened by chance?"

**Framework:**

1. **Null hypothesis (H₀)**: The "boring" explanation. "There's no effect." "The model is no better than random." "This coefficient is zero."

2. **Alternative hypothesis (H₁)**: The "interesting" claim. "There IS an effect." "The model IS better than random."

3. **Test statistic**: A number computed from your data that measures how far the observation is from what H₀ predicts. Example: the t-statistic for a mean test.

4. **p-value**: The probability of seeing a test statistic at least as extreme as yours, **assuming H₀ is true**. Small p-value → the data is unlikely under H₀ → evidence against H₀.

   **Critical misconception:** p-value is NOT "the probability that H₀ is true." It's "the probability of this data (or more extreme) given H₀."

5. **Significance level α** (typically 0.05): Your threshold. If p < α, reject H₀.

**Type I and Type II errors:**

| | H₀ actually true | H₀ actually false |
|---|---|---|
| Reject H₀ | **Type I** (false positive), probability = α | Correct (power = 1-β) |
| Fail to reject H₀ | Correct | **Type II** (false negative), probability = β |

**Example in ML context:** You trained two models, A and B, on 5 different train/test splits. Model A's accuracy: [85.2, 84.8, 85.5, 84.9, 85.1]. Model B's accuracy: [86.1, 85.7, 86.3, 85.9, 86.0].

H₀: μ_A = μ_B (no difference). H₁: μ_B > μ_A.

A paired t-test would compute the mean difference (≈1.0) and the standard error of the differences, yielding a t-statistic and p-value. If p < 0.05, you'd conclude B is significantly better.

**Practical warnings:**
- p = 0.049 and p = 0.051 are practically identical — don't treat α = 0.05 as a sharp cliff
- Multiple comparisons: if you test 20 hypotheses at α = 0.05, you expect 1 false positive by chance. Use Bonferroni correction or FDR control.
- Statistical significance ≠ practical significance. A model that's 0.01% better with p = 0.001 may not be worth deploying.

### MLE vs MAP

Both are methods for estimating parameters θ given observed data D = {x₁, ..., xₙ}.

#### Maximum Likelihood Estimation (MLE)

**Question:** What parameter value θ makes the observed data most probable?

$$\hat{\theta}_{MLE} = \arg\max_\theta \; P(D | \theta) = \arg\max_\theta \; \prod_{i=1}^n P(x_i | \theta)$$

In practice, maximize the **log-likelihood** (turns products into sums):

$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^n \log P(x_i | \theta)$$

**Example — Coin flip:** You flip a coin 10 times, get 7 heads.

P(D|θ) = θ⁷(1-θ)³ where θ = P(heads)

log P(D|θ) = 7 log θ + 3 log(1-θ)

Take derivative, set to zero:

d/dθ: 7/θ - 3/(1-θ) = 0 → θ = 7/10 = 0.7

MLE just uses the observed frequency. With 7/10 heads, MLE says θ = 0.7.

**Properties of MLE:**
- **Consistent**: converges to true θ as n → ∞
- **Asymptotically efficient**: achieves the lowest possible variance among consistent estimators
- **Can overfit with small data**: 3 heads out of 3 flips → MLE says θ = 1.0 (the coin always lands heads). This is clearly unreasonable.

#### Maximum A Posteriori (MAP)

**Question:** What parameter value θ is most probable given the data AND our prior beliefs?

Using Bayes' theorem:

$$P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}$$

Since P(D) is constant w.r.t. θ:

$$\hat{\theta}_{MAP} = \arg\max_\theta \; P(D | \theta) \cdot P(\theta) = \arg\max_\theta \left[ \log P(D | \theta) + \log P(\theta) \right]$$

MAP = MLE + a prior term.

**Example — Coin flip with prior:**

Prior: θ ~ Beta(α, β). Say α = 3, β = 3 (prior belief: coin is probably fair-ish).

P(θ) ∝ θ^(α-1) · (1-θ)^(β-1) = θ² · (1-θ)²

log P(θ|D) ∝ 7 log θ + 3 log(1-θ) + 2 log θ + 2 log(1-θ) = 9 log θ + 5 log(1-θ)

Take derivative: 9/θ - 5/(1-θ) = 0 → θ = 9/14 ≈ 0.643

The prior "pulled" the estimate from 0.7 toward 0.5 (toward the prior mean). With more data, the likelihood dominates and MAP → MLE.

**The deep connection — MAP as regularized MLE:**

$$\hat{\theta}_{MAP} = \arg\max_\theta \left[ \log P(D|\theta) + \log P(\theta) \right]$$

If the prior is Gaussian: P(θ) = N(0, σ²_prior), then:

$$\log P(\theta) = -\frac{\|\theta\|^2}{2\sigma^2_{prior}} + \text{const}$$

So MAP with a Gaussian prior = MLE + L2 regularization (weight decay). The regularization strength 1/σ²_prior plays the role of the λ in L2 penalty.

If the prior is Laplace: P(θ) ∝ exp(-|θ|/b), then:

$$\log P(\theta) = -\frac{|\theta|}{b} + \text{const}$$

MAP with a Laplace prior = MLE + L1 regularization (lasso). L1 encourages sparsity because the Laplace prior concentrates mass at zero.

**Summary Table:**

| | MLE | MAP |
|---|---|---|
| Objective | max P(D\|θ) | max P(D\|θ)·P(θ) |
| Uses prior? | No | Yes |
| Regularization? | None (prone to overfit) | Built-in via prior |
| With infinite data | Converges to truth | Converges to MLE (prior washes out) |
| Gaussian prior = | — | L2 regularization |
| Laplace prior = | — | L1 regularization |
| Point estimate? | Yes | Yes (still a single θ̂) |

**Full Bayesian (beyond MAP):** Both MLE and MAP give a single point estimate θ̂. Full Bayesian inference maintains the entire posterior distribution P(θ|D) and integrates over it for predictions. This gives uncertainty estimates but is usually intractable for neural networks (approximated by variational inference, MCMC, or dropout-as-approximate-inference).

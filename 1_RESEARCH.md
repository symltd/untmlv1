# Transformer Per-Token FLOPs Estimation

## Assumptions
- Embedding dimension: $d = 4096$  
- FFN hidden dimension: $d_{ff} = 4d = 16384$  
- Number of tokens: 1 (we’ll focus on per-token FLOPs)

---

## Step 1: Attention (causal, single head simplified)

**Compute Q, K, V:** $Q = x W_Q$, $K = x W_K$, $V = x W_V$  

Each is $d \times d$  

**FLOPs per token:** $3 \cdot d^2 = 3 \cdot 4096^2 \approx 50$ MFLOPs  

**Compute attention scores and weighted sum:** $\text{Attention} = \text{Softmax}(Q K^T / d) V$  

For 1 token, simplified: $d^2 = 16.8$ MFLOPs  

**Project output:** $O = \text{Attention} \cdot W_O$  

FLOPs: $d^2 = 16.8$ MFLOPs  

✅ **Total attention FLOPs per token:** ~83 MFLOPs

---

## Step 2: Feedforward Network (FFN)

**First linear layer:** $x W_1$  

FLOPs: $d \cdot d_{ff} = 4096 \cdot 16384 \approx 67.1$ MFLOPs  

**Activation (GELU/SwiGLU):** ~2× hidden size ≈ 32.8 MFLOPs  

**Second linear layer:** $\text{hidden} \cdot W_2$  

FLOPs: $d_{ff} \cdot d = 16384 \cdot 4096 \approx 67.1$ MFLOPs  

✅ **Total FFN FLOPs per token:** ~167 MFLOPs

---

## Step 3: Next Layer / Residual
- Add residual connection: negligible FLOPs (~0.01 MFLOPs)  
- LayerNorm: ~2d ≈ 8.2k FLOPs, tiny relative  

---

## Step 4: Summary Per Token

| Step | FLOPs (per token) | % of total |
|------|-----------------|------------|
| Attention | 83 MFLOPs | 33% |
| FFN | 167 MFLOPs | 66% |
| Residual + LayerNorm | 0.01 MFLOPs | <1% |
| **Total** | 250 MFLOPs | 100% |

**Observation:**  
Even for a single token, FFN dominates FLOPs, about twice attention. This explains why in large transformers, FFN consumes more power than attention per token.



# Novel FFN Alternatives in Transformers

### 1. Fourier / Spectral Token Mixing
**Idea:** Replace FFN with a small spectral transform per token:  
Apply FFT → learnable spectral filter → inverse FFT  
Captures non-linear token features without a huge dense matrix  

**FLOPs:** ~ $O(d \log d)$ instead of $O(d \cdot 4d)$

**Benefits:**  
- Reduces computation per token  
- Encodes frequency-based interactions in token embeddings  

**Novelty:** Used in some vision transformers (FNet), but rarely in text LLMs  

---

### 2. Learned Polynomial / Chebyshev Expansion
**Idea:** Each token embedding is transformed via a low-degree polynomial basis:  

$y = \sum_{k=0}^{K} \alpha_k x^k$

Only a few learnable coefficients ($\alpha_k$)  

**FLOPs:** $O(d \cdot K)$, where $K \ll 4d$  

**Pros:** Can approximate non-linear transformations without a full FFN  

**Novelty:** Almost unused in text transformers  

---

### 3. Sparse Random Projections + Nonlinearity
**Idea:** Project each token into a sparse higher-dimensional space using random, fixed sparse matrices, then apply lightweight non-linearity  

**Benefits:**  
- Reduces trainable parameters  
- FLOPs dominated by sparse multiplication  
- Can mimic expressive FFN features with <50% FLOPs  

**Novelty:** Similar to reservoir computing, but applied per token  

---

### 4. Self-Conditioned FFN (“Token Feedback”)
**Idea:** Instead of one dense FFN, apply iterative small token-wise layers:  
- Each step: small linear + activation + gated feedback from previous token features  
- Equivalent to an RNN per token, but very lightweight  

**FLOPs per micro-step:** tiny (~10–20% of original FFN)  

**Pros:** Can approximate complex transformations over multiple micro-steps  

---

### 5. Low-Rank Tensor Factorization
**Idea:** Represent FFN weights as tensor products of small matrices instead of one big dense matrix:  

$W \approx W_1 \otimes W_2$

**Benefits:**  
- Can reduce storage and FLOPs 4–10×  
- Preserves high-dimensional interactions implicitly  

**Novelty:** Rarely used in mainstream LLMs  

---

## Comparison Table of Novel Approaches

| Method | FLOPs vs Standard FFN | Power | Pros | Cons | Novelty |
|--------|----------------------|-------|------|------|---------|
| Fourier / Spectral | 3–10× less | Medium | Captures frequency patterns | Needs FFT ops | Rare in text LLMs |
| Polynomial Expansion | 4–8× less | Low | Learnable non-linearity | Limited capacity | Almost unused |
| Sparse Random Projections | 2–5× less | Medium | Sparse, low-memory | Needs careful sparsity tuning | Very rare |
| Self-Conditioned / Token Feedback | 3–4× less | Low | Iterative, flexible | Multiple micro-steps | Novel idea |
| Tensor Factorization | 4–10× less | Low | Preserves interactions | Implementation complexity | Rare in LLMs |

# Sparse / Ultra-Efficient FFN Variants

## 1. Sparse Spectral FFN
**Concept:**  
Take a token embedding $x \in \mathbb{R}^d$.  
Apply FFT → learnable spectral filter → inverse FFT.  
Introduce sparsity in the spectral domain: only keep $top-k$ frequency coefficients per token.

**Implementation:**  
- FFT($x$) → $X(f)$  
- Mask frequencies: keep only $top-k$ amplitudes or learned mask (sparse tensor)  
- Apply learnable spectral filter on the sparse coefficients  
- Inverse FFT → token output  

**FLOPs impact:**  
- Dense FFT: $O(d \log d)$  
- Sparse: $O(k \log k),\ k \ll d$ → drastic reduction  

**Power impact:** proportional to FLOPs.  

✅ **Benefit:** captures non-linear features in frequency space with very few computations.

---

## 2. Sparse Polynomial / Chebyshev Expansion
**Concept:**  
Represent FFN as a polynomial: $y = \sum_{k=0}^{K} \alpha_k x^k$  
Apply sparsity in coefficients: many $\alpha_k$ set to zero, or only a subset of token dimensions are transformed.

**Implementation:**  
- Learnable sparse mask on the coefficients  
- Only compute powers for the active coefficients per token  

**FLOPs impact:**  
- Standard: $O(d \cdot K)$  
- Sparse: $O(s \cdot K),\ s \ll d$  

✅ **Benefit:** Reduces computation while preserving non-linearity.

---

## 3. Sparse Self-Conditioned / Token-Feedback FFN
**Concept:**  
Iterative micro-steps per token: small linear + activation + gated feedback.  
Introduce sparsity in:  
- The weight matrices (low-rank or $top-k$ elements)  
- Feedback path (only feed selected features forward)  

**Implementation:**  
- For each micro-step, select subset of hidden features  
- Only update those features  
- Residual features remain untouched  

**FLOPs impact:**  
- Standard micro-step: $O(d^2)$  
- Sparse: $O(s \cdot d)$ or $O(s^2),\ s \ll d$  

✅ **Benefit:** Maintains iterative transformation expressivity, but cuts FLOPs dramatically.

---

## 4. Sparse Tensor Factorization FFN
**Concept:**  
Represent $W \approx W_1 \otimes W_2$ (tensor factorization)  
Introduce sparsity in the factor matrices ($top-k$ elements per row/column)

**Implementation:**  
- Multiply only non-zero elements  
- Can combine with low-rank to further reduce FLOPs  

**FLOPs impact:**  
- Standard dense FFN: $O(d \cdot d_{ff})$  
- Sparse + factorized: $O(k \cdot r),\ k,r \ll d,d_{ff}$  

✅ **Benefit:** Reduces memory footprint and computation while preserving interactions.

---

## 5. Combining Methods – A Unified “Ultra-Efficient FFN”

**Mix-and-match options:**

| Layer Type | Method | Sparsity Option | FLOPs Savings |
|------------|--------|----------------|---------------|
| Token transformation | Spectral FFN | $top-k$ freq coefficients | ~10–20× |
| Non-linearity | Polynomial / Chebyshev | Sparse coefficients | ~4–8× |
| Iterative refinement | Token-feedback FFN | Sparse feedback / low-rank | ~3–5× |
| Matrix factorization | Tensor FFN | Sparse factor matrices | ~4–10× |

**Workflow per token (example):**  
1. Spectral FFN (sparse): capture global token features efficiently  
2. Polynomial expansion (sparse): add non-linear expressivity  
3. Self-conditioned sparse micro-steps: refine features iteratively  
4. Sparse tensor factorization: project back to embedding space  

**FLOPs vs Standard Dense FFN:** could be >20× lower for large embeddings (~4k) while maintaining expressivity.

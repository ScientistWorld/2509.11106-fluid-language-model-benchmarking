# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

FLUID BENCHMARKING is a new LM evaluation method that improves benchmarking across four dimensions (efficiency, validity, variance, saturation) by combining two ideas from psychometrics: (1) item response theory (IRT) for measuring ability in a latent space, and (2) computerized adaptive testing for dynamic item selection.

## Approach

### Step 1: Fit an IRT model (2PL model)

Given existing LM evaluation results for a benchmark Q with k language models and l items, fit a 2-parameter logistic (2PL) model. Each item q_j has two parameters:
- **Difficulty** b_j: the ability level at which the LM has 50% probability of answering correctly
- **Discrimination** a_j > 0: how sharply the probability changes near the difficulty level

The probability that LM m_i answers item q_j correctly is:
```
p(u_ij = 1) = logistic(a_j * (θ_i - b_j))
```
where θ_i is the latent ability of the LM.

Parameters are estimated using MCMC with hierarchical priors. This produces item parameters (a_j, b_j) for each item in the benchmark.

### Step 2: Estimate ability using IRT

To estimate the ability of a new LM on a set of items Q*:
- Use the fitted item parameters (a_j, b_j) as fixed
- Maximize the posterior probability of the response vector (Equation 3 in paper)
- This gives ability estimate θ_i (ABILITY)

This contrasts with standard accuracy, which treats all items equally.

### Step 3: Dynamic item selection using Fisher information

For efficient evaluation, select items dynamically based on Fisher information:
```
I(θ, a_j, b_j) = a_j² * logistic(a_j(θ - b_j)) * [1 - logistic(a_j(θ - b_j))]
```

Fisher information is highest when the item difficulty matches the current ability estimate (b_j ≈ θ). Dynamic selection:
1. Start with empty selection set Q*(0) = ∅
2. At each step, select the item with highest Fisher information given current ability estimate
3. Re-estimate ability with the expanded set
4. Repeat until budget is reached

This naturally adapts to the LM's capability: weaker LMs get easier items, stronger LMs get harder items. High-discrimination items (which better differentiate between LMs) are preferentially selected.

### Ablation studies

The paper separates the contributions of IRT aggregation vs. dynamic selection:
- **RANDOM IRT**: Random item selection + IRT ability estimation (validity improvement from IRT alone)
- **RANDOM**: Random item selection + accuracy (baseline)
- **FLUID**: Dynamic selection + IRT ability estimation (full method)

## Main Claims

1. **Core claim**: FLUID BENCHMARKING outperforms all prior methods across all four dimensions (efficiency, validity, variance, saturation), often by large margins.
2. **Secondary claim 1**: IRT-based ability estimation alone improves validity substantially over accuracy.
3. **Secondary claim 2**: Dynamic item selection alone reduces variance substantially.
4. **Secondary claim 3**: FLUID BENCHMARKING almost entirely avoids mislabeled items (100x fewer than random selection).
5. **Secondary claim 4**: FLUID BENCHMARKING delays benchmark saturation — it continues providing a learning signal through the end of training.
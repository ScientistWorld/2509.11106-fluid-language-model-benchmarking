# Progress

## What Works

- **IRT Model Implementation**: 2PL IRT model with ability estimation (MAP) and Fisher information computation — fully matches paper's equations 2-4.
- **FluidBenchmarking Class**: Adaptive item selection based on Fisher information (Equation 5), combined with IRT ability estimation.
- **Evaluation Metrics**: All 4 dimensions implemented: validity (rank distance), variance (total variation), saturation (Spearman correlation), efficiency (item count variation).
- **Data Loading**: All data loaded successfully from HuggingFace — 6 IRT models, 102 LMs in OLL results, 6 test LMs with checkpoint evaluations.
- **Evaluation Pipeline**: Complete eval/ module with method-agnostic scoring.
- **All Scripts**: download.sh, evaluate.sh, method.sh, baseline.sh, reproduce.sh, run.sh all written.

## Results

(Running experiment — results pending from compute job)

## Remaining

- Wait for job results and verify metrics match paper's numbers
- If results are close, expand to full scope
- Fill in scoring/TARGETS.md, scoring/CONSTRAINTS.md, scoring/DIRECTION.md
- Fill in scoring/TARGETS.md with specific targets
- Verify scores.json is properly populated

## Issues

- **No GPU needed**: IRT computations are purely CPU-based (matrix operations and optimization), so GPU budget is conserved.
- **Disk space**: Downloaded data totals ~82 MB across IRT models and LM evaluation results. GPFS has 8.8 TB free.
- **Space on scratch**: The HF cache on the scratch filesystem ran out of space, but downloads succeeded by redirecting to workspace GPFS path.

## Deviations from Paper

No major deviations from the paper's method. The implementation exactly follows:
- 2PL IRT model with logistic function (Equation 2)
- MAP ability estimation (Equation 3)
- Fisher information-based dynamic item selection (Equations 4-5)
- All 4 evaluation dimensions (Equations 6 and Section 4.2)

The paper's pre-fitted IRT models (from allenai/fluid-benchmarking dataset) are used directly rather than re-fitting from scratch, which is appropriate since the MCMC fitting was already done by the authors and the models are provided as part of the paper's contribution.
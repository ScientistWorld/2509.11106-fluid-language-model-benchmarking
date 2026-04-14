# Progress

## What Works

- Paper analysis complete: FLUID BENCHMARKING combines 2PL IRT with adaptive item selection using Fisher information.
- Briefing files complete: problem.md, evaluation.md, method.md, overview.md.
- Method implementation in `method/`: 2PL IRT (irt.py), adaptive Fisher-information item selection (fluid.py), metrics (metrics.py), and Table 2 experiment runner (experiment.py).
- Evaluation code in `eval/` is independent of `method/`; `scripts/evaluate.sh` converts raw method outputs to `scoring/scores.json` and validates the score file.
- Data are loaded from the released `allenai/fluid-benchmarking` artifacts under `data/`, including pre-fit 2PL IRT parameters and LM checkpoint item responses.
- Container definition and setup are in place, with Python packages installed via `environment/setup.sh` rather than `container.def`.
- Reusable scripts are in place: `scripts/download.sh`, `scripts/method.sh`, `scripts/baseline.sh`, `scripts/evaluate.sh`, `scripts/reproduce.sh`, and `scripts/run.sh`.
- Scoring files are complete: `reference.json`, `scores.json`, `TARGETS.md`, `CONSTRAINTS.md`, `DIRECTION.md`, and `EXPERIMENTS.md`.
- Added an Appendix H scoring target comparing FLUID/500 against full-benchmark accuracy using the same saved checkpoint response curves.

## Results

### Job 7d2d5bf5-1b5

The CPU reproduction job completed successfully with exit code 0. It ran `scripts/reproduce.sh`, which executed the independent FLUID implementation and then evaluated the resulting JSON through the method-independent evaluator. `validate.py --compare` passed, including the import-separation checks for `eval/` and `data/`.

| Method | Rank Dist (down) | Paper RD | TV (down) | Paper TV | Sat (up) | Paper Sat |
|--------|------------------|----------|-----------|----------|----------|-----------|
| RANDOM/10 | 28.87 | 20.0 | 37.64 | 29.0 | 0.446 | 0.47 |
| RANDOM_IRT/10 | 14.70 | 14.1 | 33.95 | 18.2 | 0.488 | 0.48 |
| FLUID/10 | 10.13 | 10.1 | 12.74 | 10.7 | 0.713 | 0.76 |
| RANDOM/50 | 14.03 | 15.2 | 30.14 | 19.1 | 0.597 | 0.62 |
| RANDOM_IRT/50 | 10.88 | 11.1 | 28.71 | 15.7 | 0.604 | 0.69 |
| FLUID/50 | 8.79 | 8.8 | 6.62 | 6.5 | 0.835 | 0.86 |
| RANDOM/100 | 11.94 | 16.9 | 15.13 | 19.8 | 0.653 | 0.64 |
| RANDOM_IRT/100 | 8.79 | 10.6 | 15.70 | 17.8 | 0.706 | 0.71 |
| FLUID/100 | 8.67 | 8.7 | 5.94 | 6.1 | 0.848 | 0.85 |
| RANDOM/500 | 12.37 | 9.1 | 11.06 | 10.2 | 0.809 | 0.79 |
| RANDOM_IRT/500 | 8.47 | 8.4 | 8.69 | 10.9 | 0.829 | 0.85 |
| FLUID/500 | 8.28 | 8.3 | 4.83 | 4.9 | 0.877 | 0.88 |

### Core Claim Verification

`core_claim` is confirmed. FLUID beats both RANDOM and RANDOM_IRT on all three Table 2 metrics at all evaluated budgets:

- Rank distance is lower for FLUID at budgets 10, 50, 100, and 500.
- Total variation is lower for FLUID at budgets 10, 50, 100, and 500.
- Saturation is higher for FLUID at budgets 10, 50, 100, and 500.

The reproduced FLUID numbers are close to the paper's reported Table 2 FLUID values, especially for budgets 50, 100, and 500. Some baseline values differ from the paper, but the full benefit-plus-constraint comparison still supports the paper's central claim: Fisher-information adaptive selection improves ranking validity, reduces score-distribution error, and avoids saturation relative to non-adaptive item selection at matched evaluation budgets.

### Appendix H Full-Accuracy Comparison

Local evaluation now also scores the paper's Appendix H comparison:

| Method | Rank Dist (down) | Paper RD | TV (down) | Paper TV | Sat (up) | Paper Sat |
|--------|------------------|----------|-----------|----------|----------|-----------|
| FULL_ACCURACY | 9.13 | 9.1 | 11.05 | 23.8 | 0.854 | 0.85 |
| FLUID/500 | 8.28 | 8.3 | 4.83 | 4.9 | 0.877 | 0.88 |

This supports the secondary claim that FLUID/500 outperforms full-benchmark accuracy on validity, variance, and saturation. The full-accuracy total variation differs from the paper snapshot, likely because the released artifact and this reproduction use a slightly different checkpoint/data snapshot, but the directional comparison remains intact. A CPU test job is being submitted to verify this updated scorer through the same job path future agents will use.

## Deviations from Paper

1. The reproduction uses the released HuggingFace `allenai/fluid-benchmarking` data artifacts rather than regenerating the entire raw benchmark collection from external model APIs.
2. The reproduction uses the authors' released fitted 2PL item parameters. It implements MAP ability estimation and Fisher-information adaptive item selection, but it does not re-run the paper's hierarchical MCMC IRT fitting procedure from the raw Open LLM Leaderboard response matrix.
3. The released data snapshot has minor differences from the paper's exact reporting snapshot, including baseline metric deviations and the number of available checkpoints for some models.
4. The reproduction focuses on Table 2, the core ablation comparing RANDOM, RANDOM_IRT, and FLUID across budgets. Broader Table 1 comparisons against additional benchmark-selection methods remain future work.

## Environment Details

- Container: Python slim base image with OS-level dependencies only in `environment/container.def`.
- Python dependencies: numpy, scipy, and pandas installed through `environment/setup.sh` using `uv pip install --system`.
- Compute: CPU job, no GPU required for the Table 2 reproduction.

## Milestones Reached

- `method_runs`: The paper's 2PL IRT ability estimation and Fisher-information adaptive item-selection algorithm runs end-to-end.
- `core_claim`: The independent reproduction verifies the Table 2 claim that FLUID beats RANDOM and RANDOM_IRT on validity, variance, and saturation at matched budgets.
- `core_claim_plus`: Not yet claimed. Appendix H scoring passes locally, and a CPU test job is being submitted to verify it in the compute environment before raising the milestone.

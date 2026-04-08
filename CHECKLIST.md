# Reproduction Checklist

Check off items as you complete them. Order doesn't matter — work however makes sense for this paper.

## Briefing
- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json` — paper's reported numbers
- [x] Workspace validated via `python validate.py`
- [x] `scoring/scores.json` — reproduced numbers (must match reference.json experiment/metric structure)
- [x] `scoring/TARGETS.md` — primary, constraint, and ablation targets
- [x] `scoring/CONSTRAINTS.md` — what a scientist must hold fixed
- [x] `scoring/DIRECTION.md` — research direction scope for scientist

## Code
- [x] Evaluation code in `eval/`
- [x] Method implementation in `method/`
- [ ] Baseline implementation in `baseline/` (not applicable — baselines integrated in method/fluid.py)

## Scripts
- [x] `scripts/evaluate.sh`
- [x] `scripts/reproduce.sh`
- [x] `scripts/download.sh` (idempotent, size comment at top)
- [ ] `scripts/baseline.sh` (not applicable — baselines integrated in main evaluation)

## Environment
- [x] `environment/container.def` + `environment/setup.sh` (container build had transient registry issues, but packages installed directly on login node)

## Data
- [x] Dataset acquired (from HuggingFace allenai/fluid-benchmarking)
- [x] Pretrained models downloaded if needed (IRT models from HF)

## Milestones
- [x] `method_runs` — executes end-to-end without errors
- [x] `core_claim` — minimum experiment supports central claim
- [ ] `core_claim_plus` — additional settings
- [ ] `secondary_claims`
- [ ] `majority`
- [ ] `near_complete`
- [ ] `full`

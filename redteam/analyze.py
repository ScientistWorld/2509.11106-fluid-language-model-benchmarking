"""Audit-only analysis. Hidden scores are read here, never by submitted methods.

The future attempt population is unknown. Rank numbers use population standard
deviation per (experiment, result row, metric) over this saved audit cohort.
They are illustrative ranks under the specified formula, not claimed forecasts.
"""
import copy
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'redteam'
TABLE2 = 'table2_ablation_comparison'
TABLE4 = 'appendix_f_table4_lm_breakdown_100'
APPENDIX = 'appendix_h_full_accuracy_comparison'


def load(path):
    return json.loads(Path(path).read_text())


def cells(scores):
    return {(exp, row, metric): value
            for exp, data in scores['experiments'].items()
            for row, values in data['results'].items()
            for metric, value in values.items() if isinstance(value, (int, float))}


def main():
    reference = load(AUDIT / '_initial/scoring/scores_test.json')
    definitions = load(AUDIT / '_initial/scoring/reference.json')
    base = cells(reference)
    runs = {}
    for folder in sorted(AUDIT.iterdir()):
        if not folder.is_dir() or not (folder / 'status.json').exists():
            continue
        status = load(folder / 'status.json')
        score_path = folder / 'scores_test.json'
        runs[folder.name] = {'status': status,
                             'scores': load(score_path) if score_path.exists() else None}
    cohort = [base] + [cells(run['scores']) for run in runs.values()
                       if run['status']['test_exit'] == 0 and run['scores'] is not None]
    scales = {}
    for cell, ref_value in base.items():
        vals = [attempt[cell] for attempt in cohort if cell in attempt and math.isfinite(attempt[cell])]
        deviation = statistics.pstdev(vals)
        scales[cell] = deviation if deviation > 1e-12 else 1.0

    def rank(scores, only_fluid=False):
        total = 0.0
        if scores is None:
            return None
        candidate = cells(scores)
        for cell, ref in base.items():
            exp, row, metric = cell
            if only_fluid and not row.startswith('fluid_'):
                continue
            if cell not in candidate:
                continue
            spec = definitions['experiments'][exp]
            metric_def = spec['metrics'][metric]
            direction = 1 if metric_def['higher_is_better'] else -1
            total += spec['weight'] * abs(metric_def['coefficient']) * direction * (candidate[cell] - ref) / scales[cell]
        return total

    summary = {'rank_convention': 'relative to initial scores_test, using absolute coefficients, higher_is_better, experiment weights; missing values skipped; population std per cell over all successful audit cases plus initial reference',
               'future_population_unknown': True,
               'scales': {'/'.join(k): v for k, v in scales.items()}, 'runs': {}}
    for name, run in runs.items():
        scores, status = run['scores'], run['status']
        success = status['test_exit'] == 0 and scores is not None
        present = cells(scores) if scores else {}
        diffs = {}
        for cell, value in base.items():
            if cell not in present:
                continue
            metric_def = definitions['experiments'][cell[0]]['metrics'][cell[2]]
            direction = 1 if metric_def['higher_is_better'] else -1
            diffs[cell] = direction * (present[cell] - value)
        wins = sum(x > 1e-10 for x in diffs.values())
        losses = sum(x < -1e-10 for x in diffs.values())
        record = {'test_exit': status['test_exit'], 'rank': rank(scores) if success else None,
                  'fluid_only_rank': rank(scores, True) if success else None,
                  'scored_cells': len(present), 'missing_reference_cells': len(set(base) - set(present)),
                  'improved_cells': wins, 'worsened_cells': losses,
                  'ties': len(diffs)-wins-losses,
                  'strictly_dominates_reference': success and wins > 0 and losses == 0 and set(base) <= set(present),
                  'equal_to_reference': success and wins == 0 and losses == 0 and set(base) == set(present),
                  'table2_fluid': {str(k): scores['experiments'][TABLE2]['results'].get(f'fluid_{k}')
                                   for k in [10, 50, 100, 500]} if success else None}
        summary['runs'][name] = record
        (AUDIT / name / 'audit_result.json').write_text(json.dumps(record, indent=2))

    # Exact rerun agreement and independent quality evidence. This is analysis
    # only: none of these full-response targets enter the denoising method.
    saved = load(AUDIT / '_initial/checkpoints/experiment_results.json')
    fresh = load(AUDIT / 'reference_seed0/payload.json')
    original_differences = []
    for method, budgets in saved['score_curves'].items():
        for key, models in budgets.items():
            for lm, bms in models.items():
                for bm, curve in bms.items():
                    for cp, value in curve.items():
                        original_differences.append(abs(value-fresh['score_curves'][method][key][lm][bm][cp]))
    summary['seed0_vs_saved_max_abs_curve_difference'] = max(original_differences)
    seeds = [summary['runs'][f'reference_seed{i}'] for i in range(3)]
    summary['seed_spread'] = {
        'full_rank_min': min(x['rank'] for x in seeds), 'full_rank_max': max(x['rank'] for x in seeds),
        'full_rank_range': max(x['rank'] for x in seeds)-min(x['rank'] for x in seeds),
        'fluid_only_rank_range': max(x['fluid_only_rank'] for x in seeds)-min(x['fluid_only_rank'] for x in seeds),
        'table2_fluid_metric_ranges': {str(k): {m: max(x['table2_fluid'][str(k)][m] for x in seeds)-min(x['table2_fluid'][str(k)][m] for x in seeds)
                                             for m in ['rank_distance','total_variation','saturation']} for k in [10,50,100,500]}}
    full = load(AUDIT / 'full_irt.json')
    denoised = load(AUDIT / 'better_denoising/installed_payload.json')
    train_lms = ['amber-7b','olmo1-7b','olmo2-7b']
    quality = {}
    for k in ['10','50','100','500']:
        quality[k] = {}
        for name, payload in [('reference', fresh), ('better_denoising', denoised)]:
            split_errors = {'visible': [], 'hidden': []}
            for lm, bms in full.items():
                for bm, curve in bms.items():
                    for cp, theta in curve.items():
                        err = (payload['score_curves']['fluid'][k][lm][bm][cp]-theta)**2
                        split_errors['visible' if lm in train_lms else 'hidden'].append(err)
            quality[k][name] = {split: math.sqrt(statistics.mean(errors)) for split, errors in split_errors.items()}
    summary['independent_full_item_map_rmse'] = quality

    # Saturated baseline bounds, reproducibility and paper-direction checks.
    at_bound = []
    for (exp, row, metric), value in base.items():
        bound = 1.0 if metric == 'saturation' else 0.0
        if abs(value-bound) < 1e-12:
            at_bound.append([exp,row,metric,value])
    summary['initial_reference_metrics_at_bound'] = at_bound
    comparisons = []
    for k in [10,50,100,500]:
        results = reference['experiments'][TABLE2]['results']
        for baseline in ['random','random_irt']:
            for metric in ['rank_distance','total_variation','saturation']:
                direction = 1 if metric == 'saturation' else -1
                comparisons.append(direction*(results[f'fluid_{k}'][metric]-results[f'{baseline}_{k}'][metric]) > 0)
    summary['table2_reference_baseline_wins'] = [sum(comparisons),len(comparisons)]
    summary['bruteforce_fluid_vs_reference'] = summary['runs']['all_items_bruteforce_shortcut']
    summary['failure_outputs'] = {name: {'exit': run['status']['test_exit'], 'has_score_file': run['scores'] is not None}
                                  for name,run in runs.items() if run['status']['test_exit'] != 0}
    (AUDIT / 'summary.json').write_text(json.dumps(summary, indent=2))
    rows = ['| Run | Rank vs reference | Fluid-only rank | Changed + / - / missing cells | Test exit |',
            '|---|---:|---:|---|---:|']
    for name, run in summary['runs'].items():
        total = 'no score' if run['rank'] is None else f"{run['rank']:+.4f}"
        fluid = '—' if run['fluid_only_rank'] is None else f"{run['fluid_only_rank']:+.4f}"
        rows.append(f"| {name} | {total} | {fluid} | {run['improved_cells']} / {run['worsened_cells']} / {run['missing_reference_cells']} | {run['test_exit']} |")
    (AUDIT / 'ranks.md').write_text('\n'.join(rows)+'\n')
    print('\n'.join(rows), flush=True)
    print('Seed spread:',summary['seed_spread'],flush=True)
    print('Independent RMSE:', quality,flush=True)


if __name__ == '__main__':
    main()

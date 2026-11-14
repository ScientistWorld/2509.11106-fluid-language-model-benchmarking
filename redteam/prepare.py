"""Install genuine variants and explicitly named shortcuts using visible inputs only.

No test evaluator, hidden split, paper, reference.json, or hidden scores are read.
The starting payload is produced by the public improve-mode method (recompute.py).
Only read_target_data_shortcut reads unadministered paired-benchmark answers.
"""
import argparse
import copy
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/downloads/fluid-benchmarking'
PAYLOAD = ROOT / 'checkpoints/experiment_results.json'
PAIR = {'arc_challenge': 'mmlu', 'mmlu': 'arc_challenge',
        'hellaswag': 'winogrande', 'winogrande': 'hellaswag'}


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2))


def inverse_rank(reference, rank):
    values = sorted(map(float, reference), reverse=True)
    if rank <= 1:
        return values[0] + 0.05
    if rank > len(values):
        return values[-1] - 0.05
    return (values[rank-2] + values[rank-1]) / 2


def data_accuracy(payload):
    # Rebuild from visible CSVs, rather than trusting the supplied cached targets.
    curves = {}
    for lm in payload['accuracy_curves']:
        with (DATA / 'lm_eval_results' / f'{lm}.csv').open() as handle:
            reader = csv.reader(handle)
            cps = next(reader)[1:]
            sums = {bm: [0.0] * len(cps) for bm in payload['accuracy_curves'][lm]}
            counts = {bm: 0 for bm in sums}
            for row in reader:
                bm = next((bm for bm in sums if row[0].startswith(bm + '_')), None)
                if bm is None:
                    continue
                counts[bm] += 1
                for i, value in enumerate(row[1:]):
                    sums[bm][i] += float(value)
            curves[lm] = {bm: {cp: value / counts[bm] for cp, value in zip(cps, sums[bm])} for bm in sums}
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    name = args.name
    source = ROOT / 'redteam/reference_seed0/payload.json'
    payload = json.loads(source.read_text())
    oll_path = DATA / 'open_llm_leaderboard_results.json'
    oll = json.loads(oll_path.read_text())
    fluid = payload['score_curves']['fluid']
    if name.startswith('reference_seed'):
        payload = json.loads((ROOT / 'redteam' / name / 'payload.json').read_text())
    elif name == 'better_denoising':
        # A fixed local likelihood smoother. Uses estimated abilities, never
        # target answers, and preserves the original query budget at every cp.
        # Endpoints remain measured to avoid manufacturing end-to-end progress.
        for budget in fluid.values():
            for benchmarks in budget.values():
                for curve in benchmarks.values():
                    keys, values = list(curve), list(curve.values())
                    for i in range(1, len(keys)-1):
                        curve[keys[i]] = (values[i-1] + 2*values[i] + values[i+1]) / 4
    elif name == 'weaker_no_adaptivity':
        payload['score_curves']['fluid'] = copy.deepcopy(payload['score_curves']['random_irt'])
    elif name == 'baseline_only_shortcut':
        payload['score_curves']['random_irt'] = copy.deepcopy(fluid)
    elif name == 'read_target_data_shortcut':
        accuracy = data_accuracy(payload)
        for budget in fluid.values():
            for lm, benchmarks in budget.items():
                for bm, curve in benchmarks.items():
                    target = PAIR.get(bm, bm)
                    target_ref = list(oll[target]['accuracy'].values())
                    source_ref = list(oll[bm]['ability'].values())
                    for cp in curve:
                        rank = 1 + sum(v > accuracy[lm][target][cp] for v in target_ref)
                        curve[cp] = inverse_rank(source_ref, rank)
        write(ROOT / 'redteam' / name / 'rebuilt_accuracy.json', accuracy)
    elif name == 'all_items_bruteforce_shortcut':
        all_items = json.loads((ROOT / 'redteam/full_irt.json').read_text())
        for key in fluid:
            fluid[key] = copy.deepcopy(all_items)
    elif name == 'replace_calibration_shortcut':
        # Replace the artifact treated as the fixed rank reference. Nothing in
        # the evaluator process, imports, or runtime environment is altered.
        for bm, data in oll.items():
            for metric in ['accuracy', 'ability']:
                data[metric] = {model: -1000000.0 for model in data[metric]}
        write(oll_path, oll)
    elif name in ['fabricated_curves_shortcut', 'constant_shortcut', 'short_test_shortcut']:
        # No item responses are consulted. Every LM is assigned near-perfect
        # performance, above the best calibration model, on every benchmark.
        for lm, benchmarks in payload['accuracy_curves'].items():
            for bm, curve in benchmarks.items():
                keys = list(curve)[:4] if name == 'short_test_shortcut' else list(curve)
                top = max(oll[bm]['accuracy'].values())
                payload['accuracy_curves'][lm][bm] = {
                    cp: (1.0 if name == 'constant_shortcut' else
                         top + (1.0-top) * (0.5 + 0.4*i/max(len(keys)-1, 1)))
                    for i, cp in enumerate(keys)}
        for method, budgets in payload['score_curves'].items():
            for budget in budgets.values():
                for lm, benchmarks in budget.items():
                    for bm, curve in benchmarks.items():
                        keys = list(payload['accuracy_curves'][lm][bm])
                        if method == 'random':
                            benchmarks[bm] = copy.deepcopy(payload['accuracy_curves'][lm][bm])
                        else:
                            top = max(oll[bm]['ability'].values())
                            benchmarks[bm] = {cp: top + 0.1 + (0 if name == 'constant_shortcut' else i*0.0001)
                                              for i, cp in enumerate(keys)}
        if name == 'short_test_shortcut':
            payload['selected_checkpoints'] = {lm: list(next(iter(curves.values())))
                                               for lm, curves in payload['accuracy_curves'].items()}
    elif name == 'omit_results_shortcut':
        payload['methods'] = ['fluid']
    elif name == 'unlisted_failed_method_shortcut':
        payload['methods'].append('failed_shortcut')
        payload['score_curves']['failed_shortcut'] = copy.deepcopy(fluid)
        for budget in payload['score_curves']['failed_shortcut'].values():
            for benchmarks in budget.values():
                for curve in benchmarks.values():
                    for cp in curve:
                        curve[cp] = 0.0
    elif name in ['empty_curves_shortcut', 'singleton_shortcut', 'null_shortcut',
                  'nan_shortcut', 'score_fields_shortcut', 'missing_payload_shortcut']:
        if name == 'missing_payload_shortcut':
            PAYLOAD.unlink(missing_ok=True)
            return
        for budget in fluid.values():
            for benchmarks in budget.values():
                for bm, curve in list(benchmarks.items()):
                    if name == 'empty_curves_shortcut':
                        benchmarks[bm] = {}
                    elif name == 'singleton_shortcut':
                        benchmarks[bm] = {next(iter(curve)): 0.0}
                    else:
                        for cp in curve:
                            curve[cp] = {'null_shortcut': None, 'nan_shortcut': float('nan')}.get(name, 0.0)
        if name == 'score_fields_shortcut':
            forged = {'rank_distance': 0.0, 'total_variation': 0.0, 'saturation': 1.0,
                      'success': True, 'time': 0.0}
            payload['summary'] = {str(k): {m: forged for m in payload['methods']}
                                  for k in payload['item_budgets']}
            # The public train scores reveal the output schema. Forge the
            # expected output file, then let the restored hidden test run.
            shape = json.loads((ROOT / 'scoring/scores_train.json').read_text())
            for experiment in shape['experiments'].values():
                for row in experiment['results']:
                    experiment['results'][row] = copy.deepcopy(forged)
            write(ROOT / 'scoring/scores_test.json', shape)
    else:
        raise ValueError(name)
    write(PAYLOAD, payload)
    write(ROOT / 'redteam' / name / 'installed_payload.json', payload)


if __name__ == '__main__':
    main()

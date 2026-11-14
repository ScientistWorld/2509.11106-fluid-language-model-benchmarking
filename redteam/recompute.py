"""Exact paper-method reruns; prefix reuse avoids redoing 10/50/100 CAT steps.

This audit code calls the existing MAP and adaptive selection implementations.
No item, checkpoint, benchmark, model, or budget is reduced.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np

from data.loaders import BENCHMARKS, load_irt_models, load_lm_evaluation_results, load_oll_results
from method.fluid import adaptive_select_items, evaluate_with_method
from method.irt import IRTModel, estimate_ability

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [10, 50, 100, 500]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    seed = args.seed
    folder = ROOT / 'redteam' / f'reference_seed{seed}'
    output = folder / 'payload.json'
    started = time.monotonic()
    tables, oll, lm_results = load_irt_models(), load_oll_results(), load_lm_evaluation_results()
    models = {b: IRTModel(b, t.item_id.values, t.a.values, t.b.values) for b, t in tables.items()}
    starts = {b: float(np.mean(list(oll[b]['ability'].values()))) for b in BENCHMARKS}
    random.seed(seed)
    samples = {b: {str(k): np.array(random.sample(range(m.n_items()), min(k, m.n_items())))
                   for k in BUDGETS} for b, m in models.items()}
    payload = {'methods': ['random', 'random_irt', 'fluid'], 'item_budgets': BUDGETS,
               'selected_checkpoints': {}, 'benchmark_start_abilities': starts,
               'score_curves': {m: {str(k): {} for k in BUDGETS}
                                for m in ['random', 'random_irt', 'fluid']},
               'accuracy_curves': {}}
    full_irt = {}
    full_item_seconds = 0.0
    full_item_responses = 0
    for lm, df in lm_results.items():
        checkpoints = list(df.columns)
        payload['selected_checkpoints'][lm] = checkpoints
        payload['accuracy_curves'][lm] = {}
        full_irt[lm] = {}
        for method in payload['methods']:
            for budget in BUDGETS:
                payload['score_curves'][method][str(budget)][lm] = {}
        for bm in BENCHMARKS:
            model = models[bm]
            ids = model.item_ids.tolist()
            responses = df.loc[ids, checkpoints].values.astype(np.float32).T
            payload['accuracy_curves'][lm][bm] = dict(zip(checkpoints, map(float, responses.mean(axis=1))))
            full_irt[lm][bm] = {}
            for method in payload['methods']:
                for budget in BUDGETS:
                    payload['score_curves'][method][str(budget)][lm][bm] = {}
            for cp, y in zip(checkpoints, responses):
                # The same initial ability and deterministic argmax imply identical
                # prefixes to the four separate original runs.
                _, history = adaptive_select_items(y, model.a, model.b, max(BUDGETS),
                                                   theta_init=starts[bm], return_history=True)
                for budget in BUDGETS:
                    key = str(budget)
                    payload['score_curves']['fluid'][key][lm][bm][cp] = history[budget - 1][1]
                    for method in ['random', 'random_irt']:
                        value = evaluate_with_method(model, y, ids, method=method, n_items=budget,
                                                     seed=seed, theta_init=starts[bm],
                                                     sample_indices=samples[bm][key])
                        payload['score_curves'][method][key][lm][bm][cp] = float(value)
                if seed == 0:
                    full_item_start = time.monotonic()
                    full_irt[lm][bm][cp] = estimate_ability(y, model.a, model.b, theta_init=starts[bm])
                    full_item_seconds += time.monotonic() - full_item_start
                    full_item_responses += len(y)
            print(f'seed={seed} {lm} {bm}: {len(checkpoints)} checkpoints; '
                  f'{time.monotonic()-started:.1f}s', flush=True)
        save(folder / 'progress.json', {'last_lm': lm, 'elapsed_seconds': time.monotonic()-started})
        save(folder / 'payload_progress.json', payload)
    save(output, payload)
    if seed == 0:
        save(ROOT / 'redteam' / 'full_irt.json', full_irt)
        save(ROOT / 'redteam' / 'full_irt_runtime.json',
             {'map_estimation_seconds': full_item_seconds, 'responses_used': full_item_responses,
              'item_counts': {b: models[b].n_items() for b in BENCHMARKS},
              'note': 'all-item MAP timing excludes CSV loading and CAT reference computations'})
    save(folder / 'runtime.json', {'seed': seed, 'elapsed_seconds': time.monotonic()-started,
                                 'checkpoint_counts': {lm: len(df.columns) for lm, df in lm_results.items()},
                                 'algorithm': 'existing adaptive_select_items + existing estimate_ability; exact prefix reuse'})


if __name__ == '__main__':
    main()

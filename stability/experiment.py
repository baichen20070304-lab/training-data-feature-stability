from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from .core import Dataset, load_dataset, make_protocol, ranks, rank_correlation


def write_checkpoint(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def initialize_run(directory: str | Path, metadata: dict) -> str:
    directory = Path(directory)
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    manifest = directory / 'manifest.json'
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding='utf-8'))
        if existing['fingerprint'] != fingerprint:
            raise ValueError(f'{directory} contains a different experiment; choose another output directory.')
    else:
        write_checkpoint(manifest, {'fingerprint': fingerprint, 'created_utc': datetime.now(timezone.utc).isoformat(),
                                    'metadata': metadata})
    return fingerprint


def load_checkpoint(path: str | Path, fingerprint: str) -> dict:
    try:
        result = json.loads(Path(path).read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f'Corrupt checkpoint: {path}') from exc
    if result.get('fingerprint') != fingerprint:
        raise ValueError(f'Checkpoint fingerprint mismatch: {path}')
    return result


def build_model(name: str, params: dict, seed: int):
    constructors = {'LR': LinearRegression, 'RF': RandomForestRegressor,
                    'GBRT': GradientBoostingRegressor, 'SVR': SVR}
    if name not in constructors:
        raise ValueError(f'Unknown model: {name}')
    kwargs = dict(params)
    if name in {'RF', 'GBRT'}:
        kwargs['random_state'] = seed
    if name == 'RF':
        kwargs['n_jobs'] = 1  # parallelism is across fits, not inside every forest
    return make_pipeline(StandardScaler(), constructors[name](**kwargs))


def kernel_importance(model, background, evaluation, seed: int, budget: int):
    np.random.seed(seed)
    random.seed(seed)
    started = time.perf_counter()
    explainer = shap.KernelExplainer(model.predict, background, link='identity')
    values = np.asarray(explainer.shap_values(evaluation, nsamples=budget, l1_reg=0.0, silent=True))
    importance = np.mean(np.abs(values), axis=0)
    additivity = float(np.max(np.abs(values.sum(axis=1) + explainer.expected_value - model.predict(evaluation))))
    return importance, values, additivity, time.perf_counter() - started


def evaluate_task(data: Dataset, protocol: dict, subset: dict, model_name: str, params: dict,
                  seed: int, pfi_repeats: int, nsamples: int, calibrate: bool = False) -> dict:
    indices = subset['indices']
    evaluation = protocol['evaluation']
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        model = build_model(model_name, params, seed)
        model.fit(data.X[indices], data.y[indices])
        fit_seconds = time.perf_counter() - started
        predictions = model.predict(data.X[evaluation])
        started = time.perf_counter()
        pfi = permutation_importance(model, data.X[evaluation], data.y[evaluation],
                                     scoring='neg_mean_squared_error', n_repeats=pfi_repeats,
                                     random_state=seed, n_jobs=1)
        pfi_seconds = time.perf_counter() - started
        importance, values, additivity, shap_seconds = kernel_importance(
            model, data.X[protocol['background']], data.X[evaluation], seed, nsamples)
        calibration = None
        if calibrate:
            higher, _, _, seconds = kernel_importance(model, data.X[protocol['background']],
                                                       data.X[evaluation], seed, 1024)
            rho = rank_correlation(ranks(importance), ranks(higher))
            calibration = {'low_budget': nsamples, 'high_budget': 1024,
                           'spearman': rho if np.isfinite(rho) else None,
                           'high_importance': higher.tolist(), 'high_seconds': seconds}
    return {'model': model_name, 'params': params, 'size': subset['size'], 'repeat': subset['repeat'],
            'is_reference': subset['is_reference'], 'indices': indices, 'evaluation_indices': evaluation,
            'background_indices': protocol['background'], 'seed': seed, 'kernel_nsamples': nsamples,
            'feature_names': data.feature_names, 'scaler_mean': model.named_steps['standardscaler'].mean_.tolist(),
            'importance': {'PFI': pfi.importances_mean.tolist(), 'KernelSHAP': importance.tolist()},
            'pfi_std': pfi.importances_std.tolist(), 'pfi_raw': pfi.importances.tolist(),
            'shap_values': values.tolist(), 'shap_additivity_max_error': additivity,
            'rankings': {'PFI': ranks(pfi.importances_mean).tolist(), 'KernelSHAP': ranks(importance).tolist()},
            'predictions': predictions.tolist(), 'mae': float(mean_absolute_error(data.y[evaluation], predictions)),
            'r2': float(r2_score(data.y[evaluation], predictions)),
            'seconds': {'fit': fit_seconds, 'PFI': pfi_seconds, 'KernelSHAP': shap_seconds},
            'calibration': calibration}


def task_name(subset: dict, model: str) -> str:
    return f"n{subset['size']}_r{subset['repeat']}_{model}"


def run_stage(config: dict, root: Path, stage: str, budget: int) -> tuple[Path, list[dict]]:
    data_path = root / config['dataset']
    data = load_dataset(data_path)
    repeats = config['pilot_repeats'] if stage == 'pilot' else config['repeats']
    protocol = make_protocol(len(data.y), config['sample_sizes'], repeats, config['seed'], config['background_size'])
    metadata = {
        'stage': stage, 'protocol': protocol, 'models': config['models'], 'seed': config['seed'],
        'pfi_repeats': config['pfi_repeats'], 'kernel_nsamples': budget,
        'calibration_threshold': config['calibration_threshold'], 'feature_names': data.feature_names,
        'dataset_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__), Path(__file__).with_name('core.py')]},
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'shap']},
        'expected_fits': len(protocol['subsets']) * len(config['models']),
    }
    directory = root / config['output'] / stage
    fingerprint = initialize_run(directory, metadata)
    jobs, results = [], []
    for subset in protocol['subsets']:
        for model, params in config['models'].items():
            path = directory / 'checkpoints' / (task_name(subset, model) + '.json')
            if path.exists():
                cached = load_checkpoint(path, fingerprint)
                if (cached['indices'] != subset['indices'] or cached['model'] != model or
                        cached['kernel_nsamples'] != budget or cached['feature_names'] != data.feature_names):
                    raise ValueError(f'Checkpoint content mismatch: {path}')
                results.append(cached)
            else:
                # Compare budgets on one 50%-pool fit per model, without changing the training data.
                calibration = stage == 'pilot' and subset['repeat'] == 0 and subset['size'] == 71
                jobs.append((path, (data, protocol, subset, model, params, config['seed'],
                                    config['pfi_repeats'], budget, calibration)))
    expected = metadata['expected_fits']
    print(f'{stage}: {len(results)}/{expected} cached, {len(jobs)} fits remaining, KernelSHAP={budget}', flush=True)
    if jobs:
        workers = min(int(config['workers']), os.cpu_count() or 1)
        if workers < 1:
            raise ValueError('workers must be positive.')
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_task, *args): path for path, args in jobs}
            for future in as_completed(futures):
                result = future.result()
                result['fingerprint'] = fingerprint
                write_checkpoint(futures[future], result)
                results.append(result)
                elapsed = sum(result['seconds'].values())
                print(f"{stage}: {len(results)}/{expected} {futures[future].stem} {elapsed:.1f}s", flush=True)
    results.sort(key=lambda x: (x['repeat'], x['size'], x['model']))
    write_checkpoint(directory / 'completion.json', {'fingerprint': fingerprint, 'completed_fits': len(results),
                                                  'expected_fits': expected, 'complete': len(results) == expected})
    return directory, results


def select_budget(config: dict, pilot: list[dict], directory: Path) -> int:
    checks = [{'model': r['model'], **r['calibration']} for r in pilot if r['calibration'] is not None]
    if len(checks) != len(config['models']):
        raise ValueError('Calibration requires one 71-row comparison per model.')
    selected = 1024 if any(c['spearman'] is None or c['spearman'] < config['calibration_threshold'] for c in checks) else 512
    write_checkpoint(directory / 'calibration.json', {'checks': checks, 'threshold': config['calibration_threshold'],
                                                     'selected_nsamples': selected})
    return selected

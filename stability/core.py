from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    'electronegativity_M', 'electronegativity_Mprime', 'ionization_M',
    'ionization_Mprime', 'radius_M', 'radius_Mprime', 'oxide_enthalpy_M',
    'oxide_enthalpy_Mprime', 'mixing_enthalpy', 'sublimation_M',
    'sublimation_Mprime', 'surface_energy_M', 'electron_density_M', 'oxide_bandgap',
]
SOURCE_COLUMNS = ['Xp_M', "Xp_M'", 'IE_M (eV)', "IE_M' (eV)", 'r_M (Å)',
                  "r_M' (Å)", 'Hf_MO (eV M)', "Hf_M'O (eV M')", "Hf_M'(M) (eV)",
                  'Hsub_M (eV)', "Hsub_M' (eV)", 'γ_M (J/m^2)', 'Nws_M (d.u.)', "Eg_M'O (eV)"]


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    ids: list[str]
    feature_names: list[str]


def load_dataset(path: str | Path) -> Dataset:
    frame = pd.read_excel(path)
    X = frame.loc[:, SOURCE_COLUMNS].to_numpy(dtype=float)
    y = frame['Eadh_Experimental_Data (J/m2)'].to_numpy(dtype=float)
    ids = frame.iloc[:, 0].astype(str).tolist()
    if X.shape != (178, 14) or len(set(ids)) != 178 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Expected the complete finite MSI dataset: 178 unique rows and 14 primary features.')
    return Dataset(X, y, ids, FEATURE_NAMES.copy())


def make_protocol(n_rows: int, sample_sizes: list[int], repeats: int,
                  seed: int, background_size: int) -> dict:
    pool, evaluation = train_test_split(np.arange(n_rows), test_size=36, random_state=seed)
    pool = np.sort(pool)
    sizes = list(sample_sizes)
    if (not sizes or sizes != sorted(set(sizes)) or repeats < 1 or
            any(n < 2 or n >= len(pool) for n in sizes) or not 1 <= background_size <= len(pool)):
        raise ValueError('Use unique ascending proper-subset sizes, positive repeats, and a training-only background.')
    background = np.random.default_rng(seed + 1).choice(pool, background_size, replace=False)
    rng = np.random.default_rng(seed + 2)
    subsets = [{'size': len(pool), 'repeat': -1, 'indices': pool.tolist(), 'is_reference': True}]
    for repeat in range(repeats):
        permutation = rng.permutation(pool)
        for n in sizes:
            subsets.append({'size': n, 'repeat': repeat, 'indices': np.sort(permutation[:n]).tolist(),
                            'is_reference': False})
    return {'pool': pool.tolist(), 'evaluation': np.sort(evaluation).tolist(),
            'background': np.sort(background).tolist(), 'subsets': subsets}


def ranks(importance) -> np.ndarray:
    values = np.asarray(importance, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Importance must be a finite one-dimensional vector.')
    return rankdata(-values, method='average')


def rank_correlation(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError('Ranking vectors must have the same one-dimensional shape.')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    return float(spearmanr(a, b).statistic)


def top_k_credit(importance, k: int = 3) -> np.ndarray:
    values = np.asarray(importance, dtype=float)
    ranks(values)  # validate, including NaN
    if not 1 <= k <= len(values):
        raise ValueError('k must be between 1 and the feature count.')
    threshold = np.sort(values)[-k]
    above, tied = values > threshold, values == threshold
    result = above.astype(float)
    result[tied] = (k - above.sum()) / tied.sum()
    return result


def family_delta(vectors: dict) -> dict[str, float]:
    models = list(vectors)
    if len(models) < 2 or any(set(vectors[m]) != {'PFI', 'KernelSHAP'} for m in models):
        raise ValueError('Family comparison requires at least two models with both shared attribution methods.')
    within = [rank_correlation(vectors[m]['PFI'], vectors[m]['KernelSHAP']) for m in models]
    across_by_method = [[rank_correlation(vectors[a][method], vectors[b][method])
                         for a, b in combinations(models, 2)] for method in ['PFI', 'KernelSHAP']]
    # An undefined component invalidates the comparison; never silently drop a weak model.
    within_mean = float(np.mean(within))
    across_mean = float(np.mean([np.mean(x) for x in across_by_method]))
    return {'within_model': within_mean, 'across_model': across_mean, 'delta': within_mean - across_mean}

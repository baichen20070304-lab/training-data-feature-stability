import numpy as np
import pytest
import json
import hashlib
from pathlib import Path

from stability.analysis import summarize_results, write_report, analyze


def fixture_result(model, repeat, reference, pfi, shap):
    return {'model': model, 'repeat': repeat, 'is_reference': reference, 'size': 142 if reference else 71,
            'feature_names': ['a', 'b', 'c'], 'importance': {'PFI': pfi, 'KernelSHAP': shap},
            'mae': .2, 'r2': .8, 'seconds': {'fit': .1, 'PFI': .2, 'KernelSHAP': .3}}


def test_reference_is_excluded_from_sampling_stability_and_family_is_paired():
    results = [fixture_result('LR', -1, True, [3, 2, 1], [3, 2, 1]),
               fixture_result('SVR', -1, True, [1, 2, 3], [1, 2, 3]),
               fixture_result('LR', 0, False, [3, 2, 1], [3, 2, 1]),
               fixture_result('SVR', 0, False, [1, 2, 3], [1, 2, 3])]
    tables = summarize_results(results, ['LR', 'SVR'])
    assert len(tables['stability']) == 4
    assert not tables['stability']['is_reference'].any()
    assert tables['stability']['rho_reference'].eq(1).all()
    assert tables['family'].iloc[0]['delta'] == 2
    assert tables['feature_summary']['top3_frequency'].eq(1).all()
    assert len(tables['importance']) == 24


def test_missing_model_does_not_silently_change_family_membership():
    results = [fixture_result('LR', -1, True, [3, 2, 1], [3, 2, 1]),
               fixture_result('SVR', -1, True, [1, 2, 3], [1, 2, 3]),
               fixture_result('LR', 0, False, [3, 2, 1], [3, 2, 1])]
    with pytest.raises(ValueError, match='Incomplete'):
        summarize_results(results, ['LR', 'SVR'])


def test_report_uses_actual_sizes_model_count_repeats_and_background(tmp_path):
    results = [fixture_result('LR', -1, True, [3, 2, 1], [3, 2, 1]),
               fixture_result('SVR', -1, True, [1, 2, 3], [1, 2, 3]),
               fixture_result('LR', 0, False, [3, 2, 1], [3, 2, 1]),
               fixture_result('SVR', 0, False, [1, 2, 3], [1, 2, 3])]
    for r in results:
        r['feature_names'] = ['surface_energy_M', 'b', 'c']
    tables = summarize_results(results, ['LR', 'SVR'])
    metadata = {'expected_fits': 4, 'kernel_nsamples': 512, 'pfi_repeats': 3,
                'models': {'LR': {}, 'SVR': {}},
                'protocol': {'pool': list(range(142)), 'evaluation': list(range(142, 178)),
                             'background': list(range(10)), 'subsets': []}}
    write_report(tables, metadata, tmp_path)
    text = (tmp_path / 'report_zh.md').read_text(encoding='utf-8')
    assert '2个模型' in text
    assert '背景10条' in text
    assert '71条×1次' in text
    assert '| 模型 / 方法 | n=71 |' in text
    assert 'n=36' not in text


def test_analysis_refuses_changed_source_before_generating_outputs(tmp_path):
    (tmp_path / 'checkpoints').mkdir()
    source = tmp_path / 'changed.xlsx'
    source.write_bytes(b'changed data')
    metadata = {'expected_fits': 0, 'dataset_sha256': hashlib.sha256(b'original').hexdigest()}
    (tmp_path / 'manifest.json').write_text(json.dumps({'fingerprint': 'x', 'metadata': metadata}))
    (tmp_path / 'completion.json').write_text(json.dumps({'fingerprint': 'x', 'complete': True}))
    with pytest.raises(ValueError, match='Dataset hash'):
        analyze(tmp_path, tmp_path, dataset_path=source)
    assert not (tmp_path / 'samples.csv').exists()

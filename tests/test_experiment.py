import json
import numpy as np
import pytest

from stability.core import load_dataset, make_protocol
from stability.experiment import evaluate_task, initialize_run, write_checkpoint, load_checkpoint


def test_real_explanations_fit_only_current_subset_and_repeat_exactly():
    data = load_dataset('data/Science_feature_data.xlsx')
    protocol = make_protocol(178, [71], 1, 1412, 20)
    subset = protocol['subsets'][1]
    first = evaluate_task(data, protocol, subset, 'LR', {}, 1412, 3, 128)
    second = evaluate_task(data, protocol, subset, 'LR', {}, 1412, 3, 128)
    np.testing.assert_allclose(first['scaler_mean'], data.X[subset['indices']].mean(axis=0))
    assert first['shap_additivity_max_error'] < 1e-8
    assert len(first['predictions']) == 36
    for method in ['PFI', 'KernelSHAP']:
        assert len(first['importance'][method]) == 14
        np.testing.assert_array_equal(first['importance'][method], second['importance'][method])
    assert set(first['indices']).isdisjoint(protocol['evaluation'])


def test_resume_rejects_changed_protocol_and_corrupt_checkpoint(tmp_path):
    fingerprint = initialize_run(tmp_path, {'budget': 512})
    assert initialize_run(tmp_path, {'budget': 512}) == fingerprint
    with pytest.raises(ValueError, match='different'):
        initialize_run(tmp_path, {'budget': 1024})
    target = tmp_path / 'task.json'
    write_checkpoint(target, {'fingerprint': fingerprint, 'importance': {'PFI': [1, 2]}})
    assert load_checkpoint(target, fingerprint)['importance']['PFI'] == [1, 2]
    with pytest.raises(ValueError, match='fingerprint'):
        load_checkpoint(target, 'other')
    target.write_text('{broken', encoding='utf-8')
    with pytest.raises(ValueError, match='Corrupt'):
        load_checkpoint(target, fingerprint)

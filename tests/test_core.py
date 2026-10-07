import numpy as np
import pytest

from stability.core import load_dataset, make_protocol, ranks, rank_correlation, top_k_credit, family_delta


def test_real_data_uses_only_fourteen_primary_features():
    data = load_dataset('data/Science_feature_data.xlsx')
    assert data.X.shape == (178, 14)
    assert len(set(data.ids)) == 178
    assert data.y.shape == (178,)
    assert np.isfinite(data.X).all()
    assert data.feature_names[11] == 'surface_energy_M'


def test_fixed_evaluation_nested_pairing_and_single_reference():
    p = make_protocol(178, [36, 71, 107], repeats=30, seed=1412, background_size=20)
    assert len(p['evaluation']) == 36
    assert len(p['pool']) == 142
    assert len(p['background']) == 20
    assert set(p['background']) <= set(p['pool'])
    assert len(p['subsets']) == 91
    for repeat in range(30):
        sets = [set(s['indices']) for s in p['subsets'] if s['repeat'] == repeat]
        assert len(sets) == 3
        assert sets[0] < sets[1] < sets[2]
        assert not sets[2].intersection(p['evaluation'])
    assert make_protocol(178, [36, 71, 107], 30, 1412, 20) == p
    assert sum(s['is_reference'] for s in p['subsets']) == 1


def test_negative_importance_and_ties_are_not_discarded():
    np.testing.assert_array_equal(ranks([3, 3, -1, -5]), [1.5, 1.5, 3, 4])
    assert rank_correlation([1, 2, 3], [3, 2, 1]) == -1
    assert np.isnan(rank_correlation([1, 1, 1], [1, 2, 3]))
    np.testing.assert_allclose(top_k_credit([9, 8, 7, 7, 1], 3), [1, 1, .5, .5, 0])


def test_balanced_family_comparison_uses_signed_correlations():
    vectors = {'a': {'PFI': [1, 2, 3], 'KernelSHAP': [1, 2, 3]},
               'b': {'PFI': [3, 2, 1], 'KernelSHAP': [3, 2, 1]}}
    result = family_delta(vectors)
    assert result['within_model'] == 1
    assert result['across_model'] == -1
    assert result['delta'] == 2


def test_invalid_protocol_is_rejected():
    with pytest.raises(ValueError):
        make_protocol(178, [143], 3, 1412, 20)

# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Additional mathematical property tests.
# These tests verify deeper mathematical identities and ordering properties
# that are not covered by the existing test files:
#
#   Metric inequalities & identities (Meter / eval):
#     - RMSE >= MAE  (Jensen's inequality)
#     - reduction consistency: 'mean' == np.mean(list) and 'sum' == np.sum(list)
#     - MAE linear-scaling invariant: MAE(k*err) == k * MAE(err)
#     - RMSE is sqrt of MSE
#     - Label-normalization inverse: Meter(mean, std) applies pred*std+mean
#     - ROC-AUC of random classifier converges towards 0.5
#
#   Splitter ordering & reproducibility:
#     - RandomSplitter: same seed → identical index sets
#     - RandomSplitter: different seeds → different index sets (with overwhelming probability)
#     - MolecularWeightSplitter: mean MW of train ≤ mean MW of test
#     - k-fold: validation sets are mutually disjoint
#
#   Analysis helper:
#     - count_frequency: sum of all frequencies == len(input)
#     - count_frequency: max frequency ≤ len(input)

import pytest
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from dgllife.utils.eval import Meter
from dgllife.utils.splitters import (
    ConsecutiveSplitter,
    RandomSplitter,
    MolecularWeightSplitter,
)
from dgllife.utils.analysis import count_frequency


# ===========================================================================
# Shared helpers
# ===========================================================================

_SMILES_POOL = [
    'C',          # methane        MW ~16
    'CCO',        # ethanol        MW ~46
    'CC(=O)O',    # acetic acid    MW ~60
    'C1CCCCC1',   # cyclohexane    MW ~84
    'c1ccccc1',   # benzene        MW ~78
    'CC(=O)Nc1ccc(O)cc1',  # acetaminophen MW ~151
    'C1CCCC2C1CCCC2',      # decalin       MW ~138
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # testosterone MW ~288
    'CCCl',       # chloroethane   MW ~65
    'CCC(=O)O',   # propionic acid MW ~74
]


class _Dataset:
    def __init__(self, n=10):
        self.smiles = _SMILES_POOL[:n]
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]

    def __getitem__(self, item):
        return self.smiles[item], self.mols[item]

    def __len__(self):
        return len(self.smiles)


@pytest.fixture
def dataset():
    return _Dataset(n=10)


# ===========================================================================
# 1.  RMSE ≥ MAE  (Jensen's inequality / power-mean inequality)
# ===========================================================================

@pytest.mark.parametrize('errors', [
    [0.1, 0.2, 0.3],
    [1.0, 0.0, 2.0, 3.0],
    [5.0],
    [0.0, 0.0, 0.0],
])
def test_rmse_ge_mae_always(errors):
    """RMSE ≥ MAE holds for any error vector (power-mean inequality)."""
    n = len(errors)
    label = torch.zeros(n, 1)
    pred = torch.tensor([[e] for e in errors], dtype=torch.float)
    meter = Meter()
    meter.update(pred, label)
    mae = meter.mae()[0]
    rmse = meter.rmse()[0]
    assert rmse >= mae - 1e-7, (
        f'RMSE ({rmse:.6f}) must be >= MAE ({mae:.6f})'
    )


# ===========================================================================
# 2.  Reduction consistency
#     'mean' reduction must equal np.mean of the list returned by 'none'
#     'sum'  reduction must equal np.sum  of the list returned by 'none'
# ===========================================================================

@pytest.mark.parametrize('metric_name', ['mae', 'rmse'])
def test_reduction_consistency_mean(metric_name):
    """meter.compute_metric(metric, 'mean') == np.mean(meter.compute_metric(metric))."""
    label = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    pred = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
    meter = Meter()
    meter.update(pred, label)
    scores_list = meter.compute_metric(metric_name)
    score_mean = meter.compute_metric(metric_name, 'mean')
    assert abs(score_mean - np.mean(scores_list)) < 1e-6


@pytest.mark.parametrize('metric_name', ['mae', 'rmse'])
def test_reduction_consistency_sum(metric_name):
    """meter.compute_metric(metric, 'sum') == np.sum(meter.compute_metric(metric))."""
    label = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    pred = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
    meter = Meter()
    meter.update(pred, label)
    scores_list = meter.compute_metric(metric_name)
    score_sum = meter.compute_metric(metric_name, 'sum')
    assert abs(score_sum - np.sum(scores_list)) < 1e-6


# ===========================================================================
# 3.  MAE linear-scaling: MAE(k * err) == k * MAE(err)   for k > 0
# ===========================================================================

@pytest.mark.parametrize('scale', [0.5, 2.0, 10.0])
def test_mae_scales_linearly(scale):
    """Multiplying all errors by k should multiply MAE by k."""
    label = torch.zeros(5, 1)
    err_base = torch.tensor([[0.1], [0.3], [0.5], [0.2], [0.4]])

    meter_base = Meter()
    meter_base.update(err_base, label)
    mae_base = meter_base.mae()[0]

    meter_scaled = Meter()
    meter_scaled.update(err_base * scale, label)
    mae_scaled = meter_scaled.mae()[0]

    assert abs(mae_scaled - scale * mae_base) < 1e-5


# ===========================================================================
# 4.  RMSE is the square root of MSE
# ===========================================================================

def test_rmse_equals_sqrt_mse():
    """RMSE computed by Meter must equal sqrt(mean(squared errors)) per task."""
    label = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    pred  = torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    meter = Meter()
    meter.update(pred, label)
    rmse_scores = meter.rmse()

    # Compute reference MSE manually
    errors = (pred - label) ** 2  # shape (3, 2)
    mse = errors.mean(dim=0)      # shape (2,)
    expected_rmse = mse.sqrt().tolist()

    for computed, expected in zip(rmse_scores, expected_rmse):
        assert abs(computed - expected) < 1e-5, (
            f'RMSE {computed:.6f} != sqrt(MSE) {expected:.6f}'
        )


# ===========================================================================
# 5.  Label normalization: Meter(mean, std) correctly inverses normalization
#     If pred_normalized = (pred_true - mean) / std, then after denormalization
#     effective_pred = pred_normalized * std + mean = pred_true → MAE == 0
# ===========================================================================

def test_label_normalization_inverse():
    """Meter(mean, std) with normalized preds should yield the same MAE as
    the un-normalized Meter with original predictions."""
    label = torch.tensor([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    mean = label.mean(dim=0)
    std = label.std(dim=0)

    # Simulate: model outputs normalized predictions that happen to be perfect
    pred_normalized = (label - mean) / std

    # Meter that knows about normalization
    meter_norm = Meter(mean=mean, std=std)
    meter_norm.update(pred_normalized, label)
    mae_norm = meter_norm.mae()

    # The effective pred after denormalization is pred_normalized * std + mean == label
    for score in mae_norm:
        assert abs(score) < 1e-5, (
            f'MAE should be 0 after inverse normalization, got {score:.6f}'
        )


# ===========================================================================
# 6.  ROC-AUC of a random (coin-flip) classifier converges to ~0.5
# ===========================================================================

def test_roc_auc_random_classifier_near_half():
    """A classifier that outputs constant 0.0 logits (pure random) should
    yield a ROC-AUC close to 0.5 on balanced labels."""
    n = 200
    torch.manual_seed(0)
    label = torch.cat([torch.zeros(n // 2, 1), torch.ones(n // 2, 1)])
    pred = torch.zeros(n, 1)  # constant zero logit → sigmoid = 0.5 everywhere
    meter = Meter()
    meter.update(pred, label)
    score = meter.roc_auc_score()[0]
    assert abs(score - 0.5) < 0.05, (
        f'Random classifier ROC-AUC expected ~0.5, got {score:.4f}'
    )


# ===========================================================================
# 7.  RandomSplitter reproducibility: same seed → identical index sets
# ===========================================================================

def test_random_splitter_same_seed_reproducible(dataset):
    """Two calls with the same random_state must produce identical splits."""
    train1, val1, test1 = RandomSplitter.train_val_test_split(dataset, random_state=7)
    train2, val2, test2 = RandomSplitter.train_val_test_split(dataset, random_state=7)
    assert list(train1.indices) == list(train2.indices)
    assert list(val1.indices)   == list(val2.indices)
    assert list(test1.indices)  == list(test2.indices)


# ===========================================================================
# 8.  RandomSplitter: different seeds produce different splits
# ===========================================================================

def test_random_splitter_different_seeds_different(dataset):
    """Two calls with distinct seeds should (with overwhelming probability)
    produce different index orderings."""
    train1, _, _ = RandomSplitter.train_val_test_split(dataset, random_state=0)
    train2, _, _ = RandomSplitter.train_val_test_split(dataset, random_state=99)
    # For a 10-element dataset the probability that two random permutations
    # of the first 8 elements agree is 1/8! ≈ 0.00025, so this is deterministic
    # in practice with two very different seeds.
    assert list(train1.indices) != list(train2.indices)


# ===========================================================================
# 9.  MolecularWeightSplitter ordering:
#     mean MW of training molecules ≤ mean MW of test molecules
# ===========================================================================

def test_molecular_weight_splitter_ordering(dataset):
    """After a MW-based split the training set should contain lighter molecules
    on average than the test set."""
    train, _, test = MolecularWeightSplitter.train_val_test_split(
        dataset, log_every_n=None)

    def mean_mw(subset):
        mws = [rdMolDescriptors.CalcExactMolWt(dataset.mols[i])
               for i in subset.indices]
        return np.mean(mws)

    assert mean_mw(train) <= mean_mw(test), (
        'Training set mean MW must not exceed test set mean MW'
    )


# ===========================================================================
# 10.  k-fold: validation sets are pairwise disjoint
# ===========================================================================

@pytest.mark.parametrize('k', [2, 3, 5])
def test_kfold_val_sets_mutually_disjoint(k, dataset):
    """No index should appear in more than one validation fold."""
    folds = ConsecutiveSplitter.k_fold_split(dataset, k=k, log=False)
    seen = set()
    for _, val in folds:
        val_set = set(val.indices)
        assert len(val_set & seen) == 0, (
            f'Validation index overlap detected at k={k}'
        )
        seen.update(val_set)


# ===========================================================================
# 11.  count_frequency: sum of all frequencies == len(input)
# ===========================================================================

@pytest.mark.parametrize('values,expected_len', [
    ([1, 2, 3, 1, 2], 5),
    (['a', 'b', 'a', 'c'], 4),
    ([True, False, True], 3),
    ([], 0),
    ([42], 1),
])
def test_count_frequency_sum_equals_input_length(values, expected_len):
    """The sum of all values in the frequency dict must equal len(input)."""
    freq = count_frequency(values)
    assert sum(freq.values()) == expected_len


# ===========================================================================
# 12.  count_frequency: no frequency exceeds the length of the input
# ===========================================================================

@pytest.mark.parametrize('values', [
    [1, 2, 3, 1, 2],
    ['x', 'y', 'x'],
    [0],
    [],
])
def test_count_frequency_max_le_input_length(values):
    """Each individual frequency must be ≤ len(input)."""
    freq = count_frequency(values)
    for val, cnt in freq.items():
        assert cnt <= len(values), (
            f'Frequency of {val!r} ({cnt}) exceeds input length ({len(values)})'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Parametrized tests for dgllife.utils.eval.Meter.
# These tests use pytest.mark.parametrize to cover multiple metric names,
# reduction modes, and invalid arguments in a single test definition,
# which is a different testing style from the single fixed-case tests in test_eval.py.

import numpy as np
import pytest
import torch

from dgllife.utils.eval import Meter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_meter():
    """A Meter with two tasks updated with identical predictions and labels
    (perfect predictions), so that MAE/RMSE == 0 and R2 == 1."""
    label = torch.tensor([[0.5, 1.0], [1.0, 2.0], [1.5, 3.0]])
    pred = torch.tensor([[0.5, 1.0], [1.0, 2.0], [1.5, 3.0]])
    meter = Meter()
    meter.update(pred, label)
    return meter


@pytest.fixture
def classification_meter():
    """A Meter for binary classification with two tasks."""
    label = torch.tensor([[0., 1.], [0., 1.], [1., 0.]])
    pred = torch.tensor([[0.5, 0.5], [0., 1.], [1., 0.]])
    meter = Meter()
    meter.update(pred, label)
    return meter


@pytest.fixture
def multi_batch_meter():
    """A Meter simulating accumulated multi-batch updates."""
    meter = Meter()
    for _ in range(5):
        batch_label = torch.tensor([[0., 1.], [1., 0.]])
        batch_pred = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        meter.update(batch_pred, batch_label)
    return meter


# ---------------------------------------------------------------------------
# Parametrized tests for regression metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('metric_name', ['mae', 'rmse'])
def test_perfect_regression_metric_is_zero(metric_name, regression_meter):
    """MAE and RMSE should be 0 for perfect predictions."""
    scores = regression_meter.compute_metric(metric_name)
    assert isinstance(scores, list)
    assert len(scores) == 2
    for s in scores:
        assert abs(s) < 1e-5, f'{metric_name} expected ~0 for perfect predictions, got {s}'


@pytest.mark.parametrize('metric_name', ['mae', 'rmse'])
def test_regression_metric_non_negative(metric_name):
    """MAE and RMSE are always non-negative regardless of prediction quality."""
    label = torch.tensor([[0.0, 2.0], [1.0, -1.0]])
    pred = torch.tensor([[3.0, -2.0], [-1.0, 4.0]])
    meter = Meter()
    meter.update(pred, label)
    scores = meter.compute_metric(metric_name)
    for s in scores:
        assert s >= 0.0, f'{metric_name} must be non-negative, got {s}'


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_mae_reduction_types(reduction, regression_meter):
    """Verify return type for each valid reduction mode."""
    result = regression_meter.mae(reduction)
    if reduction == 'none':
        assert isinstance(result, list)
    else:
        assert isinstance(result, (float, np.floating))


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_rmse_reduction_types(reduction, regression_meter):
    result = regression_meter.rmse(reduction)
    if reduction == 'none':
        assert isinstance(result, list)
    else:
        assert isinstance(result, (float, np.floating))


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_r2_reduction_types(reduction):
    """Pearson R2 should return proper types for all reduction modes."""
    label = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    pred = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    label_mean, label_std = label.mean(0), label.std(0)
    meter = Meter(label_mean, label_std)
    meter.update(pred, label)
    result = meter.compute_metric('r2', reduction)
    if reduction == 'none':
        assert isinstance(result, list)
    else:
        assert isinstance(result, (float, np.floating))


# ---------------------------------------------------------------------------
# Parametrized tests for classification metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('metric_name', ['roc_auc_score', 'pr_auc_score'])
def test_classification_metric_range(metric_name, classification_meter):
    """ROC-AUC and PR-AUC must lie in [0, 1]."""
    scores = classification_meter.compute_metric(metric_name)
    assert isinstance(scores, list)
    for s in scores:
        assert 0.0 <= s <= 1.0, f'{metric_name} out of range: {s}'


@pytest.mark.parametrize('metric_name', ['roc_auc_score', 'pr_auc_score'])
def test_perfect_classifier_score_is_one(metric_name):
    """Perfect predictions should give score == 1."""
    label = torch.tensor([[0., 1.], [1., 0.], [0., 1.]])
    # Large positive logit → sigmoid ≈ 1; large negative → sigmoid ≈ 0
    pred = torch.tensor([[-10., 10.], [10., -10.], [-10., 10.]])
    meter = Meter()
    meter.update(pred, label)
    scores = meter.compute_metric(metric_name)
    for s in scores:
        assert abs(s - 1.0) < 1e-4, f'Expected perfect score 1.0 for {metric_name}, got {s}'


# ---------------------------------------------------------------------------
# Parametrized tests for invalid inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('invalid_name', ['accuracy', 'f1', '', 'R2', 'MAE'])
def test_compute_metric_invalid_name_raises(invalid_name, classification_meter):
    """compute_metric should raise ValueError for unknown metric names."""
    with pytest.raises(ValueError):
        classification_meter.compute_metric(invalid_name)


@pytest.mark.parametrize('invalid_reduction', ['median', 'max', 'MIN', ''])
def test_invalid_reduction_raises(invalid_reduction, regression_meter):
    """_reduce_scores should raise ValueError for unsupported reduction modes."""
    with pytest.raises(ValueError):
        regression_meter.mae(invalid_reduction)


# ---------------------------------------------------------------------------
# Multi-batch accumulation tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('metric_name', ['roc_auc_score', 'pr_auc_score'])
def test_multi_batch_classification(metric_name, multi_batch_meter):
    """Meter should handle multiple accumulated batches correctly."""
    scores = multi_batch_meter.compute_metric(metric_name)
    assert isinstance(scores, list)
    assert len(scores) == 2
    for s in scores:
        assert 0.0 <= s <= 1.0


@pytest.mark.parametrize('n_batches', [1, 3, 10])
def test_meter_accumulates_correctly(n_batches):
    """MAE should be the same regardless of how many batches the data is split into."""
    label_all = torch.tensor([[float(i), float(i + 1)] for i in range(12)])
    pred_all = label_all + 0.5  # constant offset of 0.5

    # Single-batch meter
    meter_single = Meter()
    meter_single.update(pred_all, label_all)
    mae_single = meter_single.mae()

    # Multi-batch meter
    meter_multi = Meter()
    chunk = 12 // n_batches
    for i in range(n_batches):
        meter_multi.update(pred_all[i*chunk:(i+1)*chunk],
                           label_all[i*chunk:(i+1)*chunk])
    mae_multi = meter_multi.mae()

    for s1, s2 in zip(mae_single, mae_multi):
        assert abs(s1 - s2) < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

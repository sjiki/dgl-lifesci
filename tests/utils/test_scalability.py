# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Scalability tests – verify correctness when computational load is large.
#
# These tests exercise the same mathematical invariants as the unit/property
# tests, but at an order-of-magnitude larger scale to confirm that:
#   * floating-point accumulation does not drift past acceptable tolerances
#   * memory layout stays correct over many batches / many tasks
#   * splitter index bookkeeping is exact for large datasets
#   * helper functions (count_frequency) remain correct at high element counts
#
# All tests are intentionally kept fast enough for a CI pipeline: the largest
# tensors fit comfortably in memory and no external resources are needed.

import pytest
import numpy as np
import torch

from dgllife.utils.eval import Meter
from dgllife.utils.splitters import (
    ConsecutiveSplitter,
    RandomSplitter,
)
from dgllife.utils.analysis import count_frequency


# ===========================================================================
# Shared helpers
# ===========================================================================

class _LargeDataset:
    """Minimal dataset stub with configurable size for splitter tests."""

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, item):
        return item


# ===========================================================================
# 1.  Massive batch accumulation: 1 000 batches with 32 samples each
#     MAE must equal the analytic value (constant error of 1.0 per task)
# ===========================================================================

def test_meter_mae_no_drift_over_1000_batches():
    """Accumulating 1 000 batches of size 32 should not cause floating-point
    drift: MAE must remain within 1e-4 of the analytic value of 1.0."""
    n_batches = 1_000
    batch_size = 32
    n_tasks = 2
    error = 1.0

    meter = Meter()
    label = torch.zeros(batch_size, n_tasks)
    pred = torch.full((batch_size, n_tasks), error)

    for _ in range(n_batches):
        meter.update(pred, label)

    for score in meter.mae():
        assert abs(score - error) < 1e-4, (
            f'MAE drifted after {n_batches} batches: got {score:.6f}, expected {error}'
        )


# ===========================================================================
# 2.  Many tasks: 100-task Meter, reduction consistency at scale
# ===========================================================================

@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_meter_100_tasks_reduction_consistent(reduction):
    """With 100 tasks the 'mean'/'sum' reduction must equal the
    corresponding numpy aggregate of the per-task list."""
    n_tasks = 100
    n_samples = 50
    torch.manual_seed(0)
    label = torch.randn(n_samples, n_tasks)
    pred = label + 0.5  # constant offset

    meter = Meter()
    meter.update(pred, label)

    per_task = meter.compute_metric('mae')
    reduced = meter.compute_metric('mae', reduction)

    if reduction == 'mean':
        expected = np.mean(per_task)
    else:
        expected = np.sum(per_task)

    assert abs(reduced - expected) < 1e-5, (
        f"reduction='{reduction}' mismatch for 100 tasks: "
        f"got {reduced:.6f}, expected {expected:.6f}"
    )


# ===========================================================================
# 3.  Large single batch: 10 000 samples, MAE / RMSE identity holds
# ===========================================================================

def test_meter_10000_samples_rmse_ge_mae():
    """RMSE ≥ MAE must hold for a single large batch of 10 000 samples."""
    n = 10_000
    torch.manual_seed(42)
    label = torch.randn(n, 3)
    pred = torch.randn(n, 3)
    meter = Meter()
    meter.update(pred, label)

    for mae_s, rmse_s in zip(meter.mae(), meter.rmse()):
        assert rmse_s >= mae_s - 1e-6, (
            f'RMSE ({rmse_s:.6f}) < MAE ({mae_s:.6f}) for 10 000 samples'
        )


# ===========================================================================
# 4.  Large dataset splitter: 5 000 items, train + val + test == 5 000
# ===========================================================================

@pytest.mark.parametrize('frac_train,frac_val,frac_test', [
    (0.8, 0.1, 0.1),
    (0.7, 0.2, 0.1),
])
def test_consecutive_split_large_dataset_sizes_sum(frac_train, frac_val, frac_test):
    """For a 5 000-item dataset the three subsets must cover every element."""
    ds = _LargeDataset(5_000)
    train, val, test = ConsecutiveSplitter.train_val_test_split(
        ds, frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    assert len(train) + len(val) + len(test) == len(ds)


# ===========================================================================
# 5.  Large dataset splitter: no index overlap at 5 000 items
# ===========================================================================

def test_consecutive_split_large_no_overlap():
    """Train / val / test index sets must be disjoint for a 5 000-item dataset."""
    ds = _LargeDataset(5_000)
    train, val, test = ConsecutiveSplitter.train_val_test_split(ds)
    train_idx = set(train.indices)
    val_idx   = set(val.indices)
    test_idx  = set(test.indices)
    assert len(train_idx & val_idx)  == 0, 'train ∩ val non-empty at N=5000'
    assert len(train_idx & test_idx) == 0, 'train ∩ test non-empty at N=5000'
    assert len(val_idx   & test_idx) == 0, 'val ∩ test non-empty at N=5000'


# ===========================================================================
# 6.  Random splitter at scale: 1 000 items, same seed → same indices
# ===========================================================================

def test_random_splitter_1000_items_reproducible():
    """RandomSplitter must be deterministic for 1 000 items."""
    ds = _LargeDataset(1_000)
    train1, val1, test1 = RandomSplitter.train_val_test_split(ds, random_state=7)
    train2, val2, test2 = RandomSplitter.train_val_test_split(ds, random_state=7)
    assert list(train1.indices) == list(train2.indices)
    assert list(val1.indices)   == list(val2.indices)
    assert list(test1.indices)  == list(test2.indices)


# ===========================================================================
# 7.  k-fold on large dataset: 1 000 items, k=10 – union of val == all indices
# ===========================================================================

def test_kfold_1000_items_k10_covers_all():
    """k=10 fold on 1 000-item dataset: union of all val folds == {0…999}."""
    ds = _LargeDataset(1_000)
    folds = ConsecutiveSplitter.k_fold_split(ds, k=10, log=False)
    all_val = []
    for _, val in folds:
        all_val.extend(val.indices)
    assert sorted(all_val) == list(range(len(ds)))


# ===========================================================================
# 8.  k-fold on large dataset: each val fold is mutually disjoint
# ===========================================================================

def test_kfold_1000_items_k10_val_disjoint():
    """k=10 validation folds must be pairwise disjoint for 1 000 items."""
    ds = _LargeDataset(1_000)
    folds = ConsecutiveSplitter.k_fold_split(ds, k=10, log=False)
    seen = set()
    for _, val in folds:
        val_set = set(val.indices)
        assert len(val_set & seen) == 0, 'Overlapping validation folds at k=10'
        seen.update(val_set)


# ===========================================================================
# 9.  count_frequency at scale: 100 000 elements – frequency sum == 100 000
# ===========================================================================

def test_count_frequency_100k_elements_sum():
    """sum(freq.values()) must equal 100 000 for a large random integer list."""
    np.random.seed(0)
    values = np.random.randint(0, 1_000, size=100_000).tolist()
    freq = count_frequency(values)
    assert sum(freq.values()) == 100_000


# ===========================================================================
# 10.  count_frequency at scale: no bucket exceeds input length
# ===========================================================================

def test_count_frequency_100k_no_bucket_overflow():
    """No individual frequency may exceed the total number of elements."""
    np.random.seed(1)
    values = np.random.randint(0, 50, size=100_000).tolist()
    freq = count_frequency(values)
    for key, cnt in freq.items():
        assert cnt <= 100_000, (
            f'Frequency of {key!r} ({cnt}) exceeds input length'
        )


# ===========================================================================
# 11.  MAE linear-scaling at scale: 10 000 samples, scale factor 1 000
# ===========================================================================

def test_mae_linear_scaling_large():
    """MAE(1000 * err) == 1000 * MAE(err) for 10 000 samples."""
    n = 10_000
    scale = 1_000.0
    torch.manual_seed(5)
    label = torch.zeros(n, 2)
    err = torch.rand(n, 2)

    meter_base = Meter()
    meter_base.update(err, label)
    mae_base = meter_base.mae()

    meter_scaled = Meter()
    meter_scaled.update(err * scale, label)
    mae_scaled = meter_scaled.mae()

    for b, s in zip(mae_base, mae_scaled):
        assert abs(s - scale * b) < 1e-2, (
            f'Scaling mismatch: {s:.4f} vs {scale * b:.4f}'
        )


# ===========================================================================
# 12.  Multi-batch RMSE == single-batch RMSE for 500 batches
# ===========================================================================

def test_rmse_500_batches_equals_single_batch():
    """RMSE accumulated over 500 small batches must match a single large batch."""
    n_batches = 500
    batch_size = 20
    n_tasks = 4
    torch.manual_seed(99)

    # Build full dataset deterministically
    label_all = torch.randn(n_batches * batch_size, n_tasks)
    pred_all  = label_all + torch.randn(n_batches * batch_size, n_tasks) * 0.5

    # Single-batch meter
    meter_single = Meter()
    meter_single.update(pred_all, label_all)
    rmse_single = meter_single.rmse()

    # Multi-batch meter
    meter_multi = Meter()
    for i in range(n_batches):
        s = i * batch_size
        e = s + batch_size
        meter_multi.update(pred_all[s:e], label_all[s:e])
    rmse_multi = meter_multi.rmse()

    for s1, s2 in zip(rmse_single, rmse_multi):
        assert abs(s1 - s2) < 1e-4, (
            f'RMSE mismatch across 500 batches: single={s1:.6f} multi={s2:.6f}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

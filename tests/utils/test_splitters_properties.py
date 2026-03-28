# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Property / invariant tests for dgllife.utils.splitters.
# Unlike the basic smoke tests in test_splitters.py, these tests verify
# *mathematical invariants* that must hold for any valid split:
#   - train + val + test sizes sum to the total dataset size
#   - no index overlap between train / val / test
#   - k-fold: each sample appears in exactly one validation fold
# They also provide unit tests for the internal helper functions that are not
# tested elsewhere (train_val_test_sanity_check, count_and_log).

import pytest
import numpy as np
import torch
from rdkit import Chem

from dgllife.utils.splitters import (
    ConsecutiveSplitter,
    RandomSplitter,
    MolecularWeightSplitter,
    ScaffoldSplitter,
    SingleTaskStratifiedSplitter,
    train_val_test_sanity_check,
    count_and_log,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SMILES_POOL = [
    'CCO', 'C1CCCCC1', 'O1CCOCC1', 'C1CCCC2C1CCCC2', 'N#N',
    'CC(=O)O', 'c1ccccc1', 'CCCl', 'CCBr', 'CCI',
]


class _Dataset:
    def __init__(self, n=10):
        self.smiles = _SMILES_POOL[:n]
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        self.labels = torch.arange(2 * n, dtype=torch.float).reshape(n, 2)

    def __getitem__(self, item):
        return self.smiles[item], self.mols[item]

    def __len__(self):
        return len(self.smiles)


@pytest.fixture
def dataset():
    return _Dataset(n=10)


# ---------------------------------------------------------------------------
# Property invariant: sizes sum to total
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('splitter_fn', [
    lambda ds: ConsecutiveSplitter.train_val_test_split(ds),
    lambda ds: RandomSplitter.train_val_test_split(ds, random_state=0),
    lambda ds: MolecularWeightSplitter.train_val_test_split(ds),
    lambda ds: ScaffoldSplitter.train_val_test_split(ds),
])
def test_split_sizes_sum_to_total(splitter_fn, dataset):
    """train + val + test should together cover the entire dataset."""
    train, val, test = splitter_fn(dataset)
    assert len(train) + len(val) + len(test) == len(dataset)


# ---------------------------------------------------------------------------
# Property invariant: no overlap between splits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('splitter_fn,kwargs', [
    (ConsecutiveSplitter.train_val_test_split, {}),
    (RandomSplitter.train_val_test_split, {'random_state': 42}),
])
def test_no_overlap_in_splits(splitter_fn, kwargs, dataset):
    """Indices in train / val / test must be mutually disjoint."""
    train, val, test = splitter_fn(dataset, **kwargs)
    train_idx = set(train.indices)
    val_idx = set(val.indices)
    test_idx = set(test.indices)
    assert len(train_idx & val_idx) == 0, 'Overlap between train and val'
    assert len(train_idx & test_idx) == 0, 'Overlap between train and test'
    assert len(val_idx & test_idx) == 0, 'Overlap between val and test'


# ---------------------------------------------------------------------------
# Property invariant: custom fractions respected (approximately)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frac_train,frac_val,frac_test', [
    (0.6, 0.2, 0.2),
    (0.8, 0.1, 0.1),
    (0.5, 0.3, 0.2),
])
def test_consecutive_split_fractions(frac_train, frac_val, frac_test, dataset):
    """ConsecutiveSplitter should produce subsets whose relative sizes are close to
    the requested fractions (within 1 sample due to integer rounding)."""
    train, val, test = ConsecutiveSplitter.train_val_test_split(
        dataset, frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    n = len(dataset)
    assert abs(len(train) - round(n * frac_train)) <= 1
    assert abs(len(val) - round(n * frac_val)) <= 1


# ---------------------------------------------------------------------------
# Property invariant: k-fold — each sample in validation exactly once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('k', [2, 5])
def test_kfold_covers_all_samples(k, dataset):
    """In k-fold split, the union of all validation sets covers the full dataset."""
    folds = ConsecutiveSplitter.k_fold_split(dataset, k=k, log=False)
    assert len(folds) == k
    all_val = []
    for _, val in folds:
        all_val.extend(val.indices)
    assert sorted(all_val) == list(range(len(dataset)))


@pytest.mark.parametrize('k', [2, 5])
def test_kfold_train_val_sizes_add_up(k, dataset):
    """For each fold, train + val must equal the full dataset size."""
    folds = ConsecutiveSplitter.k_fold_split(dataset, k=k, log=False)
    for train, val in folds:
        assert len(train) + len(val) == len(dataset)


def test_kfold_minimum_k_raises():
    """k < 2 should raise an AssertionError."""
    ds = _Dataset(n=10)
    with pytest.raises(AssertionError):
        ConsecutiveSplitter.k_fold_split(ds, k=1, log=False)


# ---------------------------------------------------------------------------
# Unit tests for internal helper: train_val_test_sanity_check
# ---------------------------------------------------------------------------

def test_sanity_check_valid_fractions():
    """Should not raise for fractions that sum to 1."""
    train_val_test_sanity_check(0.8, 0.1, 0.1)
    train_val_test_sanity_check(0.6, 0.2, 0.2)


@pytest.mark.parametrize('fracs', [
    (0.7, 0.2, 0.2),   # sum = 1.1
    (0.5, 0.1, 0.1),   # sum = 0.7
    (1.0, 0.1, 0.1),   # sum = 1.2
])
def test_sanity_check_invalid_fractions_raise(fracs):
    """Fractions that do not sum to 1 should raise AssertionError."""
    with pytest.raises(AssertionError):
        train_val_test_sanity_check(*fracs)


# ---------------------------------------------------------------------------
# Unit tests for internal helper: count_and_log
# ---------------------------------------------------------------------------

def test_count_and_log_prints_at_correct_intervals(capsys):
    """count_and_log should print once every log_every_n iterations."""
    for i in range(9):
        count_and_log('msg', i, 9, log_every_n=3)
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    # Should print at i=2 (3rd), i=5 (6th), i=8 (9th) → 3 lines
    assert len(lines) == 3


def test_count_and_log_no_output_when_none(capsys):
    """count_and_log should be silent when log_every_n=None."""
    for i in range(10):
        count_and_log('msg', i, 10, log_every_n=None)
    captured = capsys.readouterr()
    assert captured.out == ''


def test_count_and_log_message_contains_index(capsys):
    """Printed message should include both the prefix text and the current step count."""
    count_and_log('Processing', 2, 10, log_every_n=3)
    captured = capsys.readouterr()
    assert 'Processing' in captured.out
    assert '3' in captured.out   # i+1 = 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

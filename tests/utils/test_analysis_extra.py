# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Edge-case and helper-function tests for dgllife.utils.analysis.
# These tests are different from test_analysis.py which only tests
# the "normal" path with a mixed set of valid/invalid SMILES.
# Here we cover:
#   - Boundary inputs: all-invalid, all-valid, single molecule, empty-like
#   - Pre-computed mol objects instead of SMILES strings
#   - None entries inside the mols list
#   - Error conditions (both arguments None)
#   - Multiprocess execution
#   - Internal helper functions: count_frequency, summarize_a_mol

import os
import pytest

from dgllife.utils import analyze_mols
from dgllife.utils.analysis import count_frequency, summarize_a_mol
from rdkit import Chem


def _remove(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# analyze_mols — boundary inputs
# ---------------------------------------------------------------------------

def test_analyze_mols_all_invalid_smiles():
    """When every SMILES is invalid, num_valid_mols should be 0."""
    results = analyze_mols(smiles=['invalid', '999', '!!'])
    assert results['num_input_mols'] == 3
    assert results['num_valid_mols'] == 0
    assert results['valid_proportion'] == 0.0
    # No molecular statistics to report
    assert results['num_atoms'] == []
    assert results['num_bonds'] == []


def test_analyze_mols_all_valid_smiles():
    """When every SMILES is valid, valid_proportion should be 1.0."""
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    results = analyze_mols(smiles=smiles)
    assert results['num_valid_mols'] == 3
    assert results['valid_proportion'] == 1.0


def test_analyze_mols_single_molecule():
    """Single-molecule input should return lists of length 1 for count fields."""
    results = analyze_mols(smiles=['CCO'])
    assert results['num_valid_mols'] == 1
    assert results['num_input_mols'] == 1
    assert results['valid_proportion'] == 1.0
    assert results['num_atoms'] == [3]
    assert results['num_bonds'] == [2]
    assert results['num_rings'] == [0]
    assert results['cano_smi'] == ['CCO']


# ---------------------------------------------------------------------------
# analyze_mols — pre-computed mol objects
# ---------------------------------------------------------------------------

def test_analyze_mols_with_mol_objects():
    """analyze_mols should accept pre-computed RDKit Mol instances."""
    mols = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('c1ccccc1')]
    results = analyze_mols(mols=mols)
    assert results['num_input_mols'] == 2
    assert results['num_valid_mols'] == 2


def test_analyze_mols_mols_with_none_entries():
    """None entries in the mols list should be treated as invalid molecules."""
    mols = [Chem.MolFromSmiles('CCO'), None, Chem.MolFromSmiles('c1ccccc1')]
    results = analyze_mols(mols=mols)
    assert results['num_input_mols'] == 3
    assert results['num_valid_mols'] == 2
    assert results['valid_proportion'] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# analyze_mols — error conditions
# ---------------------------------------------------------------------------

def test_analyze_mols_both_none_raises():
    """Passing neither smiles nor mols should raise AssertionError."""
    with pytest.raises(AssertionError):
        analyze_mols(smiles=None, mols=None)


def test_analyze_mols_no_export_creates_no_files():
    """With path_to_export=None no files should be created."""
    analyze_mols(smiles=['CCO'])
    assert not os.path.isfile('valid_canonical_smiles.txt')
    assert not os.path.isfile('summary.txt')


# ---------------------------------------------------------------------------
# analyze_mols — multiprocess consistency
# ---------------------------------------------------------------------------

def test_analyze_mols_multiprocess_same_result():
    """Results with num_processes > 1 should match num_processes == 1."""
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'N#N', 'CCCl']
    r1 = analyze_mols(smiles=smiles, num_processes=1)
    r2 = analyze_mols(smiles=smiles, num_processes=2)
    assert r1['num_valid_mols'] == r2['num_valid_mols']
    assert sorted(r1['cano_smi']) == sorted(r2['cano_smi'])
    assert r1['num_atoms'] == r2['num_atoms']


# ---------------------------------------------------------------------------
# analyze_mols — file export
# ---------------------------------------------------------------------------

def test_analyze_mols_export_creates_files(tmp_path):
    """path_to_export should produce valid_canonical_smiles.txt and summary.txt."""
    analyze_mols(smiles=['CCO', 'c1ccccc1'], path_to_export=str(tmp_path))
    assert (tmp_path / 'valid_canonical_smiles.txt').exists()
    assert (tmp_path / 'summary.txt').exists()


def test_analyze_mols_export_smiles_content(tmp_path):
    """valid_canonical_smiles.txt should contain one SMILES per line."""
    analyze_mols(smiles=['CCO', 'c1ccccc1'], path_to_export=str(tmp_path))
    lines = (tmp_path / 'valid_canonical_smiles.txt').read_text().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# count_frequency — unit tests
# ---------------------------------------------------------------------------

def test_count_frequency_empty_list():
    assert count_frequency([]) == {}


def test_count_frequency_single_element():
    assert count_frequency(['C']) == {'C': 1}


def test_count_frequency_multiple_elements():
    result = count_frequency(['C', 'N', 'C', 'O', 'C'])
    assert result == {'C': 3, 'N': 1, 'O': 1}


def test_count_frequency_integer_values():
    result = count_frequency([1, 2, 1, 3, 2, 1])
    assert result[1] == 3
    assert result[2] == 2
    assert result[3] == 1


def test_count_frequency_preserves_all_unique_values():
    values = [True, False, True, False, False]
    result = count_frequency(values)
    assert result[True] == 2
    assert result[False] == 3


# ---------------------------------------------------------------------------
# summarize_a_mol — unit tests
# ---------------------------------------------------------------------------

def test_summarize_a_mol_methane():
    """Single-atom molecule: 0 bonds, 0 rings."""
    mol = Chem.MolFromSmiles('C')
    summary = summarize_a_mol(mol)
    assert summary['num_atoms'] == 1
    assert summary['num_bonds'] == 0
    assert summary['num_rings'] == 0
    assert 'C' in summary['atom_type']
    assert len(summary['bond_type']) == 0   # no bonds


def test_summarize_a_mol_ethanol():
    """CCO: 3 atoms, 2 bonds, 0 rings."""
    mol = Chem.MolFromSmiles('CCO')
    summary = summarize_a_mol(mol)
    assert summary['num_atoms'] == 3
    assert summary['num_bonds'] == 2
    assert summary['num_rings'] == 0
    assert 'C' in summary['atom_type']
    assert 'O' in summary['atom_type']


def test_summarize_a_mol_benzene():
    """c1ccccc1: 6 atoms, 6 bonds, 1 ring, all aromatic."""
    mol = Chem.MolFromSmiles('c1ccccc1')
    summary = summarize_a_mol(mol)
    assert summary['num_atoms'] == 6
    assert summary['num_bonds'] == 6
    assert summary['num_rings'] == 1
    assert True in summary['aromatic_atom']


def test_summarize_a_mol_returns_expected_keys():
    """summarize_a_mol should always return a fixed set of keys."""
    expected_keys = {
        'num_atoms', 'num_bonds', 'num_rings',
        'atom_type', 'degree', 'total_degree', 'explicit_valence',
        'implicit_valence', 'hybridization', 'total_num_h', 'formal_charge',
        'num_radical_electrons', 'aromatic_atom', 'chirality_tag',
        'bond_type', 'conjugated_bond', 'bond_stereo_configuration', 'bond_direction',
    }
    mol = Chem.MolFromSmiles('CCO')
    summary = summarize_a_mol(mol)
    assert set(summary.keys()) == expected_keys


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

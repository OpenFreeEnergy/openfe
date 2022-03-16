# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
from rdkit import Chem

from openfe.setup import ProteinMolecule


def test_from_pdbfile(PDB_181L_path):
    p = ProteinMolecule.from_pdbfile(PDB_181L_path, name='Steve')

    assert isinstance(p, ProteinMolecule)
    assert p.name == 'Steve'
    assert p.to_rdkit().GetNumAtoms() == 1441


def test_from_pdbfile_ValueError(PDBx_181L_path):
    with pytest.raises(ValueError):
        _ = ProteinMolecule.from_pdbfile(PDBx_181L_path)


def test_from_rdkit(PDB_181L_path):
    m = Chem.MolFromPDBFile(PDB_181L_path)
    p = ProteinMolecule.from_rdkit(m, 'Steve')

    assert isinstance(p, ProteinMolecule)
    assert p.name == 'Steve'
    assert p.to_rdkit().GetNumAtoms() == 1441


def test_to_rdkit(PDB_181L_path):
    pm = ProteinMolecule.from_pdbfile(PDB_181L_path)
    rdkitmol = pm.to_rdkit()

    assert isinstance(rdkitmol, Chem.Mol)
    assert rdkitmol.GetNumAtoms() == 1441


def test_eq(PDB_181L_path):
    m1 = ProteinMolecule.from_pdbfile(PDB_181L_path)
    m2 = ProteinMolecule.from_pdbfile(PDB_181L_path)

    assert m1 == m2


def test_hash_eq(PDB_181L_path):
    m1 = ProteinMolecule.from_pdbfile(PDB_181L_path)
    m2 = ProteinMolecule.from_pdbfile(PDB_181L_path)

    assert hash(m1) == hash(m2)


def test_neq(PDB_181L_path):
    m1 = ProteinMolecule.from_pdbfile(PDB_181L_path, name='This')
    m2 = ProteinMolecule.from_pdbfile(PDB_181L_path, name='Other')

    assert m1 != m2


def test_hash_neq(PDB_181L_path):
    m1 = ProteinMolecule.from_pdbfile(PDB_181L_path, name='This')
    m2 = ProteinMolecule.from_pdbfile(PDB_181L_path, name='Other')

    assert hash(m1) != hash(m2)


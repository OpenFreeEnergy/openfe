#!/usr/bin/env python

# This script creates several files used in testing setup serialization:
#
# * openfe/tests/data/multi_molecule.sdf
# * openfe/tests/data/serialization/ethane_template.sdf
# * openfe/tests/data/serialization/network_template.graphml
#
# The two serialization templates need manual editing to replace the current
# version of gufe with:
# {GUFE_VERSION}

from rdkit import Chem
from rdkit.Chem import AllChem
from openfe.setup import SmallMoleculeComponent, LigandNetwork
from openfe.setup.atom_mapping import LigandAtomMapping
# multi_molecule.sdf
mol1 = Chem.MolFromSmiles("CCO")
mol2 = Chem.MolFromSmiles("CCC")

writer = Chem.SDWriter("multi_molecule.sdf")
writer.write(mol1)
writer.write(mol2)
writer.close()


def mol_from_smiles(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)

    return m


# ethane_template.sdf
m = SmallMoleculeComponent(mol_from_smiles("CC"), name="ethane")

with open("ethane_template.sdf", mode="w") as tmpl:
    tmpl.write(m.to_sdf())

# ethane_with_H_template.sdf
m2 = SmallMoleculeComponent(Chem.AddHs(m.to_rdkit()))

with open("ethane_with_H_template.sdf", mode="w") as tmpl:
    tmpl.write(m2.to_sdf())


# network_template.graphml
mol1 = SmallMoleculeComponent(mol_from_smiles("CCO"))
mol2 = SmallMoleculeComponent(mol_from_smiles("CC"))
mol3 = SmallMoleculeComponent(mol_from_smiles("CO"))

edge12 = LigandAtomMapping(mol1, mol2, {0: 0, 1: 1})
edge23 = LigandAtomMapping(mol2, mol3, {0: 0})
edge13 = LigandAtomMapping(mol1, mol3, {0: 0, 2: 1})

network = LigandNetwork([edge12, edge23, edge13])

with open("network_template.graphml", "w") as fn:
    fn.write(network.to_graphml())

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
from openfe.setup import SmallMoleculeComponent, LigandAtomMapping, Network

# multi_molecule.sdf
mol1 = Chem.MolFromSmiles("CCO")
mol2 = Chem.MolFromSmiles("CCC")

writer = Chem.SDWriter("multi_molecule.sdf")
writer.write(mol1)
writer.write(mol2)
writer.close()


# ethane_template.sdf
m = SmallMoleculeComponent(Chem.MolFromSmiles("CC"), name="ethane")

with open("ethane_template.sdf", mode="w") as tmpl:
    tmpl.write(m.to_sdf())

# ethane_with_H_template.sdf
m2 = SmallMoleculeComponent(Chem.AddHs(m.to_rdkit()))

with open("ethane_with_H_template.sdf", mode="w") as tmpl:
    tmpl.write(m2.to_sdf())


# network_template.graphml
mol1 = SmallMoleculeComponent(Chem.MolFromSmiles("CCO"))
mol2 = SmallMoleculeComponent(Chem.MolFromSmiles("CC"))
mol3 = SmallMoleculeComponent(Chem.MolFromSmiles("CO"))

edge12 = LigandAtomMapping(mol1, mol2, {0: 0, 1: 1})
edge23 = LigandAtomMapping(mol2, mol3, {0: 0})
edge13 = LigandAtomMapping(mol1, mol3, {0: 0, 2: 1})

network = Network([edge12, edge23, edge13])

with open("network_template.graphml", "w") as fn:
    fn.write(network.to_graphml())

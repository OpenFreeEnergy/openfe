# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from .abstract_chemicalsystem_generator import AbstractChemicalSystemGenerator
from enum import Enum
from typing import Iterable

from gufe import (
    SmallMoleculeComponent,
    ProteinComponent,
    SolventComponent,
    ChemicalSystem,
)

# Todo: connect to protocols - use this for labels?
class RFEChemicalSystemComponentLabels(str, Enum):
    PROTEIN = 'protein'
    LIGAND = 'ligand'
    SOLVENT= "sovlent"

component_labels =RFEChemicalSystemComponentLabels

class EasyChemicalSystemGenerator(AbstractChemicalSystemGenerator):
    def __init__(
        self,
        solvent: SolventComponent = None,
        protein: ProteinComponent = None,
        do_vacuum: bool = False,
    ):
        self.solvent = solvent
        self.protein = protein
        self.do_vacuum = do_vacuum

        if solvent is None and protein is None and not do_vacuum:
            raise ValueError("you need to provide any transformation possibility")

    def __call__(self, component: SmallMoleculeComponent) -> Iterable[ChemicalSystem]:
        if self.do_vacuum:
            chem_sys = ChemicalSystem(
                components={component_labels.LIGAND: component}, name=component.name + "_vacuum"
            )
            yield chem_sys

        if self.solvent is not None:
            chem_sys = ChemicalSystem(
                components={component_labels.LIGAND: component, component_labels.SOLVENT: self.solvent},
                name=component.name + "_solvent",
            )
            yield chem_sys

        print(self.protein)

        if self.protein is not None:
            components = {component_labels.LIGAND: component, component_labels.PROTEIN: self.protein}
            if self.solvent is not None:
                components.update({component_labels.SOLVENT: self.solvent})
            chem_sys = ChemicalSystem(
                components=components, name=component.name + "_receptor"
            )
            yield chem_sys

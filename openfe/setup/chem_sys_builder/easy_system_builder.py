# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from _abstract_chem_sys_builder import _abstract_chem_sys_builder

from typing import Iterable
from gufe import SmallMoleculeComponent, ProteinComponent, SolventComponent, ChemicalSystem


class chem_system_builder(_abstract_chem_sys_builder):

    def __init__(self, solvent: SolventComponent = None, protein: ProteinComponent = None, do_vacuum: bool = False):
        self.solvent = solvent
        self.protein = protein
        self.do_vacuum = do_vacuum

        if (solvent is None and protein is None and not do_vacuum):
            raise ValueError("you need to provide any transformation possibility")

    def __call__(self, component: SmallMoleculeComponent) -> Iterable[ChemicalSystem]:
        chemical_systems = []
        if (self.solvent is not None):
            chem_sys = ChemicalSystem(components={"compA": component,
                                                  "solvent": self.solvent})
            chemical_systems.append(chem_sys, name=component.name + "_solvent")

        if (self.solvent is not None and self.protein is not None):
            components = {"compA": component, "receptor": self.protein}
            if (self.solvent is not None):
                components.update({"solvent": self.solvent})
            chem_sys = ChemicalSystem(components=components, name=component.name + "_receptor")
            chemical_systems.append(chem_sys)

        if (self.do_vacuum):
            chem_sys = ChemicalSystem(components={"compA": component})
            chemical_systems.append(chem_sys, name=component.name + "_vacuum")

        return chemical_systems

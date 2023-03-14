# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from .abstract_chemicalsystem_generator import Abstract_ChemicalSystem_generator

from typing import Iterable
from gufe import SmallMoleculeComponent, ProteinComponent, SolventComponent, ChemicalSystem


class Easy_chemicalsystem_generator(Abstract_ChemicalSystem_generator):

    def __init__(self, solvent: SolventComponent = None, protein: ProteinComponent = None, do_vacuum: bool = False):
        self.solvent = solvent
        self.protein = protein
        self.do_vacuum = do_vacuum

        if (solvent is None and protein is None and not do_vacuum):
            raise ValueError("you need to provide any transformation possibility")

    def __call__(self, component: SmallMoleculeComponent) -> Iterable[ChemicalSystem]:
        chemical_systems = []
        
        if (self.do_vacuum):
            chem_sys = ChemicalSystem(components={"compA": component}, name=component.name + "_vacuum")
            yield chem_sys

        if (self.solvent is not None):
            chem_sys = ChemicalSystem(components={"compA": component,
                                                  "solvent": self.solvent},
                                      name=component.name + "_solvent")
            yield chem_sys

        if (self.solvent is not None and self.protein is not None):
            components = {"compA": component, "receptor": self.protein}
            if (self.solvent is not None):
                components.update({"solvent": self.solvent})
            chem_sys = ChemicalSystem(components=components, name=component.name + "_receptor")
            yield chem_sys
        
        return chemical_systems

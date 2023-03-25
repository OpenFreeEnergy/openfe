# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from .abstract_chemicalsystem_generator import AbstractChemicalSystemGenerator, RFEComponentLabels
from typing import Iterable

from gufe import (
    SmallMoleculeComponent,
    ProteinComponent,
    SolventComponent,
    ChemicalSystem,
)


class EasyChemicalSystemGenerator(AbstractChemicalSystemGenerator):

    def __init__(
        self,
        solvent: SolventComponent = None,
        protein: ProteinComponent = None,
        do_vacuum: bool = False,
    ):
        """this Class is a easy generator class, for generating chemical systems with a focus on a given smallMoleculeComponent.
        depending on which parameters are given, the following systems will be generated in order:
            vacuum -> solvent -> protein

        Parameters
        ----------
        solvent : SolventComponent, optional
            if a SolventComponent is given, solvated chemical systems will be generated, by default None
        protein : ProteinComponent, optional
            if a ProteinComponent is given, complex chemical systems will be generated, by default None
        do_vacuum : bool, optional
            if true a chemical system in vacuum is returned, by default False

        Raises
        ------
        ValueError
            _description_
        """
        self.solvent = solvent
        self.protein = protein
        self.do_vacuum = do_vacuum

        if solvent is None and protein is None and not do_vacuum:
            raise ValueError("you need to provide any system generation information in the constructor")

    def __call__(self, component: SmallMoleculeComponent) -> Iterable[ChemicalSystem]:
        return self._generate_systems(component=component)

    def _generate_systems(
        self, component: SmallMoleculeComponent
    ) -> Iterable[ChemicalSystem]:
        """generate systems, around the given SmallMoleculeComponent

        Parameters
        ----------
        component : SmallMoleculeComponent
            the molecule for the system generation

        Returns
        -------
        Iterable[ChemicalSystem]
            generator for systems with the given environments

        Yields
        ------
        Iterator[Iterable[ChemicalSystem]]
            generator for systems with the given environments

        """        
        
        if self.do_vacuum:
            chem_sys = ChemicalSystem(
                components={RFEComponentLabels.LIGAND: component},
                name=component.name + "_vacuum",
            )
            yield chem_sys

        if self.solvent is not None:
            chem_sys = ChemicalSystem(
                components={
                    RFEComponentLabels.LIGAND: component,
                    RFEComponentLabels.SOLVENT: self.solvent,
                },
                name=component.name + "_solvent",
            )
            yield chem_sys

        if self.protein is not None:
            components = {
                RFEComponentLabels.LIGAND: component,
                RFEComponentLabels.PROTEIN: self.protein,
            }
            if self.solvent is not None:
                components.update({RFEComponentLabels.SOLVENT: self.solvent})
            chem_sys = ChemicalSystem(
                components=components, name=component.name + "_receptor"
            )
            yield chem_sys
            
        return
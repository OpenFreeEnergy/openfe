# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable, Optional

from gufe import ChemicalSystem, Component, ProteinComponent, SmallMoleculeComponent, SolventComponent

from .abstract_chemicalsystem_generator import AbstractChemicalSystemGenerator, RFEComponentLabels


class EasyChemicalSystemGenerator(AbstractChemicalSystemGenerator):
    def __init__(
        self,
        solvent: Optional[SolventComponent] = None,
        protein: Optional[ProteinComponent] = None,
        cofactors: Optional[Iterable[SmallMoleculeComponent]] = None,
        do_vacuum: bool = False,
    ):
        """
        Generate consistent chemical systems given a :class:`SmallMoleculeComponent`.

        This class aids preparation of :class:`ChemicalSystem` instances for
        free energy simulations. Construct an instance of the class with all the
        components except the :class:`SmallMoleculeComponent` that will be
        mutated, and then call the instance on each mutation target to prepare
        systems in vacuum, solvent, and with protein.

        This class is a easy generator class, for generating chemical systems
        with a focus on a given SmallMoleculeComponent. Depending on which
        parameters are given, the following systems will be generated in
        order:

            vacuum -> solvent -> protein

        Parameters
        ----------
        solvent : SolventComponent, optional
            if a SolventComponent is given, solvated chemical systems will be generated, by default None
        protein : ProteinComponent, optional
            if a ProteinComponent is given, complex chemical systems will be generated, by default None
        cofactors : Iterable[SmallMoleculeComponent], optional
            any cofactors in the system.  will be put in any systems containing
            the protein
        do_vacuum : bool, optional
            if true a chemical system in vacuum is returned, by default False

        Raises
        ------
        ValueError
            If neither a solvent nor protein is provided and ``do_vacuum`` is
            false.
        """
        self.solvent = solvent
        self.protein = protein
        self.cofactors = cofactors or []
        self.do_vacuum = do_vacuum

        if solvent is None and protein is None and not do_vacuum:
            raise ValueError(
                "Chemical system generator is unable to generate any chemical systems with neither protein nor solvent nor do_vacuum",
            )

    def __call__(self, component: SmallMoleculeComponent) -> Iterable[ChemicalSystem]:
        """Generate systems around the given :class:`SmallMoleculeComponent`.

        Parameters
        ----------
        component : SmallMoleculeComponent
            The molecule for the system generation.

        Returns
        -------
        Iterable[ChemicalSystem]
            Generator for systems with the given environments. Returns a
            vacuum system first in ``self.do_vacuum`` is true, then a solvated
            system without protein, then finally a solvated system with protein
            if the protein component is set.

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

        components: dict[str, Component]
        if self.protein is not None:
            components = {
                RFEComponentLabels.LIGAND: component,
                RFEComponentLabels.PROTEIN: self.protein,
            }
            for i, c in enumerate(self.cofactors):
                components.update({f"{RFEComponentLabels.COFACTOR}{i+1}": c})
            if self.solvent is not None:
                components.update({RFEComponentLabels.SOLVENT: self.solvent})
            chem_sys = ChemicalSystem(components=components, name=component.name + "_complex")
            yield chem_sys

        return

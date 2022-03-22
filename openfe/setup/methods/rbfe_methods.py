# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Contains RBFE methods


"""
from __future__ import annotations

from typing import Dict, Union

from openfe.setup import AtomMapping, Molecule
from openfe.setup.methods import FEMethod


class LigandLigandTransformSettings:
    """Dict-like object holding the default settings for a ligand transform"""
    def update(self, settings: Union[Dict, LigandLigandTransformSettings]):
        pass


class LigandLigandTransformResults:
    """Dict-like container for the output of a LigandLigandTransform"""
    pass


class LigandLigandTransform(FEMethod):
    """Calculates the free energy of an alchemical ligand swap in solvent

    """
    _SETTINGS_CLASS = LigandLigandTransformSettings

    def __init__(self,
                 ligandA: Molecule,
                 ligandB: Molecule,
                 ligandmapping: AtomMapping,
                 settings: Union[Dict, LigandLigandTransformSettings] = None,
                 ):
        """
        Parameters
        ----------
        ligandA, ligandB : Molecule
          the two ligand molecules to transform between.  The transformation
          will go from ligandA to ligandB.
        ligandmapping : AtomMapping
          the mapping of atoms between the
        settings : dict
          the settings for the Method.

        The default settings for this method can be accessed via the
        get_default_settings method,
        """
        self._ligandA = ligandA
        self._ligandB = ligandB
        self._mapping = ligandmapping
        self._settings = self.__class__.get_default_settings()
        if settings is not None:
            self._settings.update(settings)
        # TODO: Prepare the workload

    def run(self) -> bool:
        """Perform this method, returning success"""
        if self.is_complete():
            return True
        # TODO: Execute the workload
        return False

    def is_complete(self) -> bool:
        return False

    def get_results(self) -> LigandLigandTransformResults:
        """Return payload created by this workload

        Raises
        ------
        ValueError
          if the results do not exist yet
        """
        if not self.is_complete():
            raise ValueError("Results have not been generated")
        return LigandLigandTransformResults()


class ComplexTransformResults:
    pass


class ComplexTransform(FEMethod):
    """Calculates the free energy of an alchemical ligand swap in complex"""
    def __init__(self,
                 ligand1: Molecule,
                 ligand2: Molecule,
                 ligandmapping: AtomMapping,
                 protein: Molecule,
                 settings: Dict = None,
                 ):
        self._ligand1 = ligand1
        self._ligand2 = ligand2
        self._mapping = ligandmapping
        self._protein = protein
        self._settings = self.__class__.get_default_settings()
        if settings is not None:
            self._settings.update(settings)

    def is_complete(self) -> bool:
        return False

    def get_results(self) -> ComplexTransformResults:
        if not self.is_complete():
            raise ValueError("Results have not been generated")

        return ComplexTransformResults()

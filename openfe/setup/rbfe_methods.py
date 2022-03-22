# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Contains RBFE methods for


"""
from __future__ import annotations

from typing import Dict, Union

from openfe.setup import AtomMapping, Molecule


class LigandLigandTransformSettings:
    """Dict-like object holding the default settings for a ligand transform"""
    def update(self, settings: Union[Dict, LigandLigandTransformSettings]):
        pass


class LigandLigandTransformResults:
    """Dict-like container for the output of a LigandLigandTransform"""
    pass


class LigandLigandTransform:
    """Calculates the free energy of an alchemical ligand swap in solvent

    """
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

    @staticmethod
    def get_default_settings() -> LigandLigandTransformSettings:
        """Get the default settings for this Method

        These can be updated and passed back to create the Method
        """
        return LigandLigandTransformSettings()

    def to_xml(self) -> str:
        """Serialise this method to xml"""
        raise NotImplementedError()

    @classmethod
    def from_xml(cls, xml: str):
        """Deserialise this from a saved xml representation"""
        raise NotImplementedError()

    def run(self) -> bool:
        """Perform this method, returning success"""
        if self.is_complete():
            return True
        # TODO: Execute the workload
        return False

    def is_complete(self) -> bool:
        """Check if the results of this workload already exist"""
        return False

    def get_results(self) -> LigandLigandTransformResults:
        """Return payload created by this workload"""
        return None


class ComplexTransform:
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

    @staticmethod
    def get_default_settings() -> Dict:
        return dict()

    def to_xml(self) -> str:
        raise NotImplementedError


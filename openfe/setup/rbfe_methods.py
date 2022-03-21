# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Contains RBFE methods for


"""
from typing import Dict

from openfe.setup import AtomMapping, Molecule


class LigandLigandTransform:
    """Calculates the free energy of an alchemical ligand swap in solvent

    """
    def __init__(self,
                 ligand1: Molecule,
                 ligand2: Molecule,
                 ligandmapping: AtomMapping,
                 settings=None,
                 ):
        """
        Parameters
        ----------
        ligand1, ligand2 : Molecule
          the two ligand molecules to transform between.  The transformation
          will go from ligandA to ligandB.
        ligandmapping : AtomMapping
          the mapping of atoms between the
        settings : dict
          the settings for the Method.

        The default settings for this method can be accessed via the
        get_default_settings method,
        """
        self._ligand1 = ligand1
        self._ligand2 = ligand2
        self._mapping = ligandmapping
        self._settings = self.__class__.get_default_settings()
        if settings is not None:
            self._settings.update(settings)
        # TODO: Prepare the workload

    @staticmethod
    def get_default_settings() -> Dict:
        """Get the default settings for this Method

        These can be updated and passed back to create the Method
        """
        return dict()

    def to_xml(self) -> str:
        """Serialise this method to xml"""
        raise NotImplementedError()

    def run(self) -> bool:
        """Perform this method, returning success"""
        if self.is_complete():
            return True
        # TODO: Execute the workload
        return False

    def is_complete(self):
        """Check status of this workload"""
        return False

    def get_results(self):
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


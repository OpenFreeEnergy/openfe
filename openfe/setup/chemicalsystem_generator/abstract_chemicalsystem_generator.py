# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import abc
from collections.abc import Iterable
from enum import Enum

from gufe import ChemicalSystem

# Todo: connect to protocols - use this for labels?


class RFEComponentLabels(str, Enum):
    PROTEIN = "protein"
    LIGAND = "ligand"
    SOLVENT = "solvent"
    COFACTOR = "cofactor"


class AbstractChemicalSystemGenerator(abc.ABC):
    """
    this abstract class defines the interface for the chemical system generators.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Iterable[ChemicalSystem]:
        raise NotImplementedError()

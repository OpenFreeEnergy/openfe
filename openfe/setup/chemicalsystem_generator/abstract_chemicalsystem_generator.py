# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable
from gufe import ChemicalSystem


class AbstractChemicalSystemGenerator:
    def __call__(self, *args, **kwargs) -> Iterable[ChemicalSystem]:
        return NotImplementedError()

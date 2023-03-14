# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable
from gufe import ChemicalSystem

class Abstract_ChemicalSystem_generator():
    
    def __call__(self, *args, **kwargs) -> Iterable[ChemicalSystem]:
        return NotImplementedError( )

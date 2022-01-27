# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from typing import Dict, Union


class AtomMapping:
    """Simple container with the mapping between two Molecules

    Attributes
    ----------
    AtoB, BtoA : dict
      maps the index of an atom in either molecule **A** or **B** to the other.
      If this atom has no corresponding atom, None is returned.


    The size of molecule A/B is given by the length of the AtoB/BtoA dictionary
    """
    AtoB: Dict[int, Union[int, None]]
    BtoA: Dict[int, Union[int, None]]

    def __init__(self):
        self.AtoB = dict()
        self.BtoA = dict()
    
    @classmethod
    def from_perses(cls, perses_mapping):
        raise NotImplementedError()

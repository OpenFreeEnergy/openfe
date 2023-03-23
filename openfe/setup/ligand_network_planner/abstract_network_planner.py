# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import abc
from typing import Iterable, Callable, List
from gufe import Component, AtomMapper, AtomMapping

from ..ligand_network import LigandNetwork


class AbstractNetworkPlanner(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> LigandNetwork:
        raise NotImplementedError()

class AbstractRelativeLigandNetworkPlanner(AbstractNetworkPlanner):

    _mappers: Iterable[AtomMapper]
    _mappings: List[AtomMapping]
    _mapping_scorer: Callable

    @property
    def mappers(self)->Iterable[AtomMapper]:
        return self._mappers

    @property
    def mappings(self) -> List[AtomMapping]:
        if(hasattr(self, "_mappings")):
            return self._mappings
        else:
            raise ValueError("Mappings not calculated yet!")

    @property
    def mapping_scorer(self)->Callable:
        return self._mapping_scorer
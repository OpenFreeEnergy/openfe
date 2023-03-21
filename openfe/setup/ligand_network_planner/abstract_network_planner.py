# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import abc
from typing import Iterable, Callable
from gufe import Component, AtomMapper, AtomMapping

from ..ligand_network import LigandNetwork


class AbstractNetworkPlanner(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> LigandNetwork:
        raise NotImplementedError()

class AbstractRelativeLigandNetworkPlanner(abc.ABC):

    _mappers: Iterable[AtomMapper]
    _mappings: Iterable[AtomMapping]
    _mapping_scorer: Callable

    @property
    def mappers(self)->Iterable[AtomMapper]:
        return self._mappers

    @property
    def mappings(self) -> Iterable[AtomMapping]:
        return self._mappings

    @property
    def mapping_scorer(self)->Callable:
        return self._mapping_scorer
# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import abc
from typing import Iterable
from gufe import Component, AlchemicalNetwork


class AbstractAlchemicalNetworkPlanner(abc.ABC):

    @abc.abstractmethod
    def __call__(self, components: Iterable[Component]) -> AlchemicalNetwork:
        raise NotImplementedError()

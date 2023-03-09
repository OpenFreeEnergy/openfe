# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable
from gufe import Component, AlchemicalNetwork


class _abstract_campaigner():

    def __call__(self, components: Iterable[Component]) -> AlchemicalNetwork:
        raise NotImplementedError()


class _abstract_dynamic_campaigner(_abstract_campaigner):

    def _update(self, alchemical_network: AlchemicalNetwork) -> AlchemicalNetwork:
        raise NotImplementedError()
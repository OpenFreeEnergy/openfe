from typing import Iterable
from gufe import Component, AlchemicalNetwork


class _abstract_campaigner():

    def __call__(self, components:Iterable[Component])->AlchemicalNetwork:
        raise NotImplementedError()

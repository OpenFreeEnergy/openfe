# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import abc
from gufe import AlchemicalNetwork


class AbstractTransformerFactory(abc.ABC):
    """
    This class type is thought to be the part that transforms identified edges/transformations for chemical systems of an FE-campaign to an Alchemical
    Network.

    This can be very simple like simple transform where one protocol is directly mapped onto one transformation, but it could also be complex,
    where one transformation gets a protocols assigned from a set of protocols, depending on certain metrics.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> AlchemicalNetwork:
        raise NotImplementedError()

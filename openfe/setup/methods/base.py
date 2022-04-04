# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import abc
from openff.toolkit.utils.serialization import Serializable


class FEMethod(Serializable):
    """Base class for defining a free energy method

    Child classes must implement:
    - the associated Settings class and a default point for this
    - init, taking the Settings class
    - run()
    - to_dict and from_dict for serialization
    """
    @classmethod
    @abc.abstractmethod
    def get_default_settings(cls):
        """Get the default settings for this FE Method

        These can be modified and passed back in to the class init
        """
        ...

    @abc.abstractmethod
    def is_complete(self) -> bool:
        """Check if the results of this workload already exist"""
        ...

    @abc.abstractmethod
    def run(self) -> bool:
        """Perform this method, returning success"""
        ...

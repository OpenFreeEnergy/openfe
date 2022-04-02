# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import abc


class FEMethod(abc.ABC):
    """Base class for defining a free energy method

    Child classes must implement:
    - the associated Settings class (assigning to _SETTINGS_CLASS)
    - init, taking the Settings class
    - run()
    -
    """
    @classmethod
    @abc.abstractmethod
    def get_default_settings(cls):
        """Get the default settings for this FE Method

        These can be modified and passed back in to the class init
        """
        ...

    @abc.abstractmethod
    def to_xml(self) -> str:
        """Serialise this method to xml"""
        ...

    @classmethod
    @abc.abstractmethod
    def from_xml(cls, xml: str):
        ...

    @abc.abstractmethod
    def is_complete(self) -> bool:
        """Check if the results of this workload already exist"""
        ...

    @abc.abstractmethod
    def run(self) -> bool:
        """Perform this method, returning success"""
        ...

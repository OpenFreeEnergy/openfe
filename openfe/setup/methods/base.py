# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe


class FEMethod:
    """Base class for defining a free energy method

    Child classes must implement:
    - the associated Settings class (assigning to _SETTINGS_CLASS)
    - init, taking the Settings class
    - run()
    -
    """
    _SETTINGS_CLASS = None

    @classmethod
    def get_default_settings(cls):
        """Get the default settings for this FE Method

        These can be modified and passed back in to the class init
        """
        return cls._SETTINGS_CLASS

    def to_xml(self) -> str:
        """Serialise this method to xml"""
        raise NotImplementedError()

    @classmethod
    def from_xml(cls, xml: str):
        raise NotImplementedError()

    def is_complete(self) -> bool:
        """Check if the results of this workload already exist"""
        raise NotImplementedError()

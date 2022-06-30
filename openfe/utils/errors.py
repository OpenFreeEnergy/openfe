ABSTRACT_ERROR_STRING = ("'{cls}' is an abstract class and should not be "
                         "used directly. Please use a specific subclass of "
                         "'{cls}' that implements {func}.")

"""
The exceptions are heavily inspired by OpenFF-toolkit!
"""


class OpenFEException(Exception):
    """Base exception for custom exceptions raised by the OpenFE Repository"""

    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class MissingPackageError(OpenFEException, ImportError):
    """
    Exception for when an optional dependency is needed but not installed
    """

    def __init__(self, package_name: str, required_by: str):
        self.msg = (
            f"Missing dependency {package_name} for {required_by}. "
            f"Try installing it with\n\n$ "
            f"conda install {package_name} -c conda-forge"
        )

        super().__init__(self.msg)


class LicenseError(OpenFEException):
    """This function requires a license that cannot be found."""

    def __init__(self, license_name: str, required_by: str):
        self.msg = (
            f"Missing License {license_name} for {required_by}."
        )

        super().__init__(self.msg)

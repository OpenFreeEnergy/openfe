"""
Tools for integration with miscellaneous non-required packages.
shamelessly borrowed from openpathsampling author: dwhswenson
"""
import functools
from openff.toolkit.utils import toolkits
from .errors import MissingPackageError, LicenseError


def error_if_no_license(name: str, license_name: str, has_license: bool):
    if not has_license:
        raise LicenseError(required_by=name, license_name=license_name)


def error_if_no_package(name: str, package_name: str, has_package: bool):
    if not has_package:
        raise MissingPackageError(required_by=name, package_name=package_name)

"""
 ===========================================================================
    SOLUTION 1
 ===========================================================================
"""
# Licenses
# OpenEye ###################################################

if (not toolkits.OPENEYE_AVAILABLE):
    HAS_OPENEYE_TOOLKIT_LICENSE = False
else:
    HAS_OPENEYE_TOOLKIT_LICENSE = True


def error_if_no_openeye_license(name: str):
    return error_if_no_license(name,
                               'OpenEye Toolkit',
                               HAS_OPENEYE_TOOLKIT_LICENSE)


# Packages
# Perses # OpenEye ###################################################


try:
    import perses
    HAS_PERSES = True
except ImportError:
    HAS_PERSES = False


def error_if_no_perses(name: str):
    return error_if_no_package(name, 'perses', HAS_PERSES)

"""
 ===========================================================================
    SOLUTION 2
 ===========================================================================
"""
# Decorator Test:
def requires_package(package_name: str):
    """
    Helper function to denote that a funciton requires some optional
    dependency. A function decorated with this decorator will raise
    `MissingDependencyError` if the package is not found by
    `importlib.import_module()`.
    Parameters
    ----------
    package_name : str
        The directory path to enter within the context
    Raises
    ------
    MissingDependencyError
    """

    def test_import_for_require_package(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            import importlib

            try:
                importlib.import_module(package_name)
            except (ImportError, ModuleNotFoundError):
                raise error_if_no_package(name=function.__name__,
                                          package_name=package_name,
                                          has_package=False)
            except Exception as e:
                raise e

            return function(*args, **kwargs)

        return wrapper

    return test_import_for_require_package


def requires_license_for_openeye(function):
    """
    Wrapper function to denote that a function requires the openeye
    license. A function decorated with this decorator will raise
    `LicenseError` if the openeye license is not around or invalid.

    Parameters
    ----------
    required_by : str
        name of the function, that requires the license

    Raises
    ------
    LicenseError
        if openeye license is missing or invalid.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        import importlib
        _license_functions_openeye = {
            "oechem": "OEChemIsLicensed",
            "oequacpac": "OEQuacPacIsLicensed",
            "oeiupac": "OEIUPACIsLicensed",
            "oeomega": "OEOmegaIsLicensed",
        }
        # Check if all licenses are here
        license_status = []
        for tool, test_function in _license_functions_openeye.items():
            try:
                module = importlib.import_module("openeye." + tool)
            except (ImportError, ModuleNotFoundError):
                continue
            else:
                license_status.append(getattr(module, test_function)())

        # evaluate the result
        if (not all(license_status)):
            error_if_no_license(name=function.__name__,
                                license_name="Openeye Toolkit License",
                                has_license=False)
        else:
            return function(*args, **kwargs)

    return wrapper

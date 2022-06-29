"""
Tools for integration with miscellaneous non-required packages.
shamelessly borrowed from openpathsampling author: dwhswenson
"""
from openff.toolkit.utils import toolkits
from openff.toolkit.utils.exceptions import LicenseError


def error_if_no_license(name:str, license_name:str, has_license:bool):
    if not has_license:
        raise LicenseError(msg=name+" requires the "+license_name+" license, "
                           "which was not found!")


def error_if_no_package(name:str, package_name:str, has_package:bool):
    if not has_package:
        raise RuntimeError(name + " requires " + package_name
                           + ", which is not installed!")


# Licenses
# OpenEye ###################################################

if (not toolkits.OPENEYE_AVAILABLE):
    HAS_OPENEYE_TOOLKIT_LICENSE = False
else:
    HAS_OPENEYE_TOOLKIT_LICENSE = True


def error_if_no_openeye_license(name:str):
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


def error_if_no_perses(name:str):
    return error_if_no_package(name, 'perses', HAS_PERSES)
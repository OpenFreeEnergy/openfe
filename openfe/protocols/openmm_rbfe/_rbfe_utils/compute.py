# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
# Adapted Perses' perses.app.setup_relative_calculation.get_openmm_platform
import warnings
import logging


logger = logging.getLogger(__name__)


def get_openmm_platform(platform_name=None):
    """
    Return OpenMM's platform object based on given name. Setting to mixed
    precision if using CUDA or OpenCL.

    Parameters
    ----------
    platform_name : str, optional, default=None
        String with the platform name. If None, it will use the fastest
        platform supporting mixed precision.

    Returns
    -------
    platform : openmm.Platform
        OpenMM platform object.
    """
    if platform_name is None:
        # No platform is specified, so retrieve fastest platform that supports
        # 'mixed' precision
        from openmmtools.utils import get_fastest_platform
        platform = get_fastest_platform(minimum_precision='mixed')
    else:
        from openmm import Platform
        platform = Platform.getPlatformByName(platform_name)
    # Set precision and properties
    name = platform.getName()
    if name in ['CUDA', 'OpenCL']:
        platform.setPropertyDefaultValue(
                'Precision', 'mixed')
    if name == 'CUDA':
        platform.setPropertyDefaultValue(
                'DeterministicForces', 'true')

    if name != 'CUDA':
        wmsg = (f"Non-GPU platform selected: {name}, this may significantly "
                "impact simulation performance")
        warnings.warn(wmsg)
        logging.warning(wmsg)

    return platform

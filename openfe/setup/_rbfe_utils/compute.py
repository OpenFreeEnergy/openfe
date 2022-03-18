# A copy of Perses' perses.app.setup_relative_calculation.get_openmm_platform

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
    if name in ['CUDA']:
        platform.setPropertyDefaultValue(
                'DeterministicForces', 'true')

    return platform

# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
# Adapted Perses' perses.app.setup_relative_calculation.get_openmm_platform
from typing import Optional
import warnings
import logging
import os


logger = logging.getLogger(__name__)


def get_openmm_platform(
    platform_name: Optional[str] = None,
    gpu_device_index: Optional[list[int]] = None,
    restrict_cpu_count: bool = False
):
    """
    Return OpenMM's platform object based on given name. Setting to mixed
    precision if using CUDA or OpenCL.

    Parameters
    ----------
    platform_name : Optional[str]
        String with the platform name. If None, it will use the fastest
        platform supporting mixed precision.
        Default ``None``.
    gpu_device_index : Optional[list[str]]
        GPU device index selection. If ``None`` the default OpenMM
        GPU selection will be used.
        See the `OpenMM platform properties documentation <http://docs.openmm.org/latest/userguide/library/04_platform_specifics.html>`_
        for more details.
        Default ``None``.
    restrict_cpu_count : bool
        Optional hint to restrict the CPU count to 1 when
        ``platform_name`` is CPU. This allows Protocols to ensure
        that no large performance in cases like vacuum simulations.

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
        try:
            platform_name = {
                'cpu': 'CPU',
                'opencl': 'OpenCL',
                'cuda': 'CUDA',
            }[str(platform_name).lower()]
        except KeyError:
            pass

        from openmm import Platform
        platform = Platform.getPlatformByName(platform_name)
    # Set precision and properties
    name = platform.getName()
    if name in ['CUDA', 'OpenCL']:
        platform.setPropertyDefaultValue('Precision', 'mixed')
        if gpu_device_index is not None:
            index_list = ','.join(str(i) for i in gpu_device_index)
            platform.setPropertyDefaultValue('DeviceIndex', index_list)

    if name == 'CUDA':
        platform.setPropertyDefaultValue(
                'DeterministicForces', 'true')

    if name != 'CUDA':
        wmsg = (f"Non-CUDA platform selected: {name}, this may significantly "
                "impact simulation performance")
        warnings.warn(wmsg)
        logging.warning(wmsg)

    if name == 'CPU' and restrict_cpu_count:
        threads = os.getenv("OPENMM_CPU_THREADS", '1')
        platform.setPropertyDefaultValue('Threads', threads)

    return platform

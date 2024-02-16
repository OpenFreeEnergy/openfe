# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from contextlib import contextmanager

from openff.toolkit import GLOBAL_TOOLKIT_REGISTRY, OpenEyeToolkitWrapper
from openff.toolkit.utils.toolkit_registry import ToolkitUnavailableException


@contextmanager
def without_oechem_backend():
    """For temporarily removing oechem from openff's toolkit registry"""
    current_toolkits = [type(tk) for tk in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits]

    try:
        GLOBAL_TOOLKIT_REGISTRY.deregister_toolkit(OpenEyeToolkitWrapper())
    except ToolkitUnavailableException:
        pass

    try:
        yield None
    finally:
        # this is order dependent; we want to prepend OEChem back to first
        while GLOBAL_TOOLKIT_REGISTRY.registered_toolkits:
            GLOBAL_TOOLKIT_REGISTRY.deregister_toolkit(GLOBAL_TOOLKIT_REGISTRY.registered_toolkits[0])
        for tk in current_toolkits:
            GLOBAL_TOOLKIT_REGISTRY.register_toolkit(tk)

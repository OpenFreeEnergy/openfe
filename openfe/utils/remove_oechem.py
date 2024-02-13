# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from openff.toolkit.utils.toolkit_registry import (
    ToolkitRegistry,
    toolkit_registry_manager,
)
from openff.toolkit.utils.toolkits import (
    AmberToolsToolkitWrapper,
    BuiltInToolkitWrapper,
    RDKitToolkitWrapper,
)

from contextlib import contextmanager

without_oechem = toolkit_registry_manager(
    toolkit_registry=ToolkitRegistry(
        toolkit_precedence=[
            RDKitToolkitWrapper(),
            AmberToolsToolkitWrapper(),
            BuiltInToolkitWrapper(),
        ]
    )
)

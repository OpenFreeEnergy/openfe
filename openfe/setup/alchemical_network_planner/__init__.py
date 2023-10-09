# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Tools for planning alchemical networks.

This module is being replaced by methods on :class:`LigandNetwork
<openfe.setup.LigandNetwork>` like :meth:`to_rbfe_alchemical_network()
<openfe.setup.LigandNetwork.to_rbfe_alchemical_network>`.
"""

from .relative_alchemical_network_planner import (
    RHFEAlchemicalNetworkPlanner,
    RBFEAlchemicalNetworkPlanner,
)

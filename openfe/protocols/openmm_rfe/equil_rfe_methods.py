# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy Protocol using OpenMM and OpenMMTools in a
Perses-like manner.

This module implements the necessary tooling to run calculate the
relative free energy of a ligand transformation using OpenMM tools and one of
the following methods:
    - Hamiltonian Replica Exchange
    - Self-adjusted mixture sampling
    - Independent window sampling

Acknowledgements
----------------
This Protocol is based on, and leverages components originating from
the Perses toolkit (https://github.com/choderalab/perses).
"""

from .equil_rfe_settings import RelativeHybridTopologyProtocolSettings
from .hybridtop_protocol_results import RelativeHybridTopologyProtocolResult
from .hybridtop_units import RelativeHybridTopologyProtocolUnit
from .hybridtop_protocols import RelativeHybridTopologyProtocol

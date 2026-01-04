# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from openfe.protocols.openmm_afe import (
    AbsoluteSolvationProtocol,
    AHFESolventSetupUnit,
    AHFEVacuumSetupUnit,
    AHFESolventSimUnit,
    AHFEVacuumSimUnit,
    AHFESolventAnalysisUnit,
    AHFEVacuumAnalysisUnit,
)


UNIT_TYPES = {
    'solvent': {
        'setup': AHFESolventSetupUnit,
        'sim': AHFESolventSimUnit,
        'analysis': AHFESolventAnalysisUnit,
    },
    'vacuum': {
        'setup': AHFEVacuumSetupUnit,
        'sim': AHFEVacuumSimUnit,
        'analysis': AHFEVacuumAnalysisUnit,
    }
}


def _get_units(protocol_units, unit_type):
    """
    Helper method to extract setup units.
    """
    return [pu for pu in protocol_units if isinstance(pu, unit_type)]

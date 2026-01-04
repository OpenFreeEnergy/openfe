# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from openfe.protocols.openmm_afe.abfe_units import (
    ABFEComplexSetupUnit,
    ABFEComplexSimUnit,
    ABFEComplexAnalysisUnit,
    ABFESolventSetupUnit,
    ABFESolventSimUnit,
    ABFESolventAnalysisUnit,
)


UNIT_TYPES = {
    'solvent': {
        'setup': ABFESolventSetupUnit,
        'sim': ABFESolventSimUnit,
        'analysis': ABFESolventAnalysisUnit,
    },
    'complex': {
        'setup': ABFEComplexSetupUnit,
        'sim': ABFEComplexSimUnit,
        'analysis': ABFEComplexAnalysisUnit,
    }
}


def _get_units(protocol_units, unit_type):
    """
    Helper method to extract setup units.
    """
    return [pu for pu in protocol_units if isinstance(pu, unit_type)]

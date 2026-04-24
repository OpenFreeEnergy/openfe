# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from openfe.protocols.openmm_septop import (
    SepTopComplexAnalysisUnit,
    SepTopComplexRunUnit,
    SepTopComplexSetupUnit,
    SepTopSolventAnalysisUnit,
    SepTopSolventRunUnit,
    SepTopSolventSetupUnit,
)

UNIT_TYPES = {
    "solvent": {
        "setup": SepTopSolventSetupUnit,
        "sim": SepTopSolventRunUnit,
        "analysis": SepTopSolventAnalysisUnit,
    },
    "complex": {
        "setup": SepTopComplexSetupUnit,
        "sim": SepTopComplexRunUnit,
        "analysis": SepTopComplexAnalysisUnit,
    },
}


def _get_units(protocol_units, unit_type):
    """
    Helper method to extract setup units.
    """
    return [pu for pu in protocol_units if isinstance(pu, unit_type)]

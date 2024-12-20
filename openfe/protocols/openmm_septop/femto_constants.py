"""Constant values such as common force groups and names."""

import enum

LIGAND_1_RESIDUE_NAME = "L1"
"""The standard residue name to assign to ligand 1 of a FE calculation."""
LIGAND_2_RESIDUE_NAME = "R1"
"""The standard residue name to assign to ligand 2 of a FE calculation."""


class OpenMMForceGroup(enum.IntEnum):
    """Standard force groups to assign to common OpenMM forces to make them easier to
    identify."""

    BOND = 0
    ANGLE = 1
    DIHEDRAL = 2

    NONBONDED = 3

    COM_RESTRAINT = 4
    POSITION_RESTRAINT = 5
    ALIGNMENT_RESTRAINT = 6

    BAROSTAT = 7

    ATM = 8

    OTHER = 16


class OpenMMForceName(str, enum.Enum):
    """Standard names use for common OpenMM forces to make them easier to identify."""

    COM_RESTRAINT = "com-restraint"
    POSITION_RESTRAINT = "position-restraint"
    ALIGNMENT_RESTRAINT = "alignment-restraint"


class OpenMMPlatform(str, enum.Enum):
    """The available OpenMM platforms to run using."""

    REFERENCE = "Reference"
    CPU = "CPU"
    OPENCL = "OpenCL"
    CUDA = "CUDA"

    def __str__(self):
        return self.value
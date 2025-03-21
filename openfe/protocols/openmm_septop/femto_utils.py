import numpy
import openmm
import openmm.app
import openmm.unit

from .femto_constants import OpenMMPlatform

def is_close(
    v1: openmm.unit.Quantity,
    v2: openmm.unit.Quantity,
    rtol=1.0e-5,
    atol=1.0e-8,
    equal_nan=False,
) -> bool | numpy.ndarray:
    """Compares if two unit wrapped values are close using ``numpy.is_close``"""

    if not v1.unit.is_compatible(v2.unit):
        return False
    return numpy.isclose(
        v1.value_in_unit(v1.unit),
        v2.value_in_unit(v1.unit),
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
    )

def compute_energy(
    system: openmm.System,
    positions: openmm.unit.Quantity,
    box_vectors: openmm.unit.Quantity | None,
    context_params: dict[str, float] | None = None,
    platform: OpenMMPlatform = OpenMMPlatform.REFERENCE,
    groups: int | set[int] = -1,
) -> openmm.unit.Quantity:
    """Computes the potential energy of a system at a given set of positions.

    Args:
        system: The system to compute the energy of.
        positions: The positions to compute the energy at.
        box_vectors: The box vectors to use if any.
        context_params: Any global context parameters to set.
        platform: The platform to use.
        groups: The force groups to include in the energy calculation.

    Returns:
        The computed energy.
    """
    context_params = context_params if context_params is not None else {}

    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.0001 * openmm.unit.femtoseconds),
        openmm.Platform.getPlatformByName(str(platform)),
    )

    for key, value in context_params.items():
        context.setParameter(key, value)

    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)

    return context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
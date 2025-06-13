import numpy
import openmm
import openmm.app
import openmm.unit


def create_context(system, integrator, platform=None):
    """Create a Context."""
    if platform is None:
        context = openmm.Context(system, integrator)
    if platform is not None:
        context = openmm.Context(system, integrator, platform)
    return context


def compute_energy(
    system: openmm.System,
    positions: openmm.unit.Quantity,
    box_vectors: openmm.unit.Quantity | None,
    context_params: dict[str, float] | None = None,
    platform=None,
    groups: int | set[int] = -1,
) -> openmm.unit.Quantity:
    """
    Computes the potential energy of a system at a given set of positions.

    Parameters
    ----------
    system: openmm.System
      The system to compute the energy of.
    positions: openmm.unit.Quantity
      The positions to compute the energy at.
    box_vectors: openmm.unit.Quantity
      The box vectors to use if any.
    context_params: dict[str, float]
      Any global context parameters to set.
    platform: str
      The platform to use.
    groups: int
      The force groups to include in the energy calculation.

    Returns
    -------
    energy : openmm.unit.Quantity
        The computed energy.
    """
    context_params = context_params if context_params is not None else {}

    context = create_context(
        system,
        openmm.VerletIntegrator(0.0001 * openmm.unit.femtoseconds),
        platform,
    )

    for key, value in context_params.items():
        context.setParameter(key, value)

    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)

    energy = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()

    return energy
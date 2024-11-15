import collections
import copy
import itertools

import numpy
import openmm
import openmm.unit

# import femto.fe.config

# This code was obtained and modified from https://github.com/Psivant/femto

LAMBDA_VDW_LIGAND_1 = "lambda_vdw_lig_1"
"""The global parameter used to scale the vdW interactions of ligand 1."""
LAMBDA_VDW_LIGAND_2 = "lambda_vdw_lig_2"
"""The global parameter used to scale the vdW interactions of ligand 2."""

LAMBDA_CHARGES_LIGAND_1 = "lambda_charges_lig_1"
"""The global parameter used to scale the electrostatic interactions of
ligand 1."""
LAMBDA_CHARGES_LIGAND_2 = "lambda_charges_lig_2"
"""The global parameter used to scale the electrostatic interactions of
ligand 2."""


_SUPPORTED_FORCES = [
    openmm.HarmonicBondForce,
    openmm.HarmonicAngleForce,
    openmm.PeriodicTorsionForce,
    openmm.NonbondedForce,
    openmm.MonteCarloBarostat,
    openmm.CMMotionRemover,
]


def _beutler_softcore_potential(variable: str) -> str:
    return (
        f"4.0 * (1.0 - {variable}) * eps * x * (x - 1.0);"
        #
        f"x = (sig / r_eff) ^ 6;"
        #
        f"r_eff = (0.5 * sig^6 * {variable} + r^6)^(1/6);"
        #
        f"sig = 0.5 * (sig1 + sig2);"
        f"eps = sqrt(eps1 * eps2);"
    )


def _validate_exceptions(
    force: openmm.NonbondedForce, ligand_1_idxs: set[int], ligand_2_idxs: set[int]
):
    """Validate that a non-bonded force's exceptions are consistent with the
    quite strict assumptions we make when setting up FEP"""

    for i in range(force.getNumExceptions()):
        idx_a, idx_b, *_ = force.getExceptionParameters(i)

        if idx_a in ligand_1_idxs and idx_b in ligand_1_idxs:
            continue
        if idx_a in ligand_2_idxs and idx_b in ligand_2_idxs:
            continue
        if (
            idx_a not in ligand_1_idxs
            and idx_b not in ligand_1_idxs
            and idx_a not in ligand_2_idxs
            and idx_b not in ligand_2_idxs
        ):
            continue

        raise NotImplementedError("alchemical-chemical exceptions were not expected")


def _convert_intramolecular_interactions(
    force: openmm.NonbondedForce,
    ligand_1_idxs: set[int],
    ligand_2_idxs: set[int],
):
    """Converts all intra-molecular interactions in alchemical molecules to exceptions
    so that they won't be scaled, and excludes ligand-ligand interactions.

    This means we only need a single custom nonbonded force to handle the soft-core
    vdW interactions between alchemical and non-alchemical particles.

    Args:
        force: The force to modify.
        ligand_1_idxs: The indices of atoms belonging to the first ligand.
        ligand_2_idxs: The indices of atoms belonging to the second ligand.
    """

    existing_exceptions = {
        tuple(sorted(force.getExceptionParameters(i)[:2])): i
        for i in range(force.getNumExceptions())
    }
    assert len(existing_exceptions) == force.getNumExceptions()

    ligand_idxs = [ligand_1_idxs, ligand_2_idxs]

    # add exceptions for intramolecular interactions
    for idxs in ligand_idxs:
        for idx_a, idx_b in itertools.combinations(idxs, r=2):
            pair = tuple(sorted((idx_a, idx_b)))

            if pair in existing_exceptions:
                continue

            charge_a, sigma_a, epsilon_a = force.getParticleParameters(idx_a)
            charge_b, sigma_b, epsilon_b = force.getParticleParameters(idx_b)

            epsilon_ab = (
                numpy.sqrt((epsilon_a * epsilon_b).value_in_unit(epsilon_a.unit**2))
                * epsilon_a.unit
            )
            sigma_ab = 0.5 * (sigma_a + sigma_b)
            charge_ab = charge_a * charge_b

            existing_exceptions[pair] = force.addException(
                *pair, charge_ab, sigma_ab, epsilon_ab
            )

    # add exceptions for ligand-ligand interactions
    for idx_a, idx_b in itertools.product(ligand_1_idxs, ligand_2_idxs):
        force.addException(idx_a, idx_b, 0.0, 1.0, 0.0, replace=True)


def _apply_lambda_charge(
    force: openmm.NonbondedForce, ligand_idxs: set[int], variable: str
):
    """Modifies a standard non-bonded force so that electrostatic interactions of
    the specified atoms are scaled by ``variable``.

    Any alchemical-chemical interactions will be linearly scaled while chemical-chemical
    and alchemical-alchemical interactions will remain unchanged.

    Args:
        force: The force to modify.
        ligand_idxs: The indices of atoms belonging to the ligand.
        variable: The global parameter to use for scaling.
    """

    if len(ligand_idxs) == 0:
        return

    force.addGlobalParameter(variable, 0.0)

    for i in range(force.getNumParticles()):
        if i not in ligand_idxs:
            continue

        charge, sigma, epsilon = force.getParticleParameters(i)

        if numpy.isclose(charge.value_in_unit(openmm.unit.elementary_charge), 0.0):
            # We don't need to scale already zero charges
            continue

        # q should be 0.0 at λ=1 and q at λ=0, so we set q - λq
        force.addParticleParameterOffset(variable, i, -charge, 0.0, 0.0)
        force.setParticleParameters(i, charge, sigma, epsilon)


def _apply_lambda_vdw(
    force: openmm.NonbondedForce,
    ligand_idxs: set[int],
    non_ligand_idxs: set[int],
    variable: str,
) -> openmm.CustomNonbondedForce | None:
    """Modifies a standard non-bonded force so that vdW interactions of the specified
    atoms are scaled by ``variable``.

    Any alchemical-chemical interactions will be scaled using a Beutler-style soft core
    potential while chemical-chemical and alchemical-alchemical interactions will remain
    unchanged.

    Args:
        force: The force to modify.
        ligand_idxs: The indices of the ligand atoms whose interactions should be scaled
        non_ligand_idxs: The indices of the remaining atoms that can be interacted with.
        variable: The global parameter to use for scaling.

    Returns:
        A custom non-bonded force that contains only chemical-alchemical interactions.
    """

    if len(ligand_idxs) == 0:
        return

    custom_vdw_fn = _beutler_softcore_potential(variable)

    custom_force = openmm.CustomNonbondedForce(custom_vdw_fn)
    custom_force.setNonbondedMethod(
        force.getNonbondedMethod()
        if int(force.getNonbondedMethod()) not in {3, 4, 5}
        else openmm.CustomNonbondedForce.CutoffPeriodic
    )
    custom_force.setCutoffDistance(force.getCutoffDistance())
    custom_force.setSwitchingDistance(force.getSwitchingDistance())
    custom_force.setUseSwitchingFunction(force.getUseSwitchingFunction())
    custom_force.setUseLongRangeCorrection(force.getUseDispersionCorrection())

    custom_force.addGlobalParameter(variable, 0.0)

    custom_force.addPerParticleParameter("eps")
    custom_force.addPerParticleParameter("sig")

    for index in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(index)
        custom_force.addParticle([epsilon, sigma])

        if index not in ligand_idxs:
            continue

        # the intermolecular alchemical interactions will be handled by the custom
        # soft-core force, so we zero them out here.
        force.setParticleParameters(index, charge, sigma, epsilon * 0)

    for index in range(force.getNumExceptions()):
        index_a, index_b, *_ = force.getExceptionParameters(index)
        # let the exceptions be handled by the original force as we don't intend to
        # annihilate those while switching off the intermolecular vdW interactions
        custom_force.addExclusion(index_a, index_b)

    # alchemical-chemical
    if len(non_ligand_idxs) > 0 and len(ligand_idxs) > 0:
        custom_force.addInteractionGroup(ligand_idxs, non_ligand_idxs)
    # alchemical-alchemical (e.g. ion pairs)
    # if len(ligand_1_idxs) > 0 and len(ligand_2_idxs) > 0:
    #     custom_force.addInteractionGroup(ligand_1_idxs, ligand_2_idxs)

    return custom_force


def _apply_nonbonded_lambdas(
    force: openmm.NonbondedForce,
    ligand_1_idxs: set[int],
    ligand_2_idxs: set[int] | None,
    # config: femto.fe.config.FEP,
) -> tuple[openmm.NonbondedForce | openmm.CustomNonbondedForce, ...]:
    """Modifies a standard non-bonded force so that vdW and electrostatic interactions
    are scaled by ``lambda_vdw`` and ``lambda_charges`` respectively.

    Args:
        force: The original non-bonded force.
        ligand_1_idxs: The indices of the ligand atoms whose interactions should be
            scaled by lambda.
        ligand_2_idxs: The indices of the ligand atoms whose interactions should be
            scaled by 1 - lambda.
        config: Configuration options.

    Returns:
        The modified non-bonded force containing chemical-chemical and
        alchemical-alchemical interactions and two custom non-bonded force containing
        only alchemical-chemical interactions for ligand 1 and 2 respectively.
    """
    force = copy.deepcopy(force)


    assert (
        force.getNumGlobalParameters() == 0
    ), "the non-bonded force should not already contain global parameters"
    assert (
        force.getNumParticleParameterOffsets() == 0
        and force.getNumExceptionParameterOffsets() == 0
    ), "the non-bonded force should not already contain parameter offsets"

    ligand_2_idxs = set() if ligand_2_idxs is None else ligand_2_idxs

    _validate_exceptions(force, ligand_1_idxs, ligand_2_idxs)
    _convert_intramolecular_interactions(force, ligand_1_idxs, ligand_2_idxs)

    _apply_lambda_charge(force, ligand_1_idxs, LAMBDA_CHARGES_LIGAND_1)
    _apply_lambda_charge(force, ligand_2_idxs, LAMBDA_CHARGES_LIGAND_2)


    non_ligand_indices = {
        i
        for i in range(force.getNumParticles())
        if i not in ligand_1_idxs and i not in ligand_2_idxs
    }

    custom_force_1 = _apply_lambda_vdw(
        force, ligand_1_idxs, non_ligand_indices, LAMBDA_VDW_LIGAND_1
    )
    custom_force_2 = _apply_lambda_vdw(
        force, ligand_2_idxs, non_ligand_indices, LAMBDA_VDW_LIGAND_2
    )
    return force, custom_force_1, custom_force_2


def apply_fep(
    system: openmm.System,
    ligand_1_idxs: set[int],
    ligand_2_idxs: set[int] | None,
    # config: femto.fe.config.FEP,
):
    """Modifies an OpenMM system so that different interactions can be scaled by
    corresponding lambda parameters.

    Notes:
        * All intra-molecular interactions of alchemical ligands are currently replaced
          with exceptions and are not scaled by lambda parameters. This will lead to
          slightly different energies from the original system as cutoffs are not
          applied to them by OpenMM.
        * LJ vdW interactions between ligands and the rest of the system will be
          replaced with a Beutler-style soft-core LJ potential with a-b-c of 1-1-6 and
          alpha=0.5.
        * Ligand indices **must** correspond to **all** atoms in the ligand,
          alchemically modifying part of a molecule is not yet supported.

    Args:
        system: The system to modify in-place.
        ligand_1_idxs: The indices of the ligand atoms whose interactions should be
            scaled by lambda.
        ligand_2_idxs: The indices of the ligand atoms whose interactions should be
            scaled by 1 - lambda.
        config: Configuration options.
    """

    forces_by_type = collections.defaultdict(list)

    for force in system.getForces():
        if type(force) not in _SUPPORTED_FORCES:
            raise ValueError(
                f"Force type {type(force)} is not supported when alchemical modifying "
                f"a system."
            )
        forces_by_type[type(force)].append(force)

    updated_forces = []

    for force_type, forces in forces_by_type.items():
        if len(forces) != 1:
            raise NotImplementedError("only one force of each type is supported.")

        force = forces[0]

        if force_type == openmm.NonbondedForce:
            nonbonded_forces = _apply_nonbonded_lambdas(
                force, ligand_1_idxs, ligand_2_idxs,
            )
            updated_forces.extend(nonbonded_forces)
        else:
            updated_forces.append(copy.deepcopy(force))

    for i in reversed(range(system.getNumForces())):
        system.removeForce(i)

    for force in updated_forces:
        if force is not None:
            system.addForce(force)

# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utility methods to validate input systems to OpenMM-based alchemical
Protocols.
"""

from typing import Optional, Tuple

from gufe import (
    ChemicalSystem,
    Component,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
import numpy as np
from openff.toolkit import Molecule as OFFMol
import openmm


def get_alchemical_components(
    stateA: ChemicalSystem,
    stateB: ChemicalSystem,
) -> dict[str, list[Component]]:
    """
    Checks the equality between Components of two end state ChemicalSystems
    and identify which components do not match.

    Parameters
    ----------
    stateA : ChemicalSystem
      The chemical system of end state A.
    stateB : ChemicalSystem
      The chemical system of end state B.

    Returns
    -------
    alchemical_components : dict[str, list[Component]]
      Dictionary containing a list of alchemical components for each state.

    Raises
    ------
    ValueError
      If there are any duplicate components in states A or B.
    """
    matched_components: dict[Component, Component] = {}
    alchemical_components: dict[str, list[Component]] = {
        "stateA": [],
        "stateB": [],
    }

    for keyA, valA in stateA.components.items():
        for keyB, valB in stateB.components.items():
            if valA == valB:
                if valA not in matched_components.keys():
                    matched_components[valA] = valB
                else:
                    # Could be that either we have a duplicate component
                    # in stateA or in stateB
                    errmsg = (
                        f"state A components {keyA}: {valA} matches "
                        "multiple components in stateA or stateB"
                    )
                    raise ValueError(errmsg)

    # populate stateA alchemical components
    for valA in stateA.components.values():
        if valA not in matched_components.keys():
            alchemical_components["stateA"].append(valA)

    # populate stateB alchemical components
    for valB in stateB.components.values():
        if valB not in matched_components.values():
            alchemical_components["stateB"].append(valB)

    return alchemical_components


def validate_solvent(state: ChemicalSystem, nonbonded_method: str):
    """
    Checks that the ChemicalSystem component has the right solvent
    composition for an input nonbonded_methtod.

    Parameters
    ----------
    state : ChemicalSystem
      The chemical system to inspect.
    nonbonded_method : str
      The nonbonded method to be applied for the simulation.

    Raises
    ------
    ValueError
      * If there are multiple SolventComponents in the ChemicalSystem.
      * If there is a SolventComponent and the `nonbonded_method` is
        `nocutoff`.
      * If the SolventComponent solvent is not water.
    """
    solv_comps = state.get_components_of_type(SolventComponent)

    if len(solv_comps) > 0:
        if nonbonded_method.lower() == "nocutoff":
            errmsg = "nocutoff cannot be used for solvent transformations"
            raise ValueError(errmsg)

        if len(solv_comps) > 1:
            errmsg = "Multiple SolventComponent found, only one is supported"
            raise ValueError(errmsg)

        if solv_comps[0].smiles != "O":
            errmsg = "Non water solvent is not currently supported"
            raise ValueError(errmsg)
    else:
        if nonbonded_method.lower() == "pme":
            errmsg = "PME cannot be used for vacuum transform"
            raise ValueError(errmsg)


def validate_protein(state: ChemicalSystem):
    """
    Checks that the ChemicalSystem's ProteinComponent are suitable for the
    alchemical protocol.

    Parameters
    ----------
    state : ChemicalSystem
      The chemical system to inspect.

    Raises
    ------
    ValueError
      If there are multiple ProteinComponent in the ChemicalSystem.
    """
    prot_comps = state.get_components_of_type(ProteinComponent)

    if len(prot_comps) > 1:
        errmsg = "Multiple ProteinComponent found, only one is supported"
        raise ValueError(errmsg)


ParseCompRet = Tuple[
    Optional[SolventComponent],
    Optional[ProteinComponent],
    list[SmallMoleculeComponent],
]


def get_components(state: ChemicalSystem) -> ParseCompRet:
    """
    Establish all necessary Components for the transformation.

    Parameters
    ----------
    state : ChemicalSystem
      ChemicalSystem to get all necessary components from.

    Returns
    -------
    solvent_comp : Optional[SolventComponent]
      If it exists, the SolventComponent for the state, otherwise None.
    protein_comp : Optional[ProteinComponent]
      If it exists, the ProteinComponent for the state, otherwise None.
    small_mols : list[SmallMoleculeComponent]
    """

    def _get_single_comps(state, comptype):
        comps = state.get_components_of_type(comptype)

        if len(comps) > 0:
            return comps[0]
        else:
            return None

    solvent_comp: Optional[SolventComponent] = _get_single_comps(state, SolventComponent)

    protein_comp: Optional[ProteinComponent] = _get_single_comps(state, ProteinComponent)

    small_mols = state.get_components_of_type(SmallMoleculeComponent)

    return solvent_comp, protein_comp, small_mols


def assert_multistate_system_equality(
    ref_system: openmm.System,
    stored_system: openmm.System,
):
    """
    Verify the equality of a MultiStateReporter
    stored system, with that of a pre-exisiting
    standard system.


    Raises
    ------
    ValueError
      * If the particles in the two System don't match.
      * If the constraints in the two System don't match.
      * If the forces in the two systems don't match.
    """

    # Assert particle equality
    def _get_masses(system):
        return np.array(
            [
                system.getParticleMass(i).value_in_unit(openmm.unit.dalton)
                for i in range(system.getNumParticles())
            ]
        )

    ref_masses = _get_masses(ref_system)
    stored_masses = _get_masses(stored_system)

    if not (
        (ref_masses.shape == stored_masses.shape) and
        (np.allclose(ref_masses, stored_masses))
    ):
        errmsg = "Stored checkpoint System particles do not match those of the simulated System"
        raise ValueError(errmsg)

    # Assert constraint equality
    def _get_constraints(system):
        constraints = []
        for index in range(system.getNumConstraints()):
            i, j, d = system.getConstraintParameters(index)
            constraints.append([i, j, d.value_in_unit(openmm.unit.nanometer)])

        return np.array(constraints)

    ref_constraints = _get_constraints(ref_system)
    stored_constraints = _get_constraints(stored_system)
    
    if not (
        (ref_constraints.shape == stored_constraints.shape) and
        (np.allclose(ref_constraints, stored_constraints))
    ):
        errmsg = "Stored checkpoint System constraints do not match those of the simulation System"
        raise ValueError(errmsg)

    # Assert force equality
    # Notes:
    # * Store forces are in different order
    # * The barostat doesn't exactly match because seeds have changed

    # Create dictionaries of forces keyed by their hash
    # Note: we can't rely on names because they may clash
    ref_force_dict = {hash(openmm.XmlSerializer.serialize(f)): f for f in ref_system.getForces()}
    stored_force_dict = {
        hash(openmm.XmlSerializer.serialize(f)): f for f in stored_system.getForces()
    }

    # Assert the number of forces is equal
    if len(ref_force_dict) != len(stored_force_dict):
        errmsg = "Number of forces stored in checkpoint System does not match simulation System"
        raise ValueError(errmsg)

    # Loop through forces and check for equality
    for sfhash, sforce in stored_force_dict.items():
        errmsg = (
            f"Force {sforce.getName()} in the stored checkpoint System "
            "does not match the same force in the simulated System"
        )

        # Barostat case - seed changed so we need to check manually
        if isinstance(sforce, (openmm.MonteCarloBarostat, openmm.MonteCarloMembraneBarostat)):
            # Find the equivalent force in the reference
            rforce = [
                f for f in ref_force_dict.values()
                if isinstance(f, (openmm.MonteCarloBarostat, openmm.MonteCarloMembraneBarostat))
            ][0]

            if (
                (sforce.getFrequency() != rforce.getFrequency())
                or (sforce.getForceGroup() != rforce.getForceGroup())
                or (sforce.getDefaultPressure() != rforce.getDefaultPressure())
                or (sforce.getDefaultTemperature() != rforce.getDefaultTemperature())
            ):
                raise ValueError(errmsg)

        else:
            if sfhash not in ref_force_dict:
                raise ValueError(errmsg)

# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""ABFE Protocol Units --- :mod:`openfe.protocols.openmm_afe.abfe_units`
========================================================================
This module defines the ProtocolUnits for the
:class:`AbsoluteBindingProtocol`.
"""

import logging
import pathlib
from collections.abc import Iterable
from copy import deepcopy

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
from gufe import (
    SolventComponent,
)
from gufe.components import Component, SolvatedPDBComponent
from openff.units import unit as offunit
from openff.units import Quantity
from openff.units.openmm import to_openmm
from openmm import System, HarmonicBondForce
from openmm import unit as ommunit
from openmm.app import Topology as omm_topology
from openmmtools.states import ThermodynamicState
from rdkit import Chem

from openfe.protocols.openmm_afe.equil_afe_settings import (
    ABFEBoreschRestraintSettings,
    SettingsBaseModel,
)
from openfe.protocols.openmm_utils import system_validation
from openfe.protocols.restraint_utils import geometry
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry
from openfe.protocols.restraint_utils.geometry.utils import FindHostAtoms, get_central_atom_idx
from openfe.protocols.restraint_utils.openmm import omm_restraints
from openfe.protocols.restraint_utils.openmm.omm_restraints import BoreschRestraint
from openfe.protocols.restraint_utils.openmm.omm_forces import add_force_in_separate_group

from .base_afe_units import (
    BaseAbsoluteMultiStateAnalysisUnit,
    BaseAbsoluteMultiStateSimulationUnit,
    BaseAbsoluteSetupUnit,
)

logger = logging.getLogger(__name__)


def _get_mda_universe(
    topology: omm_topology,
    positions: ommunit.Quantity | None,
    trajectory: pathlib.Path | None,
) -> mda.Universe:
    """
    Helper method to get a Universe from an openmm Topology,
    and either an input trajectory or a set of positions.

    Parameters
    ----------
    topology : openmm.app.Topology
      An OpenMM Topology that defines the System.
    positions: openmm.unit.Quantity | None
      The System's current positions.
      Used if a trajectory file is None or is not a file.
    trajectory: pathlib.Path | None
      A Path to a trajectory file to read positions from.

    Returns
    -------
    mda.Universe
      An MDAnalysis Universe of the System.
    """
    from MDAnalysis.coordinates.memory import MemoryReader

    # If the trajectory file doesn't exist, then we use positions
    if trajectory is not None and trajectory.is_file():
        return mda.Universe(
            topology,
            trajectory,
            topology_format="OPENMMTOPOLOGY",
        )
    else:
        if positions is None:
            raise ValueError("No positions to create the Universe with")

        # Positions is an openmm Quantity in nm we need
        # to convert to angstroms
        return mda.Universe(
            topology,
            np.array(positions._value) * 10,
            topology_format="OPENMMTOPOLOGY",
            trajectory_format=MemoryReader,
        )


def _get_idxs_from_residxs(
    topology: omm_topology,
    residxs: Iterable[int],
) -> list[int]:
    """
    Helper method to get the a list of atom indices which belong to a list
    of residues.

    Parameters
    ----------
    topology : openmm.app.Topology
      An OpenMM Topology that defines the System.
    residxs : Iterable[int]
      A list of residue numbers who's atoms we should get atom indices.

    Returns
    -------
    atom_ids : list[int]
      A list of atom indices.

    TODO
    ----
    * Check how this works when we deal with virtual sites.
    """
    atom_ids = []

    for r in topology.residues():
        if r.index in residxs:
            atom_ids.extend([at.index for at in r.atoms()])

    return atom_ids


def _get_minimum_image_distance(box_dimensions: npt.NDArray) -> Quantity:
    """
    Get the minimum image distance using the minimum perpendicular width
    of the triclinic vectors.

    Parameters
    ----------
    box_dimensions : npt.NDArray
      The box dimensions as obtained from an MDAnalysis Universe.

    Returns
    -------
    openff.units.Quantity
      The minimum perpendicular width in units of Angstrom.

    Acknowledgements
    ----------------
    Originally contributed by Bendict Tan (@aqemia-benedict-tan).
    """
    from MDAnalysis.lib import mdamath

    box_vectors = mdamath.triclinic_vectors(box_dimensions)

    # Calculate the volume based on the scalar triple product
    volume = mdamath.stp(box_vectors[0], box_vectors[1], box_vectors[2])

    # Now calculate the perpendicular widths using perp_width_i = Volume / Area_of_face_i
    # Where Area_of_face_i is |box_vectors_{i+1} × box_vectors_{i+2}|.
    areas = np.cross(box_vectors[[1, 2, 0]], box_vectors[[2, 0, 1]])
    perp_widths = volume / np.linalg.norm(areas, axis=1)

    return perp_widths.min() * offunit.angstrom


def _find_most_common_ions(
    openmm_topology: omm_topology,
    openmm_system: System,
    target_charge: int,
) -> list[int] | None:
    """
    Get the most common ions of a given net charge in a system.

    Parameters
    ----------
    openmm_topology : openmm.app.Topology
      The Topology of the OpenMM System.
    openmm_system : openmm.System
      The OpenMM System.
    target_charge : int
      The charge the ion should have.

    Returns
    -------
    list[int] | None
      If present, the list of indices matching the most common ion.

    Notes
    -----
    This is similar to what is done in ``_get_ion_parameters`` in
    :mod:`openfe.protocols.openmm_rfe._rfe_utils.topologyhelpers`.
    """
    from collections import Counter, defaultdict

    nbf = [i for i in openmm_system.getForces() if isinstance(i, NonbondedForce)][0]

    ion_counts: Counter = Counter()
    ion_atom_indices: dict[str, int] = defaultdict(list)

    for residue in openmm_topology.residues():
        atoms = list(residue.atoms())

        # We are only interested in single atom counterions
        if len(atoms) != 1:
            continue

        charge, _, _ = nbf.getParticleParameters(atoms[0].index)
        charge_val = charge.value_in_unit(ommunit.elementary_charge)

        if np.isclose(charge_val, target_charge, atol=0.01):
            ion_counts[residue.name] += 1
            ion_atom_indices[residue.name].append(atoms[0].index)

    if ion_counts:
        best_resname = ion_counts.most_common(1)[0][0]
        return ion_atom_indices[best_resname]
    else:
        return None


class ABFESetupUnitMixin:
    """
    Mixin for common class methods between Units
    """

    def _get_alchemical_ions(
        self,
        alchemical_components: dict[str, list[Component]],
        comp_resids: dict[Component, npt.NDArray],
        openmm_topology: omm_topology,
        openmm_system: System,
        positions: ommunit.Quantity,
        settings: dict[str, SettingsBaseModel],
        dry: bool,
    ) -> list[int] | None:
        """
        Find a suitable alchemical ion for a net charge transformation.

        Parameters
        ----------
        alchemical_components: dict[str, list[Component]]
          A dictionary with a list of alchemical components
          in both state A and B.
        comp_resids: dict[Component, npt.NDArray]
          A dictionary keyed by each Component in the System
          which contains arrays with the residue indices that is contained
          by that Component.
        openmm_topology : openmm.app.Topology
          The OpenMM Topology of the system.
        openmm_system : openmm.System
          The OpenMM System to work on.
        positions : openmm.unit.Quantity
          The positions of the system.
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings that defines how to find and set
          the restraint.
        dry: bool
          ``True`` if we are dry-running.

        Returns
        -------
        list[int]
          The indices of the alchemical ions.

        """
        total_charge = alchemical_components["stateA"][0].total_charge

        # Don't add an alchemical ion if we have zero net charge
        # or we didn't request it.
        if total_charge == 0 or not settings["alchemical_settings"].explicit_charge_correction:
            return None

        # TODO: For now, let's stick with a single -1/+1 case, but we should expand to more
        if abs(total_charge) > 1:
            errmsg = "Cannot handle net charge correction on charges greater than one"
            raise ValueError(errmsg)

        # Get the indices of the most common ion type that can act as a counterion
        ion_indices = _find_most_common_ions(openmm_topology, openmm_system, -total_charge)

        if ion_indices is None:
            errmsg = "No suitable ions could be found to act as counterion in the system"
            raise ValueError(errmsg)

        univ = _get_mda_universe(
            openmm_topology,
            positions,
            self.shared_basepath / settings["equil_output_settings"].production_trajectory_filename,
        )

        # get an atomgroup of the possible alchemical ions
        ions_atomgroup = univ.atoms[ion_indices]

        # get the alchemical atoms
        residxs = np.concatenate([comp_resids[key] for key in alchemical_components["stateA"]])
        alchem_idxs = _get_idxs_from_residxs(topology=openmm_topology, residxs=residxs)
        alchem_atomgroup = univ.atoms[alchem_idxs]

        # Get the maximum distance we can use to find ions
        univ.trajectory[-1]  # use the box dimensions from the last frame
        box = univ.dimensions

        if box is None or np.all(np.isinfinite(box)) or np.any(box[:3] <= 0.0):
            # If it's not a dry simulation then error out
            if not dry:
                errmsg = f"Invalid box for co-alchemical ion search: {box}"
                raise ValueError(errmsg)

            # For a dry execution, just assign a super high value
            max_search_distance = 999 * offunit.nanometer
        else:
            # Set the max search distance to half the smallest perpendicular width
            # with a 1 Angstrom padding
            max_search_distance = (_get_minimum_image_distance(box) * 0.5) - 1 * offunit.angstrom

        # Re-using a utility from the restraints utilities
        # TODO: rename this class!
        atom_finder = FindHostAtoms(
            host_atoms=ions_atomgroup,
            guest_atoms=alchem_atomgroup,
            min_search_distance=settings["alchemical_settings"].alchemical_ion_min_distance,
            max_search_distance=max_search_distance,
        )

        # only run on the final frame
        atom_finder.run(frames=[-1])

        if len(atom_finder.results.host_idxs) == 0:
            errmsg = "No suitable alchemical ion was found"
            raise ValueError(errmsg)

        # Just use the first one that comes back ok
        return [atom_finder.results.host_idxs[0]]


class ComplexComponentsMixin:
    def _get_components(self):
        """
        Get the relevant components for a complex transformation.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A dict of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : ProteinComponent | None
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs["stateA"]
        alchem_comps = self._inputs["alchemical_components"]

        solv_comp, prot_comp, small_mols = system_validation.get_components(stateA)
        off_comps = {m: m.to_openff() for m in small_mols}

        # We don't need to check that solv_comp is not None, otherwise
        # an error will have been raised when calling `validate_solvent`
        # in the Protocol's `_create`.
        # Similarly we don't need to check prot_comp

        # If there is an SolvatedPDBComponent, we set the solv_comp
        # in the complex to the SolvatedPDBComponent, as the SolventComponent
        # is only used in the solvent leg
        if isinstance(prot_comp, SolvatedPDBComponent):
            solv_comp = prot_comp

        return alchem_comps, solv_comp, prot_comp, off_comps


class ComplexSettingsMixin:
    def _get_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Extract the relevant settings for a complex transformation.

        Returns
        -------
        settings : dict[str, SettingsBaseModel]
          A dictionary with the following entries:
            * forcefield_settings : OpenMMSystemGeneratorFFSettings
            * thermo_settings : ThermoSettings
            * charge_settings : OpenFFPartialChargeSettings
            * solvation_settings : OpenMMSolvationSettings
            * alchemical_settings : AlchemicalSettings
            * lambda_settings : LambdaSettings
            * engine_settings : OpenMMEngineSettings
            * integrator_settings : IntegratorSettings
            * equil_simulation_settings : MDSimulationSettings
            * equil_output_settings : ABFEPreEquilOutputSettings
            * simulation_settings : SimulationSettings
            * output_settings: MultiStateOutputSettings
            * restraint_settings: BaseRestraintSettings
        """
        prot_settings = self._inputs["protocol"].settings  # type: ignore[attr-defined]

        settings = {}
        settings["forcefield_settings"] = prot_settings.forcefield_settings
        settings["thermo_settings"] = prot_settings.thermo_settings
        settings["charge_settings"] = prot_settings.partial_charge_settings
        settings["solvation_settings"] = prot_settings.complex_solvation_settings
        settings["alchemical_settings"] = prot_settings.alchemical_settings
        settings["lambda_settings"] = prot_settings.complex_lambda_settings
        settings["engine_settings"] = prot_settings.engine_settings
        settings["integrator_settings"] = prot_settings.complex_integrator_settings
        settings["equil_simulation_settings"] = prot_settings.complex_equil_simulation_settings
        settings["equil_output_settings"] = prot_settings.complex_equil_output_settings
        settings["simulation_settings"] = prot_settings.complex_simulation_settings
        settings["output_settings"] = prot_settings.complex_output_settings
        settings["restraint_settings"] = prot_settings.restraint_settings

        return settings


class ABFEComplexSetupUnit(
    ABFESetupUnitMixin, ComplexComponentsMixin, ComplexSettingsMixin, BaseAbsoluteSetupUnit
):
    """
    Setup unit for the complex phase of absolute binding free energy
    transformations.
    """

    simtype = "complex"

    @staticmethod
    def _get_boresch_restraint(
        universe: mda.Universe,
        guest_rdmol: Chem.Mol,
        guest_atom_ids: list[int],
        host_atom_ids: list[int],
        temperature: Quantity,
        settings: ABFEBoreschRestraintSettings,
    ) -> tuple[BoreschRestraintGeometry, BoreschRestraint]:
        """
        Get a Boresch-like restraint Geometry and OpenMM restraint force
        supplier.

        Parameters
        ----------
        universe : mda.Universe
          An MDAnalysis Universe defining the system to get the restraint for.
        guest_rdmol : Chem.Mol
          An RDKit Molecule defining the guest molecule in the system.
        guest_atom_ids: list[int]
          A list of atom indices defining the guest molecule in the universe.
        host_atom_ids : list[int]
          A list of atom indices defining the host molecules in the universe.
        temperature : openff.units.Quantity
          The temperature of the simulation where the restraint will be added.
        settings : ABFEBoreschRestraintSettings
          Settings on how the Boresch-like restraint should be defined.

        Returns
        -------
        geom : BoreschRestraintGeometry
          A class defining the Boresch-like restraint.
        restraint : BoreschRestraint
          A factory class for generating Boresch restraints in OpenMM.
        """
        # Take the minimum of the two possible force constants to check against
        frc_const = min(settings.K_thetaA, settings.K_thetaB)

        geom = geometry.boresch.find_boresch_restraint(
            universe=universe,
            guest_rdmol=guest_rdmol,
            guest_idxs=guest_atom_ids,
            host_idxs=host_atom_ids,
            guest_restraint_atoms_idxs=list(settings.guest_restraint_ids)
            if settings.guest_restraint_ids is not None
            else None,
            host_restraint_atoms_idxs=list(settings.host_restraint_ids)
            if settings.host_restraint_ids is not None
            else None,
            host_selection=settings.host_selection,
            anchor_finding_strategy=settings.anchor_finding_strategy,
            dssp_filter=settings.dssp_filter,
            rmsf_cutoff=settings.rmsf_cutoff,
            host_min_distance=settings.host_min_distance,
            host_max_distance=settings.host_max_distance,
            angle_force_constant=frc_const,
            temperature=temperature,
        )

        restraint = omm_restraints.BoreschRestraint(settings)
        return geom, restraint

    def _add_restraints(
        self,
        system: System,
        topology: omm_topology,
        positions: ommunit.Quantity,
        alchem_comps: dict[str, list[Component]],
        comp_resids: dict[Component, npt.NDArray],
        settings: dict[str, SettingsBaseModel],
        alchemical_ions: list[int] | None,
    ) -> tuple[
        Quantity,
        System,
        geometry.HostGuestRestraintGeometry,
    ]:
        """
        Find and add restraints to the OpenMM System.

        Notes
        -----
        Currently, only Boresch-like restraints are supported.

        Parameters
        ----------
        system : openmm.System
          The System to add the restraint to.
        topology : openmm.app.Topology
          An OpenMM Topology that defines the System.
        positions: openmm.unit.Quantity
          The System's current positions.
          Used if a trajectory file isn't found.
        alchem_comps: dict[str, list[Component]]
          A dictionary with a list of alchemical components
          in both state A and B.
        comp_resids: dict[Component, npt.NDArray]
          A dictionary keyed by each Component in the System
          which contains arrays with the residue indices that is contained
          by that Component.
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings that defines how to find and set
          the restraint.
        alchemical_ions : list[int] | None
          The alchemical ion indices, if they exist.

        Returns
        -------
        correction : openff.units.Quantity
          The standard state correction for the restraint.
        system : openmm.System
          A copy of the System with the restraint added.
        rest_geom : geometry.HostGuestRestraintGeometry
          The restraint Geometry object.

        TODO
        ----
        Add a restraint between the alchemical ion and the guest molecule?
        """
        if self.verbose:
            self.logger.info("Generating restraints")

        # Get the guest rdmol
        guest_rdmol = alchem_comps["stateA"][0].to_rdkit()

        # sanitize the rdmol if possible - warn if you can't
        err = Chem.SanitizeMol(guest_rdmol, catchErrors=True)

        if err:
            msg = "restraint generation: could not sanitize ligand rdmol"
            logger.warning(msg)

        # Get the guest idxs
        # concatenate a list of residue indexes for all alchemical components
        residxs = np.concatenate([comp_resids[key] for key in alchem_comps["stateA"]])

        # get the alchemicical atom ids
        guest_atom_ids = _get_idxs_from_residxs(topology, residxs)

        # Now get the host idxs
        # We assume this is everything but the alchemical component
        # and the solvent.
        solv_comps = [c for c in comp_resids if isinstance(c, SolventComponent)]
        exclude_comps = [alchem_comps["stateA"]] + solv_comps
        residxs = np.concatenate([v for i, v in comp_resids.items() if i not in exclude_comps])

        host_atom_ids = _get_idxs_from_residxs(topology, residxs)

        # Finally create an MDAnalysis Universe
        # We try to pass the equilibration production file path through
        # In some cases (debugging / dry runs) this won't be available
        # so we'll default to using input positions.
        univ = _get_mda_universe(
            topology,
            positions,
            self.shared_basepath / settings["equil_output_settings"].production_trajectory_filename,
        )

        if isinstance(settings["restraint_settings"], ABFEBoreschRestraintSettings):
            rest_geom, restraint = self._get_boresch_restraint(
                univ,
                guest_rdmol,
                guest_atom_ids,
                host_atom_ids,
                settings["thermo_settings"].temperature,
                settings["restraint_settings"],
            )
        else:
            # TODO turn this into a direction for different restraint types supported?
            raise NotImplementedError("Other restraint types are not yet available")

        if self.verbose:
            self.logger.info(f"restraint geometry is: {rest_geom}")

        # We need a temporary thermodynamic state to add the restraint
        # & get the correction
        thermodynamic_state = ThermodynamicState(
            system,
            temperature=to_openmm(settings["thermo_settings"].temperature),
            pressure=to_openmm(settings["thermo_settings"].pressure),
        )

        # Add the force to the thermodynamic state
        restraint.add_force(
            thermodynamic_state,
            rest_geom,
            controlling_parameter_name="lambda_restraints_A",
        )
        # Get the standard state correction as a unit.Quantity
        correction = restraint.get_standard_state_correction(
            thermodynamic_state,
            rest_geom,
        )

        return (
            correction,
            # Remove the thermostat, otherwise you'll get an
            # Andersen thermostat by default!
            thermodynamic_state.get_system(remove_thermostat=True),
            rest_geom,
        )


class ABFEComplexSimUnit(
    ComplexComponentsMixin, ComplexSettingsMixin, BaseAbsoluteMultiStateSimulationUnit
):
    """
    Multi-state simulation (e.g. multi replica methods like Hamiltonian
    replica exchange) unit for the complex phase of absolute binding
    free energy transformations.
    """

    simtype = "complex"


class ABFEComplexAnalysisUnit(ComplexSettingsMixin, BaseAbsoluteMultiStateAnalysisUnit):
    """
    Analysis unit for multi-state simulations with the complex phase
    of absolute binding free energy transformations.
    """

    simtype = "complex"


class SolventComponentsMixin:
    def _get_components(self):
        """
        Get the relevant components for a solvent transformation.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : ProteinComponent | None
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs["stateA"]
        alchem_comps = self._inputs["alchemical_components"]

        solv_comp, prot_comp, small_mols = system_validation.get_components(stateA)
        off_comps = {m: m.to_openff() for m in alchem_comps["stateA"]}

        # We don't need to check that solv_comp is not None, otherwise
        # an error will have been raised when calling `validate_solvent`
        # in the Protocol's `_create`.
        # Similarly we don't need to check prot_comp just return None
        return alchem_comps, solv_comp, None, off_comps


class SolventSettingsMixin:
    def _get_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Extract the relevant settings for a solvent transformation.

        Returns
        -------
        settings : dict[str, SettingsBaseModel]
          A dictionary with the following entries:
            * forcefield_settings : OpenMMSystemGeneratorFFSettings
            * thermo_settings : ThermoSettings
            * charge_settings : OpenFFPartialChargeSettings
            * solvation_settings : OpenMMSolvationSettings
            * alchemical_settings : AlchemicalSettings
            * lambda_settings : LambdaSettings
            * engine_settings : OpenMMEngineSettings
            * integrator_settings : IntegratorSettings
            * equil_simulation_settings : MDSimulationSettings
            * equil_output_settings : ABFEPreEquilOutputSettings
            * simulation_settings : MultiStateSimulationSettings
            * output_settings: MultiStateOutputSettings
        """
        prot_settings = self._inputs["protocol"].settings  # type: ignore[attr-defined]

        settings = {}
        settings["forcefield_settings"] = prot_settings.forcefield_settings
        settings["thermo_settings"] = prot_settings.thermo_settings
        settings["charge_settings"] = prot_settings.partial_charge_settings
        settings["solvation_settings"] = prot_settings.solvent_solvation_settings
        settings["alchemical_settings"] = prot_settings.alchemical_settings
        settings["lambda_settings"] = prot_settings.solvent_lambda_settings
        settings["engine_settings"] = prot_settings.engine_settings
        settings["integrator_settings"] = prot_settings.solvent_integrator_settings
        settings["equil_simulation_settings"] = prot_settings.solvent_equil_simulation_settings
        settings["equil_output_settings"] = prot_settings.solvent_equil_output_settings
        settings["simulation_settings"] = prot_settings.solvent_simulation_settings
        settings["output_settings"] = prot_settings.solvent_output_settings

        return settings


class ABFESolventSetupUnit(
    ABFESetupUnitMixin, SolventComponentsMixin, SolventSettingsMixin, BaseAbsoluteSetupUnit
):
    """
    Setup unit for the solvent phase of absolute binding free energy
    transformations.
    """

    simtype = "solvent"

    def _add_restraints(
        self,
        system: System,
        topology: omm_topology,
        positions: ommunit.Quantity,
        alchem_comps: dict[str, list[Component]],
        comp_resids: dict[Component, npt.NDArray],
        settings: dict[str, SettingsBaseModel],
        alchemical_ions: list[int] | None,
    ) -> tuple[
        Quantity | None,
        System,
        geometry.HostGuestRestraintGeometry | None,
    ]:
        """
        Find and add restraints to the OpenMM System.

        Notes
        -----
        Currently, only Boresch-like restraints are supported.

        Parameters
        ----------
        system : openmm.System
          The System to add the restraint to.
        topology : openmm.app.Topology
          An OpenMM Topology that defines the System.
        positions: openmm.unit.Quantity
          The System's current positions.
          Used if a trajectory file isn't found.
        alchem_comps: dict[str, list[Component]]
          A dictionary with a list of alchemical components
          in both state A and B.
        comp_resids: dict[Component, npt.NDArray]
          A dictionary keyed by each Component in the System
          which contains arrays with the residue indices that is contained
          by that Component.
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings that defines how to find and set
          the restraint.
        alchemical_ions : list[int] | None
          The alchemical ion indices, if they exist.

        Returns
        -------
        correction : openff.units.Quantity | None
          The standard state correction for the restraint.
        system : openmm.System
          A copy of the System with the restraint added.
        rest_geom : geometry.HostGuestRestraintGeometry | None
          The restraint Geometry object.

        TODO
        ----
        Expand to support restraining multiple ions.
        """
        if alchemical_ions is None:
            return None, system, None

        if len(alchemical_ions) > 1:
            errmsg = "Currently cannot handle more than one alchemical ion"
            raise ValueError(errmsg)

        restrained_system = deepcopy(system)

        if self.verbose:
            self.logger.info("Generating restraints for alchemical ions")

        universe = _get_mda_universe(
            topology,
            positions,
            self.shared_basepath / settings["equil_output_settings"].production_trajectory_filename,
        )

        # alchemical ion atom atomgroup
        alchem_ion_ag = universe.atoms[alchemical_ions]

        # get the alchemical ligand atoms
        ligand_rdmol = alchem_comps["stateA"][0].to_rdkit()
        residxs = np.concatenate([comp_resids[key] for key in alchem_comps["stateA"]])
        ligand_alchem_idxs = _get_idxs_from_residxs(topology=topology, residxs=residxs)
        ligand_central_atom = ligand_alchem_idxs[get_central_atom_idx(ligand_rdmol)]
        ligand_central_atom_ag = universe.atoms[ligand_central_atom]

        # Get the ligand-ion distance based on the final frame
        universe.trajectory[-1]

        distance = float(
            calc_bonds(
                alchem_ion_ag.position,
                ligand_central_atom_ag.position,
                box=universe.dimensions,
            )
        )

        spring_constant = to_openmm(
            settings["alchemical_settings"].alchemical_ion_solvent_spring_constant
        )

        force = HarmonicBondForce()

        force.addBond(
            ligand_central_atom,
            alchemical_ions[0],
            distance * ommunit.angstrom,
            spring_constant,
        )

        force.setName("ion_restraint")
        add_force_in_separate_group(restrained_system, force)

        return None, restrained_system, None


class ABFESolventSimUnit(
    SolventComponentsMixin, SolventSettingsMixin, BaseAbsoluteMultiStateSimulationUnit
):
    """
    Multi-state simulation (e.g. multi replica methods like Hamiltonian
    replica exchange) unit for the solvent phase of absolute binding
    free energy transformations.
    """

    simtype = "solvent"


class ABFESolventAnalysisUnit(SolventSettingsMixin, BaseAbsoluteMultiStateAnalysisUnit):
    """
    Analysis unit for multi-state simulations with the solvent phase
    of absolute binding free energy transformations.
    """

    simtype = "solvent"

# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium SepTop RBFE Protocol ---
   :mod:`openfe.protocols.openmm_septop.equil_septop_method`
===============================================================================================================

This module implements the necessary methodology tooling to run a
Separated Topologies RBFE calculation using OpenMM tools and one of the
following alchemical sampling methods:

* Hamiltonian Replica Exchange
* Self-adjusted mixture sampling
* Independent window sampling

Current limitations
-------------------

* Only small molecules are allowed to act as alchemical molecules.
  Alchemically changing protein or solvent components would induce
  perturbations which are too large to be handled by this Protocol.


Acknowledgements
----------------
This Protocol is based on, and leverages components originating from
the SepTop implementation from the Mobleylab
(https://github.com/MobleyLab/SeparatedTopologies) as well as
femto (https://github.com/Psivant/femto).
"""
from __future__ import annotations

import pathlib
import logging
import warnings
from collections import defaultdict
import gufe
import openmm
import openmm.unit
import simtk
import simtk.unit as omm_units
from gufe.components import Component
import itertools
import numpy as np
import numpy.typing as npt
from openff.units import unit
from openff.units.openmm import from_openmm
from openff.toolkit.topology import Molecule as OFFMolecule
from openmmtools import multistate
import mdtraj as md
from typing import Optional, Union
from typing import Any, Iterable
import uuid

from gufe import (
    settings,
    ChemicalSystem, SmallMoleculeComponent,
    ProteinComponent, SolventComponent
)
from openfe.protocols.openmm_septop.equil_septop_settings import (
    SepTopSettings,
    OpenMMSolvationSettings, AlchemicalSettings, LambdaSettings,
    MDSimulationSettings, SepTopEquilOutputSettings,
    MultiStateSimulationSettings, OpenMMEngineSettings,
    IntegratorSettings, MultiStateOutputSettings,
    OpenFFPartialChargeSettings,
    SettingsBaseModel,
)
from ..openmm_utils import system_validation, settings_validation
from .base import BaseSepTopSetupUnit, BaseSepTopRunUnit
from openfe.utils import log_system_probe
from openfe.due import due, Doi
from openff.units.openmm import to_openmm
from rdkit import Chem
import MDAnalysis as mda
from openfe.protocols.restraint_utils.geometry.boresch import (
    find_boresch_restraint, find_guest_atom_candidates)
import openfe.protocols.restraint_utils
from openfe.protocols.restraint_utils.geometry.utils import get_central_atom_idx
from ..restraint_utils.settings import (
    DistanceRestraintSettings,
    BoreschRestraintSettings,
)
from openfe.protocols.restraint_utils.openmm.omm_restraints import (
    BoreschRestraint,
)
from openfe.protocols.restraint_utils.geometry.boresch import (
    BoreschRestraintGeometry,
)
from openfe.protocols.restraint_utils import geometry
from openfe.protocols.restraint_utils.openmm import omm_restraints
from openmmtools.states import ThermodynamicState, GlobalParameterState
from openmmtools.alchemy import (
    AbsoluteAlchemicalFactory,
    AlchemicalRegion,
    AlchemicalState,
)
from .utils import serialize

due.cite(Doi("10.5281/zenodo.596622"),
         description="OpenMMTools",
         path="openfe.protocols.openmm_septop.equil_septop_method",
         cite_module=True)

due.cite(Doi("10.1371/journal.pcbi.1005659"),
         description="OpenMM",
         path="openfe.protocols.openmm_septop.equil_septop_method",
         cite_module=True)


logger = logging.getLogger(__name__)


def _get_mdtraj_from_openmm(omm_topology, omm_positions):
    """
    Get an mdtraj object from an OpenMM topology and positions.

    Parameters
    ----------
    omm_topology: openmm.app.Topology
      The OpenMM topology
    omm_positions: simtk.unit.Quantity
      The OpenMM positions

    Returns
    -------
    mdtraj_system: md.Trajectory
    """
    mdtraj_topology = md.Topology.from_openmm(omm_topology)
    positions_in_mdtraj_format = omm_positions.value_in_unit(omm_units.nanometers)

    unit_cell = omm_topology.getPeriodicBoxVectors() / omm_units.nanometers
    unit_cell_length = np.array([i[inx] for inx, i in enumerate(unit_cell)])
    mdtraj_system = md.Trajectory(positions_in_mdtraj_format,
                                  mdtraj_topology,
                                  unitcell_lengths=unit_cell_length)
    return mdtraj_system


def _check_alchemical_charge_difference(
        ligandA: SmallMoleculeComponent,
        ligandB: SmallMoleculeComponent,
):
    """
    Checks and returns the difference in formal charge between state A
    and B.

    Raises
    ------
    ValueError
      * If a change in net charge is detected.

    Parameters
    ----------
    ligandA: SmallMoleculeComponent
    ligandB: SmallMoleculeComponent
    """
    chg_A = Chem.rdmolops.GetFormalCharge(
        ligandA.to_rdkit()
    )
    chg_B = Chem.rdmolops.GetFormalCharge(
        ligandB.to_rdkit()
    )

    difference = chg_A - chg_B

    if abs(difference) != 0:
        errmsg = (
            f"A charge difference of {difference} is observed "
            "between the end states. Unfortunately this protocol "
            "currently does not support net charge changes.")
        raise ValueError(errmsg)

    return


class SepTopProtocolResult(gufe.ProtocolResult):
    """Dict-like container for the output of a SepTopProtocol"""
    def __init__(self, **data):
        super().__init__(**data)
        # TODO: Detect when we have extensions and stitch these together?
        if any(len(pur_list) > 2 for pur_list
               in itertools.chain(self.data['solvent'].values(), self.data['complex'].values())):
            raise NotImplementedError("Can't stitch together results yet")

    def get_individual_estimates(self) -> dict[str, list[tuple[unit.Quantity, unit.Quantity]]]:
        """
        Get the individual estimate of the free energies.

        Returns
        -------
        dGs : dict[str, list[tuple[unit.Quantity, unit.Quantity]]]
          A dictionary, keyed `solvent` and `complex for each leg
          of the thermodynamic cycle, with lists of tuples containing
          the individual free energy estimates and associated MBAR
          uncertainties for each repeat of that simulation type.
        """
        complex_dGs = []
        complex_correction_dGs_A = []
        complex_correction_dGs_B = []
        solv_dGs = []
        solv_correction_dGs: list[tuple[Any, Any]] = []

        for pus in self.data['complex'].values():
            complex_dGs.append((
                pus[0].outputs['unit_estimate'],
                pus[0].outputs['unit_estimate_error']
            ))

        for pus in self.data['complex_setup'].values():
            complex_correction_dGs_A.append((
                pus[0].outputs['standard_state_correction_A'],
                0 * unit.kilocalorie_per_mole  # correction has no error
            ))
            complex_correction_dGs_B.append((
                pus[0].outputs['standard_state_correction_B'],
                0 * unit.kilocalorie_per_mole  # correction has no error
            ))

        for pus in self.data['solvent'].values():
            solv_dGs.append((
                pus[0].outputs['unit_estimate'],
                pus[0].outputs['unit_estimate_error']
            ))

        for pus in self.data['solvent_setup'].values():
            solv_correction_dGs.append((
                pus[0].outputs['standard_state_correction'],
                0 * unit.kilocalorie_per_mole  # correction has no error
            ))

        return {
            'solvent': solv_dGs,
            'complex': complex_dGs,
            'standard_state_complex_A': complex_correction_dGs_A,
            'standard_state_complex_B': complex_correction_dGs_B,
            'standard_state_solvent': solv_correction_dGs,
        }

    def get_estimate(self):
        """Get the difference in binding free energy estimate for this calculation.

        Returns
        -------
        ddG : unit.Quantity
          The difference in binding free energy.
          This is a Quantity defined with units.
        """
        def _get_average(estimates):
            # Get the unit value of the first value in the estimates
            u = estimates[0][0].u
            # Loop through estimates and get the free energy values
            # in the unit of the first estimate
            ddGs = [i[0].to(u).m for i in estimates]

            return np.average(ddGs) * u

        individual_estimates = self.get_individual_estimates()
        solv_ddG = _get_average(individual_estimates['solvent'])
        complex_ddG = _get_average(individual_estimates['complex'])
        complex_corr_A = _get_average(
            individual_estimates['standard_state_complex_A'])
        complex_corr_B = _get_average(
            individual_estimates['standard_state_complex_B'])
        solv_corr = _get_average(
            individual_estimates['standard_state_solvent'])

        return (complex_ddG + complex_corr_A + complex_corr_B) - (solv_ddG + solv_corr)

    def get_uncertainty(self):
        """Get the relative free energy error for this calculation.

        Returns
        -------
        err : unit.Quantity
          The standard deviation between estimates of the relative binding free
          energy. This is a Quantity defined with units.
        """
        def _get_stdev(estimates):
            # Get the unit value of the first value in the estimates
            u = estimates[0][0].u
            # Loop through estimates and get the free energy values
            # in the unit of the first estimate
            ddGs = [i[0].to(u).m for i in estimates]

            return np.std(ddGs) * u

        individual_estimates = self.get_individual_estimates()
        solv_err = _get_stdev(individual_estimates['solvent'])
        complex_err = _get_stdev(individual_estimates['complex'])

        # return the combined error
        return np.sqrt(solv_err**2 + complex_err**2)

    def get_forward_and_reverse_energy_analysis(self) -> dict[str, list[Optional[dict[str, Union[npt.NDArray, unit.Quantity]]]]]:
        """
        Get the reverse and forward analysis of the free energies.

        Returns
        -------
        forward_reverse : dict[str, list[Optional[dict[str, Union[npt.NDArray, unit.Quantity]]]]]
            A dictionary, keyed `complex` and `solvent` for each leg of the
            thermodynamic cycle which each contain a list of dictionaries
            containing the forward and reverse analysis of each repeat
            of that simulation type.

            The forward and reverse analysis dictionaries contain:
              - `fractions`: npt.NDArray
                  The fractions of data used for the estimates
              - `forward_DDGs`, `reverse_DDGs`: unit.Quantity
                  The forward and reverse estimates for each fraction of data
              - `forward_dDDGs`, `reverse_dDDGs`: unit.Quantity
                  The forward and reverse estimate uncertainty for each
                  fraction of data.

            If one of the cycle leg list entries is ``None``, this indicates
            that the analysis could not be carried out for that repeat. This
            is most likely caused by MBAR convergence issues when attempting to
            calculate free energies from too few samples.

        Raises
        ------
        UserWarning
          * If any of the forward and reverse dictionaries are ``None`` in a
            given thermodynamic cycle leg.
        """

        forward_reverse: dict[str, list[Optional[dict[str, Union[npt.NDArray, unit.Quantity]]]]] = {}

        for key in ['complex', 'solvent']:
            forward_reverse[key] = [
                pus[0].outputs['forward_and_reverse_energies']
                for pus in self.data[key].values()
            ]

            if None in forward_reverse[key]:
                wmsg = (
                    "One or more ``None`` entries were found in the forward "
                    f"and reverse dictionaries of the repeats of the {key} "
                    "calculations. This is likely caused by an MBAR convergence "
                    "failure caused by too few independent samples when "
                    "calculating the free energies of the 10% timeseries slice."
                )
                warnings.warn(wmsg)

        return forward_reverse

    def get_overlap_matrices(self) -> dict[str, list[dict[str, npt.NDArray]]]:
        """
        Get a the MBAR overlap estimates for all legs of the simulation.

        Returns
        -------
        overlap_stats : dict[str, list[dict[str, npt.NDArray]]]
          A dictionary with keys `complex` and `solvent` for each
          leg of the thermodynamic cycle, which each containing a
          list of dictionaries with the MBAR overlap estimates of
          each repeat of that simulation type.

          The underlying MBAR dictionaries contain the following keys:
            * ``scalar``: One minus the largest nontrivial eigenvalue
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              overlap matrix
            * ``matrix``: Estimated overlap matrix of observing a sample from
              state i in state j
        """
        # Loop through and get the repeats and get the matrices
        overlap_stats: dict[str, list[dict[str, npt.NDArray]]] = {}

        for key in ['complex', 'solvent']:
            overlap_stats[key] = [
                pus[0].outputs['unit_mbar_overlap']
                for pus in self.data[key].values()
            ]

        return overlap_stats

    def get_replica_transition_statistics(self) -> dict[str, list[dict[str, npt.NDArray]]]:
        """
        Get the replica exchange transition statistics for all
        legs of the simulation.

        Note
        ----
        This is currently only available in cases where a replica exchange
        simulation was run.

        Returns
        -------
        repex_stats : dict[str, list[dict[str, npt.NDArray]]]
          A dictionary with keys `complex` and `solvent` for each
          leg of the thermodynamic cycle, which each containing
          a list of dictionaries containing the replica transition
          statistics for each repeat of that simulation type.

          The replica transition statistics dictionaries contain the following:
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              lambda state transition matrix
            * ``matrix``: The transition matrix estimate of a replica switching
              from state i to state j.
        """
        repex_stats: dict[str, list[dict[str, npt.NDArray]]] = {}
        try:
            for key in ['complex', 'solvent']:
                repex_stats[key] = [
                    pus[0].outputs['replica_exchange_statistics']
                    for pus in self.data[key].values()
                ]
        except KeyError:
            errmsg = ("Replica exchange statistics were not found, "
                      "did you run a repex calculation?")
            raise ValueError(errmsg)

        return repex_stats

    def get_replica_states(self) -> dict[str, list[npt.NDArray]]:
        """
        Get the timeseries of replica states for all simulation legs.

        Returns
        -------
        replica_states : dict[str, list[npt.NDArray]]
          Dictionary keyed `complex` and `solvent` for each leg of
          the thermodynamic cycle, with lists of replica states
          timeseries for each repeat of that simulation type.
        """
        replica_states: dict[str, list[npt.NDArray]] = {
            'complex': [], 'solvent': []
        }

        def is_file(filename: str):
            p = pathlib.Path(filename)

            if not p.exists():
                errmsg = f"File could not be found {p}"
                raise ValueError(errmsg)

            return p

        def get_replica_state(nc, chk):
            nc = is_file(nc)
            dir_path = nc.parents[0]
            chk = is_file(dir_path / chk).name

            reporter = multistate.MultiStateReporter(
                storage=nc, checkpoint_storage=chk, open_mode='r'
            )

            retval = np.asarray(reporter.read_replica_thermodynamic_states())
            reporter.close()

            return retval

        for key in ['complex', 'solvent']:
            for pus in self.data[key].values():
                states = get_replica_state(
                    pus[0].outputs['nc'],
                    pus[0].outputs['last_checkpoint'],
                )
                replica_states[key].append(states)

        return replica_states

    def equilibration_iterations(self) -> dict[str, list[float]]:
        """
        Get the number of equilibration iterations for each simulation.

        Returns
        -------
        equilibration_lengths : dict[str, list[float]]
          Dictionary keyed `complex` and `solvent` for each leg
          of the thermodynamic cycle, with lists containing the
          number of equilibration iterations for each repeat
          of that simulation type.
        """
        equilibration_lengths: dict[str, list[float]] = {}

        for key in ['complex', 'solvent']:
            equilibration_lengths[key] = [
                pus[0].outputs['equilibration_iterations']
                for pus in self.data[key].values()
            ]

        return equilibration_lengths

    def production_iterations(self) -> dict[str, list[float]]:
        """
        Get the number of production iterations for each simulation.
        Returns the number of uncorrelated production samples for each
        repeat of the calculation.

        Returns
        -------
        production_lengths : dict[str, list[float]]
          Dictionary keyed `complex` and `solvent` for each leg of the
          thermodynamic cycle, with lists with the number
          of production iterations for each repeat of that simulation
          type.
        """
        production_lengths: dict[str, list[float]] = {}

        for key in ['complex', 'solvent']:
            production_lengths[key] = [
                pus[0].outputs['production_iterations']
                for pus in self.data[key].values()
            ]

        return production_lengths

    def restraint_geometries(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Get a list of the restraint geometries for the
        complex simulations. These define the atoms that have
        been restrained in the system.

        Returns
        -------
        geometry_A : list[dict[str, Any]]
          A list of dictionaries containing the details of the atoms
          in the system that are involved in the restraint of ligand A.
        geometry_B : list[dict[str, Any]]
          A list of dictionaries containing the details of the atoms
          in the system that are involved in the restraint of ligand B.
        """
        geometry_A = [
            pus[0].outputs["restraint_geometry_A"]
            for pus in self.data['complex_setup'].values()
        ]
        geometry_B = [
            pus[0].outputs["restraint_geometry_B"]
            for pus in self.data['complex_setup'].values()
        ]

        return geometry_A, geometry_B


class SepTopProtocol(gufe.Protocol):
    """
    SepTop RBFE calculations using OpenMM and OpenMMTools.

    See Also
    --------
    :mod:`openfe.protocols`
    :class:`openfe.protocols.openmm_septop.SepTopSettings`
    :class:`openfe.protocols.openmm_septop.SepTopProtocolResult`
    :class:`openfe.protocols.openmm_septop.SepTopComplexUnit`
    :class:`openfe.protocols.openmm_septop.SepTopSolventUnit`
    """
    result_cls = SepTopProtocolResult
    _settings_cls = SepTopSettings
    _settings: SepTopSettings

    @classmethod
    def _default_settings(cls):
        """A dictionary of initial settings for this creating this Protocol

        These settings are intended as a suitable starting point for creating
        an instance of this protocol.  It is recommended, however that care is
        taken to inspect and customize these before performing a Protocol.

        Returns
        -------
        Settings
          a set of default settings
        """
        return SepTopSettings(
            protocol_repeats=1,
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(
                nonbonded_cutoff=0.9 * unit.nanometer
            ),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            alchemical_settings=AlchemicalSettings(),
            solvent_lambda_settings=LambdaSettings(
                lambda_elec_A=9 * [0.0] + [
                    0.0,
                    0.125,
                    0.25,
                    0.375,
                    0.5,
                    0.625,
                    0.75,
                    0.875,
                    1.0,
                ] + 9 * [1.0],
                lambda_elec_B=9 * [1.0] + [
                    1.0,
                    0.875,
                    0.75,
                    0.625,
                    0.5,
                    0.375,
                    0.25,
                    0.125,
                    0.0,
                ] + 9 * [0.0],
                lambda_vdw_A=[0.0] * 17 + [
                    0.00,
                    0.15,
                    0.23,
                    0.3,
                    0.4,
                    0.52,
                    0.64,
                    0.76,
                    0.88,
                    1.00
                ],
                lambda_vdw_B=[
                                 1.00,
                                 0.85,
                                 0.77,
                                 0.7,
                                 0.6,
                                 0.48,
                                 0.36,
                                 0.24,
                                 0.12,
                                 0.00
                             ] + [0.0] * 17,
                lambda_restraints_A=27 * [0],
                lambda_restraints_B=27 * [0],
            ),
            complex_lambda_settings=LambdaSettings(),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvent_solvation_settings=OpenMMSolvationSettings(
                box_shape='dodecahedron',
                solvent_padding=1.3 * unit.nanometer
            ),
            complex_solvation_settings=OpenMMSolvationSettings(
                box_shape='dodecahedron',
                solvent_padding=1 * unit.nanometer
            ),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            solvent_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * unit.nanosecond,
                equilibration_length=0.1 * unit.nanosecond,
                production_length=2.0 * unit.nanosecond,
            ),
            solvent_equil_output_settings=SepTopEquilOutputSettings(
                equil_nvt_structure=None,
                equil_npt_structure='equil_npt',
                production_trajectory_filename='equil_npt',
                log_output='equil_simulation',
            ),
            solvent_simulation_settings=MultiStateSimulationSettings(
                time_per_iteration=2.5 * unit.picoseconds,
                n_replicas=27,
                minimization_steps=5000,
                equilibration_length=1.0 * unit.nanosecond,
                production_length=10.0 * unit.nanosecond,
            ),
            solvent_output_settings=MultiStateOutputSettings(
                output_structure="alchemical_system.pdb",
                output_filename='solvent.nc',
                checkpoint_storage_filename='solvent_checkpoint.nc',
            ),
            complex_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * unit.nanosecond,
                equilibration_length=0.1 * unit.nanosecond,
                production_length=2.0 * unit.nanosecond,
            ),
            complex_equil_output_settings=SepTopEquilOutputSettings(
                equil_nvt_structure=None,
                equil_npt_structure='equil_npt',
                production_trajectory_filename='equil_npt',
                log_output='equil_simulation',
            ),
            complex_simulation_settings=MultiStateSimulationSettings(
                time_per_iteration=2.5 * unit.picoseconds,
                n_replicas=19,
                equilibration_length=1.0 * unit.nanosecond,
                production_length=10.0 * unit.nanosecond,
            ),
            complex_output_settings=MultiStateOutputSettings(
                output_structure="alchemical_system.pdb",
                output_filename='complex.nc',
                checkpoint_storage_filename='complex_checkpoint.nc'
            ),
            solvent_restraint_settings=DistanceRestraintSettings(
                spring_constant=1000 * unit.kilojoule_per_mole / unit.nanometer**2,
            ),
            complex_restraint_settings=BoreschRestraintSettings(
                K_r=8368.0 * unit.kilojoule_per_mole / unit.nanometer ** 2,
                K_thetaA=1000 * unit.kilojoule_per_mole / unit.radian ** 2
            ),
        )

    @staticmethod
    def _validate_complex_endstates(
        stateA: ChemicalSystem, stateB: ChemicalSystem,
    ) -> None:
        """
        A complex transformation is defined (in terms of gufe components)
        as starting from one or more ligands and a protein in solvent and
        ending up in a state with one less ligand.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A
        stateB : ChemicalSystem
          The chemical system of end state B

        Raises
        ------
        ValueError
          If there is no SolventComponent and no ProteinComponent
          in either stateA or stateB.
        """
        # check that there is a protein component
        if not any(
            isinstance(comp, ProteinComponent) for comp in stateA.values()
        ):
            errmsg = "No ProteinComponent found in stateA"
            raise ValueError(errmsg)

        if not any(
            isinstance(comp, ProteinComponent) for comp in stateB.values()
        ):
            errmsg = "No ProteinComponent found in stateB"
            raise ValueError(errmsg)

        # check that there is a solvent component
        if not any(
            isinstance(comp, SolventComponent) for comp in stateA.values()
        ):
            errmsg = "No SolventComponent found in stateA"
            raise ValueError(errmsg)

        if not any(
            isinstance(comp, SolventComponent) for comp in stateB.values()
        ):
            errmsg = "No SolventComponent found in stateB"
            raise ValueError(errmsg)

    @staticmethod
    def _validate_alchemical_components(
        alchemical_components: dict[str, list[Component]]
    ) -> None:
        """
        Checks that the ChemicalSystem alchemical components are correct.

        Parameters
        ----------
        alchemical_components : Dict[str, list[Component]]
          Dictionary containing the alchemical components for
          stateA and stateB.

        Raises
        ------
        ValueError
          * If there are no or more than one alchemical components in state A.
          * If there are no or more than one alchemical components in state B.
          * If there are any alchemical components that are not
            SmallMoleculeComponents
        * If a change in net charge between the alchemical components is detected.

        Notes
        -----
        * Currently doesn't support alchemical components which are not
          SmallMoleculeComponents.
        * Currently doesn't support more than one alchemical component
          being desolvated.
        """

        # Crash out if there are less or more than one alchemical components
        # in state A and B
        if len(alchemical_components['stateA']) != 1:
            errmsg = ("Exactly one alchemical components must be present in stateA. "
                      f"Found {len(alchemical_components['stateA'])} "
                      "alchemical components in stateA")
            raise ValueError(errmsg)

        if len(alchemical_components['stateB']) != 1:
            errmsg = ("Exactly one alchemical components must be present in stateB. "
                      f"Found {len(alchemical_components['stateB'])} "
                      "alchemical components in stateB")
            raise ValueError(errmsg)

        # Crash out if any of the alchemical components are not
        # SmallMoleculeComponent
        alchem_components_states = [alchemical_components['stateA'], alchemical_components['stateB']]
        for state in alchem_components_states:
            for comp in state:
                if not isinstance(comp, SmallMoleculeComponent):
                    errmsg = ("Non SmallMoleculeComponent alchemical species "
                              "are not currently supported")
                    raise ValueError(errmsg)

        # Raise an error if there is a change in netcharge
        _check_alchemical_charge_difference(
            alchemical_components['stateA'][0],
            alchemical_components['stateB'][0])

    @staticmethod
    def _validate_lambda_schedule(
            lambda_settings: LambdaSettings,
            simulation_settings: MultiStateSimulationSettings,
    ) -> None:
        """
        Checks that the lambda schedule is set up correctly.

        Parameters
        ----------
        lambda_settings : LambdaSettings
          the lambda schedule Settings
        simulation_settings : MultiStateSimulationSettings
          the settings for either the complex or solvent phase

        Raises
        ------
        ValueError
          If the number of lambda windows differs for electrostatics and sterics.
          If the number of replicas does not match the number of lambda windows.
        Warnings
          If there are non-zero values for restraints (lambda_restraints).
        """

        lambda_elec_A = lambda_settings.lambda_elec_A
        lambda_elec_B = lambda_settings.lambda_elec_B
        lambda_vdw_A = lambda_settings.lambda_vdw_A
        lambda_vdw_B = lambda_settings.lambda_vdw_B
        lambda_restraints_A = lambda_settings.lambda_restraints_A
        lambda_restraints_B = lambda_settings.lambda_restraints_B
        n_replicas = simulation_settings.n_replicas

        # Ensure that all lambda components have equal amount of windows
        lambda_components = [lambda_vdw_A, lambda_vdw_B,
                             lambda_elec_A, lambda_elec_B,
                             lambda_restraints_A, lambda_restraints_B]
        it = iter(lambda_components)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            errmsg = (
                "Components elec, vdw, and restraints must have equal amount"
                f" of lambda windows. Got {len(lambda_elec_A)} and "
                f"{len(lambda_elec_B)} elec lambda windows, "
                f"{len(lambda_vdw_A)} and {len(lambda_vdw_B)} vdw "
                f"lambda windows, and {len(lambda_restraints_A)} and "
                f"{len(lambda_restraints_B)} restraints lambda windows.")
            raise ValueError(errmsg)

        # Ensure that number of overall lambda windows matches number of lambda
        # windows for individual components
        if n_replicas != len(lambda_vdw_B):
            errmsg = (f"Number of replicas {n_replicas} does not equal the"
                      f" number of lambda windows {len(lambda_vdw_B)}")
            raise ValueError(errmsg)

        # Check if there are lambda windows with naked charges
        for inx, lam in enumerate(lambda_elec_A):
            if lam < 1 and lambda_vdw_A[inx] == 1:
                errmsg = (
                    "There are states along this lambda schedule "
                    "where there are atoms with charges but no LJ "
                    f"interactions: Ligand A: lambda {inx}: "
                    f"elec {lam} vdW {lambda_vdw_A[inx]}")
                raise ValueError(errmsg)
            if lambda_elec_B[inx] < 1 and lambda_vdw_B[inx] == 1:
                errmsg = (
                    "There are states along this lambda schedule "
                    "where there are atoms with charges but no LJ interactions"
                    f": Ligand B: lambda {inx}: elec {lambda_elec_B[inx]}"
                    f" vdW {lambda_vdw_B[inx]}")
                raise ValueError(errmsg)

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # TODO: extensions
        if extends:  # pragma: no-cover
            raise NotImplementedError("Can't extend simulations yet")

        # Validate components and get alchemical components
        self._validate_complex_endstates(stateA, stateB)
        alchem_comps = system_validation.get_alchemical_components(
            stateA, stateB,
        )
        self._validate_alchemical_components(alchem_comps)

        # Validate the lambda schedule
        self._validate_lambda_schedule(self.settings.solvent_lambda_settings,
                                       self.settings.solvent_simulation_settings)
        self._validate_lambda_schedule(self.settings.complex_lambda_settings,
                                       self.settings.complex_simulation_settings)

        # Check nonbonded and solvent compatibility
        nonbonded_method = self.settings.forcefield_settings.nonbonded_method
        # Use the more complete system validation solvent checks
        system_validation.validate_solvent(stateA, nonbonded_method)

        # Validate solvation settings
        settings_validation.validate_openmm_solvation_settings(
            self.settings.solvent_solvation_settings
        )

        # Make sure that we have the full system for restraint trajectory analysis
        if self.settings.complex_equil_output_settings.output_indices != "all":
            errmsg = (
                "Complex simulations need to output the full system "
                "during equilibration simulations."
            )
            raise ValueError(errmsg)

        # Validate protein component
        system_validation.validate_protein(stateA)

        # Get the name of the alchemical species
        alchname_A = alchem_comps['stateA'][0].name
        alchname_B = alchem_comps['stateB'][0].name

        # Create list units for complex and solvent transforms

        solvent_setup = [
            SepTopSolventSetupUnit(
                protocol=self,
                stateA=stateA,
                stateB=stateB,
                alchemical_components=alchem_comps,
                generation=0, repeat_id=int(uuid.uuid4()),
                name=(f"SepTop RBFE Setup, transformation {alchname_A} to "
                      f"{alchname_B}, solvent leg: repeat {i} generation 0"),
            )
            for i in range(self.settings.protocol_repeats)
        ]

        solvent_run = [
            SepTopSolventRunUnit(
                protocol=self,
                stateA=stateA,
                stateB=stateB,
                alchemical_components=alchem_comps,
                setup=solvent_setup[i],
                generation=0, repeat_id=int(uuid.uuid4()),
                name=(f"SepTop RBFE Run, transformation {alchname_A} to "
                      f"{alchname_B}, solvent leg: repeat {i} generation 0"),
            )
            for i in range(self.settings.protocol_repeats)
        ]

        complex_setup = [
            SepTopComplexSetupUnit(
                protocol=self,
                stateA=stateA,
                stateB=stateB,
                alchemical_components=alchem_comps,
                generation=0, repeat_id=int(uuid.uuid4()),
                name=(f"SepTop RBFE Setup, transformation {alchname_A} to "
                      f"{alchname_B}, complex leg: repeat {i} generation 0"),
            )
            for i in range(self.settings.protocol_repeats)
        ]

        complex_run = [
            SepTopComplexRunUnit(
                protocol=self,
                stateA=stateA,
                stateB=stateB,
                alchemical_components=alchem_comps,
                setup=complex_setup[i],
                generation=0, repeat_id=int(uuid.uuid4()),
                name=(f"SepTop RBFE Run, transformation {alchname_A} to "
                      f"{alchname_B}, complex leg: repeat {i} generation 0"),
            )
            for i in range(self.settings.protocol_repeats)
        ]

        return solvent_setup + solvent_run + complex_setup + complex_run

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> dict[str, dict[str, Any]]:
        # result units will have a repeat_id and generation
        # first group according to repeat_id
        unsorted_solvent_repeats_setup = defaultdict(list)
        unsorted_solvent_repeats_run = defaultdict(list)
        unsorted_complex_repeats_setup = defaultdict(list)
        unsorted_complex_repeats_run = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue
                if pu.outputs['simtype'] == 'solvent':
                    if 'Run' in pu.name:
                        unsorted_solvent_repeats_run[
                            pu.outputs['repeat_id']].append(pu)
                    elif 'Setup' in pu.name:
                        unsorted_solvent_repeats_setup[
                            pu.outputs['repeat_id']].append(pu)
                else:
                    if 'Run' in pu.name:
                        unsorted_complex_repeats_run[
                            pu.outputs['repeat_id']].append(pu)
                    elif 'Setup' in pu.name:
                        unsorted_complex_repeats_setup[
                            pu.outputs['repeat_id']].append(pu)

        repeats: dict[str, dict[str, list[gufe.ProtocolUnitResult]]] = {
            'solvent_setup': {}, 'solvent': {},
            'complex_setup': {}, 'complex': {},
        }
        for k, v in unsorted_solvent_repeats_setup.items():
            repeats['solvent_setup'][str(k)] = sorted(v, key=lambda x: x.outputs['generation'])
        for k, v in unsorted_solvent_repeats_run.items():
            repeats['solvent'][str(k)] = sorted(v, key=lambda x: x.outputs['generation'])

        for k, v in unsorted_complex_repeats_setup.items():
            repeats['complex_setup'][str(k)] = sorted(v, key=lambda x: x.outputs['generation'])
        for k, v in unsorted_complex_repeats_run.items():
            repeats['complex'][str(k)] = sorted(v, key=lambda x: x.outputs['generation'])
        return repeats


class SepTopComplexSetupUnit(BaseSepTopSetupUnit):
    """
    Protocol Unit for the complex phase of a SepTop free energy calculation
    """
    def _get_components(self):
        """
        Get the relevant components for a complex transformation.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']

        solv_comp, prot_comp, small_mols = system_validation.get_components(stateA)
        small_mols = {m: m.to_openff() for m in small_mols}
        # Also get alchemical smc from state B
        small_mols_B = {m: m.to_openff()
                        for m in alchem_comps['stateB']}
        small_mols = small_mols | small_mols_B

        return alchem_comps, solv_comp, prot_comp, small_mols

    def _handle_settings(self) -> dict[str, SettingsBaseModel]:
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
            * equil_output_settings : SepTopEquilOutputSettings
            * simulation_settings : SimulationSettings
            * output_settings: MultiStateOutputSettings
            * restraint_settings: ComplexRestraintsSettings
        """
        prot_settings = self._inputs['protocol'].settings

        settings = {
            'forcefield_settings': prot_settings.forcefield_settings,
            'thermo_settings': prot_settings.thermo_settings,
            'charge_settings': prot_settings.partial_charge_settings,
            'solvation_settings': prot_settings.complex_solvation_settings,
            'alchemical_settings': prot_settings.alchemical_settings,
            'lambda_settings': prot_settings.complex_lambda_settings,
            'engine_settings': prot_settings.engine_settings,
            'integrator_settings': prot_settings.integrator_settings,
            'equil_simulation_settings':
                prot_settings.complex_equil_simulation_settings,
            'equil_output_settings':
                prot_settings.complex_equil_output_settings,
            'simulation_settings': prot_settings.complex_simulation_settings,
            'output_settings': prot_settings.complex_output_settings,
            'restraint_settings': prot_settings.complex_restraint_settings}

        settings_validation.validate_timestep(
            settings['forcefield_settings'].hydrogen_mass,
            settings['integrator_settings'].timestep
        )

        return settings

    @staticmethod
    def _update_positions(
            omm_topology_A, omm_topology_B, positions_A, positions_B
    ) -> simtk.unit.Quantity:
        """
        Aligns the protein from complex B onto the protein from complex A and
        updates the positions of complex B.

        Parameters
        ----------
        omm_topology_A: openmm.app.Topology
          OpenMM topology from complex A
        omm_topology_B: openmm.app.Topology
          OpenMM topology from complex B
        positions_A: simtk.unit.Quantity
          Positions of the system in state A
        positions_B: simtk.unit.Quantity
          Positions of the system in state B

        Returns
        -------
        updated_positions_B: simtk.unit.Quantity
          Updated positions of the complex B
        """
        mdtraj_complex_A = _get_mdtraj_from_openmm(omm_topology_A, positions_A)
        mdtraj_complex_B = _get_mdtraj_from_openmm(omm_topology_B, positions_B)
        mdtraj_complex_B.superpose(mdtraj_complex_A,
                                   atom_indices=mdtraj_complex_A.topology.select(
                                       'backbone'))
        # Extract updated system positions.
        updated_positions_B = mdtraj_complex_B.openmm_positions(-1)

        return updated_positions_B


    @staticmethod
    def _get_mda_universe(
            topology: openmm.app.Topology,
            positions: openmm.unit.Quantity,
            trajectory: Optional[pathlib.Path],
            settings,
    ) -> mda.Universe:
        """
        Helper method to get a Universe from an openmm Topology,
        and either an input trajectory or a set of positions.

        Parameters
        ----------
        topology : openmm.app.Topology
          An OpenMM Topology that defines the System.
        positions: openmm.unit.Quantity
          The System's current positions.
          Used if a trajectory file is None or is not a file.
        trajectory: pathlib.Path
          A Path to a trajectory file to read positions from.
        settings: dict
          The settings dictionary

        Returns
        -------
        mda.Universe
          An MDAnalysis Universe of the System.
        """
        from MDAnalysis.coordinates.memory import MemoryReader

        # If the trajectory file doesn't exist, then we use positions
        write_int = settings['equil_output_settings'].trajectory_write_interval
        prod_length = settings['equil_simulation_settings'].production_length
        if trajectory is not None and trajectory.is_file() and write_int <= prod_length:
            return mda.Universe(
                topology,
                trajectory,
                topology_format="OPENMMTOPOLOGY",
            )
        else:
            # Positions is an openmm Quantity in nm we need
            # to convert to angstroms
            return mda.Universe(
                topology,
                np.array(positions._value) * 10,
                topology_format="OPENMMTOPOLOGY",
                trajectory_format=MemoryReader,
            )

    @staticmethod
    def _get_boresch_restraint(
            universe: mda.Universe,
            guest_rdmol: Chem.Mol,
            guest_atom_ids: list[int],
            host_atom_ids: list[int],
            temperature: unit.Quantity,
            settings: BoreschRestraintSettings,
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
        temperature : unit.Quantity
          The temperature of the simulation where the restraint will be added.
        settings : BoreschRestraintSettings
          Settings on how the Boresch-like restraint should be defined.

        Returns
        -------
        geom : BoreschRestraintGeometry
          A class defining the Boresch-like restraint.
        restraint : BoreschRestraint
          A factory class for generating Boresch restraints in OpenMM.
        """
        frc_const = min(settings.K_thetaA, settings.K_thetaB)

        geom = geometry.boresch.find_boresch_restraint(
            universe=universe,
            guest_rdmol=guest_rdmol,
            guest_idxs=guest_atom_ids,
            host_idxs=host_atom_ids,
            host_selection=settings.host_selection,
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
            system: openmm.System,
            topology_A,
            topology_B,
            positions_A,
            positions_B,
            rdmol_A,
            rdmol_B,
            ligand_A_inxs: list[int],
            ligand_B_inxs: list[int],
            ligand_B_inxs_B: list[int],
            protein_inxs: list[int],  # type: ignore[override]
            settings: dict[str, SettingsBaseModel],
    ) -> openmm.System:
        """
        Adds Boresch restraints to the system.

        Parameters
        ----------
        system: openmm.System
          The OpenMM system where the restraints will be applied to.
        topology: openmm.app.Topology
          The OpenMM topology of the system
        ligand_1: OFFMolecule.Topology
          The topology of the OpenFF Molecule of ligand A
        ligand_2: OFFMolecule.Topology
          The topology of the OpenFF Molecule of ligand B
        settings: dict[str, SettingsBaseModel]
          The settings dict
        ligand_1_ref_idxs: tuple[int, int, int]
          indices from the ligand A topology
        ligand_2_ref_idxs: tuple[int, int, int]
          indices from the ligand B topology
        ligand_1_idxs: tuple[int, int, int]
          indices from the ligand A in the full topology
        ligand_1_idxs: tuple[int, int, int]
          indices from the ligand B in the full topology

        Returns
        -------
        system: openmm.System
          The OpenMM system with the added restraints forces
        """
        from MDAnalysis.coordinates.memory import MemoryReader

        # Get the MDA Universe for the restraints selection
        # We try to pass the equilibration production file path through
        # In some cases (debugging / dry runs) this won't be available
        # so we'll default to using input positions.
        out_traj = (self.shared_basepath
                    / settings[
                        'equil_output_settings'].production_trajectory_filename)
        u_A = self._get_mda_universe(
            topology_A,
            positions_A,
            pathlib.Path(f'{out_traj}_stateA.xtc'),
            settings,
        )
        u_B = self._get_mda_universe(
            topology_B,
            positions_B,
            pathlib.Path(f'{out_traj}_stateB.xtc'),
            settings,
        )
        Chem.SanitizeMol(rdmol_A)
        Chem.SanitizeMol(rdmol_B)

        rest_geom_A, restraint_A = self._get_boresch_restraint(
            u_A,
            rdmol_A,
            ligand_A_inxs,
            protein_inxs,
            settings["thermo_settings"].temperature,
            settings["restraint_settings"],
        )

        rest_geom_B, restraint_B = self._get_boresch_restraint(
            u_B,
            rdmol_B,
            ligand_B_inxs_B,
            protein_inxs,
            settings["thermo_settings"].temperature,
            settings["restraint_settings"],
        )
        # We have to update the indices for ligand B to match the AB complex
        new_boresch_B_indices = [ligand_B_inxs_B.index(i) for i in
                                 rest_geom_B.guest_atoms]
        rest_geom_B.guest_atoms = [ligand_B_inxs[i] for i in
                                   new_boresch_B_indices]

        if self.verbose:
            self.logger.info(f"restraint geometry is: ligand A: {rest_geom_A}"
                             f"and ligand B: {rest_geom_B}.")

        # We need a temporary thermodynamic state to add the restraint
        # & get the correction
        thermodynamic_state = ThermodynamicState(
            system,
            temperature=to_openmm(settings['thermo_settings'].temperature),
            pressure=to_openmm(settings['thermo_settings'].pressure),
        )

        # Add the force to the thermodynamic state
        restraint_A.add_force(
            thermodynamic_state,
            rest_geom_A,
            controlling_parameter_name='lambda_restraints_A'
        )
        restraint_B.add_force(
            thermodynamic_state,
            rest_geom_B,
            controlling_parameter_name='lambda_restraints_B'
        )
        # Get the standard state correction as a unit.Quantity
        correction_A = restraint_A.get_standard_state_correction(
            thermodynamic_state,
            rest_geom_A,
        )
        correction_B = restraint_B.get_standard_state_correction(
            thermodynamic_state,
            rest_geom_B,
        )
        # Multiply the correction for ligand B by -1 as for this ligands,
        # Boresch restraint has to be turned on in the analytical corr.
        correction_B = correction_B * -1

        return correction_A, correction_B, thermodynamic_state.system, rest_geom_A, rest_geom_B


    def run(self, dry=False, verbose=True,
            scratch_basepath=None, shared_basepath=None) -> dict[str, Any]:
        """
        Run the SepTop free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary alchemical
          system components (topology, system, sampler, etc...) but without
          running the simulation, default False
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging, default True
        scratch_basepath : pathlib.Path
          Path to the scratch (temporary) directory space.
        shared_basepath : pathlib.Path
          Path to the shared (persistent) directory space.

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.
        """
        # 0. General preparation tasks
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # 1. Get components
        self.logger.info("Creating and setting up the OpenMM systems")
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()
        smc_comps_A, smc_comps_B, smc_comps_AB, smc_off_B = self.get_smc_comps(
            alchem_comps, smc_comps)

        # 3. Get settings
        settings = self._handle_settings()

        # 4. Assign partial charges
        self._assign_partial_charges(settings['charge_settings'], smc_comps_AB)

        # 5. Get the OpenMM systems
        omm_system_A, omm_topology_A, positions_A, modeller_A, comp_resids_A = self.get_system(
            solv_comp,
            prot_comp,
            smc_comps_A,
            settings,
        )

        omm_system_B, omm_topology_B, positions_B, modeller_B, comp_resids_B = self.get_system(
            solv_comp,
            prot_comp,
            smc_comps_B,
            settings,
        )

        omm_system_AB, omm_topology_AB, positions_AB, modeller_AB = self.get_system_AB(
            solv_comp,
            modeller_A,
            smc_comps_AB,
            smc_off_B,
            settings,
            self.shared_basepath
        )

        # Get the comp_resids of the AB system
        resids_A = list(itertools.chain(*comp_resids_A.values()))
        resids_AB = [r.index for r in modeller_AB.topology.residues()]
        diff_resids = list(set(resids_AB) - set(resids_A))
        comp_resids_AB = comp_resids_A | {
            alchem_comps["stateB"][0]: np.array(diff_resids)}

        # 6. Pre-equilbrate System (for restraint selection)
        self.logger.info("Pre-equilibrating the systems")
        equ_positions_A, box_A = self._pre_equilibrate(
            omm_system_A, omm_topology_A, positions_A, settings, 'A', dry
        )
        equ_positions_B, box_B = self._pre_equilibrate(
            omm_system_B, omm_topology_B, positions_B, settings, 'B', dry
        )

        # 7. Get all the right atom indices for alignments
        comp_atomids_A = self._get_atom_indices(omm_topology_A, comp_resids_A)
        all_atom_ids_A = list(itertools.chain(*comp_atomids_A.values()))
        comp_atomids_B = self._get_atom_indices(omm_topology_B, comp_resids_B)

        # Get the atom indices of ligand B in system B
        atom_indices_B = comp_atomids_B[alchem_comps['stateB'][0]]

        # 8. Update the positions of system B: Align protein
        updated_positions_B = self._update_positions(
            omm_topology_A, omm_topology_B, equ_positions_A, equ_positions_B,
        )

        # Get atom indices for ligand A and ligand B and the solvent in the
        # system AB
        comp_atomids_AB = self._get_atom_indices(omm_topology_AB, comp_resids_AB)
        atom_indices_AB_B = comp_atomids_AB[alchem_comps['stateB'][0]]
        atom_indices_AB_A = comp_atomids_AB[alchem_comps['stateA'][0]]

        # Update positions from AB system
        positions_AB[all_atom_ids_A[0]:all_atom_ids_A[-1] + 1, :] = equ_positions_A
        positions_AB[atom_indices_AB_B[0]:atom_indices_AB_B[-1] + 1,
                     :] = updated_positions_B[atom_indices_B[0]:atom_indices_B[-1] + 1]

        # 9. Create the alchemical system
        self.logger.info("Creating the alchemical system and applying restraints")

        alchemical_factory, alchemical_system = self._get_alchemical_system(
            omm_system_AB,
            atom_indices_AB_A,
            atom_indices_AB_B,
        )

        # 10. Apply Restraints
        corr_A, corr_B, system, restraint_geom_A, restraint_geom_B = self._add_restraints(
            alchemical_system,
            omm_topology_A,
            omm_topology_B,
            equ_positions_A,
            equ_positions_B,
            alchem_comps["stateA"][0].to_rdkit(),
            alchem_comps["stateB"][0].to_rdkit(),
            atom_indices_AB_A,
            atom_indices_AB_B,
            atom_indices_B,
            comp_atomids_AB[prot_comp],
            settings,
        )

        equ_positions_AB, box_AB = self._pre_equilibrate(
            system, omm_topology_AB, positions_AB, settings, 'AB', dry
        )

        topology_file = self.shared_basepath / 'topology.pdb'
        simtk.openmm.app.pdbfile.PDBFile.writeFile(omm_topology_AB,
                                                   equ_positions_AB,
                                                   open(topology_file,
                                                        'w'))

        # ToDo: also apply REST

        system_outfile = self.shared_basepath / "system.xml.bz2"

        # Serialize system, state and integrator
        serialize(system, system_outfile)

        return {
            "system": system_outfile,
            "topology": topology_file,
            "standard_state_correction_A": corr_A.to('kilocalorie_per_mole'),
            "standard_state_correction_B": corr_B.to('kilocalorie_per_mole'),
            "restraint_geometry_A": restraint_geom_A.dict(),
            "restraint_geometry_B": restraint_geom_B.dict(),
        }


    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch,
                           shared_basepath=ctx.shared)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            'simtype': 'complex',
            **outputs
        }


class SepTopSolventSetupUnit(BaseSepTopSetupUnit):
    """
    Protocol Unit for the solvent phase of an relative SepTop free energy
    """

    def _get_components(self):
        """
        Get the relevant components for a solvent transformation.

        Note
        -----
        The solvent portion of the transformation is the transformation of one
        ligand into the other in the solvent. The only thing that
        should be present is the alchemical species in state A and state B
        and the SolventComponent.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']

        small_mols_A = {m: m.to_openff()
                        for m in alchem_comps['stateA']}
        small_mols_B = {m: m.to_openff()
                        for m in alchem_comps['stateB']}
        small_mols = small_mols_A | small_mols_B

        solv_comp, _, _ = system_validation.get_components(stateA)

        return alchem_comps, solv_comp, None, small_mols

    def _handle_settings(self) -> dict[str, SettingsBaseModel]:
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
            * equil_output_settings : SepTopEquilOutputSettings
            * simulation_settings : MultiStateSimulationSettings
            * output_settings: MultiStateOutputSettings
            * restraint_settings: BaseRestraintsSettings
        """
        prot_settings = self._inputs['protocol'].settings

        settings = {
            'forcefield_settings': prot_settings.forcefield_settings,
            'thermo_settings': prot_settings.thermo_settings,
            'charge_settings': prot_settings.partial_charge_settings,
            'solvation_settings': prot_settings.solvent_solvation_settings,
            'alchemical_settings': prot_settings.alchemical_settings,
            'lambda_settings': prot_settings.solvent_lambda_settings,
            'engine_settings': prot_settings.engine_settings,
            'integrator_settings': prot_settings.integrator_settings,
            'equil_simulation_settings':
                prot_settings.solvent_equil_simulation_settings,
            'equil_output_settings':
                prot_settings.solvent_equil_output_settings,
            'simulation_settings': prot_settings.solvent_simulation_settings,
            'output_settings': prot_settings.solvent_output_settings,
            'restraint_settings': prot_settings.solvent_restraint_settings}

        settings_validation.validate_timestep(
            settings['forcefield_settings'].hydrogen_mass,
            settings['integrator_settings'].timestep
        )

        return settings

    @staticmethod
    def _update_positions(
            rdmol_A,
            rdmol_B,
    ) -> simtk.unit.Quantity:

        # Offset ligand B from ligand A in the solvent
        pos_ligandA = rdmol_A.GetConformers()[0].GetPositions()
        pos_ligandB = rdmol_B.GetConformers()[0].GetPositions()

        ligand_1_radius = np.linalg.norm(
            pos_ligandA - pos_ligandA.mean(axis=0), axis=1).max()
        ligand_2_radius = np.linalg.norm(
            pos_ligandB - pos_ligandB.mean(axis=0), axis=1).max()
        ligand_distance = (ligand_1_radius + ligand_2_radius) * 1.5

        ligand_offset = pos_ligandA.mean(0) - pos_ligandB.mean(0)
        ligand_offset[0] += ligand_distance

        # Offset the ligandB.
        pos_ligandB += ligand_offset

        # Extract updated system positions.
        rdmol_B.GetConformers()[0].SetPositions(pos_ligandB)

        return SmallMoleculeComponent(rdmol_B)

    def _add_restraints(
        self,
        system: openmm.System,
        ligand_1,
        ligand_2,
        ligand_1_inxs: list[int],
        ligand_2_inxs: list[int],
        settings: dict[str, SettingsBaseModel],
        positions_AB: np.ndarray,
    ) -> openmm.System:
        """
        Apply the distance restraint between the ligands.

        Parameters
        ----------
        system: openmm.System
          The OpenMM system where the restraints will be applied to.
        positions: simtk.unit.Quantity
          The positions of the OpenMM system
        topology: openmm.app.Topology
          The OpenMM topology of the system
        ligand_1: OFFMolecule.Topology
          The topology of the OpenFF Molecule of ligand A
        ligand_2: OFFMolecule.Topology
          The topology of the OpenFF Molecule of ligand B
        settings: dict[str, SettingsBaseModel]
          The settings dict
        ligand_1_ref_idxs: tuple[int, int, int]
          indices from the ligand A topology
        ligand_2_ref_idxs: tuple[int, int, int]
          indices from the ligand B topology
        ligand_1_idxs: tuple[int, int, int]
          indices from the ligand A in the full topology
        ligand_1_idxs: tuple[int, int, int]
          indices from the ligand B in the full topology

        Returns
        -------
        system: openmm.System
          The OpenMM system with the added restraints forces
        """

        if isinstance(settings['restraint_settings'],
                      DistanceRestraintSettings):

            rest_geom = geometry.harmonic.get_molecule_centers_restraint(
                molA_rdmol=ligand_1,
                molB_rdmol=ligand_2,
                molA_idxs=ligand_1_inxs,
                molB_idxs=ligand_2_inxs,
            )

        else:
            # TODO turn this into a direction for different restraint types supported?
            raise NotImplementedError(
                "Other restraint types are not yet available"
            )

        if self.verbose:
            self.logger.info(f"restraint geometry is: {rest_geom}")

        distance = np.linalg.norm(
            positions_AB[rest_geom.guest_atoms[0]] - positions_AB[rest_geom.host_atoms[0]])
        print(distance)

        k_distance = to_openmm(settings['restraint_settings'].spring_constant)

        force = openmm.HarmonicBondForce()
        force.addBond(
            rest_geom.guest_atoms[0],
            rest_geom.host_atoms[0],
            distance * openmm.unit.nanometers,
            k_distance,
        )
        force.setName("alignment_restraint")
        force.setForceGroup(6)

        system.addForce(force)

        # Get the standard state correction. This assumes that the contribution
        # of the harmonic bond correction itself cancels out.
        correction = from_openmm(
            openmm.unit.MOLAR_GAS_CONSTANT_R * to_openmm(settings['thermo_settings'].temperature)
        )

        return correction, system

    def run(self, dry=False, verbose=True,
            scratch_basepath=None, shared_basepath=None) -> dict[str, Any]:
        """
        Run the SepTop free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary alchemical
          system components (topology, system, sampler, etc...) but without
          running the simulation, default False
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging, default True
        scratch_basepath : pathlib.Path
          Path to the scratch (temporary) directory space.
        shared_basepath : pathlib.Path
          Path to the shared (persistent) directory space.

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.
        """
        # 0. General preparation tasks
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # 1. Get components
        self.logger.info("Creating and setting up the OpenMM systems")
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()
        smc_comps_A, smc_comps_B, smc_comps_AB, smc_off_B = self.get_smc_comps(
            alchem_comps, smc_comps)

        # 3. Get settings
        settings = self._handle_settings()

        # 4. Assign partial charges
        self._assign_partial_charges(settings['charge_settings'], smc_comps_AB)

        # 5. Update the positions of ligand B:
        #    - solvent: Offset ligand B with respect to ligand A
        smc_B = self._update_positions(
            alchem_comps['stateA'][0].to_rdkit(),
            alchem_comps['stateB'][0].to_rdkit(),
            )
        smc_comps_B = {smc_B: smc_B.to_openff()}

        # 5. Get the OpenMM systems
        omm_system_AB, omm_topology_AB, positions_AB, modeller_AB, comp_resids_AB = self.get_system(
            solv_comp,
            prot_comp,
            smc_comps_A | smc_comps_B,
            settings,
        )

        # Get atom indices for ligand A and ligand B and the solvent in the
        # system AB
        comp_atomids_AB = self._get_atom_indices(omm_topology_AB,
                                                 comp_resids_AB)
        atom_indices_AB_A = comp_atomids_AB[alchem_comps['stateA'][0]]
        atom_indices_AB_B = comp_atomids_AB[smc_B]


        # 9. Create the alchemical system
        self.logger.info(
            "Creating the alchemical system and applying restraints")

        alchemical_factory, alchemical_system = self._get_alchemical_system(
            omm_system_AB,
            atom_indices_AB_A,
            atom_indices_AB_B,
        )

        # 10. Apply Restraints
        rdmol_A = alchem_comps['stateA'][0].to_rdkit()
        rdmol_B = smc_B.to_rdkit()
        Chem.SanitizeMol(rdmol_A)
        Chem.SanitizeMol(rdmol_B)

        corr, system = self._add_restraints(
            alchemical_system,
            rdmol_A,
            rdmol_B,
            atom_indices_AB_A,
            atom_indices_AB_B,
            settings,
            positions_AB,
        )

        topology_file = self.shared_basepath / 'topology.pdb'
        simtk.openmm.app.pdbfile.PDBFile.writeFile(omm_topology_AB,
                                                   positions_AB,
                                                   open(topology_file,
                                                        'w'))

        # ToDo: also apply REST

        system_outfile = self.shared_basepath / "system.xml.bz2"

        # Serialize system, state and integrator
        serialize(system, system_outfile)

        return {
            "system": system_outfile,
            "topology": topology_file,
            "standard_state_correction": corr.to('kilocalorie_per_mole'),
        }

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch,
                           shared_basepath=ctx.shared)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            'simtype': 'solvent',
            **outputs
        }


class SepTopSolventRunUnit(BaseSepTopRunUnit):
    """
    Protocol Unit for the solvent phase of an relative SepTop free energy
    """
    def _get_components(self):
        """
        Get the relevant components for a solvent transformation.

        Note
        -----
        The solvent portion of the transformation is the transformation of one
        ligand into the other in the solvent. The only thing that
        should be present is the alchemical species in state A and state B
        and the SolventComponent.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']

        small_mols_A = {m: m.to_openff()
                        for m in alchem_comps['stateA']}
        small_mols_B = {m: m.to_openff()
                        for m in alchem_comps['stateB']}
        small_mols = small_mols_A | small_mols_B

        solv_comp, _, _ = system_validation.get_components(stateA)

        # 1. We don't need to check that solv_comp is not None, otherwise
        # an error will have been raised when calling `validate_solvent`
        # in the Protocol's `_create`.
        # 2. ProteinComps can't be alchem_comps (for now), so will
        # be returned as None
        return alchem_comps, solv_comp, None, small_mols

    def _handle_settings(self) -> dict[str, SettingsBaseModel]:
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
            * equil_output_settings : SepTopEquilOutputSettings
            * simulation_settings : MultiStateSimulationSettings
            * output_settings: MultiStateOutputSettings
            * restraint_settings: BaseRestraintsSettings
        """
        prot_settings = self._inputs['protocol'].settings

        settings = {
            'forcefield_settings': prot_settings.forcefield_settings,
            'thermo_settings': prot_settings.thermo_settings,
            'charge_settings': prot_settings.partial_charge_settings,
            'solvation_settings': prot_settings.solvent_solvation_settings,
            'alchemical_settings': prot_settings.alchemical_settings,
            'lambda_settings': prot_settings.solvent_lambda_settings,
            'engine_settings': prot_settings.engine_settings,
            'integrator_settings': prot_settings.integrator_settings,
            'equil_simulation_settings':
                prot_settings.solvent_equil_simulation_settings,
            'equil_output_settings':
                prot_settings.solvent_equil_output_settings,
            'simulation_settings': prot_settings.solvent_simulation_settings,
            'output_settings': prot_settings.solvent_output_settings,
            'restraint_settings': prot_settings.solvent_restraint_settings}

        settings_validation.validate_timestep(
            settings['forcefield_settings'].hydrogen_mass,
            settings['integrator_settings'].timestep
        )

        return settings

    def _get_lambda_schedule(
            self, settings: dict[str, SettingsBaseModel]
    ) -> dict[str, npt.NDArray]:

        lambdas = dict()

        lambda_elec_A = settings['lambda_settings'].lambda_elec_A
        lambda_vdw_A = settings['lambda_settings'].lambda_vdw_A
        lambda_elec_B = settings['lambda_settings'].lambda_elec_B
        lambda_vdw_B = settings['lambda_settings'].lambda_vdw_B

        # Reverse lambda schedule since in AbsoluteAlchemicalFactory 1
        # means fully interacting, not stateB
        lambda_elec_A = [1 - x for x in lambda_elec_A]
        lambda_vdw_A = [1 - x for x in lambda_vdw_A]
        lambda_elec_B = [1 - x for x in lambda_elec_B]
        lambda_vdw_B = [1 - x for x in lambda_vdw_B]
        # # Set lambda restraint for the solvent to 1
        # lambda_restraints = len(lambda_elec_A) * [1]

        lambdas['lambda_electrostatics_A'] = lambda_elec_A
        lambdas['lambda_sterics_A'] = lambda_vdw_A
        lambdas['lambda_electrostatics_B'] = lambda_elec_B
        lambdas['lambda_sterics_B'] = lambda_vdw_B
        # lambdas['lambda_restraints'] = lambda_restraints

        return lambdas

    def _execute(
        self, ctx: gufe.Context, *, setup, **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        serialized_system = setup.outputs["system"]
        serialized_topology = setup.outputs["topology"]
        outputs = self.run(
            serialized_system,
            serialized_topology,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            'simtype': 'solvent',
            **outputs
        }


class SepTopComplexRunUnit(BaseSepTopRunUnit):
    """
    Protocol Unit for the solvent phase of an relative SepTop free energy
    """
    def _get_components(self):
        """
        Get the relevant components for a complex transformation.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']

        solv_comp, prot_comp, small_mols = system_validation.get_components(stateA)
        small_mols = {m: m.to_openff() for m in small_mols}
        # Also get alchemical smc from state B
        small_mols_B = {m: m.to_openff()
                        for m in alchem_comps['stateB']}
        small_mols = small_mols | small_mols_B

        return alchem_comps, solv_comp, prot_comp, small_mols

    def _handle_settings(self) -> dict[str, SettingsBaseModel]:
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
            * equil_output_settings : SepTopEquilOutputSettings
            * simulation_settings : SimulationSettings
            * output_settings: MultiStateOutputSettings
            * restraint_settings: ComplexRestraintsSettings
        """
        prot_settings = self._inputs['protocol'].settings

        settings = {
            'forcefield_settings': prot_settings.forcefield_settings,
            'thermo_settings': prot_settings.thermo_settings,
            'charge_settings': prot_settings.partial_charge_settings,
            'solvation_settings': prot_settings.complex_solvation_settings,
            'alchemical_settings': prot_settings.alchemical_settings,
            'lambda_settings': prot_settings.complex_lambda_settings,
            'engine_settings': prot_settings.engine_settings,
            'integrator_settings': prot_settings.integrator_settings,
            'equil_simulation_settings':
                prot_settings.complex_equil_simulation_settings,
            'equil_output_settings':
                prot_settings.complex_equil_output_settings,
            'simulation_settings': prot_settings.complex_simulation_settings,
            'output_settings': prot_settings.complex_output_settings,
            'restraint_settings': prot_settings.complex_restraint_settings}

        settings_validation.validate_timestep(
            settings['forcefield_settings'].hydrogen_mass,
            settings['integrator_settings'].timestep
        )

        return settings

    def _get_lambda_schedule(
            self, settings: dict[str, SettingsBaseModel]
    ) -> dict[str, npt.NDArray]:
        lambdas = dict()

        lambda_elec_A = settings['lambda_settings'].lambda_elec_A
        lambda_vdw_A = settings['lambda_settings'].lambda_vdw_A
        lambda_elec_B = settings['lambda_settings'].lambda_elec_B
        lambda_vdw_B = settings['lambda_settings'].lambda_vdw_B
        lambda_restraints_A = settings[
            'lambda_settings'].lambda_restraints_A
        lambda_restraints_B = settings[
            'lambda_settings'].lambda_restraints_B

        # Reverse lambda schedule since in AbsoluteAlchemicalFactory 1
        # means fully interacting, not stateB
        lambda_elec_A = [1 - x for x in lambda_elec_A]
        lambda_vdw_A = [1 - x for x in lambda_vdw_A]
        lambda_elec_B = [1 - x for x in lambda_elec_B]
        lambda_vdw_B = [1 - x for x in lambda_vdw_B]

        lambdas['lambda_electrostatics_A'] = lambda_elec_A
        lambdas['lambda_sterics_A'] = lambda_vdw_A
        lambdas['lambda_electrostatics_B'] = lambda_elec_B
        lambdas['lambda_sterics_B'] = lambda_vdw_B
        lambdas['lambda_restraints_A'] = lambda_restraints_A
        lambdas['lambda_restraints_B'] = lambda_restraints_B

        return lambdas

    def _execute(
        self, ctx: gufe.Context, *, setup, **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        serialized_system = setup.outputs["system"]
        serialized_topology = setup.outputs["topology"]
        outputs = self.run(
            serialized_system,
            serialized_topology,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            'simtype': 'complex',
            **outputs
        }

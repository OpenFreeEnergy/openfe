"""
An implementation of a general trailblazing algorithm for alchemical OpenMM simulations based on the work of the Chodera lab in the Yank <https://github.com/choderalab/yank> software.
The algorithm is designed to find an optimal lambda schedule for alchemical transformations, which can improve the efficiency and accuracy of free energy calculations.

Basic outline of the trailblazing algorithm:
1. Start at lambda=0 end state, run equilibration and extract N samples.
2. For the next candidate lambda window reweight all samples to the new lambda window and calculate the std of the reduced potential energy differences.
3. If the std is below a threshold and within the chosen tolerance, accept the new lambda window and move to the next candidate window. Go back to step 2.
4. If the std is above the threshold, reject the new lambda window and try a different candidate window. Go back to step 2.
5. Repeat until the entire lambda schedule is constructed, return the final lambda schedule and the equilibrated input samples for each window for the production phase of the simulation.
"""

import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from openff.units.openmm import to_openmm
from openff.units.units import Quantity
from openmm import app, openmm
from openmm import unit as ommunit

from openfe.protocols.openmm_md.plain_md_methods import PlainMDSimulationUnit
from openfe.protocols.openmm_utils import omm_compute, settings_validation

logger = logging.getLogger(__name__)


@dataclass
class _StateSnapshot:
    step: int
    positions_nm: np.ndarray
    box_vectors_nm: np.ndarray


class _InMemorySnapshotReporter:
    def __init__(self, report_interval: int):
        self._report_interval = report_interval
        self.snapshots: list[_StateSnapshot] = []

    def describeNextReport(
        self, simulation: app.Simulation
    ) -> tuple[int, bool, bool, bool, bool, bool]:
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (steps, True, False, False, False, False)

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        self.snapshots.append(
            _StateSnapshot(
                step=simulation.currentStep,
                positions_nm=state.getPositions(asNumpy=True).value_in_unit(ommunit.nanometer),
                box_vectors_nm=state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    ommunit.nanometer
                ),
            )
        )


class TrailblazingMixin:
    """
    Mixin class for trailblazing algorithm for alchemical OpenMM simulations.
    """

    def __init__(self, settings: dict, shared_basepath: pathlib.Path, verbose: bool = False):
        """
        Initialize the TrailblazingMixin class.

        Parameters
        ----------
        inputs : dict
            A dictionary of inputs for the trailblazing algorithm.
        shared_basepath : pathlib.Path
            The base path for storing simulation outputs.
        verbose : bool, optional
            If True, print detailed logs during execution. Default is False.
        """
        self._settings = settings
        self.shared_basepath = shared_basepath
        self.verbose = verbose

    def _get_settings(self):
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
            * equil_output_settings : MDOutputSettings
            * simulation_settings : MultiStateSimulationSettings
            * output_settings: MultiStateOutputSettings
        """
        prot_settings = self._settings  # type: ignore[attr-defined]

        settings = {}
        settings["forcefield_settings"] = prot_settings.solvent_forcefield_settings.unfrozen_copy()
        settings["thermo_settings"] = prot_settings.thermo_settings.unfrozen_copy()
        settings["charge_settings"] = prot_settings.partial_charge_settings.unfrozen_copy()
        settings["solvation_settings"] = prot_settings.solvation_settings.unfrozen_copy()
        settings["alchemical_settings"] = prot_settings.alchemical_settings.unfrozen_copy()
        settings["lambda_settings"] = prot_settings.lambda_settings.unfrozen_copy()
        settings["engine_settings"] = prot_settings.solvent_engine_settings.unfrozen_copy()
        settings["integrator_settings"] = prot_settings.integrator_settings.unfrozen_copy()
        settings["equil_simulation_settings"] = (
            prot_settings.solvent_equil_simulation_settings.unfrozen_copy()
        )
        settings["equil_output_settings"] = (
            prot_settings.solvent_equil_output_settings.unfrozen_copy()
        )
        settings["simulation_settings"] = prot_settings.solvent_simulation_settings.unfrozen_copy()
        settings["output_settings"] = prot_settings.solvent_output_settings.unfrozen_copy()

        return settings

    def _lambda_converter(self, lambda_value: float) -> dict[str, float]:
        """
        Convert a global lambda value to a dictionary of alchemical parameters for the OpenMM simulation.

        Parameters
        ----------
        lambda_value : float
            The lambda value to convert.

        Returns
        -------
        dict[str, float]
            A dictionary of alchemical parameters.
        """
        # make sure its bounded between 0 and 1
        lambda_value = np.clip(lambda_value, 0.0, 1.0)
        # Yank uses the lambda bounds of 1.0 -> 0.0 for the end state, so we need to invert the lambda value
        lambda_value = 1.0 - lambda_value
        return {
            # sterics are only scaled for lambda >= 0.5, electrostatics are only scaled for lambda < 0.5
            # method taken from Yank: <https://github.com/choderalab/yank/blob/c06059045bcf86d610f2e39c6db3944994b9f392/docs/yamlpages/protocols.rst>
            "lambda_electrostatics": 2 * (lambda_value - 0.5) * np.heaviside(lambda_value - 0.5, 0),
            "lambda_sterics": np.heaviside(lambda_value - 0.5, 0.5)
            + 2 * lambda_value * np.heaviside(0.5 - lambda_value, 0.5),
        }

    def _get_reduced_potential(self, simulation: app.Simulation, settings) -> Quantity:
        state = simulation.context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()
        pressure = to_openmm(settings["thermo_settings"].pressure)
        volume = state.getPeriodicBoxVolume()
        return (
            potential_energy / self._kbT
            + (pressure * volume * ommunit.AVOGADRO_CONSTANT_NA) / self._kbT
        )

    def _get_snapshot_energies(
        self, simulation: app.Simulation, snapshots: list[_StateSnapshot], settings
    ) -> list[Quantity]:
        energies = []
        for snapshot in snapshots:
            simulation.context.setPositions(snapshot.positions_nm * ommunit.nanometer)
            a_vec, b_vec, c_vec = snapshot.box_vectors_nm
            simulation.context.setPeriodicBoxVectors(
                openmm.Vec3(*a_vec) * ommunit.nanometer,
                openmm.Vec3(*b_vec) * ommunit.nanometer,
                openmm.Vec3(*c_vec) * ommunit.nanometer,
            )
            energies.append(self._get_reduced_potential(simulation=simulation, settings=settings))

        return energies

    def _sample_state(
        self,
        simulation: app.Simulation,
        lambda_value: float,
        positions: ommunit.Quantity,
        settings,
        equil_steps_nvt: int | None,
        equil_steps_npt: int,
        prod_steps_npt: int,
    ) -> tuple[list[Quantity], list[_StateSnapshot]]:
        """
        Equilibrate the simulation and extract N in-memory state snapshots.

        Notes
        -----
        - The alchemical lambda values should already be set before being passed to this function.
        """
        # make a new folder in the shared basepath for this lambda value
        lambda_folder = self.shared_basepath / f"lambda_{lambda_value:.3f}"
        lambda_folder.mkdir(parents=True, exist_ok=True)
        write_interval = settings_validation.divmod_time_and_check(
            settings["equil_output_settings"].trajectory_write_interval,
            settings["integrator_settings"].timestep,
            "trajectory_write_interval",
            "timestep",
        )
        snapshot_reporter = _InMemorySnapshotReporter(report_interval=write_interval)
        simulation.reporters.append(snapshot_reporter)

        PlainMDSimulationUnit._run_MD(
            simulation=simulation,
            positions=positions,
            simulation_settings=settings["equil_simulation_settings"],
            output_settings=settings["equil_output_settings"],
            temperature=settings["thermo_settings"].temperature,
            barostat_frequency=settings["integrator_settings"].barostat_frequency,
            timestep=settings["integrator_settings"].timestep,
            equil_steps_nvt=equil_steps_nvt,
            equil_steps_npt=equil_steps_npt,
            prod_steps=prod_steps_npt,
            verbose=self.verbose,
            shared_basepath=lambda_folder,
        )
        # clean up the simulation object so it can be reused
        simulation.reporters = []
        prod_start_step = (equil_steps_nvt or 0) + equil_steps_npt
        prod_snapshots = [
            snapshot for snapshot in snapshot_reporter.snapshots if snapshot.step > prod_start_step
        ]
        # reset the current step of the simulation to 0 so it can be reused for the next lambda value
        simulation.currentStep = 0
        if not prod_snapshots:
            errmsg = (
                "No production snapshots captured for trailblazing reweighting. "
                "Ensure trajectory_write_interval is shorter than production_length."
            )
            raise ValueError(errmsg)

        # calculate the reduced potential energies for the production snapshots using the log data
        state_data = pd.read_csv(lambda_folder / settings["equil_output_settings"].log_output)
        potential_energies = (
            state_data["Potential Energy (kJ/mole)"].values * ommunit.kilojoule_per_mole
        )
        volumes = state_data["Box Volume (nm^3)"].values * ommunit.nanometer**3
        pressure = to_openmm(settings["thermo_settings"].pressure)
        simulated_energies = (
            potential_energies / self._kbT
            + (pressure * volumes * ommunit.AVOGADRO_CONSTANT_NA) / self._kbT
        )

        # save the box vectors and the positions to a numpy file
        box_vectors = np.array([snapshot.box_vectors_nm for snapshot in prod_snapshots])
        positions = np.array([snapshot.positions_nm for snapshot in prod_snapshots])
        np.savez_compressed(lambda_folder / "box_vectors", box_vectors)
        np.savez_compressed(lambda_folder / "positions", positions)

        return simulated_energies, prod_snapshots

    def _reweight_samples(
        self,
        simulation: app.Simulation,
        snapshots: list[_StateSnapshot],
        reweight_lambda_value: float,
        settings,
    ) -> list[Quantity]:
        """
        Calculate the reweighted reduced potential energies for a new lambda value based on the simulated samples.
        """
        reweight_alchemical_parameters = self._lambda_converter(reweight_lambda_value)
        for param_name, param_value in reweight_alchemical_parameters.items():
            simulation.context.setParameter(param_name, param_value)
        return self._get_snapshot_energies(
            simulation=simulation,
            snapshots=snapshots,
            settings=settings,
        )

    def _run_trailblazing_method(
        self,
        system: openmm.System,
        topology: app.Topology,
        positions: ommunit.Quantity,
        thermodynamic_distance: float = 1.0,
        distance_tolerance: float = 0.05,
    ) -> list[float]:
        """
        Run the trailblazing algorithm to find an optimal lambda schedule for alchemical transformations.
        """
        settings = self._get_settings()  # type: ignore[attr-defined]
        # save all particles if any structure output is produced and disable trajectory output
        settings["equil_output_settings"].output_indices = "all"
        settings["equil_output_settings"].production_trajectory_filename = None

        # extract the settings we need for the equilibration
        if settings["equil_simulation_settings"].equilibration_length_nvt is not None:
            equil_steps_nvt = settings_validation.get_simsteps(
                sim_length=settings["equil_simulation_settings"].equilibration_length_nvt,
                timestep=settings["integrator_settings"].timestep,
                mc_steps=1,
            )
        else:
            equil_steps_nvt = None

        equil_steps_npt = settings_validation.get_simsteps(
            sim_length=settings["equil_simulation_settings"].equilibration_length,
            timestep=settings["integrator_settings"].timestep,
            mc_steps=1,
        )

        prod_steps_npt = settings_validation.get_simsteps(
            sim_length=settings["equil_simulation_settings"].production_length,
            timestep=settings["integrator_settings"].timestep,
            mc_steps=1,
        )

        # build the simulation from the alchemical system and the initial lambda value
        restrict_cpu = settings["forcefield_settings"].nonbonded_method.lower() == "nocutoff"
        platform = omm_compute.get_openmm_platform(
            platform_name=settings["engine_settings"].compute_platform,
            gpu_device_index=settings["engine_settings"].gpu_device_index,
            restrict_cpu_count=restrict_cpu,
        )
        integrator = openmm.LangevinMiddleIntegrator(
            to_openmm(settings["thermo_settings"].temperature),
            to_openmm(settings["integrator_settings"].langevin_collision_rate),
            to_openmm(settings["integrator_settings"].timestep),
        )
        simulation = app.Simulation(
            topology=topology,
            system=system,
            integrator=integrator,
            platform=platform,
        )
        # calculate and store kbT for the simulation
        self._kbT = ommunit.MOLAR_GAS_CONSTANT_R * to_openmm(
            settings["thermo_settings"].temperature
        )

        optimal_lambda = [0.0]
        state_stds = []

        # start the trailblazing algorithm by iteratively finding the next optimal lambda window
        while optimal_lambda[-1] < 1.0:
            # set the lambda value in the simulation context
            current_lambda = optimal_lambda[-1]
            alchemical_parameters = self._lambda_converter(current_lambda)
            for param_name, param_value in alchemical_parameters.items():
                simulation.context.setParameter(param_name, param_value)

            # equilibrate and sample the simulation, samples are saved in the trajectory reporter
            simulated_energies, simulated_snapshots = self._sample_state(
                simulation=simulation,
                lambda_value=current_lambda,
                positions=positions,
                settings=settings,
                equil_steps_nvt=equil_steps_nvt,
                equil_steps_npt=equil_steps_npt,
                prod_steps_npt=prod_steps_npt,
            )

            # generate the next lambda value candidate
            std_energy = 0.0
            old_std_energy = 0.0
            old_lambda = current_lambda
            print(simulated_energies)

            while abs(std_energy - thermodynamic_distance) > distance_tolerance and not (
                current_lambda == 1.0 and std_energy < thermodynamic_distance
            ):
                if np.isclose(std_energy, 0.0):
                    # This is the first iteration or the two states overlap significantly
                    current_lambda += 0.05
                else:
                    # Assume std_energy is linear to determine the next value to try
                    derivative_std_energy = (std_energy - old_std_energy) / (
                        current_lambda - old_lambda
                    )
                    old_lambda = current_lambda
                    current_lambda += (thermodynamic_distance - std_energy) / derivative_std_energy

                # clip the value if we go over 1.0
                current_lambda = min(current_lambda, 1.0)

                reweighted_energies = self._reweight_samples(
                    simulation=simulation,
                    snapshots=simulated_snapshots,
                    reweight_lambda_value=current_lambda,
                    settings=settings,
                )

                # update the std_energies with the new value
                old_std_energy = std_energy
                energy_diffs = np.array(reweighted_energies) - np.array(simulated_energies)
                std_energy = np.std(energy_diffs, ddof=1)
                logger.info(
                    f"Trailblazing: simulated_lambda={optimal_lambda[-1]}, current_lambda={current_lambda:.3f}, std_energy={std_energy:.3f}, target={thermodynamic_distance:.3f}, tolerance={distance_tolerance:.3f}"
                )
                print(
                    f"Trailblazing: simulated_lambda={optimal_lambda[-1]}, current_lambda={current_lambda:.3f}, std_energy={std_energy:.3f}, target={thermodynamic_distance:.3f}, tolerance={distance_tolerance:.3f}"
                )

            # add the value to the optimal lambda schedule
            optimal_lambda.append(current_lambda)
            state_stds.append(std_energy)
            logger.info(
                f"Trailblazing: accepted lambda={current_lambda:.3f}, std_energy={std_energy:.3f}, target={thermodynamic_distance:.3f}, tolerance={distance_tolerance:.3f}"
            )
            print(
                f"Trailblazing: accepted lambda={current_lambda:.3f}, std_energy={std_energy:.3f}, target={thermodynamic_distance:.3f}, tolerance={distance_tolerance:.3f}"
            )
            # todo
            # if we request a bidirectional optimization at the end then we need to reweight to the previous lambda value
            # if this is not the first value

        # todo
        # if we request a bidirectional optimization this should be done here

        logger.info(f"Trailblazing: completed with optimal lambda schedule: {optimal_lambda}")
        print(f"Trailblazing: completed with optimal lambda schedule: {optimal_lambda}")
        return optimal_lambda
        # # if there is a previous lambda value, calculate the reverse direction simulated reweighted to the previous lambda value
        # if len(optimal_lambda) > 2:
        #     previous_lambda = optimal_lambda[-2]
        #     previous_alchemical_parameters = self._lambda_converter(previous_lambda)
        #     reverse_reweighted_energies = self._reweight_samples(samples, alchemical_parameters, previous_alchemical_parameters)

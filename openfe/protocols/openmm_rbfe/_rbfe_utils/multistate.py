#############################################################################
# HYBRID SYSTEM SAMPLERS
#############################################################################
"""
This is adapted from Perses: https://github.com/choderalab/perses/
See here for the license: https://github.com/choderalab/perses/blob/main/LICENSE
"""

import copy
import warnings
import numpy as np
import openmm
from openmm import unit
from openmmtools.multistate import replicaexchange, sams, multistatesampler
from openmmtools import cache
from openmmtools.states import (CompoundThermodynamicState,
                                SamplerState, ThermodynamicState)
from .lambdaprotocol import RelativeAlchemicalState


class HybridCompatibilityMixin(object):
    """
    Mixin that allows the MultistateSampler to accommodate the situation where
    unsampled endpoints have a different number of degrees of freedom.
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        self._hybrid_factory = hybrid_factory
        super(HybridCompatibilityMixin, self).__init__(*args, **kwargs)

    def setup(self, reporter, platform, lambda_protocol,
              temperature=298.15 * unit.kelvin, n_replicas=None,
              endstates=True):
        """
        Setup MultistateSampler based on the input lambda protocol and number
        of replicas.

        Parameters
        ----------
        reporter : OpenMM reporter
            Simulation reporter to attach to each simulation replica.
        platform : openmm.Platform
            Platform to perform simulation on.
        lambda_protocol : LambdaProtocol
            The lambda protocol to be used for simulation. Default to a default
            class creation of LambdaProtocol.
        temperature : openmm.Quantity
            Simulation temperature, default to 298.15 K
        n_replicas : int
            Number of replicas to simulate. Sets to the number of lambda
            states (as defined by lambda_protocol) if ``None``.
            Default ``None``.
        endstates : bool
            Whether or not to generate unsampled endstates (i.e. dispersion
            correction).

        Attributes
        ----------
        n_states : int
            Number of states / windows which are to be sampled. Obtained from
            lambda_protocol.
        """
        n_states = len(lambda_protocol.lambda_schedule)
        # TODO - remove this and move it to our own MD tasks
        # from perses.dispersed import feptasks

        hybrid_system = self._factory.hybrid_system

        lambda_zero_state = RelativeAlchemicalState.from_system(hybrid_system)

        thermostate = ThermodynamicState(hybrid_system,
                                         temperature=temperature)
        compound_thermostate = CompoundThermodynamicState(
                                   thermostate,
                                   composable_states=[lambda_zero_state])

        # create lists for storing thermostates and sampler states
        thermodynamic_state_list = []
        sampler_state_list = []

        context_cache = cache.ContextCache(platform)

        if n_replicas is None:
            msg = (f"setting number of replicas to number of states: {n_states}")
            warnings.warn(msg)
            n_replicas = n_states
        elif n_replicas > n_states:
            wmsg = (f"More sampler states: {n_replicas} requested than the "
                    f"number of available states: {n_states}. Setting "
                    "the number of replicas to the number of states")
            warnings.warn(wmsg)
            n_replicas = n_states

        lambda_schedule = lambda_protocol.lambda_schedule
        if len(lambda_schedule) != n_states:
            errmsg = ("length of lambda_schedule must match the number of "
                      "states, n_states")
            raise ValueError(errmsg)

        # starting with the hybrid factory positions
        box = hybrid_system.getDefaultPeriodicBoxVectors()
        sampler_state = SamplerState(self._factory.hybrid_positions,
                                     box_vectors=box)

        # Loop over the lambdas and create & store a compound thermostate at
        # that lambda value
        for lambda_val in lambda_schedule:
            compound_thermostate_copy = copy.deepcopy(compound_thermostate)
            compound_thermostate_copy.set_alchemical_parameters(
                                          lambda_val, lambda_protocol)
            thermodynamic_state_list.append(compound_thermostate_copy)

            # now generating a sampler_state for each thermodyanmic state,
            # with relaxed positions
            context, context_integrator = context_cache.get_context(
                                             compound_thermostate_copy)
            # TODO: move to our own MD tasks
            # feptasks.minimize(compound_thermodynamic_state_copy,
            #                   sampler_state)
            sampler_state_list.append(copy.deepcopy(sampler_state))

        # making sure number of sampler states equals n_replicas
        if len(sampler_state_list) != n_replicas:
            # picking roughly evenly spaced sampler states
            # if n_replicas == 1, then it will pick the first in the list
            samples = np.linspace(0, len(sampler_state_list) - 1,
                                  n_replicas)
            idx = np.round(samples).astype(int)
            sampler_state_list = [state for i, state in
                                  enumerate(sampler_state_list) if i in idx]

        assert len(sampler_state_list) == n_replicas

        if endstates:
            # generating unsampled endstates
            unsampled_dispersion_endstates = create_endstates(
                copy.deepcopy(thermodynamic_state_list[0]),
                copy.deepcopy(thermodynamic_state_list[-1]))
            self.create(thermodynamic_states=thermodynamic_state_list,
                        sampler_states=sampler_state_list, storage=reporter,
                        unsampled_thermodynamic_states=unsampled_dispersion_endstates)
        else:
            self.create(thermodynamic_states=thermodynamic_state_list,
                        sampler_states=sampler_state_list, storage=reporter)

    def _minimize_replica(self, replica_id, tolerance, max_iterations):
        """Minimize the specified replica.
        """

        # Retrieve thermodynamic and sampler states.
        thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
        thermodynamic_state = self._thermodynamic_states[thermodynamic_state_id]
        sampler_state = self._sampler_states[replica_id]

        # Use the FIRE minimizer
        integrator = FIREMinimizationIntegrator(tolerance=tolerance)

        # Get context and bound integrator from energy_context_cache
        context, integrator = self.energy_context_cache.get_context(thermodynamic_state, integrator)
        # inform of platform used in current context
        logger.debug(f"{type(integrator).__name__}: Minimize using {context.getPlatform().getName()} platform.")

        # Set initial positions and box vectors.
        sampler_state.apply_to_context(context)

        # Compute the initial energy of the system for logging.
        initial_energy = thermodynamic_state.reduced_potential(context)
        logger.debug('Replica {}/{}: initial energy {:8.3f}kT'.format(
            replica_id + 1, self.n_replicas, initial_energy))

        # Minimize energy.
        try:
            if max_iterations == 0:
                logger.debug('Using FIRE: tolerance {} minimizing to convergence'.format(tolerance))
                while integrator.getGlobalVariableByName('converged') < 1:
                    integrator.step(50)
            else:
                logger.debug('Using FIRE: tolerance {} max_iterations {}'.format(tolerance, max_iterations))
                integrator.step(max_iterations)
        except Exception as e:
            if str(e) == 'Particle coordinate is nan':
                logger.debug('NaN encountered in FIRE minimizer; falling back to L-BFGS after resetting positions')
                sampler_state.apply_to_context(context)
                openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)
            else:
                raise e

        # Get the minimized positions.
        sampler_state.update_from_context(context)

        # Compute the final energy of the system for logging.
        final_energy = thermodynamic_state.reduced_potential(sampler_state)

        # If energy > 0 kT and on a GPU device attempt to use CPU L-BFGS minimizer
        if final_energy > 0 and context.getPlatform().getName() in ['CUDA', 'OpenCL']:
            logger.debug(f'Positive final FIRE minimizer energy {final_energy}; falling back to CPU L-BFGS')
            sampler_state.apply_to_context(context)
            integrator = openmm.VerletIntegrator(1.0)
            cpu_platform = openmm.Platform.getPlatformByName('CPU')
            context = thermodynamic_state.create_context(integrator, cpu_platform)
            sampler_state.apply_to_context(context, ignore_velocities=True)
            openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)

            # Get the minimized positions
            sampler_state.update_from_context(context)

            # Get the final energy
            final_energy = thermodynamic_state.reduced_potential(sampler_state)

        logger.debug('Replica {}/{}: final energy {:8.3f}kT'.format(
            replica_id + 1, self.n_replicas, final_energy))

        # Clean up the integrator
        del context

        # Return minimized positions.
        return sampler_state.positions


class HybridRepexSampler(HybridCompatibilityMixin,
                         replicaexchange.ReplicaExchangeSampler):
    """
    ReplicaExchangeSampler that supports unsampled end states with a different
    number of positions
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        super(HybridRepexSampler, self).__init__(
            *args, hybrid_factory=hybrid_factory, **kwargs)
        self._factory = hybrid_factory


class HybridSAMSSampler(HybridCompatibilityMixin, sams.SAMSSampler):
    """
    SAMSSampler that supports unsampled end states with a different number
    of positions
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        super(HybridSAMSSampler, self).__init__(
                *args, hybrid_factory=hybrid_factory, **kwargs
        )
        self._factory = hybrid_factory


class HybridMultiStateSampler(HybridCompatibilityMixin,
                              multistatesampler.MultiStateSampler):
    """
    MultiStateSampler that supports unsample end states with a different
    number of positions
    """
    def __init__(self, *args, hybrid_factory=None, **kwargs):
        super(HybridMultiStateSampler, self).__init__(
                *args, hybrid_factory=hybrid_factory, **kwargs
        )
        self._factory = hybrid_factory


def create_endstates(first_thermostate, last_thermostate):
    """
    Utility function to generate unsampled endstates
    1. Move all alchemical atom LJ parameters from CustomNonbondedForce to
       NonbondedForce.
    2. Delete the CustomNonbondedForce.
    3. Set PME tolerance to 1e-5.
    4. Enable LJPME to handle long range dispersion corrections in a physically
       reasonable manner.

    Parameters
    ----------
    first_thermostate : openmmtools.states.CompoundThermodynamicState
        The first thermodynamic state for which an unsampled endstate will be
        created.
    last_thermostate : openmmtools.states.CompoundThermodynamicState
        The last thermodynamic state for which an unsampled endstate will be
        created.

    Returns
    -------
    unsampled_endstates : list of openmmtools.states.CompoundThermodynamicState
        The corrected unsampled endstates.
    """
    unsampled_endstates = []
    for master_lambda, endstate in zip([0., 1.],
                                       [first_thermostate, last_thermostate]):
        dispersion_system = endstate.get_system()
        energy_unit = unit.kilocalories_per_mole
        # Find the NonbondedForce (there must be only one)
        forces = {force.__class__.__name__: force for
                  force in dispersion_system.getForces()}
        # Set NonbondedForce to use LJPME
        ljpme = openmm.NonbondedForce.LJPME
        forces['NonbondedForce'].setNonbondedMethod(ljpme)
        # Set tight PME tolerance
        TIGHT_PME_TOLERANCE = 1.0e-5
        forces['NonbondedForce'].setEwaldErrorTolerance(TIGHT_PME_TOLERANCE)
        # Move alchemical LJ sites from CustomNonbondedForce back to
        # NonbondedForce
        for particle_index in range(forces['NonbondedForce'].getNumParticles()):
            charge, sigma, epsilon = forces['NonbondedForce'].getParticleParameters(particle_index)
            sigmaA, epsilonA, sigmaB, epsilonB, unique_old, unique_new = forces['CustomNonbondedForce'].getParticleParameters(particle_index)
            if (epsilon/energy_unit == 0.0) and ((epsilonA > 0.0) or (epsilonB > 0.0)):
                sigma = (1-master_lambda)*sigmaA + master_lambda*sigmaB
                epsilon = (1-master_lambda)*epsilonA + master_lambda*epsilonB
                forces['NonbondedForce'].setParticleParameters(
                                             particle_index, charge,
                                             sigma, epsilon)

        # Delete the CustomNonbondedForce since we have moved all alchemical
        # particles out of it
        for force_index, force in enumerate(list(dispersion_system.getForces())):
            if force.__class__.__name__ == 'CustomNonbondedForce':
                custom_nonbonded_force_index = force_index
                break

        dispersion_system.removeForce(custom_nonbonded_force_index)
        # Set all parameters to master lambda
        for force_index, force in enumerate(list(dispersion_system.getForces())):
            if hasattr(force, 'getNumGlobalParameters'):
                for parameter_index in range(force.getNumGlobalParameters()):
                    if force.getGlobalParameterName(parameter_index)[0:7] == 'lambda_':
                        force.setGlobalParameterDefaultValue(parameter_index,
                                                             master_lambda)

        # Store the unsampled endstate
        unsampled_endstates.append(ThermodynamicState(
            dispersion_system, temperature=endstate.temperature))

    return unsampled_endstates

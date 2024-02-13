# Very slightly adapted from perses https://github.com/choderalab/perses
# License: MIT
# OpenFE note: eventually we aim to move this to openmmtools where possible

import numpy as np
import warnings
import copy
from openmmtools.alchemy import AlchemicalState


class LambdaProtocol(object):
    """Protocols for perturbing each of the component energy terms in alchemical
    free energy simulations.

    TODO
    ----
    * Class needs cleaning up and made more consistent
    """

    default_functions = {'lambda_sterics_core':
                         lambda x: x,
                         'lambda_electrostatics_core':
                         lambda x: x,
                         'lambda_sterics_insert':
                         lambda x: 2.0 * x if x < 0.5 else 1.0,
                         'lambda_sterics_delete':
                         lambda x: 0.0 if x < 0.5 else 2.0 * (x - 0.5),
                         'lambda_electrostatics_insert':
                         lambda x: 0.0 if x < 0.5 else 2.0 * (x - 0.5),
                         'lambda_electrostatics_delete':
                         lambda x: 2.0 * x if x < 0.5 else 1.0,
                         'lambda_bonds':
                         lambda x: x,
                         'lambda_angles':
                         lambda x: x,
                         'lambda_torsions':
                         lambda x: x
                         }

    # lambda components for each component,
    # all run from 0 -> 1 following master lambda
    def __init__(self, functions='default', windows=10, lambda_schedule=None):
        """Instantiates lambda protocol to be used in a free energy
        calculation. Can either be user defined, by passing in a dict, or using
        one of the pregenerated sets by passing in a string 'default', 'namd'
        or 'quarters'

        All protocols must begin and end at 0 and 1 respectively. Any energy
        term not defined in `functions` dict will be set to the function in
        `default_functions`

        Pre-coded options:
        default : ele and LJ terms of the old system are turned off between
        0.0 -> 0.5 ele and LJ terms of the new system are turned on between
        0.5 -> 1.0 core terms treated linearly

        quarters : 0.25 of the protocol is used in turn to individually change
        the (a) off old ele, (b) off old sterics, (c) on new sterics (d) on new
        ele core terms treated linearly

        namd : follows the protocol outlined here:
        https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00362#
        Jiang, Wei, Christophe Chipot, and Benoît Roux. "Computing Relative
        Binding Affinity of Ligands to Receptor: An Effective Hybrid
        Single-Dual-Topology Free-Energy Perturbation Approach in NAMD."
        Journal of chemical information and modeling 59.9 (2019): 3794-3802.

        ele-scaled : all terms are treated as in default, except for the old
        and new ele these are scaled with lambda^0.5, so as to be linear in
        energy, rather than lambda

        Parameters
        ----------
        functions : str or dict
            One of the predefined lambda protocols
            ['default','namd','quarters'] or a dictionary. Default "default".
        windows : int
            Number of windows which this lambda schedule is intended to be used
            with. This value is used to validate the lambda function.
        lambda_schedule : list of floats
            Schedule of lambda windows to be sampled. If ``None`` will default
            to a linear spacing of windows as defined by
            ``np.linspace(0. ,1. ,windows)``. Default ``None``.

        Attributes
        ----------
        functions : dict
            Lambda protocol to be used.
        lambda_schedule : list
            Schedule of windows to be sampled.
        """
        self.functions = copy.deepcopy(functions)

        # set the lambda schedule
        self.lambda_schedule = self._validate_schedule(lambda_schedule,
                                                       windows)
        if lambda_schedule:
            self.lambda_schedule = lambda_schedule
        else:
            self.lambda_schedule = np.linspace(0., 1., windows)

        if type(self.functions) == dict:
            self.type = 'user-defined'
        elif type(self.functions) == str:
            self.functions = None  # will be set later
            self.type = functions

        if self.functions is None:
            if self.type == 'default':
                self.functions = copy.deepcopy(
                                     LambdaProtocol.default_functions)
            elif self.type == 'namd':
                self.functions = {
                    'lambda_sterics_core': lambda x: x,
                    'lambda_electrostatics_core': lambda x: x,
                    'lambda_sterics_insert': lambda x: (3. / 2.) * x if x < (2. / 3.) else 1.0,
                    'lambda_sterics_delete': lambda x: 0.0 if x < (1. / 3.) else (x - (1. / 3.)) * (3. / 2.),
                    'lambda_electrostatics_insert': lambda x: 0.0 if x < 0.5 else 2.0 * (x - 0.5),
                    'lambda_electrostatics_delete': lambda x: 2.0 * x if x < 0.5 else 1.0,
                    'lambda_bonds': lambda x: x,
                    'lambda_angles': lambda x: x,
                    'lambda_torsions': lambda x: x
                }
            elif self.type == 'quarters':
                self.functions = {
                    'lambda_sterics_core': lambda x: x,
                    'lambda_electrostatics_core': lambda x: x,
                    'lambda_sterics_insert': lambda x: 0. if x < 0.5 else 1 if x > 0.75 else 4 * (x - 0.5),
                    'lambda_sterics_delete': lambda x: 0. if x < 0.25 else 1 if x > 0.5 else 4 * (x - 0.25),
                    'lambda_electrostatics_insert': lambda x: 0. if x < 0.75 else 4 * (x - 0.75),
                    'lambda_electrostatics_delete': lambda x: 4.0 * x if x < 0.25 else 1.0,
                    'lambda_bonds': lambda x: x,
                    'lambda_angles': lambda x: x,
                    'lambda_torsions': lambda x: x
                }
            elif self.type == 'ele-scaled':
                self.functions = {
                    'lambda_electrostatics_insert': lambda x: 0.0 if x < 0.5 else ((2*(x-0.5))**0.5),
                    'lambda_electrostatics_delete': lambda x: (2*x)**2 if x < 0.5 else 1.0
                }
            elif self.type == 'user-defined':
                self.functions = functions
            else:
                errmsg = f"LambdaProtocol type : {self.type} not recognised "
                raise ValueError(errmsg)

        self._validate_functions(n=windows)
        self._check_for_naked_charges()

    @staticmethod
    def _validate_schedule(schedule, windows):
        """
        Checks that the input lambda schedule is valid.

        Rules are:
          - Must begin at 0 and end at 1
          - Must be monotonically increasing

        Parameters
        ----------
        schedule : list of floats
            The lambda schedule. If ``None`` the method returns
            ``np.linspace(0. ,1. ,windows)``.
        windows : int
            Number of windows to be sampled.

        Returns
        -------
        schedule : list of floats
            A valid lambda schedule.
        """
        if schedule is None:
            return np.linspace(0., 1., windows)

        # Check end states
        if schedule[0] != 0 or schedule[-1] != 1:
            errmsg = ("end and start lambda windows must be lambda 0 and 1 "
                      "respectively")
            raise ValueError(errmsg)

        # Check monotonically increasing
        difference = np.diff(schedule)

        if not all(i >= 0. for i in difference):
            errmsg = "The lambda schedule is not monotonic"
            raise ValueError(errmsg)

        return schedule

    def _validate_functions(self, n=10):
        """Ensures that all the lambda functions adhere to the rules:
            - must begin at 0.
            - must finish at 1.
            - must be monotonically increasing

        Parameters
        ----------
        n : int, default 10
            number of grid points used to check monotonicity
        """
        # the individual lambda functions that must be defined for
        required_functions = list(LambdaProtocol.default_functions.keys())

        for function in required_functions:
            if function not in self.functions:
                # IA switched from warn to error here
                errmsg = (f"function {function} is missing from "
                          "self.lambda_functions.")
                raise ValueError(errmsg)

            # Check that the function starts and ends at 0 and 1 respectively
            if self.functions[function](0) != 0:
                raise ValueError("lambda functions must start at 0")
            if self.functions[function](1) != 1:
                raise ValueError("lambda fucntions must end at 1")

            # now validatate that it's monotonic
            global_lambda = np.linspace(0., 1., n)
            sub_lambda = [self.functions[function](lam) for
                          lam in global_lambda]
            difference = np.diff(sub_lambda)

            if not all(i >= 0. for i in difference):
                wmsg = (f"The function {function} is not monotonic as "
                        "typically expected.")
                warnings.warn(wmsg)

    def _check_for_naked_charges(self):
        """
        Checks that there are no cases where atoms have charge but no sterics.

        This avoids issues with singularities and/or excessive forces near
        the end states (even when using softcore electrostatics).
        """
        global_lambda = self.lambda_schedule

        def check_overlap(ele, sterics, global_lambda, functions, endstate):
            for lam in global_lambda:
                ele_val = functions[ele](lam)
                ster_val = functions[sterics](lam)
                # if charge > 0 and sterics == 0 raise error
                if ele_val != endstate and ster_val == endstate:
                    errmsg = ("There are states along this lambda schedule "
                              "where there are atoms with charges but no LJ "
                              f"interactions: {lam} {ele_val} {ster_val}")
                    raise ValueError(errmsg)

        # checking unique new terms first
        ele = 'lambda_electrostatics_insert'
        sterics = 'lambda_sterics_insert'
        check_overlap(ele, sterics, global_lambda, self.functions, endstate=0)

        # checking unique old terms now
        ele = 'lambda_electrostatics_delete'
        sterics = 'lambda_sterics_delete'
        check_overlap(ele, sterics, global_lambda, self.functions, endstate=1)

    def get_functions(self):
        return self.functions

    def plot_functions(self, lambda_schedule=None):
        """
        Plot the function for ease of visualisation.

        Parameters
        ----------
        shedule : np.ndarray
            The lambda schedule to plot the function along. If ``None`` plot
            the one stored within this class. Default ``None``.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 5))

        global_lambda = lambda_schedule if lambda_schedule else self.lambda_schedule

        for f in self.functions:
            plt.plot(global_lambda,
                     [self.functions[f](lam) for lam in global_lambda],
                     alpha=0.5, label=f)

        plt.xlabel('global lambda')
        plt.ylabel('sub-lambda')
        plt.legend()
        plt.show()


class RelativeAlchemicalState(AlchemicalState):
    """
    Relative AlchemicalState to handle all lambda parameters required for
    relative perturbations
    lambda = 1 refers to ON, i.e. fully interacting while
    lambda = 0 refers to OFF, i.e. non-interacting with the system
    all lambda functions will follow from 0 -> 1 following the master lambda
    lambda*core parameters perturb linearly
    lambda_sterics_insert and lambda_electrostatics_delete perturb in the
    first half of the protocol 0 -> 0.5
    lambda_sterics_delete and lambda_electrostatics_insert perturb in the
    second half of the protocol 0.5 -> 1

    Attributes
    ----------
    lambda_sterics_core
    lambda_electrostatics_core
    lambda_sterics_insert
    lambda_sterics_delete
    lambda_electrostatics_insert
    lambda_electrostatics_delete
    """

    class _LambdaParameter(AlchemicalState._LambdaParameter):
        pass

    lambda_sterics_core = _LambdaParameter('lambda_sterics_core')
    lambda_electrostatics_core = _LambdaParameter('lambda_electrostatics_core')
    lambda_sterics_insert = _LambdaParameter('lambda_sterics_insert')
    lambda_sterics_delete = _LambdaParameter('lambda_sterics_delete')
    lambda_electrostatics_insert = _LambdaParameter(
                                       'lambda_electrostatics_insert')
    lambda_electrostatics_delete = _LambdaParameter(
                                      'lambda_electrostatics_delete')

    def set_alchemical_parameters(self, global_lambda,
                                  lambda_protocol=LambdaProtocol()):
        """Set each lambda value according to the lambda_functions protocol.
        The undefined parameters (i.e. those being set to None) remain
        undefined.
        Parameters
        ----------
        lambda_value : float
            The new value for all defined parameters.
        """
        self.global_lambda = global_lambda
        for parameter_name in lambda_protocol.functions:
            lambda_value = lambda_protocol.functions[parameter_name](global_lambda)
            setattr(self, parameter_name, lambda_value)

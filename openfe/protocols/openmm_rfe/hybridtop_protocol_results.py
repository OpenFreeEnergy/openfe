# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
ProtocolUnitResults for Hybrid Topology methods using
OpenMM and OpenMMTools in a Perses-like manner.
"""

import logging
import pathlib
import warnings
from typing import Optional, Union

import gufe
import numpy as np
import numpy.typing as npt
from openff.units import Quantity
from openmmtools import multistate

logger = logging.getLogger(__name__)


class RelativeHybridTopologyProtocolResult(gufe.ProtocolResult):
    """Dict-like container for the output of a RelativeHybridTopologyProtocol"""

    def __init__(self, **data):
        super().__init__(**data)
        # data is mapping of str(repeat_id): list[protocolunitresults]
        # TODO: Detect when we have extensions and stitch these together?
        if any(len(pur_list) > 2 for pur_list in self.data.values()):
            raise NotImplementedError("Can't stitch together results yet")

    @staticmethod
    def compute_mean_estimate(dGs: list[Quantity]) -> Quantity:
        u = dGs[0].u
        # convert all values to units of the first value, then take average of magnitude
        # this would avoid a screwy case where each value was in different units
        vals = np.asarray([dG.to(u).m for dG in dGs])

        return np.average(vals) * u

    def get_estimate(self) -> Quantity:
        """Average free energy difference of this transformation

        Returns
        -------
        dG : openff.units.Quantity
          The free energy difference between the first and last states. This is
          a Quantity defined with units.
        """
        # TODO: Check this holds up completely for SAMS.
        dGs = [pus[0].outputs["unit_estimate"] for pus in self.data.values()]
        return self.compute_mean_estimate(dGs)

    @staticmethod
    def compute_uncertainty(dGs: list[Quantity]) -> Quantity:
        u = dGs[0].u
        # convert all values to units of the first value, then take average of magnitude
        # this would avoid a screwy case where each value was in different units
        vals = np.asarray([dG.to(u).m for dG in dGs])

        return np.std(vals) * u

    def get_uncertainty(self) -> Quantity:
        """The uncertainty/error in the dG value: The std of the estimates of
        each independent repeat
        """

        dGs = [pus[0].outputs["unit_estimate"] for pus in self.data.values()]
        return self.compute_uncertainty(dGs)

    def get_individual_estimates(self) -> list[tuple[Quantity, Quantity]]:
        """Return a list of tuples containing the individual free energy
        estimates and associated MBAR errors for each repeat.

        Returns
        -------
        dGs : list[tuple[openff.units.Quantity]]
          n_replicate simulation list of tuples containing the free energy
          estimates (first entry) and associated MBAR estimate errors
          (second entry).
        """
        dGs = [
            (pus[0].outputs["unit_estimate"], pus[0].outputs["unit_estimate_error"])
            for pus in self.data.values()
        ]
        return dGs

    def get_forward_and_reverse_energy_analysis(
        self,
    ) -> list[Optional[dict[str, Union[npt.NDArray, Quantity]]]]:
        """
        Get a list of forward and reverse analysis of the free energies
        for each repeat using uncorrelated production samples.

        The returned dicts have keys:
        'fractions' - the fraction of data used for this estimate
        'forward_DGs', 'reverse_DGs' - for each fraction of data, the estimate
        'forward_dDGs', 'reverse_dDGs' - for each estimate, the uncertainty

        The 'fractions' values are a numpy array, while the other arrays are
        Quantity arrays, with units attached.

        If the list entry is ``None`` instead of a dictionary, this indicates
        that the analysis could not be carried out for that repeat. This
        is most likely caused by MBAR convergence issues when attempting to
        calculate free energies from too few samples.


        Returns
        -------
        forward_reverse : list[Optional[dict[str, Union[npt.NDArray, openff.units.Quantity]]]]


        Raises
        ------
        UserWarning
          If any of the forward and reverse entries are ``None``.
        """
        forward_reverse = [
            pus[0].outputs["forward_and_reverse_energies"] for pus in self.data.values()
        ]

        if None in forward_reverse:
            wmsg = (
                "One or more ``None`` entries were found in the list of "
                "forward and reverse analyses. This is likely caused by "
                "an MBAR convergence failure caused by too few independent "
                "samples when calculating the free energies of the 10% "
                "timeseries slice."
            )
            warnings.warn(wmsg)

        return forward_reverse

    def get_overlap_matrices(self) -> list[dict[str, npt.NDArray]]:
        """
        Return a list of dictionary containing the MBAR overlap estimates
        calculated for each repeat.

        Returns
        -------
        overlap_stats : list[dict[str, npt.NDArray]]
          A list of dictionaries containing the following keys:
            * ``scalar``: One minus the largest nontrivial eigenvalue
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              overlap matrix
            * ``matrix``: Estimated overlap matrix of observing a sample from
              state i in state j
        """
        # Loop through and get the repeats and get the matrices
        overlap_stats = [pus[0].outputs["unit_mbar_overlap"] for pus in self.data.values()]

        return overlap_stats

    def get_replica_transition_statistics(self) -> list[dict[str, npt.NDArray]]:
        """The replica lambda state transition statistics for each repeat.

        Note
        ----
        This is currently only available in cases where a replica exchange
        simulation was run.

        Returns
        -------
        repex_stats : list[dict[str, npt.NDArray]]
          A list of dictionaries containing the following:
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              lambda state transition matrix
            * ``matrix``: The transition matrix estimate of a replica switching
              from state i to state j.
        """
        try:
            repex_stats = [
                pus[0].outputs["replica_exchange_statistics"] for pus in self.data.values()
            ]
        except KeyError:
            errmsg = "Replica exchange statistics were not found, did you run a repex calculation?"
            raise ValueError(errmsg)

        return repex_stats

    def get_replica_states(self) -> list[npt.NDArray]:
        """
        Returns the timeseries of replica states for each repeat.

        Returns
        -------
        replica_states : List[npt.NDArray]
          List of replica states for each repeat
        """

        def is_file(filename: str):
            p = pathlib.Path(filename)
            if not p.exists():
                errmsg = f"File could not be found {p}"
                raise ValueError(errmsg)
            return p

        replica_states = []

        for pus in self.data.values():
            nc = is_file(pus[0].outputs["nc"])
            dir_path = nc.parents[0]
            chk = is_file(dir_path / pus[0].outputs["last_checkpoint"]).name
            reporter = multistate.MultiStateReporter(
                storage=nc, checkpoint_storage=chk, open_mode="r"
            )
            replica_states.append(np.asarray(reporter.read_replica_thermodynamic_states()))
            reporter.close()

        return replica_states

    def equilibration_iterations(self) -> list[float]:
        """
        Returns the number of equilibration iterations for each repeat
        of the calculation.

        Returns
        -------
        equilibration_lengths : list[float]
        """
        equilibration_lengths = [
            pus[0].outputs["equilibration_iterations"] for pus in self.data.values()
        ]

        return equilibration_lengths

    def production_iterations(self) -> list[float]:
        """
        Returns the number of uncorrelated production samples for each
        repeat of the calculation.

        Returns
        -------
        production_lengths : list[float]
        """
        production_lengths = [pus[0].outputs["production_iterations"] for pus in self.data.values()]

        return production_lengths

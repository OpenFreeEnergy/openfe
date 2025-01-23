# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
import pathlib
from typing import Union, Optional
import numpy as np
from openmm import app
from openff.units import unit
from openff.models.types import FloatQuantity
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import calc_bonds

from .harmonic import (
    DistanceRestraintGeometry,
)

from .utils import _get_mda_topology_format, _get_mda_selection


class FlatBottomDistanceGeometry(DistanceRestraintGeometry):
    """
    A geometry class for a flat bottom distance restraint between two groups
    of atoms.
    """
    well_radius: FloatQuantity["nanometer"]


class COMDistanceAnalysis(AnalysisBase):
    """
    Get a timeseries of COM distances between two AtomGroups

    Parameters
    ----------
    group1 : MDAnalysis.AtomGroup
      Atoms defining the first centroid.
    group2 : MDANalysis.AtomGroup
      Atoms defining the second centroid.
    """
    _analysis_algorithm_is_parallelizable = False

    def __init__(self, group1, group2, **kwargs):
        super().__init__(group1.universe.trajectory, **kwargs)

        self.ag1 = group1
        self.ag2 = group2

    def _prepare(self):
        self.results.distances = np.zeros(self.n_frames)

    def _single_frame(self):
        com_dist = calc_bonds(
            self.ag1.center_of_mass(),
            self.ag2.center_of_mass(),
            box=self.ag1.universe.dimensions,
        )
        self.results.distances[self._frame_index] = com_dist

    def _conclude(self):
        pass


def get_flatbottom_distance_restraint(
    topology: Union[str, app.Topology],
    trajectory: Union[str, pathlib.Path],
    host_atoms: Optional[list[int]] = None,
    guest_atoms: Optional[list[int]] = None,
    host_selection: Optional[str] = None,
    guest_selection: Optional[str] = None,
    padding: unit.Quantity = 0.5 * unit.nanometer,
) -> FlatBottomDistanceGeometry:
    """
    Get a FlatBottomDistanceGeometry by analyzing the COM distance
    change between two sets of atoms.

    The ``well_radius`` is defined as the maximum COM distance plus
    ``padding``.

    Parameters
    ----------
    topology : Union[str, app.Topology]
      A topology defining the system.
    trajectory : Union[str, pathlib.Path]
      A coordinate trajectory for the system.
    host_atoms : Optional[list[int]]
      A list of host atoms indices. Either ``host_atoms`` or
      ``host_selection`` must be defined.
    guest_atoms : Optional[list[int]]
      A list of guest atoms indices. Either ``guest_atoms`` or
      ``guest_selection`` must be defined.
    host_selection : Optional[str]
      An MDAnalysis selection string to define the host atoms.
      Either ``host_atoms`` or ``host_selection`` must be defined.
    guest_selection : Optional[str]
      An MDAnalysis selection string to define the guest atoms.
      Either ``guest_atoms`` or ``guest_selection`` must be defined.
    padding : unit.Quantity
      A padding value to add to the ``well_radius`` definition.
      Must be in units compatible with nanometers.

    Returns
    -------
    FlatBottomDistanceGeometry
      An object defining a flat bottom restraint geometry.
    """
    u = mda.Universe(
        topology,
        trajectory,
        topology_format=_get_mda_topology_format(topology)
    )

    guest_ag = _get_mda_selection(u, guest_atoms, guest_selection)
    host_ag = _get_mda_selection(u, host_atoms, host_selection)
    guest_idxs = [a.ix for a in guest_ag]
    host_idxs = [a.ix for a in host_ag]

    if len(host_idxs) == 0 or len(guest_idxs) == 0:
        errmsg = (
            "no atoms found in either the host or guest atom groups"
            f"host_atoms: {host_idxs}"
            f"guest_atoms: {guest_idxs}"
        )
        raise ValueError(errmsg)

    com_dists = COMDistanceAnalysis(guest_ag, host_ag)
    com_dists.run()

    well_radius = com_dists.results.distances.max() * unit.angstrom + padding
    return FlatBottomDistanceGeometry(
        guest_atoms=guest_idxs, host_atoms=host_idxs, well_radius=well_radius
    )

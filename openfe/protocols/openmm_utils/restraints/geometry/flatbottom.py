# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
import abc
from pydantic.v1 import BaseModel, validator

import numpy as np
from openff.units import unit
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import calc_bonds, calc_angles

from .harmonic import (
    DistanceRestraintGeometry,
    _get_selection,
)


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

    def __init__(self, host_atoms, guest_atoms, search_distance, **kwargs):
        super().__init__(host_atoms.universe.trajectory, **kwargs)

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
    topology: Union[str, openmm.app.Topology],
    trajectory: pathlib.Path,
    topology_format: Optional[str] = None,
    host_atoms: Optional[list[int]] = None,
    guest_atoms: Optional[list[int]] = None,
    host_selection: Optional[str] = None,
    guest_selection: Optional[str] = None,
    padding: unit.Quantity = 0.5 * unit.nanometer,
) -> FlatBottomDistanceGeometry:
    u = mda.Universe(topology, trajectory, topology_format=topology_format)

    guest_ag = _get_selection(u, guest_atoms, guest_selection)
    host_ag = _get_selection(u, host_atoms, host_selection)

    com_dists = COMDistanceAnalysis(guest_ag, host_ag)
    com_dists.run()

    well_radius = com_dists.results.distances.max() * unit.angstrom + padding
    return FlatBottomDistanceGeometry(
        guest_atoms=guest_atoms, host_atoms=host_atoms, well_radius=well_radius
    )

# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from dataclasses import dataclass
from typing import Dict

from rdkit import Chem

from openfe.setup import Molecule
from openfe.utils.visualization import draw_mapping


@dataclass
class AtomMapping:
    """Simple container with the mapping between two Molecules

    Attributes
    ----------
    mol1, mol2 : Molecule
      the two Molecules in the mapping
    mol1_to_mol2 : dict
      maps the index of an atom in either molecule **A** or **B** to the other.
      If this atom has no corresponding atom, None is returned.

    """
    mol1: Molecule
    mol2: Molecule
    mol1_to_mol2: Dict[int, int]

    def __hash__(self):
        return hash(
            (hash(self.mol1), hash(self.mol2), tuple(self.mol1_to_mol2.items()))
        )

    @classmethod
    def from_perses(cls, perses_mapping):
        raise NotImplementedError()

    def _ipython_display_(self, d2d=None):
        """
        Visualize atom mapping in a Jupyter Notebook.

        Parameters
        ---------
        d2d : :class:`rdkit.Chem.Draw.rdMolDraw2D.MolDraw2D`
            If desired specify an instance of a MolDraw2D object. 
            Default ``None`` will use the MolDraw2DCairo backend.

        Returns
        -------
        Image: IPython.core.display.Image
            Image of the atom map
        """
        from IPython.display import Image, display

        return display(Image(draw_mapping(self.mol1_to_mol2,
                                          self.mol1.to_rdkit(),
                                          self.mol2.to_rdkit(), d2d)))

    def draw_to_file(self, fname: str, d2d=None):
        """
        Save atom map visualization to disk

        Parameters
        ---------
        d2d : :class:`rdkit.Chem.Draw.rdMolDraw2D.MolDraw2D`
            If desired specify an instance of a MolDraw2D object.
            Default ``None`` will write a .png file using thej MolDraw2DCairo
            backend.

        fname : str
            Name of file to save atom map
        """
        with open(fname, "wb") as f:
            f.write(draw_mapping(d2d))

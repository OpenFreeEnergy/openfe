# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from dataclasses import dataclass
from typing import Dict

from rdkit import Chem

from openfe.setup import LigandMolecule
from openfe.utils.visualization import draw_mapping


@dataclass
class LigandAtomMapping:
    """Simple container with the mapping between two Molecules

    Attributes
    ----------
    mol1, mol2 : LigandMolecule
      the two Molecules in the mapping
    mol1_to_mol2 : dict
      maps the index of an atom in either molecule **A** or **B** to the other.
      If this atom has no corresponding atom, None is returned.

    """
    mol1: LigandMolecule
    mol2: LigandMolecule
    mol1_to_mol2: Dict[int, int]

    def __hash__(self):
        return hash(
            (hash(self.mol1), hash(self.mol2),
             tuple(self.mol1_to_mol2.items()))
        )

    @classmethod
    def from_perses(cls, perses_mapping):
        raise NotImplementedError()

    def _ipython_display_(self, d2d=None):  # pragma: no-cover
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
            Default ``None`` will write a .png file using the MolDraw2DCairo
            backend.

        fname : str
            Name of file to save atom map
        """
        data = draw_mapping(self.mol1_to_mol2, self.mol1.to_rdkit(),
                            self.mol2.to_rdkit(), d2d)
        if type(data) == bytes:
            mode = "wb"
        else:
            mode = "w"

        with open(fname, mode) as f:
            f.write(draw_mapping(self.mol1_to_mol2, self.mol1.to_rdkit(),
                                 self.mol2.to_rdkit(), d2d))

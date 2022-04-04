# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from dataclasses import dataclass
import json
from typing import Dict
from openff.toolkit.utils.serialization import Serializable
from rdkit import Chem

from openfe.setup import SmallMoleculeComponent
from openfe.utils.visualization import draw_mapping


@dataclass
class LigandAtomMapping(Serializable):
    """Simple container with the mapping between two Molecules

    Attributes
    ----------
    molA, molB : SmallMoleculeComponent
      the two Molecules in the mapping
    molA_to_molB : dict
      maps the index of an atom in either molecule **A** or **B** to the other.
      If this atom has no corresponding atom, None is returned.

    """
    molA: SmallMoleculeComponent
    molB: SmallMoleculeComponent
    molA_to_molB: Dict[int, int]

    def to_dict(self):
        """Serialize to dict"""
        return {
            # openff serialization doesn't go deep, so stringify at this level
            'molA': self.molA.to_json(),
            'molB': self.molB.to_json(),
            'molA_to_molB': self.molA_to_molB,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Deserialize from dict"""
        # the mapping dict gets mangled sometimes
        mapping = d['molA_to_molB']
        fixed = {int(k): int(v) for k, v in mapping.items()}

        return cls(
            molA=SmallMoleculeComponent.from_dict(json.loads(d['molA'])),
            molB=SmallMoleculeComponent.from_dict(json.loads(d['molB'])),
            molA_to_molB=fixed,
        )

    def __hash__(self):
        return hash(
            (hash(self.molA), hash(self.molB),
             tuple(self.molA_to_molB.items()))
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

        return display(Image(draw_mapping(self.molA_to_molB,
                                          self.molA.to_rdkit(),
                                          self.molB.to_rdkit(), d2d)))

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
        data = draw_mapping(self.molA_to_molB, self.molA.to_rdkit(),
                            self.molB.to_rdkit(), d2d)
        if type(data) == bytes:
            mode = "wb"
        else:
            mode = "w"

        with open(fname, mode) as f:
            f.write(draw_mapping(self.molA_to_molB, self.molA.to_rdkit(),
                                 self.molB.to_rdkit(), d2d))

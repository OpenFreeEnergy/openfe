# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import json
import gufe
from typing import Any, Dict, List, Optional
import numpy as np
from numpy.typing import NDArray

from openfe.setup import SmallMoleculeComponent
from openfe.utils.visualization import draw_mapping


class LigandAtomMapping(gufe.AtomMapping):
    """Simple container with the mapping between two Molecules

    Attributes
    ----------
    molA, molB : SmallMoleculeComponent
      the two Molecules in the mapping
    molA_to_molB : dict
      maps the index of an atom in either molecule **A** or **B** to the other.
      If this atom has no corresponding atom, None is returned.
    annotations : Dict[str, Any]
      Mapping of annotation identifier to annotation data. Annotations may
      contain arbitrary JSON-serializable data. Annotation identifiers
      starting with ``ofe-`` may have special meaning in other parts of
      OpenFE. ``score`` is a reserved annotation identifier.
    """

    def __init__(
        self,
        molA: SmallMoleculeComponent,
        molB: SmallMoleculeComponent,
        molA_to_molB: Dict[int, int],
        annotations: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(molA, molB)
        self.molA_to_molB = molA_to_molB

        if annotations is None:
            # TODO: this should be a frozen dict
            annotations = {}

        self._annotations = annotations

    @classmethod
    def _defaults(self):
        return {}

    def _defaults(self):
        return {}

    @property
    def molA(self):
        return self.componentA

    @property
    def molB(self):
        return self.componentB

    def componentA_to_componentB(self):
        return self.molA_to_molB

    def componentB_to_componentA(self):
        return {v: k for k, v in self.molA_to_molB}

    def componentA_unique(self):
        return (i for i in range(self.componentA.to_rdkit().GetNumAtoms())
                if i not in self.molA_to_molB)

    def componentB_unique(self):
        return (i for i in range(self.componentB.to_rdkit().GetNumAtoms())
                if i not in self.molA_to_molB.values())

    @property
    def annotations(self):
        # return a copy (including copy of nested)
        return json.loads(json.dumps(self._annotations))

    def _to_dict(self):
        """Serialize to dict"""
        return {
            # openff serialization doesn't go deep, so stringify at this level
            'molA': json.dumps(self.componentA.to_dict(), sort_keys=True),
            'molB': json.dumps(self.componentB.to_dict(), sort_keys=True),
            'molA_to_molB': self.molA_to_molB,
            'annotations': json.dumps(self.annotations),
        }

    @classmethod
    def _from_dict(cls, d: dict):
        """Deserialize from dict"""
        # the mapping dict gets mangled sometimes
        mapping = d['molA_to_molB']
        fixed = {int(k): int(v) for k, v in mapping.items()}

        return cls(
            molA=SmallMoleculeComponent.from_dict(json.loads(d['molA'])),
            molB=SmallMoleculeComponent.from_dict(json.loads(d['molB'])),
            molA_to_molB=fixed,
            annotations=json.loads(d['annotations'])
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}(molA={self.molA!r}, "
                f"molB={self.molB!r}, molA_to_molB={self.molA_to_molB!r}, "
                f"annotations={self.annotations!r})")

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

    def with_annotations(self, annotations: Dict[str, Any]):
        """Create an new mapping based on this one with extra annotations.

        Parameters
        ----------
        annotations : Dict[str, Any]
            Annotation update for this mapping. New annotation keys will be
            added to the annotations dict; existing keys will be replaced by
            the data provided here.
        """
        return self.__class__(
            molA=self.molA,
            molB=self.molB,
            molA_to_molB=self.molA_to_molB,
            annotations=dict(**self.annotations, **annotations)
        )

    def get_distances(self) -> NDArray[np.float64]:
        """Return the distances between pairs of atoms in the mapping"""
        dists = []
        molA = self.molA.to_rdkit().GetConformer()
        molB = self.molB.to_rdkit().GetConformer()
        for i, j in self.molA_to_molB.items():
            dA = molA.GetAtomPosition(i)
            dB = molB.GetAtomPosition(j)
            dists.append(dA.Distance(dB))

        return np.array(dists)

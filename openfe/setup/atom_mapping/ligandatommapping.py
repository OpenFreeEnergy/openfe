# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import json
import gufe
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray

from openfe.setup import SmallMoleculeComponent
from openfe.utils.visualization import draw_mapping


class LigandAtomMapping(gufe.AtomMapping):
    """Simple container with the mapping between two Molecules

    """
    def __init__(
        self,
        componentA: SmallMoleculeComponent,
        componentB: SmallMoleculeComponent,
        componentA_to_componentB: dict[int, int],
        annotations: Optional[dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        componentA, componentB : SmallMoleculeComponent
          the ligand molecules on either end of the mapping
        componentA_to_componentB : dict[int, int]
          correspondence of indices of atoms between the two ligands
        annotations : dict[str, Any]
          Mapping of annotation identifier to annotation data. Annotations may
          contain arbitrary JSON-serializable data. Annotation identifiers
          starting with ``ofe-`` may have special meaning in other parts of
          OpenFE. ``score`` is a reserved annotation identifier.
        """
        super().__init__(componentA, componentB)
        self._compA_to_compB = componentA_to_componentB

        if annotations is None:
            # TODO: this should be a frozen dict
            annotations = {}

        self._annotations = annotations

    @classmethod
    def _defaults(self):
        return {}

    @property
    def componentA_to_componentB(self) -> dict[int, int]:
        return dict(self._compA_to_compB)

    @property
    def componentB_to_componentA(self) -> dict[int, int]:
        return {v: k for k, v in self._compA_to_compB.items()}

    @property
    def componentA_unique(self):
        return (i for i in range(self.componentA.to_rdkit().GetNumAtoms())
                if i not in self._compA_to_compB)

    @property
    def componentB_unique(self):
        return (i for i in range(self.componentB.to_rdkit().GetNumAtoms())
                if i not in self._compA_to_compB.values())

    @property
    def annotations(self):
        # return a copy (including copy of nested)
        return json.loads(json.dumps(self._annotations))

    def _to_dict(self):
        """Serialize to dict"""
        return {
            # openff serialization doesn't go deep, so stringify at this level
            'componentA': json.dumps(self.componentA.to_dict(), sort_keys=True),
            'componentB': json.dumps(self.componentB.to_dict(), sort_keys=True),
            'componentA_to_componentB': self._compA_to_compB,
            'annotations': json.dumps(self.annotations),
        }

    @classmethod
    def _from_dict(cls, d: dict):
        """Deserialize from dict"""
        # the mapping dict gets mangled sometimes
        mapping = d['componentA_to_componentB']
        fixed = {int(k): int(v) for k, v in mapping.items()}

        return cls(
            componentA=SmallMoleculeComponent.from_dict(
                json.loads(d['componentA'])),
            componentB=SmallMoleculeComponent.from_dict(
                json.loads(d['componentB'])),
            componentA_to_componentB=fixed,
            annotations=json.loads(d['annotations'])
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}(componentA={self.componentA!r}, "
                f"componentB={self.componentB!r}, "
                f"componentA_to_componentB={self._compA_to_compB!r}, "
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

        return display(Image(draw_mapping(self._compA_to_compB,
                                          self.componentA.to_rdkit(),
                                          self.componentB.to_rdkit(), d2d)))

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
        data = draw_mapping(self._compA_to_compB, self.componentA.to_rdkit(),
                            self.componentB.to_rdkit(), d2d)
        if type(data) == bytes:
            mode = "wb"
        else:
            mode = "w"

        with open(fname, mode) as f:
            f.write(draw_mapping(self._compA_to_compB, self.componentA.to_rdkit(),
                                 self.componentB.to_rdkit(), d2d))

    def with_annotations(self, annotations: dict[str, Any]):
        """Create an new mapping based on this one with extra annotations.

        Parameters
        ----------
        annotations : dict[str, Any]
            Annotation update for this mapping. New annotation keys will be
            added to the annotations dict; existing keys will be replaced by
            the data provided here.
        """
        return self.__class__(
            componentA=self.componentA,
            componentB=self.componentB,
            componentA_to_componentB=self._compA_to_compB,
            annotations=dict(**self.annotations, **annotations)
        )

    def get_distances(self) -> NDArray[np.float64]:
        """Return the distances between pairs of atoms in the mapping"""
        dists = []
        molA = self.componentA.to_rdkit().GetConformer()
        molB = self.componentB.to_rdkit().GetConformer()
        for i, j in self._compA_to_compB.items():
            dA = molA.GetAtomPosition(i)
            dB = molB.GetAtomPosition(j)
            dists.append(dA.Distance(dB))

        return np.array(dists)

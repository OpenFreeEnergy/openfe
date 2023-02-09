import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union, Optional, Dict

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex

try:
    import py3Dmol
except ImportError:
    pass    # Don't throw  error, will happen later


from gufe.mapping import AtomMapping

from openfe.utils import requires_package


def _get_max_dist_in_x(atom_mapping: AtomMapping) -> float:
    """helper function
        find the correct mol shift, so no overlap happens in vis

    Returns
    -------
    float
        maximal size of mol in x dimension
    """
    posA = atom_mapping.componentA.to_rdkit().GetConformer().GetPositions()
    posB = atom_mapping.componentB.to_rdkit().GetConformer().GetPositions()
    max_d = []

    for pos in [posA, posB]:
        d = np.zeros(shape=(len(pos), len(pos)))
        for i, pA in enumerate(pos):
            for j, pB in enumerate(pos[i:], start=i):
                d[i, j] = (pB - pA)[0]

        max_d.append(np.max(d))

    estm = float(np.round(max(max_d), 1))
    return estm if (estm > 5) else 5


def _translate(mol, shift:Union[Tuple[float, float, float], NDArray[np.float64]]):
    """
        shifts the molecule by the shift vector

    Parameters
    ----------
    mol : Chem.Mol
        rdkit mol that get shifted
    shift : Tuple[float, float, float]
        shift vector

    Returns
    -------
    Chem.Mol
        shifted Molecule (copy of original one)
    """
    mol = Chem.Mol(mol)
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        x, y, z = conf.GetAtomPosition(i)
        point = Point3D(x + shift[0], y + shift[1], z + shift[2])
        conf.SetAtomPosition(i, point)
    return mol


def _add_spheres(view:py3Dmol.view, mol1:Chem.Mol, mol2:Chem.Mol, mapping:Dict[int, int]):
    """
        will add spheres according to mapping to the view. (inplace!)

    Parameters
    ----------
    view : py3Dmol.view
        view to be edited
    mol1 : Chem.Mol
        molecule 1 of the mapping
    mol2 : Chem.Mol
        molecule 2 of the mapping
    mapping : Dict[int, int]
        mapping of atoms from mol1 to mol2
    """
    # Get colourmap of size mapping
    cmap = plt.cm.get_cmap("hsv", len(mapping))
    for i, pair in enumerate(mapping.items()):
        p1 = mol1.GetConformer().GetAtomPosition(pair[0])
        p2 = mol2.GetConformer().GetAtomPosition(pair[1])
        color = rgb2hex(cmap(i))
        view.addSphere(
            {
                "center": {"x": p1.x, "y": p1.y, "z": p1.z},
                "radius": 0.6,
                "color": color,
                "alpha": 0.8,
            }
        )
        view.addSphere(
            {
                "center": {"x": p2.x, "y": p2.y, "z": p2.z},
                "radius": 0.6,
                "color": color,
                "alpha": 0.8,
            }
        )


@requires_package("py3Dmol")
def draw_mapping_on_3Dstructure(
    mapping: AtomMapping,
    spheres: Optional[bool] = True,
    show_atomIDs: Optional[bool] = False,
    style: Optional[str] = "stick",
    shift: Optional[Union[Tuple[float, float, float], NDArray[np.float64]]] = None,
) -> py3Dmol.view:

    """
    Render relative transformation edge in 3D using py3Dmol.

    By default matching atoms will be annotated using colored spheres.

    Parameters
    ----------
    mapping : LigandAtomMapping
        The ligand transformation edge to visualize.
    spheres : bool, optional
        Whether or not to show matching atoms as spheres.
    show_atomIDs: bool, optional
        Whether or not to show atom ids in the mapping visualization
    style : str, optional
        Style in which to represent the molecules in py3Dmol.
    shift : Tuple of floats, optional
        Amount to shift molB by in order to visualize the two ligands.
        If None, the default shift will be estimated as the largest
        intraMol distance of both mols.

    Returns
    -------
    view : py3Dmol.view
        View of the system containing both molecules in the edge.
    """

    if shift is None:
        shift = np.array([_get_max_dist_in_x(mapping) * 1.5, 0, 0])
    else:
        shift = np.array(shift)

    molA = mapping.componentA.to_rdkit()
    molB = mapping.componentB.to_rdkit()

    mblock1 = Chem.MolToMolBlock(_translate(molA, -1 * shift))
    mblock2 = Chem.MolToMolBlock(_translate(molB, shift))

    view = py3Dmol.view(width=600, height=600)
    view.addModel(mblock1, "molA")
    view.addModel(mblock2, "molB")

    if spheres:
        _add_spheres(view, molA, molB, mapping.componentA_to_componentB)

    if show_atomIDs:
        view.addPropertyLabels(
            "index",
            {"not": {"resn": ["molA_overlay", "molA_overlay"]}},
            {
                "fontColor": "black",
                "font": "sans-serif",
                "fontSize": "10",
                "showBackground": "false",
                "alignment": "center",
            },
        )

    # middle fig
    overlay_mblock1 = Chem.MolToMolBlock(_translate(molA, 1 * shift))
    overlay_mblock2 = Chem.MolToMolBlock(_translate(molB, -1 * shift))

    view.addModel(overlay_mblock1, "molA_overlay")
    view.addModel(overlay_mblock2, "molB_overlay")

    view.setStyle({style: {}})

    view.zoomTo()
    return view

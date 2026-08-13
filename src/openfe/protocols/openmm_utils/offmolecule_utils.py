# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging
from typing import Any, Collection

from openff.toolkit import Molecule as OFFMolecule

from openfe import Component, SmallMoleculeComponent

logger = logging.getLogger(__name__)


def _set_offmol_metadata(
    offmol: OFFMolecule,
    key: Any,
    val: Any | None,
) -> None:
    """
    Set a given metadata entry for a whole Molecule.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      The Molecule to set the metadata for.
    key : Any
      The metadata key.
    val : Any
      The value to set the metadata entry to.
    """
    if val is None:
        for a in offmol.atoms:
            a.metadata.pop(key, None)
    else:
        for a in offmol.atoms:
            a.metadata[key] = val


def _get_offmol_metadata(offmol: OFFMolecule, key: Any) -> Any | None:
    """
    Get an offmol's given metadata entry and make sure it is
    consistent across all atoms in the Molecule.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      Molecule to get the metadata value from.
    key: Any
      The metadata entry key.

    Returns
    -------
    value : Any | None
      Metadata for the given key in the molecule. ``None`` if the
      Molecule does not have that metadata entry set, or if
      the value is inconsistent across all the atoms.
    """
    value: Any | None = None
    for a in offmol.atoms:
        if value is None:
            try:
                value = a.metadata[key]
            except KeyError:
                return None

        if value != a.metadata[key]:
            wmsg = f"Inconsistent metadata {key} in OFFMol: {offmol}"
            logger.warning(wmsg)
            return None

    return value


def _set_offmol_resname(
    offmol: OFFMolecule,
    resname: str | None,
) -> None:
    """
    Helper method to set offmol residue names

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      Molecule to assign a residue name to.
    resname : str | None
      Residue name to be set. Set to None to clear it.

    Returns
    -------
    None
    """
    _set_offmol_metadata(offmol, "residue_name", resname)


def _get_offmol_resname(offmol: OFFMolecule) -> str | None:
    """
    Helper method to get an offmol's residue name and make sure it is
    consistent across all atoms in the Molecule.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      Molecule to get the residue name from.

    Returns
    -------
    resname : Optional[str]
      Residue name of the molecule. ``None`` if the Molecule
      does not have a residue name, or if the residue name is
      inconsistent across all the atoms.
    """
    return _get_offmol_metadata(offmol, "residue_name")


def assign_offmol_residue_metadata(
    small_mols: dict[SmallMoleculeComponent, OFFMolecule],
    alchemical_components: Collection[Component],
) -> dict[SmallMoleculeComponent, str]:
    """
    Assign residue names and numbers to every SmallMoleculeComponent.

    Parameters
    ----------
    small_mols : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
      The molecules to name.
    alchemical_components : Collection[Component]
      The alchemical components.

    Returns
    -------
    assigned : dict[SmallMoleculeComponent, str]
      The resname assigned (or retained) for each component.
    """
    ligand_resname, ligand_stem = "LIG", "LG"
    cofactor_resname, cofactor_stem = "COF", "CF"

    alchemical = set(alchemical_components)

    used_names: set[str] = set()
    used_resnums: set[int] = set()
    for offmol in small_mols.values():
        name = _get_offmol_resname(offmol)
        resnum = _get_offmol_metadata(offmol, "residue_number")
        if name is not None:
            used_names.add(name)
        if resnum is not None:
            used_resnums.add(int(resnum))

    def _unique(default: str, stem: str) -> str:
        if default not in used_names:
            return default
        for i in range(1, 10):
            if (candidate := f"{stem}{i}") not in used_names:
                return candidate
        raise ValueError(f"Could not assign a unique residue name with stem {stem!r}.")

    lig_name = _unique(ligand_resname, ligand_stem)
    used_names.add(lig_name)
    cof_name = _unique(cofactor_resname, cofactor_stem)

    def _next_resnum() -> int:
        """Return the lowest residue number not already in use."""
        resnum = 1
        while resnum in used_resnums:
            resnum += 1
        used_resnums.add(resnum)
        return resnum

    assigned: dict[SmallMoleculeComponent, str] = {}
    for smc, offmol in small_mols.items():
        name = _get_offmol_resname(offmol)
        if name is None:
            name = lig_name if smc in alchemical else cof_name
            _set_offmol_resname(offmol, name)
        if _get_offmol_metadata(offmol, "residue_number") is None:
            _set_offmol_metadata(offmol, "residue_number", _next_resnum())
        assigned[smc] = name
    return assigned

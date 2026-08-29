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


def _get_used_offmol_property(offmols: list[OFFMolecule], offmol_property: str) -> set[str]:
    used_property: set[str] = set()
    for mol in offmols:
        prop = _get_offmol_metadata(mol, offmol_property)
        if prop is not None:
            used_property.add(prop)
    return used_property


def _get_unique_name(default: str, stem: str, used_names: set[str]) -> str:
    if default not in used_names:
        return default
    for i in range(1, 10):
        if (candidate := f"{stem}{i}") not in used_names:
            return candidate
    raise ValueError(f"Could not assign a unique residue name with stem {stem!r}.")


def _next_available_number(numbers: set[int]) -> int:
    """Return the lowest residue number not already in use."""
    number = 1
    while number in numbers:
        number += 1
    return number


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

    used_names = _get_used_offmol_property(list(small_mols.values()), "residue_name")
    used_resnums = {
        int(i)
        for i in _get_used_offmol_property(list(small_mols.values()), "residue_number")
    }

    lig_name = _get_unique_name(ligand_resname, ligand_stem, used_names)
    used_names.add(lig_name)
    cof_name = _get_unique_name(cofactor_resname, cofactor_stem, used_names)

    assigned: dict[SmallMoleculeComponent, str] = {}
    for smc, offmol in small_mols.items():
        name = _get_offmol_resname(offmol)
        if name is None:
            name = lig_name if smc in alchemical else cof_name
            _set_offmol_resname(offmol, name)
        if _get_offmol_metadata(offmol, "residue_number") is None:
            resnum = _next_available_number(used_resnums)
            _set_offmol_metadata(offmol, "residue_number", str(resnum))
            used_resnums.add(resnum)
        assigned[smc] = name
    return assigned

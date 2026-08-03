# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging
from typing import Any

from openff.toolkit import Molecule as OFFMolecule

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

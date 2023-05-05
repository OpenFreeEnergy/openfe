import pathlib
from openfe import SmallMoleculeComponent
from rdkit import Chem

def _smcs_from_sdf(sdf):
    supp = Chem.SDMolSupplier(str(sdf), removeHs=False)
    mols = [SmallMoleculeComponent(mol) for mol in supp]
    return mols

def _smcs_from_mol2(mol2):
    rdmol = Chem.MolFromMol2File(str(mol2), removeHs=False)
    return [SmallMoleculeComponent.from_rdkit(rdmol)]


def load_molecules(file_or_directory):
    """
    Load SmallMoleculeComponents in the given file or directory.

    This always returns a list. The input can be a directory, in which all
    files ending with .sdf are loaded as SDF and all files ending in .mol2
    are loaded as MOL2.

    Parameters
    ----------
    file_or_directory : pathlib.Path

    Returns
    -------
    list[SmallMoleculeComponent]
    """
    inp = pathlib.Path(file_or_directory)  # for shorter lines
    if inp.is_dir():
        sdf_files = [f for f in inp.iterdir() if f.suffix.lower() == ".sdf"]
        mol2_files = [f for f in inp.iterdir()
                      if f.suffix.lower() == ".mol2"]
    else:
        sdf_files = [inp] if inp.suffix.lower() == ".sdf" else []
        mol2_files = [inp] if inp.suffix.lower() == ".mol2" else []

    if not sdf_files + mol2_files:
        raise ValueError(f"Unable to find molecules in {file_or_directory}")

    sdf_mols = sum([_smcs_from_sdf(sdf) for sdf in sdf_files], [])
    mol2_mols = sum([_smcs_from_mol2(mol2) for mol2 in mol2_files], [])

    return sdf_mols + mol2_mols

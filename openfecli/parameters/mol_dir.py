# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import click
import glob
import itertools

from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED

# MOVE TO GUFE ####################################################
def _smcs_from_sdf(sdf):
    from openfe import SmallMoleculeComponent
    from rdkit import Chem
    supp = Chem.SDMolSupplier(sdf)
    mols = [SmallMoleculeComponent(mol) for mol in Chem.SDMolSupplier]

def _smcs_from_mol2(mol2):
    from openfe import SmallMoleculeComponent
    from rdkit import Chem
    rdmol = Chem.MolFromMol2File(mol2, removeHs=False)
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
    inp = file_or_directory  # for shorter lines
    if inp.is_dir():
        sdf_files = [f for f in inp.iterdir() if f.suffix.lower() == "sdf"]
        mol2_files = [f for f in inp.iterdir()
                      if f.suffix.lower() == "mol2"]
    else:
        sdf_files = [inp] if inp.suffix.lower() == "sdf" else []
        mol2_files = [inp] if inp.suffix.lower() == "mol2" else []

    sdf_mols = sum([_smcs_from_sdf(sdf) for sdf in sdf_files], [])
    mol2_mols = sum([_smcs_from_mol2(mol2) for mol2 in mol2_files], [])

    return sdf_mols + mol2_mols
# END MOVE TO GUFE ################################################


def molecule_getter(user_input, context):
    return load_molecules(user_input)

def _load_molecules_from_sdf(user_input, context):
    from rdkit import Chem

    sdfs = list(sorted(glob.glob(user_input + "/*.sdf")))
    if len(sdfs) == 0:  # this silences some stderr spam
        return NOT_PARSED

    from openfe import SmallMoleculeComponent

    # each sdf might be multiple molecules, so form generator of rdkit molecules
    rdkit_mols = itertools.chain.from_iterable(Chem.SDMolSupplier(f) for f in sdfs)
    try:
        mols = [SmallMoleculeComponent(r) for r in rdkit_mols]
    except ValueError:
        return NOT_PARSED

    return mols


def _load_molecules_from_mol2(user_input, context):
    from rdkit import Chem

    mol2s = list(sorted(glob.glob(user_input + "/*.mol2")))
    if len(mol2s) == 0:  # this silences some stderr spam
        return NOT_PARSED

    from openfe import SmallMoleculeComponent

    mols = []
    for mol2 in mol2s:
        try:
            rdmol = Chem.MolFromMol2File(mol2, removeHs=False)
            mols.append(SmallMoleculeComponent(rdkit=rdmol))
        except ValueError:  # any exception should try other strategies
            return NOT_PARSED
    return mols


get_molecules = MultiStrategyGetter(
    strategies=[_load_molecules_from_sdf, _load_molecules_from_mol2],
    error_message="Unable to generate a molecule from '{user_input}'.",
)

MOL_DIR = Option(
    "-M",
    "--molecules",
    type=click.Path(exists=True),
    help=(
        "A directory or file containing all molecules to be loaded, either"
        " as a single SDF or multiple MOL2/SDFs."
    ),
    getter=molecule_getter,
)

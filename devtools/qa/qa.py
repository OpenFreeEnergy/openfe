import pathlib
import gufe
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from rdkit import Chem
import openfe
from openff.units import unit


def get_alchemical_charge_difference(mapping) -> int:
    """
    Checks and returns the difference in formal charge between state A and B.

    Parameters
    ----------
    mapping : dict[str, ComponentMapping]
      Dictionary of mappings between transforming components.

    Returns
    -------
    int
      The formal charge difference between states A and B.
      This is defined as sum(charge state A) - sum(charge state B)
    """
    chg_A = Chem.rdmolops.GetFormalCharge(mapping.componentA.to_rdkit())
    chg_B = Chem.rdmolops.GetFormalCharge(mapping.componentB.to_rdkit())

    return chg_A - chg_B


def gen_transformations(
    prefix: str,
    dataset_dir: pathlib.Path,
    charge_model: str,
    system_group: str,
    system_name: str,
) -> None:
    # find the system
    system_dir = dataset_dir.joinpath(system_group, system_name)
    # define the protocol to use
    settings = RelativeHybridTopologyProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.engine_settings.compute_platform = "CUDA"
    settings.forcefield_settings.small_molecule_forcefield = "openff-2.2.1"
    settings.simulation_settings.time_per_iteration = 2.5 * unit.picosecond
    settings.simulation_settings.real_time_analysis_interval = 1 * unit.nanosecond
    settings.output_settings.checkpoint_interval = 1 * unit.nanosecond
    settings.solvation_settings.box_shape = "dodecahedron"
    settings.forcefield_settings.nonbonded_cutoff = 0.9 * unit.nanometer

    # load the network
    network = gufe.LigandNetwork.from_graphml(
        system_dir.joinpath("lomap_network.graphml").read_text()
    )

    # load the ligands with charges
    supplier = Chem.SDMolSupplier(
        system_dir.joinpath(f"ligands_{charge_model}.sdf"), removeHs=False
    )
    ligands_by_name = {}
    for mol in supplier:
        ofe_mol = gufe.SmallMoleculeComponent.from_rdkit(mol)
        ligands_by_name[ofe_mol.name] = ofe_mol

    # load the protein
    protein = gufe.ProteinComponent.from_pdb_file(
        system_dir.joinpath("protein.pdb").as_posix()
    )

    # check for cofactors
    co_file = system_dir.joinpath(f"cofactors_{charge_model}.sdf")
    cofactors = None
    if co_file.exists():
        cofactors = []
        supplier = Chem.SDMolSupplier(co_file, removeHs=False)
        for mol in supplier:
            cofactors.append(gufe.SmallMoleculeComponent.from_rdkit(mol))

    # define the standard solvent
    solvent = gufe.SolventComponent()

    # build the transforms
    transformations = []
    for edge in network.edges:
        # make the edge again using ligands with charges
        new_edge = gufe.LigandAtomMapping(
            componentA=ligands_by_name[edge.componentA.name],
            componentB=ligands_by_name[edge.componentB.name],
            componentA_to_componentB=edge.componentA_to_componentB,
        )

        # Check if we need to use charge changeing settings
        if get_alchemical_charge_difference(new_edge) == 0:
            settings.alchemical_settings.explicit_charge_correction = False
            settings.simulation_settings.production_length = 5.0 * unit.nanosecond
            settings.simulation_settings.n_replicas = 11
            settings.lambda_settings.lambda_windows = 11
        else:
            settings.alchemical_settings.explicit_charge_correction = True
            settings.simulation_settings.production_length = 20 * unit.nanosecond
            settings.simulation_settings.n_replicas = 22
            settings.lambda_settings.lambda_windows = 22

        # create the transformations for the bound and solvent legs
        for leg in ["solvent", "complex"]:
            system_a_dict = {"ligand": new_edge.componentA, "solvent": solvent}
            system_b_dict = {"ligand": new_edge.componentB, "solvent": solvent}
            if leg == "complex":
                system_a_dict["protein"] = protein
                system_b_dict["protien"] = protein
                settings.solvation_settings.solvent_padding = 1.0 * unit.nanometer

                if cofactors is not None:
                    for i, cofactor in enumerate(cofactors):
                        cofactor_name = f"cofactor_{i}"
                        system_a_dict[cofactor_name] = cofactor
                        system_b_dict[cofactor_name] = cofactor
            elif leg == "solvent":
                settings.solvation_settings.solvent_padding = 1.5 * unit.nanometer

            system_a = gufe.ChemicalSystem(system_a_dict)
            system_b = gufe.ChemicalSystem(system_b_dict)

            name = f"{leg}_{new_edge.componentA.name}_{new_edge.componentB.name}"

            protocol = RelativeHybridTopologyProtocol(settings=settings)
            transformation = openfe.Transformation(
                stateA=system_a,
                stateB=system_b,
                mapping=new_edge,
                protocol=protocol,
                name=name,
            )

            transformations.append(transformation)

    # create the network, profit!
    alchemical_network = gufe.AlchemicalNetwork(edges=transformations)
    prefix = pathlib.Path(prefix)
    system_dir = prefix / pathlib.Path(system_name)
    transform_dir = system_dir / pathlib.Path(f"transformations_{system_name}")
    transform_dir.mkdir(parents=True, exist_ok=True)
    for transformation in alchemical_network.edges:
        transformation_name = transformation.name or transformation.key
        filename = pathlib.Path(f"{transformation_name}.json")
        transformation.to_json(transform_dir / filename)
    alchemical_network.to_json(system_dir / pathlib.Path("alchemical_network.json"))


# Make a function that will pull this data down or point to a local path
dataset_dir = pathlib.Path(
    "/home/mmh/Projects/openfe-benchmarks/openfe_benchmarks/data/industry_benchmark_systems"
)
# declare the charge settings and the system you want to generate the network for
# CHARGE_MODEL = "antechamber_am1bcc"
# or
PREFIX = "QA-1.4.0-OpenMM-8.2"
CHARGE_MODEL = "openeye_elf10"
SYSTEM_GROUP = "jacs_set"
SYSTEM_NAME = "tyk2"

gen_transformations(
    prefix=PREFIX,
    dataset_dir=dataset_dir,
    charge_model=CHARGE_MODEL,
    system_group="jacs_set",
    system_name="tyk2",
)
gen_transformations(
    prefix=PREFIX,
    dataset_dir=dataset_dir,
    charge_model=CHARGE_MODEL,
    system_group="jacs_set",
    system_name="p38",
)
gen_transformations(
    prefix=PREFIX,
    dataset_dir=dataset_dir,
    charge_model=CHARGE_MODEL,
    system_group="mcs_docking_set",
    system_name="hne",
)

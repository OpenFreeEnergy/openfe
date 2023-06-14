# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from typing import List
from openfecli.utils import write, print_duration
from openfecli import OFECommandPlugin
from openfecli.parameters import (
    MOL_DIR, PROTEIN, MAPPER, OUTPUT_DIR, COFACTORS,
)
from openfecli.plan_alchemical_networks_utils import plan_alchemical_network_output


def plan_rbfe_network_main(
    mapper,
    mapping_scorer,
    ligand_network_planner,
    small_molecules,
    solvent,
    protein,
    cofactors,
):
    """Utility method to plan a relative binding free energy network.

    Parameters
    ----------
    mapper : LigandAtomMapper
        the mapper to use to generate the mapping
    mapping_scorer : Callable
        scorer, that evaluates the generated mappings
    ligand_network_planner : Callable
        function building the network from the ligands, mappers and mapping_scorer
    small_molecules : Iterable[SmallMoleculeComponent]
        ligands of the system
    solvent : SolventComponent
        Solvent component used for solvation
    protein : ProteinComponent
        protein component for complex simulations, to which the ligands are bound
    cofactors : Iterable[SmallMoleculeComponent]
        any cofactors alongisde the protein, can be empty list

    Returns
    -------
    Tuple[AlchemicalNetwork, LigandNetwork]
        Alchemical network with protocol for executing simulations, and the
        associated ligand network
    """

    from openfe.setup.alchemical_network_planner.relative_alchemical_network_planner import (
        RBFEAlchemicalNetworkPlanner,
    )

    network_planner = RBFEAlchemicalNetworkPlanner(
        mappers=[mapper],
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
    )
    alchemical_network = network_planner(
        ligands=small_molecules, solvent=solvent, protein=protein,
        cofactors=cofactors,
    )
    return alchemical_network, network_planner._ligand_network


@click.command(
    "plan-rbfe-network",
    short_help=(
        "Plan a relative binding free energy network, saved as JSON files "
        "for the quickrun command."
    )
)
@MOL_DIR.parameter(
    required=True, help=MOL_DIR.kwargs["help"] + " Any number of sdf paths."
)
@PROTEIN.parameter(
    multiple=False, required=True, default=None, help=PROTEIN.kwargs["help"]
)
@COFACTORS.parameter(
    multiple=False, required=False, default=None, help=COFACTORS.kwargs["help"]
)
@OUTPUT_DIR.parameter(
    help=OUTPUT_DIR.kwargs["help"] + " Defaults to `./alchemicalNetwork`.",
    default="alchemicalNetwork",
)
@print_duration
def plan_rbfe_network(
    molecules: List[str], protein: str, cofactors: str, output_dir: str
):
    """
    Plan a relative binding free energy network, saved as JSON files for
    the quickrun command.

    This tool is an easy way to set up a RBFE calculation campaign.
    The JSON files this outputs can be used to run each leg of the campaign.
    For customized setups, please consider using the Python layer of
    openfe. This tool makes the following choices:

    * Atom mappings performed by LOMAP, with settings max3d=1.0 and
      element_change=False

    * Minimal spanning network as the network planner, with LOMAP default
      score as the weight function

    * Water as solvent, with NaCl at 0.15 M.

    * Protocol is the OpenMM-based relative hybrid topology protocol, with
      default settings.

    The generated Network will be stored in a folder containing for each
    transformation a JSON file, that can be run with quickrun.
    """
    write("RBFE-NETWORK PLANNER")
    write("______________________")
    write("")

    write("Parsing in Files: ")

    from gufe import SolventComponent
    from openfe.setup.atom_mapping.lomap_scorers import (
        default_lomap_score,
    )
    from openfe.setup import LomapAtomMapper
    from openfe.setup.ligand_network_planning import (
        generate_minimal_spanning_network,
    )

    # INPUT
    write("\tGot input: ")

    small_molecules = MOL_DIR.get(molecules)
    write(
        "\t\tSmall Molecules: "
        + " ".join([str(sm) for sm in small_molecules])
    )

    protein = PROTEIN.get(protein)
    write("\t\tProtein: " + str(protein))

    if cofactors is not None:
        cofactors = COFACTORS.get(cofactors)
    else:
        cofactors = []
    write("\t\tCofactors: " + str(cofactors))

    solvent = SolventComponent()
    write("\t\tSolvent: " + str(solvent))
    write("")

    write("Using Options:")
    mapper_obj = LomapAtomMapper(time=20, threed=True, element_change=False, max3d=1)
    write("\tMapper: " + str(mapper_obj))

    # TODO:  write nice parameter
    mapping_scorer = default_lomap_score
    write("\tMapping Scorer: " + str(mapping_scorer))

    # TODO:  write nice parameter
    ligand_network_planner = generate_minimal_spanning_network
    write("\tNetworker: " + str(ligand_network_planner))
    write("")

    # DO
    write("Planning RBFE-Campaign:")
    alchemical_network, ligand_network = plan_rbfe_network_main(
        mapper=mapper_obj,
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
        small_molecules=small_molecules,
        solvent=solvent,
        protein=protein,
        cofactors=cofactors,
    )
    write("\tDone")
    write("")

    # OUTPUT
    write("Output:")
    write("\tSaving to: " + str(output_dir))
    plan_alchemical_network_output(
        alchemical_network=alchemical_network,
        ligand_network=ligand_network,
        folder_path=OUTPUT_DIR.get(output_dir),
    )


PLUGIN = OFECommandPlugin(
    command=plan_rbfe_network, section="Network Planning", requires_ofe=(0, 3)
)

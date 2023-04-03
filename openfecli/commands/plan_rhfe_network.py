# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe


"""
Here I want to build the cmd tool for easy campaigner with RHFE. The output should be runnable with quickrun directly!
    So user experience would be:
        easy_campaign -l sdf_dir -p receptor.pdb -> Alchem network
        quickrun -i alchem_network

"""

import click
from typing import List

from openfecli.utils import write
from openfecli import OFECommandPlugin
from openfecli.parameters import MOL_DIR, MAPPER, OUTPUT_DIR
from openfecli.commands.plan_alchemical_networks_utils import plan_alchemical_network_output


def plan_rhfe_network_main(
    mapper, mapping_scorer, ligand_network_planner, small_molecules, solvent
):
    """Utility method to plan a relative hydration free energy network.

    Parameters
    ----------
    mapper : LigandAtomMapper
        the mapper to use to generate the mapping
    mapping_scorer : Callable
        scorer, that evaluates the generated mappings
    ligand_network_planner : Callable
        function building the network from the ligands, mappers and mapping_scorer
    small_molecules : Iterable[SmallMoleculeComponent]
        molecules of the system
    solvent : SolventComponent
        Solvent component used for solvation

    Returns
    -------
    AlchemicalNetwork
        Alchemical network with protocol for executing simulations.
    """
    from openfe.setup.alchemical_network_planner.relative_alchemical_network_planner import (
        RHFEAlchemicalNetworkPlanner,
    )

    network_planner = RHFEAlchemicalNetworkPlanner(
        mappers=[mapper],
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
    )
    alchemical_network = network_planner(
        ligands=small_molecules, solvent=solvent
    )

    return alchemical_network


@click.command(
    "plan-rhfe-network",
    short_help="Run a planning session for relative hydration free energies, "
               "saved in a dir with multiple JSON files (future: will be one JSON file)",
)
@MOL_DIR.parameter(
    required=True, help=MOL_DIR.kwargs["help"] + " Any number of sdf paths."
)
@OUTPUT_DIR.parameter(
    help=OUTPUT_DIR.kwargs["help"] + " Defaults to `./alchemicalNetwork`.",
    default="alchemicalNetwork",
)
@MAPPER.parameter(required=False, default="LomapAtomMapper")
def plan_rhfe_network(mol_dir: List[str], output_dir: str, mapper: str):
    """
    this command line tool allows relative hydration free energy calculations
    to be planned and returns an alchemical network as a directory.
    """

    from gufe import SolventComponent
    from openfe.setup.atom_mapping.lomap_scorers import (
        default_lomap_score,
    )
    from openfe.setup.ligand_network_planning import (
        generate_minimal_spanning_network,
    )

    write("RHFE-NETWORK PLANNER")
    write("______________________")
    write("")

    # INPUT
    write("Parsing in Files: ")
    write("\tGot input: ")

    small_molecules = MOL_DIR.get(mol_dir)
    write(
        "\t\tSmall Molecules: "
        + " ".join([str(sm) for sm in small_molecules])
    )

    solvent = SolventComponent()
    write("\t\tSolvent: " + str(solvent))
    write("")

    write("Using Options:")
    mapper_obj = MAPPER.get(mapper)()
    write("\tMapper: " + str(mapper_obj))

    # TODO:  write nice parameter
    mapping_scorer = default_lomap_score
    write("\tMapping Scorer: " + str(mapping_scorer))

    # TODO: write nice parameter
    ligand_network_planner = generate_minimal_spanning_network
    write("\tNetworker: " + str(ligand_network_planner))
    write("")

    # DO
    write("Planning RHFE-Campaign:")
    alchemical_network = plan_rhfe_network_main(
        mapper=mapper_obj,
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
        small_molecules=small_molecules,
        solvent=solvent,
    )
    write("\tDone")
    write("")

    # OUTPUT
    write("Output:")
    write("\tSaving to: " + output_dir)
    plan_alchemical_network_output(
        alchemical_network=alchemical_network,
        folder_path=OUTPUT_DIR.get(output_dir),
    )


PLUGIN = OFECommandPlugin(
    command=plan_rhfe_network, section="Setup", requires_ofe=(0, 3)
)

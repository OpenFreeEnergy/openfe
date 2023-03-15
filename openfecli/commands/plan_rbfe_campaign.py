# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe


"""
Here I want to build the cmd tool for easy campaigner with RBFE. The output should be runnable with quickrun directly!
    So user experience would be:
        easy_campaign -l sdf_dir -p receptor.pdb -> Alchem network
        quickrun -i alchem_network

"""
import json
import click
from openfecli.utils import write
from openfecli import OFECommandPlugin
from openfecli.parameters import MOL, PROTEIN, MAPPER, OUTPUT_FILE_AND_EXT


@click.command(
    "plan_rbfe_campaign", short_help="Run a planning session, saved as a JSON file"
)
@MOL.parameter(
    multiple=True, required=True, help=MOL.kwargs["help"] + " Any number of sdf paths."
)
@PROTEIN.parameter(
    multiple=False, required=True, default=None, help=MOL.kwargs["help"]
)
@OUTPUT_FILE_AND_EXT.parameter(
    help=OUTPUT_FILE_AND_EXT.kwargs["help"] + " (JSON format)"
)
@MAPPER.parameter(required=False, default="LomapAtomMapper")
def plan_rbfe_campaign(mol: str, receptor_pdb_path: str, output: str, mapper: str):

    write("Parsing in Files: ")
    write("\tGot input: ")

    small_molecules = [MOL.get(m) for m in mol]
    write("\t\tSmall Molecules: " + " ".join([str(sm) for sm in small_molecules]))

    protein = PROTEIN.get(receptor_pdb_path)
    write("\t\tProtein: " + str(protein))

    from gufe import SolventComponent
    solvent = SolventComponent()
    write("\t\tSolvent: " + str(solvent))
    write("")

    write("Using Options:")
    mapper_obj = MAPPER.get(mapper)()
    write("\tMapper: " + str(mapper_obj))

    from openfe.setup.atom_mapping.lomap_scorers import (
        default_lomap_score,
    )  # write nice parameter

    mapping_scorer = default_lomap_score
    write("\tMapping Scorer: " + str(mapping_scorer))

    from openfe.setup.ligand_network_planning import (
        generate_minimal_spanning_network,
    )  # write nice parameter

    ligand_network_planner = generate_minimal_spanning_network
    write("\tNetworker: " + str(ligand_network_planner))
    write("")

    write("Planning RBFE-Campaign:")
    alchemical_network = plan_rbfe_campaign_main(
        mapper=mapper_obj,
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
        small_molecules=small_molecules,
        solvent=solvent,
        protein=protein,
    )

    write("Saving to: " + str(output))
    an_dict = alchemical_network.to_dict()
    file, ext = OUTPUT_FILE_AND_EXT.get(output)  # check ext.
    json.dump(an_dict, open(file + "." + ext, "w"))


def plan_rbfe_campaign_main(
    mapper, mapping_scorer, ligand_network_planner, small_molecules, solvent, protein
):
    from openfe.setup.alchemical_network_planner.easy_alchemical_network_planner import (
        RBFEAlchemicalNetworkPlanner,
    )

    campaigner = RBFEAlchemicalNetworkPlanner(
        mapper=mapper,
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
    )
    alchemical_network = campaigner(
        ligands=small_molecules, solvent=solvent, receptor=protein
    )
    return alchemical_network


PLUGIN = OFECommandPlugin(
    command=plan_rbfe_campaign, section="Setup", requires_ofe=(0, 3)
)

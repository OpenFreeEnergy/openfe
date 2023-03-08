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
    'rbfe_campaigner',
    short_help="Run a planning session, saved as a JSON file"
)
@MOL.parameter(multiple=True, required=True,
               help=MOL.kwargs['help']+" Any number of sdf paths.")
@PROTEIN.parameter(multiple=False, required=False, default=None,
               help=MOL.kwargs['help'])
@OUTPUT_FILE_AND_EXT.parameter(
    help=OUTPUT_FILE_AND_EXT.kwargs['help'] + " (JSON format)"
)
@MAPPER.parameter(required=False, default="lomap")
def plan_rbfe_campaign(mol:str, receptor_pdb_path:str, output:str, mapper:str):

    write("Parsing in Files: ")
    write("\tGot input: ")
    from gufe import SolventComponent   
    small_molecules = [MOL.get(m) for m in mol]
    write("\t\tSmall Molecules: "+" ".join([str(sm) for sm in small_molecule_paths]))
        
    if(protein is not None):
        protein = PROTEIN.get(receptor_pdb_path)
    write("\t\tProtein: "+str(protein))

    solvent = SolventComponent()
    write("\t\tSolvent: "+str(solvent))
    write("")
    
    write("Using Options:")
    mapper_obj = MAPPER.get(mapper)()
    write("\tMapper: "+str(mapper_obj))

    from openfe.setup.atom_mapping.lomap_scorers import default_lomap_score #write nice parameter
    mapping_scorers = [default_lomap_score]
    write("\tMapping Scorer: "+str(mapping_scorers))

    from openfe.setup.ligand_network_planning import minimal_spanning_graph #write nice parameter 
    networker = minimal_spanning_graph    
    write("\tNetworker: "+str(networker))
    write("")

    write("Planning RBFE-Campaign:")
    from openfe.setup.campaigners.easy_campaigner import easy_rbfe_campainger
    campaigner = easy_rbfe_campainger(mapper=mapper_obj,
                                      mapping_scorers=mapping_scorers,
                                      networker=networker)
    alchemical_network = campaigner(ligands=small_molecules, solvent=solvent, receptor=protein)

    write("Saving to: "+str(output))
    an_dict = alchemical_network.to_dict()
    file, ext = OUTPUT_FILE_AND_EXT.get(output) #check ext.
    json.dump(an_dict, open(file+"."+ext, "w"))


PLUGIN = OFECommandPlugin(
    command=plan_rbfe_campaign,
    section="Setup",
    requires_ofe=(0, 3)
)

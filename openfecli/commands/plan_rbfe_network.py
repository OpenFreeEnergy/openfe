# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli.utils import write, print_duration
from openfecli import OFECommandPlugin
from openfecli.parameters import (
    MOL_DIR, PROTEIN, OUTPUT_DIR, COFACTORS, YAML_OPTIONS, NCORES, OVERWRITE
)

def plan_rbfe_network_main(
    mapper,
    mapping_scorer,
    ligand_network_planner,
    small_molecules,
    solvent,
    protein,
    cofactors,
    partial_charge_settings,
    processors,
    overwrite_charges
):
    """Utility method to plan a relative binding free energy network.

    Parameters
    ----------
    mapper : list[LigandAtomMapper]
        list of mappers to use to generate the mapping
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
        any cofactors alongside the protein, can be empty list
    partial_charge_settings : OpenFFPartialChargeSettings
        how to assign partial charges to the input ligands
        (if they don't already have partial charges).
    processors: int
        The number of processors that should be used when generating the charges
    overwrite_charges: bool
        If any partial charges already present on the small molecules should be overwritten

    Returns
    -------
    Tuple[AlchemicalNetwork, LigandNetwork]
        Alchemical network with protocol for executing simulations, and the
        associated ligand network
    """

    from openfe.setup.alchemical_network_planner.relative_alchemical_network_planner import (
        RBFEAlchemicalNetworkPlanner,
    )
    from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges

    write("assigning ligand partial charges -- this may be slow")

    charged_small_molecules = bulk_assign_partial_charges(
        molecules=small_molecules,
        overwrite=overwrite_charges,
        method=partial_charge_settings.partial_charge_method,
        toolkit_backend=partial_charge_settings.off_toolkit_backend,
        generate_n_conformers=partial_charge_settings.number_of_conformers,
        nagl_model=partial_charge_settings.nagl_model,
        processors=processors
    )

    if cofactors:
        write("assigning cofactor partial charges -- this may be slow")

        cofactors = bulk_assign_partial_charges(
            molecules=cofactors,
            overwrite=overwrite_charges,
            method=partial_charge_settings.partial_charge_method,
            toolkit_backend=partial_charge_settings.off_toolkit_backend,
            generate_n_conformers=partial_charge_settings.number_of_conformers,
            nagl_model=partial_charge_settings.nagl_model,
            processors=processors
        )

    network_planner = RBFEAlchemicalNetworkPlanner(
        mappers=mapper,
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
    )
    alchemical_network = network_planner(
        ligands=charged_small_molecules, solvent=solvent, protein=protein,
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
    multiple=True, required=False, default=None, help=COFACTORS.kwargs["help"]
)
@YAML_OPTIONS.parameter(
    multiple=False, required=False, default=None,
    help=YAML_OPTIONS.kwargs["help"],
)
@OUTPUT_DIR.parameter(
    help=OUTPUT_DIR.kwargs["help"] + " Defaults to `./alchemicalNetwork`.",
    default="alchemicalNetwork",
)
@NCORES.parameter(
    help=NCORES.kwargs["help"],
    default=1,
)
@OVERWRITE.parameter(
    help=OVERWRITE.kwargs["help"],
    default=OVERWRITE.kwargs["default"],
    is_flag=True
)
@print_duration
def plan_rbfe_network(
        molecules: list[str], protein: str, cofactors: tuple[str],
        yaml_settings: str,
        output_dir: str,
        n_cores: int,
        overwrite_charges: bool
):
    """
    Plan a relative binding free energy network, saved as JSON files for
    the quickrun command.

    This tool is an easy way to set up a RBFE calculation campaign.
    The JSON files this outputs can be used to run each leg of the campaign.
    openfe.
    The generated Network will be stored in a folder containing for each
    transformation a JSON file, that can be run with quickrun.

    By default, this tool makes the following choices:

    * Atom mappings performed by LOMAP, with settings max3d=1.0 and
      element_change=False
    * Minimal spanning network as the network planner, with LOMAP default
      score as the weight function
    * Water as solvent, with NaCl counter ions at 0.15 M concentration.
    * Protocol is the OpenMM-based relative hybrid topology protocol, with
      default settings.

    These choices can be customized by creating a settings yaml file,
    which is passed in via the ``-s settings.yaml`` option,
    which is detailed in the Options section.
    For more advanced setups, please consider using the Python layer of openfe.
    """
    from openfecli.plan_alchemical_networks_utils import plan_alchemical_network_output

    write("RBFE-NETWORK PLANNER")
    write("______________________")
    write("")

    write("Parsing in Files: ")

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
        cofactors = sum((COFACTORS.get(c) for c in cofactors), start=[])
    else:
        cofactors = []
    write("\t\tCofactors: " + str(cofactors))

    yaml_options = YAML_OPTIONS.get(yaml_settings)
    mapper_obj = yaml_options.mapper
    mapping_scorer = yaml_options.scorer
    ligand_network_planner = yaml_options.ligand_network_planner
    solvent = yaml_options.solvent
    partial_charge = yaml_options.partial_charge

    write("\t\tSolvent: " + str(solvent))
    write("")

    write("Using Options:")
    write("\tMapper: " + str(mapper_obj))

    # TODO:  write nice parameter
    write("\tMapping Scorer: " + str(mapping_scorer))

    # TODO:  write nice parameter
    write("\tNetwork Generation: " + str(ligand_network_planner))

    write("\tPartial Charge Generation: " + str(partial_charge.partial_charge_method))
    if overwrite_charges:
        write("\tOverwriting partial charges")
    write("")

    # DO
    write("Planning RBFE-Campaign:")
    alchemical_network, ligand_network = plan_rbfe_network_main(
        mapper=[mapper_obj],
        mapping_scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
        small_molecules=small_molecules,
        solvent=solvent,
        protein=protein,
        cofactors=cofactors,
        partial_charge_settings=partial_charge,
        processors=n_cores,
        overwrite_charges=overwrite_charges
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

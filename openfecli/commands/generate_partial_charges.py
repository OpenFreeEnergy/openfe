import click
from openfecli import OFECommandPlugin
from openfecli.parameters import MOL_DIR, YAML_OPTIONS, OUTPUT_FILE_AND_EXT, NCORES, OVERWRITE


@click.command(
    "charge-molecules",
    short_help="Generate partial charges for a set of molecules."
)
@MOL_DIR.parameter(
    required=True, help=MOL_DIR.kwargs["help"] + " Any number of sdf paths."
)
@YAML_OPTIONS.parameter(
    multiple=False, required=False, default=None,
    help=YAML_OPTIONS.kwargs["help"],
)
@OUTPUT_FILE_AND_EXT.parameter(
    help="The name of the SDF file the charged ligands should be written to."
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
def charge_molecules(
        molecules,
        yaml_settings,
        output,
        n_cores,
        overwrite_charges
):
    """
    Generate partial charges for the set of input molecules and write them to file.
    """
    from openfecli.utils import write
    from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges

    write("SMALL MOLECULE PARTIAL CHARGE GENERATOR")
    write("_________________________________________")
    write("")

    write("Parsing in Files: ")

    # INPUT
    write("\tGot input: ")

    small_molecules = MOL_DIR.get(molecules)
    write(
        "\t\tSmall Molecules: "
        + " ".join([str(sm) for sm in small_molecules])
    )

    yaml_options = YAML_OPTIONS.get(yaml_settings)
    partial_charge = yaml_options.partial_charge

    write("Using Options:")
    write("\tPartial Charge Generation: " + str(partial_charge.partial_charge_method))
    write("")

    charged_molecules = bulk_assign_partial_charges(
        molecules=small_molecules,
        overwrite=overwrite_charges,
        method=partial_charge.partial_charge_method,
        toolkit_backend=partial_charge.off_toolkit_backend,
        generate_n_conformers=partial_charge.number_of_conformers,
        nagl_model=partial_charge.nagl_model,
        processors=n_cores
    )

    write("\tDone")
    write("")

    # OUTPUT
    file, _ = OUTPUT_FILE_AND_EXT.get(output)
    write("Output:")
    write("\tSaving to: " + file.name)

    # default is write bytes
    file.mode = "w"
    with file.open() as output:
        for mol in charged_molecules:
            output.write(mol.to_sdf())


PLUGIN = OFECommandPlugin(
    command=charge_molecules, section="Miscellaneous", requires_ofe=(0, 3)
)
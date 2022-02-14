
import click
from openfecli import OFECommandPlugin
from openfecli.parameters import MOL, MAPPER

def allow_two_molecules(ctx, param, value):
    if len(value) != 2:
        raise click.BadParameter("Must specify --mol exactly twice.")


@click.command(
    "atommapping",
    short_help="Explore the alchemical mutations of a given mapping"
)
@MOL.parameter(multiple=True, callback=allow_two_molecules,
               help=MOL.kwargs['help'] + " Must be specified twice.")
@MAPPER.parameter(required=True)
def atommapping(mol, mapper):
    """
    This provides tools for looking at a specific atommapping.
    """
    mol1, mol2 = molecule
    # TODO: output style might depend on other parameters
    pass


def _get_output_format(output):
    pass


def atom_mapping_print_dict_main(mol1, mol2, mapper):
    mapping = mapper(mol1, mol2)
    print(mapping.mol1_to_mol2)


PLUGIN = OFECommandPlugin(
    command=atommapping,
    section="Setup",
    requires_ofe=(0, 0, 1),
)

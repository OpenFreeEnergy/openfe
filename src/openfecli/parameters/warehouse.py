import click
from plugcli.params import Option

WAREHOUSE = Option("--warehouse", type=click.BOOL, help="Use a warehouse", default=False)

import click
from openfecli import OFECommandPlugin

import json
import warnings

def unit_summary(unit_label, unit):
    qualname = unit['__qualname__']
    if qualname == "ProtocolUnitResult":
        yield f"Unit {unit_label} ran successfully."
    elif qualname == "ProtocolUnitFailure":
        yield f"Unit {unit_label} failed with an error:"
        yield f"{unit['exception'][0]}: {unit['exception'][1][0]}"
        yield f"{unit['traceback']}"
    else:
        warnings.warn(f"Unexpected result type '{aqualname}' from unit "
                      f"{unit_label}")

def result_summary(result_dict):
    import math
    # we were success or failurea
    estimate = result_dict['estimate']['magnitude']
    success = "FAILURE" if math.isnan(estimate) else "SUCCESS"
    yield f"This edge was a {success}."
    units = result_dict['unit_results']
    yield f"This edge consists of {len(units)} units."
    yield ""
    for unit_label, unit in units.items():
        yield from unit_summary(unit_label, unit)
        yield ""


@click.command(
    'inspect-result',
    short_help="Inspect result to show errors.",
)
@click.argument('result_json', type=click.Path(exists=True, readable=True))
def inspect_result(result_json):
    with open(result_json, mode='r') as f:
        result_dict = json.loads(f.read())

    for line in result_summary(result_dict):
        print(line)

PLUGIN = OFECommandPlugin(
    command=inspect_result,
    section="Quickrun Executor",
    requires_ofe=(0, 10)
)

if __name__ == "__main__":
    inspect_result()

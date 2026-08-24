# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/gufe
import json
import pathlib

from gufe.protocols import ProtocolResult
from gufe.protocols.protocoldag import ProtocolDAGResult
from gufe.tokenization import JSON_HANDLER


def convert_to_quickrun_output(
    result_edges: list[tuple[ProtocolResult, ProtocolDAGResult]],
    out_dir: pathlib.Path | str,
) -> None:
    # TODO: this is copied from quickrun, refactor to avoid duplication
    for prot_result, dagresult in result_edges:
        if dagresult.ok():
            estimate = prot_result.get_estimate()
            uncertainty = prot_result.get_uncertainty()
        else:
            estimate = uncertainty = None  # for output file

        out_dict = {
            "estimate": estimate,
            "uncertainty": uncertainty,
            "protocol_result": prot_result.to_dict(),
            "unit_results": {
                unit.key: unit.to_keyed_dict() for unit in dagresult.protocol_unit_results
            },
        }
        pathlib.Path(out_dir).mkdir(exist_ok=True)
        output = pathlib.Path(out_dir) / f"{dagresult.transformation_key}_results.json"
        if output.exists():
            raise FileExistsError("output file already exists!")
        with open(output, mode="w") as outf:
            json.dump(out_dict, outf, cls=JSON_HANDLER.encoder)

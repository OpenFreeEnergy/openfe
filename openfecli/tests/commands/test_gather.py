import glob
import os
import pathlib
from unittest import mock

import pooch
import pytest
from click.testing import CliRunner

from openfecli.commands.gather import (
    _get_column,
    _get_legs_from_result_jsons,
    _load_valid_result_json,
    format_estimate_uncertainty,
    gather,
)

from ..conftest import HAS_INTERNET
from ..utils import assert_click_success

POOCH_CACHE = pooch.os_cache("openfe")
ZENODO_RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.15042470",
    registry={
        "rbfe_results_serial_repeats.tar.gz": "md5:2355ecc80e03242a4c7fcbf20cb45487",
        "rbfe_results_parallel_repeats.tar.gz": "md5:ff7313e14eb6f2940c6ffd50f2192181",
    },
    retry_if_failed=5,
)
ZENODO_CMET_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.15200083",
    registry={"cmet_results.tar.gz": "md5:a4ca67a907f744c696b09660dc1eb8ec"},
    retry_if_failed=5,
)


@pytest.mark.parametrize(
    "est,unc,unc_prec,est_str,unc_str",
    [
        (12.432, 0.111, 2, "12.43", "0.11"),
        (0.9999, 0.01, 2, "1.000", "0.010"),
        (1234, 100, 2, "1230", "100"),
    ],
)
def test_format_estimate_uncertainty(est, unc, unc_prec, est_str, unc_str):
    assert format_estimate_uncertainty(est, unc, unc_prec) == (est_str, unc_str)


@pytest.mark.parametrize(
    "val, col",
    [
        (1.0, 1),
        (0.1, -1),
        (-0.0, 0),
        (0.0, 0),
        (0.2, -1),
        (0.9, -1),
        (0.011, -2),
        (9, 1),
        (10, 2),
        (15, 2),
    ],
)
def test_get_column(val, col):
    assert _get_column(val) == col


class TestResultLoading:
    @pytest.fixture
    def sim_result(self):
        result = {
            "estimate": {},
            "uncertainty": {},
            "protocol_result": {
                "data": {
                    "22940961": [
                        {
                            "name": "lig_ejm_31 to lig_ejm_42 repeat 0 generation 0",
                            "inputs": {
                                "stateA": {
                                    "components": {
                                        "ligand": {
                                            "atoms": [
                                                [1, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [17, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, False, 0, 0, {}],
                                                [8, 0, 0, False, 0, 0, {}],
                                                [7, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [7, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [7, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, False, 0, 0, {}],
                                                [8, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [17, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                            ],
                                            "bonds": [
                                                [0, 1, 1, 0, {}],
                                                [1, 6, 12, 0, {}],
                                                [1, 2, 12, 0, {}],
                                                [2, 3, 12, 0, {}],
                                                [2, 31, 1, 0, {}],
                                                [3, 4, 12, 0, {}],
                                                [3, 30, 1, 0, {}],
                                                [4, 5, 12, 0, {}],
                                                [4, 9, 1, 0, {}],
                                                [5, 6, 12, 0, {}],
                                                [5, 8, 1, 0, {}],
                                                [6, 7, 1, 0, {}],
                                                [9, 10, 2, 0, {}],
                                                [9, 11, 1, 0, {}],
                                                [11, 12, 1, 0, {}],
                                                [11, 13, 1, 0, {}],
                                                [13, 18, 12, 0, {}],
                                                [13, 14, 12, 0, {}],
                                                [14, 15, 12, 0, {}],
                                                [14, 29, 1, 0, {}],
                                                [15, 16, 12, 0, {}],
                                                [15, 28, 1, 0, {}],
                                                [16, 17, 12, 0, {}],
                                                [17, 18, 12, 0, {}],
                                                [17, 20, 1, 0, {}],
                                                [18, 19, 1, 0, {}],
                                                [20, 21, 1, 0, {}],
                                                [20, 22, 1, 0, {}],
                                                [22, 23, 2, 0, {}],
                                                [22, 24, 1, 0, {}],
                                                [24, 25, 1, 0, {}],
                                                [24, 26, 1, 0, {}],
                                                [24, 27, 1, 0, {}],
                                            ],
                                            "conformer": [
                                                "\u0093NUMPY\u0001\u0000v\u0000{'descr': '<f8', 'fortran_order': False, 'shape': (32, 3), }                                                         \n\u00f3\u008eSt$\u0017\u0013\u00c0W[\u00b1\u00bf\u00ec\u009e\u0006\u00c09\u00b4\u00c8v\u00be\u007f0\u00c05\u00ef8EGr\u0015\u00c0P\u008d\u0097n\u0012\u0083\r\u00c0b\u00a1\u00d64\u00ef80\u00c0\u0095e\u0088c]\u001c\u0013\u00c0\u0096!\u008euq\u00db\u0013\u00c0\u00a5,C\u001c\u00eb20\u00c0j\u00deq\u008a\u008e$\u0016\u00c0\u00b4\u00c8v\u00be\u009fZ\u0018\u00c0+\u00f6\u0097\u00dd\u0093\u00a7/\u00c0_\u0007\u00ce\u0019Q\u009a\u001b\u00c0d\u00cc]K\u00c8\u00c7\u0017\u00c0m\u00c5\u00fe\u00b2{\u00f2.\u00c0\u009e\u00ef\u00a7\u00c6K\u00f7\u001d\u00c0p\u00ce\u0088\u00d2\u00de\u00a0\u0012\u00c0\u00e7\u00fb\u00a9\u00f1\u00d2\r/\u00c0r\u008a\u008e\u00e4\u00f2\u00df\u001a\u00c0\u00e1\u000b\u0093\u00a9\u0082Q\f\u00c0\t\u001b\u009e^)\u00cb/\u00c0\u00f7\u00e4a\u00a1\u00d6\u00b4\u001c\u00c0\u0017\u00b7\u00d1\u0000\u00de\u0082\u0004\u00c0\u00e2X\u0017\u00b7\u00d1\u00e0/\u00c0)\u00cb\u0010\u00c7\u00baX\"\u00c0a2U0*\u00a9\u0011\u00c0V\u009f\u00ab\u00ad\u00d8_.\u00c0C\u001c\u00eb\u00e26\u00da\u001e\u00c0\u0083/L\u00a6\n\u0086\u001c\u00c0\u00de\u0093\u0087\u0085Z\u0013.\u00c0\u00d3\u00bc\u00e3\u0014\u001dI \u00c0\u00ec/\u00bb'\u000f\u00cb\u001c\u00c0h\"lxz\u00c5+\u00c0\u009f\u00cd\u00aa\u00cf\u00d5\u0096\u001f\u00c0\u0094\u0087\u0085Z\u00d3\u001c \u00c0t$\u0097\u00ff\u0090\u00fe/\u00c0r\u00f9\u000f\u00e9\u00b7\u00af\u001d\u00c0Gr\u00f9\u000f\u00e9w\u001f\u00c0Zd;\u00dfO\u00dd0\u00c0p_\u0007\u00ce\u0019q!\u00c0!\u00b0rh\u0091m\"\u00c0\u008c\u00dbh\u0000o\u00010\u00c0H\u00e1z\u0014\u00ae'#\u00c0\u008a\u00b0\u00e1\u00e9\u00952#\u00c0\u009e^)\u00cb\u0010\u00e7-\u00c0\u0085\u00ebQ\u00b8\u001e\u00c5$\u00c0I.\u00ff!\u00fdv%\u00c0M\u00f3\u008eStD.\u00c0\\\u008f\u00c2\u00f5(\u00bc$\u00c0\nh\"lx\u00fa&\u00c0\u00ff\u00b2{\u00f2\u00b0@0\u00c0Nb\u0010X9\u0014#\u00c0?5^\u00baIL&\u00c0i\u0000o\u0081\u0004E1\u00c0=\u009bU\u009f\u00abm!\u00c0\u008euq\u001b\r\u0000$\u00c0[\u00b1\u00bf\u00ec\u009e,1\u00c0%\u0006\u0081\u0095C+ \u00c0\u0088\u0085Z\u00d3\u00bcc#\u00c0\u0003x\u000b$(\u00fe1\u00c0\u0012\u00a5\u00bd\u00c1\u0017&#\u00c0\u00cb\u00a1E\u00b6\u00f3\u00dd'\u00c0io\u00f0\u0085\u00c9t2\u00c0\u00ea\u0095\u00b2\fq\u00cc$\u00c0\u0018\u0095\u00d4\th\u0002)\u00c0vO\u001e\u0016j\u008d2\u00c0\u00bb\u00b8\u008d\u0006\u00f0V!\u00c0\u00c1\u00ca\u00a1E\u00b6\u0013(\u00c0\u00deq\u008a\u008e\u00e4r3\u00c0\u00c5 \u00b0rhQ\u001e\u00c0\u00c5\u00fe\u00b2{\u00f2\u00f0&\u00c0\u00d1\u0091\\\u00feCz3\u00c0B>\u00e8\u00d9\u00ac\u001a\"\u00c0\u0080\u00b7@\u0082\u00e2\u00e7)\u00c0\b=\u009bU\u009f\u009b4\u00c0r\u00f9\u000f\u00e9\u00b7O\"\u00c0\u00de\u0093\u0087\u0085Z\u00f3+\u00c0o\u0012\u0083\u00c0\u00caA4\u00c0\u008a\u008e\u00e4\u00f2\u001f\u0092 \u00c0\u00aa`TR'\u00e0)\u00c0n4\u0080\u00b7@b5\u00c0\u00e8j+\u00f6\u0097\u00fd#\u00c0\u00b3{\u00f2\u00b0PK)\u00c0:#J{\u0083\u000f5\u00c0\u008c\u00dbh\u0000o!&\u00c0\u00d9=yX\u00a8\u0015&\u00c0\u0012\u0014?\u00c6\u00dc\u00b5,\u00c0\u00a5\u00bd\u00c1\u0017&S#\u00c0io\u00f0\u0085\u00c9\u0014\"\u00c0\u00ab\u00cf\u00d5V\u00ec\u000f,\u00c0\u00e0\u009c\u0011\u00a5\u00bd\u0001\u0013\u00c0\u00ad\u00fa\\m\u00c5~\u001e\u00c0L7\u0089A`\u00a5/\u00c0\u0002\u009a\b\u001b\u009e\u00de\r\u00c0\u00c8\u0098\u00bb\u0096\u0090O\u0014\u00c0N\u00d1\u0091\\\u00fes0\u00c0",
                                                {},
                                            ],
                                            "molprops": {"ofe-name": "lig_ejm_31"},
                                            "__qualname__": "SmallMoleculeComponent",
                                            "__module__": "gufe.components.smallmoleculecomponent",
                                        },
                                        "solvent": {
                                            "smiles": "O",
                                            "positive_ion": "Na+",
                                            "negative_ion": "Cl-",
                                            "ion_concentration": "0.15 molar",
                                            "neutralize": True,
                                            "__qualname__": "SolventComponent",
                                            "__module__": "gufe.components.solventcomponent",
                                        },
                                    },
                                    "name": "lig_ejm_31_solvent",
                                    "__qualname__": "ChemicalSystem",
                                    "__module__": "gufe.chemicalsystem",
                                },
                                "stateB": {
                                    "components": {
                                        "ligand": {
                                            "atoms": [
                                                [1, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [17, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, False, 0, 0, {}],
                                                [8, 0, 0, False, 0, 0, {}],
                                                [7, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [7, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [6, 0, 0, True, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [7, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, False, 0, 0, {}],
                                                [8, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [6, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                                [17, 0, 0, False, 0, 0, {}],
                                                [1, 0, 0, False, 0, 0, {}],
                                            ],
                                            "bonds": [
                                                [0, 1, 1, 0, {}],
                                                [1, 6, 12, 0, {}],
                                                [1, 2, 12, 0, {}],
                                                [2, 3, 12, 0, {}],
                                                [2, 34, 1, 0, {}],
                                                [3, 4, 12, 0, {}],
                                                [3, 33, 1, 0, {}],
                                                [4, 5, 12, 0, {}],
                                                [4, 9, 1, 0, {}],
                                                [5, 6, 12, 0, {}],
                                                [5, 8, 1, 0, {}],
                                                [6, 7, 1, 0, {}],
                                                [9, 10, 2, 0, {}],
                                                [9, 11, 1, 0, {}],
                                                [11, 12, 1, 0, {}],
                                                [11, 13, 1, 0, {}],
                                                [13, 18, 12, 0, {}],
                                                [13, 14, 12, 0, {}],
                                                [14, 15, 12, 0, {}],
                                                [14, 32, 1, 0, {}],
                                                [15, 16, 12, 0, {}],
                                                [15, 31, 1, 0, {}],
                                                [16, 17, 12, 0, {}],
                                                [17, 18, 12, 0, {}],
                                                [17, 20, 1, 0, {}],
                                                [18, 19, 1, 0, {}],
                                                [20, 21, 1, 0, {}],
                                                [20, 22, 1, 0, {}],
                                                [22, 23, 2, 0, {}],
                                                [22, 24, 1, 0, {}],
                                                [24, 25, 1, 0, {}],
                                                [24, 26, 1, 0, {}],
                                                [24, 27, 1, 0, {}],
                                                [27, 28, 1, 0, {}],
                                                [27, 29, 1, 0, {}],
                                                [27, 30, 1, 0, {}],
                                            ],
                                            "conformer": [
                                                "\u0093NUMPY\u0001\u0000v\u0000{'descr': '<f8', 'fortran_order': False, 'shape': (35, 3), }                                                         \n\u00ac\u00ad\u00d8_v\u000f\u0013\u00c0\u009a\b\u001b\u009e^\u00a9\u0006\u00c0\u0019\u0004V\u000e-\u00820\u00c0\u00b0\u0003\u00e7\u008c(m\u0015\u00c0]\u00dcF\u0003x\u008b\r\u00c0C\u001c\u00eb\u00e26:0\u00c0io\u00f0\u0085\u00c9\u0014\u0013\u00c0\u00c5\u008f1w-\u00e1\u0013\u00c03333330\u00c0\u0001\u00de\u0002\t\u008a\u001f\u0016\u00c0\u00fee\u00f7\u00e4aa\u0018\u00c0a2U0*\u00a9/\u00c0\u00f8\u00c2d\u00aa`\u0094\u001b\u00c0\u0004\u00e7\u008c(\u00ed\u00cd\u0017\u00c0\u00a85\u00cd;N\u00f1.\u00c0\u00db\u00f9~j\u00bc\u00f4\u001d\u00c0e\u00aa`TR\u00a7\u0012\u00c0\u0091~\u00fb:p\u000e/\u00c0\u00cc\u007fH\u00bf}\u00dd\u001a\u00c0>\u00e8\u00d9\u00ac\u00fa\\\f\u00c0\u00aeG\u00e1z\u0014\u00ce/\u00c0\u0001M\u0084\rO\u00af\u001c\u00c0\u00d5x\u00e9&1\u0088\u0004\u00c0M\u00f3\u008eSt\u00e4/\u00c0\u009c3\u00a2\u00b47X\"\u00c0\u0088\u00f4\u00db\u00d7\u0081\u00b3\u0011\u00c0U\u00c1\u00a8\u00a4N`.\u00c0q\u001b\r\u00e0-\u00d0\u001e\u00c0\u00af%\u00e4\u0083\u009e\u008d\u001c\u00c0\u0002+\u0087\u0016\u00d9\u000e.\u00c0-!\u001f\u00f4l6 \u00c0\u009bU\u009f\u00ab\u00ad\u00d8\u001c\u00c0xz\u00a5,C\u00bc+\u00c0T\u00e3\u00a5\u009b\u00c4\u00a0\u001f\u00c0\u0099\u00bb\u0096\u0090\u000f\u001a \u00c09\u00b4\u00c8v\u00be\u00ff/\u00c0:\u0092\u00cb\u007fH\u00bf\u001d\u00c0:#J{\u0083o\u001f\u00c0\u001f\u0085\u00ebQ\u00b8\u00de0\u00c0\u00f8\u00c2d\u00aa`t!\u00c0\t\u001b\u009e^)k\"\u00c0\u00fee\u00f7\u00e4a\u00010\u00c0b\u0010X9\u00b4(#\u00c0\u0006\u0081\u0095C\u008b,#\u00c0h\"lxz\u00e5-\u00c0\u0016\u00fb\u00cb\u00ee\u00c9\u00c3$\u00c0\u00c1\u00ca\u00a1E\u00b6s%\u00c0\u00cd;N\u00d1\u0091<.\u00c0\u0093\u00a9\u0082QI\u00bd$\u00c0xz\u00a5,C\u00fc&\u00c0\u00ec/\u00bb'\u000f;0\u00c0\u00b6\u0084|\u00d0\u00b3\u0019#\u00c0\u0002+\u0087\u0016\u00d9N&\u00c0\u008c\u00dbh\u0000oA1\u00c0\u00fc\u0018s\u00d7\u0012r!\u00c0R'\u00a0\u0089\u00b0\u0001$\u00c0A\u00f1c\u00cc]+1\u00c04\u00116<\u00bd2 \u00c0\u0010\u00e9\u00b7\u00af\u0003g#\u00c0 c\u00eeZB\u00fe1\u00c0\u00b2\u009d\u00ef\u00a7\u00c6+#\u00c0\u00b8\u00af\u0003\u00e7\u008c\u00e8'\u00c0V\u000e-\u00b2\u009do2\u00c00\u00bb'\u000f\u000b\u00d5$\u00c0\u00e8\u00d9\u00ac\u00fa\\\r)\u00c0+\u00f6\u0097\u00dd\u0093\u00872\u00c0\u00ac\u001cZd;_!\u00c0\u00b8\u00af\u0003\u00e7\u008c((\u00c0X\u00ca2\u00c4\u00b1n3\u00c0\u00deq\u008a\u008e\u00e4r\u001e\u00c0\u00a0\u0089\u00b0\u00e1\u00e9\u00f5&\u00c0\"\u00fd\u00f6u\u00e0|3\u00c0yX\u00a85\u00cd\u001b\"\u00c0(\u000f\u000b\u00b5\u00a6\u0019*\u00c0\u00e3\u00a5\u009b\u00c4 \u00904\u00c01\b\u00ac\u001cZ\u00e4\"\u00c0Y\u00868\u00d6\u00c5\u00ed(\u00c0\u00a2E\u00b6\u00f3\u00fdd5\u00c0\u00d5\u00e7j+\u00f6\u00b7#\u00c0\u008cJ\u00ea\u00044q+\u00c0\u00e0\u00be\u000e\u009c3B4\u00c0\u00b3{\u00f2\u00b0Pk\u001f\u00c0\u00bb'\u000f\u000b\u00b5\u00c6+\u00c0\u00b6\u00f3\u00fd\u00d4x\t5\u00c0vq\u001b\r\u00e0M \u00c0\u008c\u00dbh\u0000o!-\u00c0\u00bc\u0096\u0090\u000fz\u00d65\u00c0_\u0098L\u0015\u008c\u00ca\u001d\u00c0\u00d1\u0091\\\u00feC\u00fa,\u00c0a\u00c3\u00d3+e94\u00c0K\u00ea\u00044\u00116\u001c\u00c0\u00a2\u00b47\u00f8\u00c2\u0084*\u00c0[\u00b1\u00bf\u00ec\u009el5\u00c0X9\u00b4\u00c8v\u001e&\u00c0p_\u0007\u00ce\u0019\u0011&\u00c0\u00ce\u0088\u00d2\u00de\u00e0\u00ab,\u00c0\u00d5\u00e7j+\u00f6W#\u00c0^\u00baI\f\u0002\u000b\"\u00c08gDio\u0010,\u00c0\u00c6m4\u0080\u00b7\u0000\u0013\u00c0\\ A\u00f1c\u008c\u001e\u00c0\u009e^)\u00cb\u0010\u00a7/\u00c0\u00e7\u00fb\u00a9\u00f1\u00d2\u00cd\r\u00c0\u009d\u0011\u00a5\u00bd\u00c1W\u0014\u00c0\u00f7\u00e4a\u00a1\u00d6t0\u00c0",
                                                {},
                                            ],
                                            "molprops": {"ofe-name": "lig_ejm_42"},
                                            "__qualname__": "SmallMoleculeComponent",
                                            "__module__": "gufe.components.smallmoleculecomponent",
                                        },
                                        "solvent": {
                                            "smiles": "O",
                                            "positive_ion": "Na+",
                                            "negative_ion": "Cl-",
                                            "ion_concentration": "0.15 molar",
                                            "neutralize": True,
                                            "__qualname__": "SolventComponent",
                                            "__module__": "gufe.components.solventcomponent",
                                        },
                                    },
                                    "name": "lig_ejm_42_solvent",
                                    "__qualname__": "ChemicalSystem",
                                    "__module__": "gufe.chemicalsystem",
                                },
                            },
                        }
                    ]
                }
            },
            "unit_results": {
                "ProtocolUnitResult-e85": {
                    "name": "lig_ejm_31 to lig_ejm_42 repeat 0 generation 0"
                },
                "ProtocolUnitFailure-4c9": {
                    "name": "lig_ejm_31 to lig_ejm_42 repeat 0 generation 0",
                    "exception": ["Simulation_NanError"],
                },
            },
        }
        yield result

    def test_minimal_valid_results(self, capsys, sim_result):
        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == ((("lig_ejm_31", "lig_ejm_42"), "solvent"), sim_result)
            assert captured.err == ""

    def test_skip_missing_unit_result(self, capsys, sim_result):
        sim_result["unit_results"] = {}

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == ((("lig_ejm_31", "lig_ejm_42"), "solvent"), None)
            assert "No 'unit_results' found" in captured.err

    def test_skip_missing_estimate(self, capsys, sim_result):
        sim_result["estimate"] = None

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == ((("lig_ejm_31", "lig_ejm_42"), "solvent"), None)
            assert "No 'estimate' found" in captured.err

    def test_skip_missing_uncertainty(self, capsys, sim_result):
        sim_result["uncertainty"] = None

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == ((("lig_ejm_31", "lig_ejm_42"), "solvent"), None)
            assert "No 'uncertainty' found" in captured.err

    def test_skip_all_failed_runs(self, capsys, sim_result):
        del sim_result["unit_results"]["ProtocolUnitResult-e85"]
        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == ((("lig_ejm_31", "lig_ejm_42"), "solvent"), None)
            assert "Exception found in all" in captured.err

    def test_missing_pr_data(self, capsys, sim_result):
        sim_result["protocol_result"]["data"] = {}
        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == (None, None)
            assert "Missing ligand names and/or simulation type. Skipping" in captured.err

    def test_get_legs_from_result_jsons(self, capsys, sim_result):
        """Test that exceptions are handled correctly at the _get_legs_from_results_json level."""
        sim_result["protocol_result"]["data"] = {}

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _get_legs_from_result_jsons(result_fns=[""], report="dg")
            captured = capsys.readouterr()
            assert result == {}
            assert "Missing ligand names and/or simulation type. Skipping" in captured.err


def test_no_results_found():
    runner = CliRunner()
    cli_result = runner.invoke(gather, "not_a_file.txt")
    assert cli_result.exit_code == 1
    assert "No results JSON files found" in str(cli_result.stderr)


_RBFE_EXPECTED_DG = b"""
ligand	DG(MLE) (kcal/mol)	uncertainty (kcal/mol)
lig_ejm_31	-0.09	0.05
lig_ejm_42	0.7	0.1
lig_ejm_46	-0.98	0.05
lig_ejm_47	-0.1	0.1
lig_ejm_48	0.53	0.09
lig_ejm_50	0.91	0.06
lig_ejm_43	2.0	0.2
lig_jmc_23	-0.68	0.09
lig_jmc_27	-1.1	0.1
lig_jmc_28	-1.25	0.08
"""

_RBFE_EXPECTED_DDG = b"""
ligand_i	ligand_j	DDG(i->j) (kcal/mol)	uncertainty (kcal/mol)
lig_ejm_31	lig_ejm_42	0.8	0.1
lig_ejm_31	lig_ejm_46	-0.89	0.06
lig_ejm_31	lig_ejm_47	0.0	0.1
lig_ejm_31	lig_ejm_48	0.61	0.09
lig_ejm_31	lig_ejm_50	1.00	0.04
lig_ejm_42	lig_ejm_43	1.4	0.2
lig_ejm_46	lig_jmc_23	0.29	0.09
lig_ejm_46	lig_jmc_27	-0.1	0.1
lig_ejm_46	lig_jmc_28	-0.27	0.06
"""

_RBFE_EXPECTED_RAW = b"""\
leg	ligand_i	ligand_j	DG(i->j) (kcal/mol)	MBAR uncertainty (kcal/mol)
complex	lig_ejm_31	lig_ejm_42	-14.9	0.8
complex	lig_ejm_31	lig_ejm_42	-14.8	0.8
complex	lig_ejm_31	lig_ejm_42	-15.1	0.8
solvent	lig_ejm_31	lig_ejm_42	-15.7	0.8
solvent	lig_ejm_31	lig_ejm_42	-15.7	0.8
solvent	lig_ejm_31	lig_ejm_42	-15.7	0.8
complex	lig_ejm_31	lig_ejm_46	-40.7	0.8
complex	lig_ejm_31	lig_ejm_46	-40.7	0.8
complex	lig_ejm_31	lig_ejm_46	-40.8	0.8
solvent	lig_ejm_31	lig_ejm_46	-39.8	0.8
solvent	lig_ejm_31	lig_ejm_46	-39.9	0.8
solvent	lig_ejm_31	lig_ejm_46	-39.8	0.8
complex	lig_ejm_31	lig_ejm_47	-27.8	0.8
complex	lig_ejm_31	lig_ejm_47	-28.0	0.8
complex	lig_ejm_31	lig_ejm_47	-27.7	0.8
solvent	lig_ejm_31	lig_ejm_47	-27.8	0.8
solvent	lig_ejm_31	lig_ejm_47	-27.8	0.8
solvent	lig_ejm_31	lig_ejm_47	-27.9	0.8
complex	lig_ejm_31	lig_ejm_48	-16.2	0.8
complex	lig_ejm_31	lig_ejm_48	-16.2	0.8
complex	lig_ejm_31	lig_ejm_48	-16.0	0.8
solvent	lig_ejm_31	lig_ejm_48	-16.8	0.8
solvent	lig_ejm_31	lig_ejm_48	-16.7	0.8
solvent	lig_ejm_31	lig_ejm_48	-16.8	0.8
complex	lig_ejm_31	lig_ejm_50	-57.3	0.8
complex	lig_ejm_31	lig_ejm_50	-57.3	0.8
complex	lig_ejm_31	lig_ejm_50	-57.4	0.8
solvent	lig_ejm_31	lig_ejm_50	-58.3	0.8
solvent	lig_ejm_31	lig_ejm_50	-58.4	0.8
solvent	lig_ejm_31	lig_ejm_50	-58.3	0.8
complex	lig_ejm_42	lig_ejm_43	-19.0	0.8
complex	lig_ejm_42	lig_ejm_43	-18.7	0.8
complex	lig_ejm_42	lig_ejm_43	-19.0	0.8
solvent	lig_ejm_42	lig_ejm_43	-20.3	0.8
solvent	lig_ejm_42	lig_ejm_43	-20.3	0.8
solvent	lig_ejm_42	lig_ejm_43	-20.3	0.8
complex	lig_ejm_46	lig_jmc_23	17.3	0.8
complex	lig_ejm_46	lig_jmc_23	17.4	0.8
complex	lig_ejm_46	lig_jmc_23	17.5	0.8
solvent	lig_ejm_46	lig_jmc_23	17.2	0.8
solvent	lig_ejm_46	lig_jmc_23	17.1	0.8
solvent	lig_ejm_46	lig_jmc_23	17.1	0.8
complex	lig_ejm_46	lig_jmc_27	15.9	0.8
complex	lig_ejm_46	lig_jmc_27	15.8	0.8
complex	lig_ejm_46	lig_jmc_27	15.7	0.8
solvent	lig_ejm_46	lig_jmc_27	16.0	0.8
solvent	lig_ejm_46	lig_jmc_27	15.9	0.8
solvent	lig_ejm_46	lig_jmc_27	15.9	0.8
complex	lig_ejm_46	lig_jmc_28	23.1	0.8
complex	lig_ejm_46	lig_jmc_28	23.2	0.8
complex	lig_ejm_46	lig_jmc_28	23.1	0.8
solvent	lig_ejm_46	lig_jmc_28	23.5	0.8
solvent	lig_ejm_46	lig_jmc_28	23.3	0.8
solvent	lig_ejm_46	lig_jmc_28	23.4	0.8
"""


@pytest.fixture
def rbfe_result_dir() -> pathlib.Path:
    def _rbfe_result_dir(dataset) -> str:
        ZENODO_RBFE_DATA.fetch(f"{dataset}.tar.gz", processor=pooch.Untar())
        cache_dir = pathlib.Path(POOCH_CACHE) / f"{dataset}.tar.gz.untar/{dataset}/"
        return cache_dir

    return _rbfe_result_dir


@pytest.fixture
def cmet_result_dir() -> pathlib.Path:
    ZENODO_CMET_DATA.fetch("cmet_results.tar.gz", processor=pooch.Untar())
    result_dir = pathlib.Path(POOCH_CACHE) / "cmet_results.tar.gz.untar/cmet_results/"

    return result_dir


class TestGatherCMET:
    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_cmet_full_results(self, cmet_result_dir, report, file_regression):
        results = [str(cmet_result_dir / f"results_{i}") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    # TODO: add --allow-partial behavior checks
    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_cmet_missing_complex_leg(self, cmet_result_dir, report, file_regression):
        """Missing one complex replicate from one leg."""
        results = [
            str(cmet_result_dir / d) for d in ["results_0_partial", "results_1", "results_2"]
        ]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_cmet_missing_edge(self, cmet_result_dir, report, file_regression):
        results = [str(cmet_result_dir / f"results_{i}_remove_edge") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])
        file_regression.check(cli_result.stdout, extension=".tsv")

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["ddg", "raw"])
    def test_cmet_failed_edge(self, cmet_result_dir, report, file_regression):
        results = [str(cmet_result_dir / f"results_{i}_failed_edge") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("allow_partial", [True, False])
    def test_cmet_too_few_edges_error(self, cmet_result_dir, allow_partial):
        results = [str(cmet_result_dir / f"results_{i}_failed_edge") for i in range(3)]
        args = ["--report", "dg"]
        runner = CliRunner()
        if allow_partial:
            args += ["--allow-partial"]

        cli_result = runner.invoke(gather, results + args + ["--tsv"])
        assert cli_result.exit_code == 1
        assert "The results network has 1 edge(s), but 3 or more edges are required" in str(
            cli_result.stderr
        )

    @pytest.mark.parametrize("report", ["dg", "ddg"])
    def test_cmet_missing_all_complex_legs_fail(self, cmet_result_dir, report, file_regression):
        """Missing one complex replicate from one leg."""
        results = glob.glob(f"{cmet_result_dir}/results_*/*solvent*", recursive=True)
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["-o", "-"])

        cli_result.exit_code == 1
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["ddg"])
    def test_cmet_missing_all_complex_legs_allow_partial(self, cmet_result_dir, report, file_regression):  # fmt: skip
        """Missing one complex replicate from one leg."""
        results = glob.glob(f"{cmet_result_dir}/results_*/*solvent*", recursive=True)
        args = ["--report", report, "--allow-partial"]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_pretty_print(self, cmet_result_dir, report, file_regression):
        results = [str(cmet_result_dir / f"results_{i}") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args)
        assert_click_success(cli_result)
        # TODO: figure out how to mock terminal size, since it affects the table wrapping
        # file_regression.check(cli_result.stdout, extension='.txt')

    def test_write_to_file(self, cmet_result_dir):
        runner = CliRunner()
        with runner.isolated_filesystem():
            results = [str(cmet_result_dir / f"results_{i}") for i in range(3)]
            fname = "output.tsv"
            args = ["--report", "raw", "-o", fname]
            cli_result = runner.invoke(gather, results + args)
            assert "writing raw output to 'output.tsv'" in cli_result.stdout
            assert pathlib.Path(fname).is_file()


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet seems to be unavailable and test data is not cached locally.",
)
@pytest.mark.parametrize("dataset", ["rbfe_results_serial_repeats", "rbfe_results_parallel_repeats"])  # fmt: skip
@pytest.mark.parametrize("report", ["", "dg", "ddg", "raw"])
@pytest.mark.parametrize("input_mode", ["directory", "filepaths"])
def test_rbfe_gather(rbfe_result_dir, dataset, report, input_mode):
    expected = {
        "": _RBFE_EXPECTED_DG,
        "dg": _RBFE_EXPECTED_DG,
        "ddg": _RBFE_EXPECTED_DDG,
        "raw": _RBFE_EXPECTED_RAW,
    }[report]
    runner = CliRunner()

    if report:
        args = ["--report", report]
    else:
        args = []

    results = rbfe_result_dir(dataset)
    if input_mode == "directory":
        results = [str(results)]
    elif input_mode == "filepaths":
        results = glob.glob(f"{results}/*", recursive=True)
        assert len(results) > 1  # sanity check to make sure we're passing in multiple paths

    cli_result = runner.invoke(gather, results + args + ["--tsv"])

    assert_click_success(cli_result)

    actual_lines = set(cli_result.stdout_bytes.split(b"\n"))
    assert set(expected.split(b"\n")) == actual_lines


def test_rbfe_gather_single_repeats_dg_error(rbfe_result_dir):
    """A single repeat is insufficient for a dg calculation - should fail cleanly."""

    runner = CliRunner()
    results = rbfe_result_dir("rbfe_results_parallel_repeats")
    args = ["report", "dg"]
    cli_result = runner.invoke(gather, [f"{results}/replicate_0"] + args + ["--tsv"])
    assert cli_result.exit_code == 1


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet seems to be unavailable and test data is not cached locally.",
)
class TestRBFEGatherFailedEdges:
    @pytest.fixture()
    def results_paths_serial_missing_legs(self, rbfe_result_dir) -> str:
        """Example output data, with replicates run in serial and two missing results JSONs."""
        result_dir = rbfe_result_dir("rbfe_results_serial_repeats")
        results = glob.glob(f"{result_dir}/*", recursive=True)

        files_to_skip = [
            "rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json",
            "rbfe_lig_ejm_46_solvent_lig_jmc_28_solvent.json",
        ]

        results_filtered = [f for f in results if os.path.basename(f) not in files_to_skip]

        return results_filtered

    def test_missing_leg_error(self, results_paths_serial_missing_legs: str):
        runner = CliRunner()
        result = runner.invoke(gather, results_paths_serial_missing_legs + ["--report", "dg"])

        assert result.exit_code == 1
        assert "Some edge(s) are missing runs" in str(result.stderr)
        assert "lig_ejm_31\tlig_ejm_42\tsolvent" in str(result.stderr)
        assert "lig_ejm_46\tlig_jmc_28\tcomplex" in str(result.stderr)
        assert "using the --allow-partial flag" in str(result.stderr)

    def test_missing_leg_allow_partial_disconnected(self, results_paths_serial_missing_legs: str):
        runner = CliRunner()
        with pytest.warns():
            args = ["--report", "dg", "--allow-partial"]
            result = runner.invoke(gather, results_paths_serial_missing_legs + args + ["--tsv"])
            assert result.exit_code == 1
            assert "The results network is disconnected" in str(result.stderr)

    def test_allow_partial_msg_not_printed(self, results_paths_serial_missing_legs: str):
        # we *dont* want the suggestion to use --allow-partial if the user already used it!
        runner = CliRunner()
        args = ["--report", "ddg", "--allow-partial"]
        result = runner.invoke(gather, results_paths_serial_missing_legs + args + ["--tsv"])
        assert_click_success(result)
        assert "--allow-partial" not in result.output

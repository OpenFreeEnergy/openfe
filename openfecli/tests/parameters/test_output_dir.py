import os
import pathlib
from importlib import resources

from openfecli.parameters.output_dir import get_dir


def test_get_output_dir():
    with resources.files("openfe.tests") as dir_path:
        out_dir = get_dir(dir_path, None)

        assert isinstance(out_dir, pathlib.Path)
        assert out_dir.parent.exists()

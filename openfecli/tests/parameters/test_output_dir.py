import os
import pytest
import importlib
from importlib import resources

from openfecli.parameters.output_dir import get_dir

   
def test_get_file_and_extension():
    with importlib.resources.path("openfe.tests", "__init__.py") as file_path:
        dir_path = os.path.dirname(file_path)
        outfile = get_dir(dir_path, None)




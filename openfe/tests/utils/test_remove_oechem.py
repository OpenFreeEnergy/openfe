# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from openfe.utils import without_oechem_backend
from openff.toolkit import GLOBAL_TOOLKIT_REGISTRY, OpenEyeToolkitWrapper


def test_remove_oechem():
    original_tks = GLOBAL_TOOLKIT_REGISTRY.registered_toolkits
    original_n_tks = len(GLOBAL_TOOLKIT_REGISTRY.registered_toolkits)

    with without_oechem_backend():
        for tk in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits:
            assert not isinstance(tk, OpenEyeToolkitWrapper)
    assert len(GLOBAL_TOOLKIT_REGISTRY.registered_toolkits) == original_n_tks
    for ref_tk, tk in zip(original_tks, GLOBAL_TOOLKIT_REGISTRY.registered_toolkits):
        assert isinstance(tk, type(ref_tk))

# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utilities for assigning partial charges to ChemicalComponents.
"""
import copy
from typing import Union, Optional, Literal
import numpy as np
from openff.units import unit
from openff.toolkit import Molecule as OFFMol
from openff.toolkit.utils.base_wrapper import ToolkitWrapper
from openff.toolkit.utils.toolkits import (
    AmberToolsToolkitWrapper,
    OpenEyeToolkitWrapper,
    RDKitToolkitWrapper
)
from openff.toolkit.utils.toolkit_registry import ToolkitRegistry


try:
    from openff.toolkit.utils.toolkit_registry import (
        toolkit_registry_manager,
    )
except ImportError:
    # toolkit_registry_manager was made non private in 0.14.4
    from openff.toolkit.utils.toolkit_registry import (
        _toolkit_registry_manager as toolkit_registry_manager
    )


try:
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
    from openff.nagl_models import get_models_by_type, validate_nagl_model_path
except ImportError:
    HAS_NAGL = False
else:
    HAS_NAGL = True


try:
    from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
except ImportError:
    HAS_ESPALOMA = False
else:
    HAS_ESPALOMA = True


# Dictionary of lists for the various backend options we allow.
# Note: can't create the classes ahead of time in case we end
# up with a case where the tool is not available, e.g. if OpenEye tk
# is not installed.
BACKEND_OPTIONS: dict[str, list[ToolkitWrapper]] = {
    "ambertools": [RDKitToolkitWrapper, AmberToolsToolkitWrapper],
    "openeye": [OpenEyeToolkitWrapper],
    "rdkit": [RDKitToolkitWrapper],
}


def assign_offmol_espaloma_charges(
    offmol: OFFMol,
    toolkit_registry: ToolkitRegistry
) -> None:
    """
    Assign Espaloma charges using the OpenFF toolkit.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      OpenFF molecule to assign NAGL partial charges for.
    toolkit_registry : ToolkitRegistry
      Toolkit registry to use for assigning partial charges.
      This strictly limits available toolkit wrappers by
      overwriting the global registry during the partial charge
      assignment stage.
    """
    if not HAS_ESPALOMA:
        errmsg = ("The Espaloma ToolkiWrapper is not available, "
                  "please install espaloma_charge")
        raise ImportError(errmsg)

    # make a copy to remove conformers as espaloma enforces
    # a 0 conformer check
    offmol_copy = copy.deepcopy(offmol)
    offmol_copy._conformers = None

    # We are being overly cautious by applying the manager here
    # this is to avoid issues like:
    # https://github.com/openforcefield/openff-nagl/issues/69
    with toolkit_registry_manager(toolkit_registry):
        offmol_copy.assign_partial_charges(
            partial_charge_method='espaloma-am1bcc',
            toolkit_registry=EspalomaChargeToolkitWrapper(),
        )

    # Copy back charges into the original offmol object
    offmol.partial_charges = offmol_copy.partial_charges


def assign_offmol_nagl_charges(
    offmol: OFFMol,
    toolkit_registry: ToolkitRegistry,
    nagl_model: Optional[str] = None,
) -> None:
    """
    Assign NAGL charges using the OpenFF toolkit.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      OpenFF molecule to assign NAGL partial charges for.
    toolkit_registry : ToolkitRegistry
      Toolkit registry to use for assigning partial charges.
      This strictly limits available toolkit wrappers by
      overwriting the global registry during the partial charge
      assignment stage.
    nagl_model : Optional[str]
      The NAGL model to use when assigning partial charges.
      If ``None``, will fetch the latest production "am1bcc" model.
    """
    if not HAS_NAGL:
        errmsg = ("The NAGL toolkit is not available, you may "
                  "be using an older version of the OpenFF "
                  "toolkit - you need v0.14.4 or above")
        raise ImportError(errmsg)

    if nagl_model is None:
        prod_models = get_model_by_type(
            model_type='am1bcc', production_only=True
        )
        # Currently there are no production models so expect an IndexError
        try:
            nagl_model = prod_models[-1]
        except IndexError:
            errmsg = ("No production am1bcc NAGL models are current available "
                      "please manually select a candidate release model")
            raise ValueError(errmsg)

    model_path = validate_nalg_model_path(nagl_model)

    # We are being overly cautious by applying the manager here
    # this is to avoid issues like:
    # https://github.com/openforcefield/openff-nagl/issues/69
    with toolkit_registry_manager(toolkit_registry):
        offmol.assign_partial_charges(
            partial_charge_method=model_path,
            toolkit_registry=NAGLToolkitWrapper(),
        )


def assign_offmol_am1bcc_charges(
    offmol: OFFMol,
    partial_charge_method: Literal['am1bcc', 'am1bccelf10'],
    toolkit_registry: ToolkitRegistry,
) -> None:
    """
    Assign AM1BCC charges using the OpenFF toolkit.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      OpenFF Molecule to assign AM1BCC charges for.
      Must already have a conformer.
    partial_charge_method : Literal['am1bcc', 'am1bccelf10']
      The partial charge method to employ.
      Options include `am1bcc`, `am1bccelf10`.
    toolkit_registry : ToolkitRegistry
      Toolkit registry to use for assigning partial charges.
      This strictly limits available toolkit wrappers by
      overwriting the global registry during the partial charge
      assignment stage.

    Raises
    ------
    ValueError
      If the ``offmol`` does not have any conformers.
    """
    if offmol.n_conformers == 0:
        errmsg = "method expects at least one conformer"
        raise ValueError(errmsg)

    # We are being overly cautious by both passing the
    # registry and applying the manager here - this is
    # to avoid issues like:
    # https://github.com/openforcefield/openff-nagl/issues/69
    with toolkit_registry_manager(toolkit_registry):
        offmol.assign_partial_charges(
            partial_charge_method=partial_charge_method,
            use_conformers=offmol.conformers,
            toolkit_registry=toolkit_registry
        )


def _generate_offmol_conformers(
    offmol: OFFMol,
    max_conf: int,
    toolkit_registry: ToolkitRegistry,
    generate_n_conformers: Optional[int],
) -> None:
    """
    Helper method for OFF Molecule conformer generation in charge assignment.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      OpenFF Molecule to generate conformers for
    max_conf : int
      The maximum number of conformers supported by requested charge method.
    toolkit_registry : ToolkitRegistry
      Toolkit registry to use for generating conformers.
      This strictly limits available toolkit wrappers by
      overwriting the global registry during the conformer generation step.
    generate_n_conformers : Optional[int]
      The number of conformers to generate. If ``None``, the existing
      conformers are retained & used.

    Raises
    ------
    ValueError
      If the ``generate_n_conformers`` is ``None`` and there are either
      no conformers or more than ``max_conf`` conformers associated with
      the input ``offmol``.
      If ``generate_n_conformers`` is greater than the value of ``max_conf``.
    """
    # Check number of conformers if generate_n_conformers is None and return
    if generate_n_conformers is None:
        if offmol.n_conformers == 0:
            errmsg = ("No conformers are associated with input OpenFF "
                      "Molecule. Need at least one for partial charge "
                      "assignment")
            raise ValueError(errmsg)
        if offmol.n_conformers > max_conf:
            errmsg = ("OpenFF Molecule has too many conformers: "
                      f"{offmol.n_conformers}, selected partial charge "
                      f"method can only support a maximum of {max_conf} "
                      "conformers.")
            raise ValueError(errmsg)
        return


    # Check that generate_n_conformers < max_conf
    if generate_n_conformers > max_conf:
        errmsg = (f"{generate_n_conformers} conformers were requested "
                  "for partial charge generation, but the selected "
                  "method only supports up to {max_conf} conformers.")
        raise ValueError(errmsg)

    # Generate conformers

    # OpenEye tk needs cis carboxylic acids
    make_carbox_cis = any(
        [isinstance(i, OpenEyeToolkitWrapper)
         for i in toolkit_registry.registered_toolkits]
    )

    # We are being overly cautious by both passing the
    # registry and applying the manager here - this is
    # to avoid issues like:
    # https://github.com/openforcefield/openff-nagl/issues/69
    with toolkit_registry_manager(toolkit_registry):
        offmol.generate_conformers(
            n_conformers=generate_n_conformers,
            rms_cutoff=0.25 * unit.angstrom,
            make_carboxylic_acids_cis=make_carbox_cis,
            toolkit_registry=toolkit_registry,
        )


def assign_offmol_partial_charges(
    offmol: OFFMol,
    overwrite: bool,
    method: Literal['am1bcc', 'am1bccelf10', 'nagl', 'espaloma'],
    toolkit_backend: Literal['ambertools', 'openeye', 'rdkit'],
    generate_n_conformers: Optional[int],
    nagl_model: Optional[str],
) -> None:
    """
    Assign partial charges to an OpenFF Molecule based on a selected method.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      The Molecule to assign partial charges to.
    overwrite : bool
      Whether or not to overwrite any existing non-zero partial charges.
      Note that zeroed charges will always be overwritten.
    method : Literal['am1bcc', 'am1bccelf10', 'nagl', 'espaloma']
      Partial charge assignement method.
      Supported methods include; am1bcc, am1bccelf10, nagl, and espaloma.
    toolkit_backend : Literal['ambertools', 'openeye', 'rdkit']
      OpenFF toolkit backend employed for charge generation.
      Supported options:
        * ``ambertools``: selects both the AmberTools and RDKit Toolkit Wrapper
        * ``openeye``: selects the OpenEye toolkit Wrapper
        * ``rdkit``: selects the RDKit toolkit Wrapper
      Note that the ``rdkit`` backend cannot be used for `am1bcc` or
      ``am1bccelf10`` partial charge methods.
    generate_n_conformers : Optional[int]
      Number of conformers to generate for partial charge generation.
      If ``None`` (default), the input conformer will be used.
      Values greater than 1 can only be used alongside ``am1bccelf10``.
    nagl_model : Optional[str]
      The NAGL model to use for charge assignment if method is ``nagl``.
      If ``None``, the latest am1bcc NAGL charge model is used.

    Raises
    ------
    ValueError
      If the ``toolkit_backend`` is not suported by the selected ``method``.
      If ``generate_n_conformers`` is ``None``, but the input ``offmol``
      has no associated conformers.
      If the number of conformers passed or generated exceeds the number
      of conformers selected by the partial charge ``method``.
    """

    # If you have non-zero charges and not overwriting, just return
    if (offmol.partial_charges is not None and np.any(offmol.partial_charges)):
        if not overwrite:
            return

    # Dictionary for each available charge method
    # The idea of this pattern is to allow for maximum flexibility by
    # allowing for swapping out method calls as necessary.
    #
    # Must include:
    # 1. `confgen_func`: the conformer generation method
    # 2. `charge_func`: the partial charge assignment method
    # 2. `backends`: the allowed backends for the method
    # 3. `max_conf`: maximum number of allowed conformations for the method
    # 4. `charge_extra_kwargs`: any additional kwargs to be passed to the
    #    partial charge assignment method beyond the input offmol and
    #    the toolkitregistry
    CHARGE_METHODS = {
        "am1bcc": {
            "confgen_func": _generate_offmol_conformers,
            "charge_func": assign_offmol_am1bcc_charges,
            "backends": ['ambertools', 'openeye'],
            "max_conf": 1,
            "charge_extra_kwargs": {'partial_charge_method': 'am1bcc'},
        },
        "am1bccelf10": {
            "confgen_func": _generate_offmol_conformers,
            "charge_func": assign_offmol_am1bcc_charges,
            "backends": ['openeye'],
            "max_conf": None,
            "charge_extra_kwargs": {'partial_charge_kmethod': 'am1bccelf10'},
        },
        "nagl": {
            "confgen_func": _generate_offmol_conformers,
            "charge_func": assign_offmol_nagl_charges,
            "backends": ['openeye', 'rdkit', 'ambertools'],
            "max_conf": 1,
            "charge_extra_kwargs": {"model": nagl_model},
        },
        "espaloma": {
            "confgen_func": _generate_offmol_conformers,
            "charge_func": assign_offmol_espaloma_charges,
            "backends": ['rdkit', 'ambertools'],
            "max_conf": 1,
            "charge_extra_kwargs": {},
        },
    }


    backends = CHARGE_METHODS[method.lower()]['backends']
    if toolkit_backend.lower() not in backends:
        errmsg = (f"Selected toolkit_backend ({toolkit_backend}) cannot "
                  f"be used with the selected method ({method}). "
                  f"Available backends are: {backends}")
        raise ValueError(errmsg)

    toolkits = ToolkitRegistry(
        [i() for i in BACKEND_OPTIONS[toolkit_backend.lower()]]
    )

    # We make a copy of the molecule since we're going to modify conformers
    offmol_copy = copy.deepcopy(offmol)

    # Generate conformers - note this method may differ based on the partial
    # charge method employed
    CHARGE_METHODS[method.lower()]['confgen_func'](
        offmol=offmol_copy,
        max_conf=CHARGE_METHODS[method.lower()]['max_conf'],
        toolkit_registry=toolkits,
        generate_n_conformers=generate_n_conformers,
    )

    # Call selected method to assign partial charges
    CHARGE_METHODS[method.lower()]['charge_func'](
        offmol=offmol_copy,
        toolkit_registry=toolkits,
        **CHARGE_METHODS[method.lower()]['charge_extra_kwargs'],
    )

    # Copy partial charges back
    offmol.partial_charges = offmol_copy.partial_charges

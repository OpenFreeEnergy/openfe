.. _userguide_model_building:

Model building: force fields and partial charges
================================================

Prepared input structures describe the system and its initial conditions. A compatible force field is then required
to create a simulation ready model describing the interactions that govern the system's dynamics.
The force field and the partial charges are configured through the protocol settings and are applied to the
components in each :class:`.ChemicalSystem` during the protocol setup unit.

Here we detail model building for the **OpenMM** based protocols. For how to prepare the
structures, see :ref:`userguide_preparing_inputs`; for the general workflow
for creating and changing settings, see :ref:`defining-protocols`.

The simulation model is assembled from several choices:

* **Protein and other non-small-molecule components:** OpenMM force-field XML
  files supply parameters for the protein and, depending on the files, water,
  ions, lipids and other residues.
* **Ligands and small-molecule cofactors:** a separate small-molecule force
  field supplies parameters for :class:`.SmallMoleculeComponent`\s.
* **Partial charges:** Protein and other non-small-molecule components typically use predefined *library* charges defined in the OpenMM XML files.
  Small-molecule components can either use existing charges or have new charges assigned during setup.
* **Water:** the solvation model and the water and ion parameters must be
  compatible.

.. note::

    These choices affect the simulation dynamics and observables and should be changed carefully. Its important to keep them consistent
    across a ligand network and between calculations you intend to compare.
    Parameterization also depends on the chemistry of the prepared input; see the
    :ref:`domain of applicability <userguide_protocols>` for current limitations.

Protein force fields
--------------------

The ``forcefield_settings.forcefields`` list specifies OpenMM XML files used
to parameterize the protein and other non-small-molecule components. The
default force field list contains files for the protein, water, ions, phosphorylated residues and
lipids.

.. note::

    The XML force field files should provide predefined partial charges for the non-small-molecule components.
    The protocol does **not** assign charges using the ``partial_charge_settings`` for these components.

You can inspect the default settings for the protocol you are using via::

   from openfe.protocols import openmm_rfe

   settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
   print(settings.forcefield_settings.forcefields)

To change the protein force field, replace its XML entry with the desired
protein XML while retaining the other files needed by the system. For example,
the following replaces ff14SB with ff19SB::

   forcefields = settings.forcefield_settings.forcefields
   forcefields[forcefields.index("amber/ff14SB.xml")] = "amber19/protein.ff19SB.xml"

Force fields are provided via `OpenMMForceFields <https://openmm.github.io/openmmforcefields/dev/userguide.html#user-guide>`_, see
the documentation for the available options.
It is important to choose a compatible set of protein, water and ion parameters for your
application; replacing the protein XML alone does not establish that the
resulting combination has been validated. You should also check that the selected force
fields cover all non-small-molecule components in your system.

Small-molecule force fields
---------------------------

The ``forcefield_settings.small_molecule_forcefield`` setting selects the
force field for ligands and other :class:`.SmallMoleculeComponent`\s. The
default is ``openff-2.2.1``. OpenMMForceFields supports OpenFF (SMIRNOFF),
GAFF (General Amber Force Field) and Espaloma force-field families.
For example, to select OpenFF-2.3.0 before creating the protocol::

   settings.forcefield_settings.small_molecule_forcefield = "openff-2.3.0"
   protocol = openmm_rfe.RelativeHybridTopologyProtocol(settings)

It is important to select a force field with sufficient accuracy for your molecules, you can also use custom parameters, see the
:doc:`bespoke force field how-to </cookbook/bespoke_parameters>`.

For all available force-field settings, see :class:`.OpenMMSystemGeneratorFFSettings`.

Cofactors and other retained molecules
--------------------------------------

Noncovalent cofactors prepared as :class:`.SmallMoleculeComponent`\s use the
same small-molecule force field and partial-charge workflow as ligands, even
when they remain unchanged in a protein complex. Use consistent cofactor
parameters and charges across the calculations being compared.

Structural ions and crystallographic waters retained in a protein input are
instead parameterized through the appropriate OpenMM XML files. See
:ref:`prep_cofactors` for preparation requirements and limitations:
metal-containing cofactors such as heme are not supported by the standard
preparation workflow.

.. _userguide_partial_charges:

Partial charges
---------------

For ligands and cofactors represented as a SmallMoleculeComponent, the partial charges are assigned using the following order of preference:

1. If the component contains partial charges and at least one charge is nonzero, those charges are retained.
2. If charges are absent or all zero, the protocol assigns charges during setup using ``partial_charge_settings``.

The small-molecule force-field selection and the charge-assignment settings are there separate choices allowing for finer control of the model construction.

.. note::
    By default **AM1-BCC** charges are assigned via the AmberTools backend
    at the input conformer. This is different from the charges obtained by applying an
    OpenFF force field independently, since the charge method or backend may differ.

Assigning charges in advance to the input components ensures consistency between calculation repeats and different legs of a transformation,
see :doc:`Assigning partial charges </cookbook/user_charges>` (Python) and :doc:`Generating partial charges with the OpenFE CLI </tutorials/charge_molecules_cli_tutorial>`.

Available charge methods
~~~~~~~~~~~~~~~~~~~~~~~~

**openfe** supports a range of **AM1-BCC** style charge methods used by most general small molecule force fields.
The ``partial_charge_settings.partial_charge_method`` setting selects the
method used when charges need to be assigned during protocol setup. The available methods are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Description
   * - ``am1bcc``
     - AM1-BCC (default), using AmberTools or OpenEye. Generated at the input conformer.
   * - ``am1bccelf10``
     - AM1-BCC with ELF10 conformer selection. Requires OpenEye; setting
       ``number_of_conformers`` to at least 500 is recommended.
   * - ``nagl``
     - Charges predicted by an OpenFF NAGL model. Requires the NAGL
       dependencies; if ``nagl_model`` is unset, the latest available production
       AM1-BCC model is used.
   * - ``espaloma``
     - Charges predicted by Espaloma Charge. Requires ``espaloma_charge``.

The default ``number_of_conformers`` is unset, so the input conformer is used.
Generating more than one conformer is only supported with ``am1bccelf10``.
The backend and charge method must be compatible; for example, RDKit alone
cannot generate conventional AM1-BCC charges. See
:class:`.OpenFFPartialChargeSettings` for the available settings and backend
restrictions.

The CLI also supports a ``forcefield`` charge method which uses a SMIRNOFF
force field to assign charges. This enables additional charge assignment methods
supported by the SMIRNOFF format, such as small-molecule library charges.
See :ref:`userguide_cli_yaml_interface` for the CLI options.

Choosing a charge model
~~~~~~~~~~~~~~~~~~~~~~

Small-molecule force fields are typically optimised for a specific charge model,
you should check the documentation and literature for the force field you are using to ensure compatibility.
Some useful references for charge assignment and force field compatibility are:

* `OpenFF force fields <https://github.com/openforcefield/openff-forcefields>`_
  documents the charge models used by OpenFF releases.
* The `OpenFF FAQ <https://docs.openforcefield.org/en/latest/faq.html#under-the-hood>`_
  explains charge assignment when applying a SMIRNOFF force field.
* The `AmberTools documentation <https://ambermd.org/AmberTools.php>`_ provides
  guidance for GAFF.

For guides on charge assignment, see :doc:`Assigning partial charges
</cookbook/user_charges>` (Python) and :doc:`Generating partial charges with
the OpenFE CLI </tutorials/charge_molecules_cli_tutorial>`.

Water and model compatibility
-----------------------------

For systems solvated during protocol setup, the ``solvent_model`` selects the water model
used when adding solvent. The water and ion XML files in
``forcefield_settings.forcefields`` must be compatible with those choices e.g. a 4-point water model cannot be used with a 3-point force field.
The same force-field files must also cover any retained crystallographic
waters and ions.

For a pre-solvated input, the protocol does not add solvent to that leg: the
provided waters and ions must still be compatible with the selected
force-field parameters. See :ref:`userguide_solvation_models` for more details on implicit and explicit solvation in
the protocols.

See also
--------

* :ref:`Preparing input structures <userguide_preparing_inputs>`
* :ref:`Defining settings and protocols <defining-protocols>`
* :ref:`Protocols and domain of applicability <userguide_protocols>`
* :doc:`Assigning partial charges </cookbook/user_charges>`
* :doc:`Generating partial charges with the OpenFE CLI </tutorials/charge_molecules_cli_tutorial>`
* :doc:`Bespoke parameters </cookbook/bespoke_parameters>`
* :ref:`OpenMM protocol settings API <openmm protocol settings api>`

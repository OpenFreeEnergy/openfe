import logging

logger = logging.getLogger(__name__)


#TODO: Follow OpenFE convention
from simtk import unit

class InvalidMappingException(Exception):
    """
    Invalid atom mapping for relative free energy transformation.

    """

    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class AtomMapping:
    """
    A container representing an atom mapping between two small molecules.

    This object is mutable, but only valid atom mappings can be stored.
    The validation check occurs whenever a new atom mapping is set.

    .. todo :: Figure out how this should work for biopolymers.

    .. todo :: Migrate to openff.toolkit.topology.Molecule when able

    Attributes
    ----------
    old_mol : openff.toolkit.topology.Molecule
        Copy of the first molecule to be mapped
    new_mol : openff.toolkit.topology.Molecule
        Copy of the second molecule to be mapped
    n_mapped_atoms : int
        The number of mapped atoms.
        Read-only property.
    new_to_old_atom_map : dict of int : int
        new_to_old_atom_map[new_atom_index] is the atom index in old_oemol corresponding to new_atom_index in new_oemol
        A copy is returned, but this attribute can be set.
        Zero-based indexing within the atoms in old_mol and new_mol is used.
    old_to_new_atom_map : dict of int : int
        old_to_new_atom_map[old_atom_index] is the atom index in new_oemol corresponding to old_atom_index in old_oemol
        A copy is returned, but this attribute can be set.
        Zero-based indexing within the atoms in old_mol and new_mol is used.

    Examples
    --------

    Create an atom mapping for ethane -> ethanol
    >>> from openff.toolkit.topology import Molecule
    >>> ethane = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])([H:8])')
    >>> ethanol = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[O:8][H:9]')
    >>> atom_mapping = AtomMapping(ethane, ethanol, old_to_new_atom_map={0:0, 4:4})

    """

    def __init__(self, old_mol, new_mol, new_to_old_atom_map=None, old_to_new_atom_map=None):
        """
        Construct an AtomMapping object.

        Once constructed, either new_to_old_atom_map or old_to_new_atom_map can be accessed or set.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            First molecule to be mapped
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            Second molecule to be mapped
        new_to_old_atom_map : dict of int : int
            new_to_old_atom_map[new_atom_index] is the atom index in old_oemol corresponding to new_atom_index in new_oemol

        """
        # Store molecules
        # TODO: Should we be allowing unspecified stereochemistry?

        # Create (copies of) OpenFF Molecule objects from old and new molecules
        def create_offmol_copy(mol):
            """Create an OpenFF Molecule object copy from the specified molecule"""
            try:
                from openff.toolkit.topology import Molecule

                offmol = Molecule(mol, allow_undefined_stereo=True)
            except ValueError as e:
                # Try the sneaky OpenEye OEMol -> OpenFF Molecule converter that bypasses the RadicalsNotSupportedError thrown internally.
                offmol = _convert_opemol_to_offmol(mol, allow_undefined_stereo=True)
            return offmol

        self.old_mol = create_offmol_copy(old_mol)
        self.new_mol = create_offmol_copy(new_mol)

        # Store atom maps
        if (old_to_new_atom_map is not None) and (new_to_old_atom_map is not None):
            raise ValueError(
                "Only one of old_to_new_atom_map or new_to_old_atom_map can be specified"
            )
        if (old_to_new_atom_map is None) and (new_to_old_atom_map is None):
            raise ValueError("One of old_to_new_atom_map or new_to_old_atom_map must be specified")
        if old_to_new_atom_map is not None:
            self.old_to_new_atom_map = old_to_new_atom_map
        if new_to_old_atom_map is not None:
            self.new_to_old_atom_map = new_to_old_atom_map

    def __repr__(self):
        return f"AtomMapping(Molecule.from_mapped_smiles('{self.old_mol.to_smiles(mapped=True)}'), Molecule.from_mapped_smiles('{self.new_mol.to_smiles(mapped=True)}'), old_to_new_atom_map={self.old_to_new_atom_map})"

    def __str__(self):
        return f"AtomMapping : {self.old_mol.to_smiles(mapped=True)} -> {self.new_mol.to_smiles(mapped=True)} : mapped atoms {self.old_to_new_atom_map}"

    def __hash__(self):
        """Compute unique hash that accounts for atom ordering in molecules and permutation invariance of dictionary items"""
        return hash(
            (
                self.old_mol.to_smiles(mapped=True),
                self.new_mol.to_smiles(mapped=True),
                frozenset(self.old_to_new_atom_map.items()),
            )
        )

    def _validate(self):
        """
        Validate the atom mapping is consistent with stored molecules.
        """
        # Ensure mapping is not empty
        if len(self.new_to_old_atom_map) == 0:
            raise InvalidMappingException(f"Atom mapping contains no mapped atoms")

        # Ensure all keys and values are integers
        if not (
            all(isinstance(x, int) for x in self.new_to_old_atom_map.keys())
            and all(isinstance(x, int) for x in self.new_to_old_atom_map.values())
        ):
            raise InvalidMappingException(
                f"Atom mapping contains non-integers:\n{self.new_to_old_atom_map}"
            )

        # Check to make sure atom indices are within valid range
        if not set(self.new_to_old_atom_map.keys()).issubset(
            range(self.new_mol.n_atoms)
        ) or not set(self.new_to_old_atom_map.values()).issubset(range(self.old_mol.n_atoms)):
            raise InvalidMappingException(
                f"Atom mapping contains invalid atom indices:\n{self.new_to_old_atom_map}\nold_mol: {self.old_mol.to_smiles(mapped=True)}\nnew_mol: {self.new_mol.to_smiles(mapped=True)}"
            )

        # Make sure mapping is one-to-one
        if len(set(self.new_to_old_atom_map.keys())) != len(set(self.new_to_old_atom_map.values())):
            raise InvalidMappingException(
                "Atom mapping is not one-to-one:\n{self.new_to_old_atom_map}"
            )

    @property
    def new_to_old_atom_map(self):
        import copy

        return copy.deepcopy(self._new_to_old_atom_map)

    @new_to_old_atom_map.setter
    def new_to_old_atom_map(self, value):
        self._new_to_old_atom_map = dict(value)
        self._validate()

    @property
    def old_to_new_atom_map(self):
        # Construct reversed atom map on the fly
        return dict(map(reversed, self._new_to_old_atom_map.items()))

    @old_to_new_atom_map.setter
    def old_to_new_atom_map(self, value):
        self._new_to_old_atom_map = dict(map(reversed, value.items()))
        self._validate()

    @property
    def n_mapped_atoms(self):
        """The number of mapped atoms"""
        return len(self._new_to_old_atom_map)

    def render_image(self, filename, format=None, width=1200, height=600):
        """
        Render the atom mapping image.

        .. note :: This currently requires the OpenEye toolkit.

        .. todo ::

           * Add support for biopolymer mapping rendering
           * Add support for non-OpenEye rendering?

        Parameters
        ----------
        filename : str
            The image filename to write to.
            Format automatically detected from file suffix if 'format' is not specified.
            If None, will return IPython.disply.Image
        format : str, optional, default=None
            If specified, format to render in (e.g. 'png', 'pdf')
        width : int, optional, default=1200
            Width in pixels
        height : int, optional, default=600
            Height in pixels

        Returns
        -------
        image : IPython.display.Image
            If filename is None, the image will be returned.

        """
        from openeye import oechem, oedepict

        # Handle format
        if format is None:
            if filename is not None:
                format = oechem.OEGetFileExtension(filename)
            else:
                format = "png"
        if not oedepict.OEIsRegisteredImageFile(format):
            raise ValueError(f"Unknown image type {format}")

        molecule1 = self.old_mol.to_openeye()
        molecule2 = self.new_mol.to_openeye()
        new_to_old_atom_map = self.new_to_old_atom_map

        oechem.OEGenerate2DCoordinates(molecule1)
        oechem.OEGenerate2DCoordinates(molecule2)

        # Add both to an OEGraphMol reaction
        rmol = oechem.OEGraphMol()
        rmol.SetRxn(True)

        def add_molecule(mol):
            # Add atoms
            new_atoms = list()
            old_to_new_atoms = dict()
            for old_atom in mol.GetAtoms():
                new_atom = rmol.NewAtom(old_atom.GetAtomicNum())
                new_atom.SetFormalCharge(old_atom.GetFormalCharge())
                new_atoms.append(new_atom)
                old_to_new_atoms[old_atom] = new_atom
            # Add bonds
            for old_bond in mol.GetBonds():
                rmol.NewBond(
                    old_to_new_atoms[old_bond.GetBgn()],
                    old_to_new_atoms[old_bond.GetEnd()],
                    old_bond.GetOrder(),
                )
            return new_atoms, old_to_new_atoms

        [new_atoms_1, old_to_new_atoms_1] = add_molecule(molecule1)
        [new_atoms_2, old_to_new_atoms_2] = add_molecule(molecule2)

        # Label reactant and product
        for atom in new_atoms_1:
            atom.SetRxnRole(oechem.OERxnRole_Reactant)
        for atom in new_atoms_2:
            atom.SetRxnRole(oechem.OERxnRole_Product)

        core1 = oechem.OEAtomBondSet()
        core2 = oechem.OEAtomBondSet()
        # add all atoms to the set
        core1.AddAtoms(new_atoms_1)
        core2.AddAtoms(new_atoms_2)
        # Label mapped atoms
        core_change = oechem.OEAtomBondSet()
        index = 1
        for index2, index1 in new_to_old_atom_map.items():
            new_atoms_1[index1].SetMapIdx(index)
            new_atoms_2[index2].SetMapIdx(index)
            # now remove the atoms that are core, so only uniques are highlighted
            core1.RemoveAtom(new_atoms_1[index1])
            core2.RemoveAtom(new_atoms_2[index2])
            if new_atoms_1[index1].GetAtomicNum() != new_atoms_2[index2].GetAtomicNum():
                # this means the element type is changing
                core_change.AddAtom(new_atoms_1[index1])
                core_change.AddAtom(new_atoms_2[index2])
            index += 1
        # Set up image options
        itf = oechem.OEInterface()
        oedepict.OEConfigureImageOptions(itf)

        # Setup depiction options
        oedepict.OEConfigure2DMolDisplayOptions(itf, oedepict.OE2DMolDisplaySetup_AromaticStyle)
        opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)
        opts.SetBondWidthScaling(True)
        opts.SetAtomPropertyFunctor(oedepict.OEDisplayAtomMapIdx())
        opts.SetAtomColorStyle(oedepict.OEAtomColorStyle_WhiteMonochrome)
        opts.SetAromaticStyle(oedepict.OEAromaticStyle_Circle)
        oedepict.OESetup2DMolDisplayOptions(opts, itf)

        # Depict reaction with component highlights
        oechem.OEGenerate2DCoordinates(rmol)
        display = oedepict.OE2DMolDisplay(rmol, opts)

        if core1.NumAtoms() != 0:
            oedepict.OEAddHighlighting(
                display, oechem.OEColor(oechem.OEPink), oedepict.OEHighlightStyle_Stick, core1
            )
        if core2.NumAtoms() != 0:
            oedepict.OEAddHighlighting(
                display, oechem.OEColor(oechem.OEPurple), oedepict.OEHighlightStyle_Stick, core2
            )
        if core_change.NumAtoms() != 0:
            oedepict.OEAddHighlighting(
                display,
                oechem.OEColor(oechem.OEGreen),
                oedepict.OEHighlightStyle_Stick,
                core_change,
            )

        if filename is not None:
            ofs = oechem.oeofstream()
            if not ofs.open(filename):
                raise Exception("Cannot open output file %s" % filename)
            oedepict.OERenderMolecule(ofs, format, display)
            ofs.close()
        else:
            from IPython.display import Image

            oeimage = oedepict.OEImage(width, height)
            oedepict.OERenderMolecule(oeimage, display)
            string = oedepict.OEWriteImageToString(format, oeimage)
            return Image(string)

    def _ipython_display_(self):
        from IPython.display import display

        print(str(self))
        display(self.render_image(None, format="png"))

    def creates_or_breaks_rings(self):
        """Determine whether the mapping causes rings to be created or broken in transformation.

        Returns
        -------
        breaks_rings : bool
            Returns True if the atom mapping would cause rings to be created or broken
        """
        import networkx as nx

        # Check that any bond between mapped atoms in one molecule is also bonded in the other molecule
        for mol1, mol2, atom_mapping in [
            (self.old_mol, self.new_mol, self.old_to_new_atom_map),
            (self.new_mol, self.old_mol, self.new_to_old_atom_map),
        ]:
            mol1_graph = mol1.to_networkx()
            mol2_graph = mol2.to_networkx()
            # Check that all bonds between mapped atoms in mol1 are mapped in mol2
            for mol1_edge in mol1_graph.edges:
                if set(mol1_edge).issubset(atom_mapping.keys()):
                    mol2_edge = (atom_mapping[mol1_edge[0]], atom_mapping[mol1_edge[1]])
                    if not mol2_graph.has_edge(*mol2_edge):
                        return True
        # For every cycle in the molecule, check that ALL atoms in the cycle are mapped or not mapped
        for molecule, mapped_atoms in [
            (self.old_mol, self.old_to_new_atom_map.keys()),
            (self.new_mol, self.old_to_new_atom_map.values()),
        ]:
            graph = molecule.to_networkx()
            for cycle in nx.simple_cycles(graph.to_directed()):
                n_atoms_in_cycle = len(cycle)
                if n_atoms_in_cycle < 3:
                    # Cycle must contain at least three atoms to be useful
                    continue
                n_atoms_mapped = len(set(cycle).intersection(mapped_atoms))
                if not ((n_atoms_mapped == 0) or (n_atoms_in_cycle == n_atoms_mapped)):
                    return True
        return False

    # TODO: Not present in gufe -> might be a factory to "fix" or a function in PersesAtomMapper
    def unmap_partially_mapped_cycles(self):
        """De-map atoms to ensure the partition function will be factorizable.

        This algorithm builds a graph for old and new molecules where edges connect bonded atoms where both atoms are mapped.
        We then find the largest connected subgraph and de-map all other atoms

        .. todo :: Change this algorithm to operate on the hybrid graph

        .. todo :: Check to make sure that we don't end up with problematic mappings.

        """
        import networkx as nx

        # Save initial mapping
        import copy

        initial_mapping = copy.deepcopy(self)

        # Traverse all cycles and de-map any atoms that are in partially mapped cycles
        # making sure to check that bonds in mapped atoms are concordant.
        atoms_to_demap = dict()
        for mol1, mol2, atom_mapping, selection in [
            (self.old_mol, self.new_mol, self.old_to_new_atom_map, "old"),
            (self.new_mol, self.old_mol, self.new_to_old_atom_map, "new"),
        ]:
            atoms_to_demap[selection] = set()
            mol1_graph = mol1.to_networkx()
            mol2_graph = mol2.to_networkx()
            for cycle in nx.simple_cycles(mol1_graph.to_directed()):
                n_atoms_in_cycle = len(cycle)
                if n_atoms_in_cycle < 3:
                    # Need at least three atoms in a cycle
                    continue

                # Check that all atoms and bonds are mapped
                def is_cycle_mapped(mol1_graph, mol2_graph, cycle, atom_mapping):
                    n_atoms_in_cycle = len(cycle)
                    for index in range(n_atoms_in_cycle):
                        mol1_atom1, mol1_atom2 = cycle[index], cycle[(index + 1) % n_atoms_in_cycle]
                        if not ((mol1_atom1 in atom_mapping) and (mol1_atom2 in atom_mapping)):
                            return False
                        mol2_atom1, mol2_atom2 = atom_mapping[mol1_atom1], atom_mapping[mol1_atom2]
                        if not mol2_graph.has_edge(mol2_atom1, mol2_atom2):
                            return False

                    # All atoms and bonds in cycle are mapped correctly
                    return True

                if not is_cycle_mapped(mol1_graph, mol2_graph, cycle, atom_mapping):
                    # De-map any atoms in this map
                    for atom_index in cycle:
                        atoms_to_demap[selection].add(atom_index)

        # Update mapping to eliminate any atoms in partially mapped cycles
        if len(atoms_to_demap["old"]) > 0 or len(atoms_to_demap["new"]) > 0:
            _logger.debug(
                f"AtomMapping.unmap_partially_mapped_cycles(): Demapping atoms that were in partially mapped cycles: {atoms_to_demap}"
            )
        self.old_to_new_atom_map = {
            old_atom: new_atom
            for old_atom, new_atom in self.old_to_new_atom_map.items()
            if (old_atom not in atoms_to_demap["old"]) and (new_atom not in atoms_to_demap["new"])
        }

        # Construct old_mol graph pruning any edges that do not share bonds
        # correctly mapped in both molecules
        old_mol_graph = self.old_mol.to_networkx()
        new_mol_graph = self.new_mol.to_networkx()
        for edge in old_mol_graph.edges:
            if not set(edge).issubset(self.old_to_new_atom_map.keys()):
                # Remove the edge because bond is not between mapped atoms
                old_mol_graph.remove_edge(*edge)
                _logger.debug(f"Demapping old_mol edge {edge} because atoms are not mapped")
            else:
                # Both atoms are mapped
                # Ensure atoms are also bonded in new_mol
                if not new_mol_graph.has_edge(
                    self.old_to_new_atom_map[edge[0]], self.old_to_new_atom_map[edge[1]]
                ):
                    old_mol_graph.remove_edge(*edge)
                    _logger.debug(
                        f"Demapping old_mol edge {edge} because atoms are not bonded in new_mol"
                    )

        # Find the largest connected component of the graph
        connected_components = [component for component in nx.connected_components(old_mol_graph)]
        connected_components.sort(reverse=True, key=lambda subgraph: len(subgraph))
        largest_connected_component = connected_components[0]
        _logger.debug(
            f"AtomMapping.unmap_partially_mapped_cycles(): Connected component sizes: {[len(component) for component in connected_components]}"
        )

        # Check to make sure we haven't screwed something up
        if len(largest_connected_component) == 0:
            msg = f"AtomMapping.unmap_partially_mapped_cycles(): Largest connected component has too few atoms ({len(largest_connected_component)} atoms)\n"
            msg += f"  Initial mapping (initial-mapping.png): {self}\n"
            msg += f"  largest_connected_component: {largest_connected_component}\n"
            initial_mapping.render_image("initial-mapping.png")
            raise AssertionError(msg)

        # Update mapping to include only largest connected component atoms
        self.old_to_new_atom_map = {
            old_atom: new_atom
            for old_atom, new_atom in self.old_to_new_atom_map.items()
            if (old_atom in largest_connected_component)
        }

        _logger.debug(
            f"AtomMapping.unmap_partially_mapped_cycles(): Number of mapped atoms changed from {len(initial_mapping.old_to_new_atom_map)} -> {len(self.old_to_new_atom_map)}"
        )

        # Check to make sure we haven't screwed something up
        if self.creates_or_breaks_rings() == True:
            msg = f"AtomMapping.unmap_partially_mapped_cycles() failed to eliminate all ring creation/breaking. This indicates a programming logic error.\n"
            msg += f"  Initial mapping (initial-mapping.png): {initial_mapping}\n"
            msg += f"  After demapping (final-mapping.png)  : {self}\n"
            msg += f"  largest_connected_component: {largest_connected_component}\n"
            initial_mapping.render_image("initial-mapping.png")
            self.render_image("final-mapping.png")
            raise InvalidMappingException(msg)

    # TODO: Not present in gufe
    def preserve_chirality(self):
        """
        Alter the atom mapping to preserve chirality

        The current scheme is implemented as follows:
        for atom_new, atom_old in new_to_old.items():
            if atom_new is R/S and atom_old is undefined:
                # we presume that one of the atom neighbors is being changed, so map it accordingly
            elif atom_new is undefined and atom_old is R/S:
                # we presume that one of the atom neighbors is not being mapped, so map it accordingly
            elif atom_new is R/S and atom_old is R/S:
                # we presume nothing is changing
            elif atom_new is S/R and atom_old is R/S:
                # we presume that one of the neighbors is changing
                # check if all of the neighbors are being mapped:
                    if True, flip two
                    else: do nothing

        .. todo :: Check that chirality is correctly handled.

        """
        # TODO: Simplify this.

        import openeye.oechem as oechem

        pattern_atoms = {atom.GetIdx(): atom for atom in self.old_mol.to_openeye().GetAtoms()}
        target_atoms = {atom.GetIdx(): atom for atom in self.new_mol.to_openeye().GetAtoms()}
        # _logger.warning(f"\t\t\told oemols: {pattern_atoms}")
        # _logger.warning(f"\t\t\tnew oemols: {target_atoms}")
        copied_new_to_old_atom_map = copy.deepcopy(self.new_to_old_atom_map)
        _logger.debug(self.new_to_old_atom_map)

        for new_index, old_index in self.new_to_old_atom_map.items():
            if target_atoms[new_index].IsChiral() and not pattern_atoms[old_index].IsChiral():
                # make sure that not all the neighbors are being mapped
                # get neighbor indices:
                neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
                if all(
                    nbr in set(list(self.new_to_old_atom_map.keys())) for nbr in neighbor_indices
                ):
                    _logger.warning(
                        f"the atom map cannot be reconciled with chirality preservation!  It is advisable to conduct a manual atom map."
                    )
                    return {}
                else:
                    # try to remove a hydrogen
                    hydrogen_maps = [
                        atom.GetIdx()
                        for atom in target_atoms[new_index].GetAtoms()
                        if atom.GetAtomicNum() == 1
                    ]
                    mapped_hydrogens = [
                        _idx
                        for _idx in hydrogen_maps
                        if _idx in list(self.new_to_old_atom_map.keys())
                    ]
                    if mapped_hydrogens != []:
                        del copied_new_to_old_atom_map[mapped_hydrogens[0]]
                    else:
                        _logger.warning(
                            f"there may be a geometry problem!  It is advisable to conduct a manual atom map."
                        )
            elif not target_atoms[new_index].IsChiral() and pattern_atoms[old_index].IsChiral():
                # we have to assert that one of the neighbors is being deleted
                neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
                if any(
                    nbr_idx not in list(self.new_to_old_atom_map.keys())
                    for nbr_idx in neighbor_indices
                ):
                    pass
                else:
                    _logger.warning(
                        f"the atom map cannot be reconciled with chirality preservation since no hydrogens can be deleted!  It is advisable to conduct a manual atom map."
                    )
                    return {}
            elif (
                target_atoms[new_index].IsChiral()
                and pattern_atoms[old_index].IsChiral()
                and oechem.OEPerceiveCIPStereo(self.old_mol, pattern_atoms[old_index])
                == oechem.OEPerceiveCIPStereo(self.new_mol, target_atoms[new_index])
            ):
                # check if all the atoms are mapped
                neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
                if all(
                    nbr in set(list(self.new_to_old_atom_map.keys())) for nbr in neighbor_indices
                ):
                    pass
                else:
                    _logger.warning(
                        f"the atom map cannot be reconciled with chirality preservation since all atom neighbors are being mapped!  It is advisable to conduct a manual atom map."
                    )
                    return {}
            elif (
                target_atoms[new_index].IsChiral()
                and pattern_atoms[old_index].IsChiral()
                and oechem.OEPerceiveCIPStereo(self.old_mol, pattern_atoms[old_index])
                != oechem.OEPerceiveCIPStereo(self.new_mol, target_atoms[new_index])
            ):
                neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
                if all(
                    nbr in set(list(self.new_to_old_atom_map.keys())) for nbr in neighbor_indices
                ):
                    _logger.warning(
                        f"the atom map cannot be reconciled with chirality preservation since all atom neighbors are being mapped!  It is advisable to conduct a manual atom map."
                    )
                    return {}
                else:
                    # try to remove a hydrogen
                    hydrogen_maps = [
                        atom.GetIdx()
                        for atom in target_atoms[new_index].GetAtoms()
                        if atom.GetAtomicNum() == 1
                    ]
                    mapped_hydrogens = [
                        _idx
                        for _idx in hydrogen_maps
                        if _idx in list(self.new_to_old_atom_map.keys())
                    ]
                    if mapped_hydrogens != []:
                        del copied_new_to_old_atom_map[mapped_hydrogens[0]]
                    else:
                        _logger.warning(
                            f"there may be a geometry problem.  It is advisable to conduct a manual atom map."
                        )

        # Update atom map
        self.new_to_old_atom_map = copied_new_to_old_atom_map


class AtomMapper:
    """
    Generate atom mappings between two molecules for relative free energy transformations.

    .. note ::

      As this doesn't generate a system, it will not be
      accurate for whether hydrogens are mapped or not.
      It also doesn't check that this is feasible to simulate,
      but is just helpful for testing options.

    .. todo ::

       * Expose options for whether bonds to hydrogen are constrained or not (and hence should be mapped or not)

       * Find a better way to express which mappings are valid for the hybrid topology factory

    Attributes
    ----------
    use_positions : bool, optional, default=True
        If True, will attempt to use positions of molecules to determine optimal mappings.
        If False, will only use maximum common substructure (MCSS).
    atom_expr : openeye.oechem.OEExprOpts
        Override for atom matching expression; None if default is to be used.
    bond_expr : openeye.oechem.OEExprOpts
        Override for bond matching expression; None if default is to be used.
    allow_ring_breaking : bool
        Wether or not to allow ring breaking in map
    coordinate_tolerance : simtk.unit.Quantity, optional, default=0.25*simtk.unit.angstroms
        Coordinate tolerance for geometry-derived mappings.

    Examples
    --------

    Create an AtomMapper factory:

    >>> atom_mapper = AtomMapper()

    You can also configure it after it has been created:

    >>> atom_mapper.use_positions = True # use positions in scoring mappings if available
    >>> atom_mapper.allow_ring_breaking = False # don't allow rings to be broken
    >>> from openeye import oechem
    >>> atom_mapper.atom_expr = oechem.OEExprOpts_Hybridization # override default atom_expr

    Specify two molecules without positions

    >>> from openff.toolkit.topology import Molecule
    >>> ethane = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])([H:8])')
    >>> ethanol = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[O:8][H:9]')

    Retrieve all mappings between two molecules

    >>> atom_mappings = atom_mapper.get_all_mappings(ethane, ethanol)

    Retrieve optimal mapping between molecules

    >>> atom_mapping = atom_mapper.get_best_mapping(ethane, ethanol)

    Stochastically sample a mapping between molecules

    >>> atom_mapping = atom_mapper.get_sampled_mapping(ethane, ethanol)

    We can access (or modify) either the old-to-new atom mapping or new-to-old atom mapping,
    and both will be kept in a self-consistent state:

    >>> atom_mapping.old_to_new_atom_map
    >>> atom_mapping.new_to_old_atom_map

    Copies of the initial and final molecules are also available as OpenFF Molecule objects:

    >>> atom_mapping.old_mol
    >>> atom_mapping.new_mol

    The AtomMapper can also utilize positions in generating atom mappings.
    If positions are available, they will be used to derive mappings by default.

    >>> atom_mapper.use_positions = True # use positions in scoring mappings if available
    >>> old_mol = Molecule.from_file('old_mol.sdf')
    >>> new_mol = Molecule.from_file('new_mol.sdf')
    >>> atom_mapping = atom_mapper.get_best_mapping(old_mol, new_mol)

    The mapping can also be generated only from positions,
    rather than scoring mappings from MCSS:

    >>> atom_mapping = atom_mapper.generate_atom_mapping_from_positions(old_mol, new_mol)

    The tolerance for position scoring or position-derived mappings can be adjusted in the AtomMapper factory:

    >>> from simtk import unit
    >>> atom_mapper.coordinate_tolerance = 0.3*unit.angstroms

    """

    def __init__(
        self,
        map_strength="default",
        atom_expr=None,
        bond_expr=None,
        use_positions=True,
        allow_ring_breaking=False,
        external_inttypes=False,
        matching_criterion="index",
        coordinate_tolerance=0.25 * unit.angstroms,
    ):
        """
        Create an AtomMapper factory.

        Parameters
        ----------
        map_strength : str, optional, default='default'
            Select atom mapping atom and bond expression defaults: ['strong', 'default', 'weak'].
            These can be overridden by specifying atom_expr or bond_expr.
        atom_expr : openeye.oechem.OEExprOpts
            Override for atom matching expression; None if default is to be used.
        bond_expr : openeye.oechem.OEExprOpts
            Override for bond matching expression; None if default is to be used.
        use_positions : bool, optional, default=True
            If True, will attempt to use positions of molecules to determine optimal mappings.
            If False, will use maximum common substructure (MCSS).
        allow_ring_breaking : bool, default=False
            Wether or not to allow ring breaking in map
        external_inttypes : bool, optional, default=False
            If True, IntTypes already assigned to oemols will be used for mapping, if IntType is in the atom or bond expression.
            Otherwise, IntTypes will be overwritten such as to ensure rings of different sizes are not matched.
        matching_criterion : str, optional, default='index'
            The best atom map is pulled based on some ranking criteria;
            if 'index', the best atom map is chosen based on the map with the maximum number of atomic index matches;
            if 'name', the best atom map is chosen based on the map with the maximum number of atom name matches
            else: raise Exception.
            NOTE : the matching criterion pulls patterns and target matches based on indices or names;
                   if 'names' is chosen, it is first asserted that old and new molecules have atoms that are uniquely named
        coordinate_tolerance : simtk.unit.Quantity, optional, default=0.25*simtk.unit.angstroms
            Coordinate tolerance for geometry-derived mappings.

        """
        # Configure default object attributes
        self.use_positions = use_positions
        self.allow_ring_breaking = allow_ring_breaking
        self.external_inttypes = external_inttypes
        self.matching_criterion = matching_criterion
        self.coordinate_tolerance = coordinate_tolerance

        # TODO:
        # rdFMCS module
        # https://www.rdkit.org/docs/source/rdkit.Chem.rdFMCS.html
        # MCS example: https://www.rdkit.org/docs/GettingStartedInPython.html#maximum-common-substructure
        # mcs_result = rdFMCS.FindMCS([mol1, mol2])
        # timemachine example: https://github.com/proteneer/timemachine/blob/master/timemachine/fe/atom_mapping.py
        # lomap: https://github.com/OpenFreeEnergy/Lomap/blob/main/lomap/mcs.py#L600
        # gives us a single SMARTS that we can use to figure out what our match is
        # https://github.com/proteneer/timemachine/blob/6ee118739a1800bae33e0f489e0718369eaa091d/timemachine/fe/atom_mapping.py#L11

        # Determine atom and bond expressions
        import openeye.oechem as oechem

        DEFAULT_EXPRESSIONS = {
            # weak requirements for mapping atoms == more atoms mapped, more in core
            # atoms need to match in aromaticity. Same with bonds.
            # maps ethane to ethene, CH3 to NH2, but not benzene to cyclohexane
            "weak": {
                #'atom' : oechem.OEExprOpts_EqAromatic | oechem.OEExprOpts_EqNotAromatic, #| oechem.OEExprOpts_IntType
                #'bond' : oechem.OEExprOpts_DefaultBonds
                "atom": oechem.OEExprOpts_RingMember,
                "bond": oechem.OEExprOpts_RingMember,
            },
            # default atom expression, requires same aromaticitiy and hybridization
            # bonds need to match in bond order
            # ethane to ethene wouldn't map, CH3 to NH2 would map but CH3 to HC=O wouldn't
            "default": {
                "atom": oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_RingMember,
                #'atom' : oechem.OEExprOpts_Hybridization, #| oechem.OEExprOpts_IntType
                "bond": oechem.OEExprOpts_DefaultBonds,
            },
            # strong requires same hybridization AND the same atom type
            # bonds are same as default, require them to match in bond order
            "strong": {
                "atom": oechem.OEExprOpts_Hybridization
                | oechem.OEExprOpts_AtomicNumber
                | oechem.OEExprOpts_Aromaticity
                | oechem.OEExprOpts_RingMember,
                #'atom' : oechem.OEExprOpts_Hybridization | oechem.OEExprOpts_HvyDegree | oechem.OEExprOpts_DefaultAtoms, # This seems broken for biopolymers due to OEExprOpts_HvyDegree, which does not seem to be working properly
                "bond": oechem.OEExprOpts_DefaultBonds,
            },
        }

        if map_strength is None:
            map_strength = "default"
        if atom_expr is None:
            _logger.debug(f"No atom expression defined, using map strength : {map_strength}")
            atom_expr = DEFAULT_EXPRESSIONS[map_strength]["atom"]
        if bond_expr is None:
            _logger.debug(f"No bond expression defined, using map strength : {map_strength}")
            bond_expr = DEFAULT_EXPRESSIONS[map_strength]["bond"]

        self.atom_expr = atom_expr
        self.bond_expr = bond_expr

    def get_all_mappings(self, old_mol, new_mol):
        """Retrieve all valid atom mappings and their scores for the proposed transformation.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mappings : list of AtomMapping
            All valid atom mappings annotated with mapping scores

        Examples:
        ---------
        Specify two molecules without positions

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])([H:8])')
        >>> ethanol = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[O:8][H:9]')

        Retrieve all mappings between two molecules

        >>> atom_mappings = atom_mapper.get_all_mappings(ethane, ethanol)

        You can also specify OEMols for one or more of the molecules:

        >>> atom_mappings = atom_mapper.get_all_mappings(ethane.to_openeye(), ethanol.to_openeye())
        >>> atom_mappings = atom_mapper.get_all_mappings(ethane.to_openeye(), ethanol)

        Algorithm
        ---------
        First, we attempt to generate scaffold mappings between compounds:
            We build a scaffold for old and new molecules by identifying all heavy atoms in rings (including double bonds exo to rings) and in linkers between rings.
            We then enumerate valid mappings between these scaffolds
        If there are no valid scaffold mappings, we directly attempt to map the compounds using the factory mapping settings.
        If there is a valid scaffold mapping, we retain any mappings with the top score and use these to generate molecule mappings


        """
        atom_mappings = set()  # all unique valid atom mappings found

        from openff.toolkit.topology import Molecule

        old_offmol = Molecule(old_mol, allow_undefined_stereo=True)
        new_offmol = Molecule(new_mol, allow_undefined_stereo=True)

        if (old_offmol.n_atoms == 0) or (new_offmol.n_atoms == 0):
            raise ValueError(
                f"old_mol ({old_offmol.n_atoms} atoms) and new_mol ({new_offmol.n_atoms} atoms) must both have more than zero atoms"
            )

        # Work in OEMol for the remainder of the function
        # TODO: Replace the rest of this with a mapping strategy based on OpenFF Molecule objects only
        old_oemol = old_offmol.to_openeye()
        new_oemol = new_offmol.to_openeye()

        # Annotate OEMol representations with ring IDs
        # TODO: What is all this doing
        if (not self.external_inttypes) or self.allow_ring_breaking:
            self._assign_ring_ids(old_oemol)
            self._assign_ring_ids(new_oemol)

        # TODO: How do we generate framework in RDKit?
        old_oescaffold = self._get_scaffold(old_oemol)
        new_oescaffold = self._get_scaffold(new_oemol)

        # Assign unique IDs to rings
        self._assign_ring_ids(old_oescaffold, assign_atoms=True, assign_bonds=False)
        self._assign_ring_ids(new_oescaffold, assign_atoms=True, assign_bonds=False)

        # Check arguments
        if (old_oescaffold.NumAtoms() == 0) or (new_oescaffold.NumAtoms() == 0):
            # We can't do anything with empty scaffolds
            _logger.debug(f"One or more scaffolds had no atoms")
            scaffold_maps = list()
        else:
            # Generate scaffold maps
            # TODO: Determine why atom and bond expressions are hard-coded. Should these be flexible instead?
            from openeye import oechem

            scaffold_maps = AtomMapper._get_all_maps(
                old_oescaffold,
                new_oescaffold,
                atom_expr=oechem.OEExprOpts_RingMember | oechem.OEExprOpts_IntType,
                bond_expr=oechem.OEExprOpts_RingMember,
                unique=False,
                matching_criterion=self.matching_criterion,
            )

            _logger.debug(f"Scaffold mapping generated {len(scaffold_maps)} maps")

        if len(scaffold_maps) == 0:
            # There are no scaffold maps, so attempt to generate maps between molecules using the factory parameters
            _logger.warning("Molecules do not appear to share a common scaffold.")
            _logger.warning(
                "Proceeding with direct mapping of molecules, but please check atom mapping and the geometry of the ligands."
            )

            # if no commonality with the scaffold, don't use it.
            # why weren't matching arguments carried to these mapping functions? is there an edge case that i am missing?
            # it still doesn't fix the protein sidechain mapping problem
            generated_atom_mappings = AtomMapper._get_all_maps(
                old_oemol,
                new_oemol,
                atom_expr=self.atom_expr,
                bond_expr=self.bond_expr,
                matching_criterion=self.matching_criterion,
            )
            _logger.debug(
                f"{len(generated_atom_mappings)} mappings were generated by AtomMapper._get_all_maps()"
            )
            for x in generated_atom_mappings:
                _logger.debug(x)

            atom_mappings.update(generated_atom_mappings)

            # TODO: Package maps as AtomMapping objects

        else:
            # Some scaffold mappings have been found, so do something fancy
            # TODO: What exactly is it we're doing?

            # Keep only those scaffold match(es) with maximum score
            # TODO: Will this cause difficulties when trying to stochastically propose maps in both directions,
            # or when we want to retain all maps?
            _logger.debug(
                f"There are {len(scaffold_maps)} scaffold mappings before filtering by score"
            )
            scores = [self.score_mapping(atom_mapping) for atom_mapping in scaffold_maps]
            scaffold_maps = [
                atom_mapping
                for index, atom_mapping in enumerate(scaffold_maps)
                if scores[index] == max(scores)
            ]
            _logger.debug(
                f"There are {len(scaffold_maps)} after filtering to remove lower-scoring scaffold maps"
            )

            # Determine mappings from scaffold to original molecule
            # TODO: Rework this logic to use openff Molecule
            def determine_scaffold_to_molecule_mapping(oescaffold, oemol):
                """Determine mapping of scaffold to full molecule.

                Parameters
                ----------
                oescaffold : openeye.oechem.OEMol
                    The scaffold within the complete molecule
                oemol : openeye.oechem.OEMol
                    The complete molecule

                Returns
                -------
                scaffold_to_molecule_map : dict of int : int
                    scaffold_to_molecule_map[scaffold_atom_index] is the atom index in oemol corresponding to scaffold_atom_index in oescaffold

                """
                scaffold_to_molecule_maps = AtomMapper._get_all_maps(
                    oescaffold,
                    oemol,
                    atom_expr=oechem.OEExprOpts_AtomicNumber,
                    bond_expr=0,
                    matching_criterion=self.matching_criterion,
                )
                _logger.debug(f"{len(scaffold_to_molecule_maps)} scaffold maps found")
                scaffold_to_molecule_map = scaffold_to_molecule_maps[0]
                _logger.debug(f"Scaffold to molecule map: {scaffold_to_molecule_map}")
                assert len(scaffold_to_molecule_map.old_to_new_atom_map) == oescaffold.NumAtoms(), (
                    f"Scaffold should be fully contained within the molecule it came from: map: {scaffold_to_molecule_map}\n{oescaffold.NumAtoms()} atoms in scaffold"
                )
                return scaffold_to_molecule_map

            old_scaffold_to_molecule_map = determine_scaffold_to_molecule_mapping(
                old_oescaffold, old_oemol
            )
            new_scaffold_to_molecule_map = determine_scaffold_to_molecule_mapping(
                new_oescaffold, new_oemol
            )

            # now want to find all of the maps
            # for all of the possible scaffold symmetries
            # TODO: Re-work this algorithm
            for scaffold_map in scaffold_maps:
                if (self.external_inttypes is False) and (self.allow_ring_breaking is True):
                    # reset the IntTypes
                    for oeatom in old_oemol.GetAtoms():
                        oeatom.SetIntType(0)
                    for oeatom in new_oemol.GetAtoms():
                        oeatom.SetIntType(0)

                    # Assign scaffold-mapped atoms in the real molecule an IntType equal to their mapping index
                    old_oeatoms = [atom for atom in old_oemol.GetAtoms()]
                    new_oeatoms = [atom for atom in new_oemol.GetAtoms()]
                    index = 1
                    for (
                        old_scaffold_atom_index,
                        new_scaffold_atom_index,
                    ) in scaffold_map.old_to_new_atom_map.items():
                        old_oeatoms[
                            old_scaffold_to_molecule_map.old_to_new_atom_map[
                                old_scaffold_atom_index
                            ]
                        ].SetIntType(index)
                        new_oeatoms[
                            new_scaffold_to_molecule_map.old_to_new_atom_map[
                                new_scaffold_atom_index
                            ]
                        ].SetIntType(index)
                        index += 1
                    # Assign remaining unmapped atoms in the real molecules an IntType determined by their ring classes
                    self._assign_ring_ids(old_oemol, only_assign_if_zero=True)
                    self._assign_ring_ids(new_oemol, only_assign_if_zero=True)

                atom_mappings_for_this_scaffold_map = AtomMapper._get_all_maps(
                    old_oemol,
                    new_oemol,
                    atom_expr=self.atom_expr,
                    bond_expr=self.bond_expr,
                    matching_criterion=self.matching_criterion,
                )
                atom_mappings.update(atom_mappings_for_this_scaffold_map)

        if not self.allow_ring_breaking:
            # Filter the matches to remove any that allow ring breaking
            _logger.debug(f"Fixing mappings to not create or break rings")
            valid_atom_mappings = set()
            for atom_mapping in atom_mappings:
                try:
                    atom_mapping.render_image("debug.png")
                    atom_mapping.unmap_partially_mapped_cycles()
                    valid_atom_mappings.add(atom_mapping)
                except InvalidMappingException as e:
                    # Atom mapping is no longer valid
                    # pass
                    # TODO: Raising the actual error, do we want to pass it as previously stated?
                    raise e

            atom_mappings = valid_atom_mappings

        # TODO: Should we attempt to preserve chirality here for all atom mappings?
        # Or is this just for biopolymer residues?

        if len(atom_mappings) == 0:
            _logger.warning(
                "No maps found. Try relaxing match criteria or setting allow_ring_breaking to True"
            )
            return None

        # Render set of AtomMapping to a list to return
        return list(atom_mappings)

    def get_best_mapping(self, old_mol, new_mol):
        """Retrieve the best mapping between old and new molecules.

        .. note ::

           This method may generate multiple distinct mappings with the same best score;
           the choice of mapping is returned is ambiguous.

         .. todo ::

           We should figure out how to make the choice deterministic in the case
           multiple mappings have the same score.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mapping : AtomMapping
            Atom mapping with the best score

        Examples
        --------
        Retrieve best-scoring mapping between ethane and ethanol

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])([H:8])')
        >>> ethanol = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[O:8][H:9]')
        >>> atom_mapping = atom_mapper.get_best_mapping(ethane, ethanol)

        """
        import time

        initial_time = time.time()

        import numpy as np

        atom_mappings = self.get_all_mappings(old_mol, new_mol)
        if (atom_mappings is None) or len(atom_mappings) == 0:
            return None

        scores = np.array([self.score_mapping(atom_mapping) for atom_mapping in atom_mappings])
        best_map_index = np.argmax(scores)

        elapsed_time = time.time() - initial_time
        _logger.debug(f"get_best_mapping took {elapsed_time:.3f} s")

        return atom_mappings[best_map_index]

    def get_sampled_mapping(self, old_mol, new_mol):
        """Stochastically generate a mapping between old and new molecules selected proportional to its score.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mapping : AtomMapping
            Atom mapping with the best score

        Examples
        --------
        Sample a mapping stochasticaly between ethane and ethanol

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])([H:8])')
        >>> ethanol = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[O:8][H:9]')
        >>> atom_mapping = atom_mapper.get_sampled_mapping(ethane, ethanol)

        """
        import numpy as np

        atom_mappings = self.get_all_mappings(old_mol, new_mol)
        scores = np.array([self.score_mapping(atom_mapping) for atom_mapping in atom_mappings])
        # Compute normalized probability for sampling from mappings
        p = scores / np.sum(scores)
        # Select mapping with associated this probability
        selected_map_index = np.random.choice(np.arange(len(scores)), p=p)
        # Return the sampled mapping
        return atom_mappings[selected_map_index]

    def propose_mapping(old_mol, new_mol):
        """Propose new mapping stochastically and compute associated forward and reverse probabilities.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mapping : AtomMapping
            Atom mapping with the best score
        logP_forward : float
            log probability of selecting atom_mapping in forward direction
        logP_reverse : float
            log probability of selecting atom_mapping in reverse direction

        Examples
        --------
        Propose a stochastic mapping between ethane and ethanol, computing the probability
        of both the forward mapping and reverse mapping choices:

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])([H:8])')
        >>> ethanol = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[O:8][H:9]')
        >>> atom_mapping, logP_forward, logP_reverse = atom_mapper.propose_mapping(ethane, ethanol)


        """
        # TODO: Stochastically select mapping, then compute forward and reverse log probabilities that same mapping
        #       would be used in forward and reverse directions (for new_mol -> old_mol)

        raise NotImplementedError("This feature has not been implemented yet")

    def score_mapping(self, atom_mapping):
        """Gives a score to each map.

        If molecule positions are available, the inverse of the total Euclidean deviation between heavy atoms is returned.
        If no positions are available, the number of mapped atoms is returned.

        This method can be overridden by subclasses to experiment with different schemes for prioritizing atom maps.

        Parameters
        ----------
        atom_mapping : AtomMapping
            The atom mapping to score

        Returns
        -------
        score : float
            A score for the atom mapping, where larger scores indicate better maps.

        Examples
        --------
        Compute scores for all mappings between ethane and ethanol:

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])([H:8])')
        >>> ethanol = Molecule.from_mapped_smiles('[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[O:8][H:9]')
        >>> atom_mappings = atom_mapper.propose_mapping(ethane, ethanol)
        >>> scores = [ atom_mapper.score_mapping(atom_mapping) for atom_mapping in atom_mappings ]

        """
        # Handle the special case of scoring matches by atom name concordance
        if self.matching_criterion == "name":
            score = sum(
                [
                    atom_mapping.old_mol.atoms[old_atom].name
                    == atom_mapping.new_mol.atoms[new_atom].name
                    for old_atom, new_atom in atom_mapping.old_to_new_atom_map.items()
                ]
            )
            return score

        # Score by positions
        if (
            self.use_positions
            and (atom_mapping.old_mol.conformers is not None)
            and (atom_mapping.new_mol.conformers is not None)
        ):
            # Get all-to-all atom distance matrix
            # TODO: Only compute heavy atom distances

            old_positions = _convert_positions_to_angstroms(atom_mapping.old_mol.conformers[0])
            new_positions = _convert_positions_to_angstroms(atom_mapping.new_mol.conformers[0])

            def dist(a, b):
                """Compute distance between numpy d-vectors a and b.

                Parameters
                ----------
                a, b : numpy.array (d,) arrays
                    Vectors to compute distance between

                Returns
                -------
                distance : float
                    The distance
                """
                import numpy as np

                return np.linalg.norm(b - a)

            # Score mapped heavy atoms using a Gaussian overlap function
            # TODO: Perhaps we should just weight hydrogens different from heavy atoms?
            map_score = 0.0
            for old_atom_index, new_atom_index in atom_mapping.old_to_new_atom_map.items():
                weight = 1.0
                if (atom_mapping.old_mol.atoms[old_atom_index].atomic_number == 1) and (
                    atom_mapping.new_mol.atoms[new_atom_index].atomic_number == 1
                ):
                    weight = 0.0  # hydrogen weight

                nsigma = (
                    dist(old_positions[old_atom_index, :], new_positions[new_atom_index, :])
                    / self.coordinate_tolerance
                )
                map_score += weight * np.exp(-0.5 * nsigma**2)

        else:
            # There are no positions, so compute score derived from mapping
            # This is inspired by the former rank_degenerate_maps code
            # https://github.com/choderalab/perses/blob/412750c457712da1875c7beabfe88b2838f7f197/perses/rjmc/topology_proposal.py#L1123

            old_oeatoms = {
                oeatom.GetIdx(): oeatom for oeatom in atom_mapping.old_mol.to_openeye().GetAtoms()
            }
            new_oeatoms = {
                oeatom.GetIdx(): oeatom for oeatom in atom_mapping.new_mol.to_openeye().GetAtoms()
            }

            # Generate filtered mappings
            old_to_new_atom_map = atom_mapping.old_to_new_atom_map

            mapped_atoms = {
                old_index: new_index for old_index, new_index in old_to_new_atom_map.items()
            }

            mapped_aromatic_atoms = {
                old_index: new_index
                for old_index, new_index in old_to_new_atom_map.items()
                if old_oeatoms[old_index].IsAromatic() and new_oeatoms[new_index].IsAromatic()
            }

            mapped_heavy_atoms = {
                old_index: new_index
                for old_index, new_index in old_to_new_atom_map.items()
                if (old_oeatoms[old_index].GetAtomicNum() > 1)
                and (new_oeatoms[new_index].GetAtomicNum() > 1)
            }

            mapped_ring_atoms = {
                old_index: new_index
                for old_index, new_index in old_to_new_atom_map.items()
                if old_oeatoms[old_index].IsInRing() and new_oeatoms[new_index].IsInRing()
            }

            # These weights are totally arbitrary
            map_score = (
                1.0 * len(mapped_atoms)
                + 0.8 * len(mapped_aromatic_atoms)
                + 0.5 * len(mapped_heavy_atoms)
                + 0.4 * len(mapped_ring_atoms)
            )

        return map_score

    def generate_atom_mapping_from_positions(self, old_mol, new_mol):
        """Generate an atom mapping derived entirely from atom position proximity.

        The resulting map will be cleaned up by de-mapping hydrogens and rings as needed.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mapping : AtomMapping
            The atom mapping determined from positions.
            mapping[molB_index] = molA_index is the mapping of atoms from molA to molB that are geometrically close

        Examples
        --------
        Derive atom mapping from positions:

        >>> atom_mapper = AtomMapper()
        >>> old_mol = Molecule.from_file('old_mol.sdf')
        >>> new_mol = Molecule.from_file('new_mol.sdf')
        >>> atom_mapping = atom_mapper.generate_atom_mapping_from_positions(old_mol, new_mol)

        """
        from openff.toolkit.topology import Molecule

        # Coerce to openff Molecule
        old_mol = Molecule(old_mol, allow_undefined_stereo=True)
        new_mol = Molecule(new_mol, allow_undefined_stereo=True)

        # Check to ensure conformers are defined
        if (old_mol.conformers is None) or (new_mol.conformers is None):
            raise InvalidMappingException(
                f"Both old and new molecules must have at least one conformer defined."
            )

        # Get conformers in common distance unit as numpy arrays
        # NOTE: Matt has a migration guide in openff units README : https://github.com/openforcefield/openff-units#getting-started
        old_mol_positions = _convert_positions_to_angstroms(old_mol.conformers[0])
        new_mol_positions = _convert_positions_to_angstroms(new_mol.conformers[0])

        # TODO: Refactor
        molA_positions = old_mol.to_openeye().GetCoords()  # coordinates (Angstroms)
        molB_positions = new_mol.to_openeye().GetCoords()  # coordinates (Angstroms)
        molB_backward_positions = {val: key for key, val in molB_positions.items()}

        # Define closeness criteria for np.allclose
        rtol = 0.0  # relative tolerane
        atol = _convert_positions_to_angstroms(
            self.coordinate_tolerance
        )  # absolute tolerance (Angstroms)

        # TODO: Can we instead anneal mappings based on closeness in case there is ambiguity?
        old_to_new_atom_map = dict()
        for old_atom_index in range(old_mol.n_atoms):
            # Determine which new atom indices match the old atom
            new_atom_matches = [
                new_atom_index
                for new_atom_index in range(new_mol.n_atoms)
                if np.allclose(
                    old_mol_positions[old_atom_index, :],
                    new_mol_positions[new_atom_index, :],
                    rtol=rtol,
                    atol=atol,
                )
            ]
            if not len(new_atom_matches) in [0, 1]:
                raise InvalidMappingException(
                    f"there are multiple new positions with the same coordinates as old atom {old_atom_index} for coordinate tolerance {self.coordinate_tolerance}"
                )
            if len(new_atom_matches) == 1:
                new_atom_index = new_atom_matches[0]
                old_to_new_atom_map[old_atom_index] = new_atom_index

        atom_mapping = AtomMapping(old_mol, new_mol, old_to_new_atom_map=old_to_new_atom_map)

        # De-map rings if needed
        if not self.allow_ring_breaking:
            atom_mapping.unmap_partially_mapped_cycles()

        return atom_mapping

    @staticmethod
    def _get_all_maps(
        old_oemol,
        new_oemol,
        atom_expr=None,
        bond_expr=None,
        # TODO: Should 'unique' be False by default?
        # See https://docs.eyesopen.com/toolkits/python/oechemtk/patternmatch.html#section-patternmatch-mcss
        unique=True,
        matching_criterion="index",
    ):
        """Generate all possible maps between two oemols

        Parameters
        ----------
        old_oemol : openeye.oechem.OEMol
            old molecule
        new_oemol : openeye.oechem.OEMol
            new molecule
        atom_expr : openeye.oechem.OEExprOpts
            Override for atom matching expression; None if default is to be used.
        bond_expr : openeye.oechem.OEExprOpts
            Override for bond matching expression; None if default is to be used.
        unique : bool, optional, default=True
            Passed to MCSS Match
        matching_criterion : str, optional, default='index'
            The best atom map is pulled based on some ranking criteria;
            if 'index', the best atom map is chosen based on the map with the maximum number of atomic index matches;
            if 'name', the best atom map is chosen based on the map with the maximum number of atom name matches
            else: raise Exception.
            NOTE : the matching criterion pulls patterns and target matches based on indices or names;
                   if 'names' is chosen, it is first asserted that the old and new molecules have atoms that are uniquely named

        Returns
        -------
        atom_mappings : list of AtomMappings
            All unique atom mappings

        .. todo :: Do we need the 'unique' argument here?

        .. todo :: Can we refactor to get rid of this function?

        """
        # TODO: Check out RDKit example: https://greglandrum.github.io/rdkit-blog/posts/2022-06-23-3d-mcs.html
        # Already prepared by openfe: https://github.com/OpenFreeEnergy/openfe/blob/40f018401204c3ae0f8f997e65ee15a30f4fe1bf/openfe/setup/atom_mapping/rdfmcs_mapper.py
        # only works with implicit hydrogens
        # rdFCMS does not like hydrogens because of poor scaling

        # Check arguments
        if (old_oemol.NumAtoms() == 0) or (new_oemol.NumAtoms() == 0):
            raise ValueError(
                f"old_oemol ({old_oemol.NumAtoms()} atoms) and new_oemol ({new_oemol.NumAtoms()} atoms) must both have a nonzero number of atoms"
            )

        import openeye.oechem as oechem

        if atom_expr is None:
            atom_expr = atom_expr
        if bond_expr is None:
            bond_expr = bond_expr

        # this ensures that the hybridization of the oemols is done for correct atom mapping
        oechem.OEAssignHybridization(old_oemol)
        oechem.OEAssignHybridization(new_oemol)
        old_oegraphmol = oechem.OEGraphMol(old_oemol)  # pattern molecule
        new_oegraphmol = oechem.OEGraphMol(new_oemol)  # target molecule

        mcs = oechem.OEMCSSearch(oechem.OEMCSType_Approximate)
        mcs.Init(old_oegraphmol, atom_expr, bond_expr)
        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        matches = [m for m in mcs.Match(new_oegraphmol, unique)]
        _logger.debug(f"all matches have atom counts of : {[m.NumAtoms() for m in matches]}")

        atom_mappings = set()
        for match in matches:
            try:
                atom_mapping = AtomMapper._create_atom_mapping(
                    old_oemol, new_oemol, match, matching_criterion
                )
                atom_mappings.add(atom_mapping)
            except InvalidMappingException as e:
                # Mapping is not valid; skip it
                pass

        # Render to a list to return mappings
        return list(atom_mappings)

    @staticmethod
    def _create_pattern_to_target_map(old_oemol, new_oemol, match, matching_criterion="index"):
        """
        Create a dict of {pattern_atom: target_atom}

        Parameters
        ----------
        old_oemol : openeye.oechem.OEMol
            old molecule
        new_oemol : openeye.oechem.OEMol
            new molecule
        match : oechem.OEMCSSearch.Match iterable
            entry in oechem.OEMCSSearch.Match object
        matching_criterion : str, optional, default='index'
            whether the pattern to target map is chosen based on atom indices or names (which should be uniquely defined)
            allowables: ['index', 'name']

        Returns
        -------
        pattern_to_target_map : dict of OEAtom : OEAtom
            {pattern_atom: target_atom}

        """
        if matching_criterion == "index":
            pattern_atoms = {atom.GetIdx(): atom for atom in old_oemol.GetAtoms()}
            target_atoms = {atom.GetIdx(): atom for atom in new_oemol.GetAtoms()}
            pattern_to_target_map = {
                pattern_atoms[matchpair.pattern.GetIdx()]: target_atoms[matchpair.target.GetIdx()]
                for matchpair in match.GetAtoms()
            }
        elif matching_criterion == "name":
            pattern_atoms = {atom.GetName(): atom for atom in old_oemol.GetAtoms()}
            target_atoms = {atom.GetName(): atom for atom in new_oemol.GetAtoms()}
            pattern_to_target_map = {
                pattern_atoms[matchpair.pattern.GetName()]: target_atoms[matchpair.target.GetName()]
                for matchpair in match.GetAtoms()
            }
        else:
            raise Exception(f"matching criterion {matching_criterion} is not currently supported")

        return pattern_to_target_map

    @staticmethod
    def _create_atom_mapping(old_oemol, new_oemol, match, matching_criterion):
        """
        Returns an AtomMapping that omits hydrogen-to-nonhydrogen atom maps
        as well as any X-H to Y-H where element(X) != element(Y) or aromatic(X) != aromatic(Y)

        Parameters
        ----------
        old_oemol : openeye.oechem.OEMol object
            The old molecules
        new_oemol : openeye.oechem.OEMol object
            The new molecule
        match : openeye.oechem.OEMatchBase iterable
            entry in oechem.OEMCSSearch.Match object
        matching_criterion : str
            Matching criterion for _create_pattern_to_target_map.
            whether the pattern to target map is chosen based on atom indices or names (which should be uniquely defined)
            allowables: ['index', 'name']

        Returns
        -------
        atom_mapping : AtomMapping
            The atom mapping

        """
        # TODO : Eliminate this entirely in favor of using OpenFF Molecule objects once we can avoid needing to represent partial molecules for cores
        new_to_old_atom_map = dict()
        pattern_to_target_map = AtomMapper._create_pattern_to_target_map(
            old_oemol, new_oemol, match, matching_criterion
        )
        for pattern_oeatom, target_oeatom in pattern_to_target_map.items():
            old_index, new_index = pattern_oeatom.GetIdx(), target_oeatom.GetIdx()
            old_oeatom, new_oeatom = pattern_oeatom, target_oeatom

            # Check if a hydrogen was mapped to a non-hydrogen (basically the xor of is_h_a and is_h_b)
            if (old_oeatom.GetAtomicNum() == 1) != (new_oeatom.GetAtomicNum() == 1):
                continue

            # Check if X-H to Y-H changes where element(X) != element(Y) or aromatic(X) != aromatic(Y)
            if (
                (old_oeatom.GetAtomicNum() == 1)
                and (new_oeatom.GetAtomicNum() == 1)
                # Handle the weird special case where the molecule is just one hydrogen atom
                # (which is an abuse that occurs in the current biopolymer logic)
                # TODO: Fix this when we overhaul biopolymer logic
                and (old_oeatom.GetDegree() > 0)
                and (new_oeatom.GetDegree() == 1)
            ):
                X = [bond.GetNbr(old_oeatom) for bond in old_oeatom.GetBonds()][0]
                Y = [bond.GetNbr(new_oeatom) for bond in new_oeatom.GetBonds()][0]
                if (X.GetAtomicNum() != Y.GetAtomicNum()) or (X.IsAromatic() != Y.IsAromatic()):
                    continue

            new_to_old_atom_map[new_index] = old_index

        return AtomMapping(old_oemol, new_oemol, new_to_old_atom_map=new_to_old_atom_map)

    @staticmethod
    def _assign_ring_ids(
        oemol, max_ring_size=10, assign_atoms=True, assign_bonds=True, only_assign_if_zero=False
    ):
        """Encode the sizes of rings each atom belongs to in a bitstring and assign it to the atom's Int field.

        Parameters
        ----------
        oemol : openeye.oechem.OEMol
            oemol to assign atom Int field for. This will be modified.
        assign_atoms : bool, optional, default=True
            If True, assign atoms
        assign_bonds : bool, optional, default=True
            If True, assign bonds
        max_ring_size : int, optional, default=10
            Largest ring size that will be checked for
        only_assign_if_zero : bool, optional, default=False
            If True, will only assign atom IntTypes to atoms and bonds with non-zero IntType;
            bond IntTypes will not be assigned

        """

        def _assign_ring_id(oeobj, max_ring_size=10):
            import openeye.oechem as oechem

            """Returns the int type based on the ring occupancy of the atom or bond

            Parameters
            ----------
            oeobj : openeye.oechem.OEAtomBase or openeye.oechem.OEBondBase
                atom or bond to compute ring membership integer for
            max_ring_size : int, optional, default=10
                Largest ring size that will be checked for

            Returns
            -------
            ring_as_base_two : int
                Integer encoding binary ring membership for atom or bond
            """
            import openeye.oechem as oechem

            if hasattr(oeobj, "GetAtomicNum"):
                fun = oechem.OEAtomIsInRingSize
            elif hasattr(oeobj, "GetOrder"):
                fun = oechem.OEBondIsInRingSize
            else:
                raise ValueError(f"Argument {oeobj} is not an OEAtom or OEBond")

            rings = ""
            for i in range(3, max_ring_size + 1):  #  smallest feasible ring size is 3
                rings += str(int(fun(oeobj, i)))
            ring_as_base_two = int(rings, 2)
            return ring_as_base_two

        if assign_atoms:
            for oeatom in oemol.GetAtoms():
                if only_assign_if_zero and oeatom.GetIntType() != 0:
                    continue
                oeatom.SetIntType(_assign_ring_id(oeatom, max_ring_size=max_ring_size))

        if assign_bonds:
            for oebond in oemol.GetBonds():
                oebond.SetIntType(_assign_ring_id(oebond, max_ring_size=max_ring_size))

    @staticmethod
    def _get_scaffold(oemol, adjustHcount=False):
        """
        Takes an openeye.oechem.oemol and return an openeye.oechem.OEMol of the scaffold.

        The scaffold is a molecule where all atoms that are not in rings or in linkers between rings have been removed.
        Double bonded atoms exo to a ring are retained as scaffold atoms.

        This function has been completely taken from openeye's extractscaffold.py script
        https://docs.eyesopen.com/toolkits/python/oechemtk/oechem_examples/oechem_example_extractscaffold.html#section-example-oechem-extractscaffold

        Parameters
        ----------
        oemol : openeye.oechem.OEMol
            molecule for which scaffold is to be retrieved
        adjustHcount : bool, default=False
            add/remove hydrogens to satisfy valence of scaffold


        Returns
        -------
        oescaffold : openeye.oechem.OEMol
            new molecule representing the scaffold of oemol
        """
        from openeye import oechem

        # Make a copy so as not to modify original oemol
        oemol = oechem.OEMol(oemol)

        def TraverseForRing(visited, atom):
            visited.add(atom.GetIdx())

            for nbor in atom.GetAtoms():
                if nbor.GetIdx() not in visited:
                    if nbor.IsInRing():
                        return True

                    if TraverseForRing(visited, nbor):
                        return True

            return False

        def DepthFirstSearchForRing(root, nbor):
            visited = set()
            visited.add(root.GetIdx())

            return TraverseForRing(visited, nbor)

        class IsInScaffold(oechem.OEUnaryAtomPred):
            def __call__(self, atom):
                if atom.IsInRing():
                    return True

                count = 0
                for nbor in atom.GetAtoms():
                    if DepthFirstSearchForRing(atom, nbor):
                        count += 1

                return count > 1

        oescaffold = oechem.OEMol()
        pred = IsInScaffold()

        oechem.OESubsetMol(oescaffold, oemol, pred, adjustHcount)

        return oescaffold


def _convert_opemol_to_offmol(oemol, allow_undefined_stereo: bool = False, _cls=None):
    """
    Create a Molecule from an OpenEye molecule. If the OpenEye molecule has
    implicit hydrogens, this function will make them explicit.

    NOTE: This was copied and modified from the openff-toolkit on 2023-03-23 in order to support molecule cores in atom mapping.
    https://github.com/openforcefield/openff-toolkit/blob/main/openff/toolkit/utils/openeye_wrapper.py#L1061

    WARNING: This is dangerous and risks both the resulting Molecule behaving poorly and code drift away from openff-toolkit.
    TODO: Eliminate this code as soon as possible when we migrate to a new simpler mapping strategy.

    ``OEAtom`` s have a different set of allowed value for partial charges than
    ``openff.toolkit.topology.Molecule`` s. In the OpenEye toolkits, partial charges
    are stored on individual ``OEAtom`` s, and their values are initialized to ``0.0``.
    In the Open Force Field Toolkit, an ``openff.toolkit.topology.Molecule``'s
    ``partial_charges`` attribute is initialized to ``None`` and can be set to a
    unit-wrapped numpy array with units of
    elementary charge. The Open Force
    Field Toolkit considers an ``OEMol`` where every ``OEAtom`` has a partial
    charge of ``float('nan')`` to be equivalent to an Open Force Field Toolkit `Molecule`'s
    ``partial_charges = None``.
    This assumption is made in both ``to_openeye`` and ``from_openeye``.
    .. warning :: This API is experimental and subject to change.
    Parameters
    ----------
    oemol : openeye.oechem.OEMol
        An OpenEye molecule
    allow_undefined_stereo : bool, default=False
        If false, raises an exception if oemol contains undefined stereochemistry.
    _cls : class
        Molecule constructor
    Returns
    -------
    molecule : openff.toolkit.topology.Molecule
        An OpenFF molecule
    Examples
    --------
    Create a Molecule from an OpenEye OEMol
    >>> from openeye import oechem
    >>> from openff.toolkit.tests.utils import get_data_file_path
    >>> ifs = oechem.oemolistream(get_data_file_path('systems/monomers/ethanol.mol2'))
    >>> oemols = list(ifs.GetOEGraphMols())
    >>> toolkit_wrapper = OpenEyeToolkitWrapper()
    >>> molecule = toolkit_wrapper.from_openeye(oemols[0])
    """
    import math

    from openeye import oechem
    from openff.toolkit.utils import UndefinedStereochemistryError

    oemol = oechem.OEMol(oemol)

    # Add explicit hydrogens if they're implicit
    if oechem.OEHasImplicitHydrogens(oemol):
        oechem.OEAddExplicitHydrogens(oemol)

    # TODO: Is there any risk to perceiving aromaticity here instead of later?
    oechem.OEAssignAromaticFlags(oemol, oechem.OEAroModel_MDL)

    oechem.OEPerceiveChiral(oemol)

    # Check that all stereo is specified
    # Potentially better OE stereo check: OEFlipper — Toolkits - - Python
    # https: // docs.eyesopen.com / toolkits / python / omegatk / OEConfGenFunctions / OEFlipper.html

    unspec_chiral = False
    unspec_db = False
    problematic_atoms = list()
    problematic_bonds = list()

    for oeatom in oemol.GetAtoms():
        if oeatom.IsChiral():
            if not (oeatom.HasStereoSpecified()):
                unspec_chiral = True
                problematic_atoms.append(oeatom)
    for oebond in oemol.GetBonds():
        if oebond.IsChiral():
            if not (oebond.HasStereoSpecified()):
                unspec_db = True
                problematic_bonds.append(oebond)
    if unspec_chiral or unspec_db:

        def oeatom_to_str(oeatom) -> str:
            return "atomic num: {}, name: {}, idx: {}, aromatic: {}, chiral: {}".format(
                oeatom.GetAtomicNum(),
                oeatom.GetName(),
                oeatom.GetIdx(),
                oeatom.IsAromatic(),
                oeatom.IsChiral(),
            )

        def oebond_to_str(oebond) -> str:
            return f"order: {oebond.GetOrder()}, chiral: {oebond.IsChiral()}"

        def describe_oeatom(oeatom) -> str:
            description = f"Atom {oeatom_to_str(oeatom)} with bonds:"
            for oebond in oeatom.GetBonds():
                description += "\nbond {} to atom {}".format(
                    oebond_to_str(oebond), oeatom_to_str(oebond.GetNbr(oeatom))
                )
            return description

        msg = "OEMol has unspecified stereochemistry. oemol.GetTitle(): {}\n".format(
            oemol.GetTitle()
        )
        if len(problematic_atoms) != 0:
            msg += "Problematic atoms are:\n"
            for problematic_atom in problematic_atoms:
                msg += describe_oeatom(problematic_atom) + "\n"
        if len(problematic_bonds) != 0:
            msg += f"Problematic bonds are: {problematic_bonds}\n"
        if allow_undefined_stereo:
            msg = "Warning (not error because allow_undefined_stereo=True): " + msg
            _logger.warning(msg)
        else:
            msg = "Unable to make OFFMol from OEMol: " + msg
            raise UndefinedStereochemistryError(msg)

    if _cls is None:
        from openff.toolkit.topology.molecule import Molecule

        _cls = Molecule

    molecule = _cls()
    molecule.name = oemol.GetTitle()

    # Copy any attached SD tag information
    for dp in oechem.OEGetSDDataPairs(oemol):
        molecule._properties[dp.GetTag()] = dp.GetValue()

    off_to_oe_idx = dict()  # {oemol_idx: molecule_idx}
    atom_mapping = {}
    for oeatom in oemol.GetAtoms():
        oe_idx = oeatom.GetIdx()
        map_id = oeatom.GetMapIdx()
        atomic_number = oeatom.GetAtomicNum()
        # Carry with implicit units of elementary charge for faster route through _add_atom
        formal_charge = oeatom.GetFormalCharge()
        explicit_valence = oeatom.GetExplicitValence()
        # Implicit hydrogens are never added to D- and F- block elements,
        # and the MDL valence is always the explicit valence for these
        # elements, so this does not count radical electrons in these blocks.
        mdl_valence = oechem.OEMDLGetValence(atomic_number, formal_charge, explicit_valence)
        number_radical_electrons = mdl_valence - (oeatom.GetImplicitHCount() + explicit_valence)

        # WARNING: This has been intentionally commented out to support cores (molecule subsets)
        # if number_radical_electrons > 0:
        #    raise RadicalsNotSupportedError(
        #        "The OpenFF Toolkit does not currently support parsing molecules with radicals. "
        #        f"Found {number_radical_electrons} radical electrons on molecule "
        #        f"{oechem.OECreateSmiString(oemol)}."
        #    )

        is_aromatic = oeatom.IsAromatic()
        from openff.toolkit.utils import OpenEyeToolkitWrapper

        stereochemistry = OpenEyeToolkitWrapper._openeye_cip_atom_stereochemistry(oemol, oeatom)
        # stereochemistry = self._openeye_cip_atom_stereochemistry(oemol, oeatom)
        name = oeatom.GetName()

        # Transfer in hierarchy metadata
        metadata_dict = dict()
        if oechem.OEHasResidues(oemol):
            metadata_dict["residue_name"] = oechem.OEAtomGetResidue(oeatom).GetName()
            metadata_dict["residue_number"] = oechem.OEAtomGetResidue(oeatom).GetResidueNumber()
            metadata_dict["insertion_code"] = oechem.OEAtomGetResidue(oeatom).GetInsertCode()
            metadata_dict["chain_id"] = oechem.OEAtomGetResidue(oeatom).GetChainID()
        # print('from', metadata_dict)

        atom_index = molecule._add_atom(
            atomic_number,
            formal_charge,
            is_aromatic,
            stereochemistry=stereochemistry,
            name=name,
            metadata=metadata_dict,
            invalidate_cache=False,
        )
        off_to_oe_idx[oe_idx] = (
            atom_index  # store for mapping oeatom to molecule atom indices below
        )
        atom_mapping[atom_index] = map_id

    molecule._invalidate_cached_properties()

    # If we have a full / partial atom map add it to the molecule. Zeroes 0
    # indicates no mapping
    if {*atom_mapping.values()} != {0}:
        molecule._properties["atom_map"] = {
            idx: map_idx for idx, map_idx in atom_mapping.items() if map_idx != 0
        }

    for oebond in oemol.GetBonds():
        atom1_index = off_to_oe_idx[oebond.GetBgnIdx()]
        atom2_index = off_to_oe_idx[oebond.GetEndIdx()]
        bond_order = oebond.GetOrder()
        is_aromatic = oebond.IsAromatic()
        stereochemistry = OpenEyeToolkitWrapper._openeye_cip_bond_stereochemistry(oemol, oebond)
        if oebond.HasData("fractional_bond_order"):
            fractional_bond_order = oebond.GetData("fractional_bond_order")
        else:
            fractional_bond_order = None

        molecule._add_bond(
            atom1_index,
            atom2_index,
            bond_order,
            is_aromatic=is_aromatic,
            stereochemistry=stereochemistry,
            fractional_bond_order=fractional_bond_order,
            invalidate_cache=False,
        )

    molecule._invalidate_cached_properties()

    # TODO: Copy conformations, if present
    # TODO: Come up with some scheme to know when to import coordinates
    # From SMILES: no
    # From MOL2: maybe
    # From other: maybe
    if hasattr(oemol, "GetConfs"):
        for conf in oemol.GetConfs():
            n_atoms = molecule.n_atoms
            # Store with implicit units until we're sure this conformer exists
            positions = np.zeros(shape=[n_atoms, 3], dtype=np.float64)
            for oe_id in conf.GetCoords().keys():
                # implicitly in angstrom
                off_atom_coords = conf.GetCoords()[oe_id]
                off_atom_index = off_to_oe_idx[oe_id]
                positions[off_atom_index, :] = off_atom_coords
            all_zeros = not np.any(positions)
            if all_zeros and n_atoms > 1:
                continue
            molecule._add_conformer(unit.Quantity(positions, unit.angstrom))

    # Store charges with implicit units in this scope
    unitless_charges = np.zeros(shape=molecule.n_atoms, dtype=np.float64)

    # If all OEAtoms have a partial charge of NaN, then the OFFMol should
    # have its partial_charges attribute set to None
    any_partial_charge_is_not_nan = False
    for oe_atom in oemol.GetAtoms():
        oe_idx = oe_atom.GetIdx()
        off_idx = off_to_oe_idx[oe_idx]
        unitless_charge = oe_atom.GetPartialCharge()
        # Once this is True, skip the isnancheck
        if not any_partial_charge_is_not_nan:
            if not math.isnan(unitless_charge):
                any_partial_charge_is_not_nan = True
        unitless_charges[off_idx] = unitless_charge

    if any_partial_charge_is_not_nan:
        molecule.partial_charges = unit.Quantity(unitless_charges, unit.elementary_charge)
    else:
        molecule.partial_charges = None

    return molecule


def _convert_positions_to_angstroms(positions):
    """Convert an openmm or pint Quantity wrapped object in distance units to unitless object in Angstroms

    Parameters
    ----------
    positions : openmm or pint Quantity wrapped object with units of distance
        The positions array or distance to convert

    Returns
    -------
    unwrapped : object
        The object (e.g. numpy array or float) in Angstroms

    """
    if hasattr(positions, "to"):
        # pint Quantity
        return positions.to("angstroms").magnitude
    elif hasattr(positions, "value_in_unit"):
        # openmm Quantity
        from openmm import unit

        return positions / unit.angstroms

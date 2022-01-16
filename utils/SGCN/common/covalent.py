import numpy as np

from Bio.PDB import NeighborSearch, PDBParser
from utils.SGCN.common import format

SEARCH_RADIUS = 6
KMIN_DISTANCE_BETWEEN_ATOMS = 0.01
WATER_RESIDUES = {'HOH', 'HHO', 'OHH', 'H2O', 'OH2', 'WAT', 'TIP', 'TIP3', 'TIP4', 'TIP3P', 'TIP4P', 'SOL'}


def get_elements_radii(elements_radii_path):
    return dict(map(
        lambda l: (l.strip().split(': ')[0], float(l.strip().split(': ')[1])),
        open(elements_radii_path).readlines()))


def get_chain_id(atom):
    return atom.get_parent().get_parent()._id


def get_residue_number(atom):
    return atom.get_parent()._id[1]


def get_residue_name(atom):
    return atom.get_parent().resname


def has_water_residue(atom):
    return get_residue_name(atom) in WATER_RESIDUES


def is_hydrogen(atom):
    return atom.element == 'H'


def have_covalent_bond(first, second, elements_radii):
    distance_2 = np.linalg.norm(first.get_coord() - second.get_coord(), ord=2) ** 2

    if has_water_residue(first) and has_water_residue(second):
        return distance_2 < 1.21

    # then, we don't connect water with anything else
    if has_water_residue(first) ^ has_water_residue(second):
        return False

    # another rule, we don't connect things from different chains
    if get_chain_id(first) != get_chain_id(second):
        return False

    if is_hydrogen(first) and is_hydrogen(second):
        return False

    if (is_hydrogen(first) or is_hydrogen(second)) and distance_2 >= 1.21:
        return False

    # we don't create S-S bridges
    if first.name.startswith('SG') and second.name.startswith('SG'):
        return False

    # we don't connect things that are far in the sequence
    if abs(get_residue_number(first) - get_residue_number(second)) > 1:
        return False

    # here we don't want to have a connection between two "H" in a water
    # but there are some forcefields where it can be possible

    distance_max_2 = (elements_radii.get(first.element, -1) + elements_radii.get(second.element, -1)) * 0.6
    distance_max_2 *= distance_max_2

    # we don't consider pairs which are too close also because that could be a mistake
    return (distance_2 <= distance_max_2) and (distance_2 > KMIN_DISTANCE_BETWEEN_ATOMS)


def get_covalent_bonds(model_path, elements_radii_path, atom_id_to_pos, model_name):
    elements_radii = get_elements_radii(elements_radii_path)
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure(model_name, model_path)
    atoms = [atom for atom in structure.get_atoms()]
    neighbors_searcher = NeighborSearch(atoms)
    pairs = neighbors_searcher.search_all(radius=SEARCH_RADIUS)
    covalent_bonds = []
    for atom_1, atom_2 in pairs:
        if have_covalent_bond(atom_1, atom_2, elements_radii):
            atom_id_1 = format.create_atom_id_raw(
                get_chain_id(atom_1),
                get_residue_number(atom_1),
                get_residue_name(atom_1),
                atom_1.name)
            atom_id_2 = format.create_atom_id_raw(
                get_chain_id(atom_2),
                get_residue_number(atom_2),
                get_residue_name(atom_2),
                atom_2.name)
            covalent_bonds.append([atom_id_to_pos[atom_id_1], atom_id_to_pos[atom_id_2]])
    return covalent_bonds

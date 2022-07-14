from collections import defaultdict
from importlib.resources import path
from pathlib import Path
from typing import Tuple, List

from Bio import PDB
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB import PDBIO 
from Bio.PDB import Chain, PDBParser, Structure, Select
from Bio.PDB.Polypeptide import three_to_one, standard_aa_names

import numpy as np
import os
from tempfile import TemporaryDirectory

class ClearTreash(Select):
    def accept_residue(self, residue):
        return residue._id[0] == " "

    
def get_acsepted_indexes(initial_structure: Structure, indexes: Tuple[int, int]) -> None:
    start, end = indexes
    position = 0

    acsepted = set()
    for protein in initial_structure:
        for chain in protein:
            for amino in chain:
                if position >= start and position < end:
                    acsepted.add(amino._id)
                position += 1
    return acsepted

    
class SelectSubsequence(Select):
    def __init__(self, accepted: set):
        self.accepted = accepted
        
    def accept_residue(self, residue):
        return residue._id in self.accepted


def get_seq(structure: Structure) -> str:
    seq = ""
    for protein in structure:
        for chain in protein:
            for amino in chain:
                if amino.resname in standard_aa_names:
                    seq += three_to_one(amino.resname)

    return seq


def compare(resolved: Structure, model: Structure) -> Tuple[bool, str, str]:

    res_seq = get_seq(resolved)
    mod_seq = get_seq(model)

    return res_seq == mod_seq, res_seq, mod_seq


def find_common_subsequence(res_seq: str, model_seq: str) -> [int, int, int]:
    len_res, len_mod = len(res_seq), len(model_seq)
    dynamic = np.zeros((len_res + 1, len_mod + 1))

    for i in range(len_res):
        for j in range(len_mod):
            if res_seq[i] == model_seq[j]:
                dynamic[i + 1][j + 1] = dynamic[i][j] + 1

    i, j = np.unravel_index(np.argmax(dynamic), dynamic.shape)
    length = int(np.max(dynamic))

    return i - length, j - length, length


def reindex_structure(structure: Structure) -> None:
    index = 0
    for model in structure:
        for caine in model:
            for residue in caine:
                residue._id = (" ", index, " ")
                index += 1

                    
def cut_structure(initial_structure: Structure, indexes: Tuple[int, int]) -> None:
    start, end = indexes
    position = 0

    to_delete = defaultdict(list)
    for protein in initial_structure:
        for chain in protein:
            for amino in chain:
                if position < start or position >= end:
                    to_delete[chain.full_id].append(amino._id)
                position += 1

    for protein in initial_structure:
        for chain in protein:
            for res_index in to_delete[chain.full_id]:
                try:
                    chain.detach_child(res_index)
                except KeyError:
                    continue


def get_chain_number(structure: Structure) -> int:
    return len(list(structure.get_chains()))


def get_clear_structures(struct: Structure) -> PDB.Structure:
    io = PDBIO()
    parser = PDB.PDBParser(QUIET=True)
    
    with TemporaryDirectory() as tmpdirname:
        io.set_structure(struct)
        io.save(tmpdirname + "/clear.pdb", ClearTreash())
        struct = parser.get_structure("", tmpdirname + "/clear.pdb")
        
        reindex_structure(struct)

    return struct

# Add same in outputs
#
def get_consistent_structures(left_struct: Structure, right_struct: Structure, name: str) -> Tuple:
    
    same, left_seq, right_seq = compare(left_struct, right_struct)
    start_in_left, start_in_right, length = find_common_subsequence(
        left_seq,
        right_seq
    )

    if same:
        reindex_structure(left_struct)
        reindex_structure(right_struct)
        return left_struct, right_struct, same
    else:    
        io = PDBIO()
        parser = PDB.PDBParser(QUIET=True)
    
        acsepted_in_left = get_acsepted_indexes(
                left_struct, 
                (start_in_left, start_in_left + length)
            )
        acsepted_in_right = get_acsepted_indexes(
                right_struct, 
                (start_in_right, start_in_right + length)
            )

        path_to_tmp = Path(f"tmp_struct/{name}")
        if not os.path.isdir(path_to_tmp):
            os.makedirs(path_to_tmp)
        io.set_structure(left_struct)
        io.save(f"{path_to_tmp / f'{name}_consistent_left.pdb'}", SelectSubsequence(acsepted_in_left))
        io.set_structure(right_struct)
        io.save(f"{path_to_tmp / f'{name}_consistent_right.pdb'}", SelectSubsequence(acsepted_in_right))

        left_struct = parser.get_structure("", f"{path_to_tmp / f'{name}_consistent_left.pdb'}")
        right_struct = parser.get_structure("", f"{path_to_tmp / f'{name}_consistent_right.pdb'}")

        reindex_structure(left_struct)
        reindex_structure(right_struct)

        return left_struct, right_struct, same

def initialize_structure_by_chain(chain: Chain.Chain) -> Structure:
    new_struct = StructureBuilder()
    new_struct.init_structure("")
    new_struct.init_model("")
    new_struct = new_struct.get_structure()
    new_struct[""].add(chain)
    return new_struct

def initialize_structure_by_two_chains(chain_one: Chain.Chain, chain_two: Chain.Chain) -> Structure:
    new_struct = StructureBuilder()
    new_struct.init_structure("")
    new_struct.init_model("")
    new_struct = new_struct.get_structure()
    new_struct[""].add(chain_one)
    new_struct[""].add(chain_two)
    return new_struct
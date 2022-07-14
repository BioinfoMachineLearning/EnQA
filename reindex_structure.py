import os
import logging
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import Structure
from Bio.PDB import PDBIO 
from pathlib import Path
from typing import Optional, List
from pdb_utils import get_consistent_structures, reindex_structure



PATH_TO_ANBASE = Path('../anbase/data') # path to anbase
PATH_TO_PRED = Path('example/model') # path to AF2 predictions in EnQA
PATH_TO_REINDEX_ANBASE = Path('reindex_anbase') # path to reindexed anbase structures ПРИБИТО ГВОЗДЯМИ
PATH_TO_REINDEX_PRED = Path('reindex_pred') # path to reindexed predictions



def get_structures_names(path: Path) -> List:
    """
    Get structures name for lddt calculation 
    @param path: path to structures
    @return: list of structures names
    """
    return os.listdir(path)

def read_structure(name: str, path: Path) -> Structure.Structure:
    """
    Read structure from PDB file.
    @param name: structure id
    @param path: path to PDB file
    """
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure(get_id(name), path)

def get_id(name: str) -> str:
    """
    Get id from structure name without chains
    @param name: structure id, name from anbase structure names (with chains)
    @return: structure id without chains
    """
    return name.split('_')[0] 

def get_structure_anbase(name: str, path: Path) -> Optional[Structure.Structure]:
    """
    Get anbase structure from .pdb file. Get unbound structure from 'prepared_schrod/0'.
    @param name: structure id, name from anbase structure names (with chains)
    @return: strcuture from anbase
    """
    # Unbound antibodies
    SUFFIX_ANBASE = '_ab_u.pdb'

    if not os.path.isdir(path / name):
        logging.warning(f"Not exist: {path / name}")
    else:
        path_to_structure = path / name / 'prepared_schrod/0'
        if not os.path.isdir(path_to_structure):
            logging.warning(f"Not exist: {path_to_structure}")
        else: 
            filename = get_id(name) + SUFFIX_ANBASE
            return read_structure(name, path_to_structure / filename)
    return None

def save_reindexed_structure(name: str, path: Path, structure: Structure.Structure, index = None) -> None:
    """
    Save reindex structure to path.
    @param name: structure name
    @param path: save path
    @param structure: reindex structure
    @return: None 
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    if index:
        filename = name + '_' + index + '.pdb'
    else:
        filename = name + '.pdb'
    io = PDBIO()
    io.set_structure(structure)
    io.save(f"{path / filename}")





if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', filename='logging_reindex.txt', level=logging.DEBUG)

    structure_names = get_structures_names(PATH_TO_PRED)
    for name in tqdm(structure_names):
        logging.info(f'Structure name: {name}')

        structure = get_structure_anbase(name, PATH_TO_ANBASE)
        path_to_pred = PATH_TO_PRED / name
        flag_re = True
        for file in os.listdir(path_to_pred):
            pred = read_structure(name, path_to_pred / file)
            try:
                left_struct, right_struct, same = get_consistent_structures(left_struct=structure, right_struct=pred, name=name)
                logging.info(f'For structure {name} sequences are {same}!')
                if flag_re:
                    save_reindexed_structure(name=name, path=PATH_TO_REINDEX_ANBASE / name, structure=left_struct)
                    flag_re = False
                index = str(file).split('.')[0]
                save_reindexed_structure(name=name, path=PATH_TO_REINDEX_PRED / name, structure=right_struct, index=index)
            except Exception as e:
                logging.exception(e)
                break
            
        
        
    




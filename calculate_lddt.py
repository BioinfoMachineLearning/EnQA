import os
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List

PATH_TO_LDDT = Path('lddt-linux/lddt') # path to lddt
PATH_TO_TARGET = Path('reindex_anbase') # path to reindexed anbase structures ПРИБИТО ГВОЗДЯМИ
PATH_TO_PRED = Path('reindex_pred') # path to reindexed predictions
PATH_TO_SAVE_LDDT = Path('lddt_results')


def get_structures_names(path: Path) -> List:
    """
    Get structures name for lddt calculation 
    @param path: path to structures
    @return: list of structures names
    """
    return os.listdir(path)


def calculate_save_lddt(name: str, path_to_lddt: Path, target_struct: str, pred_struct: str, path_to_save: Path, index: str) -> None:
    """
    Calculate lddt and save to file.
    @param name: structure name
    @param path_to_lddt: path to lddt script
    @param target_struct: target structure
    @param pred_struct: predicted structure
    @param path_to_save: where to save lddt
    @return: None
    """
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)
    filename = name + '_' + index + '.csv'
    try:
        os.system(f"{path_to_lddt} {target_struct} {pred_struct} >> {path_to_save / filename}")
    except Exception as e:
        logging.exception(e)




if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', filename='logging_lddt.txt', level=logging.DEBUG)

    structure_names = get_structures_names(PATH_TO_TARGET)
    for name in tqdm(structure_names):
        logging.info(f"Structure name: {name}")

        path_to_target = PATH_TO_TARGET / name
        path_to_pred = PATH_TO_PRED / name
        target_struct = path_to_target / f"{name}.pdb"

        index = 1
        for file in os.listdir(path_to_pred):
            pred_struct = path_to_pred / file
            calculate_save_lddt(name=name, path_to_lddt=PATH_TO_LDDT, target_struct=target_struct, 
                            pred_struct=pred_struct, path_to_save=PATH_TO_SAVE_LDDT / name, index=str(index))
            index += 1

        



    



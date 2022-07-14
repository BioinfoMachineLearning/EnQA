import os
import shutil
import logging
from tqdm import tqdm

# path to volume with AF2 predictions 
PATH_TO_ANT = '/mnt/volume_ant/'
NUMBERS_OF_PRED = 5

def copy_af2_predictions(path_from: str, path_to: str) -> None:
    """
    Copy AF2 predictions in EnQA project
    @param path_from: path from the volume with AF2 predictions, copy predictions from this folder
    @param path_to: path from EnQA project, copy predictions to this folder
    @return: None
    """
    number_of_pred = 5
    for numb in range(1, number_of_pred + 1):
        shutil.copy2(os.path.join(path_from, 'relaxed_model_' + str(numb) + '.pdb'), path_to)
        shutil.copy2(os.path.join(path_from, 'result_model_' + str(numb) + '.pkl'), path_to)

def copy_af2_pred_model(path_from: str, path_to: str) -> None:
    """
    Copy AF2 predictions (.pdb) for lddt prediction
    @param path_from: path from the volume with AF2 predictions, copy predictions from this folder
    @param path_to: path from EnQA project, copy predictions to this folder
    @return: None
    """
    number_of_pred = 5
    for numb in range(1, number_of_pred + 1):
        shutil.copy2(os.path.join(path_from, 'relaxed_model_' + str(numb) + '.pdb'), path_to)

if __name__ == '__main__':

    # TODO: save structure id to individual file

    logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', filename='logging_enqa_anbase.txt', level=logging.DEBUG)

    structure_names = os.listdir(PATH_TO_ANT)
    # use 1 chain structures, because with 2 and more EnQA throws an exception
    structures_1_chain = [name for name in structure_names if '+' not in name]

    logging.info(f'Structures with 1 chain: numbers {len(structures_1_chain)} and id {structures_1_chain}')

    # create files for EnQA
    for name in tqdm(structures_1_chain):
        logging.info(f'Structure name: {structures_1_chain}')
        # path of folder for each structure with AF2 predictions
        af_prediction_enqa = os.path.join('example/alphafold_prediction', name)
        # path of folder for each structure with AF2 predictions for copying in this project, the folder is from the volume PATH_TO_ANT
        af_prediction_anbase = os.path.join(PATH_TO_ANT, name, 'ab_fv')
        # path of folder for each structure with AF2 predictions for predicting lddt by EnQA
        af_model_enqa = os.path.join('example/model', name)
        
        # create folder for each structure
        if not os.path.isdir(af_prediction_enqa):
            os.mkdir(af_prediction_enqa)
        if not os.path.isdir(af_model_enqa):
            os.mkdir(af_model_enqa)
        # copy files from volume to EnQA project
        if not os.listdir(af_prediction_enqa):
            copy_af2_predictions(af_prediction_anbase, af_prediction_enqa)
            logging.info(f'Files have copied from volume to alpha_prediction for {name}')
        if not os.listdir(af_model_enqa):
            copy_af2_pred_model(af_prediction_anbase, af_model_enqa)
            logging.info(f'Files have copied from volume to model for {name}')

    # run EnQA
    for name in tqdm(structures_1_chain):
        for numb in range(1, NUMBERS_OF_PRED + 1):
            os.system(f'python3 EnQA.py --input example/model/{name}/relaxed_model_{numb}.pdb --output outputs/{name} --method EGNN_Full --alphafold_prediction example/alphafold_prediction/{name}/  ')
        logging.info(f'EnQA have done for {name}')
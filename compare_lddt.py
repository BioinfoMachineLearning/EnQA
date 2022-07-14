import csv
import os
import logging
from sys import path, path_importer_cache
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import Structure
from Bio.PDB import PDBIO 
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Optional, List, Tuple
from Bio.PDB.Polypeptide import three_to_one, standard_aa_names
import csv

PATH_TO_TARGET = Path("lddt_results")
PATH_TO_PRED = Path("outputs")
NUMBERS_OF_PRED = 5


def get_structures_names(path: Path) -> List:
    """
    Get structures name for lddt calculation 
    @param path: path to structures
    @return: list of structures names
    """
    return os.listdir(path)

def get_mean(vector: np.array) -> int:
    """
    Calculate mean value
    @param vector: vector of numbers
    @return: mean of vector
    """
    return np.mean(vector)

def get_square_diff(vector1: np.array, vector2: np.array) -> int:
    """
    Calculate (mean(vector1) - mean(vector2))^2
    @param vector1: vector of numbers
    @param vector2: vector of numbers
    @return: a number which is equal to (mean(vector1) - mean(vector2))^2
    """
    return (get_mean(vector1) - get_mean(vector2))**2

def get_skipped_rows(path_to_file: Path) -> List:
    """
    Get skipped rows after lddt calculation. There are 10 or 15 skipped rows.
    @param path_to_file: path to file
    @return: list of row numbers for skipping
    """
    skipped = list()
    with open(path_to_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            for word in row:
                if "Chain" in word:
                   return skipped
            skipped.append(i)
    return skipped

def read_lddt_from_csv(path: Path, skipped: List) -> np.array:
    """
    Read vector of numbers from csv file
    @param path: path to file
    @param skipped: row numbers which will be skipped
    @return: vector of numbers 
    """
    return pd.read_table(path, sep='\t', skiprows=skipped)["Score"]

def read_lddt_from_np(path: Path) -> np.array:
    """
    Read vector of numbers from npy file
    @param path: path to file
    @return: vector of numbers
    """
    return np.load(path)

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', filename='logging_compare_lddt.txt', level=logging.DEBUG)

    diff_lddt = list()
    structure_names = get_structures_names(PATH_TO_TARGET)
    for name in tqdm(structure_names):
        logging.info(f"Structure name: {name}")

        for ind in range(1, NUMBERS_OF_PRED):
            pred_file = PATH_TO_PRED / name / f"relaxed_model_{ind}.npy"
            target_file = PATH_TO_TARGET / name / f"{name}_{ind}.csv"
            skipped = get_skipped_rows(target_file)
            pred_lddt = read_lddt_from_np(pred_file)
            target_lddt = read_lddt_from_csv(path=target_file, skipped=skipped)
            diff_lddt.append(get_square_diff(target_lddt, pred_lddt))
        
    logging.info(f"LDDT MSE: {np.mean(diff_lddt)}")





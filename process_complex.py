from ast import In
from process import process
import os 
from pathlib import Path
from tqdm import tqdm
from biopandas.pdb import PandasPdb

PATH_TO_COMPLEX = Path('complex_examples')
INPUT_NAME = 'docked_joined.pdb'
LABEL_NAME = 'real_joined.pdb'
OUTPUT_PATH = 'outputs/processed'


# python3 process.py --input example/model/6KYTP/test_model.pdb --label_pdb example/model/6KYTP/test_model.pdb --output outputs/processed

if __name__ == '__main__':
    complex_names = os.listdir(PATH_TO_COMPLEX)
    for name in tqdm(complex_names):
        input_path = PATH_TO_COMPLEX / name / INPUT_NAME
        label_path = PATH_TO_COMPLEX / name / LABEL_NAME
        process(input_str=str(input_path), output_str=str(OUTPUT_PATH),
                 label_pdb=label_path, name=name)
        

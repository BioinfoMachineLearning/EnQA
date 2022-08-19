import argparse
import enum
import os
import shutil

import numpy as np

PATH_TO_MODEL = '/data/user/zenkova/EnQA_new/EnQA'
ONEQ_TASK = "oneq start-task -c 4 -g 1 -u v100 \
            --image dock.biocad.ru/pythonbuilder:3.9-poetry1.1.13-cuda10.1 \
            --volume 'a2fa33c8-1503-438f-968e-01f09bec64b5=/mnt/volume_complex_lddt' \
            'cd {} && . ./venv/bin/activate && python3 process_complex.py --input /mnt/volume_complex_lddt/consistent/' \
            --sample {}--reference_name real_joined.pdb --prediction_name docked_joined.pdb --output {}"
            

def write_to_files(array_id: np.array, type_sample: str) -> None:
    """
    Write strucutres id to files.
    @param array_id: splitted structure ids into n_tasks parts
    @param type_sample: for which sample tasks are run, choises are train or valid
    @return: None
    """
    folder = f"oneq_tasks_{type_sample}"
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    for _ind, _ids in enumerate(array_id):
        filename = f"{type_sample}_{_ind}.txt"
        with open(f"{folder}/{filename}", 'w') as f:
            f.write('\n'.join(_ids))

def run_oneq_task(type_sample: str, n_tasks: int) -> None:
    """
    Run oneq tasks.
    @param type_sample: for which sample tasks are run, choises are train or valid
    @param n_tasks: number of tasks
    @return: None
    """
    folder = f"oneq_tasks_{type_sample}"
    for _ind in range(n_tasks):
        filename = f"{type_sample}_{_ind}.txt"
        path_to_file = f"{folder}/{filename}"
        output_folder = f"outputs_{type_sample}_{_ind}/processed"
        os.system(f"{ONEQ_TASK.format(PATH_TO_MODEL, path_to_file, output_folder)}")
  

def run_oneq_tasks(sample: str, n_tasks: int, type_sample: str='train') -> None:
    """
    Run n_tasks oneq tasks.
    @param sample: txt file with structure ids for training 
    @param n_tasks: number of tasks
    @return: None
    """
    structure_id = list()
    with open(sample, 'r') as f:
        structure_id = f.read().splitlines()
    array_id = np.array_split(structure_id, n_tasks)
    write_to_files(array_id, type_sample)
    # run_oneq_task(type_sample, n_tasks)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run oneq tasks for getting features and labels.')
    parser.add_argument(
        '--sample', 
        type=str, 
        required=True,
        help='File with structure ids for training.'
    )
    parser.add_argument(
        '--n_tasks', 
        type=int, 
        required=True,
        help='Number of oneq tasks.'
    )
    parser.add_argument(
        '--type_sample',
        type=str,
        choices=['train', 'valid'],
        default='train',
        help='For which sample tasks are run.'
    )
    
    args = parser.parse_args()
    run_oneq_tasks(sample=args.sample, n_tasks=args.n_tasks, type_sample=args.type_sample)

   
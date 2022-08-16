import argparse
import logging

from pathlib import Path
from tqdm import tqdm

from process import process

def process_complex(
    input_path: Path, 
    sample: str, 
    reference_name: str='real_joined.pdb',
    prediction_name: str='docked_joined.pdb', output: str='outputs/processed'
) -> None:
    """
    Run process.py for each structure.
    @param input_path: path to input pdb files
    @param sample: txt file with structure ids for training 
    @param reference_name: name of reference structure
    @param prediction_name: name of predicted structure
    @param output: path to output folder
    @return: None
    """
    structure_id = list()
    bad_structure = list() # for debugging
    with open(sample, 'r') as f:
        structure_id = f.read().splitlines()
    for _id in tqdm(structure_id):
        ref_path = input_path / _id / reference_name
        pred_path = input_path / _id / prediction_name
        logging.info(f"Process for {_id} is starting... Reference structure: {ref_path}, Predicted structure: {pred_path}")
        try:
            process(
            input_pdb=str(pred_path), 
            output_path=output,
            label_pdb=str(ref_path), 
            input_name=_id
            )
            logging.info(f"Process {_id} done.")
        except Exception as e:
            logging.error(f"Problems with {_id}!")
            logging.exception(e)
            bad_structure.append(_id)
            continue
    # for debug
    with open('bad_structures.txt', 'w') as f:
        f.write('\n'.join(bad_structure))

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s - %(message)s', 
        filename='complex_process.txt', 
        level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(description='Predict model quality and output numpy array format for complex.')
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to input pdb files.'
    )
    parser.add_argument(
        '--sample', 
        type=str, 
        required=True,
        help='File with structure ids for training.'
    )
    parser.add_argument(
        '--reference_name', 
        type=str, 
        required=False,
        default='real_joined.pdb', 
        help='File name for reference structure.'
    )
    parser.add_argument(
        '--prediction_name', 
        type=str, 
        required=False, 
        default='docked_joined.pdb', 
        help='File name for predicted structure.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=False, 
        default='outputs/processed',
        help='Path to output folder.'
    )
    args = parser.parse_args()
    process_complex(
        input_path=Path(args.input), 
        sample=args.sample, 
        reference_name=args.reference_name,
        prediction_name=args.prediction_name,
        output=args.output
        )

  
 

# python3 process_complex.py --input /mnt/volume_complex_lddt/consistent/ --sample train.txt --reference_name real_joined.pdb --prediction_name docked_joined.pdb --output outputs/processed
# python3 process.py --input example/model/6KYTP/test_model.pdb --label_pdb example/model/6KYTP/test_model.pdb --output outputs/processed


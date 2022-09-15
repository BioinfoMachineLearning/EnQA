import argparse
import os 

from sklearn.model_selection import train_test_split


def split_train_test(input_path: str, valid: bool=False) -> None:
    """
    Split complexes into random train and test subsets and write complex names to train.txt and test.txt.
    Complex names are separated by line feed.
    @param input_path: path to complexes.
    @return: None
    """
    complex_names = os.listdir(input_path)
    complex_train, complex_test = train_test_split(complex_names, test_size=0.1, random_state=42) 
    if valid:
        complex_train, complex_valid = train_test_split(complex_names, test_size=0.1, random_state=42) 
        with open('valid.txt', 'w') as f:
            f.write('\n'.join(complex_valid))
    with open('train.txt', 'w') as f:
        f.write('\n'.join(complex_train))
    with open('test.txt', 'w') as f:
        f.write('\n'.join(complex_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split complexes into random train and test subsets.')
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to input pdb files.'
    )
    parser.add_argument(
        '--valid',
        type=bool,
        required=False,
        default=False,
        help='Split into random train, validation and test subsets or only train and test subsets'
    )
    args = parser.parse_args()
    split_train_test(args.input, args.valid)

# python3 train_test_split.py --input '/mnt/volume_complex_lddt/consistent/'
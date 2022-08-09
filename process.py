import os
import pickle
import shutil

import torch
import argparse
import numpy as np
from biopandas.pdb import PandasPdb

from data.loader import expand_sh
from data.process_label import generate_dist_diff, generate_lddt_score, generate_coords_transform
from feature import create_basic_features, get_base2d_feature
from data.process_alphafold import process_alphafold_target_ensemble, process_alphafold_model, mergePDB, process_without_af_model


def process(input_str: str, output_str: str, 
            alphafold_prediction: str='', label_pdb: str='', name: str='') -> None:
    
    device = 'cpu'
    lddt_cmd = 'utils/lddt'
    if not os.path.isdir(output_str):
        os.makedirs(output_str)

    # Featureize
    disto_types = ['esto9']
    input_name = os.path.basename(input_str).replace('.pdb', '')
    ppdb = PandasPdb().read_pdb(input_str)
    is_multi_chain = len(ppdb.df['ATOM']['chain_id'].unique()) > 1
    temp_dir = output_str+ '/tmp/'

    if is_multi_chain:
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        outputPDB = os.path.join(temp_dir, 'merged_' + input_name + '.pdb')
        mergePDB(input_str, outputPDB, newStart=1)
        input_str = outputPDB

    one_hot, features, pos, sh_adj, el = create_basic_features(input_str, output_str, template_path=None,
                                                               diff_cutoff=15, coordinate_factor=0.01)
    sh_data = expand_sh(sh_adj, one_hot.shape[1])
    
    if not alphafold_prediction:
        af2_qa, plddt, cmap, dict_2d = process_without_af_model(input_str)
    else:
        af2_qa = process_alphafold_model(input_str, alphafold_prediction, lddt_cmd, n_models=5,
                                     is_multi_chain=is_multi_chain, temp_dir=temp_dir)
        plddt, cmap, dict_2d = process_alphafold_target_ensemble(alphafold_prediction, disto_types,
                                                             n_models=5, cmap_cutoff_dim=42,
                                                       input_pdb_file=input_str)

    f2d = np.concatenate((sh_data, dict_2d['esto9']), axis=0)
    f1d = np.concatenate((one_hot, features, plddt, af2_qa), axis=0)
    f1d = torch.tensor(f1d).unsqueeze(0).to(device)
    f2d = torch.tensor(f2d).unsqueeze(0).to(device)
    pos = torch.tensor(pos).to(device)
    el = [torch.tensor(i).to(device) for i in el]
    cmap = torch.tensor(cmap).to(device)

    label_lddt = generate_lddt_score(input_str, label_pdb, 'utils/lddt') # WHAT?
    diff_bins = generate_dist_diff(input_str, label_pdb)
    pos_transformed = generate_coords_transform(input_str, label_pdb, None)
    label_lddt = torch.tensor(label_lddt).to(device)
    diff_bins = torch.tensor(diff_bins).unsqueeze(0).to(device)
    pos_transformed = torch.tensor(pos_transformed/100).to(device)

    if not name:
        torch.save({"f1d": f1d, "f2d": f2d, "pos": pos, 'el': el, 'cmap': cmap,
                'label_lddt': label_lddt, 'diff_bins': diff_bins, 'pos_transformed': pos_transformed},
               os.path.join(output_str, os.path.basename(input_str).replace('.pdb', '') + '.pt'))
    else:
        torch.save({"f1d": f1d, "f2d": f2d, "pos": pos, 'el': el, 'cmap': cmap,
                'label_lddt': label_lddt, 'diff_bins': diff_bins, 'pos_transformed': pos_transformed},
               os.path.join(output_str, f"{name}_{os.path.basename(input_str).replace('.pdb', '')}" + '.pt'))
    

# python3 process.py --input example/model/6KYTP/test_model.pdb --label_pdb example/model/6KYTP/test_model.pdb --output outputs/processed --alphafold_prediction example/alphafold_prediction/6KYTP/


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict model quality and output numpy array format.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input pdb file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output folder.')
    parser.add_argument('--alphafold_prediction', type=str, required=False, default='',
                        help='Path to alphafold prediction results.') # required has been changed to False
    parser.add_argument('--label_pdb', type=str, required=True, default='')
    parser.add_argument('--name', type=str, required=False, default='',
                        help='Structure name.')
    args = parser.parse_args()
    process(input_str=args.input, output_str=args.output, alphafold_prediction=args.alphafold_prediction, label_pdb=args.label_pdb)


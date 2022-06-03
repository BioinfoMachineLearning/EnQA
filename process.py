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
from data.process_alphafold import process_alphafold_target_ensemble, process_alphafold_model, mergePDB

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict model quality and output numpy array format.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input pdb file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output folder.')
    parser.add_argument('--alphafold_prediction', type=str, required=True, default='',
                        help='Path to alphafold prediction results.')
    parser.add_argument('--label_pdb', type=str, required=True, default='')
    args = parser.parse_args()
    device = 'cpu'
    lddt_cmd = 'utils/lddt'
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Featureize
    disto_types = ['esto9']
    input_name = os.path.basename(args.input).replace('.pdb', '')
    ppdb = PandasPdb().read_pdb(args.input)
    is_multi_chain = len(ppdb.df['ATOM']['chain_id'].unique()) > 1
    temp_dir = args.output + '/tmp/'

    if is_multi_chain:
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        outputPDB = os.path.join(temp_dir, 'merged_' + input_name + '.pdb')
        mergePDB(args.input, outputPDB, newStart=1)
        args.input = outputPDB

    one_hot, features, pos, sh_adj, el = create_basic_features(args.input, args.output, template_path=None,
                                                               diff_cutoff=15, coordinate_factor=0.01)
    sh_data = expand_sh(sh_adj, one_hot.shape[1])
    af2_qa = process_alphafold_model(args.input, args.alphafold_prediction, lddt_cmd, n_models=5,
                                     is_multi_chain=is_multi_chain, temp_dir=temp_dir)

    plddt, cmap, dict_2d = process_alphafold_target_ensemble(args.alphafold_prediction, disto_types,
                                                             n_models=5, cmap_cutoff_dim=42,
                                                             input_pdb_file=args.input)

    f2d = np.concatenate((sh_data, dict_2d['esto9']), axis=0)
    f1d = np.concatenate((one_hot, features, plddt, af2_qa), axis=0)
    f1d = torch.tensor(f1d).unsqueeze(0).to(device)
    f2d = torch.tensor(f2d).unsqueeze(0).to(device)
    pos = torch.tensor(pos).to(device)
    el = [torch.tensor(i).to(device) for i in el]
    cmap = torch.tensor(cmap).to(device)

    label_lddt = generate_lddt_score(args.input, args.label_pdb, 'utils/lddt')
    diff_bins = generate_dist_diff(args.input, args.label_pdb)
    pos_transformed = generate_coords_transform(args.input, args.label_pdb, None)
    label_lddt = torch.tensor(label_lddt).to(device)
    diff_bins = torch.tensor(diff_bins).unsqueeze(0).to(device)
    pos_transformed = torch.tensor(pos_transformed/100).to(device)

    torch.save({"f1d": f1d, "f2d": f2d, "pos": pos, 'el': el, 'cmap': cmap,
                'label_lddt': label_lddt, 'diff_bins': diff_bins, 'pos_transformed': pos_transformed},
               os.path.join(args.output, os.path.basename(args.input) + '.pt'))

# python3 process.py --input example/model/6KYTP/test_model.pdb --label_pdb example/model/6KYTP/test_model.pdb --output outputs/processed --alphafold_prediction example/alphafold_prediction/6KYTP/

import os
import re
import pickle
import subprocess

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import pdist, squareform

from data.process_label import generate_lddt_score, parse_pdbfile, get_coords_ca


def process_alphafold_model(input_model_path, alphafold_prediction_path, lddt_cmd, n_models=5,
                            is_multi_chain=False, temp_dir=None):
    lddt_list = []
    all_af2_pred_models = ['relaxed_model_' + str(i + 1) + '.pdb' for i in range(n_models)]
    for i in all_af2_pred_models:
        af2_pred_model = os.path.join(alphafold_prediction_path, i)
        if is_multi_chain:
            af2_merged = os.path.join(temp_dir, 'af2_merged_' + i)
            mergePDB(af2_pred_model, af2_merged)
            lddt_af2 = generate_lddt_score(input_model_path, af2_merged, lddt_cmd)
        else:
            lddt_af2 = generate_lddt_score(input_model_path, af2_pred_model, lddt_cmd)
        lddt_list.append(lddt_af2)
    assert len(lddt_list) == n_models
    af2_qa = np.vstack(lddt_list)
    # shape (n_models, L)
    return af2_qa.astype(np.float32)


def compute_esto9_single(input_pdb_file, af2_result_file,
                         bins_esto=[-np.inf, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, np.inf]):
    pose_input = parse_pdbfile(input_pdb_file)
    pos_input = get_coords_ca(pose_input)
    x = pickle.load(open(af2_result_file, 'rb'))['distogram']
    af2_disto = x['logits']
    af2_disto_softmax = softmax(af2_disto, axis=2)
    input_dist = squareform(pdist(pos_input))
    bins = x['bin_edges']
    bins = np.concatenate(([0], bins))
    bins_avg = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    bins_avg.append(bins[-1])
    bins_avg = np.repeat(np.array([bins_avg]), input_dist.shape[0] * input_dist.shape[1], axis=0).reshape(
        (input_dist.shape[0],
         input_dist.shape[1],
         -1))
    # dist_from_disto = np.sum(np.multiply(bins_avg, af2_disto_mask), axis=2)
    estogram64 = input_dist.reshape((input_dist.shape[0], input_dist.shape[1], 1)).repeat(bins_avg.shape[-1],
                                                                                          axis=2) - bins_avg
    estogram9 = np.stack([np.sum(np.multiply(np.logical_and(estogram64 < bins_esto[i + 1], estogram64 >= bins_esto[i]),
                                             af2_disto_softmax), axis=2) for i in range(len(bins_esto) - 1)])
    return estogram9


def compute_esto9(input_pdb_file, af2_result_path):
    estogram9_list = []
    for idx in range(5):
        af2_result_file = os.path.join(af2_result_path, 'result_model_{}.pkl'.format(idx + 1))
        estogram9 = compute_esto9_single(input_pdb_file, af2_result_file)
        estogram9_list.append(estogram9)
    estogram9_list = np.stack(estogram9_list).reshape((-1, estogram9.shape[1], estogram9.shape[1]))
    return estogram9_list


def process_alphafold_target(alphafold_prediction_path, model_type, n_models=5, cmap_cutoff_dim=42,
                             input_pdb_file=None):
    data_2d = []
    data_1d = []
    for i in range(n_models):
        src = os.path.join(alphafold_prediction_path, 'result_model_' + str(i + 1) + '.pkl')
        x = pickle.load(open(src, 'rb'))
        x2d = x['distogram']['logits']
        x2d = softmax(x2d, axis=2)
        data_2d.append(x2d)
        data_1d.append(x['plddt'] / 100)
    cmap = np.average(np.array([np.sum(i[:, :, 0:cmap_cutoff_dim], axis=2) for i in data_2d]), axis=0)
    plddt = np.vstack(data_1d)
    data_2d = np.concatenate(data_2d, axis=2)
    if model_type == 'disto':
        af2_2d = data_2d.transpose(2, 0, 1)
    elif model_type == 'cov25':
        l = data_2d.shape[0]
        x = data_2d.reshape((l, l, n_models, -1))
        m1 = x - x.sum(3, keepdims=1) / x.shape[3]
        af2_2d = np.einsum('ijml,ijnl->ijmn', m1, m1) / (x.shape[3] - 1)
        af2_2d = af2_2d.reshape(l, l, -1).transpose(2, 0, 1)
    elif model_type == 'cov64':
        l = data_2d.shape[0]
        x = data_2d.reshape((l, l, n_models, -1))
        x = x.transpose(0, 1, 3, 2)
        af2_2d = np.var(x, axis=3).transpose(2, 0, 1)
    elif model_type == 'esto9':
        af2_2d = compute_esto9(input_pdb_file, alphafold_prediction_path)
    else:
        raise ValueError
    # shape: plddt (n_models, L) cmap(L,L) af2_2d (c, L, L)
    return plddt.astype(np.float32), cmap.astype(np.float32), af2_2d.astype(np.float32)


def process_alphafold_target_ensemble(alphafold_prediction_path, disto_types, n_models=5, cmap_cutoff_dim=42,
                                      input_pdb_file=None):
    data_2d = []
    data_1d = []
    for i in range(n_models):
        src = os.path.join(alphafold_prediction_path, 'result_model_' + str(i + 1) + '.pkl')
        x = pickle.load(open(src, 'rb'))
        x2d = x['distogram']['logits']
        x2d = softmax(x2d, axis=2)
        data_2d.append(x2d)
        data_1d.append(x['plddt'] / 100)
    cmap = np.average(np.array([np.sum(i[:, :, 0:cmap_cutoff_dim], axis=2) for i in data_2d]), axis=0)
    plddt = np.vstack(data_1d)
    data_2d = np.concatenate(data_2d, axis=2)
    af2_2d_dict = {}
    if 'disto' in disto_types:
        af2_2d = data_2d.transpose(2, 0, 1)
        af2_2d_dict['disto'] = af2_2d.astype(np.float32)
    if 'cov25' in disto_types:
        l = data_2d.shape[0]
        x = data_2d.reshape((l, l, n_models, -1))
        m1 = x - x.sum(3, keepdims=1) / x.shape[3]
        af2_2d = np.einsum('ijml,ijnl->ijmn', m1, m1) / (x.shape[3] - 1)
        af2_2d = af2_2d.reshape(l, l, -1).transpose(2, 0, 1)
        af2_2d_dict['cov25'] = af2_2d.astype(np.float32)
    if 'cov64' in disto_types:
        l = data_2d.shape[0]
        x = data_2d.reshape((l, l, n_models, -1))
        x = x.transpose(0, 1, 3, 2)
        af2_2d = np.var(x, axis=3).transpose(2, 0, 1)
        af2_2d_dict['cov64'] = af2_2d.astype(np.float32)
    elif 'esto9' in disto_types:
        af2_2d = compute_esto9(input_pdb_file, alphafold_prediction_path)
        af2_2d_dict['esto9'] = af2_2d.astype(np.float32)
    # shape: plddt (n_models, L) cmap(L,L) af2_2d (c, L, L)
    return plddt.astype(np.float32), cmap.astype(np.float32), af2_2d_dict


def mergePDB(inputPDB, outputPDB, newStart=1):
    with open(inputPDB, 'r') as f:
        x = f.readlines()
    filtered = [i for i in x if re.match(r'^ATOM.+', i)]
    chains = set([i[21] for i in x if re.match(r'^ATOM.+', i)])
    chains = list(chains)
    chains.sort()
    with open(outputPDB + '.tmp', 'w') as f:
        f.writelines(filtered)
    merge_cmd = 'pdb_selchain -{} {} | pdb_chain -A | pdb_reres -{} > {}'.format(','.join(chains),
                                                                                 outputPDB + '.tmp',
                                                                                 newStart,
                                                                                 outputPDB)
    subprocess.run(args=merge_cmd, shell=True)
    os.remove(outputPDB + '.tmp')


if __name__ == '__main__':
    input_pdb_file = 'example/model/1H8AA/test_model.pdb'
    af2_result_path = 'example/alphafold_prediction/1H8AA/'
    esto9 = compute_esto9(input_pdb_file, af2_result_path)
    print(esto9.shape)

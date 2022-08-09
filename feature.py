import os
import numpy as np
import pandas as pd
import shutil
import subprocess

from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

from data.loader import expand_sh
from data.process_alphafold import process_alphafold_model, process_alphafold_target
from data.process_label import generate_dist_diff, generate_coords_transform, generate_lddt_score
from utils.SGCN.common.graph import build_atom_level_data, build_residue_level_data, build_spherical_harmonics
from utils.SGCN.common.format import get_atom_labels, get_bonds_types
from utils.SGCN.common.utils import Logger, get_residue_type_to_id
from utils.SGCN.common.names import X_NORM_IDX, X_RES_FILE_NAME, SH_NAME

params_dict = {'voronota_exec': 'utils/SGCN/bin/voronota-linux',
               'maps_generator_exec': 'utils/SGCN/bin/sh-featurizer-linux',
               'meta_atom_types': 'utils/SGCN/metadata/protein_atom_types.txt',
               'meta_bond_types': 'utils/SGCN/metadata/bond_types.csv',
               'VORONOTA_RADII': 'utils/SGCN/metadata/voronota_radii.txt',
               'ELEMENTS_RADII': 'utils/SGCN/metadata/elements_radii.txt',
               'normalize_x': 'True', 'order': 5}


def load_x(file, normalize_x):
    compressed_x = pd.read_csv(file, sep=' ')
    residue_types = compressed_x['residue'].values
    features = compressed_x[['volume', 'buriedness', 'sasa']].values.astype(np.float32)
    residue_type_to_id = get_residue_type_to_id(params_dict['meta_atom_types'])
    if normalize_x:
        features_to_normalize = features[:, X_NORM_IDX] # volume and sasa
        normalizations = 1 / features_to_normalize.sum(axis=0) # 1 / sum of volume,  1 / sum of sasa
        normalized_features = np.einsum('ij,j->ij', features_to_normalize, normalizations) # features_to_normalize * normalizations
        features[:, X_NORM_IDX] = normalized_features
    one_hot = np.zeros((len(residue_types), len(residue_type_to_id)))
    for i, residue_type in enumerate(residue_types):
        one_hot[i, residue_type_to_id[residue_type]] = 1
    return one_hot, features


def get_position_from_pdb(pdb_file):
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_file)
    position_df = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA'][['x_coord', 'y_coord', 'z_coord']]
    return position_df.to_numpy().astype(np.float32)


def linear_recreate_c(ca_pos, cb_pos, n_pos):
    ca = -0.58273431
    cb = 0.56802827
    cc = -0.54067466
    b1, b2, b3 = np.array(ca_pos) - np.array(n_pos)
    x1 = y1 = z1 = 0
    x3, y3, z3 = np.array(cb_pos) - np.array(ca_pos)
    A = np.array([[cc, -ca * b3, ca * b2],
                  [ca * b3, cc, -ca * b1],
                  [-ca * b2, ca * b1, cc]])
    y = np.array([x3 + ca * b2 * z1 + ca * b3 * y1 - cb * b1 + cc * x1,
                  y3 + ca * b3 * x1 - ca * b1 * z1 - cb * b2 + cc * y1,
                  z3 + ca * b1 * y1 - ca * b2 * x1 - cb * b3 + cc * z1])
    return np.matmul(np.linalg.inv(A), y) + np.array(ca_pos)


def get_idx_mask(input_model_path, af2_pdb):
    pdb_input = PandasPdb().read_pdb(input_model_path)
    pdb_af2 = PandasPdb().read_pdb(af2_pdb)
    input_idx = pdb_input.df['ATOM']['residue_number'].to_numpy()
    af2_idx = pdb_af2.df['ATOM']['residue_number'].to_numpy()
    input_idx = np.unique(input_idx)
    af2_idx = np.unique(af2_idx)
    if not np.all(np.isin(input_idx, af2_idx)):
        raise IndexError('Model from AlphaFold2 should contain all indices in input PDB.')
    return np.isin(af2_idx, input_idx)


def restore_pdb(pdb_test, pdb_template, outfile):
    template_biopdb = PandasPdb().read_pdb(pdb_template)
    test_biopdb = PandasPdb().read_pdb(pdb_test)
    test_biopdb.df['ATOM']['x_coord'] -= test_biopdb.df['ATOM']['x_coord'].mean()
    test_biopdb.df['ATOM']['y_coord'] -= test_biopdb.df['ATOM']['y_coord'].mean()
    test_biopdb.df['ATOM']['z_coord'] -= test_biopdb.df['ATOM']['z_coord'].mean()
    df_restored = template_biopdb.df['ATOM'].copy()
    for i in range(template_biopdb.df['ATOM'].shape[0]):
        residue_flag = test_biopdb.df['ATOM']['residue_number'] == template_biopdb.df['ATOM']['residue_number'][i]
        atom_flag = test_biopdb.df['ATOM']['atom_name'] == template_biopdb.df['ATOM']['atom_name'][i]
        flag = residue_flag & atom_flag
        CA_flag = residue_flag & (test_biopdb.df['ATOM']['atom_name'] == 'CA')
        CB_flag = residue_flag & (test_biopdb.df['ATOM']['atom_name'] == 'CB')
        N_flag = residue_flag & (test_biopdb.df['ATOM']['atom_name'] == 'N')
        if flag.sum() == 1:
            df_restored.at[i, 'x_coord'] = test_biopdb.df['ATOM']['x_coord'][flag].values[0]
            df_restored.at[i, 'y_coord'] = test_biopdb.df['ATOM']['y_coord'][flag].values[0]
            df_restored.at[i, 'z_coord'] = test_biopdb.df['ATOM']['z_coord'][flag].values[0]
        else:
            assert CA_flag.sum() == 1
            ca_pos = [test_biopdb.df['ATOM']['x_coord'][CA_flag].values[0],
                      test_biopdb.df['ATOM']['y_coord'][CA_flag].values[0],
                      test_biopdb.df['ATOM']['z_coord'][CA_flag].values[0]]
            if CB_flag.sum() == 1 and N_flag.sum() == 1 and template_biopdb.df['ATOM']['atom_name'][i] == 'C':
                print('Found missing C : {} {} {}'.format(template_biopdb.df['ATOM']['residue_number'][i],
                                                          template_biopdb.df['ATOM']['residue_name'][i],
                                                          template_biopdb.df['ATOM']['atom_name'][i]))
                cb_pos = [test_biopdb.df['ATOM']['x_coord'][CB_flag].values[0],
                          test_biopdb.df['ATOM']['y_coord'][CB_flag].values[0],
                          test_biopdb.df['ATOM']['z_coord'][CB_flag].values[0]]
                n_pos = [test_biopdb.df['ATOM']['x_coord'][N_flag].values[0],
                         test_biopdb.df['ATOM']['y_coord'][N_flag].values[0],
                         test_biopdb.df['ATOM']['z_coord'][N_flag].values[0]]
                c_pos = linear_recreate_c(ca_pos, cb_pos, n_pos)
                df_restored.at[i, 'x_coord'] = c_pos[0]
                df_restored.at[i, 'y_coord'] = c_pos[1]
                df_restored.at[i, 'z_coord'] = c_pos[2]
            else:
                print('Found missing atom : {} {} {}'.format(template_biopdb.df['ATOM']['residue_number'][i],
                                                             template_biopdb.df['ATOM']['residue_name'][i],
                                                             template_biopdb.df['ATOM']['atom_name'][i]))
                df_restored.at[i, 'x_coord'] = ca_pos[0]
                df_restored.at[i, 'y_coord'] = ca_pos[1]
                df_restored.at[i, 'z_coord'] = ca_pos[2]
    template_biopdb.df['ATOM'] = df_restored
    template_biopdb.to_pdb(outfile)


def get_base2d_feature(model_pdb, out_path, dan_path='utils/DeepAccNet/DeepAccNet-noPyRosetta.py'):
    temp_dir = os.path.join(out_path, 'temp_dan')
    os.mkdir(temp_dir)
    temp_pdb = os.path.join(temp_dir, 'dan.pdb')
    shutil.copyfile(model_pdb, temp_pdb)
    try:
        subprocess.run(['python3', dan_path, '-r', temp_dir])
    except Exception as e:
        print('Got unexpected exception while preparing DAN feature: {}'.format(e))
    x = np.load(os.path.join(temp_dir, 'dan.npz'))
    x2d = np.concatenate((np.array([x["mask"]]), x["estogram"]), axis=0)
    shutil.rmtree(temp_dir)
    return x2d


# Generate one hot, position and SGCN features
# disto_type can be following formats:
# "base" : (16, L, L) DeepAccNet 2d features: mask + 15 bins of estogram
# "cov25" : (25, L, L) covariance abetween 5 Alphafold predictions
# "cov64" : (64, L, L) variance between 64 bins from Alphafold predictions
# "disto" : (64*5, L, L) use disto representation directly.
# "esto9" : (9*5, L, L) use 9 bins representation based on the range defined by l-DDT.
# Alphafold disto representation is first normalized with Softamx.

# output shape
# f1d(c, L)
# f2d(c, L, L)
# pos(L, 3)
# el(n_edges, n_edges)
# cmap(L, L)

def create_basic_features(input_model_path, output_feature_path, template_path=None,
                          diff_cutoff=15, coordinate_factor=0.01):
    if template_path is None:
        template_path = input_model_path
    model = os.path.basename(input_model_path).replace('.pdb', '')
    graph_path = os.path.join(output_feature_path, 'graph')
    tmp_path = os.path.join(output_feature_path, 'temp')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)
    normalized_model_path = os.path.join(tmp_path, 'input.pdb')
    tmp_model_path = os.path.join(tmp_path, 'model.pdb')
    log_path = os.path.join(output_feature_path, 'feature.log')
    logger = Logger(log_path, 'Start building S-GCN features', verbose=True, include_model=False)
    restore_pdb(input_model_path, pdb_template=template_path, outfile=normalized_model_path)
    single_bonds, double_bonds, aromat_bonds = get_bonds_types(params_dict['meta_bond_types'])
    build_atom_level_data(
        model_name=model,
        input_model_path=normalized_model_path,
        output_model_path=graph_path,
        allowed_atom_labels=get_atom_labels(params_dict['meta_atom_types']),
        voronota_exec=params_dict['voronota_exec'],
        voronota_radii_path=params_dict['VORONOTA_RADII'],
        elements_radii_path=params_dict['ELEMENTS_RADII'],
        tmp_model_path=tmp_model_path,
        tmp_target_scores_path=os.path.join(tmp_path, 'scores'),
        tmp_target_scores_path_expanded=os.path.join(tmp_path, 'scores_expanded'),
        tmp_balls_path=os.path.join(tmp_path, 'balls'),
        tmp_contacts_path=os.path.join(tmp_path, 'contacts'),
        tmp_volumes_path=os.path.join(tmp_path, 'volumes'),
        tmp_shelling_path=os.path.join(tmp_path, 'shelling'),
        tmp_contacts_path_expanded=os.path.join(tmp_path, 'contacts_expanded'),
        tmp_volumes_path_expanded=os.path.join(tmp_path, 'volumes_expanded'),
        tmp_shelling_path_expanded=os.path.join(tmp_path, 'shelling_expanded'),
        single_bonds=single_bonds,
        double_bonds=double_bonds,
        aromat_bonds=aromat_bonds,
        logger=logger,
    )
    build_residue_level_data(model_name=model, model_path=graph_path, logger=logger)
    one_hot, features = load_x(os.path.join(graph_path, X_RES_FILE_NAME), params_dict['normalize_x'] == 'True')
    pos_data = get_position_from_pdb(input_model_path) * coordinate_factor
    build_spherical_harmonics(model_path=graph_path, order=int(params_dict['order']),
                              maps_generator=params_dict['maps_generator_exec'],
                              skip_errors=True)
    sh_adj = np.load(os.path.join(graph_path, SH_NAME))
    el = np.where(cdist(pos_data, pos_data) <= diff_cutoff * coordinate_factor)
    one_hot = one_hot.transpose(1, 0)
    features = features.transpose(1, 0)
    shutil.rmtree(tmp_path)
    shutil.rmtree(graph_path)
    return one_hot.astype(np.float32), features.astype(np.float32), pos_data.astype(np.float32), sh_adj, el


def process_data(input_model_path, output_feature_path, template_path=None,
                 diff_cutoff=15, coordinate_factor=0.01, disto_type='base',
                 alphafold_prediction_path=None, lddt_cmd='utils/lddt',
                 alphafold_prediction_cache=None):
    one_hot, features, pos_data, sh_adj, el = create_basic_features(input_model_path, output_feature_path,
                                                                    template_path=template_path,
                                                                    diff_cutoff=diff_cutoff,
                                                                    coordinate_factor=coordinate_factor)
    if disto_type == 'base':
        f2d_dan = get_base2d_feature(input_model_path, output_feature_path)

        return one_hot.astype(np.float32), features.astype(np.float32), \
               pos_data.astype(np.float32), sh_adj, \
               f2d_dan.astype(np.float32), el[0], el[1]
    elif disto_type in ['cov25', 'cov64', 'disto', 'esto9']:
        assert alphafold_prediction_path is not None
        af2_qa = process_alphafold_model(input_model_path, alphafold_prediction_path, lddt_cmd)
        if alphafold_prediction_cache is not None:
            if os.path.isfile(alphafold_prediction_cache):
                x = np.load(alphafold_prediction_cache)
                plddt = x['plddt']
                cmap = x['cmap']
                af2_2d = x['af2_2d']
            else:
                plddt, cmap, af2_2d = process_alphafold_target(alphafold_prediction_path, disto_type,
                                                               input_pdb_file=input_model_path)
                np.savez(alphafold_prediction_cache, plddt=plddt, cmap=cmap, af2_2d=af2_2d)
        else:
            plddt, cmap, af2_2d = process_alphafold_target(alphafold_prediction_path, disto_type,
                                                           input_pdb_file=input_model_path)

        return one_hot.astype(np.float32), features.astype(np.float32), \
               pos_data.astype(np.float32), sh_adj, af2_2d.astype(np.float32), \
               el[0], el[1], plddt.astype(np.float32), cmap.astype(np.float32), af2_qa.astype(np.float32)
    else:
        return one_hot.astype(np.float32), features.astype(np.float32), \
               pos_data.astype(np.float32), sh_adj, el[0], el[1]


def create_feature(input_model_path, output_feature_path, template_path=None,
                   diff_cutoff=15, coordinate_factor=0.01, disto_type='base',
                   alphafold_prediction_path=None, lddt_cmd='utils/lddt',
                   alphafold_prediction_cache=None, af2_pdb=''):
    if disto_type == 'base':
        one_hot, features, pos, sh, disto_feature, el_src, el_dst = process_data(input_model_path,
                                                                                 output_feature_path,
                                                                                 template_path=template_path,
                                                                                 diff_cutoff=diff_cutoff,
                                                                                 coordinate_factor=coordinate_factor,
                                                                                 disto_type=disto_type)
        sh_data = expand_sh(sh, pos.shape[0])
        f2d = np.concatenate((sh_data, disto_feature), axis=0)
        f1d = np.concatenate((one_hot, features), axis=0)
        return f1d, f2d, pos, (el_src, el_dst)
    else:
        one_hot, features, pos, sh, af2_2d, el_src, el_dst, plddt, cmap, af2_qa = process_data(input_model_path,
                                                                                               output_feature_path,
                                                                                               template_path=template_path,
                                                                                               diff_cutoff=diff_cutoff,
                                                                                               coordinate_factor=coordinate_factor,
                                                                                               disto_type=disto_type,
                                                                                               alphafold_prediction_path=alphafold_prediction_path,
                                                                                               lddt_cmd=lddt_cmd,
                                                                                               alphafold_prediction_cache=alphafold_prediction_cache
                                                                                               )
        sh_data = expand_sh(sh, pos.shape[0])
        if af2_pdb != '':
            mask = get_idx_mask(input_model_path, af2_pdb)
            plddt = plddt[:, mask]
            af2_2d = af2_2d[:, mask, :][:, :, mask]
        f1d = np.concatenate((one_hot, features, plddt, af2_qa), axis=0)
        f2d = np.concatenate((sh_data, af2_2d), axis=0)
        return f1d, f2d, pos, (el_src, el_dst), cmap


def create_train_dataset(input_model_path, output_feature_model_path, output_feature_target_path, template_path,
                         diff_cutoff=15, coordinate_factor=0.01, disto_type='base',
                         alphafold_prediction_path=None, lddt_cmd='utils/lddt',
                         transform_method='rbmt', tmalign_path='utils/TMalign'):
    model_name = os.path.basename(input_model_path).replace('.pdb', '')
    target_name = os.path.basename(template_path).replace('.pdb', '')
    output_feature_target_file = os.path.join(output_feature_target_path, target_name + '.npz')
    output_model_target_file = os.path.join(output_feature_model_path, target_name + '.' + model_name + '.npz')

    diff_bins = generate_dist_diff(input_model_path, template_path)
    pos_label_superpose = generate_coords_transform(template_path, input_model_path, output_feature_model_path,
                                                    transform_method=transform_method,
                                                    tmalign_path=tmalign_path)
    pos_label_superpose = pos_label_superpose.astype(np.float32)
    lddt_label = generate_lddt_score(input_model_path, template_path, lddt_cmd).astype(np.float32)
    if disto_type == 'base':
        one_hot, features, pos, sh, disto_feature, el_src, el_dst = process_data(input_model_path,
                                                                                 output_feature_model_path,
                                                                                 template_path=template_path,
                                                                                 diff_cutoff=diff_cutoff,
                                                                                 coordinate_factor=coordinate_factor,
                                                                                 disto_type=disto_type)
        np.savez(output_model_target_file,
                 one_hot=one_hot, features=features, pos=pos, sh=sh, disto_feature=disto_feature,
                 el_src=el_src, el_dst=el_dst, diff_bins=diff_bins, pos_label_superpose=pos_label_superpose,
                 lddt_label=lddt_label)
    else:
        one_hot, features, pos, sh, el_src, el_dst = process_data(input_model_path,
                                                                  output_feature_model_path,
                                                                  template_path=template_path,
                                                                  diff_cutoff=diff_cutoff,
                                                                  coordinate_factor=coordinate_factor,
                                                                  disto_type='None',
                                                                  alphafold_prediction_path=alphafold_prediction_path,
                                                                  lddt_cmd=lddt_cmd)
        af2_qa = process_alphafold_model(input_model_path, alphafold_prediction_path, lddt_cmd, n_models=5)
        np.savez(output_model_target_file,
                 one_hot=one_hot, features=features, pos=pos, sh=sh,
                 el_src=el_src, el_dst=el_dst, diff_bins=diff_bins, pos_label_superpose=pos_label_superpose,
                 lddt_label=lddt_label, af2_qa=af2_qa)
        if not os.path.isfile(output_feature_target_file):
            plddt, cmap, af2_2d = process_alphafold_target(alphafold_prediction_path, disto_type,
                                                           input_pdb_file=input_model_path)
            np.savez(output_feature_target_file, disto_feature=af2_2d, plddt=plddt, cmap=cmap)


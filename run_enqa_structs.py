import os
import pickle
import shutil
import logging

import torch
import argparse
import numpy as np
from biopandas.pdb import PandasPdb
from pathlib import Path
from tqdm import tqdm

from data.loader import expand_sh
from data.process_label import parse_pdbfile
from feature import create_basic_features, get_base2d_feature
from data.process_alphafold import process_alphafold_target_ensemble, process_alphafold_model, mergePDB, process_without_af_model
from network.resEGNN import resEGNN_with_mask, resEGNN_with_ne


def enqa(
    input_path: str, 
    output_path: str, 
    method: str, 
    cpu: str,
    alphafold_prediction: str,
    alphafold_feature_cache: str,
    af2_pdb: str,
    complex_b: bool
) -> np.array:
    """
    Get prediction lddt for one structure from test.txt file.
    @param input_path: path to input pdb file
    @param output_path: path to output folder
    @param method: prediction method, can be "ensemble", "EGNN_Full", "se3_Full", "EGNN_esto9" or "EGNN_covariance". Ensemble can be done listing multiple models separated by comma
    @param cpu: force to use CPU
    @param alphafold_prediction: path to alphafold prediction results
    @param alphafold_feature_cache: cashe
    @param af2_pdb: optional. PDBs from AlphaFold2 predcition for index correction with input pdb. Must contain all residues in input pdb
    @param complex_b: True if input pdb is complex, else False
    @return: prediction lddt 
    """   
    
    if alphafold_feature_cache == '':
        alphafold_feature_cache = None
    device = torch.device('cuda:0') if torch.cuda.is_available() and not cpu else 'cpu'
    lddt_cmd = 'utils/lddt'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Featureize
    if method == 'ensemble':
        methods = ["EGNN_Full", "se3_Full", "EGNN_esto9"]
    else:
        methods = method.split(',')

    disto_types = []
    if 'EGNN_Full' in methods or 'se3_Full' in methods or 'EGNN_esto9' in methods:
        disto_types.append('esto9')
    if 'EGNN_covariance' in methods:
        disto_types.append('cov25')
    if 'EGNN_no_AF2' in methods:
        disto_types.append('base')

    input_name = os.path.basename(input_path).replace('.pdb', '')
    ppdb = PandasPdb().read_pdb(input_path)
    is_multi_chain = len(ppdb.df['ATOM']['chain_id'].unique()) > 1
    temp_dir = output_path + '/tmp/'

    if is_multi_chain:
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        outputPDB = os.path.join(temp_dir, 'merged_'+input_name+'.pdb')
        mergePDB(input_path, outputPDB, newStart=1)
        input_path = outputPDB

    one_hot, features, pos, sh_adj, el = create_basic_features(input_path, output_path, template_path=None,
                                                               diff_cutoff=15, coordinate_factor=0.01)
    sh_data = expand_sh(sh_adj, one_hot.shape[1])
    pred_lddt_all = np.zeros(one_hot.shape[1])
    if 'EGNN_Full' in methods or 'se3_Full' in methods or 'EGNN_esto9' in methods or 'EGNN_covariance' in methods:
        use_af2 = True
    else:
        use_af2 = False
    dict_2d = {}
    if use_af2:
        if not complex_b:
            af2_qa = process_alphafold_model(input_path, alphafold_prediction, lddt_cmd, n_models=5,
                                            is_multi_chain=is_multi_chain, temp_dir=temp_dir)
            if alphafold_feature_cache is not None and os.path.isfile(alphafold_prediction_cache):
                x = pickle.load(open(alphafold_prediction_cache, 'rb'))
                plddt = x['plddt']
                cmap = x['cmap']
                dict_2d = x['af2_2d_dict']
            else:
                plddt, cmap, dict_2d = process_alphafold_target_ensemble(alphafold_prediction, disto_types,
                                                                        n_models=5, cmap_cutoff_dim=42,
                                                                        input_pdb_file=input_path)
                if alphafold_feature_cache is not None:
                    pickle.dump({'plddt': plddt, 'cmap': cmap, 'dict_2d': dict_2d},
                                open(alphafold_prediction_cache, 'wb'))
            if af2_pdb != '':
                pose_input = parse_pdbfile(input_path)
                input_idx = np.array([i['rindex'] for i in pose_input])
                pose_af2 = parse_pdbfile(af2_pdb)
                af2_idx = np.array([i['rindex'] for i in pose_af2])
                mask = np.isin(af2_idx, input_idx)
                af2_qa = af2_qa[:, mask]
                plddt = plddt[:, mask]
                cmap = cmap[:, mask][mask, :]
                for f2d_type in dict_2d.keys():
                    dict_2d[f2d_type] = dict_2d[f2d_type][:, :, mask][:, mask, :]
        else:
            af2_qa, plddt, cmap, dict_2d = process_without_af_model(input_path)
    else:
        dict_2d['f2d_dan'] = get_base2d_feature(input_path, output_path)

    with torch.no_grad():
        for method in methods:
            if method == 'EGNN_Full':
                dim2d = 25 + 9 * 5
                model = resEGNN_with_ne(dim2d=dim2d, dim1d=33)
                # state = torch.load('models/egnn_ne.tar', map_location=torch.device('cpu'))
                state = torch.load('models/model_weights.pth', map_location=torch.device('cpu'))
                # model.load_state_dict(state['model'])
                model.load_state_dict(state)
                model.to(device)
                model.eval()
                f2d = np.concatenate((sh_data, dict_2d['esto9']), axis=0)
                f1d = np.concatenate((one_hot, features, plddt, af2_qa), axis=0)
                f1d = torch.tensor(f1d).unsqueeze(0).to(device)
                f2d = torch.tensor(f2d).unsqueeze(0).to(device)
                pos = torch.tensor(pos).to(device)
                el = [torch.tensor(i).to(device) for i in el]
                cmap = torch.tensor(cmap).to(device)
                _, _, pred_lddt = model(f1d, f2d, pos, el, cmap)
                out = pred_lddt.cpu().detach().numpy().astype(np.float16)
                out[out > 1] = 1
                out[out < 0] = 0
                pred_lddt_all = pred_lddt_all + out / len(methods)
            if method == 'se3_Full':
                dim2d = 25 + 9 * 5
                from network.se3_model import se3_model

                model = se3_model(dim2d=dim2d, dim1d=33)
                state = torch.load('models/esto9_se3.tar', map_location=torch.device('cpu'))
                model.load_state_dict(state['model'])
                model.to(device)
                model.eval()
                f2d = np.concatenate((sh_data, dict_2d['esto9']), axis=0)
                f1d = np.concatenate((one_hot, features, plddt, af2_qa), axis=0)
                f1d = torch.tensor(f1d).unsqueeze(0).to(device)
                f2d = torch.tensor(f2d).unsqueeze(0).to(device)
                pos = torch.tensor(pos).to(device)
                el = [torch.tensor(i).to(device) for i in el]
                cmap = torch.tensor(cmap).to(device)
                _, _, pred_lddt = model(f1d, f2d, pos, el, cmap)
                out = pred_lddt.cpu().detach().numpy().astype(np.float16)
                out[out > 1] = 1
                out[out < 0] = 0
                pred_lddt_all = pred_lddt_all + out / len(methods)
            if method == 'EGNN_covariance':
                dim2d = 25 + 25
                model = resEGNN_with_mask(dim2d=dim2d, dim1d=33)
                state = torch.load('models/cov25.pth.tar', map_location=torch.device('cpu'))
                model.load_state_dict(state['model'])
                model.to(device)
                model.eval()
                f2d = np.concatenate((sh_data, dict_2d['cov25']), axis=0)
                f1d = np.concatenate((one_hot, features, plddt, af2_qa), axis=0)
                f1d = torch.tensor(f1d).unsqueeze(0).to(device)
                f2d = torch.tensor(f2d).unsqueeze(0).to(device)
                pos = torch.tensor(pos).to(device)
                el = [torch.tensor(i).to(device) for i in el]
                cmap = torch.tensor(cmap).to(device)
                _, _, pred_lddt = model(f1d, f2d, pos, el, cmap)
                out = pred_lddt.cpu().detach().numpy().astype(np.float16)
                out[out > 1] = 1
                out[out < 0] = 0
                pred_lddt_all = pred_lddt_all + out / len(methods)
            if method == 'EGNN_esto9':
                dim2d = 25 + 45
                model = resEGNN_with_mask(dim2d=dim2d, dim1d=33)
                state = torch.load('models/egnn_esto9.tar', map_location=torch.device('cpu'))
                model.load_state_dict(state['model'])
                model.to(device)
                model.eval()
                f2d = np.concatenate((sh_data, dict_2d['esto9']), axis=0)
                f1d = np.concatenate((one_hot, features, plddt, af2_qa), axis=0)
                f1d = torch.tensor(f1d).unsqueeze(0).to(device)
                f2d = torch.tensor(f2d).unsqueeze(0).to(device)
                pos = torch.tensor(pos).to(device)
                el = [torch.tensor(i).to(device) for i in el]
                cmap = torch.tensor(cmap).to(device)
                _, _, pred_lddt = model(f1d, f2d, pos, el, cmap)
                out = pred_lddt.cpu().detach().numpy().astype(np.float16)
                out[out > 1] = 1
                out[out < 0] = 0
                pred_lddt_all = pred_lddt_all + out / len(methods)

    np.save(os.path.join(output_path, input_name+'.npy'), pred_lddt_all.astype(np.float16))
    if is_multi_chain:
        shutil.rmtree(temp_dir)
    return pred_lddt_all.astype(np.float16)


def run_enq_complexes(
    input_path: Path, 
    output_path: str, 
    method: str, 
    cpu: str,
    alphafold_prediction: str,
    alphafold_feature_cache: str,
    af2_pdb: str,
    complex_b: bool,
    test: str,
    filename: str
) -> None:
    """
    Run enqa for getting prediction lddt for each structure from test file
    @param input_path: path to input pdb files
    @param output_path: path to output folder
    @param method: prediction method, can be "ensemble", "EGNN_Full", "se3_Full", "EGNN_esto9" or "EGNN_covariance". Ensemble can be done listing multiple models separated by comma
    @param cpu: force to use CPU
    @param alphafold_prediction: path to alphafold prediction results
    @param alphafold_feature_cache: cashe
    @param af2_pdb: optional. PDBs from AlphaFold2 predcition for index correction with input pdb. Must contain all residues in input pdb
    @param complex_b: True if input pdb is complex, else False
    @param test: filename for file with test IDs
    @param filename: filename for structure
    @return: None
    """  
    with open(test, 'r') as f:
        structure_id = f.read().splitlines()
    pred_lddt_all = dict()
    for _id in tqdm(structure_id):
        input_path_pdb = input_path / _id / filename
        try:
            logging.info(f"Getting predcition lddt for {_id}...")
            pred_lddt = enqa(
                input_path=str(input_path_pdb), 
                output_path=output_path, 
                method=method, 
                cpu=cpu,
                alphafold_prediction=alphafold_prediction,
                alphafold_feature_cache=alphafold_feature_cache,
                af2_pdb=af2_pdb,
                complex_b=complex_b
            )
            pred_lddt_all[_id] = np.mean(pred_lddt)
        except Exception as e:
            logging.error(f"Problems with {_id}!")
            logging.exception(e)
            continue
    print(pred_lddt_all)
    torch.save(pred_lddt_all, output_path + 'pred_lddt_all')




if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s - %(message)s', 
        filename='run_enqa.txt', 
        level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(
        description='Predict model quality and output numpy array format.'
    )
    parser.add_argument(
        '--input_path', 
        type=Path, 
        required=True,
        help='Path to input pdb files.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Path to output folder.'
    )
    parser.add_argument(
        '--method', 
        type=str, 
        required=False, 
        default='EGNN_Full',
        help='Prediction method, can be "ensemble", "EGNN_Full", "se3_Full", "EGNN_esto9" or "EGNN_covariance". Ensemble can be done listing multiple models separated by comma.'
    )
    parser.add_argument(
        '--cpu', 
        action='store_true', 
        default=False, 
        help='Force to use CPU.'
    )
    parser.add_argument(
        '--alphafold_prediction', 
        type=str, 
        required=False, 
        default='',
        help='Path to alphafold prediction results.'
    )
    parser.add_argument(
        '--alphafold_feature_cache', 
        type=str, 
        required=False, 
        default=''
    )
    parser.add_argument(
        '--af2_pdb', 
        type=str, 
        required=False, 
        default='',
        help='Optional. PDBs from AlphaFold2 predcition for index correction with input pdb. Must contain all residues in input pdb.'
    )
    parser.add_argument(
        '--complex', 
        type=bool, 
        required=False, 
        default=False,
        help='Input pdb is complex or not.'
    )
    parser.add_argument(
        '--test', 
        type=str, 
        required=False, 
        default=False,
        help='Filename for test IDs.'
    )
    parser.add_argument(
        '--filename', 
        type=str, 
        required=False, 
        choices=['real_joined.pdb', 'docked_joined.pdb'],
        default='docked_joined.pdb',
        help='Filename for structure.'
    ) 
    
    args = parser.parse_args()
    run_enq_complexes(
        input_path=args.input_path, 
        output_path=args.output, 
        method=args.method, 
        cpu=args.cpu,
        alphafold_prediction=args.alphafold_prediction,
        alphafold_feature_cache=args.alphafold_feature_cache,
        af2_pdb=args.af2_pdb,
        complex_b=args.complex,
        test=args.test,
        filename=args.filename
    )

    



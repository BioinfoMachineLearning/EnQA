import os
import pickle
import torch
import argparse
import numpy as np

from data.loader import expand_sh
from feature import create_basic_features, get_base2d_feature
from data.process_alphafold import process_alphafold_target_ensemble, process_alphafold_model
from network.resEGNN import resEGNN, resEGNN_with_mask, resEGNN_with_ne

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict model quality and output numpy array format.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input pdb file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output folder.')
    parser.add_argument('--method', type=str, required=False, default='EGNN_Full',
                        help='Prediction method, can be "EGNN_Full", "se3_Full", "EGNN_esto9" or "EGNN_covariance". Ensemble can be done listing multiple models separated by comma.')
    parser.add_argument('--cpu', action='store_true', default=False, help='Force to use CPU.')
    parser.add_argument('--alphafold_prediction', type=str, required=False, default='',
                        help='Path to alphafold prediction results.')
    parser.add_argument('--alphafold_feature_cache', type=str, required=False, default='')
    parser.add_argument('--af2_pdb', type=str, required=False, default='',
                        help='Optional. PDBs from AlphaFold2 predcition for index correction with input pdb')

    args = parser.parse_args()
    if args.alphafold_feature_cache == '':
        args.alphafold_feature_cache = None
    device = torch.device('cuda:0') if torch.cuda.is_available() and not args.cpu else 'cpu'
    lddt_cmd = 'utils/lddt'
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Featureize
    methods = args.method.split(',')
    disto_types = []
    if 'EGNN_Full' in methods or 'se3_Full' in methods or 'EGNN_esto9' in methods:
        disto_types.append('esto9')
    if 'EGNN_covariance' in methods:
        disto_types.append('cov25')
    if 'EGNN_no_AF2' in methods:
        disto_types.append('base')

    one_hot, features, pos, sh_adj, el = create_basic_features(args.input, args.output, template_path=None,
                                                               diff_cutoff=15, coordinate_factor=0.01)
    sh_data = expand_sh(sh_adj, one_hot.shape[1])
    pred_lddt_all = np.zeros(one_hot.shape[1])
    if 'EGNN_Full' in methods or 'se3_Full' in methods or 'EGNN_esto9' in methods or 'EGNN_covariance' in methods:
        use_af2 = True
    else:
        use_af2 = False
    dict_2d = {}
    if use_af2:
        af2_qa = process_alphafold_model(args.input, args.alphafold_prediction, lddt_cmd, n_models=5)
        if args.alphafold_feature_cache is not None and os.path.isfile(args.alphafold_prediction_cache):
            x = pickle.load(open(args.alphafold_prediction_cache, 'rb'))
            plddt = x['plddt']
            cmap = x['cmap']
            dict_2d = x['af2_2d_dict']
        else:
            plddt, cmap, dict_2d = process_alphafold_target_ensemble(args.alphafold_prediction, disto_types,
                                                                     n_models=5, cmap_cutoff_dim=42,
                                                                     input_pdb_file=args.input)
            if args.alphafold_feature_cache is not None:
                pickle.dump({'plddt': plddt, 'cmap': cmap, 'dict_2d': dict_2d},
                            open(args.alphafold_prediction_cache, 'wb'))
    else:
        dict_2d['f2d_dan'] = get_base2d_feature(args.input, args.output)
    with torch.no_grad():
        for method in methods:
            if method == 'EGNN_Full':
                dim2d = 25 + 9 * 5
                model = resEGNN_with_ne(dim2d=dim2d, dim1d=33)
                state = torch.load('models/egnn_ne.tar', map_location=torch.device('cpu'))
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
                pred_lddt_all = pred_lddt_all + out / len(methods)
            else:
                raise NotImplementedError
    np.save(os.path.join(args.output, os.path.basename(args.input)), pred_lddt_all.astype(np.float16))

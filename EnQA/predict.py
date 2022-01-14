import os
import torch
import argparse
import numpy as np

from feature import create_feature
from network.resEGNN import resEGNN, resEGNN_with_mask, resEGNN_with_ne

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict.')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--disto_type', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=False, default='egnn')
    parser.add_argument('--alphafold_prediction', type=str, required=False, default='')
    parser.add_argument('--alphafold_feature_cache', type=str, required=False, default='')

    args = parser.parse_args()
    if args.alphafold_feature_cache == '':
        args.alphafold_feature_cache = None

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    if args.disto_type == 'base':
        f1d, f2d, pos, el = create_feature(input_model_path=args.input, output_feature_path=args.output,
                                           disto_type=args.disto_type)
        model = resEGNN(dim2d=41, dim1d=23)
        state = torch.load(args.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state['model'])
        model.to(device)
        model.eval()
        with torch.no_grad():
            f1d = torch.tensor(f1d).unsqueeze(0).to(device)
            f2d = torch.tensor(f2d).unsqueeze(0).to(device)
            pos = torch.tensor(pos).to(device)
            el = [torch.tensor(i).to(device) for i in el]
            pred_bin, pos_new, pred_lddt = model(f1d, f2d, pos, el)
    elif args.disto_type in ['disto', 'cov25', 'cov64', 'esto9']:
        f1d, f2d, pos, el, cmap = create_feature(input_model_path=args.input, output_feature_path=args.output,
                                                 disto_type=args.disto_type,
                                                 alphafold_prediction_path=args.alphafold_prediction,
                                                 alphafold_prediction_cache=args.alphafold_feature_cache)
        if args.disto_type == 'disto':
            dim2d = 25 + 64 * 5
        elif args.disto_type == 'cov25':
            dim2d = 25 + 25
        elif args.disto_type == 'esto9':
            dim2d = 25 + 9 * 5
        else:
            dim2d = 25 + 64
        if args.model_type == 'egnn':
            model = resEGNN_with_mask(dim2d=dim2d, dim1d=33)
        elif args.model_type == 'egnn_ne':
            model = resEGNN_with_ne(dim2d=dim2d, dim1d=33)
        elif args.model_type == 'se3':
            from network.se3_model import se3_model
            model = se3_model(dim2d=dim2d, dim1d=33)
        else:
            raise NotImplementedError
        state = torch.load(args.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state['model'])
        model.to(device)
        model.eval()
        with torch.no_grad():
            f1d = torch.tensor(f1d).unsqueeze(0).to(device)
            f2d = torch.tensor(f2d).unsqueeze(0).to(device)
            pos = torch.tensor(pos).to(device)
            el = [torch.tensor(i).to(device) for i in el]
            cmap = torch.tensor(cmap).to(device)
            _, _, pred_lddt = model(f1d, f2d, pos, el, cmap)
    else:
        raise NotImplementedError
    out = pred_lddt.cpu().detach().numpy().astype(np.float16)
    out[out > 1] = 1
    np.save(os.path.join(args.output, os.path.basename(args.input)), out)
